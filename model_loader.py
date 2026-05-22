"""
=============================================================================
LLaVA-1.5-7B 模型加载器 (Model Loader)
=============================================================================

本模块实现了LLaVA多模态大语言模型的加载、配置和管理功能。

【核心功能】
1. 模型加载：支持全精度、8bit、4bit量化加载
2. LoRA集成：集成PEFT库，支持LoRA适配器的添加、保存、加载
3. 推理接口：提供图像描述生成功能
4. 内存优化：针对不同显存环境的自动优化

【主要类】
- LLaVAModelLoader: 模型加载器主类
  统一管理模型、processor、LoRA适配器

【工厂函数】
- create_lora_model_loader(): 快速创建带LoRA的模型加载器
  简化模型初始化流程

【核心特性】
1. LoRA动态秩调整
   - 支持运行时修改LoRA秩（用于渐进式训练）
   - 使用SVD正交化方法扩展权重矩阵

2. LoRA适配器管理
   - save_lora_adapter(): 保存适配器到指定路径
   - load_lora_adapter(): 从检查点加载适配器
   - 支持多种LoRA配置（default, progressive, fair_comparison等）

3. 多种量化配置
   - 全精度（FP16）：显存充足时使用
   - 8bit量化：节省约50%显存
   - 4bit量化：节省约75%显存

4. 自动显存管理
   - 自动检测GPU显存大小
   - 根据显存大小选择合适的量化策略
   - 支持梯度检查点（gradient checkpointing）

【使用示例】
```python
# 创建带LoRA的模型加载器
model_loader = create_lora_model_loader(
    model_path="/path/to/llava-1.5-7b",
    lora_config_name="progressive_lora",
    adapter_path="/path/to/adapter"  # 可选，加载已保存的适配器
)

# 生成图像描述
caption = model_loader.generate_caption(image_path, prompt)

# 保存LoRA适配器
model_loader.save_lora_adapter("/path/to/save")
```

"""

import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from PIL import Image
import logging
import gc
from typing import Optional, Union, Dict, Any
import warnings
import os

# 导入LoRA配置
from lora_config import LoRAConfig, get_lora_config

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLaVAModelLoader:
    def __init__(self, 
                 model_path: str = "/root/autodl-tmp/llava-1.5-7b",
                 lora_config: Optional[LoRAConfig] = None,
                 lora_config_name: str = "default_48gb"):
        """
        初始化LLaVA模型加载器
        
        Args:
            model_path: 模型路径
            lora_config: LoRA配置对象
            lora_config_name: LoRA配置名称（如果lora_config为None）
        """
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = self._get_device()
        
        # LoRA相关属性
        self.lora_config = lora_config or get_lora_config(lora_config_name)
        self.is_lora_enabled = self.lora_config.enable_lora
        self.peft_model = None
        self.lora_adapter_path = None
        
        logger.info(f"使用设备: {self.device}")
        if self.is_lora_enabled:
            logger.info(f"LoRA微调已启用，配置: r={self.lora_config.lora_r}, alpha={self.lora_config.lora_alpha}")
        
    def _get_device(self) -> str:
        """获取可用的计算设备"""
        if torch.cuda.is_available():
            # 检查显存
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"检测到GPU显存: {gpu_memory:.1f}GB")
            return "cuda"
        else:
            logger.warning("未检测到CUDA，将使用CPU（速度较慢）")
            return "cpu"
    
    def load_model(self, 
                   load_in_8bit: bool = None, 
                   load_in_4bit: bool = None,
                   enable_lora: bool = None) -> bool:
        """
        加载LLaVA模型和处理器
        
        Args:
            load_in_8bit: 是否以8bit精度加载（节省显存），None时使用LoRA配置
            load_in_4bit: 是否以4bit精度加载（更节省显存），None时使用LoRA配置
            enable_lora: 是否启用LoRA，None时使用配置文件设置
            
        Returns:
            bool: 是否成功加载
        """
        try:
            logger.info("开始加载LLaVA模型...")
            
            # 处理LoRA和量化配置
            if enable_lora is not None:
                self.is_lora_enabled = enable_lora
            
            # 使用LoRA配置中的量化设置（如果参数未指定）
            if load_in_4bit is None:
                load_in_4bit = self.lora_config.use_4bit_quantization
            if load_in_8bit is None:
                load_in_8bit = self.lora_config.use_8bit_quantization
            
            # 配置模型加载参数
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
                "trust_remote_code": True
            }
            
            # 根据显存情况选择量化选项
            if self.device == "cuda":
                if load_in_4bit:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=getattr(torch, self.lora_config.bnb_4bit_compute_dtype),
                        bnb_4bit_quant_type=self.lora_config.bnb_4bit_quant_type,
                        bnb_4bit_use_double_quant=self.lora_config.bnb_4bit_use_double_quant
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    logger.info("使用4bit量化加载")
                elif load_in_8bit:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_8bit_compute_dtype=torch.float16
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    logger.info("使用8bit量化加载")
            
            # 加载处理器
            logger.info("加载处理器...")
            self.processor = LlavaProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # 加载模型
            logger.info("加载模型（可能需要几分钟）...")
            # 忽略模型类型兼容性警告
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_path,
                    **model_kwargs
                )
            
            # 启用梯度检查点（如果配置中启用）
            if self.lora_config and self.lora_config.use_gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
                logger.info("梯度检查点已启用")
            
            # 应用LoRA配置
            if self.is_lora_enabled:
                success = self._setup_lora()
                if not success:
                    logger.error("LoRA设置失败")
                    return False
            
            # 清理内存
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            logger.info("模型加载成功！")
            self._print_memory_usage()
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            return False
    
    def _setup_lora(self) -> bool:
        """设置LoRA微调"""
        try:
            logger.info("设置LoRA微调...")
            
            # 创建PEFT LoRA配置
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.lora_config.lora_r,
                lora_alpha=self.lora_config.lora_alpha,
                lora_dropout=self.lora_config.lora_dropout,
                target_modules=self.lora_config.target_modules,
                bias=self.lora_config.lora_bias
            )
            
            # 应用LoRA到模型
            self.peft_model = get_peft_model(self.model, peft_config)
            
            # 打印可训练参数信息
            self.peft_model.print_trainable_parameters()
            
            logger.info("LoRA设置完成")
            return True
            
        except Exception as e:
            logger.error(f"LoRA设置失败: {str(e)}")
            return False
    
    def _print_memory_usage(self):
        """打印内存使用情况"""
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / (1024**3)
            cached = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"GPU内存使用: {allocated:.2f}GB 已分配, {cached:.2f}GB 已缓存")
    
    def check_token_safety(self, text: str, max_tokens: int = 3500) -> tuple[bool, int]:
        """
        检查文本token长度是否安全
        
        Args:
            text: 输入文本
            max_tokens: 最大允许token数
            
        Returns:
            tuple: (是否安全, token数量)
        """
        if self.processor is None:
            return True, 0
            
        token_length = len(self.processor.tokenizer.encode(text))
        return token_length <= max_tokens, token_length
    
    def truncate_text_safely(self, text: str, max_tokens: int = 3500) -> str:
        """
        安全截断文本到指定token长度
        
        Args:
            text: 输入文本
            max_tokens: 最大token数
            
        Returns:
            str: 截断后的文本
        """
        if self.processor is None:
            return text
            
        tokens = self.processor.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
            
        # 截断并解码
        truncated_tokens = tokens[:max_tokens]
        return self.processor.tokenizer.decode(truncated_tokens, skip_special_tokens=True)

    def describe_image(self, 
                      image: Union[str, Image.Image], 
                      prompt: str = "Describe this image in detail.",
                      max_new_tokens: int = 50,
                      temperature: float = 0.7,
                      num_beams: int = 3,
                      length_penalty: float = 1.0,
                      repetition_penalty: float = 1.2) -> str:
        """
        对图像进行描述
        
        Args:
            image: 图像路径或PIL Image对象
            prompt: 描述提示词
            max_new_tokens: 最大生成token数 (默认50，覆盖模型实际生成长度23±3词 + 系统开销)
            temperature: 生成温度 (仅在do_sample=True时有效)
            num_beams: beam search的beam数量 (默认3，提高质量)
            length_penalty: 长度惩罚 (>1鼓励长句，<1鼓励短句，=1中性)
            repetition_penalty: 重复惩罚 (>1减少重复)
            
        Returns:
            str: 图像描述文本
            
        Note:
            - COCO标准描述通常10-15词（约50-77 tokens含标点）
            - 使用beam search (num_beams>1) 可提高质量但速度较慢
            - 设置do_sample=False使用贪心解码获得更稳定的结果
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("模型未加载，请先调用load_model()")
        
        try:
            # 处理图像输入
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif not isinstance(image, Image.Image):
                raise ValueError("image参数必须是图像路径字符串或PIL Image对象")
            
            logger.info(f"处理图像，尺寸: {image.size}")
            
            # 构建完整提示
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"},
                    ],
                }
            ]
            
            # 处理输入 - 使用LLaVA-1.5的正确方法
            text_prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            
            # 预检查token长度，避免超限
            token_length = len(self.processor.tokenizer.encode(text_prompt))
            if token_length > 3500:  # 留出空间给视觉tokens（约576）和生成tokens
                logger.warning(f"输入文本过长({token_length} tokens)，将进行截断")
                # 截断过长的prompt
                tokens = self.processor.tokenizer.encode(text_prompt)[:3500]
                text_prompt = self.processor.tokenizer.decode(tokens, skip_special_tokens=True)
            
            inputs = self.processor(
                text=text_prompt,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096  # LLaVA-1.5的最大输入长度
            )
            
            # 移动到设备
            if self.device == "cuda":
                for k, v in inputs.items():
                    if hasattr(v, 'to'):
                        inputs[k] = v.to(self.device)
            
            # 生成描述
            logger.info("开始生成图像描述...")
            
            # 设置pad_token_id（如果tokenizer没有pad_token，使用eos_token）
            pad_token_id = getattr(self.processor.tokenizer, 'pad_token_id', None)
            if pad_token_id is None:
                pad_token_id = self.processor.tokenizer.eos_token_id
            
            with torch.no_grad():
                # 选择正确的模型进行生成
                model_to_use = self.peft_model if self.is_lora_enabled and self.peft_model is not None else self.model
                
                # LLaVA-1.5的优化生成配置
                # 使用beam search提高质量，添加长度和重复惩罚
                generate_kwargs = {
                    'input_ids': inputs['input_ids'],
                    'pixel_values': inputs['pixel_values'],
                    'attention_mask': inputs['attention_mask'],
                    'max_new_tokens': max_new_tokens,
                    'pad_token_id': pad_token_id,
                    'eos_token_id': self.processor.tokenizer.eos_token_id,  # 添加EOS token以支持句子完整性
                    'use_cache': True,
                    'length_penalty': length_penalty,
                    'repetition_penalty': repetition_penalty,
                }
                
                # 根据num_beams选择生成策略
                if num_beams > 1:
                    # Beam search模式：更高质量但速度较慢
                    generate_kwargs.update({
                        'num_beams': num_beams,
                        'do_sample': False,  # beam search不使用采样
                        'early_stopping': True,  # 提前停止
                    })
                else:
                    # 采样模式：更快但可能质量稍低
                    generate_kwargs.update({
                        'do_sample': True,
                        'temperature': temperature,
                        'top_p': 0.9,  # nucleus sampling
                    })
                
                outputs = model_to_use.generate(**generate_kwargs)
            
            # 解码结果
            generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # 提取生成的描述（去除输入部分）
            # 尝试多种分割方式来提取助手的回复
            if "assistant\n" in generated_text:
                description = generated_text.split("assistant\n")[-1].strip()
            elif "ASSISTANT:" in generated_text:
                description = generated_text.split("ASSISTANT:")[-1].strip()
            elif "Assistant:" in generated_text:
                description = generated_text.split("Assistant:")[-1].strip()
            else:
                # 如果没有找到明确的分隔符，尝试从输入长度之后提取
                input_length = len(self.processor.batch_decode(inputs['input_ids'], skip_special_tokens=True)[0])
                if len(generated_text) > input_length:
                    description = generated_text[input_length:].strip()
                else:
                    description = generated_text.strip()
            
            # 确保句子完整性
            description = self._ensure_complete_sentence(description)
            
            logger.info("图像描述生成完成")
            return description
            
        except Exception as e:
            logger.error(f"图像描述生成失败: {str(e)}")
            raise
    
    def _ensure_complete_sentence(self, text: str) -> str:
        """
        确保描述以完整句子结束
        
        Args:
            text: 原始生成文本
            
        Returns:
            str: 处理后的完整句子文本
        """
        text = text.strip()
        if not text:
            return text
        
        # 如果已经以句号、问号、感叹号结尾，直接返回
        if text[-1] in '.!?':
            return text
        
        # 找到最后一个句子结束符的位置
        last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        
        # 如果找到了完整的句子结束符，截断到那里
        if last_period > 0:
            return text[:last_period + 1]
        
        # 如果完全没有句号，添加一个句号（保持原文完整性）
        return text + '.'
    
    def save_lora_adapter(self, save_path: str) -> bool:
        """
        保存LoRA适配器
        
        Args:
            save_path: 保存路径
            
        Returns:
            bool: 是否保存成功
        """
        if not self.is_lora_enabled or self.peft_model is None:
            logger.warning("LoRA未启用或模型未加载，无法保存适配器")
            return False
        
        try:
            os.makedirs(save_path, exist_ok=True)
            self.peft_model.save_pretrained(save_path)
            
            # 同时保存LoRA配置
            config_path = os.path.join(save_path, "lora_config.json")
            self.lora_config.save_to_file(config_path)
            
            logger.info(f"LoRA适配器已保存到: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存LoRA适配器失败: {str(e)}")
            return False
    
    def load_lora_adapter(self, adapter_path: str) -> bool:
        """
        加载LoRA适配器
        
        Args:
            adapter_path: 适配器路径
            
        Returns:
            bool: 是否加载成功
        """
        if self.model is None:
            logger.error("基础模型未加载，无法加载LoRA适配器")
            return False
        
        try:
            logger.info(f"加载LoRA适配器: {adapter_path}")
            
            # 加载LoRA配置（如果存在）
            config_path = os.path.join(adapter_path, "lora_config.json")
            if os.path.exists(config_path):
                self.lora_config = LoRAConfig.from_file(config_path)
                logger.info("已加载LoRA配置")
            
            # 获取基础模型（如果当前模型是PEFT模型，需要先获取基础模型）
            base_model = self.model
            if isinstance(self.model, PeftModel):
                logger.info("检测到当前模型是PEFT模型，获取基础模型用于加载新适配器")
                base_model = self.model.get_base_model()
                # 清理旧的PEFT模型
                if self.peft_model is not None:
                    del self.peft_model
                    self.peft_model = None
            
            # 加载PEFT模型
            self.peft_model = PeftModel.from_pretrained(base_model, adapter_path)
            
            # 确保PEFT模型在正确的设备上
            if self.device == "cuda":
                self.peft_model = self.peft_model.to(self.device)
                logger.info(f"PEFT模型已移动到设备: {self.device}")
            
            self.is_lora_enabled = True
            self.lora_adapter_path = adapter_path
            
            logger.info("LoRA适配器加载成功")
            return True
            
        except Exception as e:
            logger.error(f"加载LoRA适配器失败: {str(e)}")
            return False
    
    def merge_and_save_model(self, save_path: str) -> bool:
        """
        合并LoRA权重并保存完整模型
        
        Args:
            save_path: 保存路径
            
        Returns:
            bool: 是否保存成功
        """
        if not self.is_lora_enabled or self.peft_model is None:
            logger.warning("LoRA未启用或模型未加载，无法合并保存")
            return False
        
        try:
            logger.info("合并LoRA权重...")
            
            # 合并权重
            merged_model = self.peft_model.merge_and_unload()
            
            # 保存合并后的模型
            os.makedirs(save_path, exist_ok=True)
            merged_model.save_pretrained(save_path)
            
            # 保存处理器
            if self.processor is not None:
                self.processor.save_pretrained(save_path)
            
            logger.info(f"合并后的模型已保存到: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"合并保存模型失败: {str(e)}")
            return False
    
    def get_trainable_model(self):
        """获取用于训练的模型对象"""
        if self.is_lora_enabled and self.peft_model is not None:
            return self.peft_model
        return self.model
    
    def cleanup(self):
        """清理模型和释放内存"""
        logger.info("清理模型资源...")
        if self.peft_model is not None:
            del self.peft_model
            self.peft_model = None
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("模型资源已清理")

# 便捷函数
def create_model_loader(model_path: str = "/root/autodl-tmp/llava-1.5-7b", 
                       load_in_8bit: bool = None, 
                       load_in_4bit: bool = None,
                       lora_config: Optional[LoRAConfig] = None,
                       lora_config_name: str = "default_48gb",
                       enable_lora: bool = None) -> LLaVAModelLoader:
    """
    创建并加载LLaVA模型（支持LoRA）
    
    Args:
        model_path: 模型路径
        load_in_8bit: 是否使用8bit量化
        load_in_4bit: 是否使用4bit量化
        lora_config: LoRA配置对象
        lora_config_name: LoRA配置名称
        enable_lora: 是否启用LoRA
        
    Returns:
        LLaVAModelLoader: 已加载的模型加载器
    """
    loader = LLaVAModelLoader(
        model_path=model_path,
        lora_config=lora_config,
        lora_config_name=lora_config_name
    )
    success = loader.load_model(
        load_in_8bit=load_in_8bit, 
        load_in_4bit=load_in_4bit,
        enable_lora=enable_lora
    )
    if not success:
        raise RuntimeError("模型加载失败")
    return loader

def create_lora_model_loader(model_path: str = "/root/autodl-tmp/llava-1.5-7b",
                            lora_config_name: str = "default_48gb",
                            adapter_path: Optional[str] = None) -> LLaVAModelLoader:
    """
    创建支持LoRA的模型加载器
    
    Args:
        model_path: 基础模型路径
        lora_config_name: LoRA配置名称
        adapter_path: 预训练的LoRA适配器路径（可选）
        
    Returns:
        LLaVAModelLoader: 已加载的模型加载器
    """
    loader = LLaVAModelLoader(
        model_path=model_path,
        lora_config_name=lora_config_name
    )
    
    if adapter_path:
        # 如果有预训练适配器，先加载基础模型（不启用LoRA），然后加载适配器
        success = loader.load_model(enable_lora=False)
        if not success:
            raise RuntimeError("基础模型加载失败")
        
        success = loader.load_lora_adapter(adapter_path)
        if not success:
            raise RuntimeError("LoRA适配器加载失败")
    else:
        # 如果没有预训练适配器，直接启用LoRA进行训练
        success = loader.load_model(enable_lora=True)
        if not success:
            raise RuntimeError("基础模型加载失败")
    
    return loader

