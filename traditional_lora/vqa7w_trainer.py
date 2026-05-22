"""
VQA7W 传统 LoRA 训练器 (Traditional LoRA Training)

与渐进式 LoRA 版本的区别：
- 使用固定的 LoRA rank（从配置读取）
- 标准的 epoch 循环训练（不是三阶段渐进式）
- 没有动态 rank 调整和 SVD 权重继承

其余功能与渐进式版本完全相同。
"""

import os
import logging
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from tqdm import tqdm

from model_loader import LLaVAModelLoader, create_lora_model_loader
# 导入 VQA7W 相关模块（需要从父目录导入）
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# 导入 VQA7W adapter（从父目录导入，与渐进式版本保持一致）
try:
    from vqa7w_adapter import (
        VQA7WAdapter,
        VQA7WDataset,
        evaluate_vqa7w_accuracy,
    )
except ImportError:
    # 如果从 traditional_lora 目录运行，尝试从 llava_expe 导入
    try:
        from llava_expe.vqa7w_adapter import (
            VQA7WAdapter,
            VQA7WDataset,
            evaluate_vqa7w_accuracy,
        )
    except ImportError:
        # 最后尝试：添加项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from llava_expe.vqa7w_adapter import (
            VQA7WAdapter,
            VQA7WDataset,
            evaluate_vqa7w_accuracy,
        )

logger = logging.getLogger(__name__)

@dataclass
class VQA7WTrainingConfig:
    """VQA7W 传统 LoRA 训练配置"""

    model_path: str = "/root/autodl-tmp/llava-1.5-7b"
    data_root: str = "/root/autodl-tmp/VQA7W"
    output_dir: str = os.path.join(os.getcwd(), "vqa7w_traditional_lora_output")

    # 训练参数
    num_epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # 学习率调度
    scheduler_type: str = "cosine"  # "linear" or "cosine"
    cosine_num_cycles: float = 0.5

    # 日志与训练细节
    logging_steps: int = 10  # 训练中详细日志的步长

    # 数据与生成
    num_workers: int = 4
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    max_test_samples: Optional[int] = None
    max_new_tokens: int = 32
    temperature: float = 0.7

    # 复现性
    seed: Optional[int] = 42

    # LoRA / 量化配置（与图像描述管线保持一致的接口）
    lora_config_name: str = "traditional_lora_r32"  # 传统 LoRA 使用固定 rank 32（从 16-24-32 中选择中间值）
    enable_lora: bool = True
    # 默认跟随 LoRA 配置，由 traditional_lora_r128 等配置文件统一控制
    load_in_4bit: Optional[bool] = None
    load_in_8bit: Optional[bool] = None

    def prepare_output_dirs(self):
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.logs_dir = os.path.join(self.output_dir, "logs")
        self.lora_adapters_dir = os.path.join(self.output_dir, "lora_adapters")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.lora_adapters_dir, exist_ok=True)

class VQA7WLoRATrainer:
    """
    VQA7W 传统 LoRA 训练器
    
    - 使用固定的 LoRA rank（从配置读取）
    - 标准的 epoch 循环训练
    - 与渐进式版本在非渐进式部分完全一致
    """

    def __init__(self, config: VQA7WTrainingConfig):
        self.config = config
        self.config.prepare_output_dirs()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loader: Optional[LLaVAModelLoader] = None
        self.model = None
        self.processor = None

        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        self.current_epoch = 0
        self._training_start_time: Optional[float] = None
        self._batch_start_time: Optional[float] = None

        self.best_val_acc = 0.0
        self.training_history = []

        self.adapter = VQA7WAdapter(self.config.data_root)
        # 原始完整样本
        train_all = self.adapter.get_split("train")
        val_all = self.adapter.get_split("val")
        test_all = self.adapter.get_split("test")

        # 按配置限制样本数量，便于小规模全流程测试
        if self.config.max_train_samples is not None:
            max_n = min(self.config.max_train_samples, len(train_all))
            train_all = train_all[:max_n]
        if self.config.max_val_samples is not None:
            max_n = min(self.config.max_val_samples, len(val_all))
            val_all = val_all[:max_n]
        if self.config.max_test_samples is not None:
            max_n = min(self.config.max_test_samples, len(test_all))
            test_all = test_all[:max_n]

        self.train_samples = train_all
        self.val_samples = val_all
        self.test_samples = test_all

        logger.info(
            f"VQA7WLoRATrainer 初始化完成，设备={self.device}, "
            f"train={len(self.train_samples)}, val={len(self.val_samples)}, test={len(self.test_samples)}"
        )

        # 设置随机种子以提升复现性
        if self.config.seed is not None:
            self._set_seed(self.config.seed)

    # -------------------- 模型 & 数据 --------------------

    @staticmethod
    def _set_seed(seed: int):
        """固定随机种子，提升复现性。"""
        random.seed(seed)
        import numpy as np
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def setup_model(self) -> bool:
        """设置模型（与渐进式版本保持一致）"""
        try:
            logger.info("正在加载LLaVA模型...")
            self.model_loader = LLaVAModelLoader(
                model_path=self.config.model_path,
                lora_config=None,
                lora_config_name=self.config.lora_config_name,
            )
            
            success = self.model_loader.load_model(
                load_in_8bit=self.config.load_in_8bit,
                load_in_4bit=self.config.load_in_4bit,
                enable_lora=self.config.enable_lora
            )
            
            if not success:
                logger.error("模型加载失败")
                return False
            
            # 获取用于训练的模型（可能是PEFT模型）
            self.model = self.model_loader.get_trainable_model()
            self.processor = self.model_loader.processor
            
            # 设置模型为训练模式
            self.model.train()
            
            # 为了配合梯度检查点，显式关闭 use_cache，避免与缓存机制冲突
            if hasattr(self.model, "config"):
                try:
                    self.model.config.use_cache = False
                except Exception:
                    pass
            
            if self.config.enable_lora:
                logger.info("LoRA微调模式已启用")
                # 打印可训练参数信息
                if hasattr(self.model, 'print_trainable_parameters'):
                    self.model.print_trainable_parameters()
            
            logger.info("模型设置完成")
            return True
            
        except Exception as e:
            logger.error(f"模型设置失败: {str(e)}")
            return False

    def _build_dataloader(self, samples: List[Dict], shuffle: bool = True) -> DataLoader:
        dataset = VQA7WDataset(samples)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            collate_fn=self._collate_fn,
        )

    @staticmethod
    def _collate_fn(batch: List[Dict]) -> Dict:
        # 简单字典打包，后续处理由 trainer 负责
        images = [b["image"] for b in batch]
        questions = [b["question"] for b in batch]
        answers = [b["answer"] for b in batch]
        metas = [b["meta"] for b in batch]
        return {"images": images, "questions": questions, "answers": answers, "metas": metas}

    # -------------------- 优化器 --------------------

    def setup_optimizer(self) -> bool:
        """设置优化器和学习率调度器（标准版本，非渐进式）"""
        try:
            logger.info("正在设置优化器...")
            
            # 获取可训练参数
            trainable_params = []
            trainable_param_count = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    trainable_params.append(param)
                    trainable_param_count += param.numel()
            
            logger.info(f"可训练参数数量: {len(trainable_params)} 个参数组, {trainable_param_count:,} 个参数")
            
            # 创建优化器
            self.optimizer = AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                eps=1e-8,
                betas=(0.9, 0.999),
                amsgrad=False,
                maximize=False,
                foreach=None,
                capturable=False,
                differentiable=False,
                fused=None
            )
            
            # 计算总训练步数
            train_loader = self._build_dataloader(self.train_samples, shuffle=True)
            total_steps = len(train_loader) * self.config.num_epochs
            
            # 计算warmup步数
            if hasattr(self.config, 'warmup_ratio'):
                warmup_steps = int(total_steps * self.config.warmup_ratio)
            else:
                warmup_steps = max(10, total_steps // 10)  # 至少10步warmup
            
            # 根据配置选择学习率调度器
            scheduler_type = getattr(self.config, 'scheduler_type', 'cosine')
            
            if scheduler_type == "linear":
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps
                )
                logger.info(f"使用线性学习率调度器")
                
            elif scheduler_type == "cosine_with_restarts":
                from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
                num_cycles = getattr(self.config, 'cosine_num_cycles', 1.0)
                self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps,
                    num_cycles=int(num_cycles)
                )
                logger.info(f"使用带重启的余弦学习率调度器，cycles: {int(num_cycles)}")
                
            else:  # 默认使用余弦退火
                num_cycles = getattr(self.config, 'cosine_num_cycles', 0.5)
                self.scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps,
                    num_cycles=num_cycles
                )
                logger.info(f"使用余弦退火学习率调度器，cycles: {num_cycles}")
            
            logger.info(f"优化器设置完成，总训练步数: {total_steps}, warmup步数: {warmup_steps}")
            return True
            
        except Exception as e:
            logger.error(f"优化器设置失败: {str(e)}")
            return False

    # -------------------- 参数统计 & 日志摘要 --------------------

    def _compute_parameter_stats(self) -> Dict:
        """
        计算模型参数统计信息
        """
        # 方法1: 优先使用PEFT的标准方法（最准确）
        if hasattr(self.model, 'get_nb_trainable_parameters'):
            trainable_params, total_params = self.model.get_nb_trainable_parameters()
        else:
            # 方法2: 手动递归统计所有参数（包括所有子模块）
            trainable_params = 0
            total_params = 0
            
            # 使用named_parameters()递归获取所有参数（包括嵌套模块）
            for name, param in self.model.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
        
        # 计算可训练参数的范数
        param_norm = 0.0
        for param in self.model.parameters():
            if param.requires_grad:
                param_norm += param.data.norm(2).item() ** 2
        param_norm = param_norm ** 0.5
        
        return {
            'trainable_params': trainable_params,
            'total_params': total_params,
            'param_norm': param_norm,
            'trainable_ratio': trainable_params / total_params if total_params > 0 else 0.0
        }

    def _log_training_summary(self):
        """显示训练配置摘要"""
        param_stats = self._compute_parameter_stats()
        
        logger.info("=" * 80)
        logger.info("🚀 训练配置摘要")
        logger.info("=" * 80)
        logger.info(f"📊 模型参数:")
        logger.info(f"  - 总参数数量: {param_stats['total_params']:,}")
        logger.info(f"  - 可训练参数: {param_stats['trainable_params']:,}")
        logger.info(f"  - 可训练比例: {param_stats['trainable_ratio']:.2%}")
        logger.info(f"  - 参数范数: {param_stats['param_norm']:.3f}")
        
        logger.info(f"📈 训练配置:")
        logger.info(f"  - 训练轮数: {self.config.num_epochs}")
        logger.info(f"  - 批次大小: {self.config.batch_size}")
        logger.info(f"  - 学习率: {self.config.learning_rate:.2e}")
        logger.info(f"  - 权重衰减: {self.config.weight_decay:.2e}")
        logger.info(f"  - 梯度裁剪: {self.config.max_grad_norm}")
        logger.info(f"  - 预热比例: {self.config.warmup_ratio}")
        
        logger.info(f"📋 数据配置:")
        logger.info(f"  - 训练样本: {len(self.train_samples):,}")
        logger.info(f"  - 验证样本: {len(self.val_samples):,}")
        logger.info(f"  - 测试样本: {len(self.test_samples):,}")
        
        logger.info(f"⚙️ 训练策略:")
        logger.info(f"  - 日志步数: {self.config.logging_steps}")
        logger.info(f"  - 输出目录: {os.path.abspath(self.config.output_dir)}")
        
        logger.info("=" * 80)

    # -------------------- 训练核心 --------------------

    def _compute_batch_loss(self, batch: Dict) -> torch.Tensor:
        """
        计算批次损失（与渐进式版本完全相同）
        - 按样本逐个前向，构造 full_text 并手动计算交叉熵
        - 只对回答（assistant / answer）部分计算 loss
        - 对极长样本进行长度检查，避免数值不稳定
        """
        images = batch["images"]
        questions = batch["questions"]
        answers = batch["answers"]

        total_loss = 0.0
        valid_samples = 0

        for img, q, a in zip(images, questions, answers):
            # 跳过无图像或答案为空的样本
            if img is None or not a:
                continue

            # 构建对话：user 提问 + 图像
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Question: {q}\nAnswer the question based on the image.",
                        },
                        {"type": "image"},
                    ],
                }
            ]

            # 得到带 generation prompt 的文本前缀（仅 user 部分）
            try:
                text_prompt = self.processor.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"apply_chat_template 失败，跳过该样本: {e}")
                continue

            # full_text = prompt + 正确答案 + eos
            eos_token = self.processor.tokenizer.eos_token or ""
            full_text = text_prompt + a + eos_token

            # 预检查 token 长度（预留空间给图像tokens约576个）
            try:
                token_length = len(self.processor.tokenizer.encode(full_text))
            except Exception as e:
                logger.warning(f"tokenize full_text 失败，跳过该样本: {e}")
                continue

            if token_length > 3500:  # 留出空间给视觉tokens（约576）和生成tokens
                logger.warning(f"VQA7W 样本 token 过长 ({token_length})，跳过此样本")
                continue

            # 编码单个样本
            inputs = self.processor(
                text=full_text,
                images=img,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            )

            # 移动到设备
            for k, v in inputs.items():
                if hasattr(v, "to"):
                    inputs[k] = v.to(self.device)

            # 前向传播（不直接使用 HF 内置 loss，而是手写 CE，与 COCO 路径对齐）
            outputs = self.model(**inputs)
            logits = outputs.logits

            labels = inputs["input_ids"].clone()

            # 只对回答部分计算 loss：mask 掉 prompt 部分
            # 正确计算prompt长度：先处理只有prompt的输入，获取实际长度（包含图像tokens）
            prompt_inputs = self.processor(
                text=text_prompt,
                images=img,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            )
            prompt_length = prompt_inputs["input_ids"].shape[1]
            # 安全检查：确保prompt_length不超过labels长度
            prompt_length = min(prompt_length, labels.shape[1])
            labels[:, :prompt_length] = -100

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            # 检查损失是否 finite
            if not torch.isfinite(loss):
                logger.warning(f"检测到无效损失值: {loss.item()}，跳过该样本")
                continue

            total_loss += loss
            valid_samples += 1

        if valid_samples == 0:
            logger.warning("VQA7W 批次中没有有效样本，返回零损失")
            return torch.tensor(0.0, requires_grad=True, device=self.device)

        return total_loss / valid_samples

    def train_epoch(self) -> Dict:
        """训练一个epoch（标准循环，非渐进式）"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        train_loader = self._build_dataloader(self.train_samples, shuffle=True)

        # 初始化训练开始时间
        if not hasattr(self, '_training_start_time') or self._training_start_time is None:
            self._training_start_time = time.time()
        
        self._batch_start_time = time.time()

        # 创建进度条
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            try:
                # 清零梯度（在前向传播之前）
                self.optimizer.zero_grad()

                # 前向传播
                loss = self._compute_batch_loss(batch)

                # 检查损失是否有效
                if not torch.isfinite(loss):
                    logger.warning(f"检测到无效损失值: {loss.item()}，跳过此批次")
                    continue

                # 反向传播
                loss.backward()

                # 梯度裁剪
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                # 优化器步进
                self.optimizer.step()
                self.scheduler.step()

                # 更新统计
                total_loss += loss.item()
                num_batches += 1
                self.global_step += 1

                # 计算当前学习率
                current_lr = self.optimizer.param_groups[0]['lr']

                # 更新进度条
                avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

                # 计算内存使用情况
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                    gpu_cached = torch.cuda.memory_reserved() / 1024**3   # GB
                    gpu_total = gpu_memory + gpu_cached  # 总占用
                else:
                    gpu_total = 0.0

                # 计算训练速度 (samples/sec)
                now = time.time()
                batch_time = now - self._batch_start_time if self._batch_start_time else 0.0
                samples_per_sec = self.config.batch_size / batch_time if batch_time > 0 else 0
                self._batch_start_time = now

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'grad_norm': f'{grad_norm:.3f}',
                    'gpu_mem': f'{gpu_total:.1f}GB',
                    'samples/s': f'{samples_per_sec:.1f}'
                })

                # 定期详细日志记录
                if self.global_step % self.config.logging_steps == 0:
                    logger.info(
                        f"📊 Step {self.global_step}/{len(train_loader) * self.config.num_epochs}: "
                        f"loss={loss.item():.4f}, avg_loss={avg_loss:.4f}, "
                        f"lr={current_lr:.2e}, grad_norm={grad_norm:.3f}, "
                        f"gpu_mem={gpu_total:.1f}GB, samples/s={samples_per_sec:.1f}"
                    )

            except Exception as e:
                logger.error(f"训练批次失败: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        # 计算epoch结束时的参数统计
        param_stats = self._compute_parameter_stats()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # 详细的epoch完成日志
        logger.info(
            f"🎯 Epoch {self.current_epoch + 1} 训练完成: "
            f"avg_loss={avg_loss:.4f}, "
            f"total_batches={num_batches}, "
            f"global_step={self.global_step}, "
            f"lr={self.optimizer.param_groups[0]['lr']:.2e}, "
            f"trainable_params={param_stats['trainable_params']:,}, "
            f"param_norm={param_stats['param_norm']:.3f}"
        )

        # 返回epoch统计
        # 获取 epoch 结束时的学习率（在最后一个 step 之前，避免获取到 0）
        # 如果已经执行了所有步骤，则使用当前学习率
        if self.optimizer is not None and len(self.optimizer.param_groups) > 0:
            epoch_end_lr = self.optimizer.param_groups[0]['lr']
        else:
            epoch_end_lr = self.config.learning_rate
        
        epoch_metrics = {
            'epoch': self.current_epoch + 1,
            'avg_loss': avg_loss,
            'learning_rate': epoch_end_lr,
            'global_step': self.global_step,
            'total_batches': num_batches,
            'param_stats': param_stats
        }

        return epoch_metrics

    # -------------------- 验证 & 测试 --------------------

    def _predict_answers(self, samples: List[Dict]) -> Dict[int, str]:
        """预测答案（与渐进式版本完全相同）"""
        self.model.eval()
        predictions: Dict[int, str] = {}
        with torch.no_grad():
            for s in tqdm(samples, desc="VQA7W 推理", leave=False):
                img_path = s["image_path"]
                q = s["question"]
                if img_path is None or not os.path.exists(img_path):
                    continue
                prompt = f"Question: {q}\nAnswer the question based on the image."
                ans = self.model_loader.describe_image(
                    image=img_path,
                    prompt=prompt,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                )
                predictions[int(s["qa_id"])] = ans
        return predictions

    def validate(self) -> Tuple[float, float]:
        """
        验证（计算损失和准确率）
        
        Returns:
            Tuple[float, float]: (验证集损失, 验证集准确率)
        """
        if not self.val_samples:
            logger.warning("没有验证集样本，跳过验证")
            return float('inf'), 0.0
        
        # 计算验证集损失
        self.model.eval()
        val_dataloader = self._build_dataloader(self.val_samples, shuffle=False)
        total_val_loss = 0.0
        valid_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="验证集损失计算", leave=False):
                try:
                    loss = self._compute_batch_loss(batch)
                    if torch.isfinite(loss):
                        total_val_loss += loss.item()
                        valid_batches += 1
                except Exception as e:
                    logger.warning(f"验证批次损失计算失败: {str(e)}")
                    continue
        
        avg_val_loss = total_val_loss / valid_batches if valid_batches > 0 else float('inf')
        
        # 计算验证集准确率
        preds = self._predict_answers(self.val_samples)
        acc = evaluate_vqa7w_accuracy(preds, self.val_samples)
        
        logger.info(f"验证集 VQA7W Loss: {avg_val_loss:.4f}, Accuracy: {acc:.4f}")
        return avg_val_loss, acc

    def test(self) -> float:
        """测试（与渐进式版本完全相同）"""
        if not self.test_samples:
            logger.warning("没有测试集样本，跳过测试")
            return 0.0
        preds = self._predict_answers(self.test_samples)
        acc = evaluate_vqa7w_accuracy(preds, self.test_samples)
        logger.info(f"测试集 VQA7W Accuracy: {acc:.4f}")
        return acc

    # -------------------- 模型保存 --------------------

    def save_best_model(self):
        """保存最佳模型"""
        try:
            best_adapter_path = os.path.join(self.config.lora_adapters_dir, "best_vqa7w_adapter")
            os.makedirs(best_adapter_path, exist_ok=True)
            if self.model_loader.save_lora_adapter(best_adapter_path):
                logger.info(f"最佳LoRA适配器已保存: {os.path.abspath(best_adapter_path)}")
            else:
                logger.warning("保存最佳LoRA适配器失败")
        except Exception as e:
            logger.error(f"保存最佳模型失败: {str(e)}")

    # -------------------- 主训练流程 --------------------

    def train(self) -> bool:
        """执行完整训练流程（标准 epoch 循环，非渐进式）"""
        try:
            logger.info("🎯 开始VQA7W传统LoRA训练...")
            logger.info("=" * 80)
            
            # 设置模型
            if not self.setup_model():
                return False

            # 设置优化器
            if not self.setup_optimizer():
                return False

            # 显示训练配置摘要
            self._log_training_summary()

            # 训练循环
            for epoch in range(self.config.num_epochs):
                self.current_epoch = epoch

                # 计算训练开始时的参数统计
                param_stats = self._compute_parameter_stats()
                # 获取当前学习率：优先从 optimizer 获取
                # 注意：在第一个 epoch 开始前，scheduler 还没有执行 step，所以 optimizer 的 lr 可能还是初始值
                # 对于 warmup scheduler，初始学习率应该是 0，但实际训练时会从 0 开始 warmup
                if self.optimizer is not None and len(self.optimizer.param_groups) > 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    # 如果学习率为 0（warmup 开始前），显示初始学习率配置值
                    if current_lr == 0.0 and epoch == 0:
                        current_lr = self.config.learning_rate
                else:
                    current_lr = self.config.learning_rate  # Fallback
                logger.info(
                    f"🚀 开始训练 Epoch {epoch + 1}/{self.config.num_epochs}: "
                    f"trainable_params={param_stats['trainable_params']:,}, "
                    f"total_params={param_stats['total_params']:,}, "
                    f"trainable_ratio={param_stats['trainable_ratio']:.2%}, "
                    f"current_lr={current_lr:.2e}"
                )

                # 训练一个epoch
                epoch_metrics = self.train_epoch()
                self.training_history.append(epoch_metrics)

                # 每个epoch结束后验证
                logger.info(f"🔍 开始 Epoch {epoch + 1} 验证...")
                val_loss, val_acc = self.validate()

                # 判断是否改善
                improvement = val_acc > self.best_val_acc
                if improvement:
                    self.best_val_acc = val_acc
                    self.save_best_model()

                # 详细的epoch总结日志
                logger.info(
                    f"📈 Epoch {epoch + 1} 总结: "
                    f"train_loss={epoch_metrics['avg_loss']:.4f}, "
                    f"val_loss={val_loss:.4f}, "
                    f"val_accuracy={val_acc:.4f}, "
                    f"best_accuracy={self.best_val_acc:.4f}, "
                    f"improvement={'✅' if improvement else '❌'}, "
                    f"lr={epoch_metrics['learning_rate']:.2e}, "
                    f"global_step={epoch_metrics['global_step']}"
                )

            # 保存最终模型
            final_adapter_path = os.path.join(self.config.lora_adapters_dir, "final_vqa7w_adapter")
            os.makedirs(final_adapter_path, exist_ok=True)
            if self.model_loader.save_lora_adapter(final_adapter_path):
                logger.info(f"最终LoRA适配器已保存: {os.path.abspath(final_adapter_path)}")

            # 完成摘要
            logger.info("\n" + "=" * 80)
            logger.info("🎉 传统LoRA训练完成！")
            logger.info(f"📊 训练统计:")
            logger.info(f"   - 总训练轮数: {self.config.num_epochs}")
            logger.info(f"   - 总训练步数: {self.global_step:,}")
            logger.info(f"🏆 最佳验证指标:")
            logger.info(f"   - ⭐ VQA7W Accuracy: {self.best_val_acc:.4f}")
            logger.info("=" * 80)

            # 测试前加载最佳模型（按照图像描述的处理方式：重新创建model_loader）
            best_adapter_path = os.path.join(self.config.lora_adapters_dir, "best_vqa7w_adapter")
            if os.path.exists(best_adapter_path):
                logger.info(f"📥 加载最佳验证集模型用于测试: {best_adapter_path}")
                try:
                    # 按照图像描述的处理方式：重新创建model_loader，先加载基础模型，再加载适配器
                    test_model_loader = create_lora_model_loader(
                        model_path=self.config.model_path,
                        lora_config_name=self.config.lora_config_name,
                        adapter_path=best_adapter_path
                    )
                    # 替换当前的model_loader用于测试
                    self.model_loader = test_model_loader
                    self.model = test_model_loader.get_trainable_model()
                    logger.info("✅ 最佳模型加载成功")
                except Exception as e:
                    logger.warning(f"⚠️  最佳模型加载失败: {str(e)}，将使用当前模型进行测试")
            else:
                logger.warning(f"⚠️  未找到最佳模型路径 {best_adapter_path}，将使用当前模型进行测试")

            # 测试
            logger.info("开始测试...")
            test_acc = self.test()
            logger.info(f"测试集 VQA7W Accuracy: {test_acc:.4f}")

            return True

        except Exception as e:
            logger.error(f"训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


