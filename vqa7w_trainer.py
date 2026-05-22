"""
VQA7W 渐进式 LoRA 训练器 (与图像描述管线完全隔离，不影响现有功能)
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
import numpy as np

from model_loader import LLaVAModelLoader, create_lora_model_loader
from vqa7w_adapter import (
    VQA7WAdapter,
    VQA7WDataset,
    build_vqa7w_questions_and_meta,
    evaluate_vqa7w_accuracy,
)
from coco_dataset import compute_complexity_for_all_samples, calculate_vqa_complexity

logger = logging.getLogger(__name__)

@dataclass
class VQA7WTrainingConfig:
    """VQA7W 训练配置（保持与 COCOTrainingConfig 相似接口，但更精简）"""

    model_path: str = "/root/autodl-tmp/llava-1.5-7b"
    data_root: str = "/root/autodl-tmp/VQA7W"
    output_dir: str = os.path.join(os.getcwd(), "vqa7w_training_output")

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

    # 渐进式训练设置（仅控制样本分层，不改动 LoRA rank）
    easy_epochs: int = 1
    medium_epochs: int = 1
    hard_epochs: int = 1
    complexity_thresholds: Tuple[float, float] = (33.33, 66.67)
    # 渐进式 LoRA 秩（默认值：16-24-32）
    easy_lora_rank: int = 16
    medium_lora_rank: int = 24
    hard_lora_rank: int = 32
    enable_weight_inheritance: bool = True  # 调整 LoRA 秩时是否进行 SVD 权重继承

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
    lora_config_name: str = "progressive_lora"
    enable_lora: bool = True
    # 默认跟随 LoRA 配置，由 progressive_lora 等配置文件统一控制
    load_in_4bit: Optional[bool] = None
    load_in_8bit: Optional[bool] = None

    def prepare_output_dirs(self):
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.logs_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

class VQA7WLoRATrainer:
    """
    VQA7W 渐进式 LoRA 训练器

    - 不修改任何现有 COCO/Flickr/VizWiz 代码
    - 复用 LLaVAModelLoader 和 VQA7WAdapter
    - 按 question 复杂度（calculate_vqa_complexity）做三阶段 curriculum
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
        self._training_start_time: Optional[float] = None
        self._batch_start_time: Optional[float] = None

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
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def setup_model(self) -> bool:
        """设置模型（与 COCOTrainer.setup_model 保持一致）"""
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

    def setup_optimizer(self, train_steps: int) -> bool:
        try:
            logger.info("设置优化器和学习率调度器...")
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            warmup_steps = int(train_steps * self.config.warmup_ratio)
            if self.config.scheduler_type == "linear":
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=train_steps,
                )
            else:
                self.scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=train_steps,
                    num_cycles=self.config.cosine_num_cycles,
                )
            logger.info(f"优化器设置完成，总步数={train_steps}, warmup={warmup_steps}")
            return True
        except Exception as e:
            logger.error(f"优化器设置失败: {e}")
            return False

    def _setup_optimizer_for_progressive(self) -> bool:
        """
        为渐进式训练重新初始化优化器和学习率调度器
        
        这个方法用于在更新LoRA秩后重新创建优化器，确保每个阶段都有独立的优化器状态。
        注意：这里需要重新计算剩余的训练步数。
        
        Returns:
            是否设置成功
        """
        try:
            logger.info("正在重新初始化优化器...")
            
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
            
            # 使用当前数据加载器的大小计算剩余步数
            # 这样每个阶段都有独立的学习率调度
            if hasattr(self, 'train_dataloader') and self.train_dataloader is not None:
                steps_per_epoch = len(self.train_dataloader)
            else:
                # 如果没有 dataloader，使用默认值
                steps_per_epoch = 1
            
            stage_total_epochs = self._current_stage_total_epochs if hasattr(self, '_current_stage_total_epochs') and self._current_stage_total_epochs is not None else 1
            current_stage_steps = steps_per_epoch * stage_total_epochs
            
            # 计算warmup步数
            if hasattr(self.config, 'warmup_ratio'):
                warmup_steps = int(current_stage_steps * self.config.warmup_ratio)
            else:
                warmup_steps = max(10, current_stage_steps // 10)  # 至少10步warmup
            
            # 根据配置选择学习率调度器
            scheduler_type = getattr(self.config, 'scheduler_type', 'cosine')
            
            if scheduler_type == "linear":
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=current_stage_steps
                )
                logger.info(f"使用线性学习率调度器")
                
            elif scheduler_type == "cosine_with_restarts":
                from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
                num_cycles = getattr(self.config, 'cosine_num_cycles', 1.0)
                self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=current_stage_steps,
                    num_cycles=int(num_cycles)
                )
                logger.info(f"使用带重启的余弦学习率调度器，cycles: {int(num_cycles)}")
                
            else:  # 默认使用余弦退火
                num_cycles = getattr(self.config, 'cosine_num_cycles', 0.5)
                self.scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=current_stage_steps,
                    num_cycles=num_cycles
                )
                logger.info(f"使用余弦退火学习率调度器，cycles: {num_cycles}")
            
            logger.info(f"优化器重新初始化完成")
            logger.info(f"  - 当前阶段步数: {current_stage_steps}")
            logger.info(f"  - Warmup步数: {warmup_steps}")
            logger.info(f"  - 初始学习率: {self.config.learning_rate:.2e}")
            return True
            
        except Exception as e:
            logger.error(f"优化器重新初始化失败: {str(e)}")
            return False

    # -------------------- 训练核心 --------------------

    def _compute_batch_loss(self, batch: Dict) -> torch.Tensor:
        """
        参考 COCOTrainer.compute_loss 风格：
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

    def train_one_epoch(self, dataloader: DataLoader, epoch_idx: int, stage_name: str) -> float:
        self.model.train()
        total_loss = 0.0
        steps = 0

        if self._training_start_time is None:
            self._training_start_time = time.time()
        self._batch_start_time = time.time()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx} [{stage_name}]", leave=False)
        for batch_idx, batch in enumerate(pbar):
            try:
                # 清零梯度（在前向前）
                self.optimizer.zero_grad()

                # 前向计算 batch loss
                loss = self._compute_batch_loss(batch)

                # NaN 检查
                if not torch.isfinite(loss):
                    logger.warning(
                        f"检测到无效损失值: {loss.item()}，跳过此批次 "
                        f"(epoch={epoch_idx}, stage={stage_name}, step={steps})"
                    )
                    continue

                # 反向传播
                loss.backward()

                # 梯度裁剪
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )

                # 优化器 & 调度器步进
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                # 统计量更新
                steps += 1
                self.global_step += 1
                total_loss += loss.item()

                # 学习率
                current_lr = (
                    self.scheduler.get_last_lr()[0]
                    if self.scheduler is not None
                    else self.config.learning_rate
                )

                # GPU 内存
                if torch.cuda.is_available():
                    gpu_mem = (
                        torch.cuda.memory_allocated() + torch.cuda.memory_reserved()
                    ) / 1024**3
                else:
                    gpu_mem = 0.0

                # 速度统计
                now = time.time()
                batch_time = now - self._batch_start_time if self._batch_start_time else 0.0
                samples_per_sec = (
                    self.config.batch_size / batch_time if batch_time > 0 else 0.0
                )
                self._batch_start_time = now

                avg_loss = total_loss / max(steps, 1)

                # 更新 tqdm 展示（尽量对齐 COCO 日志风格）
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    avg_loss=f"{avg_loss:.4f}",
                    lr=f"{current_lr:.2e}",
                    grad_norm=f"{float(grad_norm):.3f}",
                    gpu_mem=f"{gpu_mem:.1f}GB",
                    samples_s=f"{samples_per_sec:.1f}",
                )

                # 周期性详细日志
                if self.global_step % self.config.logging_steps == 0:
                    logger.info(
                        f"[VQA][Stage={stage_name}] Epoch {epoch_idx} Step {steps} "
                        f"global_step={self.global_step} "
                        f"loss={loss.item():.4f} avg_loss={avg_loss:.4f} "
                        f"lr={current_lr:.2e} grad_norm={float(grad_norm):.3f} "
                        f"gpu_mem={gpu_mem:.1f}GB samples/s={samples_per_sec:.1f}"
                    )

            except Exception as e:
                logger.error(f"VQA: 训练 step 失败 (epoch={epoch_idx}, step={steps}): {e}")
                continue

        avg_loss = total_loss / max(steps, 1)
        logger.info(f"Epoch {epoch_idx} [{stage_name}] 平均训练 Loss: {avg_loss:.4f}")
        return avg_loss

    # -------------------- 参数统计 & 日志摘要 --------------------

    def _compute_parameter_stats(self) -> Dict:
        """
        计算模型参数统计信息
        
        修复说明：
        - 使用PEFT库的标准方法获取准确的参数统计
        - 解决了之前只统计主模型参数，遗漏Vision Encoder等模块的问题
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

    def _log_progressive_training_summary(self, stage_epochs, lora_ranks, thresholds):
        """显示渐进式训练配置摘要"""
        logger.info("\n" + "=" * 80)
        logger.info("📋 渐进式训练配置摘要")
        logger.info("=" * 80)
        logger.info(f"训练策略: 课程学习(Curriculum Learning) + 动态LoRA秩调整")
        logger.info(f"复杂度阈值: easy < {thresholds[0]:.2f} | medium < {thresholds[1]:.2f} | hard >= {thresholds[1]:.2f}")
        logger.info("")
        
        # 显示高级配置
        logger.info("🔧 高级配置:")
        enable_optimizer_reset = getattr(self.config, 'enable_optimizer_reset', True)
        logger.info(f"  - 优化器重置: {'启用（阶段独立训练）' if enable_optimizer_reset else '禁用（连续训练模式）'}")
        logger.info(f"  - 权重继承: {'启用（LoRA权重扩展）' if self.config.enable_weight_inheritance else '禁用（随机初始化）'}")
        logger.info("")
        
        logger.info(f"Stage 1 (EASY):   {stage_epochs['easy']} epochs, LoRA rank = {lora_ranks['easy']}")
        logger.info(f"Stage 2 (MEDIUM): {stage_epochs['medium']} epochs, LoRA rank = {lora_ranks['medium']}")
        logger.info(f"Stage 3 (HARD):   {stage_epochs['hard']} epochs, LoRA rank = {lora_ranks['hard']}")
        logger.info("")
        logger.info(f"总训练轮数: {sum(stage_epochs.values())} epochs")
        logger.info(f"输出目录: {os.path.abspath(self.config.output_dir)}")
        logger.info("=" * 80 + "\n")

    # -------------------- 验证 & 测试 --------------------

    def _predict_answers(self, samples: List[Dict]) -> Dict[int, str]:
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
        if not self.test_samples:
            logger.warning("没有测试集样本，跳过测试")
            return 0.0
        preds = self._predict_answers(self.test_samples)
        acc = evaluate_vqa7w_accuracy(preds, self.test_samples)
        logger.info(f"测试集 VQA7W Accuracy: {acc:.4f}")
        return acc

    # -------------------- 渐进式训练（基于复杂度分层） --------------------

    def _build_complexity_splits(self) -> Dict[str, List[int]]:
        logger.info("基于问题文本和答案计算 VQA7W 复杂度并按数值百分比分层（与图像描述逻辑保持一致）...")
        questions_dict, meta_map = build_vqa7w_questions_and_meta(self.train_samples)
        complexity_map = compute_complexity_for_all_samples(
            captions_data=questions_dict,
            cache_file=None,
            complexity_fn=calculate_vqa_complexity,
            meta_map=meta_map,
        )

        # 与图像描述分层保持一致：基于复杂度数值的百分位阈值分层
        vals = np.array(list(complexity_map.values()), dtype=float)
        p1, p2 = self.config.complexity_thresholds
        t1 = np.percentile(vals, p1)
        t2 = np.percentile(vals, p2)

        easy_ids, mid_ids, hard_ids = [], [], []
        for qid, c in complexity_map.items():
            if c <= t1:
                easy_ids.append(qid)
            elif c <= t2:
                mid_ids.append(qid)
            else:
                hard_ids.append(qid)

        logger.info(
            f"复杂度分层(按百分位): easy={len(easy_ids)}, medium={len(mid_ids)}, hard={len(hard_ids)} "
            f"(t1={t1:.3f}, t2={t2:.3f}, p1={p1}, p2={p2})"
        )
        return {"easy": easy_ids, "medium": mid_ids, "hard": hard_ids}

    # -------------------- 渐进式 LoRA：SVD 权重继承辅助工具 --------------------

    def _extract_lora_weights(self) -> Dict[str, torch.Tensor]:
        """
        提取当前模型的 LoRA 权重（用于在调高 rank 时做 SVD 扩展继承）。
        参考 COCOTrainer._extract_lora_weights 的实现，但限定在 VQA 路径内使用。
        """
        lora_weights: Dict[str, torch.Tensor] = {}
        layer_names = set()
        try:
            model = self.model_loader.model
            # PEFT LoRA 模型通常在 base_model 下挂真实模型
            if not hasattr(model, "base_model"):
                logger.warning("模型不是PEFT模型，无法提取LoRA权重")
                return lora_weights

            for name, module in model.named_modules():
                if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                    layer_names.add(name)
                    # 提取 lora_A 权重
                    if hasattr(module.lora_A, "default") and hasattr(module.lora_A.default, "weight"):
                        w_a = module.lora_A.default.weight.detach().clone().cpu()
                        lora_weights[f"{name}.lora_A"] = w_a
                    # 提取 lora_B 权重
                    if hasattr(module.lora_B, "default") and hasattr(module.lora_B.default, "weight"):
                        w_b = module.lora_B.default.weight.detach().clone().cpu()
                        lora_weights[f"{name}.lora_B"] = w_b

            if layer_names:
                logger.info(f"成功提取 {len(layer_names)} 个LoRA层用于权重继承")
            else:
                logger.warning("未找到任何LoRA层，无法提取权重")
            return lora_weights
        except Exception as e:
            logger.error(f"提取LoRA权重失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return lora_weights

    @staticmethod
    def _gram_schmidt_orthogonalize(vectors: torch.Tensor, existing_basis: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        对给定向量集进行 Gram-Schmidt 正交化，可选地相对于已有基底做正交化。
        """
        if vectors.ndim != 2:
            raise ValueError("vectors 必须是二维张量 (d, k)")

        device = vectors.device
        dtype = vectors.dtype

        orth_vectors = vectors.clone()

        # 先相对于已有基底正交化
        if existing_basis is not None and existing_basis.shape[1] > 0:
            for i in range(existing_basis.shape[1]):
                basis_vec = existing_basis[:, i : i + 1]
                proj_coef = torch.matmul(orth_vectors.T, basis_vec)
                projection = basis_vec * proj_coef.T
                orth_vectors = orth_vectors - projection

        # 内部 Gram-Schmidt
        k = orth_vectors.shape[1]
        for i in range(k):
            for j in range(i):
                prev_vec = orth_vectors[:, j : j + 1]
                curr_vec = orth_vectors[:, i : i + 1]
                proj_coef = torch.matmul(curr_vec.T, prev_vec)
                orth_vectors[:, i : i + 1] = curr_vec - proj_coef * prev_vec

            norm = torch.norm(orth_vectors[:, i])
            if norm > 1e-10:
                orth_vectors[:, i] = orth_vectors[:, i] / norm
            else:
                # 退化向量，用随机向量替换
                rand_vec = torch.randn(orth_vectors.shape[0], device=device, dtype=dtype)
                orth_vectors[:, i] = rand_vec / torch.norm(rand_vec)

        return orth_vectors

    def _expand_lora_with_svd(
        self,
        old_lora_A: torch.Tensor,
        old_lora_B: torch.Tensor,
        new_rank: int,
        use_orthogonalization: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用 SVD + 正交化策略扩展 LoRA 权重（从 old_rank 扩展到 new_rank）。
        参考 COCOTrainer._expand_lora_with_svd。
        """
        old_rank = old_lora_A.shape[0]
        in_features = old_lora_A.shape[1]
        out_features = old_lora_B.shape[0]

        device = old_lora_A.device
        dtype = old_lora_A.dtype

        assert old_lora_B.shape[1] == old_rank
        assert new_rank > old_rank

        # 步骤1：重构 ΔW
        delta_W = torch.matmul(old_lora_B, old_lora_A)

        # 步骤2：SVD 分解
        try:
            U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
        except Exception as e:
            logger.warning(f"SVD分解失败，回退到torch.svd: {e}")
            U, S, V = torch.svd(delta_W)
            Vh = V.T

        U = U[:, :old_rank]
        S = S[:old_rank]
        Vh = Vh[:old_rank, :]

        # 步骤3：扩展奇异值（指数衰减）
        num_new = new_rank - old_rank
        last_singular = S[-1].item()
        decay_factor = 0.3
        new_singulars = torch.ones(num_new, device=device, dtype=dtype) * last_singular * decay_factor
        S_expanded = torch.cat([S, new_singulars])

        # 步骤4：扩展 U 和 Vh
        U_new_part = torch.randn(out_features, num_new, device=device, dtype=dtype) * 0.1
        U_expanded = torch.cat([U, U_new_part], dim=1)

        if in_features > old_rank:
            random_matrix = torch.randn(in_features, in_features, device=device, dtype=dtype)
            Q, _ = torch.linalg.qr(random_matrix)
            overlap_scores = torch.sum((Vh @ Q) ** 2, dim=0)
            _, indices = torch.sort(overlap_scores)
            selected_indices = indices[:num_new]
            Vh_new_part = Q[:, selected_indices].T
        else:
            Vh_new_part = torch.randn(num_new, in_features, device=device, dtype=dtype) * 0.1

        Vh_expanded = torch.cat([Vh, Vh_new_part], dim=0)

        # 步骤5：正交化新增维度
        if use_orthogonalization and num_new > 0:
            U_new_orth = self._gram_schmidt_orthogonalize(
                U_expanded[:, old_rank:],
                U_expanded[:, :old_rank],
            )
            U_expanded[:, old_rank:] = U_new_orth

            for i in range(num_new):
                new_row = Vh_expanded[old_rank + i, :].clone()
                for j in range(old_rank):
                    old_row = Vh_expanded[j, :]
                    proj_coef = torch.dot(new_row, old_row)
                    new_row = new_row - proj_coef * old_row
                for j in range(i):
                    prev_row = Vh_expanded[old_rank + j, :]
                    proj_coef = torch.dot(new_row, prev_row)
                    new_row = new_row - proj_coef * prev_row
                norm = torch.norm(new_row)
                if norm > 1e-8:
                    Vh_expanded[old_rank + i, :] = new_row / norm

        # 步骤6：重构新的 A, B
        sqrt_S = torch.sqrt(S_expanded)
        sqrt_S_diag = torch.diag(sqrt_S)
        new_A = torch.matmul(sqrt_S_diag, Vh_expanded)
        new_B = torch.matmul(U_expanded, sqrt_S_diag)

        return new_A, new_B

    def _inherit_lora_weights(self, old_weights: Dict[str, torch.Tensor], old_rank: int, new_rank: int):
        """
        使用 SVD + 正交化策略，将旧 LoRA 权重继承到新的更大 rank 的 LoRA 层。
        """
        if new_rank <= old_rank:
            logger.warning(f"新rank ({new_rank}) <= 旧rank ({old_rank})，跳过权重继承")
            return
        if not old_weights:
            logger.warning("旧LoRA权重为空，跳过权重继承")
            return

        try:
            model = self.model_loader.model
            inherited_layers = 0
            failed_layers = 0

            logger.info("🔬 使用 SVD+正交化 方法扩展 LoRA 权重...")

            for name, module in model.named_modules():
                if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                    lora_a_key = f"{name}.lora_A"
                    lora_b_key = f"{name}.lora_B"
                    if lora_a_key not in old_weights or lora_b_key not in old_weights:
                        continue
                    try:
                        old_a = old_weights[lora_a_key]
                        old_b = old_weights[lora_b_key]

                        if not (hasattr(module.lora_A, "default") and hasattr(module.lora_A.default, "weight")):
                            failed_layers += 1
                            continue
                        if not (hasattr(module.lora_B, "default") and hasattr(module.lora_B.default, "weight")):
                            failed_layers += 1
                            continue

                        new_a_tensor = module.lora_A.default.weight.data
                        new_b_tensor = module.lora_B.default.weight.data

                        if old_a.shape[1] != new_a_tensor.shape[1] or old_b.shape[0] != new_b_tensor.shape[0]:
                            failed_layers += 1
                            continue

                        device = new_a_tensor.device
                        old_a = old_a.to(device)
                        old_b = old_b.to(device)

                        new_a, new_b = self._expand_lora_with_svd(
                            old_a, old_b, new_rank, use_orthogonalization=True
                        )
                        new_a_tensor.copy_(new_a)
                        new_b_tensor.copy_(new_b)
                        inherited_layers += 1
                    except Exception as e:
                        logger.warning(f"继承 {name} 的LoRA权重失败: {e}")
                        failed_layers += 1
                        continue

            logger.info(f"✅ 成功继承 {inherited_layers} 个LoRA层的权重 (rank {old_rank} → {new_rank})")
            if failed_layers > 0:
                logger.warning(f"⚠️ 有 {failed_layers} 个LoRA层权重继承失败")
        except Exception as e:
            logger.error(f"继承LoRA权重失败: {e}")
            import traceback
            traceback.print_exc()

    def _update_model_lora_rank(self, new_rank: int, skip_optimizer_reset: bool = False) -> bool:
        """
        动态更新模型的LoRA秩（支持权重继承和条件优化器重置）
        
        新功能：
        1. 权重继承（enable_weight_inheritance=True）：
           将旧的LoRA权重继承到新的更大rank的LoRA层，避免从头训练
        2. 条件优化器重置（enable_optimizer_reset）：
           - True: 阶段重启模式，每个阶段重置优化器和学习率
           - False: 连续训练模式，优化器状态保持，学习率平滑衰减
        
        Args:
            new_rank: 新的LoRA秩
            skip_optimizer_reset: 是否跳过优化器重置（用于初始化阶段，此时数据加载器还未设置）
            
        注意：这需要重新初始化LoRA适配器，但会保留预训练模型的权重
        """
        logger.info(f"开始更新LoRA秩到 {new_rank}...")
        
        try:
            # 🔥 新功能1：提取旧的LoRA权重用于继承
            old_lora_weights = None
            old_rank = getattr(self, '_current_lora_rank', None)
            
            if self.config.enable_weight_inheritance and old_rank and old_rank < new_rank:
                logger.info(f"📥 提取当前LoRA权重用于继承 (rank={old_rank})...")
                old_lora_weights = self._extract_lora_weights()
                if old_lora_weights:
                    logger.info(f"   已提取 {len(old_lora_weights)} 个LoRA参数")
                else:
                    logger.warning("   提取LoRA权重失败，将不进行权重继承")
            
            # 保存当前模型状态（用于备份）
            temp_model_path = os.path.join(self.config.output_dir, "temp_model_state")
            os.makedirs(temp_model_path, exist_ok=True)
            
            # 获取当前LoRA适配器的权重（用于warmstart）
            if hasattr(self.model_loader.model, 'base_model'):
                logger.info("保存当前LoRA适配器状态...")
                self.model_loader.save_lora_adapter(temp_model_path)
            
            # 更新配置中的LoRA秩
            if hasattr(self.model_loader, "lora_config") and self.model_loader.lora_config:
                self.model_loader.lora_config.update_lora_rank(new_rank)
            
            # 重新加载模型（使用新的LoRA秩）
            logger.info("使用新LoRA秩重新初始化模型...")
            old_model = self.model_loader.model
            
            # 重新设置模型
            if not self.setup_model():
                logger.error("重新设置模型失败")
                return False
            
            # 清理旧模型
            del old_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 🔥 新功能2：继承旧的LoRA权重到新模型
            if self.config.enable_weight_inheritance and old_lora_weights and old_rank and old_rank < new_rank:
                logger.info(f"🔄 开始继承LoRA权重: rank {old_rank} → {new_rank}")
                self._inherit_lora_weights(old_lora_weights, old_rank, new_rank)
            
            # 🔥 新功能3：条件优化器重置
            # 如果 skip_optimizer_reset=True（初始化阶段），跳过优化器重置
            # 因为此时数据加载器还未设置，优化器会在 easy 阶段正确初始化
            if skip_optimizer_reset:
                logger.info(f"✅ LoRA秩更新完成: {new_rank} (跳过优化器重置，将在 easy 阶段初始化)")
            else:
                # VQA 默认使用阶段重启模式（enable_optimizer_reset=True）
                enable_optimizer_reset = getattr(self.config, 'enable_optimizer_reset', True)
                if enable_optimizer_reset:
                    # 阶段重启模式：重新初始化优化器和调度器
                    logger.info("🔄 阶段重启模式：重新初始化优化器和调度器...")
                    if not self._setup_optimizer_for_progressive():
                        logger.error("重新初始化优化器失败")
                        return False
                    logger.info(f"✅ LoRA秩更新完成: {new_rank} (优化器已重置)")
                else:
                    # 连续训练模式：保持优化器状态，但需要更新优化器的参数组
                    logger.info("🔄 连续训练模式：更新优化器参数组...")
                    try:
                        # 获取新的可训练参数
                        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                        
                        # 更新优化器的param_groups
                        if self.optimizer is not None and self.optimizer.param_groups:
                            self.optimizer.param_groups[0]['params'] = trainable_params
                        
                        logger.info(f"✅ LoRA秩更新完成: {new_rank} (优化器状态保持)")
                    except Exception as e:
                        logger.error(f"更新优化器参数组失败: {str(e)}")
                        logger.warning("回退到重新初始化优化器...")
                        if not self._setup_optimizer_for_progressive():
                            logger.error("重新初始化优化器失败")
                            return False
            
            # 更新当前秩记录（用于下次权重继承）
            self._current_lora_rank = new_rank
            
            return True
            
        except Exception as e:
            logger.error(f"更新LoRA秩失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def progressive_train(self) -> bool:
        """
        执行渐进式LoRA训练（创新方法！）
        
        训练策略：
        - Stage 1 (easy): 只用简单样本，小LoRA秩 (r=32)
        - Stage 2 (medium): 简单+中等样本，中等LoRA秩 (r=64)
        - Stage 3 (hard): 所有样本，大LoRA秩 (r=128)
        
        Returns:
            训练是否成功
        """
        try:
            logger.info("🎯 开始渐进式LoRA训练（Progressive LoRA Training）...")
            logger.info("=" * 80)
            logger.info("📚 创新方法：结合课程学习(Curriculum Learning)与参数高效微调(LoRA)")
            logger.info("=" * 80)
            
            # 1) 初始化模型（使用 progressive_lora 配置），随后根据阶段动态调整 LoRA 秩
            logger.info(f"初始化模型，初始LoRA秩: {self.config.easy_lora_rank}")
            if not self.setup_model():
                return False

            # 初始 LoRA 秩（easy 阶段）
            stage_lora_ranks = {
                "easy": self.config.easy_lora_rank,
                "medium": self.config.medium_lora_rank,
                "hard": self.config.hard_lora_rank,
            }
            initial_rank = stage_lora_ranks["easy"]
            current_rank = getattr(self.model_loader.lora_config, "lora_r", initial_rank)
            self._current_lora_rank = current_rank

            if current_rank != initial_rank:
                logger.info(f"初始化阶段将 LoRA 秩从 {current_rank} 调整到 {initial_rank}")
                # 初始化阶段只更新模型和 LoRA 秩，不重置优化器（因为数据加载器还未设置）
                # 优化器会在 easy 阶段正确初始化
                if not self._update_model_lora_rank(initial_rank, skip_optimizer_reset=True):
                    return False
            else:
                logger.info(f"初始 LoRA 秩 = {initial_rank}")

            # 2) 构建复杂度分层与阶段样本
            logger.info("使用累积式渐进训练策略：Easy → Easy+Medium → All")
            splits = self._build_complexity_splits()
            id2sample = {int(s["qa_id"]): s for s in self.train_samples}

            stages = {
                "easy": splits["easy"],
                "medium": splits["easy"] + splits["medium"],
                "hard": splits["easy"] + splits["medium"] + splits["hard"],
            }
            stage_epochs = {
                "easy": self.config.easy_epochs,
                "medium": self.config.medium_epochs,
                "hard": self.config.hard_epochs,
            }
            best_val_acc = 0.0

            # 显示训练配置摘要
            self._log_training_summary()
            self._log_progressive_training_summary(stage_epochs, stage_lora_ranks, self.config.complexity_thresholds)

            # ==================== 三阶段渐进式训练 ====================
            total_epochs = 0
            
            for stage_name in ['easy', 'medium', 'hard']:
                logger.info("\n" + "=" * 80)
                logger.info(f"🌟 Stage: {stage_name.upper()} | LoRA Rank: {stage_lora_ranks[stage_name]} | Epochs: {stage_epochs[stage_name]}")
                logger.info("=" * 80)
                
                # 更新当前阶段的数据加载器（必须在调整LoRA秩之前）
                # 因为优化器重新初始化需要使用新阶段的数据加载器来计算步数
                stage_ids = stages[stage_name]
                stage_samples = [id2sample[qid] for qid in stage_ids]
                train_loader = self._build_dataloader(stage_samples, shuffle=True)
                self.train_dataloader = train_loader
                
                # 设置当前阶段的总epoch数（用于正确计算训练进度）
                self._current_stage_total_epochs = stage_epochs[stage_name]
                
                # 记录当前阶段开始时的global_step（用于计算阶段内进度）
                if not hasattr(self, 'global_step'):
                    self.global_step = 0
                self._stage_start_step = self.global_step
                
                # 动态调整LoRA秩（除了第一阶段）
                if stage_name != 'easy':
                    logger.info(f"📊 动态调整LoRA秩: {stage_lora_ranks[stage_name]}")
                    if not self._update_model_lora_rank(stage_lora_ranks[stage_name]):
                        logger.error(f"LoRA秩更新失败，中止训练")
                        return False
                else:
                    # Easy阶段：初始化优化器（使用当前阶段的步数）
                    logger.info(f"📊 初始化优化器（Easy阶段）")
                    if not self._setup_optimizer_for_progressive():
                        logger.error(f"优化器初始化失败，中止训练")
                        return False
                    
                    # 记录当前LoRA秩（用于后续阶段的权重继承）
                    self._current_lora_rank = stage_lora_ranks['easy']
                    logger.info(f"记录初始LoRA秩: {self._current_lora_rank}")
                
                # 训练当前阶段
                
                for stage_epoch in range(stage_epochs[stage_name]):
                    self.current_epoch = total_epochs
                    epoch_in_stage = stage_epoch + 1
                    
                    param_stats = self._compute_parameter_stats()
                    # 获取当前学习率：优先从 optimizer 获取，如果 scheduler 已初始化则从 scheduler 获取
                    if self.optimizer is not None and len(self.optimizer.param_groups) > 0:
                        current_lr = self.optimizer.param_groups[0]['lr']
                    elif self.scheduler is not None:
                        try:
                            current_lr = self.scheduler.get_last_lr()[0]
                        except (IndexError, AttributeError):
                            current_lr = self.config.learning_rate
                    else:
                        current_lr = self.config.learning_rate
                    logger.info(
                        f"🚀 Stage [{stage_name}] Epoch {epoch_in_stage}/{stage_epochs[stage_name]} "
                        f"(Global Epoch {total_epochs + 1}): "
                        f"trainable_params={param_stats['trainable_params']:,}, "
                        f"total_params={param_stats['total_params']:,}, "
                        f"trainable_ratio={param_stats['trainable_ratio']:.2%}, "
                        f"lora_rank={stage_lora_ranks[stage_name]}, "
                        f"lr={current_lr:.2e}"
                    )
                    
                    # 训练一个epoch
                    epoch_avg_loss = self.train_one_epoch(train_loader, epoch_in_stage, stage_name)
                    
                    # 验证
                    logger.info(f"🔍 Stage [{stage_name}] Epoch {epoch_in_stage} 验证...")
                    val_loss, val_acc = self.validate()
                    
                    # 总结
                    improvement = "✅" if val_acc > best_val_acc else "❌"
                    logger.info(
                        f"📈 Stage [{stage_name}] Epoch {epoch_in_stage} 总结: "
                        f"train_loss={epoch_avg_loss:.4f}, "
                        f"val_loss={val_loss:.4f}, "
                        f"val_accuracy={val_acc:.4f}, "
                        f"best_accuracy={max(best_val_acc, val_acc):.4f}, "
                        f"improvement={improvement}"
                    )
                    
                    # 保存检查点
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        save_dir = os.path.join(self.config.output_dir, "best_vqa7w_adapter")
                        os.makedirs(save_dir, exist_ok=True)
                        if self.model_loader.save_lora_adapter(save_dir):
                            logger.info(f"💾 保存最佳模型 (Accuracy={best_val_acc:.4f})")
                        else:
                            logger.warning("⚠️  保存最佳模型失败")
                    
                    total_epochs += 1
                
                logger.info(f"✅ Stage [{stage_name}] 完成！")
                
                # 清理阶段特定的临时变量
                self._current_stage_total_epochs = None
                self._stage_start_step = 0
            
            # ==================== 训练完成 ====================
            
            # 保存最终LoRA适配器
            if self.config.enable_lora:
                final_adapter_path = os.path.join(self.config.output_dir, "final_progressive")
                if self.model_loader.save_lora_adapter(final_adapter_path):
                    logger.info(f"最终渐进式LoRA适配器已保存: {os.path.abspath(final_adapter_path)}")
            
            # 完成摘要
            logger.info("\n" + "=" * 80)
            logger.info("🎉 渐进式LoRA训练完成！")
            logger.info(f"📊 训练统计:")
            logger.info(f"   - 总训练轮数: {total_epochs}")
            logger.info(f"   - Easy阶段: {stage_epochs['easy']} epochs (rank={stage_lora_ranks['easy']})")
            logger.info(f"   - Medium阶段: {stage_epochs['medium']} epochs (rank={stage_lora_ranks['medium']})")
            logger.info(f"   - Hard阶段: {stage_epochs['hard']} epochs (rank={stage_lora_ranks['hard']})")
            logger.info(f"🏆 最佳验证指标:")
            logger.info(f"   - ⭐ VQA7W Accuracy: {best_val_acc:.4f}")
            logger.info("=" * 80)
            
            # 测试前加载最佳模型（按照图像描述的处理方式：重新创建model_loader）
            best_adapter_path = os.path.join(self.config.output_dir, "best_vqa7w_adapter")
            if os.path.exists(best_adapter_path):
                logger.info(f"📥 加载最佳验证集模型用于测试: {best_adapter_path}")
                try:
                    # 按照图像描述的处理方式：重新创建model_loader，先加载基础模型，再加载适配器
                    test_model_loader = create_lora_model_loader(
                        model_path=self.config.model_path,
                        lora_config_name=self.config.lora_config_name if hasattr(self.config, 'lora_config_name') else "progressive_lora",
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
            logger.error(f"渐进式训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


