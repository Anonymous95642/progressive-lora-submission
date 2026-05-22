"""
=============================================================================
LLaVA + Progressive LoRA 训练器 (COCO Trainer)
=============================================================================

本模块实现了LLaVA模型在图像描述任务上的完整训练流程，是项目的核心模块。

【核心功能】
1. 标准LoRA训练：传统的LoRA微调（用于对比实验）
2. 渐进式LoRA训练：Progressive LoRA训练（本项目核心创新）
3. 训练管理：检查点保存、验证集评估、训练历史记录

【主要类】
- COCOTrainingConfig: 训练配置类
  包含所有训练参数（学习率、batch size、LoRA配置等）

- COCOTrainer: 训练器主类
  实现完整的训练、验证、保存流程

【核心方法】
- train(): 标准LoRA训练流程（用于对比）
- progressive_train(): 渐进式训练流程（本项目核心）
  └── 阶段1: Easy阶段   (rank=32,  简单样本)
  └── 阶段2: Medium阶段 (rank=64,  中等样本)
  └── 阶段3: Hard阶段   (rank=128, 困难样本)
  
- _train_one_epoch(): 单轮训练实现
- validate(): 验证集评估（计算Loss和BLEU/CIDEr等指标）

【渐进式训练的关键技术】
1. 样本复杂度分层（coco_dataset.py）
   - 基于语言学特征评估描述复杂度
   - 按百分位数自动分层（避免硬编码阈值）

2. LoRA秩动态扩展（lora_config.py）
   - 使用SVD正交化方法扩展权重矩阵
   - 保持旧维度冻结，新维度与旧维度正交
   - 确保知识继承和训练稳定性

3. 阶段间优化器重置
   - 每个阶段独立的optimizer和scheduler
   - 避免动量和学习率调度的干扰

4. 最佳模型选择
   - 基于验证集CIDEr指标选择最佳模型（而非Loss）
   - 符合学术界标准做法（使用主要评估指标）

【数据集支持】
- COCO 2017 Caption（主要）
- Flickr30K（通过适配器支持）

【输出文件】
- checkpoints/: 训练检查点（每N步或每个阶段结束保存）
- lora_adapters/best_model: 基于验证集CIDEr的最佳模型
- lora_adapters/final_progressive: 最终训练完成的模型
- progressive_training_history.json: 训练历史和最佳指标

"""

import os
# 设置tokenizers并行化环境变量，避免fork警告
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from tqdm import tqdm
import numpy as np

from model_loader import LLaVAModelLoader
from coco_dataset import COCODatasetConfig, COCODataLoader
from lora_config import LoRAConfig, get_lora_config

# 配置日志
logger = logging.getLogger(__name__)

class COCOTrainingConfig:
    """
    COCO训练配置类
    
    包含训练所需的所有参数配置：
    - 模型和数据路径
    - LoRA配置
    - 训练超参数（学习率、batch size、epochs等）
    - 优化器和调度器配置
    - 保存和验证策略
    
    支持从LoRA配置文件加载参数，实现配置的统一管理。
    """
    
    def __init__(self, lora_config_name: str = None):
        """
        初始化训练配置
        
        Args:
            lora_config_name: LoRA配置名称，如果提供则启用LoRA训练
                            支持的配置：default_48gb, progressive_lora, fair_comparison等
        """
        # 基础配置
        self.model_path = "/root/autodl-tmp/llava-1.5-7b"
        self.coco_data_root = "/root/autodl-tmp/COCO2017"
        self.output_dir = os.path.join(os.getcwd(), "coco_training_output")  # 确保在当前项目目录
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.logs_dir = os.path.join(self.output_dir, "logs")
        
        # LoRA配置
        self.lora_config_name = lora_config_name
        # 暂时存储LoRA配置名称，稍后应用
        self.enable_lora = lora_config_name is not None
        self.lora_config = None
        
        # 训练参数
        self.num_epochs = 5
        self.batch_size = 16  # 默认批次大小，会被LoRA配置覆盖
        self.learning_rate = 2e-5
        self.weight_decay = 0.01
        self.warmup_steps = 1000
        self.warmup_ratio = 0.1  # 默认warmup比例
        self.max_grad_norm = 1.0
        
        # 学习率调度器默认配置
        self.scheduler_type = "cosine"  # 默认使用余弦退火
        self.cosine_num_cycles = 0.5
        
        # 渐进式训练高级配置
        self.enable_optimizer_reset = True  # 是否在阶段切换时重置优化器（True=阶段重启，False=连续训练）
        self.enable_weight_inheritance = True  # 是否启用LoRA权重继承（将小rank权重扩展到大rank）
        
        # 模型参数
        self.load_in_8bit = False
        self.load_in_4bit = False
        self.max_new_tokens = 50  # 最优值：覆盖模型实际生成长度（23±3词）+ 系统开销，防止截断
        self.temperature = 0.7
        
        # 保存和验证
        self.save_steps = 1000
        self.eval_steps = 2000
        self.logging_steps = 10  # 更频繁的日志记录
        self.save_total_limit = 3
        
        # 数据处理
        self.num_workers = 8  # 默认工作进程数，会被LoRA配置覆盖
        self.pin_memory = True
        self.gradient_accumulation_steps = 1  # 默认梯度累积步数，会被LoRA配置覆盖
        self.max_train_samples = None  # 最大训练样本数
        self.max_val_samples = None    # 最大验证样本数
        self.max_eval_samples = None   # 最大评估样本数
        self.max_test_samples = None   # 最大测试样本数
        self.max_caption_length = 200  # 最大标题长度
        self.spice_cache_fix = True    # SPICE缓存修复
        
        # 应用LoRA配置（在默认参数设置之后）
        if self.enable_lora:
            self.lora_config = get_lora_config(lora_config_name)
            # 使用LoRA配置覆盖相关参数
            self._apply_lora_config()
            
            # 直接从配置文件加载额外参数（不在LoRAConfig类中的参数）
            self._load_extra_config_params(lora_config_name)
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
    
    def _apply_lora_config(self):
        """应用LoRA配置参数"""
        if not self.lora_config:
            return
            
        # 从LoRA配置中获取训练参数
        if hasattr(self.lora_config, 'batch_size'):
            self.batch_size = self.lora_config.batch_size
        if hasattr(self.lora_config, 'gradient_accumulation_steps'):
            self.gradient_accumulation_steps = self.lora_config.gradient_accumulation_steps
        if hasattr(self.lora_config, 'lora_learning_rate'):
            self.learning_rate = self.lora_config.lora_learning_rate
        if hasattr(self.lora_config, 'lora_weight_decay'):
            self.weight_decay = self.lora_config.lora_weight_decay
        if hasattr(self.lora_config, 'lora_warmup_ratio'):
            # 计算warmup步数 - 这里需要在后面的setup_optimizer中重新计算
            self.warmup_ratio = self.lora_config.lora_warmup_ratio
        
        # 学习率调度器配置
        if hasattr(self.lora_config, 'scheduler_type'):
            self.scheduler_type = self.lora_config.scheduler_type
        else:
            self.scheduler_type = "cosine"  # 默认使用余弦退火
            
        if hasattr(self.lora_config, 'cosine_num_cycles'):
            self.cosine_num_cycles = self.lora_config.cosine_num_cycles
        else:
            self.cosine_num_cycles = 0.5
        
        if hasattr(self.lora_config, 'use_4bit_quantization'):
            self.load_in_4bit = self.lora_config.use_4bit_quantization
        if hasattr(self.lora_config, 'use_8bit_quantization'):
            self.load_in_8bit = self.lora_config.use_8bit_quantization
        if hasattr(self.lora_config, 'dataloader_num_workers'):
            self.num_workers = self.lora_config.dataloader_num_workers
        if hasattr(self.lora_config, 'dataloader_pin_memory'):
            self.pin_memory = self.lora_config.dataloader_pin_memory
            
        # 确保梯度累积步数存在
        if not hasattr(self, 'gradient_accumulation_steps'):
            self.gradient_accumulation_steps = 1
            
        logger.info("COCO训练配置初始化完成")
    
    def update_output_paths(self):
        """更新所有与output_dir相关的派生路径"""
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.logs_dir = os.path.join(self.output_dir, "logs")
        self.lora_adapters_dir = os.path.join(self.output_dir, "lora_adapters")
        
        # 创建所有目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        if self.enable_lora:
            os.makedirs(self.lora_adapters_dir, exist_ok=True)
    
    def _load_extra_config_params(self, config_name: str):
        """加载配置文件中不在LoRAConfig类中的额外参数"""
        import json
        from pathlib import Path
        
        # 构建配置文件路径
        config_file = Path("./configs") / f"{config_name}.json"
        
        if not config_file.exists():
            logger.warning(f"配置文件不存在: {config_file}")
            return
            
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # 加载数据采样相关参数
            if 'max_train_samples' in config_dict:
                self.max_train_samples = config_dict['max_train_samples']
            if 'max_val_samples' in config_dict:
                self.max_val_samples = config_dict['max_val_samples']
            if 'max_eval_samples' in config_dict:
                self.max_eval_samples = config_dict['max_eval_samples']
            if 'max_test_samples' in config_dict:
                self.max_test_samples = config_dict['max_test_samples']
            if 'max_caption_length' in config_dict:
                self.max_caption_length = config_dict['max_caption_length']
            if 'spice_cache_fix' in config_dict:
                self.spice_cache_fix = config_dict['spice_cache_fix']
            
            # 加载生成参数
            if 'max_new_tokens' in config_dict:
                self.max_new_tokens = config_dict['max_new_tokens']
            if 'temperature' in config_dict:
                self.temperature = config_dict['temperature']
                
            logger.info(f"从配置文件加载额外参数: {config_file}")
            
        except Exception as e:
            logger.error(f"加载额外配置参数失败: {e}")

class COCOTrainer:
    """COCO训练器"""
    
    def __init__(self, config: COCOTrainingConfig):
        """
        初始化COCO训练器
        
        Args:
            config: 训练配置对象
        """
        self.config = config
        
        # 更新输出路径（确保使用正确的output_dir）
        self.config.update_output_paths()
        
        self.model_loader = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.optimizer = None
        self.scheduler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_weighted_score = 0.0  # 加权综合得分（主要选择标准）
        self.best_val_cider = 0.0
        self.best_val_bleu_sum = 0.0  # BLEU总和（用于记录）
        # 详细的最佳验证指标
        self.best_val_bleu_1 = 0.0
        self.best_val_bleu_2 = 0.0
        self.best_val_bleu_3 = 0.0
        self.best_val_bleu_4 = 0.0
        self.best_val_rouge_l = 0.0
        self.best_val_meteor = 0.0
        self.training_history = []
        
        # 加权综合指标的权重配置
        self.metric_weights = {
            'cider': 0.55,    # 主导指标（55%）
            'rouge_l': 0.30,  # 流畅性（30%）
            'bleu_4': 0.15    # 基准（15%）
        }
        logger.info(f"📊 使用加权综合指标选择最佳模型:")
        logger.info(f"   - CIDEr: {self.metric_weights['cider']*100:.0f}% (主导)")
        logger.info(f"   - ROUGE-L: {self.metric_weights['rouge_l']*100:.0f}% (流畅性)")
        logger.info(f"   - BLEU-4: {self.metric_weights['bleu_4']*100:.0f}% (基准)")
        
        # 渐进式训练状态跟踪（用于正确计算步数和进度）
        self._current_stage_total_epochs = None  # 当前阶段的总epoch数
        self._stage_start_step = 0  # 当前阶段开始时的global_step
        
        logger.info(f"COCO训练器初始化完成，使用设备: {self.device}")
    
    def setup_model(self) -> bool:
        """设置模型"""
        try:
            logger.info("正在加载LLaVA模型...")
            self.model_loader = LLaVAModelLoader(
                model_path=self.config.model_path,
                lora_config=self.config.lora_config
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
            self.training_model = self.model_loader.get_trainable_model()
            
            # 设置模型为训练模式
            self.training_model.train()
            
            if self.config.enable_lora:
                logger.info("LoRA微调模式已启用")
                # 打印可训练参数信息
                if hasattr(self.training_model, 'print_trainable_parameters'):
                    self.training_model.print_trainable_parameters()
            
            logger.info("模型设置完成")
            return True
            
        except Exception as e:
            logger.error(f"模型设置失败: {str(e)}")
            return False
    
    def setup_data(self) -> bool:
        """设置数据加载器"""
        try:
            logger.info("正在设置数据加载器...")
            
            # 自动检测数据集类型（支持 COCO / Flickr8k / Flickr30K / COCO Karpathy / VizWiz-Captions）
            data_root = self.config.coco_data_root
            lower_root = data_root.lower()
            
            if 'flickr8k' in lower_root:
                logger.info("检测到Flickr8k数据集，使用适配器...")
                from flickr8k_adapter import Flickr8kAdapter
                coco_config = Flickr8kAdapter(data_root)
            # 如果路径包含flickr30k或通用flickr，使用Flickr30K适配器
            elif 'flickr30k' in lower_root or 'flickr' in lower_root:
                logger.info("检测到Flickr30K数据集，使用适配器...")
                from flickr30k_adapter import Flickr30KAdapter
                coco_config = Flickr30KAdapter(data_root)
            elif 'vizwiz' in lower_root:
                logger.info("检测到VizWiz-Captions数据集，使用适配器...")
                from vizwiz_adapter import VizWizCaptionAdapter
                coco_config = VizWizCaptionAdapter(data_root)
            elif 'karpathy' in lower_root or 'coco2014' in lower_root:
                logger.info("检测到COCO Karpathy数据集，使用适配器...")
                from coco_karpathy_adapter import COCOKarpathyAdapter
                coco_config = COCOKarpathyAdapter(data_root)
            else:
                logger.info("使用COCO数据集...")
                coco_config = COCODatasetConfig(data_root)
            
            if not coco_config.validate_paths():
                logger.error("数据集路径验证失败")
                return False
            
            # 创建数据加载器（COCODataLoader可以处理适配器）
            data_loader = COCODataLoader(coco_config)
            
            # 训练集数据加载器
            self.train_dataloader = data_loader.create_dataloader(
                split="train",
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                max_samples=self.config.max_train_samples
            )
            
            # 验证集数据加载器
            self.val_dataloader = data_loader.create_dataloader(
                split="val",
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                max_samples=self.config.max_val_samples
            )
            
            logger.info(f"数据加载器设置完成")
            logger.info(f"训练集批次数: {len(self.train_dataloader)}")
            logger.info(f"验证集批次数: {len(self.val_dataloader)}")
            
            return True
            
        except Exception as e:
            logger.error(f"数据设置失败: {str(e)}")
            return False
    
    def setup_optimizer(self) -> bool:
        """设置优化器和学习率调度器"""
        try:
            logger.info("正在设置优化器...")
            
            # 获取可训练参数
            trainable_params = []
            for name, param in self.model_loader.model.named_parameters():
                if param.requires_grad:
                    trainable_params.append(param)
            
            logger.info(f"可训练参数数量: {len(trainable_params)}")
            
            # 创建优化器，添加更稳定的参数
            self.optimizer = AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                eps=1e-8,  # 增加数值稳定性
                betas=(0.9, 0.999),  # 使用标准的beta值
                amsgrad=False,  # 禁用amsgrad以避免额外的状态
                maximize=False,
                foreach=None,  # 让PyTorch自动选择最优实现
                capturable=False,
                differentiable=False,
                fused=None  # 让PyTorch自动选择是否使用融合实现
            )
            
            # 计算总训练步数
            total_steps = len(self.train_dataloader) * self.config.num_epochs
            
            # 计算warmup步数
            if hasattr(self.config, 'warmup_ratio'):
                warmup_steps = int(total_steps * self.config.warmup_ratio)
            else:
                warmup_steps = self.config.warmup_steps
            
            # 根据配置选择学习率调度器
            scheduler_type = getattr(self.config, 'scheduler_type', 'cosine')
            
            if scheduler_type == "linear":
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps
                )
                logger.info(f"使用线性学习率调度器，warmup步数: {warmup_steps}")
                
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
            
            logger.info(f"优化器设置完成，总训练步数: {total_steps}，warmup步数: {warmup_steps}")
            return True
            
        except Exception as e:
            logger.error(f"优化器设置失败: {str(e)}")
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
            for name, param in self.model_loader.model.named_parameters():
                if param.requires_grad:
                    trainable_params.append(param)
            
            logger.info(f"可训练参数数量: {len(trainable_params)}")
            
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
            steps_per_epoch = len(self.train_dataloader)
            stage_total_epochs = self._current_stage_total_epochs if self._current_stage_total_epochs is not None else 1
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
    
    def _setup_progressive_optimizer(self, total_train_steps: int) -> bool:
        """
        为渐进式训练设置优化器和学习率调度器（首次初始化）
        
        Args:
            total_train_steps: 所有阶段的总训练步数
            
        Returns:
            是否设置成功
        """
        try:
            logger.info("正在为渐进式训练设置优化器...")
            
            # 获取可训练参数
            trainable_params = []
            for name, param in self.model_loader.model.named_parameters():
                if param.requires_grad:
                    trainable_params.append(param)
            
            logger.info(f"可训练参数数量: {len(trainable_params)}")
            
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
            
            # 计算warmup步数
            if hasattr(self.config, 'warmup_ratio'):
                warmup_steps = int(total_train_steps * self.config.warmup_ratio)
            else:
                warmup_steps = self.config.warmup_steps
            
            # 根据配置选择学习率调度器
            scheduler_type = getattr(self.config, 'scheduler_type', 'cosine')
            
            if scheduler_type == "linear":
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_train_steps
                )
                logger.info(f"使用线性学习率调度器")
                
            elif scheduler_type == "cosine_with_restarts":
                from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
                num_cycles = getattr(self.config, 'cosine_num_cycles', 1.0)
                self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_train_steps,
                    num_cycles=int(num_cycles)
                )
                logger.info(f"使用带重启的余弦学习率调度器，cycles: {int(num_cycles)}")
                
            else:  # 默认使用余弦退火
                num_cycles = getattr(self.config, 'cosine_num_cycles', 0.5)
                self.scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_train_steps,
                    num_cycles=num_cycles
                )
                logger.info(f"使用余弦退火学习率调度器，cycles: {num_cycles}")
            
            logger.info(f"渐进式训练优化器设置完成")
            logger.info(f"  - 总训练步数: {total_train_steps}")
            logger.info(f"  - Warmup步数: {warmup_steps}")
            logger.info(f"  - 初始学习率: {self.config.learning_rate:.2e}")
            return True
            
        except Exception as e:
            logger.error(f"渐进式训练优化器设置失败: {str(e)}")
            return False
    
    def reset_optimizer_state(self):
        """重置优化器状态（用于错误恢复）"""
        try:
            logger.info("重置优化器状态...")
            if self.optimizer is not None:
                # 清理所有优化器状态
                self.optimizer.state.clear()
                # 重新初始化优化器
                for group in self.optimizer.param_groups:
                    for p in group['params']:
                        if p.grad is not None:
                            p.grad.detach_()
                            p.grad.zero_()
            logger.info("优化器状态重置完成")
        except Exception as e:
            logger.error(f"重置优化器状态失败: {str(e)}")
    
    def compute_loss(self, batch: Dict) -> torch.Tensor:
        """计算训练损失"""
        try:
            images = batch['images']
            captions = batch['captions']
            
            total_loss = 0.0
            batch_size = len(images)
            valid_samples = 0
            
            for i in range(batch_size):
                image = images[i]
                caption = captions[i]
                
                # 构建对话格式
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in detail."},
                            {"type": "image"},
                        ],
                    }
                ]
                
                # 处理输入
                text_prompt = self.model_loader.processor.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True
                )
                
                # 添加目标回答
                full_text = text_prompt + caption + self.model_loader.processor.tokenizer.eos_token
                
                # 预检查token长度（预留空间给图像tokens约576个）
                token_length = len(self.model_loader.processor.tokenizer.encode(full_text))
                if token_length > 3500:  # 留出空间给视觉tokens（约576）和生成tokens
                    logger.warning(f"训练样本token过长({token_length})，跳过此样本")
                    continue
                
                # 编码
                inputs = self.model_loader.processor(
                    text=full_text,
                    images=image,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=4096  # 统一使用4096作为最大长度
                )
                
                # 移动到设备
                for k, v in inputs.items():
                    if hasattr(v, 'to'):
                        inputs[k] = v.to(self.device)
                
                # 前向传播
                outputs = self.model_loader.model(**inputs)
                
                # 计算损失（只对生成部分计算损失）
                logits = outputs.logits
                labels = inputs['input_ids'].clone()
                
                # 掩码：只对回答部分计算损失
                # 正确计算prompt长度：先处理只有prompt的输入，获取实际长度（包含图像tokens）
                prompt_inputs = self.model_loader.processor(
                    text=text_prompt,
                    images=image,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=4096,
                )
                prompt_length = prompt_inputs["input_ids"].shape[1]
                # 安全检查：确保prompt_length不超过labels长度
                prompt_length = min(prompt_length, labels.shape[1])
                labels[:, :prompt_length] = -100
                
                # 计算交叉熵损失
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                total_loss += loss
                valid_samples += 1
            
            # 如果没有有效样本，返回零损失
            if valid_samples == 0:
                logger.warning("批次中没有有效样本，返回零损失")
                return torch.tensor(0.0, requires_grad=True, device=self.device)
            
            return total_loss / valid_samples
            
        except Exception as e:
            logger.error(f"损失计算失败: {str(e)}")
            # 返回一个有效的损失张量，确保在正确的设备上
            return torch.tensor(0.0, requires_grad=True, device=self.device)
    
    def train_epoch(self) -> Dict:
        """训练一个epoch"""
        self.model_loader.model.train()
        
        total_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        # 初始化训练开始时间
        if not hasattr(self, '_training_start_time'):
            self._training_start_time = time.time()
        
        # 创建进度条
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # 清零梯度（在前向传播之前）
                self.optimizer.zero_grad()
                
                # 前向传播
                loss = self.compute_loss(batch)
                
                # 检查损失是否有效
                if not torch.isfinite(loss):
                    logger.warning(f"检测到无效损失值: {loss.item()}，跳过此批次")
                    continue
                
                # 反向传播
                loss.backward()
                
                # 计算梯度统计（在裁剪前）
                grad_norm_before = 0.0
                grad_count = 0
                for param in self.model_loader.model.parameters():
                    if param.grad is not None:
                        grad_norm_before += param.grad.data.norm(2).item() ** 2
                        grad_count += 1
                grad_norm_before = grad_norm_before ** 0.5 if grad_count > 0 else 0.0
                
                # 梯度裁剪
                grad_norm_after = torch.nn.utils.clip_grad_norm_(
                    self.model_loader.model.parameters(),
                    self.config.max_grad_norm
                )
                
                # 优化器步进
                self.optimizer.step()
                self.scheduler.step()
                
                # 更新统计
                total_loss += loss.item()
                self.global_step += 1
                
                # 计算当前学习率
                current_lr = self.scheduler.get_last_lr()[0]
                
                # 更新进度条（包含更多详细指标）
                avg_loss = total_loss / (batch_idx + 1)
                
                # 计算内存使用情况
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                    gpu_cached = torch.cuda.memory_reserved() / 1024**3   # GB
                    gpu_total = gpu_memory + gpu_cached  # 总占用
                else:
                    gpu_memory = gpu_cached = gpu_total = 0.0
                
                # 计算训练速度 (samples/sec)
                if hasattr(self, '_batch_start_time'):
                    batch_time = time.time() - self._batch_start_time
                    samples_per_sec = self.config.batch_size / batch_time if batch_time > 0 else 0
                else:
                    samples_per_sec = 0
                    
                self._batch_start_time = time.time()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'grad_norm': f'{grad_norm_after:.3f}',
                    'gpu_mem': f'{gpu_total:.1f}GB',
                    'samples/s': f'{samples_per_sec:.1f}'
                })
                
                # 定期详细日志记录
                if self.global_step % self.config.logging_steps == 0:
                    # 计算参数统计
                    param_stats = self._compute_parameter_stats()
                    
                    # 计算训练进度 - 区分全局进度和阶段进度
                    if self._current_stage_total_epochs is not None:
                        # 渐进式训练：显示阶段内的进度
                        stage_total_steps = len(self.train_dataloader) * self._current_stage_total_epochs
                        stage_current_step = self.global_step - self._stage_start_step
                        progress_percent = (stage_current_step / stage_total_steps) * 100 if stage_total_steps > 0 else 0
                        step_display = f"{stage_current_step}/{stage_total_steps}"
                        
                        # ETA基于阶段进度
                        if hasattr(self, '_training_start_time'):
                            elapsed_time = time.time() - self._training_start_time
                            if stage_current_step > 0:
                                time_per_step = elapsed_time / self.global_step
                                remaining_steps = stage_total_steps - stage_current_step
                                eta_seconds = remaining_steps * time_per_step
                                eta_str = f"{eta_seconds/60:.1f}m" if eta_seconds < 3600 else f"{eta_seconds/3600:.1f}h"
                            else:
                                eta_str = "N/A"
                        else:
                            eta_str = "N/A"
                    else:
                        # 普通训练：显示全局进度
                        total_steps = len(self.train_dataloader) * self.config.num_epochs
                        progress_percent = (self.global_step / total_steps) * 100 if total_steps > 0 else 0
                        step_display = f"{self.global_step}/{total_steps}"
                        
                        # ETA基于全局进度
                        if hasattr(self, '_training_start_time'):
                            elapsed_time = time.time() - self._training_start_time
                            if self.global_step > 0:
                                time_per_step = elapsed_time / self.global_step
                                remaining_steps = total_steps - self.global_step
                                eta_seconds = remaining_steps * time_per_step
                                eta_str = f"{eta_seconds/60:.1f}m" if eta_seconds < 3600 else f"{eta_seconds/3600:.1f}h"
                            else:
                                eta_str = "N/A"
                        else:
                            eta_str = "N/A"
                    
                    logger.info(
                        f"📊 Step {step_display} ({progress_percent:.1f}%) [Global: {self.global_step}]: "
                        f"loss={loss.item():.4f}, avg_loss={avg_loss:.4f}, "
                        f"lr={current_lr:.2e}, grad_norm={grad_norm_after:.3f}, "
                        f"gpu_mem={gpu_total:.1f}GB, samples/s={samples_per_sec:.1f}, "
                        f"ETA={eta_str}, grad_clipped={'✓' if grad_norm_before > self.config.max_grad_norm else '✗'}"
                    )
                
                # 定期保存检查点
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
                
                # 定期验证
                if self.global_step % self.config.eval_steps == 0:
                    logger.info(f"🔍 开始第 {self.global_step} 步验证...")
                    val_metrics = self.validate()
                    
                    # 详细的验证结果日志
                    logger.info(
                        f"📊 验证结果 (Step {self.global_step}): "
                        f"val_loss={val_metrics['val_loss']:.4f}, "
                        f"best_val_loss={val_metrics['best_val_loss']:.4f}, "
                        f"valid_batches={val_metrics['valid_batches']}, "
                        f"improvement={'✅' if val_metrics['improvement'] else '❌'}, "
                        f"loss_diff={val_metrics.get('loss_diff', 0):.4f}"
                    )
                    
                    self.model_loader.model.train()  # 切回训练模式
                
            except Exception as e:
                logger.error(f"训练批次失败: {str(e)}")
                # 清理梯度以防止优化器状态损坏
                self.optimizer.zero_grad()
                
                # 检查是否是优化器状态相关错误
                error_str = str(e)
                is_optimizer_error = any(keyword in error_str.lower() for keyword in [
                    "cuda out of memory", "exp_avg_sq", "exp_avg", "keyerror", 
                    "optimizer", "state_dict", "momentum_buffer"
                ])
                
                if is_optimizer_error:
                    logger.warning("检测到优化器状态错误，尝试重置优化器状态")
                    try:
                        # 安全地重置优化器状态
                        for group in self.optimizer.param_groups:
                            for p in group['params']:
                                if p.grad is not None:
                                    p.grad.detach_()
                                    p.grad.zero_()
                        
                        # 清理优化器状态
                        if hasattr(self.optimizer, 'state'):
                            self.optimizer.state.clear()
                        
                        # 清理GPU缓存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                        logger.info("优化器状态重置成功")
                    except Exception as reset_e:
                        logger.error(f"重置优化器状态失败: {str(reset_e)}")
                        # 如果重置失败，尝试重新创建优化器
                        try:
                            logger.warning("尝试重新创建优化器")
                            # 获取可训练参数
                            trainable_params = [p for p in self.model_loader.model.parameters() if p.requires_grad]
                            
                            self.optimizer = torch.optim.AdamW(
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
                            logger.info("优化器重新创建成功")
                        except Exception as recreate_e:
                            logger.error(f"重新创建优化器失败: {str(recreate_e)}")
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
            f"lr={self.scheduler.get_last_lr()[0]:.2e}, "
            f"trainable_params={param_stats['trainable_params']:,}, "
            f"param_norm={param_stats['param_norm']:.3f}"
        )
        
        # 返回epoch统计
        epoch_metrics = {
            'epoch': self.current_epoch + 1,
            'avg_loss': avg_loss,
            'learning_rate': self.scheduler.get_last_lr()[0],
            'global_step': self.global_step,
            'total_batches': num_batches,
            'param_stats': param_stats
        }
        
        return epoch_metrics
    
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
        logger.info(f"  - 预热步数: {self.config.warmup_steps}")
        
        logger.info(f"📋 数据配置:")
        logger.info(f"  - 训练样本: {len(self.train_dataloader.dataset):,}")
        logger.info(f"  - 验证样本: {len(self.val_dataloader.dataset):,}")
        logger.info(f"  - 训练批次: {len(self.train_dataloader):,}")
        logger.info(f"  - 验证批次: {len(self.val_dataloader):,}")
        
        logger.info(f"⚙️ 训练策略:")
        logger.info(f"  - 日志步数: {self.config.logging_steps}")
        logger.info(f"  - 验证步数: {self.config.eval_steps}")
        logger.info(f"  - 保存步数: {self.config.save_steps}")
        logger.info(f"  - 输出目录: {os.path.abspath(self.config.output_dir)}")
        
        logger.info("=" * 80)
    
    def _log_training_completion_summary(self):
        """显示训练完成摘要"""
        if not self.training_history:
            return
            
        # 计算训练统计
        total_epochs = len(self.training_history)
        final_metrics = self.training_history[-1]
        initial_loss = self.training_history[0]['avg_loss'] if self.training_history else 0
        final_loss = final_metrics['avg_loss']
        loss_improvement = initial_loss - final_loss
        improvement_percent = (loss_improvement / initial_loss * 100) if initial_loss > 0 else 0
        
        logger.info("=" * 80)
        logger.info("🎉 训练完成摘要")
        logger.info("=" * 80)
        logger.info(f"📊 训练统计:")
        logger.info(f"  - 完成轮数: {total_epochs}/{self.config.num_epochs}")
        logger.info(f"  - 总训练步数: {final_metrics['global_step']:,}")
        logger.info(f"  - 最终学习率: {final_metrics['learning_rate']:.2e}")
        
        logger.info(f"📈 损失变化:")
        logger.info(f"  - 初始损失: {initial_loss:.4f}")
        logger.info(f"  - 最终损失: {final_loss:.4f}")
        logger.info(f"  - 损失改善: {loss_improvement:.4f} ({improvement_percent:+.1f}%)")
        logger.info(f"  - 最佳验证损失: {self.best_val_loss:.4f}")
        
        logger.info(f"🏆 最佳验证指标（选择策略：加权综合得分）:")
        logger.info(f"  - ⭐ 加权综合得分: {self.best_val_weighted_score:.4f}")
        logger.info(f"  - 分项指标:")
        logger.info(f"     • CIDEr (55%): {self.best_val_cider:.4f}")
        logger.info(f"     • ROUGE-L (30%): {self.best_val_rouge_l:.4f}")
        logger.info(f"     • BLEU-4 (15%): {self.best_val_bleu_4:.4f}")
        logger.info(f"  - BLEU总和: {self.best_val_bleu_sum:.4f}")
        logger.info(f"  - 最佳Loss: {self.best_val_loss:.4f}")
        
        logger.info(f"💾 保存的模型:")
        logger.info(f"  - 最佳模型: {os.path.abspath(os.path.join(self.config.output_dir, 'best_model'))}")
        logger.info(f"  - 最终模型: {os.path.abspath(os.path.join(self.config.output_dir, 'final_model'))}")
        logger.info(f"  - 训练历史: {os.path.abspath(os.path.join(self.config.output_dir, 'training_history.json'))}")
        
        # 计算最终参数统计
        param_stats = self._compute_parameter_stats()
        logger.info(f"🔧 最终参数状态:")
        logger.info(f"  - 可训练参数: {param_stats['trainable_params']:,}")
        logger.info(f"  - 参数范数: {param_stats['param_norm']:.3f}")
        
        logger.info("=" * 80)
    
    def _compute_parameter_stats(self) -> Dict:
        """
        计算模型参数统计信息
        
        修复说明：
        - 使用PEFT库的标准方法获取准确的参数统计
        - 解决了之前只统计主模型参数，遗漏Vision Encoder等模块的问题
        """
        # 方法1: 优先使用PEFT的标准方法（最准确）
        if hasattr(self.model_loader.model, 'get_nb_trainable_parameters'):
            trainable_params, total_params = self.model_loader.model.get_nb_trainable_parameters()
        else:
            # 方法2: 手动递归统计所有参数（包括所有子模块）
            trainable_params = 0
            total_params = 0
            
            # 使用named_parameters()递归获取所有参数（包括嵌套模块）
            for name, param in self.model_loader.model.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
        
        # 计算可训练参数的范数
        param_norm = 0.0
        for param in self.model_loader.model.parameters():
            if param.requires_grad:
                param_norm += param.data.norm(2).item() ** 2
        param_norm = param_norm ** 0.5
        
        return {
            'trainable_params': trainable_params,
            'total_params': total_params,
            'param_norm': param_norm,
            'trainable_ratio': trainable_params / total_params if total_params > 0 else 0.0
        }
    
    def validate(self) -> Dict:
        """
        验证模型 - 计算loss和评估指标
        
        Returns:
            Dict: 包含loss、指标和改善状态的字典
        """
        self.model_loader.model.eval()
        
        total_loss = 0.0
        valid_batches = 0
        max_batches = min(100, len(self.val_dataloader))  # 限制验证批次数
        
        # 记录验证开始时间
        val_start_time = time.time()
        
        # 创建验证进度条
        val_pbar = tqdm(
            enumerate(self.val_dataloader), 
            total=max_batches,
            desc="🔍 Validation (Loss)",
            leave=False
        )
        
        # 阶段1: 计算验证loss
        with torch.no_grad():
            for batch_idx, batch in val_pbar:
                if batch_idx >= max_batches:
                    break
                
                try:
                    loss = self.compute_loss(batch)
                    if torch.isfinite(loss):
                        total_loss += loss.item()
                        valid_batches += 1
                        
                        # 更新验证进度条
                        current_avg = total_loss / valid_batches
                        
                        # 计算验证内存使用
                        if torch.cuda.is_available():
                            val_gpu_memory = torch.cuda.memory_allocated() / 1024**3
                        else:
                            val_gpu_memory = 0.0
                            
                        val_pbar.set_postfix({
                            'val_loss': f'{loss.item():.4f}',
                            'avg_val_loss': f'{current_avg:.4f}',
                            'gpu_mem': f'{val_gpu_memory:.1f}GB',
                            'batch': f'{valid_batches}/{max_batches}'
                        })
                    else:
                        logger.warning(f"验证中检测到无效损失: {loss.item()}")
                        
                except Exception as e:
                    logger.error(f"验证批次失败: {str(e)}")
                    continue
        
        val_pbar.close()
        
        if valid_batches == 0:
            logger.error("验证阶段没有有效批次")
            return {
                'val_loss': float('inf'),
                'best_val_loss': self.best_val_loss,
                'val_cider': 0.0,
                'best_val_cider': self.best_val_cider,
                'valid_batches': 0,
                'improvement': False
            }
        
        avg_val_loss = total_loss / valid_batches
        
        # 计算验证loss耗时
        loss_duration = time.time() - val_start_time
        
        # 阶段2: 计算评估指标（CIDEr/BLEU等）
        logger.info("🔍 计算验证集评估指标...")
        metrics_start_time = time.time()
        val_metrics = self._compute_validation_metrics(max_samples=self.config.max_val_samples)
        metrics_duration = time.time() - metrics_start_time
        
        val_cider = val_metrics.get('CIDEr', 0.0)
        val_rouge_l = val_metrics.get('ROUGE_L', 0.0)
        val_bleu_4 = val_metrics.get('Bleu_4', 0.0)
        # 计算BLEU总和（用于记录）
        val_bleu_sum = (
            val_metrics.get('Bleu_1', 0.0) + 
            val_metrics.get('Bleu_2', 0.0) + 
            val_metrics.get('Bleu_3', 0.0) + 
            val_bleu_4
        )
        
        # 计算加权综合得分
        val_weighted_score = (
            self.metric_weights['cider'] * val_cider +
            self.metric_weights['rouge_l'] * val_rouge_l +
            self.metric_weights['bleu_4'] * val_bleu_4
        )
        
        # 总耗时
        total_val_duration = time.time() - val_start_time
        val_samples_per_sec = (valid_batches * self.config.batch_size) / total_val_duration if total_val_duration > 0 else 0
        
        # 判断是否改善（基于加权综合得分）
        improvement = False
        is_better = False
        improvement_reason = ""
        
        # 主要判断标准：加权综合得分
        if val_weighted_score > self.best_val_weighted_score:
            is_better = True
            improvement_reason = "加权综合得分提升"
        elif abs(val_weighted_score - self.best_val_weighted_score) < 1e-6 and self.best_val_weighted_score > 0:
            # 综合得分相等（罕见情况），使用Loss作为次要标准
            if avg_val_loss < self.best_val_loss:
                is_better = True
                improvement_reason = "综合得分相等，Loss降低"
        
        if is_better:
            improvement = True
            old_best_weighted_score = self.best_val_weighted_score
            old_best_cider = self.best_val_cider
            old_best_rouge_l = self.best_val_rouge_l
            old_best_bleu_4 = self.best_val_bleu_4
            old_best_loss = self.best_val_loss
            
            # 计算改善百分比
            improvement_percent = ((val_weighted_score - old_best_weighted_score) / old_best_weighted_score * 100) if old_best_weighted_score > 0 else 100.0
            
            # 更新所有最佳指标
            self.best_val_weighted_score = val_weighted_score
            self.best_val_cider = val_cider
            self.best_val_rouge_l = val_rouge_l
            self.best_val_bleu_sum = val_bleu_sum
            self.best_val_loss = avg_val_loss
            # 保存详细的BLEU指标
            self.best_val_bleu_1 = val_metrics.get('Bleu_1', 0.0)
            self.best_val_bleu_2 = val_metrics.get('Bleu_2', 0.0)
            self.best_val_bleu_3 = val_metrics.get('Bleu_3', 0.0)
            self.best_val_bleu_4 = val_bleu_4
            self.best_val_meteor = val_metrics.get('METEOR', 0.0)
            
            # 保存最佳模型
            self.save_best_model()
            
            logger.info(f"🎉 验证指标改善！({improvement_reason})")
            logger.info(
                f"   ⭐ 加权综合得分: {old_best_weighted_score:.4f} → {val_weighted_score:.4f} "
                f"({val_weighted_score - old_best_weighted_score:+.4f}, {improvement_percent:+.1f}%)"
            )
            logger.info(f"   📊 各项指标变化:")
            logger.info(
                f"      - CIDEr (55%权重): {old_best_cider:.4f} → {val_cider:.4f} "
                f"({val_cider - old_best_cider:+.4f})"
            )
            logger.info(
                f"      - ROUGE-L (30%权重): {old_best_rouge_l:.4f} → {val_rouge_l:.4f} "
                f"({val_rouge_l - old_best_rouge_l:+.4f})"
            )
            logger.info(
                f"      - BLEU-4 (15%权重): {old_best_bleu_4:.4f} → {val_bleu_4:.4f} "
                f"({val_bleu_4 - old_best_bleu_4:+.4f})"
            )
            logger.info(
                f"   📉 Loss: {old_best_loss:.4f} → {avg_val_loss:.4f} "
                f"({avg_val_loss - old_best_loss:+.4f})"
            )
            logger.info(
                f"   ⏱️  耗时: Loss {loss_duration:.1f}s + Metrics {metrics_duration:.1f}s = {total_val_duration:.1f}s"
            )
        else:
            weighted_score_diff = val_weighted_score - self.best_val_weighted_score
            cider_diff = val_cider - self.best_val_cider
            rouge_l_diff = val_rouge_l - self.best_val_rouge_l
            bleu_4_diff = val_bleu_4 - self.best_val_bleu_4
            loss_diff = avg_val_loss - self.best_val_loss
            logger.info(
                f"📊 验证结果: 综合得分={val_weighted_score:.4f} (最佳: {self.best_val_weighted_score:.4f}, {weighted_score_diff:+.4f}), "
                f"Loss={avg_val_loss:.4f} (最佳: {self.best_val_loss:.4f}, {loss_diff:+.4f}), "
                f"耗时 {total_val_duration:.1f}s"
            )
            logger.info(
                f"   分项: CIDEr={val_cider:.4f} ({cider_diff:+.4f}), "
                f"ROUGE-L={val_rouge_l:.4f} ({rouge_l_diff:+.4f}), "
                f"BLEU-4={val_bleu_4:.4f} ({bleu_4_diff:+.4f})"
            )
        
        return {
            'val_loss': avg_val_loss,
            'best_val_loss': self.best_val_loss,
            'val_weighted_score': val_weighted_score,
            'best_val_weighted_score': self.best_val_weighted_score,
            'val_cider': val_cider,
            'val_rouge_l': val_rouge_l,
            'val_bleu_4': val_bleu_4,
            'val_bleu_sum': val_bleu_sum,
            'best_val_cider': self.best_val_cider,
            'best_val_rouge_l': self.best_val_rouge_l,
            'best_val_bleu_4': self.best_val_bleu_4,
            'best_val_bleu_sum': self.best_val_bleu_sum,
            'valid_batches': valid_batches,
            'improvement': improvement,
            'loss_diff': avg_val_loss - self.best_val_loss,
            'weighted_score_diff': val_weighted_score - self.best_val_weighted_score,
            'cider_diff': val_cider - self.best_val_cider,
            'rouge_l_diff': val_rouge_l - self.best_val_rouge_l,
            'bleu_4_diff': val_bleu_4 - self.best_val_bleu_4,
            'bleu_sum_diff': val_bleu_sum - self.best_val_bleu_sum
        }
    
    def _compute_validation_metrics(self, max_samples: int = 500) -> Dict:
        """
        计算验证集评估指标（CIDEr、BLEU等）
        
        Args:
            max_samples: 最大评估样本数（默认500，平衡速度和准确性）
            
        Returns:
            Dict: 评估指标字典
        """
        try:
            from coco_evaluator import COCOCaptionEvaluator, COCOEvaluationConfig
            
            # 创建临时评估器配置
            eval_config = COCOEvaluationConfig()
            eval_config.coco_data_root = self.config.coco_data_root
            eval_config.max_eval_samples = max_samples  # 限制样本数以加快验证速度
            eval_config.batch_size = self.config.batch_size
            eval_config.eval_splits = ["val"]  # 只评估验证集
            
            # 传递生成参数
            eval_config.max_new_tokens = self.config.max_new_tokens
            eval_config.temperature = self.config.temperature
            
            # 创建评估器（不加载新模型，直接使用当前model_loader）
            evaluator = COCOCaptionEvaluator(eval_config)
            evaluator.model_loader = self.model_loader  # 复用当前模型
            evaluator.device = self.device
            
            # 设置数据加载器
            if not evaluator.setup_data():
                logger.warning("验证集数据加载失败，返回空指标")
                return {}
            
            # 只评估验证集
            results = evaluator.evaluate_split("val")
            
            if results and 'coco_metrics' in results:
                # 返回 coco_metrics 字典（包含 CIDEr, Bleu_4 等）
                return results['coco_metrics']
            else:
                logger.warning("验证集指标计算失败，返回空指标")
                return {}
                
        except Exception as e:
            logger.error(f"计算验证集指标失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def save_checkpoint(self, suffix: str = ""):
        """
        保存训练检查点
        
        Args:
            suffix: 检查点名称后缀，用于区分不同阶段的检查点
        """
        try:
            checkpoint_name = f"checkpoint-step-{self.global_step}{suffix}"
            checkpoint_path = os.path.join(
                self.config.checkpoint_dir,
                checkpoint_name
            )
            
            # 保存模型状态
            self.model_loader.model.save_pretrained(checkpoint_path)
            self.model_loader.processor.save_pretrained(checkpoint_path)
            
            # 如果启用了LoRA，同时保存LoRA适配器
            if self.config.enable_lora and hasattr(self.config, 'lora_adapters_dir'):
                adapter_checkpoint_path = os.path.join(
                    self.config.lora_adapters_dir,
                    checkpoint_name
                )
                if self.model_loader.save_lora_adapter(adapter_checkpoint_path):
                    logger.info(f"LoRA适配器检查点已保存: {os.path.abspath(adapter_checkpoint_path)}")
                else:
                    logger.warning(f"保存LoRA适配器检查点失败: {adapter_checkpoint_path}")
            
            # 保存训练状态
            training_state = {
                'global_step': self.global_step,
                'current_epoch': self.current_epoch,
                'best_val_loss': self.best_val_loss,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'training_history': self.training_history
            }
            
            torch.save(training_state, os.path.join(checkpoint_path, "training_state.pt"))
            
            logger.info(f"检查点已保存: {os.path.abspath(checkpoint_path)}")
            
            # 清理旧检查点，只保留最新的2个
            self._cleanup_old_checkpoints()
            
        except Exception as e:
            logger.error(f"保存检查点失败: {str(e)}")
    
    def _cleanup_old_checkpoints(self, keep_last_n: int = 2):
        """清理旧检查点，只保留最新的N个"""
        try:
            if not os.path.exists(self.config.checkpoint_dir):
                return
            
            # 获取所有检查点目录
            checkpoint_dirs = []
            for item in os.listdir(self.config.checkpoint_dir):
                item_path = os.path.join(self.config.checkpoint_dir, item)
                if os.path.isdir(item_path) and item.startswith("checkpoint-step-"):
                    try:
                        step_num = int(item.split("-")[-1])
                        checkpoint_dirs.append((step_num, item_path))
                    except ValueError:
                        continue
            
            # 按步数排序，保留最新的N个
            if len(checkpoint_dirs) > keep_last_n:
                checkpoint_dirs.sort(key=lambda x: x[0])  # 按步数排序
                dirs_to_remove = checkpoint_dirs[:-keep_last_n]  # 除了最新N个，其他都删除
                
                for step_num, dir_path in dirs_to_remove:
                    try:
                        import shutil
                        shutil.rmtree(dir_path)
                        logger.info(f"已清理旧检查点: checkpoint-step-{step_num}")
                    except Exception as e:
                        logger.warning(f"清理检查点失败 {dir_path}: {str(e)}")
            
            # 同时清理LoRA适配器检查点
            if self.config.enable_lora and hasattr(self.config, 'lora_adapters_dir'):
                self._cleanup_old_lora_checkpoints(keep_last_n)
                
        except Exception as e:
            logger.warning(f"清理检查点时出错: {str(e)}")
    
    def _cleanup_old_lora_checkpoints(self, keep_last_n: int = 2):
        """清理旧的LoRA适配器检查点"""
        try:
            if not os.path.exists(self.config.lora_adapters_dir):
                return
            
            # 获取所有LoRA检查点目录
            checkpoint_dirs = []
            for item in os.listdir(self.config.lora_adapters_dir):
                item_path = os.path.join(self.config.lora_adapters_dir, item)
                if os.path.isdir(item_path) and item.startswith("checkpoint-step-"):
                    try:
                        step_num = int(item.split("-")[-1])
                        checkpoint_dirs.append((step_num, item_path))
                    except ValueError:
                        continue
            
            # 按步数排序，保留最新的N个
            if len(checkpoint_dirs) > keep_last_n:
                checkpoint_dirs.sort(key=lambda x: x[0])
                dirs_to_remove = checkpoint_dirs[:-keep_last_n]
                
                for step_num, dir_path in dirs_to_remove:
                    try:
                        import shutil
                        shutil.rmtree(dir_path)
                        logger.info(f"已清理旧LoRA检查点: checkpoint-step-{step_num}")
                    except Exception as e:
                        logger.warning(f"清理LoRA检查点失败 {dir_path}: {str(e)}")
                        
        except Exception as e:
            logger.warning(f"清理LoRA检查点时出错: {str(e)}")
    
    def save_best_model(self):
        """保存最佳模型"""
        try:
            best_model_path = os.path.join(self.config.output_dir, "best_model")
            
            self.model_loader.model.save_pretrained(best_model_path)
            self.model_loader.processor.save_pretrained(best_model_path)
            
            logger.info(f"最佳模型已保存: {os.path.abspath(best_model_path)}")
            
            # 如果启用了LoRA，同时保存LoRA适配器
            if self.config.enable_lora and hasattr(self.config, 'lora_adapters_dir'):
                best_adapter_path = os.path.join(self.config.lora_adapters_dir, "best_model")
                if self.model_loader.save_lora_adapter(best_adapter_path):
                    logger.info(f"最佳LoRA适配器已保存: {os.path.abspath(best_adapter_path)}")
                else:
                    logger.warning("保存最佳LoRA适配器失败")
            
        except Exception as e:
            logger.error(f"保存最佳模型失败: {str(e)}")
    
    def train(self) -> bool:
        """执行完整训练流程"""
        try:
            logger.info("🎯 开始COCO训练流程...")
            
            # 设置模型
            if not self.setup_model():
                return False
            
            # 设置数据
            if not self.setup_data():
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
                logger.info(
                    f"🚀 开始训练 Epoch {epoch + 1}/{self.config.num_epochs}: "
                    f"trainable_params={param_stats['trainable_params']:,}, "
                    f"total_params={param_stats['total_params']:,}, "
                    f"trainable_ratio={param_stats['trainable_ratio']:.2%}, "
                    f"current_lr={self.scheduler.get_last_lr()[0]:.2e}"
                )
                
                # 训练一个epoch
                epoch_metrics = self.train_epoch()
                self.training_history.append(epoch_metrics)
                
                # 每个epoch结束后验证
                logger.info(f"🔍 开始 Epoch {epoch + 1} 验证...")
                val_metrics = self.validate()
                
                # 详细的epoch总结日志
                logger.info(
                    f"📈 Epoch {epoch + 1} 总结: "
                    f"train_loss={epoch_metrics['avg_loss']:.4f}, "
                    f"val_loss={val_metrics['val_loss']:.4f}, "
                    f"val_weighted_score={val_metrics.get('val_weighted_score', 0.0):.4f}, "
                    f"best_weighted_score={val_metrics.get('best_val_weighted_score', 0.0):.4f}, "
                    f"improvement={'✅' if val_metrics['improvement'] else '❌'}, "
                    f"lr={epoch_metrics['learning_rate']:.2e}, "
                    f"global_step={epoch_metrics['global_step']}"
                )
                
                # 保存epoch检查点
                self.save_checkpoint()
            
            # 保存最终模型
            final_model_path = os.path.join(self.config.output_dir, "final_model")
            self.model_loader.model.save_pretrained(final_model_path)
            self.model_loader.processor.save_pretrained(final_model_path)
            
            # 如果启用了LoRA，同时保存最终LoRA适配器
            if self.config.enable_lora and hasattr(self.config, 'lora_adapters_dir'):
                final_adapter_path = os.path.join(self.config.lora_adapters_dir, "final_model")
                if self.model_loader.save_lora_adapter(final_adapter_path):
                    logger.info(f"最终LoRA适配器已保存: {os.path.abspath(final_adapter_path)}")
                else:
                    logger.warning("保存最终LoRA适配器失败")
            
            # 保存训练历史（包含最佳验证指标）
            history_path = os.path.join(self.config.output_dir, "training_history.json")
            history_data = {
                'epochs': self.training_history,
                'best_val_metrics': {
                    'weighted_score': self.best_val_weighted_score,
                    'CIDEr': self.best_val_cider,
                    'Bleu_1': getattr(self, 'best_val_bleu_1', 0.0),
                    'Bleu_2': getattr(self, 'best_val_bleu_2', 0.0),
                    'Bleu_3': getattr(self, 'best_val_bleu_3', 0.0),
                    'Bleu_4': getattr(self, 'best_val_bleu_4', 0.0),
                    'ROUGE_L': getattr(self, 'best_val_rouge_l', 0.0),
                    'METEOR': getattr(self, 'best_val_meteor', 0.0),
                    'bleu_sum': self.best_val_bleu_sum,
                    'loss': self.best_val_loss
                },
                'metric_weights': self.metric_weights
            }
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            
            # 显示训练完成摘要
            self._log_training_completion_summary()
            
            logger.info("🎉 COCO训练完成！")
            return True
            
        except Exception as e:
            logger.error(f"训练失败: {str(e)}")
            return False
    
    def progressive_train(self, 
                         stage_epochs: Dict[str, int] = None,
                         lora_ranks: Dict[str, int] = None,
                         complexity_thresholds: Tuple[float, float] = (33.33, 66.67)) -> bool:
        """
        执行渐进式LoRA训练（创新方法！）
        
        训练策略：
        - Stage 1 (easy): 只用简单样本，小LoRA秩 (r=32)
        - Stage 2 (medium): 简单+中等样本，中等LoRA秩 (r=64)
        - Stage 3 (hard): 所有样本，大LoRA秩 (r=128)
        
        Args:
            stage_epochs: 每个阶段的训练轮数，默认 {'easy': 1, 'medium': 1, 'hard': 1}
            lora_ranks: 每个阶段的LoRA秩，默认 {'easy': 32, 'medium': 64, 'hard': 128}
            complexity_thresholds: 复杂度分层百分位数 (默认 33.33, 66.67 表示三等分)
                                  使用百分位数自适应分层，确保各阶段样本分布均匀
            
        Returns:
            训练是否成功
        """
        try:
            logger.info("🎯 开始渐进式LoRA训练（Progressive LoRA Training）...")
            logger.info("=" * 80)
            logger.info("📚 创新方法：结合课程学习(Curriculum Learning)与参数高效微调(LoRA)")
            logger.info("=" * 80)
            
            # 默认参数
            if stage_epochs is None:
                stage_epochs = {'easy': 1, 'medium': 1, 'hard': 1}
            if lora_ranks is None:
                lora_ranks = {'easy': 32, 'medium': 64, 'hard': 128}
            
            # 设置模型（使用初始的小LoRA秩）
            logger.info(f"初始化模型，初始LoRA秩: {lora_ranks['easy']}")
            if hasattr(self.config, 'lora_config') and self.config.lora_config:
                self.config.lora_config.update_lora_rank(lora_ranks['easy'])
            
            if not self.setup_model():
                return False
            
            # 创建渐进式数据加载器
            logger.info(f"创建渐进式数据加载器 (thresholds={complexity_thresholds})...")
            from coco_dataset import COCODatasetConfig, COCODataLoader
            
            # 自动检测数据集类型（支持 COCO / Flickr8k / Flickr30K / COCO Karpathy / VizWiz-Captions）
            data_root = self.config.coco_data_root
            lower_root = data_root.lower()
            if 'flickr8k' in lower_root:
                logger.info("使用Flickr8k适配器创建渐进式数据加载器...")
                from flickr8k_adapter import Flickr8kAdapter
                coco_config = Flickr8kAdapter(data_root)
            elif 'flickr30k' in lower_root or 'flickr' in lower_root:
                logger.info("使用Flickr30K适配器创建渐进式数据加载器...")
                from flickr30k_adapter import Flickr30KAdapter
                coco_config = Flickr30KAdapter(data_root)
            elif 'vizwiz' in lower_root:
                logger.info("使用VizWiz-Captions适配器创建渐进式数据加载器...")
                from vizwiz_adapter import VizWizCaptionAdapter
                coco_config = VizWizCaptionAdapter(data_root)
            elif 'karpathy' in lower_root or 'coco2014' in lower_root:
                logger.info("使用COCO Karpathy适配器创建渐进式数据加载器...")
                from coco_karpathy_adapter import COCOKarpathyAdapter
                coco_config = COCOKarpathyAdapter(data_root)
            else:
                coco_config = COCODatasetConfig(data_root)
            
            data_loader = COCODataLoader(coco_config)
            
            # 获取三个阶段的数据加载器（累积式：easy → easy+medium → all）
            logger.info("使用累积式渐进训练策略：Easy → Easy+Medium → All")
            progressive_loaders = data_loader.create_progressive_dataloaders(
                split='train',
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                thresholds=complexity_thresholds,
                max_samples=self.config.max_train_samples
            )
            
            # 验证集加载器（保持不变）
            self.val_dataloader = data_loader.create_dataloader(
                split='val',
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                max_samples=self.config.max_val_samples
            )
            
            # 临时设置第一个阶段的dataloader，以便_log_training_summary()能正常工作
            self.train_dataloader = progressive_loaders['easy']
            
            # 显示训练配置摘要（包含参数统计）
            self._log_training_summary()
            
            # 显示渐进式训练配置
            self._log_progressive_training_summary(stage_epochs, lora_ranks, complexity_thresholds)
            
            # ==================== 三阶段渐进式训练 ====================
            total_epochs = 0
            
            for stage_name in ['easy', 'medium', 'hard']:
                logger.info("\n" + "=" * 80)
                logger.info(f"🌟 Stage: {stage_name.upper()} | LoRA Rank: {lora_ranks[stage_name]} | Epochs: {stage_epochs[stage_name]}")
                logger.info("=" * 80)
                
                # 更新当前阶段的数据加载器（必须在调整LoRA秩之前）
                # 因为优化器重新初始化需要使用新阶段的数据加载器来计算步数
                self.train_dataloader = progressive_loaders[stage_name]
                
                # 设置当前阶段的总epoch数（用于正确计算训练进度）
                self._current_stage_total_epochs = stage_epochs[stage_name]
                
                # 记录当前阶段开始时的global_step（用于计算阶段内进度）
                self._stage_start_step = self.global_step
                
                # 动态调整LoRA秩（除了第一阶段）
                if stage_name != 'easy':
                    logger.info(f"📊 动态调整LoRA秩: {lora_ranks[stage_name]}")
                    if not self._update_model_lora_rank(lora_ranks[stage_name]):
                        logger.error(f"LoRA秩更新失败，中止训练")
                        return False
                else:
                    # Easy阶段：初始化优化器（使用当前阶段的步数）
                    logger.info(f"📊 初始化优化器（Easy阶段）")
                    if not self._setup_optimizer_for_progressive():
                        logger.error(f"优化器初始化失败，中止训练")
                        return False
                    
                    # 记录当前LoRA秩（用于后续阶段的权重继承）
                    self._current_lora_rank = lora_ranks['easy']
                    logger.info(f"记录初始LoRA秩: {self._current_lora_rank}")
                
                # 训练当前阶段
                for stage_epoch in range(stage_epochs[stage_name]):
                    self.current_epoch = total_epochs
                    epoch_in_stage = stage_epoch + 1
                    
                    param_stats = self._compute_parameter_stats()
                    logger.info(
                        f"🚀 Stage [{stage_name}] Epoch {epoch_in_stage}/{stage_epochs[stage_name]} "
                        f"(Global Epoch {total_epochs + 1}): "
                        f"trainable_params={param_stats['trainable_params']:,}, "
                        f"total_params={param_stats['total_params']:,}, "
                        f"trainable_ratio={param_stats['trainable_ratio']:.2%}, "
                        f"lora_rank={lora_ranks[stage_name]}, "
                        f"lr={self.scheduler.get_last_lr()[0]:.2e}"
                    )
                    
                    # 训练一个epoch
                    epoch_metrics = self.train_epoch()
                    epoch_metrics['stage'] = stage_name
                    epoch_metrics['lora_rank'] = lora_ranks[stage_name]
                    self.training_history.append(epoch_metrics)
                    
                    # 验证
                    logger.info(f"🔍 Stage [{stage_name}] Epoch {epoch_in_stage} 验证...")
                    val_metrics = self.validate()
                    
                    # 总结
                    logger.info(
                        f"📈 Stage [{stage_name}] Epoch {epoch_in_stage} 总结: "
                        f"train_loss={epoch_metrics['avg_loss']:.4f}, "
                        f"val_loss={val_metrics['val_loss']:.4f}, "
                        f"val_weighted_score={val_metrics.get('val_weighted_score', 0.0):.4f}, "
                        f"best_weighted_score={val_metrics.get('best_val_weighted_score', 0.0):.4f}, "
                        f"improvement={'✅' if val_metrics['improvement'] else '❌'}"
                    )
                    
                    # 保存检查点
                    self.save_checkpoint(suffix=f"_stage_{stage_name}_epoch_{epoch_in_stage}")
                    
                    total_epochs += 1
                
                logger.info(f"✅ Stage [{stage_name}] 完成！")
                
                # 清理阶段特定的临时变量
                self._current_stage_total_epochs = None
                self._stage_start_step = 0
            
            # ==================== 训练完成 ====================
            
            # 保存最终模型
            final_model_path = os.path.join(self.config.output_dir, "final_model_progressive")
            self.model_loader.model.save_pretrained(final_model_path)
            self.model_loader.processor.save_pretrained(final_model_path)
            
            # 保存最终LoRA适配器
            if self.config.enable_lora and hasattr(self.config, 'lora_adapters_dir'):
                final_adapter_path = os.path.join(self.config.lora_adapters_dir, "final_progressive")
                if self.model_loader.save_lora_adapter(final_adapter_path):
                    logger.info(f"最终渐进式LoRA适配器已保存: {os.path.abspath(final_adapter_path)}")
            
            # 保存训练历史（包含最佳验证指标）
            history_path = os.path.join(self.config.output_dir, "progressive_training_history.json")
            history_data = {
                'epochs': self.training_history,
                'best_val_metrics': {
                    'weighted_score': self.best_val_weighted_score,
                    'CIDEr': self.best_val_cider,
                    'Bleu_1': getattr(self, 'best_val_bleu_1', 0.0),
                    'Bleu_2': getattr(self, 'best_val_bleu_2', 0.0),
                    'Bleu_3': getattr(self, 'best_val_bleu_3', 0.0),
                    'Bleu_4': getattr(self, 'best_val_bleu_4', 0.0),
                    'ROUGE_L': getattr(self, 'best_val_rouge_l', 0.0),
                    'METEOR': getattr(self, 'best_val_meteor', 0.0),
                    'bleu_sum': self.best_val_bleu_sum,
                    'loss': self.best_val_loss
                },
                'metric_weights': self.metric_weights,
                'training_config': {
                    'stage_epochs': stage_epochs,
                    'lora_ranks': lora_ranks,
                    'complexity_thresholds': complexity_thresholds
                }
            }
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            
            # 完成摘要
            logger.info("\n" + "=" * 80)
            logger.info("🎉 渐进式LoRA训练完成！")
            logger.info(f"📊 训练统计:")
            logger.info(f"   - 总训练轮数: {total_epochs}")
            logger.info(f"   - Easy阶段: {stage_epochs['easy']} epochs (rank={lora_ranks['easy']})")
            logger.info(f"   - Medium阶段: {stage_epochs['medium']} epochs (rank={lora_ranks['medium']})")
            logger.info(f"   - Hard阶段: {stage_epochs['hard']} epochs (rank={lora_ranks['hard']})")
            logger.info(f"🏆 最佳验证指标（选择策略：加权综合得分）:")
            logger.info(f"   - ⭐ 加权综合得分: {self.best_val_weighted_score:.4f}")
            logger.info(f"   - 分项指标:")
            logger.info(f"      • CIDEr (55%): {self.best_val_cider:.4f}")
            logger.info(f"      • ROUGE-L (30%): {self.best_val_rouge_l:.4f}")
            logger.info(f"      • BLEU-4 (15%): {self.best_val_bleu_4:.4f}")
            logger.info(f"   - BLEU总和: {self.best_val_bleu_sum:.4f}")
            logger.info(f"   - 最佳Loss: {self.best_val_loss:.4f}")
            logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            logger.error(f"渐进式训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _extract_lora_weights(self) -> Dict[str, torch.Tensor]:
        """
        提取当前模型的LoRA权重
        
        这个方法遍历模型中所有的LoRA层，提取lora_A和lora_B矩阵的权重。
        提取的权重将用于后续的权重继承过程。
        
        Returns:
            Dict[str, torch.Tensor]: 包含所有LoRA层权重的字典
                键格式: "layer_name.lora_A" 或 "layer_name.lora_B"
                值: 对应的权重张量（已从GPU移到CPU）
        """
        lora_weights = {}
        
        try:
            model = self.model_loader.model
            
            # 检查是否是PEFT模型
            if not hasattr(model, 'base_model'):
                logger.warning("模型不是PEFT模型，无法提取LoRA权重")
                return lora_weights
            
            # 遍历所有模块，查找LoRA层
            for name, module in model.named_modules():
                # PEFT的LoRA层通常包含lora_A和lora_B属性
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    # 提取lora_A权重
                    if hasattr(module.lora_A, 'default') and hasattr(module.lora_A.default, 'weight'):
                        lora_a_weight = module.lora_A.default.weight.detach().clone().cpu()
                        lora_weights[f"{name}.lora_A"] = lora_a_weight
                        logger.debug(f"提取 {name}.lora_A: shape={lora_a_weight.shape}")
                    
                    # 提取lora_B权重
                    if hasattr(module.lora_B, 'default') and hasattr(module.lora_B.default, 'weight'):
                        lora_b_weight = module.lora_B.default.weight.detach().clone().cpu()
                        lora_weights[f"{name}.lora_B"] = lora_b_weight
                        logger.debug(f"提取 {name}.lora_B: shape={lora_b_weight.shape}")
            
            logger.info(f"成功提取 {len(lora_weights)} 个LoRA参数")
            return lora_weights
            
        except Exception as e:
            logger.error(f"提取LoRA权重失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _gram_schmidt_orthogonalize(self, vectors: torch.Tensor, existing_basis: torch.Tensor = None) -> torch.Tensor:
        """
        Gram-Schmidt正交化
        
        将输入向量正交化,如果提供existing_basis,则正交化后的向量会垂直于existing_basis中的所有向量
        
        Args:
            vectors: 待正交化的向量 (d, k) - k个d维向量
            existing_basis: 已有的正交基 (d, r) - r个d维正交向量 (可选)
        
        Returns:
            正交化后的向量 (d, k)
        """
        device = vectors.device
        dtype = vectors.dtype
        
        # 复制向量以避免修改原始数据
        orth_vectors = vectors.clone()
        
        # 先对existing_basis进行投影并减去
        if existing_basis is not None and existing_basis.shape[1] > 0:
            for i in range(existing_basis.shape[1]):
                basis_vec = existing_basis[:, i:i+1]  # (d, 1)
                # 投影: proj = (v·b)b
                proj_coef = torch.matmul(orth_vectors.T, basis_vec)  # (k, 1)
                projection = basis_vec * proj_coef.T  # (d, k)
                orth_vectors = orth_vectors - projection
        
        # 对orth_vectors内部进行Gram-Schmidt正交化
        k = orth_vectors.shape[1]
        for i in range(k):
            # 正交化当前向量(相对于之前的向量)
            for j in range(i):
                prev_vec = orth_vectors[:, j:j+1]
                curr_vec = orth_vectors[:, i:i+1]
                # 投影并减去
                proj_coef = torch.matmul(curr_vec.T, prev_vec)
                orth_vectors[:, i:i+1] = curr_vec - proj_coef * prev_vec
            
            # 归一化
            norm = torch.norm(orth_vectors[:, i])
            if norm > 1e-10:  # 避免除以0
                orth_vectors[:, i] = orth_vectors[:, i] / norm
            else:
                # 如果向量几乎为0,用随机向量替换
                logger.warning(f"正交化时遇到近零向量,使用随机向量替换 (索引={i})")
                orth_vectors[:, i] = torch.randn(orth_vectors.shape[0], device=device, dtype=dtype)
                orth_vectors[:, i] = orth_vectors[:, i] / torch.norm(orth_vectors[:, i])
        
        return orth_vectors
    
    def _expand_lora_with_svd(self, old_lora_A: torch.Tensor, old_lora_B: torch.Tensor, 
                              new_rank: int, use_orthogonalization: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用SVD分解方法扩展LoRA权重(方案B: SVD + 正交化)
        
        核心思想:
        1. 重构原始的ΔW = B @ A
        2. 对ΔW进行SVD分解,得到主要成分
        3. 扩展奇异值(指数衰减)和对应的U、Vh矩阵
        4. (可选)对新增维度进行正交化,确保独立性
        5. 重新构造扩展后的A和B矩阵
        
        Args:
            old_lora_A: 旧的A矩阵 (old_rank, in_features)
            old_lora_B: 旧的B矩阵 (out_features, old_rank)
            new_rank: 新的秩 (必须 > old_rank)
            use_orthogonalization: 是否对新增维度进行正交化 (默认True)
        
        Returns:
            (new_A, new_B): 扩展后的LoRA权重
                new_A: (new_rank, in_features)
                new_B: (out_features, new_rank)
        """
        old_rank = old_lora_A.shape[0]
        in_features = old_lora_A.shape[1]
        out_features = old_lora_B.shape[0]
        
        device = old_lora_A.device
        dtype = old_lora_A.dtype
        
        # 验证维度
        assert old_lora_B.shape[1] == old_rank, f"维度不匹配: B.shape[1]={old_lora_B.shape[1]} != old_rank={old_rank}"
        assert new_rank > old_rank, f"新rank ({new_rank}) 必须大于旧rank ({old_rank})"
        
        # === 步骤1: 重构ΔW ===
        delta_W = torch.matmul(old_lora_B, old_lora_A)  # (out_features, in_features)
        logger.debug(f"  重构ΔW: shape={delta_W.shape}, norm={torch.norm(delta_W).item():.4f}")
        
        # === 步骤2: SVD分解 ===
        # 注意: torch.svd已弃用,使用torch.linalg.svd
        try:
            U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
            # U: (out_features, min(out_features, in_features))
            # S: (min(out_features, in_features),)
            # Vh: (min(out_features, in_features), in_features)
        except Exception as e:
            logger.warning(f"  SVD分解失败,回退到torch.svd: {e}")
            U, S, V = torch.svd(delta_W)
            Vh = V.T
        
        # 截取到old_rank维度
        U = U[:, :old_rank]  # (out_features, old_rank)
        S = S[:old_rank]  # (old_rank,)
        Vh = Vh[:old_rank, :]  # (old_rank, in_features)
        
        logger.debug(f"  SVD分解: U={U.shape}, S={S.shape}, Vh={Vh.shape}")
        logger.debug(f"  奇异值范围: [{S.min().item():.6f}, {S.max().item():.6f}]")
        
        # === 步骤3: 扩展奇异值 ===
        # 新增奇异值采用指数衰减,避免突变
        num_new = new_rank - old_rank
        last_singular = S[-1].item()
        decay_factor = 0.3  # 新增奇异值为最小奇异值的30%，平衡初始梯度强度和稳定性
        
        new_singulars = torch.ones(num_new, device=device, dtype=dtype) * last_singular * decay_factor
        S_expanded = torch.cat([S, new_singulars])  # (new_rank,)
        
        logger.debug(f"  扩展奇异值: 新增{num_new}个, 值={last_singular * decay_factor:.6f}")
        
        # === 步骤4: 扩展U和Vh ===
        # U的扩展：随机初始化（列空间有足够的自由度）
        U_new_part = torch.randn(out_features, num_new, device=device, dtype=dtype) * 0.1
        U_expanded = torch.cat([U, U_new_part], dim=1)  # (out_features, new_rank)
        
        # Vh的扩展：从正交补空间采样（避免随机向量落在旧空间内）
        # 关键：delta_W的秩=old_rank，Vh的old_rank行已张满行空间，
        # 需要从in_features维空间的正交补空间中选择新向量
        if in_features > old_rank:
            # 生成完整的正交基
            random_matrix = torch.randn(in_features, in_features, device=device, dtype=dtype)
            Q, _ = torch.linalg.qr(random_matrix)  # Q: (in_features, in_features)
            
            # 计算每一列与Vh的重叠程度
            overlap_scores = torch.sum((Vh @ Q) ** 2, dim=0)  # (in_features,)
            
            # 选择重叠最小的num_new个向量（来自正交补空间）
            _, indices = torch.sort(overlap_scores)
            selected_indices = indices[:num_new]
            Vh_new_part = Q[:, selected_indices].T  # (num_new, in_features)
            
            logger.debug(f"  从正交补空间采样Vh新向量，重叠范围: [{overlap_scores.min().item():.6f}, {overlap_scores[selected_indices].max().item():.6f}]")
        else:
            # 如果in_features <= old_rank，使用随机初始化（虽然这种情况很少见）
            Vh_new_part = torch.randn(num_new, in_features, device=device, dtype=dtype) * 0.1
        
        Vh_expanded = torch.cat([Vh, Vh_new_part], dim=0)  # (new_rank, in_features)
        
        # === 步骤5: (可选)正交化新增维度 ===
        if use_orthogonalization and num_new > 0:
            logger.debug(f"  正交化新增维度...")
            
            # 正交化U的新增部分(相对于旧部分) - 列空间正交化
            U_new_orth = self._gram_schmidt_orthogonalize(
                U_expanded[:, old_rank:],  # 新增部分 (out_features, num_new)
                U_expanded[:, :old_rank]   # 已有部分 (out_features, old_rank)
            )
            U_expanded[:, old_rank:] = U_new_orth
            
            # 正交化Vh的新增部分(相对于旧部分) - 行空间正交化
            # 直接对行向量进行Gram-Schmidt，无需转置
            for i in range(num_new):
                new_row = Vh_expanded[old_rank + i, :].clone()
                
                # 减去在所有旧行上的投影
                for j in range(old_rank):
                    old_row = Vh_expanded[j, :]
                    # 假设old_row已归一化（来自SVD）
                    proj_coef = torch.dot(new_row, old_row)
                    new_row = new_row - proj_coef * old_row
                
                # 减去在之前新行上的投影
                for j in range(i):
                    prev_row = Vh_expanded[old_rank + j, :]
                    proj_coef = torch.dot(new_row, prev_row)
                    new_row = new_row - proj_coef * prev_row
                
                # 归一化
                norm = torch.norm(new_row)
                if norm > 1e-8:
                    Vh_expanded[old_rank + i, :] = new_row / norm
                else:
                    logger.warning(f"  Vh新行{i}正交化后范数过小({norm:.2e})，保持原值")
            
            logger.debug(f"  正交化完成")
        
        # === 步骤6: 重新构造A和B ===
        # 将奇异值的平方根分配给A和B,保持对称性
        sqrt_S = torch.sqrt(S_expanded)  # (new_rank,)
        sqrt_S_diag = torch.diag(sqrt_S)  # (new_rank, new_rank)
        
        new_A = torch.matmul(sqrt_S_diag, Vh_expanded)  # (new_rank, in_features)
        new_B = torch.matmul(U_expanded, sqrt_S_diag)   # (out_features, new_rank)
        
        logger.debug(f"  重构A和B: A={new_A.shape}, B={new_B.shape}")
        
        # === 验证: 检查扩展后的ΔW是否接近原始ΔW ===
        delta_W_new = torch.matmul(new_B, new_A)
        reconstruction_error = torch.norm(delta_W_new - delta_W).item()
        relative_error = reconstruction_error / (torch.norm(delta_W).item() + 1e-10)
        logger.debug(f"  重构误差: 绝对={reconstruction_error:.6f}, 相对={relative_error:.6f}")
        
        if relative_error > 0.1:  # 相对误差超过10%则警告
            logger.warning(f"  ⚠️ SVD扩展重构误差较大: {relative_error:.2%}")
        
        return new_A, new_B
    
    def _inherit_lora_weights(self, old_weights: Dict[str, torch.Tensor], old_rank: int, new_rank: int):
        """
        将旧的LoRA权重继承到新的更大rank的LoRA层中
        
        使用SVD分解 + 正交化策略(方案B):
        1. 重构原始的ΔW = B @ A
        2. 对ΔW进行SVD分解,提取主要成分
        3. 扩展奇异值(指数衰减)和对应的基向量
        4. 对新增维度进行正交化,确保独立性
        5. 重新构造扩展后的A和B矩阵
        
        相比零填充策略,该方法的优势:
        - 保持已学习的信息(通过SVD主成分)
        - 新增维度正交于已有维度,充分利用表示空间
        - 平滑扩展,避免性能突降
        
        Args:
            old_weights: 旧的LoRA权重字典（来自_extract_lora_weights）
            old_rank: 旧的LoRA秩
            new_rank: 新的LoRA秩（必须 >= old_rank）
        """
        if new_rank <= old_rank:
            logger.warning(f"新rank ({new_rank}) <= 旧rank ({old_rank})，跳过权重继承")
            return
        
        if not old_weights:
            logger.warning("旧权重为空，跳过权重继承")
            return
        
        try:
            model = self.model_loader.model
            inherited_count = 0
            failed_count = 0
            
            logger.info(f"🔬 使用SVD+正交化方法扩展LoRA权重...")
            
            # 遍历新模型的所有LoRA层
            for name, module in model.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    # 构造权重键名
                    lora_a_key = f"{name}.lora_A"
                    lora_b_key = f"{name}.lora_B"
                    
                    # 检查旧权重中是否存在对应的层
                    if lora_a_key in old_weights and lora_b_key in old_weights:
                        try:
                            # 获取旧权重
                            old_a = old_weights[lora_a_key]  # (old_rank, in_features)
                            old_b = old_weights[lora_b_key]  # (out_features, old_rank)
                            
                            # 获取新权重的引用
                            if hasattr(module.lora_A, 'default') and hasattr(module.lora_A.default, 'weight'):
                                new_a_tensor = module.lora_A.default.weight.data
                            else:
                                logger.warning(f"跳过 {name}: lora_A结构不符合预期")
                                continue
                            
                            if hasattr(module.lora_B, 'default') and hasattr(module.lora_B.default, 'weight'):
                                new_b_tensor = module.lora_B.default.weight.data
                            else:
                                logger.warning(f"跳过 {name}: lora_B结构不符合预期")
                                continue
                            
                            # 验证维度兼容性
                            if old_a.shape[1] != new_a_tensor.shape[1]:
                                logger.warning(f"跳过 {name}: in_features不匹配 {old_a.shape[1]} vs {new_a_tensor.shape[1]}")
                                failed_count += 1
                                continue
                            
                            if old_b.shape[0] != new_b_tensor.shape[0]:
                                logger.warning(f"跳过 {name}: out_features不匹配 {old_b.shape[0]} vs {new_b_tensor.shape[0]}")
                                failed_count += 1
                                continue
                            
                            # 移动到正确的设备
                            device = new_a_tensor.device
                            old_a = old_a.to(device)
                            old_b = old_b.to(device)
                            
                            # === 使用SVD方法扩展权重 ===
                            logger.debug(f"扩展 {name}: rank {old_rank} → {new_rank}")
                            new_a, new_b = self._expand_lora_with_svd(
                                old_a, old_b, new_rank, 
                                use_orthogonalization=True
                            )
                            
                            # 将扩展后的权重写入模型
                            new_a_tensor.copy_(new_a)
                            new_b_tensor.copy_(new_b)
                            
                            inherited_count += 1
                            
                        except Exception as e:
                            logger.warning(f"继承 {name} 失败: {str(e)}")
                            failed_count += 1
                            continue
            
            logger.info(f"✅ 成功继承 {inherited_count} 个LoRA层的权重 (rank {old_rank} → {new_rank})")
            if failed_count > 0:
                logger.warning(f"⚠️ {failed_count} 个LoRA层继承失败")
            
        except Exception as e:
            logger.error(f"继承LoRA权重失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _setup_continuous_optimizer(self, total_training_steps: int):
        """
        设置连续训练模式的优化器和调度器
        
        在连续训练模式下（enable_optimizer_reset=False），优化器不会在阶段切换时重置，
        学习率调度器会在整个训练过程中平滑衰减。
        
        这个方法在第一个阶段初始化时调用，计算所有阶段的总步数。
        
        Args:
            total_training_steps: 所有阶段的总训练步数
        """
        logger.info(f"设置连续训练模式优化器 (total_steps={total_training_steps})...")
        
        try:
            # 获取可训练参数
            trainable_params = [p for p in self.model_loader.model.parameters() if p.requires_grad]
            
            if not trainable_params:
                logger.error("没有可训练参数，无法创建优化器")
                return False
            
            # 创建优化器
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                eps=1e-8,
                betas=(0.9, 0.999)
            )
            
            # 计算warmup步数
            warmup_steps = int(total_training_steps * self.config.warmup_ratio)
            
            # 创建学习率调度器（覆盖所有训练阶段）
            if self.config.scheduler_type == "linear":
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_training_steps
                )
            elif self.config.scheduler_type == "cosine":
                self.scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_training_steps,
                    num_cycles=self.config.cosine_num_cycles
                )
            else:
                logger.warning(f"未知的调度器类型: {self.config.scheduler_type}，使用cosine")
                self.scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_training_steps,
                    num_cycles=self.config.cosine_num_cycles
                )
            
            logger.info(f"✅ 连续训练优化器初始化完成:")
            logger.info(f"   - 总训练步数: {total_training_steps}")
            logger.info(f"   - Warmup步数: {warmup_steps}")
            logger.info(f"   - 学习率: {self.config.learning_rate}")
            logger.info(f"   - 调度器类型: {self.config.scheduler_type}")
            
            return True
            
        except Exception as e:
            logger.error(f"设置连续训练优化器失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _update_model_lora_rank(self, new_rank: int):
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
            if hasattr(self.config, 'lora_config') and self.config.lora_config:
                self.config.lora_config.update_lora_rank(new_rank)
            
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
            if self.config.enable_weight_inheritance and old_lora_weights and old_rank < new_rank:
                logger.info(f"🔄 开始继承LoRA权重: rank {old_rank} → {new_rank}")
                self._inherit_lora_weights(old_lora_weights, old_rank, new_rank)
            
            # 🔥 新功能3：条件优化器重置
            if self.config.enable_optimizer_reset:
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
                    trainable_params = [p for p in self.model_loader.model.parameters() if p.requires_grad]
                    
                    # 更新优化器的param_groups
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
        logger.info(f"  - 优化器重置: {'启用（阶段独立训练）' if self.config.enable_optimizer_reset else '禁用（连续训练模式）'}")
        logger.info(f"  - 权重继承: {'启用（LoRA权重扩展）' if self.config.enable_weight_inheritance else '禁用（随机初始化）'}")
        logger.info("")
        
        logger.info(f"Stage 1 (EASY):   {stage_epochs['easy']} epochs, LoRA rank = {lora_ranks['easy']}")
        logger.info(f"Stage 2 (MEDIUM): {stage_epochs['medium']} epochs, LoRA rank = {lora_ranks['medium']}")
        logger.info(f"Stage 3 (HARD):   {stage_epochs['hard']} epochs, LoRA rank = {lora_ranks['hard']}")
        logger.info("")
        logger.info(f"总训练轮数: {sum(stage_epochs.values())} epochs")
        logger.info(f"输出目录: {os.path.abspath(self.config.output_dir)}")
        logger.info("=" * 80 + "\n")
    
    def cleanup(self):
        """清理资源"""
        if self.model_loader:
            self.model_loader.cleanup()
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# 便捷函数
def create_coco_trainer(model_path: str = "/root/autodl-tmp/llava-1.5-7b",
                       coco_data_root: str = "/root/autodl-tmp/COCO2017",
                       output_dir: str = "./coco_training_output") -> COCOTrainer:
    """创建COCO训练器的便捷函数"""
    config = COCOTrainingConfig()
    config.model_path = model_path
    config.coco_data_root = coco_data_root
    config.output_dir = output_dir
    
    return COCOTrainer(config)

def run_coco_training(model_path: str = "/root/autodl-tmp/llava-1.5-7b",
                     coco_data_root: str = "/root/autodl-tmp/COCO2017",
                     num_epochs: int = 3) -> bool:
    """运行COCO训练的便捷函数"""
    trainer = create_coco_trainer(model_path, coco_data_root)
    trainer.config.num_epochs = num_epochs
    
    try:
        success = trainer.train()
        return success
    finally:
        trainer.cleanup()

