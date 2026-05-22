"""
=============================================================================
LoRA 微调统一配置系统 (LoRA Config)
=============================================================================

本模块提供了LLaVA模型LoRA微调的完整配置管理方案。

【核心组件】
1. LoRAConfig类：定义所有LoRA相关参数
   - LoRA核心参数（秩r、alpha、dropout）
   - 目标模块选择（q_proj, k_proj, v_proj等）
   - 量化配置（4bit/8bit）
   - 训练优化参数

2. LoRAConfigManager：管理多个预定义配置模板
   - 加载配置文件或使用内置模板
   - 支持配置的保存和加载

3. SVD正交化扩展：支持LoRA秩的动态扩展
   - update_lora_rank(): 使用SVD方法扩展权重矩阵
   - 保持旧维度不变，新维度与旧维度正交

【预定义配置模板】
- default_48gb: 48GB显存的默认配置（r=64）
- memory_efficient: 内存高效配置（r=32）
- high_performance: 高性能配置（r=128，不量化）
- progressive_lora: 渐进式训练配置（r=32→64→128动态调整）⭐
- fair_comparison: 公平对比实验配置（与Traditional LoRA对比）

【配置优先级】
1. 文件配置（configs/*.json）- 最高优先级
2. 内置模板配置 - 次优先级
3. 默认配置 - 兜底配置

【核心方法】
- get_lora_config(config_name): 获取指定名称的配置
- update_lora_rank(new_rank, model): 动态扩展LoRA秩（用于渐进式训练）
- save_to_file(path): 保存配置到文件
- from_file(path): 从文件加载配置

【LoRA秩扩展原理】
渐进式训练的核心技术：
1. 提取旧权重矩阵（A_old, B_old）
2. 对A_old进行SVD分解：A_old = U @ S @ Vh
3. 扩展Vh矩阵：
   - Vh_old保持不变（继承旧知识）
   - Vh_new通过QR正交化与Vh_old正交（学习新知识）
4. 拼接成新的权重矩阵

为什么要正交化？
- 防止新旧维度重叠导致梯度冲突
- 确保新维度学习正交的新能力
- 保持训练稳定性

"""

import os
import json
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class LoRAConfig:
    """
    LoRA微调配置类
    
    包含LoRA微调的所有参数配置，支持：
    - LoRA核心参数（秩、alpha、dropout）
    - 目标模块选择
    - 量化配置（4bit/8bit）
    - 训练优化参数
    - 学习率调度
    
    Attributes:
        enable_lora: 是否启用LoRA微调
        lora_r: LoRA秩（rank），控制低秩矩阵的维度，越大表达能力越强
        lora_alpha: LoRA缩放因子，通常设为r的2倍，控制LoRA更新的强度
        lora_dropout: Dropout比例，防止过拟合
    """
    
    # LoRA核心参数
    enable_lora: bool = True
    lora_r: int = 64  # LoRA rank，48GB显存可以用较大值
    lora_alpha: int = 128  # LoRA alpha，通常是r的2倍
    lora_dropout: float = 0.1
    
    # 目标模块配置
    target_modules: List[str] = None  # 将在__post_init__中设置默认值
    
    # LoRA训练策略
    lora_bias: str = "none"  # "none", "all", "lora_only"
    task_type: str = "CAUSAL_LM"  # PEFT任务类型
    
    # 模型保存配置
    save_lora_adapters_only: bool = True  # 只保存LoRA适配器
    merge_and_save_full_model: bool = False  # 是否合并并保存完整模型
    
    # 内存优化配置（48GB显存优化）
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True  # 如果支持的话
    max_memory_per_gpu: str = "46GB"  # 为系统预留2GB
    
    # 量化配置（与LoRA结合）
    use_4bit_quantization: bool = True  # 4bit + LoRA是经典组合
    use_8bit_quantization: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # 训练优化配置
    batch_size: int = 4  # 训练批次大小
    gradient_accumulation_steps: int = 4  # 48GB显存可以用较小值
    dataloader_num_workers: int = 8
    dataloader_pin_memory: bool = True
    
    # 学习率调度
    lora_learning_rate: float = 1e-4  # LoRA通常用较高学习率
    lora_weight_decay: float = 0.01
    lora_warmup_ratio: float = 0.1
    scheduler_type: str = "cosine"  # "linear", "cosine", "cosine_with_restarts"
    cosine_num_cycles: float = 0.5  # 余弦调度的周期数
    
    def __post_init__(self):
        """初始化后处理"""
        # 强制使用8个模块（包含lm_head），无论配置文件如何设置
        self.target_modules = [
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head"  # 语言模型头部
        ]
        
        # 验证配置合理性
        self._validate_config()
    
    def _validate_config(self):
        """验证配置参数的合理性"""
        if self.lora_r <= 0:
            raise ValueError("lora_r必须大于0")
        
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha必须大于0")
        
        if not 0 <= self.lora_dropout <= 1:
            raise ValueError("lora_dropout必须在0-1之间")
        
        if self.use_4bit_quantization and self.use_8bit_quantization:
            logger.warning("同时启用4bit和8bit量化，将优先使用4bit")
            self.use_8bit_quantization = False
        
        logger.info("LoRA配置验证通过")
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return asdict(self)
    
    def save_to_file(self, file_path: str):
        """保存配置到文件"""
        config_dict = self.to_dict()
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"LoRA配置已保存到: {file_path}")
    
    @classmethod
    def from_file(cls, file_path: str) -> 'LoRAConfig':
        """从文件加载配置"""
        if not os.path.exists(file_path):
            logger.warning(f"配置文件不存在: {file_path}，使用默认配置")
            return cls()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # 过滤掉不属于LoRAConfig的参数
        valid_params = cls._get_valid_params()
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_params}
        
        logger.info(f"从文件加载LoRA配置: {file_path}")
        logger.info(f"过滤后的配置参数: {list(filtered_config.keys())}")
        return cls(**filtered_config)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'LoRAConfig':
        """从字典创建配置"""
        # 过滤掉不属于LoRAConfig的参数
        valid_params = cls._get_valid_params()
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_params}
        return cls(**filtered_config)
    
    @classmethod
    def _get_valid_params(cls) -> set:
        """获取LoRAConfig类的有效参数名"""
        from dataclasses import fields
        # 获取dataclass的所有字段名
        return {field.name for field in fields(cls)}
    
    def update_lora_rank(self, new_rank: int, new_alpha: Optional[int] = None):
        """
        动态更新LoRA秩（用于渐进式训练）
        
        这是渐进式训练的关键方法，允许在训练过程中动态调整LoRA秩：
        - Easy阶段：小秩（如32）
        - Medium阶段：中等秩（如64）
        - Hard阶段：大秩（如128）
        
        Args:
            new_rank: 新的LoRA秩
            new_alpha: 新的LoRA alpha（如果为None，则自动设置为rank的2倍）
            
        Raises:
            ValueError: 如果new_rank <= 0
        """
        if new_rank <= 0:
            raise ValueError(f"LoRA秩必须大于0，当前值: {new_rank}")
        
        old_rank = self.lora_r
        self.lora_r = new_rank
        
        if new_alpha is None:
            self.lora_alpha = new_rank * 2  # 通常alpha是rank的2倍
        else:
            self.lora_alpha = new_alpha
        
        logger.info(f"LoRA秩已更新: {old_rank} -> {new_rank}, alpha: {self.lora_alpha}")
    
    def clone(self) -> 'LoRAConfig':
        """克隆一个新的配置对象"""
        return LoRAConfig(**self.to_dict())

class LoRAConfigManager:
    """LoRA配置管理器"""
    
    def __init__(self, config_dir: str = "./configs"):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # 预定义配置模板
        self.templates = {
            "default_48gb": self._create_default_48gb_config(),
            "memory_efficient": self._create_memory_efficient_config(),
            "high_performance": self._create_high_performance_config(),
            "experimental": self._create_experimental_config(),
            "paper_experiment": self._create_paper_experiment_config(),
            "full_coco2017": self._create_full_coco2017_config(),
            "progressive_lora": self._create_progressive_lora_config()
        }
        
        logger.info(f"LoRA配置管理器初始化完成，配置目录: {config_dir}")
    
    def _create_default_48gb_config(self) -> LoRAConfig:
        """创建48GB显存的默认配置"""
        return LoRAConfig(
            enable_lora=True,
            lora_r=64,
            lora_alpha=128,
            lora_dropout=0.1,
            use_4bit_quantization=True,
            use_gradient_checkpointing=True,
            gradient_accumulation_steps=2,  # 48GB可以用较小值
            lora_learning_rate=1e-4,
            max_memory_per_gpu="46GB"
        )
    
    def _create_memory_efficient_config(self) -> LoRAConfig:
        """创建内存高效配置"""
        return LoRAConfig(
            enable_lora=True,
            lora_r=32,  # 较小的rank
            lora_alpha=64,
            lora_dropout=0.1,
            use_4bit_quantization=True,
            use_gradient_checkpointing=True,
            gradient_accumulation_steps=8,  # 更多的梯度累积
            lora_learning_rate=2e-4,
            max_memory_per_gpu="44GB"
        )
    
    def _create_high_performance_config(self) -> LoRAConfig:
        """创建高性能配置（充分利用48GB显存）"""
        return LoRAConfig(
            enable_lora=True,
            lora_r=128,  # 更大的rank
            lora_alpha=256,
            lora_dropout=0.05,
            use_4bit_quantization=False,  # 不量化，使用全精度
            use_8bit_quantization=False,
            use_gradient_checkpointing=False,  # 不使用梯度检查点
            gradient_accumulation_steps=1,
            lora_learning_rate=5e-5,
            max_memory_per_gpu="47GB"
        )
    
    def _create_experimental_config(self) -> LoRAConfig:
        """创建实验性配置"""
        return LoRAConfig(
            enable_lora=True,
            lora_r=96,
            lora_alpha=192,
            lora_dropout=0.15,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "lm_head", "embed_tokens"  # 包含embedding层
            ],
            use_4bit_quantization=True,
            bnb_4bit_use_double_quant=True,
            use_gradient_checkpointing=True,
            gradient_accumulation_steps=3,
            lora_learning_rate=8e-5
        )
    
    def _create_paper_experiment_config(self) -> LoRAConfig:
        """创建论文实验专用配置（48GB显存优化）"""
        return LoRAConfig(
            enable_lora=True,
            lora_r=32,  # 较小的秩，适合论文对比实验
            lora_alpha=64,
            lora_dropout=0.1,
            use_4bit_quantization=True,
            use_gradient_checkpointing=True,
            batch_size=16,  # 48GB显存可以支持更大的batch size
            gradient_accumulation_steps=1,  # 直接使用大batch size
            lora_learning_rate=2e-4,  # 较小秩配合较高学习率
            max_memory_per_gpu="46GB"
        )
    
    def _create_full_coco2017_config(self) -> LoRAConfig:
        """创建完整COCO2017数据集LoRA微调配置（48GB显存优化）"""
        return LoRAConfig(
            enable_lora=True,
            lora_r=32,  # 平衡性能与效率
            lora_alpha=64,
            lora_dropout=0.1,
            use_4bit_quantization=True,
            use_gradient_checkpointing=True,
            batch_size=8,  # 降低batch size以适应完整数据集
            gradient_accumulation_steps=2,  # 通过梯度累积保持有效batch size
            lora_learning_rate=1e-4,  # 稍微降低学习率
            lora_weight_decay=0.01,
            lora_warmup_ratio=0.05,  # 降低warmup比例
            max_memory_per_gpu="46GB",
            dataloader_num_workers=8,
            dataloader_pin_memory=True
        )
    
    def _create_progressive_lora_config(self) -> LoRAConfig:
        """
        创建渐进式LoRA训练配置（核心创新方法）
        
        本配置是Progressive LoRA方法的核心，实现了：
        
        1. 渐进式秩增长策略：
           - Stage 1 (Easy): r=32, 85.8M参数 (1.20%)
           - Stage 2 (Medium): r=64, 171.7M参数 (2.40%)
           - Stage 3 (Hard): r=128, 343.3M参数 (4.80%)
        
        2. 样本复杂度分层：
           - Easy: 简单描述（词少、物体少）
           - Medium: 中等描述
           - Hard: 复杂描述（词多、物体多、关系复杂）
        
        3. 公平对比设计：
           - 总训练轮数：3 epochs（与Traditional相同）
           - 最终参数量：343.3M（与Traditional r=128相同）
           - 超参数：batch_size=32, lr=1e-4, warmup=0.1（完全一致）
        
        Returns:
            LoRAConfig: 渐进式LoRA配置对象
        """
        return LoRAConfig(
            enable_lora=True,
            lora_r=32,  # 初始秩，会在训练中动态调整
            lora_alpha=64,
            lora_dropout=0.05,
            use_4bit_quantization=True,
            use_gradient_checkpointing=True,
            batch_size=32,
            gradient_accumulation_steps=1,
            lora_learning_rate=1e-4,
            lora_weight_decay=0.01,
            lora_warmup_ratio=0.1,
            scheduler_type="cosine",
            max_memory_per_gpu="46GB",
            dataloader_num_workers=8,
            dataloader_pin_memory=True
        )
    
    def get_config(self, config_name: str = "default_48gb") -> LoRAConfig:
        """
        获取指定配置
        
        配置加载优先级：
        1. 文件配置（configs/{config_name}.json）- 最高优先级
        2. 内置模板配置 - 次优先级
        3. 默认配置 - 兜底配置
        
        Args:
            config_name: 配置名称
            
        Returns:
            LoRAConfig: LoRA配置对象
        """
        # 优先尝试从文件加载（允许用户覆盖内置配置）
        config_file = self.config_dir / f"{config_name}.json"
        if config_file.exists():
            logger.info(f"从文件加载配置: {config_file}")
            return LoRAConfig.from_file(str(config_file))
        
        # 如果文件不存在，使用内置模板
        if config_name in self.templates:
            logger.info(f"使用内置模板配置: {config_name}")
            return self.templates[config_name]
        
        logger.warning(f"未找到配置 '{config_name}'，使用默认配置")
        return self.templates["default_48gb"]
    
    def save_template_configs(self):
        """保存所有模板配置到文件"""
        for name, config in self.templates.items():
            config_file = self.config_dir / f"{name}.json"
            config.save_to_file(str(config_file))
        
        logger.info("所有模板配置已保存到文件")
    
    def list_available_configs(self) -> List[str]:
        """列出所有可用的配置"""
        configs = list(self.templates.keys())
        
        # 添加文件中的配置
        for config_file in self.config_dir.glob("*.json"):
            config_name = config_file.stem
            if config_name not in configs:
                configs.append(config_name)
        
        return sorted(configs)
    
    def create_custom_config(self, 
                           config_name: str,
                           base_config: str = "default_48gb",
                           **kwargs) -> LoRAConfig:
        """
        创建自定义配置
        
        Args:
            config_name: 新配置名称
            base_config: 基础配置名称
            **kwargs: 要修改的参数
            
        Returns:
            LoRAConfig: 新的配置对象
        """
        base = self.get_config(base_config)
        config_dict = base.to_dict()
        config_dict.update(kwargs)
        
        new_config = LoRAConfig.from_dict(config_dict)
        
        # 保存到文件
        config_file = self.config_dir / f"{config_name}.json"
        new_config.save_to_file(str(config_file))
        
        logger.info(f"创建自定义配置: {config_name}")
        return new_config

# 全局配置管理器实例
_config_manager = None

def get_lora_config_manager(config_dir: str = "./configs") -> LoRAConfigManager:
    """获取全局LoRA配置管理器"""
    global _config_manager
    if _config_manager is None:
        _config_manager = LoRAConfigManager(config_dir)
    return _config_manager

def get_lora_config(config_name: str = "default_48gb") -> LoRAConfig:
    """Convenience function to get LoRA configuration"""
    manager = get_lora_config_manager()
    return manager.get_config(config_name)

# 预定义的配置常量
DEFAULT_48GB_CONFIG = "default_48gb"
MEMORY_EFFICIENT_CONFIG = "memory_efficient"
HIGH_PERFORMANCE_CONFIG = "high_performance"
EXPERIMENTAL_CONFIG = "experimental"

if __name__ == "__main__":
    # 测试配置系统
    manager = get_lora_config_manager()
    
    # 保存所有模板配置
    manager.save_template_configs()
    
    # 测试配置加载
    config = get_lora_config("default_48gb")
    print("默认48GB配置:")
    print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
    
    # 列出所有配置
    print("\n可用配置:")
    for config_name in manager.list_available_configs():
        print(f"  - {config_name}")


