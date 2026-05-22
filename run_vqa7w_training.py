"""
VQA7W 渐进式 LoRA 训练入口脚本

与 run_progressive_training.py 平行存在，不修改图像描述训练逻辑。
"""

import os
# 设置tokenizers并行化环境变量，避免fork警告
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import logging

from vqa7w_trainer import VQA7WTrainingConfig, VQA7WLoRATrainer

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="VQA7W Progressive LoRA Training (LLaVA-1.5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 路径
    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/autodl-tmp/llava-1.5-7b",
        help="LLaVA 模型路径",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/root/autodl-tmp/VQA7W",
        help="VQA7W 数据根目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./vqa7w_training_output",
        help="输出目录",
    )

    # 训练配置
    parser.add_argument("--num_epochs", type=int, default=5, help="总 epoch 数（用于估算步数）")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="warmup 比例")

    parser.add_argument("--scheduler_type", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--cosine_num_cycles", type=float, default=0.5, help="cosine 调度 cycles 数")

    # 渐进式配置
    parser.add_argument("--easy_epochs", type=int, default=1, help="Easy 阶段 epoch 数")
    parser.add_argument("--medium_epochs", type=int, default=1, help="Medium 阶段 epoch 数")
    parser.add_argument("--hard_epochs", type=int, default=1, help="Hard 阶段 epoch 数")
    parser.add_argument(
        "--complexity_thresholds",
        type=float,
        nargs=2,
        default=[33.33, 66.67],
        help="复杂度分层百分位数",
    )

    # 样本/生成
    parser.add_argument("--max_train_samples", type=int, default=None, help="最大训练样本数")
    parser.add_argument("--max_val_samples", type=int, default=None, help="最大验证样本数")
    parser.add_argument("--max_test_samples", type=int, default=None, help="最大测试样本数")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="VQA 回答最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")

    parser.add_argument(
        "--lora_config_name",
        type=str,
        default="progressive_lora",
        help="LoRA 配置名称（与现有 lora_config 保持一致）",
    )

    # 渐进式 LoRA 秩配置（默认值：16-24-32）
    parser.add_argument(
        "--easy_lora_rank",
        type=int,
        default=16,
        help="Easy 阶段 LoRA 秩",
    )
    parser.add_argument(
        "--medium_lora_rank",
        type=int,
        default=24,
        help="Medium 阶段 LoRA 秩",
    )
    parser.add_argument(
        "--hard_lora_rank",
        type=int,
        default=32,
        help="Hard 阶段 LoRA 秩",
    )
    parser.add_argument(
        "--disable_weight_inheritance",
        action="store_true",
        help="禁用 LoRA 权重继承（SVD 扩展），每个阶段新秩随机初始化",
    )

    return parser.parse_args()

def check_environment() -> bool:
    """简化版环境检查，不修改原有 caption 流程的 check_environment。"""
    import torch
    logger.info("检查运行环境 (VQA7W)...")
    logger.info(f"PyTorch 版本: {torch.__version__}")
    if not torch.cuda.is_available():
        logger.warning("未检测到 GPU，VQA7W 训练可能非常慢")
    return True

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()

    if not check_environment():
        return

    cfg = VQA7WTrainingConfig()
    cfg.model_path = args.model_path
    cfg.data_root = args.data_path
    cfg.output_dir = args.output_dir

    cfg.num_epochs = args.num_epochs
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.learning_rate
    cfg.weight_decay = args.weight_decay
    cfg.warmup_ratio = args.warmup_ratio

    cfg.scheduler_type = args.scheduler_type
    cfg.cosine_num_cycles = args.cosine_num_cycles

    cfg.easy_epochs = args.easy_epochs
    cfg.medium_epochs = args.medium_epochs
    cfg.hard_epochs = args.hard_epochs
    cfg.complexity_thresholds = tuple(args.complexity_thresholds)

    cfg.max_train_samples = args.max_train_samples
    cfg.max_val_samples = args.max_val_samples
    cfg.max_test_samples = args.max_test_samples

    cfg.max_new_tokens = args.max_new_tokens
    cfg.temperature = args.temperature
    cfg.lora_config_name = args.lora_config_name

    # 渐进式 LoRA 配置
    cfg.easy_lora_rank = args.easy_lora_rank
    cfg.medium_lora_rank = args.medium_lora_rank
    cfg.hard_lora_rank = args.hard_lora_rank
    cfg.enable_weight_inheritance = not args.disable_weight_inheritance

    trainer = VQA7WLoRATrainer(cfg)
    success = trainer.progressive_train()
    if success:
        logger.info("✅ VQA7W 渐进式 LoRA 训练完成")
    else:
        logger.error("❌ VQA7W 渐进式 LoRA 训练失败")

if __name__ == "__main__":
    main()


