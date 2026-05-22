"""
VQA7W 传统 LoRA 训练入口脚本

与渐进式版本的区别：
- 使用固定的 LoRA rank（从配置读取）
- 标准的 epoch 循环训练（不是三阶段渐进式）

其余功能与渐进式版本完全相同。
"""

import os
# 设置tokenizers并行化环境变量，避免fork警告
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import logging

# 导入传统 LoRA 版本的 VQA trainer
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from vqa7w_trainer import VQA7WTrainingConfig, VQA7WLoRATrainer

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="VQA7W Traditional LoRA Training (LLaVA-1.5)",
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
        default="./vqa7w_traditional_lora_output",
        help="输出目录",
    )

    # 训练配置
    parser.add_argument("--num_epochs", type=int, default=5, help="训练 epoch 数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="warmup 比例")

    parser.add_argument("--scheduler_type", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--cosine_num_cycles", type=float, default=0.5, help="cosine 调度 cycles 数")

    # 样本/生成
    parser.add_argument("--max_train_samples", type=int, default=None, help="最大训练样本数")
    parser.add_argument("--max_val_samples", type=int, default=None, help="最大验证样本数")
    parser.add_argument("--max_test_samples", type=int, default=None, help="最大测试样本数")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="VQA 回答最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")

    parser.add_argument(
        "--lora_config_name",
        type=str,
        default="traditional_lora_r32",
        help="LoRA 配置名称（传统 LoRA 使用固定 rank 32）",
    )

    return parser.parse_args()

def check_environment() -> bool:
    """环境检查"""
    import torch
    logger.info("检查运行环境 (VQA7W Traditional LoRA)...")
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

    cfg.max_train_samples = args.max_train_samples
    cfg.max_val_samples = args.max_val_samples
    cfg.max_test_samples = args.max_test_samples

    cfg.max_new_tokens = args.max_new_tokens
    cfg.temperature = args.temperature
    cfg.lora_config_name = args.lora_config_name

    trainer = VQA7WLoRATrainer(cfg)
    success = trainer.train()
    if success:
        logger.info("✅ VQA7W 传统 LoRA 训练完成")
    else:
        logger.error("❌ VQA7W 传统 LoRA 训练失败")

if __name__ == "__main__":
    main()


