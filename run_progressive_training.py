#!/usr/bin/env python3
"""
=============================================================================
渐进式LoRA训练主程序 (Progressive LoRA Training Main Script)
=============================================================================

本脚本实现了Progressive LoRA训练的完整流程，是整个项目的执行入口。

【核心思想】
渐进式学习（Curriculum Learning）：
- 从简单样本开始训练，逐步增加样本复杂度
- 随着样本复杂度增加，动态扩展LoRA秩（模型容量）
- 每个阶段都继承上一阶段的知识，避免灾难性遗忘

【执行流程】
阶段1: 渐进式训练（包含验证集评估）
  ├── Easy阶段:   rank=32,  简单样本（词数少、物体简单）
  ├── Medium阶段: rank=64,  中等样本
  └── Hard阶段:   rank=128, 困难样本（词数多、关系复杂）

阶段2: 测试集预测（可选）
  └── 在测试集上生成预测结果，可提交至官方评测

【核心创新】
1. 样本复杂度评估系统（coco_dataset.py）
   - 基于词数、物体数、属性数、关系复杂度的多维度评估
   - 自动将数据集按复杂度三等分（Easy/Medium/Hard）

2. 渐进式LoRA秩扩展（coco_trainer.py）
   - 使用SVD正交化方法扩展权重矩阵
   - 保持旧维度不变，新维度与旧维度正交
   - 确保训练稳定性和知识继承

3. 公平对比设计
   - 与Traditional LoRA保持相同的总训练量（epoch数相同）
   - 最终参数量相同（都是rank=128）
   - 唯一区别：训练顺序和秩的增长方式

【数据集支持】
- COCO 2017 Caption：官方标准数据集
- Flickr30K：小规模数据集，测试集有标注

【输出文件】
- lora_adapters/best_model: 基于验证集CIDEr指标选择的最佳模型
- lora_adapters/final_progressive: 最终训练完成的模型
- progressive_training_history.json: 训练历史和验证集指标
- test_predictions/: 测试集预测结果

"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# ============================================================
# 日志配置
# ============================================================
# 配置双输出：同时输出到控制台和文件
# - 控制台：实时查看训练进度
# - 文件：保存完整训练日志，便于后续分析
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # 输出到控制台
        logging.FileHandler('progressive_training.log', encoding='utf-8')  # 输出到文件
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """
    解析命令行参数
    
    支持自定义模型路径、数据路径、训练配置、复杂度阈值等参数。
    详细使用方法请参考 PROJECT_GUIDE.md 或运行 python run_progressive_training.py --help
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description='渐进式LoRA训练启动脚本 - Progressive LoRA Training for LLaVA on COCO2017',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 基础用法（使用默认COCO配置）
  python run_progressive_training.py

  # 使用Flickr30K数据集
  python run_progressive_training.py \\
      --dataset flickr30k \\
      --data_path /path/to/Flickr30K

  # 使用COCO Karpathy Split数据集
  python run_progressive_training.py \\
      --dataset coco_karpathy \\
      --data_path /root/autodl-tmp/coco2014

  # 自定义路径
  python run_progressive_training.py \\
      --model_path /path/to/llava-1.5-7b \\
      --data_path /path/to/COCO2017 \\
      --output_dir ./my_output

  # 自定义训练配置
  python run_progressive_training.py \\
      --easy_epochs 2 \\
      --medium_epochs 2 \\
      --hard_epochs 1 \\
      --easy_rank 16 \\
      --medium_rank 48 \\
      --hard_rank 96

  # 小规模测试运行
  python run_progressive_training.py \\
      --max_train_samples 1000 \\
      --max_val_samples 500

  # 自定义复杂度阈值
  python run_progressive_training.py \\
      --complexity_thresholds 0.25 0.75
        """
    )
    
    # 路径配置
    parser.add_argument('--model_path', type=str,
                       default='/root/autodl-tmp/llava-1.5-7b',
                       help='LLaVA模型路径（默认：/root/autodl-tmp/llava-1.5-7b）')
    parser.add_argument('--data_path', type=str,
                       default='/root/autodl-tmp/COCO2017',
                       help='数据集路径（COCO默认：/root/autodl-tmp/COCO2017）')
    parser.add_argument('--dataset', type=str,
                       default='coco',
                       choices=['coco', 'flickr8k', 'flickr30k', 'coco_karpathy', 'vizwiz'],
                       help='数据集类型：coco、flickr8k、flickr30k、coco_karpathy 或 vizwiz（默认：coco）')
    parser.add_argument('--output_dir', type=str,
                       default='./progressive_training_output',
                       help='输出目录（默认在当前项目目录）')
    
    # 训练配置
    parser.add_argument('--easy_epochs', type=int, default=1,
                       help='简单样本训练轮数')
    parser.add_argument('--medium_epochs', type=int, default=1,
                       help='中等样本训练轮数')
    parser.add_argument('--hard_epochs', type=int, default=1,
                       help='困难样本训练轮数')
    
    parser.add_argument('--easy_rank', type=int, default=32,
                       help='简单阶段LoRA秩')
    parser.add_argument('--medium_rank', type=int, default=64,
                       help='中等阶段LoRA秩')
    parser.add_argument('--hard_rank', type=int, default=128,
                       help='困难阶段LoRA秩')
    
    parser.add_argument('--complexity_thresholds', type=float, nargs=2,
                       default=[33.33, 66.67],
                       help='复杂度分层百分位数 (默认33.33 66.67表示三等分)')
    
    # 超参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='预热比例')
    
    # 数据配置
    parser.add_argument('--max_train_samples', type=int, default=None,
                       help='最大训练样本数（None=全部）')
    parser.add_argument('--max_val_samples', type=int, default=None,
                       help='最大验证样本数（None=全部）')
    
    # 生成参数
    parser.add_argument('--max_new_tokens', type=int, default=50,
                       help='生成的最大token数（默认：50）')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='生成温度（默认：0.7）')
    
    # 流程控制选项
    parser.add_argument('--test_only', action='store_true',
                       help='仅运行功能测试，不进行实际训练')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='从检查点恢复训练（暂未实现）')
    parser.add_argument('--skip_test_prediction', action='store_true',
                       help='跳过训练后的测试预测阶段（阶段2）')
    parser.add_argument('--full_pipeline', action='store_true',
                       help='运行完整流程：训练+评估+测试预测（默认）')
    
    # 评估和测试参数
    parser.add_argument('--max_eval_samples', type=int, default=None,
                       help='评估阶段的最大样本数（默认使用max_val_samples）')
    parser.add_argument('--max_test_samples', type=int, default=None,
                       help='测试预测阶段的最大样本数（None=全部测试集）')
    
    return parser.parse_args()

def check_environment():
    """
    检查运行环境
    
    检查项包括：
    - Python版本
    - PyTorch和CUDA是否可用
    - GPU显存大小
    - Transformers和PEFT库是否安装
    
    Returns:
        bool: 环境检查是否通过
    """
    logger.info("=" * 80)
    logger.info("🔍 检查运行环境...")
    logger.info("=" * 80)
    
    # 检查Python版本
    import sys
    python_version = sys.version_info
    logger.info(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查PyTorch
    try:
        import torch
        logger.info(f"PyTorch版本: {torch.__version__}")
        logger.info(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA版本: {torch.version.cuda}")
            logger.info(f"GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"    显存: {memory:.2f} GB")
    except ImportError:
        logger.error("❌ PyTorch未安装！")
        return False
    
    # 检查transformers
    try:
        import transformers
        logger.info(f"Transformers版本: {transformers.__version__}")
    except ImportError:
        logger.error("❌ Transformers未安装！")
        return False
    
    # 检查PEFT
    try:
        import peft
        logger.info(f"PEFT版本: {peft.__version__}")
    except ImportError:
        logger.error("❌ PEFT未安装！")
        return False
    
    logger.info("✅ 环境检查通过！\n")
    return True

def run_test():
    """
    运行功能测试
    
    执行test_progressive_lora.py中的测试用例，验证：
    - 数据集加载
    - 模型初始化
    - LoRA配置
    - 渐进式训练逻辑
    
    Returns:
        bool: 测试是否通过
    """
    logger.info("=" * 80)
    logger.info("🧪 运行功能测试...")
    logger.info("=" * 80)
    
    try:
        import test_progressive_lora
        success = test_progressive_lora.main()
        if success:
            logger.info("✅ 功能测试通过！")
            return True
        else:
            logger.error("❌ 功能测试失败！")
            return False
    except Exception as e:
        logger.error(f"❌ 测试脚本执行失败: {str(e)}")
        return False

def run_evaluation(args, best_adapter_path):
    """
    运行验证集评估（训练后的确认步骤）
    
    【重要】本阶段仅在验证集上评估模型性能，用于确认训练效果。
    测试集的评估将在后续的"阶段2: 测试预测"中完成。
    
    评估指标：
    - BLEU-1/2/3/4：n-gram精确匹配
    - ROUGE-L：最长公共子序列
    - CIDEr：共识评分（COCO/Flickr30K最重要指标）
    - METEOR：基于词义的匹配
    
    Args:
        args: 命令行参数
        best_adapter_path: 最佳LoRA适配器路径
        
    Returns:
        tuple: (success: bool, eval_results: dict) 评估是否成功和验证集评估结果
    """
    logger.info("\n" + "=" * 80)
    logger.info("📊 验证集评估")
    logger.info("=" * 80)
    logger.info("本阶段仅评估验证集，测试集评估将在'阶段2: 测试预测'中完成")
    logger.info("=" * 80 + "\n")
    
    try:
        from model_loader import create_lora_model_loader
        
        # 导入评估器（所有数据集类型都使用COCOCaptionEvaluator，通过适配器实现兼容）
        from coco_evaluator import COCOCaptionEvaluator as Evaluator, COCOEvaluationConfig as EvalConfig
        logger.info(f"使用COCOCaptionEvaluator（通过适配器支持{args.dataset.upper()}数据集）")
        
        # 加载训练好的模型
        logger.info(f"正在加载训练好的模型: {best_adapter_path}")
        model_loader = create_lora_model_loader(
            model_path=args.model_path,
            lora_config_name="progressive_lora",
            adapter_path=best_adapter_path
        )
        
        # 创建评估配置（确保在当前项目目录）
        eval_config = EvalConfig()
        
        # 设置数据集路径（统一使用coco_data_root，适配器会自动处理）
        eval_config.coco_data_root = args.data_path
        
        eval_config.output_dir = os.path.join(os.getcwd(), args.output_dir, "evaluation_output")
        eval_config.results_dir = os.path.join(eval_config.output_dir, "results")
        eval_config.predictions_dir = os.path.join(eval_config.output_dir, "predictions")
        eval_config.max_eval_samples = args.max_eval_samples or args.max_val_samples
        eval_config.max_new_tokens = args.max_new_tokens
        eval_config.temperature = args.temperature
        
        # 设置评估的数据分割
        # 标准流程：训练后只评估验证集，测试集留到阶段2
        eval_config.eval_splits = ["val"]
        logger.info("📋 评估数据集: 验证集 (val)")
        logger.info("💡 测试集将在'阶段2: 测试预测'中处理")
        
        # 创建输出目录
        os.makedirs(eval_config.results_dir, exist_ok=True)
        os.makedirs(eval_config.predictions_dir, exist_ok=True)
        
        logger.info(f"评估输出目录: {os.path.abspath(eval_config.output_dir)}")
        
        # 创建评估器
        evaluator = Evaluator(eval_config)
        
        # 运行评估
        success = evaluator.evaluate_model(model_loader)
        
        if success:
            logger.info("✅ 验证集评估成功完成")
            
            # 读取并返回评估结果
            results_file = os.path.join(eval_config.results_dir, "evaluation_results.json")
            if os.path.exists(results_file):
                with open(results_file, 'r', encoding='utf-8') as f:
                    eval_results = json.load(f)
                logger.info("📊 验证集评估指标已保存")
                return True, eval_results
            else:
                logger.warning("未找到验证集评估结果文件")
                return True, None
        else:
            logger.warning("⚠️  验证集评估失败")
            return False, None
        
    except Exception as e:
        logger.error(f"评估阶段异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None

def run_test_prediction(args, best_adapter_path):
    """
    运行测试预测阶段
    
    对测试集进行推理预测，生成官方提交格式的JSON文件。
    测试集没有ground truth，需要提交到官方服务器进行评测。
    
    Args:
        args: 命令行参数
        best_adapter_path: 最佳LoRA适配器路径
        
    Returns:
        tuple: (success: bool, test_metrics: dict) 测试预测是否成功和测试集评估指标
    """
    logger.info("\n" + "=" * 80)
    logger.info("🔮 开始测试预测阶段")
    logger.info("=" * 80 + "\n")
    
    # 数据集信息
    if args.dataset == 'flickr30k':
        logger.info(f"ℹ️  将在Flickr30K测试集上进行预测评估")
    elif args.dataset == 'coco_karpathy':
        logger.info(f"ℹ️  将在COCO Karpathy测试集上进行预测评估")
    elif args.dataset == 'vizwiz':
        logger.info(f"ℹ️  将在VizWiz-Captions 测试集上进行预测评估（若提供测试标注将计算指标）")
    
    try:
        from model_loader import create_lora_model_loader
        from coco_test_predictor import COCOTestPredictor, COCOTestPredictorConfig
        
        # 加载训练好的模型
        logger.info(f"正在加载训练好的模型: {best_adapter_path}")
        model_loader = create_lora_model_loader(
            model_path=args.model_path,
            lora_config_name="progressive_lora",
            adapter_path=best_adapter_path
        )
        
        # 创建测试配置（确保在当前项目目录）
        test_config = COCOTestPredictorConfig()
        test_config.coco_data_root = args.data_path
        test_config.output_dir = os.path.join(os.getcwd(), args.output_dir, "test_predictions")
        test_config.max_test_samples = args.max_test_samples
        test_config.max_new_tokens = args.max_new_tokens
        test_config.temperature = args.temperature
        test_config.team_name = "LLaVA-Progressive-LoRA"
        test_config.method_description = f"LLaVA-1.5-7B with Progressive LoRA fine-tuning on {args.dataset.upper()}"
        
        logger.info(f"测试预测输出目录: {os.path.abspath(test_config.output_dir)}")
        
        # 创建测试预测器
        predictor = COCOTestPredictor(test_config)
        
        # 运行测试预测
        success = predictor.predict_test_set(model_loader)
        
        if success:
            logger.info("✅ 测试预测阶段成功完成")
            
            # 尝试读取测试集评估指标（如果测试集有标注）
            test_metrics = None
            metrics_file = os.path.join(test_config.output_dir, "test_metrics.json")
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r', encoding='utf-8') as f:
                        metrics_data = json.load(f)
                    # 提取metrics部分（文件结构：{'metrics': {...}, 'metadata': {...}}）
                    test_metrics = metrics_data.get('metrics', metrics_data)
                    logger.info("✅ 成功读取测试集评估指标")
                except Exception as e:
                    logger.warning(f"读取测试集指标文件失败: {e}")
            
            return True, test_metrics
        else:
            logger.warning("⚠️  测试预测阶段失败")
            return False, None
        
    except Exception as e:
        logger.error(f"测试预测阶段异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None

def find_best_adapter(output_dir):
    """
    查找最佳的LoRA适配器
    
    查找优先级：
    1. best_model - 验证集CIDEr指标最高的模型（优先）
    2. final_progressive - 最终训练完成的模型
    3. checkpoint-step-* - 最新的检查点
    
    注意：best_model是训练过程中基于验证集CIDEr指标选择的最佳checkpoint，
         符合学术界标准做法（使用主要评估指标而非loss选择最佳模型）
    
    Args:
        output_dir: 输出目录
        
    Returns:
        str: 最佳适配器路径，如果未找到则返回None
    """
    try:
        adapters_dir = os.path.join(output_dir, "lora_adapters")
        
        # 优先查找 best_model
        best_model_path = os.path.join(adapters_dir, "best_model")
        if os.path.exists(best_model_path):
            logger.info(f"找到最佳模型: {best_model_path}")
            return best_model_path
        
        # 查找 final_progressive
        final_progressive_path = os.path.join(adapters_dir, "final_progressive")
        if os.path.exists(final_progressive_path):
            logger.info(f"找到最终模型: {final_progressive_path}")
            return final_progressive_path
        
        # 查找最新的checkpoint
        if os.path.exists(adapters_dir):
            checkpoints = [d for d in os.listdir(adapters_dir) 
                         if d.startswith("checkpoint-step-")]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
                latest_checkpoint = os.path.join(adapters_dir, checkpoints[-1])
                logger.info(f"找到最新检查点: {latest_checkpoint}")
                return latest_checkpoint
        
        logger.error("未找到任何可用的LoRA适配器")
        return None
        
    except Exception as e:
        logger.error(f"查找最佳适配器失败: {str(e)}")
        return None

def main():
    """
    主函数 - 渐进式LoRA训练完整流程
    
    执行流程：
    1. 环境检查
    2. 路径验证
    3. 渐进式训练（Easy→Medium→Hard）
    4. 模型评估（可选）
    5. 测试预测（可选）
    
    Returns:
        bool: 训练是否成功
    """
    args = parse_args()
    
    logger.info("\n" + "=" * 80)
    logger.info("🚀 渐进式LoRA训练启动 - 完整流程版")
    logger.info("=" * 80 + "\n")
    
    # 检查环境
    if not check_environment():
        logger.error("环境检查失败，请先安装必要的依赖")
        return False
    
    # 如果只是测试
    if args.test_only:
        return run_test()
    
    # 检查路径
    logger.info("📁 检查路径...")
    if not os.path.exists(args.model_path):
        logger.error(f"❌ 模型路径不存在: {args.model_path}")
        logger.info("请使用 --model_path 指定正确的LLaVA模型路径")
        return False
    
    if not os.path.exists(args.data_path):
        logger.error(f"❌ 数据集路径不存在: {args.data_path}")
        logger.info(f"请使用 --data_path 指定正确的{args.dataset.upper()}数据集路径")
        return False
    
    logger.info(f"✅ 模型路径: {args.model_path}")
    logger.info(f"✅ 数据集类型: {args.dataset.upper()}")
    logger.info(f"✅ 数据集路径: {args.data_path}")
    logger.info(f"✅ 输出目录: {os.path.abspath(args.output_dir)}\n")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 显示训练配置
    logger.info("=" * 80)
    logger.info("📋 训练配置")
    logger.info("=" * 80)
    logger.info(f"训练轮数配置:")
    logger.info(f"  - Easy阶段: {args.easy_epochs} epochs")
    logger.info(f"  - Medium阶段: {args.medium_epochs} epochs")
    logger.info(f"  - Hard阶段: {args.hard_epochs} epochs")
    logger.info(f"  - 总计: {args.easy_epochs + args.medium_epochs + args.hard_epochs} epochs")
    logger.info(f"\nLoRA秩配置:")
    logger.info(f"  - Easy阶段: rank = {args.easy_rank}")
    logger.info(f"  - Medium阶段: rank = {args.medium_rank}")
    logger.info(f"  - Hard阶段: rank = {args.hard_rank}")
    logger.info(f"\n复杂度阈值: {args.complexity_thresholds}")
    logger.info(f"\n超参数:")
    logger.info(f"  - Batch size: {args.batch_size}")
    logger.info(f"  - Learning rate: {args.learning_rate}")
    logger.info(f"  - Warmup ratio: {args.warmup_ratio}")
    logger.info(f"\n数据配置:")
    logger.info(f"  - 训练样本数: {args.max_train_samples if args.max_train_samples else '全部'}")
    logger.info(f"  - 验证样本数: {args.max_val_samples}")
    logger.info(f"  - 评估样本数: {args.max_eval_samples if args.max_eval_samples else args.max_val_samples}")
    logger.info(f"  - 测试样本数: {args.max_test_samples if args.max_test_samples else '全部测试集'}")
    logger.info(f"\n流程配置:")
    logger.info(f"  - 阶段1: 渐进式训练 ✅")
    logger.info(f"  - 阶段2: 测试预测 {'⏭️ 跳过' if args.skip_test_prediction else '✅'}")
    logger.info("=" * 80 + "\n")
    
    # 导入训练模块（所有数据集类型都使用COCOTrainer，通过适配器实现兼容）
    try:
        from coco_trainer import COCOTrainer as Trainer, COCOTrainingConfig as TrainingConfig
        logger.info(f"使用COCOTrainer（通过适配器支持{args.dataset.upper()}数据集）")
    except ImportError as e:
        logger.error(f"❌ 导入训练模块失败: {str(e)}")
        return False
    
    # ========== 阶段1: 训练（包含验证） ==========
    logger.info("⚙️  创建训练配置...")
    config = TrainingConfig(lora_config_name="progressive_lora")
    config.model_path = args.model_path
    
    # 设置数据集路径（统一使用coco_data_root，适配器会自动处理）
    config.coco_data_root = args.data_path
    
    config.output_dir = args.output_dir
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.warmup_ratio = args.warmup_ratio
    config.max_train_samples = args.max_train_samples
    config.max_val_samples = args.max_val_samples
    config.max_new_tokens = args.max_new_tokens
    config.temperature = args.temperature
    
    # 设置LoRA初始秩
    if hasattr(config, 'lora_config') and config.lora_config:
        config.lora_config.update_lora_rank(args.easy_rank)
    
    # 创建训练器
    logger.info("🔧 创建训练器...")
    try:
        trainer = Trainer(config)
    except Exception as e:
        logger.error(f"❌ 创建训练器失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # 开始训练
    logger.info("\n" + "=" * 80)
    logger.info("🎯 阶段1/2: 渐进式LoRA训练（包含验证集评估）")
    logger.info("=" * 80)
    logger.info("💡 训练过程中会定期在验证集上评估Loss和标准指标（BLEU/CIDEr/ROUGE等）")
    logger.info("=" * 80 + "\n")
    
    training_success = False
    try:
        success = trainer.progressive_train(
            stage_epochs={
                'easy': args.easy_epochs,
                'medium': args.medium_epochs,
                'hard': args.hard_epochs
            },
            lora_ranks={
                'easy': args.easy_rank,
                'medium': args.medium_rank,
                'hard': args.hard_rank
            },
            complexity_thresholds=tuple(args.complexity_thresholds)
        )
        
        if success:
            logger.info("\n" + "=" * 80)
            logger.info("🎉 训练阶段成功完成！")
            logger.info("=" * 80)
            logger.info(f"\n📂 训练输出文件位置:")
            logger.info(f"  - 最终模型: {os.path.join(args.output_dir, 'final_model_progressive')}")
            logger.info(f"  - LoRA适配器: {os.path.join(args.output_dir, 'lora_adapters/final_progressive')}")
            logger.info(f"  - 训练历史: {os.path.join(args.output_dir, 'progressive_training_history.json')}")
            logger.info(f"  - 检查点: {os.path.join(args.output_dir, 'checkpoints/')}")
            training_success = True
        else:
            logger.error("❌ 训练阶段失败！")
            return False
            
    except KeyboardInterrupt:
        logger.warning("\n⚠️  训练被用户中断")
        logger.info("正在保存当前状态...")
        try:
            trainer.save_checkpoint(suffix="_interrupted")
            logger.info("✅ 当前状态已保存")
        except:
            logger.error("❌ 保存失败")
        return False
        
    except Exception as e:
        logger.error(f"❌ 训练过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        logger.info("\n🧹 清理训练资源...")
        try:
            trainer.cleanup()
            logger.info("✅ 训练资源清理完成")
        except:
            logger.warning("⚠️  训练资源清理失败（可能已经清理）")
    
    # 如果训练失败，不继续后续步骤
    if not training_success:
        return False
    
    # 查找最佳适配器
    best_adapter_path = find_best_adapter(args.output_dir)
    if not best_adapter_path:
        logger.error("未找到训练好的模型，无法继续测试预测")
        return False
    
    logger.info(f"\n✅ 训练阶段已完成，验证集评估已在训练过程中完成")
    logger.info(f"📊 最佳模型位于: {best_adapter_path}")
    logger.info(f"💡 提示: 验证集评估指标已保存在训练历史文件中")
    
    # ========== 阶段2: 测试预测 ==========
    test_prediction_success = True
    test_metrics = None
    if not args.skip_test_prediction:
        # 标准流程：所有数据集都在阶段2进行测试集预测
        test_prediction_success, test_metrics = run_test_prediction(args, best_adapter_path)
        if not test_prediction_success:
            logger.warning("⚠️  测试预测阶段失败")
    else:
        logger.info("\n⏭️  跳过测试预测阶段")
    
    # ========== 总结 ==========
    logger.info("\n" + "=" * 80)
    logger.info("🏁 完整流程执行总结")
    logger.info("=" * 80)
    logger.info(f"✅ 阶段1 - 训练（包含验证集评估）: 成功")
    
    # 测试预测阶段状态
    if args.skip_test_prediction:
        test_status = "跳过"
    elif test_prediction_success:
        test_status = "成功"
    else:
        test_status = "失败"
    logger.info(f"{'✅' if test_prediction_success else '⏭️' if '跳过' in test_status else '❌'} 阶段2 - 测试预测: {test_status}")
    logger.info("=" * 80)
    
    # ========== 显示评估指标 ==========
    # 读取训练历史中的验证集指标
    history_file = os.path.join(args.output_dir, "progressive_training_history.json")
    val_metrics_from_training = None
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                training_history = json.load(f)
                # 获取最佳验证指标
                if 'best_val_metrics' in training_history:
                    val_metrics_from_training = training_history['best_val_metrics']
        except Exception as e:
            logger.warning(f"无法读取训练历史: {e}")
    
    display_metrics = None
    display_split = None
    
    if (args.dataset in ['flickr30k', 'coco_karpathy', 'vizwiz']) and test_metrics:
        # Flickr30K / COCO Karpathy / VizWiz: 显示测试集指标作为最终结果
        display_metrics = test_metrics
        display_split = "TEST"
        logger.info("\n" + "=" * 80)
        logger.info("📊 最终评估指标（测试集）")
        logger.info("=" * 80)
    elif val_metrics_from_training:
        # 显示训练过程中的最佳验证集指标
        display_split = "VAL"
        display_metrics = val_metrics_from_training
        logger.info("\n" + "=" * 80)
        if args.dataset == 'flickr30k' or args.dataset == 'coco_karpathy':
            logger.info("📊 验证集评估指标（训练时最佳，用于模型选择）")
        else:
            logger.info("📊 验证集评估指标（训练时最佳）")
        logger.info("=" * 80)
    
    # 统一显示指标
    if display_metrics:
        if display_split == "TEST":
            logger.info(f"\n【{display_split}数据集评估结果】⭐ (最终结果)")
        
        logger.info(f"  📈 BLEU指标:")
        logger.info(f"    - BLEU-1: {display_metrics.get('Bleu_1', 0):.4f}")
        logger.info(f"    - BLEU-2: {display_metrics.get('Bleu_2', 0):.4f}")
        logger.info(f"    - BLEU-3: {display_metrics.get('Bleu_3', 0):.4f}")
        logger.info(f"    - BLEU-4: {display_metrics.get('Bleu_4', 0):.4f}")
        
        logger.info(f"\n  📈 其他指标:")
        logger.info(f"    - ROUGE-L: {display_metrics.get('ROUGE_L', 0):.4f}")
        logger.info(f"    - CIDEr: {display_metrics.get('CIDEr', 0):.4f} ⭐ (最重要)")
        
        if 'METEOR' in display_metrics and display_metrics['METEOR'] > 0:
            logger.info(f"    - METEOR: {display_metrics.get('METEOR', 0):.4f}")
        if 'SPICE' in display_metrics and display_metrics['SPICE'] > 0:
            logger.info(f"    - SPICE: {display_metrics.get('SPICE', 0):.4f}")
        
        if display_split == "TEST" and (args.dataset in ['flickr30k', 'coco_karpathy', 'vizwiz']):
            logger.info(f"\n  💡 说明: 这是测试集的最终评估结果")
            logger.info(f"         验证集指标仅用于训练过程中的模型选择")
        
        logger.info("\n" + "=" * 80)
    
    # 获取绝对路径
    output_base = os.path.abspath(args.output_dir)
    test_output = os.path.abspath(os.path.join(output_base, "test_predictions"))
    
    logger.info(f"\n📂 所有输出文件位置:")
    logger.info(f"主输出目录: {os.path.abspath(output_base)}")
    logger.info(f"  ├── lora_adapters/          # LoRA适配器")
    logger.info(f"  ├── checkpoints/            # 训练检查点")
    logger.info(f"  ├── progressive_training_history.json  # 训练历史（包含验证集评估指标）")
    
    if not args.skip_test_prediction and os.path.exists(test_output):
        logger.info(f"\n测试预测: {test_output}")
        # 列出测试预测目录中的文件
        test_files = [f for f in os.listdir(test_output) if f.endswith('.json')]
        if test_files:
            for test_file in sorted(test_files):
                file_path = os.path.join(test_output, test_file)
                if 'coco' in test_file.lower() and 'test_predictions' in test_file:
                    logger.info(f"  ├── {test_file}  ✅ ⭐ (主要预测文件)")
                elif 'raw' in test_file.lower():
                    logger.info(f"  ├── {test_file}  ✅ (原始格式)")
                elif 'metadata' in test_file.lower():
                    logger.info(f"  ├── {test_file}  ✅ (元数据)")
                elif 'evaluation' in test_file.lower():
                    logger.info(f"  ├── {test_file}  ✅ (测试集评估)")
                else:
                    logger.info(f"  ├── {test_file}  ✅")
    
    logger.info("\n💡 使用提示:")
    logger.info(f"  1. 查看训练历史和验证集指标:")
    logger.info(f"     cat {os.path.join(output_base, 'progressive_training_history.json')}")
    
    if not args.skip_test_prediction and os.path.exists(test_output):
        # 根据数据集类型给出不同提示
        if args.dataset == 'coco':
            logger.info(f"  2. COCO测试集预测文件（用于提交）:")
            logger.info(f"     查看目录: {test_output}")
            logger.info(f"     主要文件: test_predictions_coco_*.json")
            logger.info(f"     提交至: https://competitions.codalab.org/competitions/3221")
        elif args.dataset == 'flickr30k':
            logger.info(f"  2. Flickr30K测试集结果:")
            logger.info(f"     评估文件: {os.path.join(test_output, 'test_evaluation_results.json')}")
        elif args.dataset == 'coco_karpathy':
            logger.info(f"  2. COCO Karpathy测试集结果:")
            logger.info(f"     评估文件: {os.path.join(test_output, 'test_evaluation_results.json')}")
        elif args.dataset == 'vizwiz':
            logger.info(f"  2. VizWiz-Captions 测试集结果（如有标注）:")
            logger.info(f"     评估文件: {os.path.join(test_output, 'test_evaluation_results.json')}")
    
    logger.info("=" * 80 + "\n")
    
    return training_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


