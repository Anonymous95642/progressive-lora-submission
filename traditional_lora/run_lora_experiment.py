#!/usr/bin/env python3
"""
LoRA微调实验运行脚本
专为48GB显存环境优化的LLaVA + COCO2017 LoRA微调实验

"""

import os
# 设置tokenizers并行化环境变量，避免fork警告
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lora_config import LoRAConfigManager, get_lora_config
from model_loader import create_lora_model_loader
from coco_trainer import COCOTrainer, COCOTrainingConfig
from coco_evaluator import COCOCaptionEvaluator, COCOEvaluationConfig
from coco_test_predictor import COCOTestPredictor, COCOTestPredictorConfig

# 配置日志（带完整日期时间和毫秒级时间戳 YYYY-MM-DD HH:MM:SS,mmm）
class MillisecondFormatter(logging.Formatter):
    """自定义格式化器，显示 年-月-日 时:分:秒,毫秒 格式"""
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            s = time.strftime("%Y-%m-%d %H:%M:%S", ct)
        s = f"{s},{int(record.msecs):03d}"
        return s

# 设置日志handlers（文件+控制台）
formatter = MillisecondFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('traditional_lora_experiment.log', encoding='utf-8')
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger(__name__)

class LoRAExperimentRunner:
    """LoRA微调实验运行器"""
    
    def __init__(self, config_name: str = "default_48gb"):
        """
        初始化实验运行器
        
        Args:
            config_name: LoRA配置名称
        """
        self.config_name = config_name
        self.lora_config = get_lora_config(config_name)
        self.experiment_dir = f"./lora_experiments/{config_name}"
        
        # 加载完整配置文件（包含训练参数）
        self.full_config = self._load_full_config(config_name)
        
        # 创建实验目录
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        logger.info(f"LoRA实验初始化完成，配置: {config_name}")
        logger.info(f"实验目录: {self.experiment_dir}")
    
    def _load_full_config(self, config_name: str) -> Dict:
        """加载完整配置文件（包含训练参数）"""
        try:
            # 尝试从文件加载
            config_file = f"./configs/{config_name}.json"
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    full_config = json.load(f)
                logger.info(f"从文件加载完整配置: {config_file}")
                return full_config
            else:
                logger.info(f"配置文件不存在: {config_file}，使用默认配置")
                return {}
        except Exception as e:
            logger.warning(f"加载完整配置失败: {str(e)}，使用默认配置")
            return {}
    
    def run_training_experiment(self, 
                              model_path: str = "/root/autodl-tmp/llava-1.5-7b",
                              coco_data_root: str = "/root/autodl-tmp/COCO2017",
                              num_epochs: int = 3,
                              max_train_samples: int = None,
                              max_val_samples: int = None,
                              dataset: str = 'coco',  # 'coco', 'flickr30k', or 'coco_karpathy'
                              batch_size: int = None,
                              learning_rate: float = None,
                              warmup_ratio: float = None) -> bool:
        """
        运行LoRA训练实验
        
        Args:
            model_path: 模型路径
            coco_data_root: 数据集路径（支持COCO和Flickr30K）
            num_epochs: 训练轮数
            max_train_samples: 最大训练样本数
            max_val_samples: 最大验证样本数
            dataset: 数据集类型 ('coco' 或 'flickr30k')
            batch_size: 批次大小（可选，覆盖配置文件）
            learning_rate: 学习率（可选，覆盖配置文件）
            warmup_ratio: 预热比例（可选，覆盖配置文件）
            
        Returns:
            bool: 是否成功
        """
        try:
            logger.info(f"开始Traditional LoRA训练实验 (dataset={dataset})...")
            logger.info("=" * 80)
            logger.info("📚 训练方法：Traditional LoRA（固定秩，随机样本顺序）")
            logger.info("=" * 80)
            
            # 创建训练配置
            training_config = COCOTrainingConfig(lora_config_name=self.config_name)
            training_config.model_path = model_path
            training_config.coco_data_root = coco_data_root
            training_config.num_epochs = num_epochs
            training_config.output_dir = os.path.join(self.experiment_dir, "training_output")
            training_config.checkpoint_dir = os.path.join(training_config.output_dir, "checkpoints")
            training_config.lora_adapters_dir = os.path.join(training_config.output_dir, "lora_adapters")
            
            # 从配置文件或命令行参数设置样本限制
            training_config.max_train_samples = (
                max_train_samples or 
                self.full_config.get('max_train_samples') or 
                training_config.max_train_samples
            )
            training_config.max_val_samples = (
                max_val_samples or 
                self.full_config.get('max_val_samples') or 
                training_config.max_val_samples
            )
            
            # 命令行参数覆盖配置文件（如果提供）
            if batch_size is not None:
                training_config.batch_size = batch_size
                logger.info(f"✅ 使用命令行batch_size: {batch_size}")
            if learning_rate is not None:
                training_config.learning_rate = learning_rate
                logger.info(f"✅ 使用命令行learning_rate: {learning_rate}")
            if warmup_ratio is not None:
                training_config.warmup_ratio = warmup_ratio
                logger.info(f"✅ 使用命令行warmup_ratio: {warmup_ratio}")
            
            # 创建训练器
            trainer = COCOTrainer(training_config)
            
            # 运行训练
            success = trainer.train()
            
            if success:
                logger.info("LoRA训练实验完成")
                
                # 保存实验配置
                self._save_experiment_config(training_config)
                
                return True
            else:
                logger.error("LoRA训练实验失败")
                return False
                
        except Exception as e:
            logger.error(f"LoRA训练实验异常: {str(e)}")
            return False
    
    def run_evaluation_experiment(self, 
                                adapter_path: str,
                                model_path: str = "/root/autodl-tmp/llava-1.5-7b",
                                coco_data_root: str = "/root/autodl-tmp/COCO2017") -> bool:
        """
        运行LoRA评估实验（已废弃，保留为兼容接口）
        
        注意：在新的训练流程中，验证集评估已集成到训练过程中。
        此方法保留仅用于向后兼容。
        
        Args:
            adapter_path: LoRA适配器路径
            model_path: 基础模型路径
            coco_data_root: COCO数据集路径
            
        Returns:
            bool: 是否成功
        """
        try:
            logger.warning("⚠️  run_evaluation_experiment已废弃！")
            logger.warning("验证集评估已集成到训练过程中，无需单独调用。")
            logger.info("开始LoRA评估实验（兼容模式）...")
            
            # 加载LoRA模型
            model_loader = create_lora_model_loader(
                model_path=model_path,
                lora_config_name=self.config_name,
                adapter_path=adapter_path
            )
            
            # 创建评估配置
            eval_config = COCOEvaluationConfig()
            eval_config.coco_data_root = coco_data_root
            eval_config.output_dir = os.path.join(self.experiment_dir, "evaluation_output")
            
            # 从配置文件设置评估样本限制
            eval_config.max_eval_samples = (
                self.full_config.get('max_val_samples') or 
                eval_config.max_eval_samples
            )
            
            # 创建评估器
            evaluator = COCOCaptionEvaluator(eval_config)
            
            # 运行评估
            success = evaluator.evaluate_model(model_loader)
            
            if success:
                logger.info("LoRA评估实验完成")
                return True
            else:
                logger.error("LoRA评估实验失败")
                return False
                
        except Exception as e:
            logger.error(f"LoRA评估实验异常: {str(e)}")
            return False
    
    def run_test_prediction_experiment(self,
                                     adapter_path: str,
                                     model_path: str = "/root/autodl-tmp/llava-1.5-7b",
                                     coco_data_root: str = "/root/autodl-tmp/COCO2017",
                                     max_test_samples: int = None) -> bool:
        """
        运行LoRA测试预测实验
        
        Args:
            adapter_path: LoRA适配器路径
            model_path: 基础模型路径
            coco_data_root: COCO数据集路径
            max_test_samples: 最大测试样本数
            
        Returns:
            bool: 是否成功
        """
        try:
            logger.info("开始LoRA测试预测实验...")
            
            # 加载LoRA模型
            model_loader = create_lora_model_loader(
                model_path=model_path,
                lora_config_name=self.config_name,
                adapter_path=adapter_path
            )
            
            # 创建测试配置
            test_config = COCOTestPredictorConfig()
            test_config.coco_data_root = coco_data_root
            test_config.output_dir = os.path.join(self.experiment_dir, "test_predictions")
            test_config.team_name = f"LLaVA-LoRA-{self.config_name}"
            
            # 从配置文件或命令行参数设置测试样本限制
            test_config.max_test_samples = (
                max_test_samples or 
                self.full_config.get('max_test_samples') or 
                test_config.max_test_samples
            )
            
            # 创建测试预测器
            predictor = COCOTestPredictor(test_config)
            
            # 运行测试预测
            success = predictor.predict_test_set(model_loader)
            
            if success:
                logger.info("LoRA测试预测实验完成")
                return True
            else:
                logger.error("LoRA测试预测实验失败")
                return False
                
        except Exception as e:
            logger.error(f"LoRA测试预测实验异常: {str(e)}")
            return False
    
    def run_full_experiment(self,
                          model_path: str = "/root/autodl-tmp/llava-1.5-7b",
                          coco_data_root: str = "/root/autodl-tmp/COCO2017",
                          num_epochs: int = 3,
                          max_train_samples: int = None,
                          max_val_samples: int = None,
                          max_test_samples: int = None,
                          dataset: str = 'coco',
                          batch_size: int = None,
                          learning_rate: float = None,
                          warmup_ratio: float = None,
                          skip_training: bool = False,
                          skip_test_prediction: bool = False) -> bool:
        """
        运行完整的LoRA实验流水线
        
        流程设计（与Progressive LoRA严格一致）：
        阶段1: 训练（包含验证集评估）
          ├── 训练循环
          └── 每个epoch验证（选择best_model）
        
        阶段2: 测试预测（可选）
          └── 在测试集上预测/评估
        
        注意：验证集评估已集成到训练过程中，无需单独指定！
        
        Args:
            model_path: 模型路径
            coco_data_root: COCO数据集路径
            num_epochs: 训练轮数
            max_train_samples: 最大训练样本数
            max_val_samples: 最大验证样本数
            max_test_samples: 最大测试样本数
            dataset: 数据集类型 ('coco' 或 'flickr30k')
            batch_size: 批次大小（覆盖配置文件）
            learning_rate: 学习率（覆盖配置文件）
            warmup_ratio: 预热比例（覆盖配置文件）
            skip_training: 跳过训练
            skip_test_prediction: 跳过测试预测
            
        Returns:
            bool: 是否成功
        """
        try:
            logger.info("=" * 80)
            logger.info("开始Traditional LoRA实验流水线...")
            logger.info("流程设计与Progressive LoRA严格一致：")
            logger.info("  阶段1: 训练（包含验证集评估）")
            logger.info("  阶段2: 测试预测（可选）")
            logger.info("=" * 80)
            
            # ========== 阶段1: 训练（包含验证集评估） ==========
            if not skip_training:
                logger.info("\n" + "=" * 80)
                logger.info("🎯 阶段1/2: Traditional LoRA训练（包含验证集评估）")
                logger.info("=" * 80)
                logger.info("💡 训练过程中会定期在验证集上评估Loss和标准指标（BLEU/CIDEr/ROUGE等）")
                logger.info("=" * 80 + "\n")
                
                success = self.run_training_experiment(
                    model_path=model_path,
                    coco_data_root=coco_data_root,
                    num_epochs=num_epochs,
                    max_train_samples=max_train_samples,
                    max_val_samples=max_val_samples,
                    dataset=dataset,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    warmup_ratio=warmup_ratio
                )
                if not success:
                    logger.error("❌ 训练阶段失败")
                    return False
                
                logger.info("\n" + "=" * 80)
                logger.info("🎉 训练阶段成功完成！")
                logger.info("=" * 80)
            
            # 寻找最佳模型适配器
            best_adapter_path = self._find_best_adapter()
            if not best_adapter_path:
                logger.error("未找到训练好的LoRA适配器")
                return False
            
            logger.info(f"\n✅ 训练阶段已完成，验证集评估已在训练过程中完成")
            logger.info(f"📊 最佳模型位于: {best_adapter_path}")
            logger.info(f"💡 提示: 验证集评估指标已保存在训练历史文件中")
            
            # ========== 阶段2: 测试预测 ==========
            test_prediction_success = True
            if not skip_test_prediction:
                logger.info("\n" + "=" * 80)
                logger.info("🎯 阶段2/2: 测试集预测与评估")
                logger.info("=" * 80 + "\n")
                
                try:
                    test_prediction_success = self.run_test_prediction_experiment(
                        adapter_path=best_adapter_path,
                        model_path=model_path,
                        coco_data_root=coco_data_root,
                        max_test_samples=max_test_samples
                    )
                    if not test_prediction_success:
                        logger.warning("⚠️ 测试预测阶段失败")
                    else:
                        logger.info("✅ 测试预测阶段成功完成")
                except Exception as e:
                    logger.warning(f"⚠️ 测试预测阶段异常: {str(e)}")
                    test_prediction_success = False
            else:
                logger.info("\n⏭️  跳过测试预测阶段")
            
            # ========== 总结 ==========
            logger.info("\n" + "=" * 80)
            logger.info("🏁 完整流程执行总结")
            logger.info("=" * 80)
            logger.info(f"✅ 阶段1 - 训练（包含验证集评估）: {'成功' if not skip_training else '跳过'}")
            
            # 测试预测阶段状态
            if skip_test_prediction:
                test_status = "跳过"
                test_icon = "⏭️"
            elif test_prediction_success:
                test_status = "成功"
                test_icon = "✅"
            else:
                test_status = "失败"
                test_icon = "❌"
            logger.info(f"{test_icon} 阶段2 - 测试预测: {test_status}")
            logger.info("=" * 80)
            
            # ========== 显示评估指标 ==========
            # 读取训练历史中的验证集指标
            history_file = os.path.join(self.experiment_dir, "training_output", "training_history.json")
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
            
            # 读取测试集指标
            test_metrics = None
            if test_prediction_success and not skip_test_prediction:
                test_metrics_file = os.path.join(self.experiment_dir, "test_predictions", "test_metrics.json")
                if os.path.exists(test_metrics_file):
                    try:
                        with open(test_metrics_file, 'r', encoding='utf-8') as f:
                            test_metrics = json.load(f)
                        logger.info("✅ 成功读取测试集评估指标")
                    except Exception as e:
                        logger.warning(f"无法读取测试集指标: {e}")
            
            display_metrics = None
            display_split = None
            
            if dataset in ['flickr30k', 'vizwiz'] and test_metrics:
                # Flickr30K / VizWiz: 显示测试集指标作为最终结果
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
                if dataset in ['flickr30k', 'vizwiz']:
                    logger.info("📊 验证集评估指标（训练时最佳，用于模型选择）")
                else:
                    logger.info("📊 验证集评估指标（训练时最佳）")
                logger.info("=" * 80)
            
            # 统一显示指标
            if display_metrics:
                if display_split == "TEST":
                    logger.info(f"\n【{display_split}数据集评估结果】⭐ (最终结果)")
                else:
                    logger.info(f"\n【{display_split}数据集评估结果】")
                
                # 处理不同的指标存储格式
                # test_metrics.json 格式: {"metrics": {...}}
                # training_history.json 格式: {...}
                if 'metrics' in display_metrics:
                    metrics_data = display_metrics['metrics']
                else:
                    metrics_data = display_metrics
                
                logger.info("  📈 BLEU指标:")
                logger.info(f"    - BLEU-1: {metrics_data.get('Bleu_1', 0):.4f}")
                logger.info(f"    - BLEU-2: {metrics_data.get('Bleu_2', 0):.4f}")
                logger.info(f"    - BLEU-3: {metrics_data.get('Bleu_3', 0):.4f}")
                logger.info(f"    - BLEU-4: {metrics_data.get('Bleu_4', 0):.4f}")
                logger.info("\n  📈 其他指标:")
                logger.info(f"    - ROUGE-L: {metrics_data.get('ROUGE_L', 0):.4f}")
                logger.info(f"    - CIDEr: {metrics_data.get('CIDEr', 0):.4f} ⭐ (最重要)")
                logger.info(f"    - METEOR: {metrics_data.get('METEOR', 0):.4f}")
                
                if display_split == "TEST":
                    logger.info("\n  💡 说明: 这是测试集的最终评估结果")
                    logger.info("         验证集指标仅用于训练过程中的模型选择")
                else:
                    logger.info("\n  💡 说明: 这是训练过程中的最佳验证集指标")
                    logger.info("         用于模型选择，测试集结果请查看test_predictions/目录")
                
                logger.info("\n" + "=" * 80)
            
            # ========== 显示所有输出文件位置 ==========
            logger.info("\n📂 所有输出文件位置:")
            logger.info(f"主输出目录: {os.path.abspath(self.experiment_dir)}")
            logger.info(f"  ├── training_output/")
            logger.info(f"  │   ├── lora_adapters/          # LoRA适配器")
            logger.info(f"  │   ├── checkpoints/            # 训练检查点")
            logger.info(f"  │   └── training_history.json   # 训练历史（包含验证集评估指标）")
            
            if not skip_test_prediction:
                test_pred_dir = os.path.join(self.experiment_dir, "test_predictions")
                if os.path.exists(test_pred_dir):
                    logger.info(f"\n测试预测: {os.path.abspath(test_pred_dir)}")
                    test_files = sorted(os.listdir(test_pred_dir))
                    for file in test_files:
                        file_path = os.path.join(test_pred_dir, file)
                        if os.path.isfile(file_path):
                            if 'test_predictions_coco' in file and file.endswith('.json'):
                                logger.info(f"  ├── {file}  ✅ ⭐ (主要预测文件)")
                            elif file == 'test_metrics.json':
                                logger.info(f"  ├── {file}  ✅")
                            elif file.endswith('.json'):
                                logger.info(f"  ├── {file}  ✅")
                            else:
                                logger.info(f"  ├── {file}")
            
            logger.info("\n💡 使用提示:")
            logger.info("  1. 查看训练历史和验证集指标:")
            logger.info(f"     cat {os.path.join(os.path.abspath(self.experiment_dir), 'training_output', 'training_history.json')}")
            
            if dataset == 'flickr30k' and not skip_test_prediction:
                logger.info("  2. Flickr30K测试集结果:")
                logger.info(f"     评估文件: {os.path.join(os.path.abspath(self.experiment_dir), 'test_predictions', 'test_evaluation_results.json')}")
            elif dataset == 'vizwiz' and not skip_test_prediction:
                logger.info("  2. VizWiz-Captions 测试集结果（如有标注）:")
                logger.info(f"     评估文件: {os.path.join(os.path.abspath(self.experiment_dir), 'test_predictions', 'test_evaluation_results.json')}")
            elif not skip_test_prediction:
                logger.info("  2. 测试集预测结果:")
                logger.info(f"     查看目录: {os.path.join(os.path.abspath(self.experiment_dir), 'test_predictions')}")
            
            logger.info("=" * 80)
            
            logger.info("\n完整LoRA实验流水线完成")
            return True
            
        except Exception as e:
            logger.error(f"完整LoRA实验流水线异常: {str(e)}")
            return False
    
    def _find_best_adapter(self) -> str:
        """寻找最佳的LoRA适配器"""
        try:
            adapters_dir = os.path.join(self.experiment_dir, "training_output", "lora_adapters")
            
            # 优先寻找best_model
            best_model_path = os.path.join(adapters_dir, "best_model")
            if os.path.exists(best_model_path):
                return best_model_path
            
            # 寻找最新的checkpoint
            if os.path.exists(adapters_dir):
                checkpoints = [d for d in os.listdir(adapters_dir) 
                             if d.startswith("checkpoint-step-")]
                if checkpoints:
                    # 按步数排序，取最新的
                    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
                    return os.path.join(adapters_dir, checkpoints[-1])
            
            return None
            
        except Exception as e:
            logger.error(f"寻找最佳适配器失败: {str(e)}")
            return None
    
    def _save_experiment_config(self, training_config):
        """保存实验配置"""
        try:
            config_data = {
                "experiment_name": self.config_name,
                "lora_config": self.lora_config.to_dict(),
                "training_config": {
                    "model_path": training_config.model_path,
                    "coco_data_root": training_config.coco_data_root,
                    "num_epochs": training_config.num_epochs,
                    "batch_size": training_config.batch_size,
                    "learning_rate": training_config.learning_rate,
                    "max_train_samples": training_config.max_train_samples,
                    "max_val_samples": training_config.max_val_samples
                }
            }
            
            config_file = os.path.join(self.experiment_dir, "experiment_config.json")
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"实验配置已保存: {config_file}")
            
        except Exception as e:
            logger.error(f"保存实验配置失败: {str(e)}")

def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(description="Traditional LoRA微调实验运行器")
    
    # 基础参数（与Progressive LoRA保持一致的命名风格）
    parser.add_argument("--config_name", type=str, default="traditional_lora_r128", 
                       help="LoRA配置名称（默认：traditional_lora_r128）")
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/llava-1.5-7b",
                       help="模型路径")
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/COCO2017",
                       help="数据集根目录（支持COCO和Flickr30K）")
    parser.add_argument("--dataset", type=str, default="coco",
                       choices=['coco', 'flickr8k', 'flickr30k', 'vizwiz'],
                       help="数据集类型：coco、flickr8k、flickr30k 或 vizwiz（默认：coco）")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出目录（默认：./traditional_lora_output）")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=9, 
                       help="训练轮数（默认9，与Progressive LoRA的3+3+3对应）")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="批次大小（覆盖配置文件，默认使用配置文件值）")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="学习率（覆盖配置文件，默认使用配置文件值）")
    parser.add_argument("--warmup_ratio", type=float, default=None,
                       help="预热比例（覆盖配置文件，默认使用配置文件值）")
    parser.add_argument("--max_train_samples", type=int, default=None,
                       help="最大训练样本数")
    parser.add_argument("--max_val_samples", type=int, default=None,
                       help="最大验证样本数（用于训练过程中的验证）")
    parser.add_argument("--max_test_samples", type=int, default=None,
                       help="最大测试样本数")
    
    # 执行选项
    parser.add_argument("--mode", type=str, default="full",
                       choices=["full", "training", "test"],
                       help="运行模式: full=训练+测试（推荐）, training=仅训练, test=仅测试")
    parser.add_argument("--adapter_path", type=str, default=None,
                       help="LoRA适配器路径（test模式需要）")
    
    # 跳过选项
    parser.add_argument("--skip_training", action="store_true", help="跳过训练阶段")
    parser.add_argument("--skip_test_prediction", action="store_true", help="跳过测试预测阶段")
    
    # 配置管理
    parser.add_argument("--list_configs", action="store_true", help="列出所有可用配置")
    parser.add_argument("--create_config_templates", action="store_true", help="创建配置模板")
    
    args = parser.parse_args()
    
    # 配置管理操作
    if args.list_configs:
        manager = LoRAConfigManager()
        configs = manager.list_available_configs()
        print("可用的LoRA配置:")
        for config in configs:
            print(f"  - {config}")
        return
    
    if args.create_config_templates:
        manager = LoRAConfigManager()
        manager.save_template_configs()
        print("配置模板已创建")
        return
    
    # 数据路径配置
    data_path = args.data_path
    
    # 输出目录配置
    if args.output_dir:
        experiment_dir = args.output_dir
    else:
        # 根据数据集类型设置默认输出目录
        if args.dataset == 'flickr30k':
            experiment_dir = f"./traditional_lora_{args.dataset}_output"
        else:
            experiment_dir = "./traditional_lora_output"
    
    # 创建实验运行器
    runner = LoRAExperimentRunner(args.config_name)
    runner.experiment_dir = experiment_dir  # 覆盖默认实验目录
    
    logger.info("=" * 80)
    logger.info("🔧 Traditional LoRA实验配置")
    logger.info("=" * 80)
    logger.info(f"训练方法: Traditional LoRA（固定秩，随机样本顺序）")
    logger.info(f"配置名称: {args.config_name}")
    logger.info(f"数据集类型: {args.dataset.upper()}")
    logger.info(f"数据路径: {data_path}")
    logger.info(f"输出目录: {experiment_dir}")
    logger.info(f"训练轮数: {args.epochs}")
    if args.batch_size:
        logger.info(f"批次大小: {args.batch_size} (命令行覆盖)")
    if args.learning_rate:
        logger.info(f"学习率: {args.learning_rate} (命令行覆盖)")
    if args.warmup_ratio:
        logger.info(f"预热比例: {args.warmup_ratio} (命令行覆盖)")
    logger.info("=" * 80)
    
    # 根据模式运行实验
    if args.mode == "full":
        success = runner.run_full_experiment(
            model_path=args.model_path,
            coco_data_root=data_path,
            num_epochs=args.epochs,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            max_test_samples=args.max_test_samples,
            dataset=args.dataset,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            skip_training=args.skip_training,
            skip_test_prediction=args.skip_test_prediction
        )
    elif args.mode == "training":
        success = runner.run_training_experiment(
            model_path=args.model_path,
            coco_data_root=data_path,
            num_epochs=args.epochs,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            dataset=args.dataset,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio
        )
    elif args.mode == "test":
        if not args.adapter_path:
            print("测试模式需要指定 --adapter_path")
            return
        success = runner.run_test_prediction_experiment(
            adapter_path=args.adapter_path,
            model_path=args.model_path,
            coco_data_root=data_path,
            max_test_samples=args.max_test_samples
        )
    
    exit(0 if success else 1)

if __name__ == "__main__":
    main()

