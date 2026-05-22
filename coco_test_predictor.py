"""
=============================================================================
COCO 测试集预测器 (Test Predictor)
=============================================================================

本模块负责在测试集上生成预测结果，并生成官方提交格式的文件。

【核心功能】
1. 测试集推理：对测试集进行批量推理预测
2. 格式转换：生成COCO官方评测所需的JSON格式
3. 结果保存：保存预测结果、原始输出、元数据
4. Flickr30K支持：测试集有标注时，额外计算评估指标

【COCO vs Flickr30K 测试集差异】
- COCO测试集：无ground truth，需提交到官方服务器评测
  └── 输出文件：test_predictions_coco_*.json (官方格式)
  
- Flickr30K测试集：有ground truth，可直接评估
  └── 额外输出：test_evaluation_results.json (包含BLEU/CIDEr等指标)

【输出文件】
1. test_predictions_coco_*.json (主要)
   - COCO官方提交格式：[{"image_id": ..., "caption": ...}, ...]
   - 可直接提交到: https://competitions.codalab.org/competitions/3221

2. test_predictions_raw_*.json
   - 原始预测结果，包含完整信息

3. test_prediction_metadata_*.json
   - 预测元数据（时间戳、模型信息、配置等）

4. test_evaluation_results.json (仅Flickr30K)
   - 测试集评估指标（BLEU, CIDEr, ROUGE等）

【使用示例】
```python
# 在run_progressive_training.py中自动调用
# 或手动运行：
from coco_test_predictor import COCOTestPredictor, COCOTestPredictorConfig

config = COCOTestPredictorConfig()
config.coco_data_root = "/path/to/COCO2017"
config.max_test_samples = None  # 预测全部测试集

predictor = COCOTestPredictor(config)
predictor.predict_test_set(model_loader)
```

【评估指标说明】
- BLEU-1/2/3/4: n-gram精确匹配（值越高越好）
- CIDEr: Consensus-based Image Description Evaluation（最重要指标）⭐
- ROUGE-L: 最长公共子序列
- METEOR: 基于词义的匹配

"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
from datetime import datetime

# 自定义JSON编码器，处理numpy类型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

from model_loader import LLaVAModelLoader
from coco_dataset import COCODatasetConfig, COCOCaptionDataset, COCODataLoader

# 配置日志
logger = logging.getLogger(__name__)

class COCOTestPredictorConfig:
    """COCO测试预测配置类"""
    
    def __init__(self):
        """初始化测试预测配置"""
        # 基础路径
        self.model_path = "/root/autodl-tmp/llava-1.5-7b"
        self.coco_data_root = "/root/autodl-tmp/COCO2017"
        self.output_dir = os.path.join(os.getcwd(), "coco_test_predictions")  # 确保在当前项目目录
        self.submission_dir = os.path.join(self.output_dir, "submissions")
        
        # 预测参数
        self.batch_size = 8
        self.max_new_tokens = 50  # 最优值：覆盖模型实际生成长度（23±3词）+ 系统开销，防止截断
        self.temperature = 0.7
        self.num_beams = 3  # beam search提高质量 (model_loader默认值)
        self.length_penalty = 1.0  # 长度惩罚中性 (model_loader默认值)
        self.repetition_penalty = 1.2  # 减少重复 (model_loader默认值)
        self.num_workers = 4
        
        # 模型加载选项
        self.load_in_8bit = False
        self.load_in_4bit = False
        
        # LoRA适配器选项
        self.adapter_path = None  # LoRA适配器路径，None表示不使用适配器
        
        # 预测选项
        self.max_test_samples = None  # None表示预测所有测试样本
        self.save_intermediate_results = True
        self.create_submission_file = True
        
        # 提交文件配置
        self.team_name = "LLaVA-COCO-Team"
        self.method_description = "LLaVA-1.5-7B fine-tuned on COCO2017 captions"
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)
        
        logger.info("测试预测配置初始化完成")

class COCOTestPredictor:
    """COCO测试集预测器"""
    
    def __init__(self, config: COCOTestPredictorConfig):
        """
        初始化COCO测试预测器
        
        Args:
            config: 测试预测配置对象
        """
        self.config = config
        self.model_loader = None
        self.coco_config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 预测结果存储
        self.test_predictions = {}
        self.prediction_metadata = {}
        
        logger.info(f"测试预测器初始化完成，使用设备: {self.device}")
    
    def setup_model(self) -> bool:
        """设置预测模型"""
        try:
            logger.info("正在加载LLaVA模型用于测试预测...")
            self.model_loader = LLaVAModelLoader(self.config.model_path)
            
            success = self.model_loader.load_model(
                load_in_8bit=self.config.load_in_8bit,
                load_in_4bit=self.config.load_in_4bit
            )
            
            if not success:
                logger.error("模型加载失败")
                return False
            
            # 如果指定了适配器路径，加载LoRA适配器
            if self.config.adapter_path:
                logger.info(f"正在加载LoRA适配器: {self.config.adapter_path}")
                if self.model_loader.load_lora_adapter(self.config.adapter_path):
                    logger.info("LoRA适配器加载成功")
                else:
                    logger.warning("LoRA适配器加载失败，将使用基础模型")
            
            # 设置模型为评估模式
            self.model_loader.model.eval()
            
            logger.info("测试预测模型设置完成")
            return True
            
        except Exception as e:
            logger.error(f"模型设置失败: {str(e)}")
            return False
    
    def setup_data(self) -> bool:
        """设置数据配置"""
        try:
            logger.info("正在设置测试数据配置...")
            
            # 自动检测数据集类型（支持 COCO / Flickr8k / Flickr30K / COCO Karpathy / VizWiz-Captions）
            data_root = self.config.coco_data_root
            lower_root = data_root.lower()
            if 'flickr8k' in lower_root:
                logger.info("检测到Flickr8k数据集，正在配置测试集...")
                from flickr8k_adapter import Flickr8kAdapter
                self.coco_config = Flickr8kAdapter(data_root)

                if not self.coco_config.validate_paths():
                    logger.error("Flickr8k 数据集路径验证失败")
                    return False

                logger.info("Flickr8k 测试数据配置设置完成")
                return True
            elif 'flickr30k' in lower_root or 'flickr' in lower_root:
                logger.info("检测到Flickr30K数据集，正在配置测试集...")
                from flickr30k_adapter import Flickr30KAdapter
                self.coco_config = Flickr30KAdapter(data_root)

                if not self.coco_config.validate_paths():
                    logger.error("Flickr30K数据集路径验证失败")
                    return False

                logger.info("Flickr30K测试数据配置设置完成")
                return True
            elif 'vizwiz' in lower_root:
                logger.info("检测到VizWiz-Captions数据集，正在配置测试集...")
                from vizwiz_adapter import VizWizCaptionAdapter
                self.coco_config = VizWizCaptionAdapter(data_root)
                
                # 验证路径
                if not self.coco_config.validate_paths():
                    logger.error("VizWiz-Captions数据集路径验证失败")
                    return False
                
                logger.info("VizWiz-Captions测试数据配置设置完成")
                return True
            elif 'karpathy' in lower_root or 'coco2014' in lower_root:
                logger.info("检测到COCO Karpathy数据集，正在配置测试集...")
                
                # 使用COCO Karpathy适配器
                from coco_karpathy_adapter import COCOKarpathyAdapter
                self.coco_config = COCOKarpathyAdapter(data_root)
                
                # 验证路径
                if not self.coco_config.validate_paths():
                    logger.error("COCO Karpathy数据集路径验证失败")
                    return False
                
                logger.info("COCO Karpathy测试数据配置设置完成")
                return True
            else:
                # 使用COCO数据集
                self.coco_config = COCODatasetConfig(data_root)
                
                # 检查测试集路径
                if not os.path.exists(self.coco_config.test_image_dir):
                    logger.error(f"测试集图像目录不存在: {self.coco_config.test_image_dir}")
                    return False
                
                # 测试集信息文件是可选的，如果不存在会给出警告但不阻止运行
                if not os.path.exists(self.coco_config.test_info_file):
                    logger.warning(f"测试集信息文件不存在: {self.coco_config.test_info_file}")
                    logger.info("这是正常情况！测试集通常不提供信息文件")
                    logger.info("将直接从图像目录加载测试图像，功能完全正常")
                
                logger.info("测试数据配置设置完成")
                return True
            
        except Exception as e:
            logger.error(f"数据设置失败: {str(e)}")
            return False
    
    def generate_test_predictions(self) -> Dict[int, str]:
        """
        生成测试集预测结果
        
        Returns:
            Dict[int, str]: 图像ID到预测描述的映射
        """
        logger.info("开始生成测试集预测结果...")
        
        # 所有数据集都使用test split
        split = "test"
        data_root = self.config.coco_data_root
        lower_root = data_root.lower()
        if 'flickr8k' in lower_root:
            logger.info("Flickr8k数据集：使用测试集进行预测")
        elif 'flickr30k' in lower_root or 'flickr' in lower_root:
            logger.info("Flickr30K数据集：使用测试集进行预测")
        elif 'vizwiz' in lower_root:
            logger.info("VizWiz-Captions数据集：使用测试集进行预测")
        elif 'karpathy' in lower_root or 'coco2014' in lower_root:
            logger.info("COCO Karpathy数据集：使用测试集进行预测")
        else:
            logger.info("COCO数据集：使用测试集进行预测")
        
        # 创建数据加载器
        data_loader = COCODataLoader(self.coco_config)
        dataloader = data_loader.create_dataloader(
            split=split,
            batch_size=1,  # 逐个处理
            shuffle=False,
            num_workers=0,  # 避免多进程问题
            max_samples=self.config.max_test_samples
        )
        
        predictions = {}
        prediction_times = {}
        
        # 获取总样本数
        total_samples = len(dataloader)
        
        # 创建进度条
        pbar = tqdm(dataloader, desc="生成测试预测", total=total_samples)
        
        for idx, batch in enumerate(pbar):
            try:
                # 获取样本（批次大小为1）
                image_id = batch['image_ids'][0]
                image = batch['images'][0]
                
                # 记录开始时间
                start_time = datetime.now()
                
                # 生成描述
                description = self.model_loader.describe_image(
                    image=image,
                    prompt="Describe this image in detail.",
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    num_beams=self.config.num_beams,
                    length_penalty=self.config.length_penalty,
                    repetition_penalty=self.config.repetition_penalty
                )
                
                # 记录结束时间
                end_time = datetime.now()
                prediction_time = (end_time - start_time).total_seconds()
                
                predictions[image_id] = description
                prediction_times[image_id] = prediction_time
                
                # 更新进度条
                avg_time = sum(prediction_times.values()) / len(prediction_times)
                pbar.set_postfix({
                    'completed': f'{idx+1}/{total_samples}',
                    'avg_time': f'{avg_time:.2f}s'
                })
                
                # 定期保存中间结果
                if self.config.save_intermediate_results and (idx + 1) % 1000 == 0:
                    self._save_intermediate_predictions(predictions, idx + 1)
                
            except Exception as e:
                image_id = batch.get('image_ids', [None])[0] if 'image_ids' in batch else 'unknown'
                logger.error(f"生成测试预测失败，图像ID {image_id}: {str(e)}")
                continue
        
        # 保存预测元数据
        self.prediction_metadata = {
            'total_predictions': len(predictions),
            'avg_prediction_time': float(np.mean(list(prediction_times.values()))) if prediction_times else 0.0,
            'total_time': float(sum(prediction_times.values())),
            'model_config': {
                'model_path': self.config.model_path,
                'max_new_tokens': self.config.max_new_tokens,
                'temperature': self.config.temperature,
                'load_in_8bit': self.config.load_in_8bit,
                'load_in_4bit': self.config.load_in_4bit
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"测试集预测生成完成，共{len(predictions)}个预测结果")
        logger.info(f"平均预测时间: {self.prediction_metadata['avg_prediction_time']:.2f}秒")
        
        return predictions
    
    def _save_intermediate_predictions(self, predictions: Dict[int, str], count: int):
        """保存中间预测结果"""
        try:
            intermediate_file = os.path.join(
                self.config.output_dir,
                f"intermediate_predictions_{count}.json"
            )
            # 确保数据类型正确
            safe_predictions = {int(k): str(v) for k, v in predictions.items()}
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump(safe_predictions, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
            logger.info(f"中间结果已保存: {os.path.abspath(intermediate_file)}")
            
        except Exception as e:
            logger.error(f"保存中间结果失败: {str(e)}")
    
    def save_test_predictions(self, predictions: Dict[int, str]):
        """
        保存测试预测结果
        
        Args:
            predictions: 测试预测结果字典
        """
        try:
            # 确保输出目录存在
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存原始预测结果
            raw_pred_file = os.path.join(
                self.config.output_dir,
                f"test_predictions_raw_{timestamp}.json"
            )
            # 确保数据类型正确
            safe_predictions = {int(k): str(v) for k, v in predictions.items()}
            with open(raw_pred_file, 'w', encoding='utf-8') as f:
                json.dump(safe_predictions, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
            # 保存COCO格式预测结果
            coco_format_predictions = []
            for image_id, caption in predictions.items():
                coco_format_predictions.append({
                    "image_id": int(image_id),
                    "caption": caption
                })
            
            coco_pred_file = os.path.join(
                self.config.output_dir,
                f"test_predictions_coco_{timestamp}.json"
            )
            with open(coco_pred_file, 'w', encoding='utf-8') as f:
                json.dump(coco_format_predictions, f, indent=2, cls=NumpyEncoder)
            
            # 保存预测元数据
            metadata_file = os.path.join(
                self.config.output_dir,
                f"prediction_metadata_{timestamp}.json"
            )
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.prediction_metadata, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
            logger.info("=" * 80)
            logger.info("📁 测试预测文件已保存:")
            logger.info("=" * 80)
            logger.info(f"✅ [主要预测文件] 标准格式: {os.path.basename(coco_pred_file)}")
            logger.info(f"   完整路径: {os.path.abspath(coco_pred_file)}")
            logger.info("")
            logger.info(f"📄 [辅助文件] 原始格式: {os.path.basename(raw_pred_file)}")
            logger.info(f"📄 [辅助文件] 预测元数据: {os.path.basename(metadata_file)}")
            logger.info("=" * 80)
            logger.info(f"💡 提示: 主要预测文件是 '{os.path.basename(coco_pred_file)}'")
            logger.info("=" * 80)
            
            return coco_pred_file
            
        except Exception as e:
            logger.error(f"保存测试预测结果失败: {str(e)}")
            return None
    
    def create_submission_file(self, predictions: Dict[int, str]) -> str:
        """
        创建官方提交文件
        
        Args:
            predictions: 测试预测结果字典
            
        Returns:
            str: 提交文件路径
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 准备提交数据
            submission_data = []
            for image_id, caption in predictions.items():
                submission_data.append({
                    "image_id": int(image_id),
                    "caption": caption
                })
            
            # 创建提交文件
            submission_file = os.path.join(
                self.config.submission_dir,
                f"coco_test_submission_{timestamp}.json"
            )
            
            with open(submission_file, 'w', encoding='utf-8') as f:
                json.dump(submission_data, f, separators=(',', ':'), cls=NumpyEncoder)  # 紧凑格式
            
            # 创建提交信息文件
            submission_info = {
                "team_name": self.config.team_name,
                "method_description": self.config.method_description,
                "submission_file": os.path.basename(submission_file),
                "num_predictions": len(predictions),
                "model_details": {
                    "model_name": "LLaVA-1.5-7B",
                    "fine_tuned_on": "COCO2017",
                    "max_tokens": self.config.max_new_tokens,
                    "temperature": self.config.temperature,
                    "num_beams": self.config.num_beams,
                    "length_penalty": self.config.length_penalty,
                    "repetition_penalty": self.config.repetition_penalty
                },
                "submission_timestamp": datetime.now().isoformat(),
                "file_size_mb": os.path.getsize(submission_file) / (1024 * 1024)
            }
            
            info_file = os.path.join(
                self.config.submission_dir,
                f"submission_info_{timestamp}.json"
            )
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(submission_info, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
            logger.info(f"官方提交文件已创建:")
            logger.info(f"  - 提交文件: {submission_file}")
            logger.info(f"  - 信息文件: {info_file}")
            logger.info(f"  - 文件大小: {submission_info['file_size_mb']:.2f} MB")
            
            return submission_file
            
        except Exception as e:
            logger.error(f"创建提交文件失败: {str(e)}")
            return ""
    
    def compute_test_metrics(self, predictions: Dict[int, str]) -> Dict:
        """
        计算测试集评估指标（如果有标注）
        
        Args:
            predictions: 测试预测结果字典
            
        Returns:
            Dict: 评估指标结果（如果没有标注则返回空字典）
        """
        try:
            # 检查是否有测试集标注文件
            test_ann_file = self.coco_config.test_captions_file
            
            if not os.path.exists(test_ann_file):
                logger.info("⚠️  测试集没有标注文件，跳过指标计算")
                logger.info("    说明：COCO测试集标注未公开，需要提交到官方评估服务器")
                return {}
            
            logger.info("✅ 测试集有标注文件，开始计算评估指标...")
            
            # 导入评估工具
            try:
                from pycocoevalcap.cider.cider import Cider
                from pycocoevalcap.bleu.bleu import Bleu
                from pycocoevalcap.rouge.rouge import Rouge
                try:
                    from pycocoevalcap.meteor.meteor import Meteor
                    meteor_available = True
                except ImportError:
                    meteor_available = False
                    logger.warning("METEOR工具不可用（需要Java环境）")
            except ImportError as e:
                logger.error(f"pycocoevalcap未安装: {e}")
                logger.info("安装命令: pip install pycocoevalcap")
                return {}
            
            # 加载标注
            from coco_dataset import COCOCaptionLoader
            coco_gt = COCOCaptionLoader(test_ann_file)
            
            # 准备真实标注数据
            gts = {}
            for img_id in predictions.keys():
                ann_ids = coco_gt.getAnnIds(imgIds=int(img_id))
                anns = coco_gt.loadAnns(ann_ids)
                captions = [ann['caption'].strip() for ann in anns if ann.get('caption', '').strip()]
                if captions:
                    gts[int(img_id)] = captions
            
            # 准备预测数据
            res = {}
            for img_id, caption in predictions.items():
                img_id = int(img_id)
                if img_id in gts and caption and caption.strip():
                    res[img_id] = [caption.strip()]
            
            if not res:
                logger.error("没有有效的预测结果可用于评估")
                return {}
            
            logger.info(f"正在评估 {len(res)} 个测试样本...")
            
            # 计算评估指标
            metrics = {}
            
            # BLEU
            try:
                logger.info("计算 BLEU 指标...")
                bleu_scorer = Bleu(n=4)
                bleu_scores, _ = bleu_scorer.compute_score(gts, res)
                metrics['Bleu_1'] = float(bleu_scores[0])
                metrics['Bleu_2'] = float(bleu_scores[1])
                metrics['Bleu_3'] = float(bleu_scores[2])
                metrics['Bleu_4'] = float(bleu_scores[3])
                logger.info(f"✅ BLEU-4: {metrics['Bleu_4']:.4f}")
            except Exception as e:
                logger.warning(f"BLEU计算失败: {e}")
            
            # CIDEr
            try:
                logger.info("计算 CIDEr 指标...")
                cider_scorer = Cider()
                cider_score, _ = cider_scorer.compute_score(gts, res)
                metrics['CIDEr'] = float(cider_score)
                logger.info(f"✅ CIDEr: {metrics['CIDEr']:.4f}")
            except Exception as e:
                logger.warning(f"CIDEr计算失败: {e}")
            
            # ROUGE-L
            try:
                logger.info("计算 ROUGE-L 指标...")
                rouge_scorer = Rouge()
                rouge_score, _ = rouge_scorer.compute_score(gts, res)
                metrics['ROUGE_L'] = float(rouge_score)
                logger.info(f"✅ ROUGE-L: {metrics['ROUGE_L']:.4f}")
            except Exception as e:
                logger.warning(f"ROUGE-L计算失败: {e}")
            
            # METEOR（可选，需要Java环境）
            if meteor_available:
                try:
                    logger.info("计算 METEOR 指标...")
                    meteor_scorer = Meteor()
                    meteor_score, _ = meteor_scorer.compute_score(gts, res)
                    if isinstance(meteor_score, (int, float)):
                        metrics['METEOR'] = float(meteor_score)
                        logger.info(f"✅ METEOR: {metrics['METEOR']:.4f}")
                except Exception as e:
                    logger.warning(f"METEOR计算失败（需要Java环境）: {e}")
            
            logger.info(f"✅ 测试集评估指标计算完成！")
            return metrics
            
        except Exception as e:
            logger.error(f"测试集指标计算失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def analyze_predictions(self, predictions: Dict[int, str]) -> Dict:
        """
        分析预测结果
        
        Args:
            predictions: 预测结果字典
            
        Returns:
            Dict: 分析结果
        """
        try:
            logger.info("正在分析预测结果...")
            
            # 基本统计
            caption_lengths = [len(caption.split()) for caption in predictions.values()]
            
            # 词汇分析
            all_words = []
            for caption in predictions.values():
                all_words.extend(caption.lower().split())
            
            from collections import Counter
            word_counts = Counter(all_words)
            
            analysis = {
                'basic_stats': {
                    'total_predictions': len(predictions),
                    'avg_caption_length': float(np.mean(caption_lengths)),
                    'std_caption_length': float(np.std(caption_lengths)),
                    'min_caption_length': int(np.min(caption_lengths)),
                    'max_caption_length': int(np.max(caption_lengths)),
                    'median_caption_length': float(np.median(caption_lengths))
                },
                'vocabulary_stats': {
                    'total_words': len(all_words),
                    'unique_words': len(word_counts),
                    'vocabulary_diversity': float(len(word_counts) / len(all_words)),
                    'most_common_words': [(word, int(count)) for word, count in word_counts.most_common(20)]
                },
                'length_distribution': {
                    'length_1_5': sum(1 for l in caption_lengths if 1 <= l <= 5),
                    'length_6_10': sum(1 for l in caption_lengths if 6 <= l <= 10),
                    'length_11_15': sum(1 for l in caption_lengths if 11 <= l <= 15),
                    'length_16_20': sum(1 for l in caption_lengths if 16 <= l <= 20),
                    'length_20_plus': sum(1 for l in caption_lengths if l > 20)
                }
            }
            
            logger.info("预测结果分析完成")
            return analysis
            
        except Exception as e:
            logger.error(f"预测结果分析失败: {str(e)}")
            return {}
    
    def run_test_prediction(self) -> bool:
        """
        运行完整测试预测流程
        
        Returns:
            bool: 是否成功完成预测
        """
        try:
            logger.info("开始测试预测流程...")
            
            # 设置模型
            if not self.setup_model():
                return False
            
            # 设置数据
            if not self.setup_data():
                return False
            
            # 生成预测
            predictions = self.generate_test_predictions()
            
            if not predictions:
                logger.error("测试预测生成失败")
                return False
            
            # 保存预测结果
            pred_file = self.save_test_predictions(predictions)
            
            # 创建提交文件（仅COCO需要，Flickr30K和COCO Karpathy有测试集标注可直接评估）
            data_root = self.config.coco_data_root
            is_flickr = 'flickr30k' in data_root.lower() or 'flickr' in data_root.lower()
            is_karpathy = 'karpathy' in data_root.lower() or 'coco2014' in data_root.lower()
            
            if self.config.create_submission_file:
                if not is_flickr and not is_karpathy:
                    # COCO: 生成提交文件
                    submission_file = self.create_submission_file(predictions)
                    if submission_file:
                        logger.info(f"提交文件创建成功: {submission_file}")
                else:
                    # Flickr30K/COCO Karpathy: 跳过提交文件生成
                    dataset_name = "Flickr30K" if is_flickr else "COCO Karpathy"
                    logger.info(f"{dataset_name}数据集：测试集有标注，可通过评估阶段直接获取指标，无需生成提交文件")
            
            # 分析预测结果
            analysis = self.analyze_predictions(predictions)
            if analysis:
                self._save_analysis_report(analysis)
                self._print_analysis_summary(analysis)
            
            # 计算评估指标（如果测试集有标注）
            metrics = self.compute_test_metrics(predictions)
            if metrics:
                self._save_metrics_report(metrics)
                self._print_metrics_summary(metrics)
            
            logger.info("测试预测流程完成！")
            return True
            
        except Exception as e:
            logger.error(f"测试预测失败: {str(e)}")
            return False
    
    def _save_analysis_report(self, analysis: Dict):
        """保存分析报告"""
        try:
            # 确保输出目录存在
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = os.path.join(
                self.config.output_dir,
                f"prediction_analysis_{timestamp}.json"
            )
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
            logger.info(f"分析报告已保存: {os.path.abspath(report_file)}")
            
        except Exception as e:
            logger.error(f"保存分析报告失败: {str(e)}")
    
    def _save_metrics_report(self, metrics: Dict):
        """保存评估指标报告"""
        try:
            # 确保输出目录存在
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 添加元数据
            report = {
                'metrics': metrics,
                'metadata': {
                    'timestamp': timestamp,
                    'model_path': self.config.model_path,
                    'data_root': self.config.coco_data_root,
                    'adapter_path': self.config.adapter_path
                }
            }
            
            # 保存带时间戳的版本（用于历史记录）
            metrics_file_timestamped = os.path.join(
                self.config.output_dir,
                f"test_metrics_{timestamp}.json"
            )
            with open(metrics_file_timestamped, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            logger.info(f"✅ 评估指标报告已保存: {os.path.abspath(metrics_file_timestamped)}")
            
            # 同时保存固定文件名的版本（用于程序读取）
            metrics_file_fixed = os.path.join(
                self.config.output_dir,
                "test_metrics.json"
            )
            with open(metrics_file_fixed, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            logger.info(f"✅ 最新指标已保存: {os.path.abspath(metrics_file_fixed)}")
            
        except Exception as e:
            logger.error(f"保存评估指标报告失败: {str(e)}")
    
    def _print_metrics_summary(self, metrics: Dict):
        """打印评估指标摘要"""
        print(f"\n{'='*80}")
        print("📊 测试集评估指标")
        print(f"{'='*80}")
        
        if 'Bleu_1' in metrics:
            print(f"\n🔹 BLEU指标:")
            print(f"  BLEU-1: {metrics.get('Bleu_1', 0):.4f}")
            print(f"  BLEU-2: {metrics.get('Bleu_2', 0):.4f}")
            print(f"  BLEU-3: {metrics.get('Bleu_3', 0):.4f}")
            print(f"  BLEU-4: {metrics.get('Bleu_4', 0):.4f}")
        
        if 'CIDEr' in metrics:
            print(f"\n🔹 CIDEr: {metrics['CIDEr']:.4f}")
        
        if 'ROUGE_L' in metrics:
            print(f"🔹 ROUGE-L: {metrics['ROUGE_L']:.4f}")
        
        if 'METEOR' in metrics:
            print(f"🔹 METEOR: {metrics['METEOR']:.4f}")
        
        print(f"\n{'='*80}\n")
    
    def _print_analysis_summary(self, analysis: Dict):
        """打印分析摘要"""
        print(f"\n{'='*80}")
        print("测试预测分析摘要")
        print(f"{'='*80}")
        
        basic_stats = analysis.get('basic_stats', {})
        vocab_stats = analysis.get('vocabulary_stats', {})
        length_dist = analysis.get('length_distribution', {})
        
        print(f"基本统计:")
        print(f"  - 总预测数量: {basic_stats.get('total_predictions', 0):,}")
        print(f"  - 平均描述长度: {basic_stats.get('avg_caption_length', 0):.2f} 词")
        print(f"  - 描述长度范围: {basic_stats.get('min_caption_length', 0)} ~ {basic_stats.get('max_caption_length', 0)} 词")
        
        print(f"\n词汇统计:")
        print(f"  - 总词数: {vocab_stats.get('total_words', 0):,}")
        print(f"  - 独特词汇: {vocab_stats.get('unique_words', 0):,}")
        print(f"  - 词汇多样性: {vocab_stats.get('vocabulary_diversity', 0):.4f}")
        
        print(f"\n长度分布:")
        print(f"  - 1-5词: {length_dist.get('length_1_5', 0)}")
        print(f"  - 6-10词: {length_dist.get('length_6_10', 0)}")
        print(f"  - 11-15词: {length_dist.get('length_11_15', 0)}")
        print(f"  - 16-20词: {length_dist.get('length_16_20', 0)}")
        print(f"  - 20+词: {length_dist.get('length_20_plus', 0)}")
        
        print(f"\n{'='*80}\n")
    
    def predict_test_set(self, model_loader: 'LLaVAModelLoader') -> bool:
        """
        使用提供的模型加载器对测试集进行预测
        
        Args:
            model_loader: 已加载的模型加载器
            
        Returns:
            bool: 预测是否成功
        """
        try:
            logger.info("开始使用提供的模型加载器进行测试集预测...")
            
            # 使用提供的模型加载器
            self.model_loader = model_loader
            
            # 确保模型处于评估模式
            if self.model_loader.model is not None:
                self.model_loader.model.eval()
                logger.info("模型已设置为评估模式")
            
            # 设置数据配置
            if not self.setup_data():
                return False
            
            # 直接生成预测，跳过模型设置
            predictions = self.generate_test_predictions()
            
            if not predictions:
                logger.error("测试预测生成失败")
                return False
            
            # 保存预测结果
            pred_file = self.save_test_predictions(predictions)
            
            # 创建提交文件（仅COCO需要，Flickr30K和COCO Karpathy有测试集标注可直接评估）
            data_root = self.config.coco_data_root
            is_flickr = 'flickr30k' in data_root.lower() or 'flickr' in data_root.lower()
            is_karpathy = 'karpathy' in data_root.lower() or 'coco2014' in data_root.lower()
            
            if self.config.create_submission_file:
                if not is_flickr and not is_karpathy:
                    # COCO: 生成提交文件
                    submission_file = self.create_submission_file(predictions)
                    if submission_file:
                        logger.info(f"提交文件创建成功: {submission_file}")
                else:
                    # Flickr30K/COCO Karpathy: 跳过提交文件生成
                    dataset_name = "Flickr30K" if is_flickr else "COCO Karpathy"
                    logger.info(f"{dataset_name}数据集：测试集有标注，可通过评估阶段直接获取指标，无需生成提交文件")
            
            # 分析预测结果
            analysis = self.analyze_predictions(predictions)
            if analysis:
                self._save_analysis_report(analysis)
                self._print_analysis_summary(analysis)
            
            # 计算评估指标（如果测试集有标注）
            metrics = self.compute_test_metrics(predictions)
            if metrics:
                self._save_metrics_report(metrics)
                self._print_metrics_summary(metrics)
            
            logger.info("测试集预测完成")
            return True
                
        except Exception as e:
            logger.error(f"测试集预测异常: {str(e)}")
            return False
    
    def cleanup(self):
        """清理资源"""
        if self.model_loader:
            self.model_loader.cleanup()

# 便捷函数
def create_test_predictor(model_path: str = "/root/autodl-tmp/llava-1.5-7b",
                         coco_data_root: str = "/root/autodl-tmp/COCO2017",
                         output_dir: str = "./coco_test_predictions",
                         adapter_path: Optional[str] = None) -> COCOTestPredictor:
    """创建COCO测试预测器的便捷函数"""
    config = COCOTestPredictorConfig()
    config.model_path = model_path
    config.coco_data_root = coco_data_root
    config.output_dir = output_dir
    config.adapter_path = adapter_path
    
    return COCOTestPredictor(config)

def run_test_prediction(model_path: str = "/root/autodl-tmp/llava-1.5-7b",
                       coco_data_root: str = "/root/autodl-tmp/COCO2017",
                       max_samples: Optional[int] = None,
                       team_name: str = "LLaVA-COCO-Team",
                       adapter_path: Optional[str] = None) -> bool:
    """运行COCO测试预测的便捷函数"""    
    predictor = create_test_predictor(model_path, coco_data_root, adapter_path=adapter_path)
    predictor.config.max_test_samples = max_samples
    predictor.config.team_name = team_name
    
    try:
        success = predictor.run_test_prediction()
        return success
    finally:
        predictor.cleanup()

