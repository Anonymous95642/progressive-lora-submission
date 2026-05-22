"""
COCO评估器
实现COCO数据集的标准评估指标计算和结果分析

"""

import os
# 设置tokenizers并行化环境变量，避免fork警告
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
from collections import defaultdict
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pandas as pd
from tqdm import tqdm

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

class COCOEvaluationConfig:
    """COCO评估配置类"""
    
    def __init__(self):
        """初始化评估配置"""
        # 基础路径
        self.model_path = "/root/autodl-tmp/llava-1.5-7b"
        self.coco_data_root = "/root/autodl-tmp/COCO2017"
        self.output_dir = os.path.join(os.getcwd(), "coco_evaluation_output")  # 确保在当前项目目录
        self.results_dir = os.path.join(self.output_dir, "results")
        self.predictions_dir = os.path.join(self.output_dir, "predictions")
        
        # 评估参数
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
        
        # 评估选项
        self.eval_splits = ["val"]  # 可以是 ["train", "val"] 
        self.max_eval_samples = None  # None表示评估所有样本
        self.save_predictions = True
        self.compute_detailed_metrics = True
        
        # 评估指标配置
        # 
        # 核心指标（已启用）：BLEU-1/2/3/4, ROUGE-L, CIDEr
        # 可选指标（已禁用）：METEOR, SPICE
        # 
        # 💡 说明：
        # - METEOR和SPICE需要Java环境，非COCO官方必需指标
        # - 当前启用的BLEU/ROUGE/CIDEr已充分评估模型质量
        # - 大部分COCO论文仅报告BLEU-4和CIDEr即可发表
        # - 详细说明见文档: EVALUATION_METRICS.md
        # 
        self.compute_spice = False   # ❌ 已禁用（需Java+CoreNLP，计算慢，非必需）
        self.compute_meteor = False  # ❌ 已禁用（需Java，与BLEU高度相关，非必需）
        self.require_java_metrics = False  # 不强制要求Java依赖的指标必须成功
        
        # 💡 说明：验证集最优模型基于validation loss保存，不是基于BLEU/CIDEr
        # 原因：1) 训练时计算BLEU等指标太耗时 2) validation loss更稳定可靠
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.predictions_dir, exist_ok=True)
        
        logger.info("COCO评估配置初始化完成")

class COCOCaptionEvaluator:
    """COCO图像描述评估器"""
    
    def __init__(self, config: COCOEvaluationConfig):
        """
        初始化COCO评估器
        
        Args:
            config: 评估配置对象
        """
        self.config = config
        self.model_loader = None
        self.coco_config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 评估结果存储
        self.predictions = {}
        self.evaluation_results = {}
        
        logger.info(f"COCO评估器初始化完成，使用设备: {self.device}")
    
    def setup_model(self) -> bool:
        """设置评估模型"""
        try:
            logger.info("正在加载LLaVA模型用于评估...")
            self.model_loader = LLaVAModelLoader(self.config.model_path)
            
            success = self.model_loader.load_model(
                load_in_8bit=self.config.load_in_8bit,
                load_in_4bit=self.config.load_in_4bit
            )
            
            if not success:
                logger.error("模型加载失败")
                return False
            
            # 设置模型为评估模式
            self.model_loader.model.eval()
            
            logger.info("评估模型设置完成")
            return True
            
        except Exception as e:
            logger.error(f"模型设置失败: {str(e)}")
            return False
    
    def setup_data(self) -> bool:
        """设置数据配置"""
        try:
            logger.info("正在设置数据配置...")
            
            # 自动检测数据集类型（支持 COCO / Flickr8k / Flickr30K / COCO Karpathy / VizWiz-Captions）
            data_root = self.config.coco_data_root
            lower_root = data_root.lower()
            if 'flickr8k' in lower_root:
                logger.info("检测到Flickr8k数据集，使用适配器...")
                from flickr8k_adapter import Flickr8kAdapter
                self.coco_config = Flickr8kAdapter(data_root)
            elif 'flickr30k' in lower_root or 'flickr' in lower_root:
                logger.info("检测到Flickr30K数据集，使用适配器...")
                from flickr30k_adapter import Flickr30KAdapter
                self.coco_config = Flickr30KAdapter(data_root)
            elif 'vizwiz' in lower_root:
                logger.info("检测到VizWiz-Captions数据集，使用适配器...")
                from vizwiz_adapter import VizWizCaptionAdapter
                self.coco_config = VizWizCaptionAdapter(data_root)
            elif 'karpathy' in lower_root or 'coco2014' in lower_root:
                logger.info("检测到COCO Karpathy数据集，使用适配器...")
                import sys
                import os
                # 添加父目录到路径以导入适配器
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                from coco_karpathy_adapter import COCOKarpathyAdapter
                self.coco_config = COCOKarpathyAdapter(data_root)
            else:
                logger.info("使用COCO数据集...")
                self.coco_config = COCODatasetConfig(data_root)
            
            if not self.coco_config.validate_paths():
                logger.error("数据集路径验证失败")
                return False
            
            logger.info("数据配置设置完成")
            return True
            
        except Exception as e:
            logger.error(f"数据设置失败: {str(e)}")
            return False
    
    def generate_predictions(self, split: str = "val") -> Dict[int, str]:
        """
        生成模型预测结果
        
        Args:
            split: 数据集分割类型
            
        Returns:
            Dict[int, str]: 图像ID到预测描述的映射
        """
        logger.info(f"开始生成{split}数据集的预测结果...")
        
        # 创建数据加载器
        data_loader = COCODataLoader(self.coco_config)
        dataloader = data_loader.create_dataloader(
            split=split,
            batch_size=1,  # 逐个处理
            shuffle=False,
            num_workers=0,  # 避免多进程问题
            max_samples=self.config.max_eval_samples
        )
        
        predictions = {}
        
        # 创建进度条
        pbar = tqdm(dataloader, desc=f"生成{split}预测")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # 获取样本（批次大小为1）
                image_id = batch['image_ids'][0]
                image = batch['images'][0]
                
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
                
                predictions[image_id] = description
                
                # 更新进度条
                pbar.set_postfix({'completed': f'{batch_idx+1}'})
                
            except Exception as e:
                logger.error(f"生成预测失败，图像ID {batch.get('image_ids', [None])[0]}: {str(e)}")
                continue
        
        logger.info(f"{split}数据集预测生成完成，共{len(predictions)}个预测结果")
        return predictions
    
    def save_predictions(self, predictions: Dict[int, str], split: str):
        """
        保存预测结果
        
        Args:
            predictions: 预测结果字典
            split: 数据集分割类型
        """
        try:
            # 保存为COCO格式的JSON文件
            coco_format_predictions = []
            for image_id, caption in predictions.items():
                coco_format_predictions.append({
                    "image_id": int(image_id),  # 确保是Python int类型
                    "caption": str(caption)     # 确保是字符串类型
                })
            
            # 确保predictions字典中的键值都是JSON可序列化的
            safe_predictions = {}
            for k, v in predictions.items():
                safe_predictions[str(k)] = str(v)
            
            # 保存COCO格式
            coco_pred_file = os.path.join(
                self.config.predictions_dir,
                f"predictions_{split}_coco_format.json"
            )
            with open(coco_pred_file, 'w', encoding='utf-8') as f:
                json.dump(coco_format_predictions, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
            # 保存简单格式
            simple_pred_file = os.path.join(
                self.config.predictions_dir,
                f"predictions_{split}_simple.json"
            )
            with open(simple_pred_file, 'w', encoding='utf-8') as f:
                json.dump(safe_predictions, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
            logger.info(f"{split}预测结果已保存:")
            logger.info(f"  - COCO格式: {os.path.abspath(coco_pred_file)}")
            logger.info(f"  - 简单格式: {os.path.abspath(simple_pred_file)}")
            
        except Exception as e:
            logger.error(f"保存预测结果失败: {str(e)}")
    
    def compute_coco_metrics(self, predictions: Dict[int, str], split: str) -> Dict:
        """
        计算图像描述评估指标（使用pycocoevalcap）
        
        Args:
            predictions: 预测结果字典
            split: 数据集分割类型
            
        Returns:
            Dict: 评估指标结果
        """
        try:
            logger.info(f"正在计算{split}数据集的图像描述评估指标...")
            
            # 导入图像描述评估工具
            try:
                from pycocoevalcap.cider.cider import Cider
                from pycocoevalcap.bleu.bleu import Bleu
                from pycocoevalcap.meteor.meteor import Meteor
                from pycocoevalcap.rouge.rouge import Rouge
                from pycocoevalcap.spice.spice import Spice
                evaluation_tools_available = True
            except ImportError:
                logger.warning("pycocoevalcap未安装，将跳过COCO标准指标计算")
                logger.warning("安装命令: pip install pycocoevalcap")
                evaluation_tools_available = False
            
            if not evaluation_tools_available:
                return {}
            
            # 加载真实标注
            if split == "train":
                ann_file = self.coco_config.train_captions_file
            elif split == "val":
                ann_file = self.coco_config.val_captions_file
            elif split == "test":
                ann_file = self.coco_config.test_captions_file
            else:
                logger.error(f"不支持的数据集分割: {split}")
                return {}
            
            # 检查标注文件是否存在
            if not os.path.exists(ann_file):
                logger.error(f"标注文件不存在: {ann_file}")
                return {}
            
            coco_gt = COCO(ann_file)
            
            # 准备真实标注数据
            gts = {}
            for img_id in predictions.keys():
                ann_ids = coco_gt.getAnnIds(imgIds=int(img_id))
                anns = coco_gt.loadAnns(ann_ids)
                # 确保每个标注都是非空字符串
                captions = [ann['caption'].strip() for ann in anns if ann.get('caption', '').strip()]
                if captions:  # 只添加有有效标注的图像
                    gts[int(img_id)] = captions
            
            # 准备预测数据 - 确保格式正确且非空
            res = {}
            for img_id, caption in predictions.items():
                img_id = int(img_id)
                if img_id in gts and caption and caption.strip():  # 只处理有真实标注且预测非空的图像
                    res[img_id] = [caption.strip()]
            
            # 过滤gts，只保留res中存在的图像ID（BLEU要求keys完全一致）
            gts_filtered = {img_id: gts[img_id] for img_id in res.keys() if img_id in gts}
            
            logger.info(f"评估数据: {len(gts)}张图有标注, {len(res)}张图有预测, {len(gts_filtered)}张图用于计算")
            
            # 计算评估指标
            metrics = {}
            
            try:
                # BLEU指标
                logger.info("正在计算BLEU指标...")
                bleu_scorer = Bleu(n=4)
                bleu_scores, _ = bleu_scorer.compute_score(gts_filtered, res)
                metrics['Bleu_1'] = float(bleu_scores[0])
                metrics['Bleu_2'] = float(bleu_scores[1])
                metrics['Bleu_3'] = float(bleu_scores[2])
                metrics['Bleu_4'] = float(bleu_scores[3])
                logger.info(f"BLEU指标计算完成: BLEU-1={metrics['Bleu_1']:.4f}, BLEU-2={metrics['Bleu_2']:.4f}, BLEU-3={metrics['Bleu_3']:.4f}, BLEU-4={metrics['Bleu_4']:.4f}")
            except Exception as e:
                logger.warning(f"BLEU指标计算失败: {str(e)}")
            
            if self.config.compute_meteor:
                try:
                    # 首先检查Java环境是否可用
                    import subprocess
                    try:
                        result = subprocess.run(['java', '-version'], 
                                              capture_output=True, text=True, timeout=10)
                        if result.returncode != 0:
                            raise RuntimeError(f"Java环境检查失败: {result.stderr}")
                    except (subprocess.TimeoutExpired, FileNotFoundError, RuntimeError) as java_e:
                        logger.warning(f"Java环境不可用，跳过METEOR指标: {str(java_e)}")
                        logger.info("建议使用无Java环境的配置: python no_java_config.py")
                        raise java_e
                    
                    # METEOR指标
                    logger.info("正在计算METEOR指标...")
                    meteor_scorer = Meteor()
                    meteor_score, _ = meteor_scorer.compute_score(gts_filtered, res)
                    
                    # 处理METEOR返回值可能是bytes或其他格式的问题
                    if isinstance(meteor_score, bytes):
                        # 如果是bytes，尝试解码并提取第一个数值
                        try:
                            score_str = meteor_score.decode('utf-8').strip()
                            # 检查是否包含错误信息
                            if 'Error:' in score_str or 'specify SCORE or EVAL or SING' in score_str:
                                raise ValueError(f"METEOR工具返回错误: {score_str}")
                            # 提取第一个浮点数
                            import re
                            numbers = re.findall(r'\d+\.?\d*', score_str)
                            if numbers:
                                meteor_score = float(numbers[0]) / 100.0  # METEOR通常以百分比形式返回
                            else:
                                raise ValueError("无法从METEOR输出中提取数值")
                        except Exception as parse_e:
                            logger.warning(f"METEOR输出解析失败: {str(parse_e)}, 原始输出: {meteor_score}")
                            raise parse_e
                    elif isinstance(meteor_score, str):
                        # 检查字符串是否包含错误信息
                        if 'Error:' in meteor_score or 'specify SCORE or EVAL or SING' in meteor_score:
                            raise ValueError(f"METEOR工具返回错误: {meteor_score}")
                        # 如果是字符串，尝试直接转换或提取数值
                        try:
                            meteor_score = float(meteor_score)
                        except ValueError:
                            import re
                            numbers = re.findall(r'\d+\.?\d*', meteor_score)
                            if numbers:
                                # METEOR分数通常是0-1之间的小数，如果大于1则除以100
                                parsed_score = float(numbers[0])
                                meteor_score = parsed_score / 100.0 if parsed_score > 1 else parsed_score
                            else:
                                raise ValueError(f"无法解析METEOR分数: {meteor_score}")
                    
                    # 确保METEOR分数在合理范围内
                    if isinstance(meteor_score, (int, float)):
                        meteor_score = float(meteor_score)
                        if meteor_score > 1:
                            meteor_score = meteor_score / 100.0
                    
                    metrics['METEOR'] = meteor_score
                    logger.info(f"METEOR指标计算完成: {metrics['METEOR']:.4f}")
                except Exception as e:
                    logger.warning(f"METEOR指标计算失败: {str(e)}")
                    logger.warning("METEOR指标由于Java环境问题失败，建议使用无Java环境配置")
                    logger.info("解决方案: 1) 修复Java环境 2) 使用 python no_java_config.py")
                    if self.config.require_java_metrics:
                        raise e
            else:
                logger.info("METEOR指标已禁用，跳过计算")
            
            try:
                # ROUGE-L指标
                logger.info("正在计算ROUGE-L指标...")
                rouge_scorer = Rouge()
                rouge_score, _ = rouge_scorer.compute_score(gts_filtered, res)
                metrics['ROUGE_L'] = float(rouge_score)
                logger.info(f"ROUGE-L指标计算完成: {metrics['ROUGE_L']:.4f}")
            except Exception as e:
                logger.warning(f"ROUGE-L指标计算失败: {str(e)}")
            
            try:
                # CIDEr指标
                logger.info("正在计算CIDEr指标...")
                
                # 验证输入数据格式
                if not gts_filtered or not res:
                    raise ValueError("CIDEr计算需要有效的真实标注和预测数据")
                
                # 检查数据一致性
                common_ids = set(gts_filtered.keys()) & set(res.keys())
                if len(common_ids) == 0:
                    raise ValueError("真实标注和预测数据没有共同的图像ID")
                
                logger.info(f"CIDEr计算将使用 {len(common_ids)} 个共同图像ID")
                
                # 过滤数据，只保留共同的ID，并进行严格的数据清理
                filtered_gts = {}
                filtered_res = {}
                
                for img_id in common_ids:
                    if img_id in gts_filtered and img_id in res:
                        # 清理真实标注
                        gt_captions = []
                        for caption in gts_filtered[img_id]:
                            if caption and isinstance(caption, str) and caption.strip():
                                clean_caption = caption.strip()
                                # 移除可能导致问题的特殊字符
                                clean_caption = ' '.join(clean_caption.split())
                                if len(clean_caption) > 0:
                                    gt_captions.append(clean_caption)
                        
                        # 清理预测数据
                        pred_captions = []
                        for caption in res[img_id]:
                            if caption and isinstance(caption, str) and caption.strip():
                                clean_caption = caption.strip()
                                # 移除可能导致问题的特殊字符
                                clean_caption = ' '.join(clean_caption.split())
                                if len(clean_caption) > 0:
                                    pred_captions.append(clean_caption)
                        
                        # 只有当真实标注和预测都有效时才添加
                        if gt_captions and pred_captions:
                            filtered_gts[img_id] = gt_captions
                            filtered_res[img_id] = pred_captions
                
                # 验证过滤后的数据
                if not filtered_gts or not filtered_res:
                    raise ValueError("数据清理后没有有效的数据用于CIDEr计算")
                
                if len(filtered_gts) != len(filtered_res):
                    raise ValueError("清理后的真实标注和预测数据数量不匹配")
                
                # 检查样本数量是否足够进行可靠的CIDEr计算
                sample_count = len(filtered_gts)
                if sample_count < 50:
                    warning_msg = f"⚠️  警告: 样本数量过少 ({sample_count} 个)，CIDEr分数可能不可靠！"
                    print(warning_msg)
                    logger.warning(warning_msg)
                    if sample_count < 10:
                        warning_msg = f"⚠️  严重警告: 样本数量极少 ({sample_count} 个)，CIDEr分数几乎没有意义！"
                        print(warning_msg)
                        logger.warning(warning_msg)
                        print("建议: 使用至少50个样本进行CIDEr评估，或者关注其他指标如BLEU、METEOR等")
                
                # 添加详细调试信息
                print(f"开始CIDEr计算 - 清理后数据量: gts={len(filtered_gts)}, res={len(filtered_res)}")
                sample_key = list(filtered_gts.keys())[0]
                print(f"数据格式检查 - Key: {sample_key}")
                print(f"  参考数量: {len(filtered_gts[sample_key])}, 样本: {filtered_gts[sample_key][0][:50]}...")
                print(f"  预测数量: {len(filtered_res[sample_key])}, 样本: {filtered_res[sample_key][0][:50]}...")
                
                # 验证数据类型
                for img_id in list(filtered_gts.keys())[:3]:  # 检查前3个样本
                    for caption in filtered_gts[img_id]:
                        if not isinstance(caption, str):
                            raise ValueError(f"真实标注包含非字符串数据: {type(caption)}")
                    for caption in filtered_res[img_id]:
                        if not isinstance(caption, str):
                            raise ValueError(f"预测数据包含非字符串数据: {type(caption)}")
                
                # 创建CIDEr评分器并计算
                cider_scorer = Cider()
                cider_score, cider_scores = cider_scorer.compute_score(filtered_gts, filtered_res)
                
                print(f"CIDEr原始返回: score={cider_score}, type={type(cider_score)}")
                if hasattr(cider_scores, '__len__') and len(cider_scores) > 0:
                    print(f"CIDEr详细分数样本: {cider_scores[:5]}")
                
                # 处理CIDEr分数
                if isinstance(cider_score, (list, tuple, np.ndarray)):
                    if len(cider_score) > 0:
                        cider_score = float(cider_score[0])
                    else:
                        raise ValueError("CIDEr返回了空的分数列表")
                elif isinstance(cider_score, (int, float, np.number)):
                    cider_score = float(cider_score)
                else:
                    raise ValueError(f"CIDEr返回了不支持的分数类型: {type(cider_score)}")
                
                # 验证CIDEr分数的有效性
                if cider_score is None or np.isnan(cider_score) or np.isinf(cider_score):
                    raise ValueError(f"CIDEr计算返回了无效分数: {cider_score}")
                
                metrics['CIDEr'] = cider_score
                print(f"CIDEr最终设置值: {metrics['CIDEr']}")
                logger.info(f"CIDEr指标计算完成: {metrics['CIDEr']:.4f}")
                
            except Exception as e:
                logger.error(f"CIDEr指标计算失败: {str(e)}")
                logger.error(f"错误类型: {type(e).__name__}")
                
                # 提供详细的调试信息
                try:
                    logger.debug(f"原始真实标注数量: {len(gts) if gts else 0}")
                    logger.debug(f"原始预测数据数量: {len(res) if res else 0}")
                    if gts and res:
                        common_ids = set(gts.keys()) & set(res.keys())
                        logger.debug(f"共同图像ID数量: {len(common_ids)}")
                        if common_ids:
                            sample_id = list(common_ids)[0]
                            logger.debug(f"样本数据 - ID: {sample_id}")
                            logger.debug(f"  真实标注: {gts.get(sample_id, [])}")
                            logger.debug(f"  预测: {res.get(sample_id, [])}")
                except Exception as debug_e:
                    logger.debug(f"无法获取调试信息: {str(debug_e)}")
                
                logger.warning("CIDEr指标计算失败，将跳过该指标")
                # 不设置CIDEr键，让调用者知道该指标不可用
            
            if self.config.compute_spice:
                try:
                    # SPICE指标
                    logger.info("正在计算SPICE指标...")
                    
                    # 检查并创建缓存目录
                    try:
                        import pycocoevalcap.spice.spice as spice_module
                        cache_dir = os.path.join(os.path.dirname(spice_module.__file__), 'cache')
                        os.makedirs(cache_dir, exist_ok=True)
                        logger.info(f"SPICE缓存目录已确保存在: {cache_dir}")
                    except Exception as cache_e:
                        logger.warning(f"无法创建SPICE缓存目录: {str(cache_e)}")
                    
                    # 限制描述长度以避免缓存问题
                    filtered_res = {}
                    for img_id, captions in res.items():
                        filtered_captions = []
                        for caption in captions:
                            # 限制描述长度，避免"Caption may be too long"错误
                            if len(caption.split()) > 50:  # 限制为50个词
                                caption = ' '.join(caption.split()[:50])
                            filtered_captions.append(caption)
                        filtered_res[img_id] = filtered_captions
                    
                    spice_scorer = Spice()
                    spice_score, _ = spice_scorer.compute_score(gts_filtered, filtered_res)
                    metrics['SPICE'] = float(spice_score)
                    logger.info(f"SPICE指标计算完成: {metrics['SPICE']:.4f}")
                except Exception as e:
                    logger.warning(f"SPICE指标计算失败: {str(e)}")
                    logger.warning("这通常是由于描述过长或缓存权限问题导致的")
                    if "cache" in str(e).lower():
                        logger.warning("建议检查SPICE缓存目录权限或使用较短的描述")
                    if self.config.require_java_metrics:
                        raise e
            else:
                logger.info("SPICE指标已禁用，跳过计算")
            
            # 打印所有成功计算的指标摘要
            logger.info(f"{split}数据集图像描述评估指标计算完成:")
            for metric_name, metric_value in metrics.items():
                logger.info(f"  - {metric_name}: {metric_value:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"图像描述评估指标计算失败: {str(e)}")
            return {}
    
    def compute_additional_metrics(self, predictions: Dict[int, str], split: str) -> Dict:
        """
        计算额外的评估指标
        
        Args:
            predictions: 预测结果字典
            split: 数据集分割类型
            
        Returns:
            Dict: 额外指标结果
        """
        try:
            logger.info(f"正在计算{split}数据集的额外指标...")
            
            # 创建数据集获取真实标注
            dataset = COCOCaptionDataset(
                config=self.coco_config,
                split=split,
                transform=None
            )
            
            # 统计指标
            caption_lengths = []
            word_counts = defaultdict(int)
            unique_words = set()
            
            for image_id, pred_caption in predictions.items():
                # 描述长度统计
                words = pred_caption.split()
                caption_lengths.append(len(words))
                
                # 词汇统计
                for word in words:
                    word_counts[word] += 1
                    unique_words.add(word.lower())
            
            # 计算统计量，确保所有值都是JSON可序列化的
            avg_length = float(np.mean(caption_lengths))
            std_length = float(np.std(caption_lengths))
            min_length = int(np.min(caption_lengths))
            max_length = int(np.max(caption_lengths))
            total_unique = int(len(unique_words))
            total_predictions = int(len(predictions))
            vocab_diversity = float(len(unique_words) / sum(len(cap.split()) for cap in predictions.values()))
            
            additional_metrics = {
                'avg_caption_length': avg_length,
                'std_caption_length': std_length,
                'min_caption_length': min_length,
                'max_caption_length': max_length,
                'total_unique_words': total_unique,
                'total_predictions': total_predictions,
                'vocabulary_diversity': vocab_diversity
            }
            
            # 最常见词汇
            most_common_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            additional_metrics['most_common_words'] = most_common_words
            
            # 打印额外指标摘要
            logger.info(f"{split}数据集额外指标计算完成:")
            logger.info(f"  - 平均描述长度: {avg_length:.2f} 词")
            logger.info(f"  - 长度标准差: {std_length:.2f}")
            logger.info(f"  - 最短描述: {min_length} 词")
            logger.info(f"  - 最长描述: {max_length} 词")
            logger.info(f"  - 独特词汇数: {total_unique}")
            logger.info(f"  - 预测数量: {total_predictions}")
            logger.info(f"  - 词汇多样性: {vocab_diversity:.4f}")
            logger.info(f"  - 最常见词汇: {most_common_words[:5]}")  # 只显示前5个
            return additional_metrics
            
        except Exception as e:
            logger.error(f"额外指标计算失败: {str(e)}")
            return {}
    
    def evaluate_split(self, split: str) -> Dict:
        """
        评估单个数据分割
        
        Args:
            split: 数据集分割类型
            
        Returns:
            Dict: 评估结果
        """
        logger.info(f"开始评估{split}数据集...")
        
        # 生成预测
        predictions = self.generate_predictions(split)
        
        if not predictions:
            logger.error(f"{split}数据集预测生成失败")
            return {}
        
        # 保存预测结果
        if self.config.save_predictions:
            self.save_predictions(predictions, split)
        
        # 计算COCO标准指标（检查是否有标注文件）
        coco_metrics = {}
        if split == "train":
            logger.info(f"⚠️  训练集不需要计算评估指标（训练集用于训练，不用于评估）")
        else:
            # 检查是否有标注文件
            ann_file = None
            if split == "val":
                ann_file = self.coco_config.val_captions_file
            elif split == "test":
                ann_file = self.coco_config.test_captions_file
            
            if ann_file and os.path.exists(ann_file):
                logger.info(f"计算{split}数据集的标准评估指标...")
                coco_metrics = self.compute_coco_metrics(predictions, split)
            else:
                logger.warning(f"⚠️  {split}数据集没有标注文件，跳过评估指标计算")
                if split == "test":
                    logger.info(f"提示: 如果是COCO数据集，要获取测试集的官方评估指标，请将预测结果提交到官方评估服务器")
                    logger.info(f"COCO评估服务器: https://competitions.codalab.org/competitions/3221")
        
        # 计算额外指标（统计类指标，不需要标注）
        additional_metrics = {}
        if self.config.compute_detailed_metrics:
            logger.info(f"计算{split}数据集的统计指标...")
            additional_metrics = self.compute_additional_metrics(predictions, split)
        
        # 合并结果
        evaluation_result = {
            'split': split,
            'num_predictions': len(predictions),
            'coco_metrics': coco_metrics,
            'additional_metrics': additional_metrics,
            'predictions': predictions if len(predictions) <= 100 else {}  # 只保存少量样例
        }
        
        return evaluation_result
    
    def run_evaluation(self) -> Dict:
        """
        运行完整评估流程
        
        Returns:
            Dict: 完整评估结果
        """
        try:
            logger.info("开始COCO评估流程...")
            
            # 设置模型
            if not self.setup_model():
                return {}
            
            # 设置数据
            if not self.setup_data():
                return {}
            
            # 评估各个数据分割
            all_results = {}
            for split in self.config.eval_splits:
                logger.info(f"评估{split}数据集...")
                split_results = self.evaluate_split(split)
                
                # 检查评估结果是否有效
                if not split_results:
                    logger.error(f"{split}数据集评估失败，返回空结果")
                    all_results[split] = {}
                elif not split_results.get('coco_metrics'):
                    logger.warning(f"{split}数据集评估部分失败，缺少COCO指标")
                    all_results[split] = split_results
                else:
                    logger.info(f"✅ {split}数据集评估成功完成")
                all_results[split] = split_results
            
            # 保存完整评估结果
            results_file = os.path.join(
                self.config.results_dir,
                "evaluation_results.json"
            )
            with open(results_file, 'w', encoding='utf-8') as f:
                # 移除predictions字段以减少文件大小
                save_results = {}
                for split, results in all_results.items():
                    save_results[split] = {k: v for k, v in results.items() if k != 'predictions'}
                json.dump(save_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
            logger.info(f"评估结果已保存: {os.path.abspath(results_file)}")
            
            # 为每个分割保存详细指标（用于公平对比）
            for split, results in all_results.items():
                if results and results.get('coco_metrics'):
                    split_metrics_file = os.path.join(
                        self.config.results_dir,
                        f"{split}_metrics_with_additional.json"
                    )
                    split_metrics_data = {
                        'coco_metrics': results['coco_metrics'],
                        'additional_metrics': results.get('additional_metrics', {}),
                        'num_predictions': results.get('num_predictions', 0),
                        'split': split
                    }
                    with open(split_metrics_file, 'w', encoding='utf-8') as f:
                        json.dump(split_metrics_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
                    logger.info(f"{split.capitalize()}指标已保存: {split_metrics_file}")
            
            # 打印评估摘要
            logger.info("正在生成评估摘要...")
            try:
                self.print_evaluation_summary(all_results)
                logger.info("评估摘要生成完成")
            except Exception as e:
                logger.warning(f"评估摘要生成失败: {str(e)}")
                logger.info("跳过摘要生成，继续执行...")
            
            # 检查是否有有效的评估结果
            has_valid_results = any(
                results and isinstance(results.get('coco_metrics'), dict) and results['coco_metrics']
                for results in all_results.values()
            )
            
            if has_valid_results:
                logger.info("✅ 评估阶段成功完成！")
                # 提示下一步操作（仅在直接运行评估器时显示）
                if __name__ == "__main__":
                    logger.info("=" * 80)
                    logger.info("🎯 评估阶段完成！下一步建议:")
                    logger.info("1. 运行测试预测: python run.py prediction --model-path ./coco_training_output/best_model")
                    logger.info("2. 或运行完整流水线: python main_pipeline.py")
                    logger.info("3. 查看评估结果: ls -la ./coco_evaluation_output/results/")
                    logger.info("=" * 80)
            else:
                logger.warning("⚠️  评估完成但未获得有效结果")
            
            # 清理资源
            logger.info("开始清理评估资源...")
            try:
                # 清理CUDA缓存（如果使用GPU）
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("CUDA缓存已清理")
                
                # 强制垃圾回收
                import gc
                gc.collect()
                logger.info("垃圾回收完成")
                
            except Exception as cleanup_e:
                logger.warning(f"资源清理时出现警告: {str(cleanup_e)}")
            
            logger.info("评估流程完全结束")
            return all_results
            
        except Exception as e:
            logger.error(f"评估失败: {str(e)}")
            logger.error(f"异常详情: {type(e).__name__}")
            import traceback
            logger.error(f"完整错误堆栈:\n{traceback.format_exc()}")
            
            # 即使出错也尝试清理资源
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
            except:
                pass
                
            return {}
    
    def print_evaluation_summary(self, results: Dict):
        """打印评估结果摘要"""
        try:
            print(f"\n{'='*80}")
            print("COCO评估结果摘要")
            print(f"{'='*80}")
            logger.info("开始打印评估结果摘要...")
            
            for split, split_results in results.items():
                print(f"\n{split.upper()}数据集结果:")
                print("-" * 50)
                
                if 'coco_metrics' in split_results:
                    coco_metrics = split_results['coco_metrics']
                    print("COCO标准指标:")
                    
                    # 按照标准顺序显示指标，跳过不可用的指标
                    standard_metrics = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE']
                    for metric in standard_metrics:
                        if metric in coco_metrics:
                            print(f"  {metric}: {coco_metrics[metric]:.4f}")
                        else:
                            print(f"  {metric}: N/A (计算失败或已禁用)")
                    
                    # 显示其他可能的指标
                    for metric, value in coco_metrics.items():
                        if metric not in standard_metrics:
                            print(f"  {metric}: {value:.4f}")
                
                if 'additional_metrics' in split_results:
                    add_metrics = split_results['additional_metrics']
                    print("\n额外指标:")
                    print(f"  平均描述长度: {add_metrics.get('avg_caption_length', 0):.2f} 词")
                    print(f"  词汇多样性: {add_metrics.get('vocabulary_diversity', 0):.4f}")
                    print(f"  独特词汇数: {add_metrics.get('total_unique_words', 0)}")
                    print(f"  预测数量: {add_metrics.get('total_predictions', 0)}")
                    
                    # 添加样本数量警告
                    total_preds = add_metrics.get('total_predictions', 0)
                    if total_preds < 50:
                        print(f"  ⚠️  样本数量警告: {total_preds} 个样本可能导致CIDEr等指标不可靠")
                        if total_preds < 10:
                            print(f"  ⚠️  建议增加样本数量至少到50个以获得可靠的评估结果")
        
            print(f"\n{'='*80}\n")
            logger.info("评估结果摘要打印完成")
        
        except Exception as e:
            logger.error(f"打印评估摘要时发生错误: {str(e)}")
            print(f"\n⚠️  评估摘要生成失败: {str(e)}\n")
            # 确保即使出错也能继续执行
    
    def evaluate_model(self, model_loader: 'LLaVAModelLoader') -> bool:
        """
        使用提供的模型加载器评估模型
        
        Args:
            model_loader: 已加载的模型加载器
            
        Returns:
            bool: 评估是否成功
        """
        try:
            logger.info("开始使用提供的模型加载器进行评估...")
            
            # 使用提供的模型加载器
            self.model_loader = model_loader
            
            # 确保模型处于评估模式
            if self.model_loader.model is not None:
                self.model_loader.model.eval()
                logger.info("模型已设置为评估模式")
            
            # 设置数据配置
            if not self.setup_data():
                return False
            
            # 直接运行评估各个数据分割，跳过模型设置
            all_results = {}
            for split in self.config.eval_splits:
                logger.info(f"评估{split}数据集...")
                split_results = self.evaluate_split(split)
                all_results[split] = split_results
            
            if not all_results:
                logger.error("评估失败，没有生成任何结果")
                return False
            
            # 保存完整评估结果
            results_file = os.path.join(
                self.config.results_dir,
                "evaluation_results.json"
            )
            with open(results_file, 'w', encoding='utf-8') as f:
                # 移除predictions字段以减少文件大小
                save_results = {}
                for split, results in all_results.items():
                    save_results[split] = {k: v for k, v in results.items() if k != 'predictions'}
                json.dump(save_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
            logger.info(f"评估结果已保存: {os.path.abspath(results_file)}")
            
            # 打印评估摘要
            logger.info("正在生成模型评估摘要...")
            try:
                self.print_evaluation_summary(all_results)
                logger.info("模型评估摘要生成完成")
            except Exception as e:
                logger.warning(f"模型评估摘要生成失败: {str(e)}")
                logger.info("跳过摘要生成，继续执行...")
            
            logger.info("模型评估完成")
            
            # 清理资源
            logger.info("开始清理评估资源...")
            try:
                # 清理CUDA缓存（如果使用GPU）
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("CUDA缓存已清理")
                
                # 强制垃圾回收
                import gc
                gc.collect()
                logger.info("垃圾回收完成")
                
            except Exception as cleanup_e:
                logger.warning(f"资源清理时出现警告: {str(cleanup_e)}")
            
            logger.info("评估流程完全结束")
            return True
                
        except Exception as e:
            logger.error(f"模型评估异常: {str(e)}")
            logger.error(f"异常详情: {type(e).__name__}")
            import traceback
            logger.error(f"完整错误堆栈:\n{traceback.format_exc()}")
            
            # 即使出错也尝试清理资源
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
            except:
                pass
                
            return False
    
    def cleanup(self):
        """清理资源"""
        if self.model_loader:
            self.model_loader.cleanup()

# 便捷函数
def create_coco_evaluator(model_path: str = "/root/autodl-tmp/llava-1.5-7b",
                         coco_data_root: str = "/root/autodl-tmp/COCO2017",
                         output_dir: str = "./coco_evaluation_output") -> COCOCaptionEvaluator:
    """创建COCO评估器的便捷函数"""
    config = COCOEvaluationConfig()
    config.model_path = model_path
    config.coco_data_root = coco_data_root
    config.output_dir = output_dir
    
    return COCOCaptionEvaluator(config)

def run_coco_evaluation(model_path: str = "/root/autodl-tmp/llava-1.5-7b",
                       coco_data_root: str = "/root/autodl-tmp/COCO2017",
                       eval_splits: List[str] = ["val"],
                       max_samples: Optional[int] = None) -> Dict:
    """运行COCO评估的便捷函数"""
    evaluator = create_coco_evaluator(model_path, coco_data_root)
    evaluator.config.eval_splits = eval_splits
    evaluator.config.max_eval_samples = max_samples
    
    try:
        results = evaluator.run_evaluation()
        return results
    finally:
        evaluator.cleanup()

def test_cider_calculation():
    """测试CIDEr计算逻辑"""
    try:
        from pycocoevalcap.cider.cider import Cider
        
        # 创建测试数据
        gts = {
            1: ["A man is riding a horse", "A person on a horse"],
            2: ["A cat is sleeping", "The cat is resting"]
        }
        
        res = {
            1: ["A man rides a horse"],
            2: ["A cat sleeps"]
        }
        
        print("测试CIDEr计算...")
        print(f"真实标注: {gts}")
        print(f"预测结果: {res}")
        
        cider_scorer = Cider()
        cider_score, cider_scores = cider_scorer.compute_score(gts, res)
        
        print(f"CIDEr分数: {cider_score}")
        print(f"详细分数: {cider_scores}")
        print(f"分数类型: {type(cider_score)}")
        
        if isinstance(cider_score, (list, tuple, np.ndarray)):
            final_score = float(cider_score[0]) if len(cider_score) > 0 else 0.0
        else:
            final_score = float(cider_score)
        
        print(f"最终分数: {final_score}")
        return final_score
        
    except Exception as e:
        print(f"CIDEr测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import sys
    
    # 设置日志级别
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 检查是否是测试模式
    if len(sys.argv) > 1 and sys.argv[1] == "test_cider":
        print("运行CIDEr测试...")
        result = test_cider_calculation()
        if result is not None:
            print(f"✅ CIDEr测试成功，分数: {result}")
        else:
            print("❌ CIDEr测试失败")
        sys.exit(0)
    
    # 正常评估流程
    print("开始COCO评估...")
    
    # 创建评估配置
    config = COCOEvaluationConfig()
    
    # 创建评估器
    evaluator = COCOCaptionEvaluator(config)
    
    try:
        # 运行评估
        results = evaluator.run_evaluation()
        
        if results:
            print("✅ 评估完成")
        else:
            print("❌ 评估失败")
            
    except KeyboardInterrupt:
        print("\n用户中断评估")
    except Exception as e:
        print(f"❌ 评估异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        evaluator.cleanup()

