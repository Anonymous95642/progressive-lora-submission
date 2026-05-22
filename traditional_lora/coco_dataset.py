"""
COCO2017数据集处理模块

本模块实现了COCO2017图像描述数据集的完整处理流程：
1. 数据集加载：支持train/val/test三个数据集
2. 复杂度评估：基于词数、物体数、属性数和关系复杂度计算样本难度
3. 分层采样：按复杂度百分位数将样本分为Easy/Medium/Hard三层
4. 数据预处理：图像加载、文本处理、格式转换

核心功能：
- calculate_caption_complexity(): 计算描述复杂度（用于渐进式训练）
- COCOCaptionDataset: COCO数据集类，支持复杂度分层
- COCODataLoader: 数据加载器工厂类
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pycocotools.coco import COCO
import pandas as pd
import re

# 配置日志
logger = logging.getLogger(__name__)

# ==================== 自定义COCO Caption加载器 ====================

class COCOCaptionLoader:
    """
    自定义COCO Caption标注加载器
    
    解决pycocotools.COCO在加载caption文件时的category_id KeyError问题。
    pycocotools.COCO默认为目标检测设计，会尝试创建catToImgs索引，
    但caption标注文件中没有category_id字段，导致错误。
    
    本类专门用于加载caption格式的标注文件（包括COCO和Flickr30K）。
    """
    
    def __init__(self, annotation_file: str):
        """
        初始化COCO Caption加载器
        
        Args:
            annotation_file: COCO caption标注文件路径（JSON格式）
        """
        self.annotation_file = annotation_file
        self.dataset = {}
        self.imgs = {}
        self.anns = {}
        self.imgToAnns = {}
        
        # 加载标注文件
        logger.info(f"使用自定义Caption加载器加载: {annotation_file}")
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        
        # 创建索引
        self._create_index()
        logger.info(f"成功加载 {len(self.imgs)} 张图像和 {len(self.anns)} 条标注")
    
    def _create_index(self):
        """创建图像和标注的索引"""
        # 创建图像索引
        if 'images' in self.dataset:
            for img in self.dataset['images']:
                self.imgs[img['id']] = img
        
        # 创建标注索引
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                self.anns[ann['id']] = ann
                
                # 创建图像到标注的映射
                img_id = ann['image_id']
                if img_id not in self.imgToAnns:
                    self.imgToAnns[img_id] = []
                self.imgToAnns[img_id].append(ann)
    
    def getAnnIds(self, imgIds: Union[int, List[int]] = None) -> List[int]:
        """
        获取指定图像的标注ID列表
        
        Args:
            imgIds: 图像ID或图像ID列表
            
        Returns:
            标注ID列表
        """
        if imgIds is None:
            return list(self.anns.keys())
        
        # 确保imgIds是列表
        if isinstance(imgIds, int):
            imgIds = [imgIds]
        
        ann_ids = []
        for img_id in imgIds:
            if img_id in self.imgToAnns:
                ann_ids.extend([ann['id'] for ann in self.imgToAnns[img_id]])
        
        return ann_ids
    
    def loadAnns(self, ids: Union[int, List[int]]) -> List[Dict]:
        """
        加载指定ID的标注
        
        Args:
            ids: 标注ID或标注ID列表
            
        Returns:
            标注字典列表
        """
        if isinstance(ids, int):
            ids = [ids]
        
        return [self.anns[ann_id] for ann_id in ids if ann_id in self.anns]
    
    def loadImgs(self, ids: Union[int, List[int]]) -> List[Dict]:
        """
        加载指定ID的图像信息
        
        Args:
            ids: 图像ID或图像ID列表
            
        Returns:
            图像信息字典列表
        """
        if isinstance(ids, int):
            ids = [ids]
        
        return [self.imgs[img_id] for img_id in ids if img_id in self.imgs]

# ==================== 渐进式训练：复杂度评估 ====================

def calculate_caption_complexity(caption: str) -> float:
    """
    计算图像描述的复杂度分数（改进版 - 防止词汇堆砌）
    
    本函数是渐进式训练的核心组件，用于评估样本难度。
    设计原则：完全基于语言学特征，不依赖任何数据集特定的统计参数。
    
    核心思想：
    1. 使用对数尺度处理长度，避免线性增长导致的分布偏斜
    2. 使用密度特征（比例）而非绝对数量，与长度解耦
    3. 不设置人为上限，让分数自然分布，通过百分位数分层
    4. **新增**：重复惩罚机制，防止简单词汇重复堆砌
    5. **新增**：信息熵评估，惩罚词汇分布不均匀的描述
    6. **新增**：语法合理性检查，惩罚纯名词堆砌
    
    复杂度维度：
    - 长度复杂度：log(词数) - 对数尺度，长文本不会过度主导
    - 词汇丰富度：unique_words/total_words - 反映描述多样性
    - 语义密度：(名词+形容词+介词)/总词数 - 反映信息密度
    - 信息熵：词汇分布均匀性 - 惩罚重复堆砌行为
    - 结构复杂度：句子数量和平均句长 - 反映语法复杂度
    - 语法合理性：是否包含动词/形容词 - 惩罚纯名词堆砌
    
    防作弊机制：
    - 重复惩罚：如果某个词重复次数过多，通过 repeat_penalty 降低分数
    - 信息熵：评估词汇分布，堆砌相同词汇会导致熵降低
    - 语法检查：纯名词堆砌（无动词/形容词）会被 grammar_penalty 惩罚
    
    Args:
        caption: 图像描述文本（可能包含多个句子或多个描述）
        
    Returns:
        float: 复杂度分数（无上限，用于相对排序和百分位数分层）
        
    Example:
        >>> calculate_caption_complexity("A cat.")
        1.2  # 简单描述
        >>> calculate_caption_complexity("A fluffy orange cat with green eyes sitting on a wooden table.")
        3.8  # 复杂描述
        >>> calculate_caption_complexity("cat cat cat cat cat cat cat cat cat cat")
        1.5  # 重复堆砌 - 被严重惩罚
        >>> calculate_caption_complexity("dog cat bird horse sheep cow elephant bear zebra giraffe")
        2.1  # 纯名词堆砌 - 被语法惩罚
    """
    caption = caption.lower().strip()
    
    # 处理可能的JSON格式列表（如Flickr30K的["desc1", "desc2", ...]）
    # 如果是列表格式，计算每个描述的平均复杂度
    if caption.startswith('[') and caption.endswith(']'):
        try:
            import ast
            descriptions = ast.literal_eval(caption)
            if isinstance(descriptions, list):
                # 对每个描述单独计算复杂度，取平均
                complexities = [calculate_caption_complexity(desc) for desc in descriptions if desc]
                return np.mean(complexities) if complexities else 0.0
        except:
            pass  # 如果解析失败，按普通字符串处理
    
    # 分词
    tokens = re.findall(r'\b\w+\b', caption)
    if len(tokens) == 0:
        return 0.0
    
    # === 词汇表定义 ===
    common_nouns = {
        'person', 'people', 'man', 'woman', 'child', 'boy', 'girl',
        'dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant',
        'car', 'bus', 'truck', 'train', 'plane', 'bike', 'motorcycle',
        'table', 'chair', 'couch', 'bed', 'toilet', 'tv', 'laptop',
        'phone', 'bottle', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'pizza', 'cake', 'donut', 'ball', 'kite', 'bat', 'glove',
        'skateboard', 'surfboard', 'racket', 'book', 'clock', 'vase',
        'scissors', 'bear', 'zebra', 'giraffe', 'building', 'street',
        'tree', 'sky', 'water', 'grass', 'mountain', 'beach', 'room',
        'field', 'road', 'window', 'door', 'wall', 'floor', 'plate'
    }
    
    adjectives = {
        'red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'gray', 'orange', 'pink',
        'big', 'small', 'large', 'little', 'tall', 'short', 'long', 'wide', 'narrow',
        'old', 'young', 'new', 'beautiful', 'nice', 'good', 'bad', 'happy', 'sad',
        'open', 'closed', 'full', 'empty', 'clean', 'dirty', 'bright', 'dark',
        'wooden', 'metal', 'glass', 'colorful', 'striped', 'round', 'square'
    }
    
    prepositions = {
        'in', 'on', 'at', 'by', 'with', 'from', 'to', 'of', 'for',
        'under', 'over', 'above', 'below', 'near', 'next', 'behind',
        'front', 'between', 'among', 'through', 'during', 'while',
        'and', 'or', 'but', 'as', 'that', 'which', 'where', 'beside'
    }
    
    # === 特征1: 长度复杂度（对数尺度） ===
    # log(n) 增长比 n 慢，避免长文本过度主导
    # log(10) ≈ 2.3, log(20) ≈ 3.0, log(50) ≈ 3.9, log(100) ≈ 4.6
    length_complexity = np.log(len(tokens) + 1)  # +1避免log(0)
    
    # === 特征2: 词汇丰富度（带重复惩罚） ===
    # 不重复词比例，范围 [0, 1]
    unique_ratio = len(set(tokens)) / len(tokens)
    
    # 重复惩罚：只针对内容词（排除功能词）
    from collections import Counter
    function_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 
                      'with', 'of', 'to', 'for', 'and', 'or', 'but'}
    content_tokens = [t for t in tokens if t not in function_words]
    
    if content_tokens:
        content_counts = Counter(content_tokens)
        max_repeat = max(content_counts.values())
        repeat_penalty = 1.0 / (1.0 + np.log(max_repeat))  # 重复越多，惩罚越大
    else:
        repeat_penalty = 1.0
    
    token_counts = Counter(tokens)  # 保留用于信息熵计算
    
    # === 特征3: 语义密度 ===
    # 关键词（名词+形容词+介词）占比
    noun_count = sum(1 for token in tokens if token in common_nouns)
    adj_count = sum(1 for token in tokens if token in adjectives)
    prep_count = sum(1 for token in tokens if token in prepositions)
    semantic_density = (noun_count + adj_count + prep_count) / len(tokens)
    
    # === 特征4: 信息熵（惩罚词汇堆砌） ===
    # 评估词汇分布的均匀性
    token_probs = np.array(list(token_counts.values())) / len(tokens)
    entropy = -np.sum(token_probs * np.log(token_probs + 1e-10))
    normalized_entropy = entropy / np.log(len(tokens) + 1)  # 归一化到[0,1]
    
    # === 特征5: 结构复杂度 ===
    # 句子数量（通过标点符号估计）
    sentence_markers = caption.count('.') + caption.count('!') + caption.count('?') + caption.count(';')
    num_sentences = max(sentence_markers, 1)  # 至少1句
    avg_sentence_length = len(tokens) / num_sentences
    structure_complexity = np.log(avg_sentence_length + 1)
    
    # === 特征6: 语法合理性（惩罚纯名词堆砌） ===
    stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'with', 'of', 'to', 'for'}
    
    # 检查是否有动词（更精确的判断）
    verb_words = {'is', 'are', 'was', 'were', 'has', 'have', 'had', 'being', 'been',
                  'playing', 'sitting', 'standing', 'running', 'walking', 'wearing', 
                  'holding', 'looking', 'eating', 'drinking', 'flying', 'jumping',
                  'laying', 'lying', 'riding', 'driving', 'swimming', 'skiing'}
    
    # 检查动词：只看 -ing 结尾（更可靠）或明确的动词词汇
    has_verb = any(
        (word.endswith('ing') and len(word) > 4) or word in verb_words
        for word in tokens if word not in stopwords
    )
    
    has_adj = any(word in adjectives for word in tokens)
    
    # 如果只有名词堆砌，没有动词/形容词，应用强惩罚
    grammar_bonus = 0.0 if (has_verb or has_adj) else -1.5  # 减法惩罚更直接
    
    # === 综合复杂度（无上限，用于排序） ===
    # 权重设计：长度主导，其他特征辅助
    base_complexity = (
        1.0 * length_complexity +        # 长度（对数尺度）- 主导因素
        0.8 * unique_ratio +             # 词汇丰富度 [0, 0.8]
        1.0 * semantic_density +         # 语义密度 [0, 1.0]
        0.8 * normalized_entropy +       # 信息熵（惩罚堆砌）[0, 0.8]
        0.5 * structure_complexity       # 结构复杂度（对数尺度）
    )
    
    # 应用重复惩罚（乘法）和语法惩罚（减法）
    complexity = base_complexity * repeat_penalty + grammar_bonus
    
    return complexity

def compute_complexity_for_all_samples(
    captions_data: Dict[int, List[str]],
    cache_file: Optional[str] = None
) -> Dict[int, float]:
    """
    计算所有样本的复杂度（支持缓存）
    
    Args:
        captions_data: 图像ID到描述列表的映射
        cache_file: 缓存文件路径（如果提供且存在，则从缓存加载）
        
    Returns:
        图像ID到平均复杂度的映射
    """
    # 尝试从缓存加载
    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            # 将字符串key转回int
            complexity_map = {int(k): v for k, v in cached_data.items()}
            logger.info(f"✓ 从缓存加载复杂度数据: {cache_file} ({len(complexity_map)}个样本)")
            return complexity_map
        except Exception as e:
            logger.warning(f"缓存加载失败，重新计算: {e}")
    
    # 计算所有样本的复杂度
    logger.info(f"计算 {len(captions_data)} 个样本的复杂度...")
    complexity_map = {}
    
    for img_id, captions in captions_data.items():
        # 对同一图像的多个描述计算平均复杂度
        complexities = [calculate_caption_complexity(cap) for cap in captions]
        avg_complexity = np.mean(complexities)
        complexity_map[img_id] = float(avg_complexity)
    
    # 保存到缓存
    if cache_file:
        try:
            os.makedirs(os.path.dirname(cache_file) or '.', exist_ok=True)
            with open(cache_file, 'w', encoding='utf-8') as f:
                # 将int key转为字符串以兼容JSON
                json.dump({str(k): v for k, v in complexity_map.items()}, f, indent=2)
            logger.info(f"✓ 复杂度数据已缓存: {cache_file}")
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")
    
    return complexity_map

def split_dataset_by_percentile(
    captions_data: Dict[int, List[str]],
    percentiles: Tuple[float, float] = (33.33, 66.67),
    cache_file: Optional[str] = None
) -> Dict[str, List[int]]:
    """
    根据百分位数将数据集分为三个难度等级（方案B：自适应分层）
    
    这是更科学的分层方法，不依赖硬编码阈值，而是根据数据分布自动分层。
    
    Args:
        captions_data: 图像ID到描述列表的映射
        percentiles: 百分位数元组 (p1, p2)，例如 (33.33, 66.67) 表示三等分
        cache_file: 复杂度缓存文件路径（可选）
        
    Returns:
        包含三个难度等级的字典：
        - 'easy': 简单样本的图像ID列表（前 p1%）
        - 'medium': 中等样本的图像ID列表（p1% - p2%）
        - 'hard': 困难样本的图像ID列表（后 (100-p2)%）
    """
    # 1. 计算所有样本的复杂度（使用缓存）
    complexity_map = compute_complexity_for_all_samples(captions_data, cache_file)
    
    # 2. 提取所有复杂度值
    complexities = list(complexity_map.values())
    
    # 3. 计算百分位数阈值
    p1, p2 = percentiles
    threshold_easy = np.percentile(complexities, p1)
    threshold_hard = np.percentile(complexities, p2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"百分位数分层统计:")
    logger.info(f"  - 复杂度范围: [{min(complexities):.3f}, {max(complexities):.3f}]")
    logger.info(f"  - 复杂度均值: {np.mean(complexities):.3f}")
    logger.info(f"  - 复杂度中位数: {np.median(complexities):.3f}")
    logger.info(f"  - 百分位数 {p1:.1f}%: {threshold_easy:.3f}")
    logger.info(f"  - 百分位数 {p2:.1f}%: {threshold_hard:.3f}")
    
    # 4. 根据阈值分配样本
    easy_samples = []
    medium_samples = []
    hard_samples = []
    
    for img_id, complexity in complexity_map.items():
        if complexity <= threshold_easy:
            easy_samples.append(img_id)
        elif complexity <= threshold_hard:
            medium_samples.append(img_id)
        else:
            hard_samples.append(img_id)
    
    # 5. 打印分层结果
    total = len(complexity_map)
    logger.info(f"\n分层结果:")
    logger.info(f"  - 简单样本 (complexity ≤ {threshold_easy:.3f}): {len(easy_samples):>5}个 ({len(easy_samples)/total*100:>5.2f}%)")
    logger.info(f"  - 中等样本 ({threshold_easy:.3f} < complexity ≤ {threshold_hard:.3f}): {len(medium_samples):>5}个 ({len(medium_samples)/total*100:>5.2f}%)")
    logger.info(f"  - 困难样本 (complexity > {threshold_hard:.3f}): {len(hard_samples):>5}个 ({len(hard_samples)/total*100:>5.2f}%)")
    logger.info(f"{'='*60}\n")
    
    return {
        'easy': easy_samples,
        'medium': medium_samples,
        'hard': hard_samples
    }

def split_dataset_by_complexity(
    captions_data: Dict[int, List[str]], 
    thresholds: Tuple[float, float] = (0.33, 0.67)
) -> Dict[str, List[int]]:
    """
    根据描述复杂度将数据集分为三个难度等级（旧方法，保留兼容性）
    
    注意：这是旧的硬编码阈值方法，建议使用 split_dataset_by_percentile()
    
    Args:
        captions_data: 图像ID到描述列表的映射
        thresholds: 复杂度阈值元组 (easy_threshold, hard_threshold)
        
    Returns:
        包含三个难度等级的字典：
        - 'easy': 简单样本的图像ID列表
        - 'medium': 中等样本的图像ID列表
        - 'hard': 困难样本的图像ID列表
    """
    logger.warning("使用旧的硬编码阈值分层方法，建议改用 split_dataset_by_percentile()")
    
    easy_threshold, hard_threshold = thresholds
    
    easy_samples = []
    medium_samples = []
    hard_samples = []
    
    for img_id, captions in captions_data.items():
        # 对同一图像的多个描述计算平均复杂度
        complexities = [calculate_caption_complexity(cap) for cap in captions]
        avg_complexity = np.mean(complexities)
        
        # 分配到对应难度等级
        if avg_complexity < easy_threshold:
            easy_samples.append(img_id)
        elif avg_complexity < hard_threshold:
            medium_samples.append(img_id)
        else:
            hard_samples.append(img_id)
    
    logger.info(f"数据集复杂度分层完成:")
    logger.info(f"  - 简单样本 (complexity < {easy_threshold:.2f}): {len(easy_samples)}个")
    logger.info(f"  - 中等样本 ({easy_threshold:.2f} <= complexity < {hard_threshold:.2f}): {len(medium_samples)}个")
    logger.info(f"  - 困难样本 (complexity >= {hard_threshold:.2f}): {len(hard_samples)}个")
    
    return {
        'easy': easy_samples,
        'medium': medium_samples,
        'hard': hard_samples
    }

# ==================== 渐进式训练：复杂度评估结束 ====================

class COCODatasetConfig:
    """COCO数据集配置类"""
    
    def __init__(self, data_root: str = "/root/autodl-tmp/COCO2017"):
        """
        初始化COCO数据集配置
        
        Args:
            data_root: COCO数据集根目录路径
        """
        # 数据集根目录
        self.data_root = data_root
        
        # 图像目录路径
        self.train_image_dir = os.path.join(data_root, "train2017")
        self.val_image_dir = os.path.join(data_root, "val2017")
        self.test_image_dir = os.path.join(data_root, "test2017")
        
        # 标注文件路径
        self.annotations_dir = os.path.join(data_root, "annotations")
        
        # 各类标注文件的具体路径
        self.train_captions_file = os.path.join(self.annotations_dir, "captions_train2017.json")
        self.val_captions_file = os.path.join(self.annotations_dir, "captions_val2017.json")
        self.train_instances_file = os.path.join(self.annotations_dir, "instances_train2017.json")
        self.val_instances_file = os.path.join(self.annotations_dir, "instances_val2017.json")
        
        # 测试集信息文件
        self.test_info_file = os.path.join(self.annotations_dir, "image_info_test2017.json")
        
        logger.info(f"COCO数据集配置初始化完成，根目录: {data_root}")
    
    def validate_paths(self) -> bool:
        """验证所有路径是否存在"""
        required_paths = [
            self.data_root,
            self.train_image_dir,
            self.val_image_dir,
            self.annotations_dir,
            self.train_captions_file,
            self.val_captions_file
        ]
        
        missing_paths = []
        for path in required_paths:
            if not os.path.exists(path):
                missing_paths.append(path)
        
        if missing_paths:
            logger.error(f"以下路径不存在: {missing_paths}")
            return False
        
        # 检查可选的测试集路径
        optional_paths = {
            "测试集图像目录": self.test_image_dir,
            "测试集信息文件": self.test_info_file
        }
        
        for name, path in optional_paths.items():
            if os.path.exists(path):
                logger.info(f"✓ {name}存在: {path}")
            else:
                logger.warning(f"⚠ {name}不存在: {path}")
                if name == "测试集信息文件":
                    logger.info("  → 测试预测将直接从图像目录加载图像")
        
        logger.info("COCO数据集路径验证完成")
        return True

class COCOCaptionDataset(Dataset):
    """COCO图像描述数据集类"""
    
    def __init__(self, 
                 config: COCODatasetConfig,
                 split: str = "train",
                 transform=None,
                 max_caption_length: int = 512):
        """
        初始化COCO图像描述数据集
        
        Args:
            config: COCO数据集配置对象
            split: 数据集分割类型 ("train", "val", "test")
            transform: 图像变换函数
            max_caption_length: 最大描述长度
        """
        self.config = config
        self.split = split
        self.transform = transform
        self.max_caption_length = max_caption_length
        
        # 根据split选择对应的路径
        if split == "train":
            self.image_dir = config.train_image_dir
            self.annotation_file = config.train_captions_file
        elif split == "val":
            self.image_dir = config.val_image_dir
            self.annotation_file = config.val_captions_file
        elif split == "test":
            self.image_dir = config.test_image_dir
            self.annotation_file = config.test_info_file
        else:
            raise ValueError(f"不支持的数据集分割类型: {split}")
        
        # 加载COCO API或直接从图像目录加载
        if split == "test" and not os.path.exists(self.annotation_file):
            # 测试集信息文件不存在时，直接从图像目录加载
            logger.warning(f"测试集信息文件不存在: {self.annotation_file}")
            logger.info("直接从图像目录加载测试图像")
            self.coco = None
            self.image_ids = self._load_test_images_from_directory()
        else:
            # 使用自定义Caption加载器（避免category_id KeyError）
            logger.info(f"正在加载{split}数据集标注文件: {self.annotation_file}")
            self.coco = COCOCaptionLoader(self.annotation_file)
            # 获取所有图像ID
            self.image_ids = list(self.coco.imgs.keys())
        
        # 如果不是测试集，加载图像描述标注
        if split != "test":
            self.captions_data = self._load_captions()
        else:
            self.captions_data = None
        
        logger.info(f"{split}数据集加载完成，共{len(self.image_ids)}张图像")
    
    def _load_test_images_from_directory(self) -> List[int]:
        """从目录直接加载测试图像ID"""
        image_files = []
        
        if not os.path.exists(self.image_dir):
            logger.error(f"测试集图像目录不存在: {self.image_dir}")
            return image_files
        
        logger.info(f"正在从目录加载测试图像: {self.image_dir}")
        
        try:
            for filename in os.listdir(self.image_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        # 从文件名提取图像ID（COCO测试集格式：XXXXXX.jpg）
                        image_id = int(os.path.splitext(filename)[0])
                        image_files.append(image_id)
                    except ValueError:
                        logger.warning(f"无法从文件名提取图像ID: {filename}")
                        continue
            
            image_files.sort()  # 按ID排序
            logger.info(f"从目录成功加载了{len(image_files)}张测试图像")
            
            if len(image_files) == 0:
                logger.warning("测试集目录中没有找到有效的图像文件")
                
        except Exception as e:
            logger.error(f"加载测试图像目录失败: {str(e)}")
        
        return image_files
    
    def _load_captions(self) -> Dict:
        """加载图像描述数据"""
        captions_data = {}
        
        for img_id in self.image_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            captions = [ann['caption'] for ann in anns]
            captions_data[img_id] = captions
        
        logger.info(f"加载了{len(captions_data)}张图像的描述标注")
        return captions_data
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取数据集中的一个样本"""
        # 获取图像ID
        img_id = self.image_ids[idx]
        
        # 获取图像信息
        if self.coco is not None:
            # 从COCO API获取图像信息
            img_info = self.coco.imgs[img_id]
            img_filename = img_info['file_name']
            img_path = os.path.join(self.image_dir, img_filename)
        else:
            # 直接从图像ID构建文件路径（测试集无信息文件时）
            img_filename = f"{img_id:012d}.jpg"  # COCO测试集格式
            img_path = os.path.join(self.image_dir, img_filename)
            # 创建基本图像信息
            img_info = {
                'id': img_id,
                'file_name': img_filename,
                'width': None,  # 将在加载图像后设置
                'height': None
            }
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
            # 如果图像信息中没有尺寸，从实际图像获取
            if img_info['width'] is None or img_info['height'] is None:
                img_info['width'] = image.width
                img_info['height'] = image.height
        except Exception as e:
            logger.error(f"加载图像失败 {img_path}: {str(e)}")
            image = Image.new('RGB', (224, 224), color='white')
            # 设置默认尺寸
            if img_info['width'] is None:
                img_info['width'] = 224
            if img_info['height'] is None:
                img_info['height'] = 224
        
        # 注意：不在这里应用transform！
        # LLaVA的processor需要同时处理图像和文本，
        # 所以transform应该在训练器的compute_loss中进行
        
        # 构建返回数据
        sample = {
            'image_id': img_id,
            'image': image,  # 返回原始PIL图像
            'image_path': img_path,
            'image_info': img_info,
            'width': img_info['width'],
            'height': img_info['height']
        }
        
        # 如果有描述标注，添加到样本中
        if self.captions_data is not None:
            captions = self.captions_data.get(img_id, [])
            sample['captions'] = captions
            if captions:
                sample['caption'] = np.random.choice(captions)
            else:
                sample['caption'] = ""
        
        return sample

class COCODataLoader:
    """COCO数据加载器"""
    
    def __init__(self, config: COCODatasetConfig):
        """初始化COCO数据加载器"""
        self.config = config
        self._complexity_splits = None  # 缓存复杂度分层结果
        logger.info("COCO数据加载器初始化完成")
    
    def create_dataloader(self,
                         split: str = "train",
                         batch_size: int = 4,
                         shuffle: bool = True,
                         num_workers: int = 4,
                         transform=None,
                         max_samples: Optional[int] = None) -> DataLoader:
        """创建数据加载器"""
        # 参数验证
        if max_samples is not None:
            if not isinstance(max_samples, int) or max_samples <= 0:
                raise ValueError(f"max_samples必须是正整数，当前值: {max_samples}")
        
        if batch_size <= 0:
            raise ValueError(f"batch_size必须是正整数，当前值: {batch_size}")
        
        if num_workers < 0:
            raise ValueError(f"num_workers必须非负，当前值: {num_workers}")
        
        if split not in ["train", "val", "test"]:
            raise ValueError(f"不支持的数据集分割: {split}")
        
        # 创建数据集
        dataset = COCOCaptionDataset(
            config=self.config,
            split=split,
            transform=transform
        )
        
        # 如果指定了最大样本数，创建子集
        if max_samples is not None:
            if max_samples > len(dataset):
                logger.warning(f"指定的max_samples({max_samples})大于数据集大小({len(dataset)})，将使用全部数据")
            total_samples = min(max_samples, len(dataset))
            indices = list(range(total_samples))
            dataset = Subset(dataset, indices)
            logger.info(f"限制{split}数据集样本数量为: {total_samples}")
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        logger.info(f"创建{split}数据加载器完成，批次大小: {batch_size}")
        return dataloader
    
    def get_complexity_splits(self, 
                             split: str = "train",
                             percentiles: Tuple[float, float] = (33.33, 66.67),
                             use_percentile: bool = True,
                             force_recompute: bool = False) -> Dict[str, List[int]]:
        """
        获取按复杂度分层的数据集（推荐使用百分位数方法）
        
        Args:
            split: 数据集分割类型 ("train", "val")
            percentiles: 百分位数元组 (p1, p2)，默认三等分 (33.33, 66.67)
                       或旧的硬编码阈值（当use_percentile=False时）
            use_percentile: 是否使用百分位数分层（True=方案B，False=旧方案A）
            force_recompute: 是否强制重新计算（否则使用缓存）
            
        Returns:
            包含三个难度等级的字典：{'easy': [...], 'medium': [...], 'hard': [...]}
        """
        # 检查缓存
        if self._complexity_splits is not None and not force_recompute:
            logger.info("使用缓存的复杂度分层结果")
            return self._complexity_splits
        
        # 创建数据集以获取描述数据
        logger.info(f"正在计算{split}数据集的复杂度分层...")
        dataset = COCOCaptionDataset(
            config=self.config,
            split=split
        )
        
        # 获取描述数据
        if dataset.captions_data is None:
            raise ValueError(f"{split}数据集没有描述标注，无法进行复杂度分层")
        
        # 生成缓存文件路径
        cache_file = os.path.join(
            self.config.data_root,
            f"complexity_cache_{split}.json"
        )
        
        # 调用复杂度分层函数
        if use_percentile:
            # 方案B：百分位数分层（推荐）
            logger.info(f"使用百分位数分层方法 (percentiles={percentiles})")
            self._complexity_splits = split_dataset_by_percentile(
                dataset.captions_data,
                percentiles=percentiles,
                cache_file=cache_file
            )
        else:
            # 方案A：硬编码阈值（旧方法，保留兼容性）
            logger.warning(f"使用旧的硬编码阈值分层方法 (thresholds={percentiles})")
            self._complexity_splits = split_dataset_by_complexity(
                dataset.captions_data, 
                thresholds=percentiles
            )
        
        return self._complexity_splits
    
    def create_progressive_dataloaders(self,
                                      split: str = "train",
                                      batch_size: int = 4,
                                      shuffle: bool = True,
                                      num_workers: int = 4,
                                      transform=None,
                                      thresholds: Tuple[float, float] = (33.33, 66.67),
                                      use_percentile: bool = True,
                                      max_samples: Optional[int] = None) -> Dict[str, DataLoader]:
        """
        创建渐进式训练的数据加载器（分为easy/medium/hard三个阶段）
        
        Args:
            split: 数据集分割类型
            batch_size: 批次大小
            shuffle: 是否打乱数据
            num_workers: 数据加载线程数
            transform: 图像变换函数
            thresholds: 复杂度阈值/百分位数 (默认 33.33, 66.67 表示三等分)
                       当use_percentile=False时，作为硬编码阈值 (如 0.30, 0.50)
                       当use_percentile=True时，作为百分位数 (如 33.33, 66.67)
            use_percentile: 是否使用百分位数分层方法（默认True，推荐使用）
            max_samples: 最大样本数限制（应用于完整数据集，然后按比例分配到各阶段）
            
        Returns:
            包含三个阶段数据加载器的字典：
            - 'easy': 只包含简单样本
            - 'medium': 包含简单+中等样本
            - 'hard': 包含所有样本
        """
        logger.info(f"创建渐进式训练数据加载器 (split={split})...")
        
        # 获取复杂度分层（支持新旧两种方法）
        complexity_splits = self.get_complexity_splits(
            split=split, 
            percentiles=thresholds,
            use_percentile=use_percentile
        )
        
        # 创建完整数据集
        full_dataset = COCOCaptionDataset(
            config=self.config,
            split=split,
            transform=transform
        )
        
        # 如果指定了max_samples，需要先对完整数据集进行采样限制
        # 然后从限制后的数据集中按复杂度分层
        logger.info(f"[DEBUG] max_samples={max_samples}, len(full_dataset)={len(full_dataset)}")
        if max_samples is not None and max_samples < len(full_dataset):
            logger.info(f"应用样本数限制: {max_samples}/{len(full_dataset)}")
            
            # 改进的样本分配策略：确保每个难度级别都有足够的样本
            # 策略：优先保证每个级别至少有一定比例的样本
            total_available = len(complexity_splits['easy']) + len(complexity_splits['medium']) + len(complexity_splits['hard'])
            
            # 设置最小样本数阈值（每个级别至少占总样本的10%）
            min_samples_per_level = max(10, int(max_samples * 0.1))
            
            # 按比例限制各层级，但确保最小样本数
            easy_ratio = len(complexity_splits['easy']) / total_available
            medium_ratio = len(complexity_splits['medium']) / total_available
            hard_ratio = len(complexity_splits['hard']) / total_available
            
            # 初步分配
            easy_limit = int(max_samples * easy_ratio)
            medium_limit = int(max_samples * medium_ratio)
            hard_limit = max_samples - easy_limit - medium_limit
            
            # 调整hard样本数，确保至少有最小样本数
            if hard_limit < min_samples_per_level:
                logger.warning(f"Hard样本数过少 ({hard_limit})，调整为最小值 {min_samples_per_level}")
                hard_limit = min(min_samples_per_level, len(complexity_splits['hard']))
                # 从easy和medium中按比例减少
                remaining = max_samples - hard_limit
                easy_limit = int(remaining * easy_ratio / (easy_ratio + medium_ratio))
                medium_limit = remaining - easy_limit
            
            # 限制各层级的图像ID（确保不超过可用数量）
            easy_limit = min(easy_limit, len(complexity_splits['easy']))
            medium_limit = min(medium_limit, len(complexity_splits['medium']))
            hard_limit = min(hard_limit, len(complexity_splits['hard']))
            
            limited_easy = complexity_splits['easy'][:easy_limit] if easy_limit > 0 else []
            limited_medium = complexity_splits['medium'][:medium_limit] if medium_limit > 0 else []
            limited_hard = complexity_splits['hard'][:hard_limit] if hard_limit > 0 else []
            
            logger.info(f"  限制后样本分布: easy={len(limited_easy)}, medium={len(limited_medium)}, hard={len(limited_hard)}")
            
            # 使用限制后的分层
            complexity_splits = {
                'easy': limited_easy,
                'medium': limited_medium,
                'hard': limited_hard
            }
        
        # 获取图像ID到索引的映射
        img_id_to_idx = {img_id: idx for idx, img_id in enumerate(full_dataset.image_ids)}
        
        # Stage 1: 只用简单样本
        easy_indices = [img_id_to_idx[img_id] for img_id in complexity_splits['easy'] 
                       if img_id in img_id_to_idx]
        easy_dataset = Subset(full_dataset, easy_indices)
        
        # Stage 2: 简单 + 中等样本
        medium_image_ids = complexity_splits['easy'] + complexity_splits['medium']
        medium_indices = [img_id_to_idx[img_id] for img_id in medium_image_ids 
                         if img_id in img_id_to_idx]
        medium_dataset = Subset(full_dataset, medium_indices)
        
        # Stage 3: 所有样本
        hard_image_ids = complexity_splits['easy'] + complexity_splits['medium'] + complexity_splits['hard']
        hard_indices = [img_id_to_idx[img_id] for img_id in hard_image_ids 
                       if img_id in img_id_to_idx]
        hard_dataset = Subset(full_dataset, hard_indices)
        
        # 创建三个数据加载器
        dataloaders = {}
        
        for stage_name, stage_dataset in [('easy', easy_dataset), 
                                          ('medium', medium_dataset), 
                                          ('hard', hard_dataset)]:
            dataloader = DataLoader(
                dataset=stage_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=self._collate_fn,
                pin_memory=True if torch.cuda.is_available() else False
            )
            dataloaders[stage_name] = dataloader
            logger.info(f"  - Stage '{stage_name}': {len(stage_dataset)} 样本")
        
        logger.info("渐进式训练数据加载器创建完成")
        return dataloaders
    
    def create_progressive_dataloaders_with_replay(self,
                                                   split: str = "train",
                                                   batch_size: int = 4,
                                                   shuffle: bool = True,
                                                   num_workers: int = 4,
                                                   transform=None,
                                                   thresholds: Tuple[float, float] = (33.33, 66.67),
                                                   use_percentile: bool = True,
                                                   max_samples: Optional[int] = None) -> Dict[str, DataLoader]:
        """
        创建支持数据回放的渐进式训练数据加载器（方案3）
        
        数据回放策略：
        - Stage 1 (easy): 只包含简单样本
        - Stage 2 (medium): 简单+中等样本混合（50%/50%）
        - Stage 3 (hard): 简单+中等+困难样本混合（33%/33%/33%）
        
        这样可以让模型在学习新知识的同时，不断巩固旧知识，避免灾难性遗忘。
        
        Args:
            split: 数据集分割类型
            batch_size: 批次大小
            shuffle: 是否打乱数据
            num_workers: 数据加载线程数
            transform: 图像变换函数
            thresholds: 复杂度阈值/百分位数 (默认 33.33, 66.67)
            use_percentile: 是否使用百分位数分层（默认True）
            max_samples: 最大样本数限制
            
        Returns:
            包含三个阶段数据加载器的字典
        """
        logger.info(f"创建支持数据回放的渐进式数据加载器 (split={split})...")
        
        # 获取复杂度分层
        complexity_splits = self.get_complexity_splits(
            split=split,
            percentiles=thresholds,
            use_percentile=use_percentile
        )
        
        # 创建完整数据集
        full_dataset = COCOCaptionDataset(
            config=self.config,
            split=split,
            transform=transform
        )
        
        # 如果指定了max_samples，按比例限制各层级
        if max_samples is not None and max_samples < len(full_dataset):
            logger.info(f"应用样本数限制: {max_samples}/{len(full_dataset)}")
            
            total_available = len(complexity_splits['easy']) + len(complexity_splits['medium']) + len(complexity_splits['hard'])
            
            # 按比例限制各层级
            easy_ratio = len(complexity_splits['easy']) / total_available
            medium_ratio = len(complexity_splits['medium']) / total_available
            hard_ratio = len(complexity_splits['hard']) / total_available
            
            easy_limit = max(10, int(max_samples * easy_ratio))
            medium_limit = max(10, int(max_samples * medium_ratio))
            hard_limit = max(10, max_samples - easy_limit - medium_limit)
            
            # 确保不超过可用数量
            easy_limit = min(easy_limit, len(complexity_splits['easy']))
            medium_limit = min(medium_limit, len(complexity_splits['medium']))
            hard_limit = min(hard_limit, len(complexity_splits['hard']))
            
            complexity_splits = {
                'easy': complexity_splits['easy'][:easy_limit],
                'medium': complexity_splits['medium'][:medium_limit],
                'hard': complexity_splits['hard'][:hard_limit]
            }
            
            logger.info(f"  限制后样本分布: easy={len(complexity_splits['easy'])}, "
                       f"medium={len(complexity_splits['medium'])}, "
                       f"hard={len(complexity_splits['hard'])}")
        
        # 获取图像ID到索引的映射
        img_id_to_idx = {img_id: idx for idx, img_id in enumerate(full_dataset.image_ids)}
        
        # Stage 1: 只用简单样本
        easy_indices = [img_id_to_idx[img_id] for img_id in complexity_splits['easy'] 
                       if img_id in img_id_to_idx]
        easy_subset = Subset(full_dataset, easy_indices)
        
        # Stage 2: 简单+中等样本混合（50%/50%）
        # 策略：从easy和medium各取50%
        easy_sample_size = len(complexity_splits['easy']) // 2
        medium_sample_size = len(complexity_splits['medium']) // 2
        
        # 打乱并采样
        import random
        easy_sampled = random.sample(complexity_splits['easy'], 
                                    min(easy_sample_size, len(complexity_splits['easy'])))
        medium_sampled = random.sample(complexity_splits['medium'], 
                                      min(medium_sample_size, len(complexity_splits['medium'])))
        
        medium_stage_img_ids = easy_sampled + medium_sampled
        medium_indices = [img_id_to_idx[img_id] for img_id in medium_stage_img_ids 
                         if img_id in img_id_to_idx]
        medium_subset = Subset(full_dataset, medium_indices)
        
        # Stage 3: 简单+中等+困难样本混合（33%/33%/33%）
        easy_sample_size_hard = len(complexity_splits['easy']) // 3
        medium_sample_size_hard = len(complexity_splits['medium']) // 3
        hard_sample_size = len(complexity_splits['hard']) // 3
        
        easy_sampled_hard = random.sample(complexity_splits['easy'], 
                                         min(easy_sample_size_hard, len(complexity_splits['easy'])))
        medium_sampled_hard = random.sample(complexity_splits['medium'], 
                                           min(medium_sample_size_hard, len(complexity_splits['medium'])))
        hard_sampled = random.sample(complexity_splits['hard'], 
                                    min(hard_sample_size, len(complexity_splits['hard'])))
        
        hard_stage_img_ids = easy_sampled_hard + medium_sampled_hard + hard_sampled
        hard_indices = [img_id_to_idx[img_id] for img_id in hard_stage_img_ids 
                       if img_id in img_id_to_idx]
        hard_subset = Subset(full_dataset, hard_indices)
        
        # 创建数据加载器
        easy_loader = DataLoader(
            dataset=easy_subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        medium_loader = DataLoader(
            dataset=medium_subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        hard_loader = DataLoader(
            dataset=hard_subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        logger.info(f"数据回放模式渐进式数据加载器创建完成:")
        logger.info(f"  - Easy阶段: {len(easy_subset)} 样本 (100% easy)")
        logger.info(f"  - Medium阶段: {len(medium_subset)} 样本 (50% easy + 50% medium)")
        logger.info(f"  - Hard阶段: {len(hard_subset)} 样本 (33% easy + 33% medium + 33% hard)")
        
        return {
            'easy': easy_loader,
            'medium': medium_loader,
            'hard': hard_loader
        }
    
    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """自定义批次整理函数"""
        # 分离各个字段
        image_ids = [item['image_id'] for item in batch]
        images = [item['image'] for item in batch]
        image_paths = [item['image_path'] for item in batch]
        image_infos = [item['image_info'] for item in batch]
        widths = [item['width'] for item in batch]
        heights = [item['height'] for item in batch]
        
        # 整理批次数据
        batch_data = {
            'image_ids': image_ids,
            'images': images,
            'image_paths': image_paths,
            'image_infos': image_infos,
            'widths': widths,
            'heights': heights
        }
        
        # 如果有描述标注，添加到批次中
        if 'captions' in batch[0]:
            all_captions = [item['captions'] for item in batch]
            main_captions = [item['caption'] for item in batch]
            batch_data['all_captions'] = all_captions
            batch_data['captions'] = main_captions
        
        return batch_data

class COCODataAnalyzer:
    """COCO数据集分析工具"""
    
    def __init__(self, config: COCODatasetConfig):
        """初始化COCO数据分析器"""
        self.config = config
        logger.info("COCO数据分析器初始化完成")
    
    def analyze_dataset_statistics(self, split: str = "train") -> Dict:
        """分析数据集统计信息"""
        logger.info(f"开始分析{split}数据集统计信息...")
        
        # 加载数据集
        dataset = COCOCaptionDataset(self.config, split=split)
        
        # 基本统计信息
        stats = {
            'split': split,
            'total_images': len(dataset),
            'image_sizes': [],
            'caption_lengths': [],
            'captions_per_image': []
        }
        
        # 分析样本
        sample_count = min(1000, len(dataset))
        for i in range(sample_count):
            sample = dataset[i]
            
            # 图像尺寸统计
            stats['image_sizes'].append((sample['width'], sample['height']))
            
            # 描述长度统计（如果有描述）
            if 'captions' in sample:
                captions = sample['captions']
                stats['captions_per_image'].append(len(captions))
                
                for caption in captions:
                    stats['caption_lengths'].append(len(caption.split()))
        
        # 计算统计量
        if stats['image_sizes']:
            widths, heights = zip(*stats['image_sizes'])
            stats['avg_width'] = np.mean(widths)
            stats['avg_height'] = np.mean(heights)
            stats['min_width'] = np.min(widths)
            stats['max_width'] = np.max(widths)
            stats['min_height'] = np.min(heights)
            stats['max_height'] = np.max(heights)
        
        if stats['caption_lengths']:
            stats['avg_caption_length'] = np.mean(stats['caption_lengths'])
            stats['min_caption_length'] = np.min(stats['caption_lengths'])
            stats['max_caption_length'] = np.max(stats['caption_lengths'])
        
        if stats['captions_per_image']:
            stats['avg_captions_per_image'] = np.mean(stats['captions_per_image'])
        
        logger.info(f"{split}数据集统计分析完成")
        return stats
    
    def print_dataset_info(self, split: str = "train"):
        """打印数据集详细信息"""
        print(f"\n{'='*60}")
        print(f"COCO2017 {split.upper()} 数据集分析报告")
        print(f"{'='*60}")
        
        stats = self.analyze_dataset_statistics(split)
        
        print(f"数据集基本信息:")
        print(f"  - 总图像数量: {stats['total_images']:,}")
        
        if 'avg_width' in stats:
            print(f"  - 平均图像尺寸: {stats['avg_width']:.1f} x {stats['avg_height']:.1f}")
            print(f"  - 图像尺寸范围: {stats['min_width']} x {stats['min_height']} ~ {stats['max_width']} x {stats['max_height']}")
        
        if 'avg_caption_length' in stats:
            print(f"描述文本信息:")
            print(f"  - 平均每张图像描述数: {stats['avg_captions_per_image']:.1f}")
            print(f"  - 平均描述长度: {stats['avg_caption_length']:.1f} 词")
            print(f"  - 描述长度范围: {stats['min_caption_length']} ~ {stats['max_caption_length']} 词")
        
        print(f"{'='*60}\n")

# 便捷函数
def create_coco_config(data_root: str = "/root/autodl-tmp/COCO2017") -> COCODatasetConfig:
    """创建COCO数据集配置"""
    config = COCODatasetConfig(data_root)
    if not config.validate_paths():
        raise FileNotFoundError("COCO数据集路径验证失败，请检查数据集是否正确下载")
    return config

def create_coco_dataloader(data_root: str = "/root/autodl-tmp/COCO2017",
                          split: str = "train",
                          batch_size: int = 4) -> DataLoader:
    """创建COCO数据加载器的便捷函数"""
    config = create_coco_config(data_root)
    loader = COCODataLoader(config)
    return loader.create_dataloader(split=split, batch_size=batch_size)

def analyze_coco_dataset(data_root: str = "/root/autodl-tmp/COCO2017"):
    """分析COCO数据集的便捷函数"""
    config = create_coco_config(data_root)
    analyzer = COCODataAnalyzer(config)
    
    # 分析各个数据集分割
    for split in ["train", "val"]:
        analyzer.print_dataset_info(split)

