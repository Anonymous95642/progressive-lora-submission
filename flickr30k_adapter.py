"""
=============================================================================
Flickr30K 数据集适配器 (Flickr30K Adapter)
=============================================================================

本模块实现了Flickr30K数据集与COCO接口的适配，使其可以无缝接入渐进式训练流程。

【核心功能】
1. 数据集格式转换
   - 读取Flickr30K原始数据（CSV格式标注）
   - 转换为COCO格式的JSON文件
   - 兼容COCOCaptionDataset的所有接口

2. 接口兼容
   - 提供与COCODatasetConfig完全一致的接口
   - 复用COCO的复杂度计算和分层逻辑
   - 无需修改训练器和评估器代码

3. 智能路径检测
   - 自动检测图片目录结构（双层/单层）
   - 兼容不同的数据集组织方式

【数据集结构要求】
```
/root/autodl-tmp/flickr30k/
├── flickr30k-images/              # 图片目录（单层或双层结构均可）
│   ├── 1000092795.jpg
│   ├── 10002456.jpg
│   └── ...（31,783张图片）
└── flickr_annotations_30k.csv     # CSV标注文件
    列: filename, split (train/val/test), raw (5条描述的列表)
```

【Flickr30K vs COCO】
- 数据规模：Flickr30K（~32K图片）vs COCO（~123K图片）
- 测试集标注：Flickr30K测试集有标注，COCO测试集无标注
- 评估方式：Flickr30K可以直接评估测试集，COCO需要提交到官方服务器

【使用示例】
```python
# 在run_progressive_training.py中使用
python run_progressive_training.py \\
    --dataset flickr30k \\
    --data_path /path/to/flickr30k
```

【生成的COCO格式文件】
生成在 `.coco_format_cache/` 目录下：
- captions_train_flickr30k.json: 训练集标注
- captions_val_flickr30k.json: 验证集标注  
- captions_test_flickr30k.json: 测试集标注

"""

import os
import json
import logging
from typing import Dict, List, Optional
from pathlib import Path
from PIL import Image
import pandas as pd

logger = logging.getLogger(__name__)

# ==================== Flickr30K适配器 ====================

class Flickr30KAdapter:
    """
    Flickr30K数据集适配器
    
    提供与COCODatasetConfig完全兼容的接口，让Flickr30K可以无缝接入渐进式训练流程。
    继承COCODatasetConfig的所有方法和属性，确保接口一致性。
    """
    
    def __init__(self, data_root: str = "/root/autodl-tmp/flickr30k"):
        """
        初始化Flickr30K适配器
        
        Args:
            data_root: Flickr30K数据集根目录
        """
        # 数据集根目录
        self.data_root = data_root
        
        # 智能检测图片目录（支持两种常见结构）
        # 结构1: flickr30k-images/flickr30k-images/ (双层)
        # 结构2: flickr30k-images/ (单层)
        image_dir_double = os.path.join(data_root, "flickr30k-images", "flickr30k-images")
        image_dir_single = os.path.join(data_root, "flickr30k-images")
        
        if os.path.exists(image_dir_double):
            self.image_dir = image_dir_double
            logger.info(f"检测到双层图片目录结构: {image_dir_double}")
        elif os.path.exists(image_dir_single):
            self.image_dir = image_dir_single
            logger.info(f"检测到单层图片目录结构: {image_dir_single}")
        else:
            # 默认使用双层结构，后续validate_paths会报错
            self.image_dir = image_dir_double
            logger.warning(f"未找到图片目录，使用默认路径: {image_dir_double}")
        
        # 设置图片目录路径（兼容COCODatasetConfig接口）
        self.train_image_dir = self.image_dir
        self.val_image_dir = self.image_dir
        self.test_image_dir = self.image_dir
        
        # 标注文件路径
        self.annotation_file = os.path.join(data_root, "flickr_annotations_30k.csv")
        
        # 为了兼容COCO接口，创建临时的COCO格式JSON文件
        self.coco_format_dir = os.path.join(data_root, ".coco_format_cache")
        os.makedirs(self.coco_format_dir, exist_ok=True)
        
        # 设置COCO格式的标注文件路径（兼容COCODatasetConfig接口）
        self.train_captions_file = os.path.join(self.coco_format_dir, "captions_train_flickr30k.json")
        self.val_captions_file = os.path.join(self.coco_format_dir, "captions_val_flickr30k.json")
        self.test_captions_file = os.path.join(self.coco_format_dir, "captions_test_flickr30k.json")
        
        # 兼容COCODatasetConfig的其他属性
        self.annotations_dir = self.coco_format_dir
        self.train_instances_file = None  # Flickr30K没有实例分割
        self.val_instances_file = None
        self.test_info_file = self.test_captions_file  # 使用test的captions文件
        
        logger.info(f"Flickr30K适配器初始化完成，根目录: {data_root}")
        
        # 加载标注
        self.annotations = self._load_annotations()
        logger.info(f"  - 加载图片数: {len(self.annotations)}")
        logger.info(f"  - 总描述数: {sum(len(ann['captions']) for ann in self.annotations.values())}")
        
        # 生成COCO格式的标注文件（如果不存在）
        self._generate_coco_format_annotations()
    
    def validate_paths(self) -> bool:
        """
        验证所有路径是否存在（兼容COCODatasetConfig接口）
        
        Returns:
            bool: 所有必需路径是否存在
        """
        required_paths = [
            self.data_root,
            self.train_image_dir,
            self.val_image_dir,
            self.annotation_file,
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
            if path and os.path.exists(path):
                logger.info(f"✓ {name}存在: {path}")
            elif path:
                logger.warning(f"⚠ {name}不存在: {path}")
                if name == "测试集信息文件":
                    logger.info("  → 测试预测将直接从图像目录加载图像")
        
        logger.info("Flickr30K数据集路径验证完成")
        return True
    
    def _load_annotations(self) -> Dict[str, Dict]:
        """
        加载Flickr30K标注文件
        
        Returns:
            Dict[str, Dict]: 文件名到标注信息的映射
        """
        logger.info(f"正在加载Flickr30K标注文件: {self.annotation_file}")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(self.annotation_file)
            logger.info(f"CSV文件列名: {list(df.columns)}")
            
            # 检查必需的列
            required_columns = ['filename', 'raw', 'split']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"CSV文件缺少必需的列: {missing_columns}")
            
            annotations = {}
            
            for _, row in df.iterrows():
                filename = row['filename']
                raw_captions = row['raw']
                split = row['split']
                
                # 解析captions（raw列存的是字符串形式的列表）
                # 例如: '["caption1", "caption2", ...]'
                try:
                    import ast
                    # 尝试解析为Python列表
                    captions_list = ast.literal_eval(raw_captions)
                    if not isinstance(captions_list, list):
                        captions_list = [str(raw_captions)]
                except (ValueError, SyntaxError):
                    # 如果解析失败，直接作为单个caption
                    captions_list = [str(raw_captions)]
                
                if filename not in annotations:
                    annotations[filename] = {
                        'filename': filename,
                        'captions': [],
                        'split': split
                    }
                
                # 添加所有captions
                annotations[filename]['captions'].extend(captions_list)
            
            logger.info(f"成功加载 {len(annotations)} 张图片的标注")
            
            # 统计各split的图片数量
            split_counts = {}
            for ann in annotations.values():
                split = ann['split']
                split_counts[split] = split_counts.get(split, 0) + 1
            
            logger.info(f"各split图片数量: {split_counts}")
            
            return annotations
            
        except Exception as e:
            logger.error(f"加载标注文件失败: {str(e)}")
            raise
    
    def _generate_coco_format_annotations(self):
        """
        生成COCO格式的标注文件
        
        将Flickr30K的CSV格式转换为COCO的JSON格式，以便复用COCO的数据加载逻辑。
        """
        logger.info("正在生成COCO格式的标注文件...")
        
        # 为每个split生成COCO格式的JSON文件
        splits = ['train', 'val', 'test']
        
        for split in splits:
            json_file = getattr(self, f'{split}_captions_file')
            
            # 检查文件是否已存在且格式正确
            need_regenerate = False
            if os.path.exists(json_file):
                # 验证文件格式是否包含category_id
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'annotations' in data and len(data['annotations']) > 0:
                            # 检查第一个annotation是否包含category_id
                            if 'category_id' not in data['annotations'][0]:
                                logger.warning(f"检测到旧格式的COCO文件（缺少category_id），将重新生成: {json_file}")
                                need_regenerate = True
                            else:
                                logger.info(f"COCO格式文件已存在且格式正确: {json_file}")
                                continue
                        else:
                            logger.info(f"COCO格式文件已存在: {json_file}")
                            continue
                except Exception as e:
                    logger.warning(f"读取COCO格式文件失败，将重新生成: {e}")
                    need_regenerate = True
            
            if not os.path.exists(json_file) or need_regenerate:
                logger.info(f"生成{split} split的COCO格式文件...")
                
                # 收集该split的所有图片
                split_images = []
                split_annotations = []
                image_id = 1
                annotation_id = 1
                
                for filename, ann_data in self.annotations.items():
                    if ann_data['split'] == split:
                        # 添加图像信息
                        image_info = {
                            'id': image_id,
                            'file_name': filename,
                            'width': None,  # 将在加载图像时设置
                            'height': None
                        }
                        split_images.append(image_info)
                        
                        # 添加标注信息
                        for caption in ann_data['captions']:
                            annotation_info = {
                                'id': annotation_id,
                                'image_id': image_id,
                                'caption': caption,
                                'category_id': 1  # 添加category_id以兼容pycocotools.COCO
                            }
                            split_annotations.append(annotation_info)
                            annotation_id += 1
                        
                        image_id += 1
                
                # 构建COCO格式的JSON结构
                coco_format = {
                    'images': split_images,
                    'annotations': split_annotations,
                    'categories': [{'id': 1, 'name': 'image', 'supercategory': 'image'}]
                }
                
                # 保存JSON文件
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(coco_format, f, indent=2, ensure_ascii=False)
                
                logger.info(f"✓ 生成{split} split: {len(split_images)}张图片, {len(split_annotations)}个标注")
        
        logger.info("COCO格式标注文件生成完成")
    
    def get_image_path(self, filename: str) -> str:
        """
        获取图片的完整路径
        
        Args:
            filename: 图片文件名
            
        Returns:
            str: 图片的完整路径
        """
        return os.path.join(self.image_dir, filename)
    
    def get_split_images(self, split: str) -> List[str]:
        """
        获取指定split的所有图片文件名
        
        Args:
            split: 数据集分割 ('train', 'val', 'test')
            
        Returns:
            List[str]: 图片文件名列表
        """
        split_images = []
        for filename, ann_data in self.annotations.items():
            if ann_data['split'] == split:
                split_images.append(filename)
        return split_images
    
    def get_image_captions(self, filename: str) -> List[str]:
        """
        获取指定图片的所有描述
        
        Args:
            filename: 图片文件名
            
        Returns:
            List[str]: 描述列表
        """
        if filename in self.annotations:
            return self.annotations[filename]['captions']
        return []
    
    def print_dataset_info(self, split: str = "train"):
        """
        打印数据集详细信息（兼容COCODatasetConfig接口）
        
        Args:
            split: 数据集分割类型
        """
        print(f"\n{'='*60}")
        print(f"Flickr30K {split.upper()} 数据集分析报告")
        print(f"{'='*60}")
        
        # 统计该split的图片
        split_images = self.get_split_images(split)
        total_captions = 0
        
        for filename in split_images:
            captions = self.get_image_captions(filename)
            total_captions += len(captions)
        
        print(f"数据集基本信息:")
        print(f"  - 总图像数量: {len(split_images):,}")
        print(f"  - 总描述数量: {total_captions:,}")
        print(f"  - 平均每张图像描述数: {total_captions/len(split_images):.1f}")
        
        # 统计描述长度
        caption_lengths = []
        for filename in split_images:
            captions = self.get_image_captions(filename)
            for caption in captions:
                caption_lengths.append(len(caption.split()))
        
        if caption_lengths:
            print(f"描述文本信息:")
            print(f"  - 平均描述长度: {sum(caption_lengths)/len(caption_lengths):.1f} 词")
            print(f"  - 描述长度范围: {min(caption_lengths)} ~ {max(caption_lengths)} 词")
        
        print(f"{'='*60}\n")

# ==================== 便捷函数 ====================

def create_flickr30k_config(data_root: str = "/root/autodl-tmp/flickr30k") -> Flickr30KAdapter:
    """
    创建Flickr30K数据集配置的便捷函数
    
    Args:
        data_root: Flickr30K数据集根目录
        
    Returns:
        Flickr30KAdapter: 配置好的适配器对象
    """
    config = Flickr30KAdapter(data_root)
    if not config.validate_paths():
        raise FileNotFoundError("Flickr30K数据集路径验证失败，请检查数据集是否正确下载")
    return config

def analyze_flickr30k_dataset(data_root: str = "/root/autodl-tmp/flickr30k"):
    """
    分析Flickr30K数据集的便捷函数
    
    Args:
        data_root: Flickr30K数据集根目录
    """
    config = create_flickr30k_config(data_root)
    
    # 分析各个数据集分割
    for split in ["train", "val", "test"]:
        config.print_dataset_info(split)
