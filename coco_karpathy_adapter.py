"""
=============================================================================
COCO Karpathy Split 数据集适配器 (COCO Karpathy Adapter)
=============================================================================

本模块实现了COCO Karpathy Split数据集与COCO接口的适配，使其可以无缝接入渐进式训练流程。

【核心功能】
1. 数据集格式转换
   - 读取Karpathy格式的dataset_coco.json文件
   - 转换为COCO格式的JSON文件
   - 兼容COCOCaptionDataset的所有接口

2. 接口兼容
   - 提供与COCODatasetConfig完全一致的接口
   - 复用COCO的复杂度计算和分层逻辑
   - 无需修改训练器和评估器代码

3. 数据划分支持
   - train: 训练集（82,783张，来自train2014）
   - val: 验证集（5,000张，来自val2014）
   - test: 测试集（5,000张，来自val2014）
   - restval: 剩余的val2014图像（用于评估）

【数据集结构要求】
```
/root/autodl-tmp/coco2014/
├── train2014/                    # 训练集图片目录
│   ├── COCO_train2014_000000391895.jpg
│   └── ...（82,783张图片）
├── val2014/                      # 验证集图片目录
│   ├── COCO_val2014_000000391895.jpg
│   └── ...（40,504张图片）
└── dataset_coco.json             # Karpathy划分标注文件
```

【COCO Karpathy vs 标准COCO】
- 数据划分：Karpathy split 使用特定的train/val/test划分
- 测试集标注：Karpathy split的测试集有标注，可以直接评估
- 划分方式：从val2014中选出5,000张作为val，5,000张作为test

【使用示例】
```python
# 在run_progressive_training.py中使用
python run_progressive_training.py \\
    --dataset coco_karpathy \\
    --data_path /root/autodl-tmp/coco2014
```

【生成的COCO格式文件】
生成在 `.coco_format_cache/` 目录下：
- captions_train_coco_karpathy.json: 训练集标注
- captions_val_coco_karpathy.json: 验证集标注  
- captions_test_coco_karpathy.json: 测试集标注

"""

import os
import json
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# ==================== COCO Karpathy适配器 ====================

class COCOKarpathyAdapter:
    """
    COCO Karpathy Split数据集适配器
    
    提供与COCODatasetConfig完全兼容的接口，让COCO Karpathy Split可以无缝接入渐进式训练流程。
    """
    
    def __init__(self, data_root: str = "/root/autodl-tmp/coco2014"):
        """
        初始化COCO Karpathy适配器
        
        Args:
            data_root: COCO 2014数据集根目录（包含train2014、val2014和dataset_coco.json）
        """
        # 数据集根目录
        self.data_root = data_root
        
        # 图像目录路径
        self.train_image_dir = os.path.join(data_root, "train2014")
        self.val_image_dir = os.path.join(data_root, "val2014")
        self.test_image_dir = os.path.join(data_root, "val2014")  # test图片也在val2014目录中
        
        # Karpathy格式标注文件
        self.karpathy_annotation_file = os.path.join(data_root, "dataset_coco.json")
        
        # 为了兼容COCO接口，创建临时的COCO格式JSON文件
        self.coco_format_dir = os.path.join(data_root, ".coco_format_cache")
        os.makedirs(self.coco_format_dir, exist_ok=True)
        
        # 设置COCO格式的标注文件路径（兼容COCODatasetConfig接口）
        self.train_captions_file = os.path.join(self.coco_format_dir, "captions_train_coco_karpathy.json")
        self.val_captions_file = os.path.join(self.coco_format_dir, "captions_val_coco_karpathy.json")
        self.test_captions_file = os.path.join(self.coco_format_dir, "captions_test_coco_karpathy.json")
        
        # 兼容COCODatasetConfig的其他属性
        self.annotations_dir = self.coco_format_dir
        self.train_instances_file = None  # Karpathy split没有实例分割
        self.val_instances_file = None
        self.test_info_file = self.test_captions_file  # 使用test的captions文件
        
        logger.info(f"COCO Karpathy适配器初始化完成，根目录: {data_root}")
        
        # 加载Karpathy格式的标注
        self.karpathy_data = self._load_karpathy_annotations()
        logger.info(f"  - 加载图像数: {len(self.karpathy_data)}")
        logger.info(f"  - 总描述数: {sum(len(img['sentences']) for img in self.karpathy_data)}")
        
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
            self.karpathy_annotation_file,
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
        
        logger.info("COCO Karpathy数据集路径验证完成")
        return True
    
    def _load_karpathy_annotations(self) -> List[Dict]:
        """
        加载Karpathy格式的标注文件
        
        Returns:
            List[Dict]: 图像标注列表，每个元素包含图像信息和sentences
        """
        logger.info(f"正在加载COCO Karpathy标注文件: {self.karpathy_annotation_file}")
        
        try:
            with open(self.karpathy_annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Karpathy格式：{"images": [...]}
            if 'images' in data:
                images = data['images']
            elif isinstance(data, list):
                images = data
            else:
                raise ValueError(f"未知的JSON格式: {list(data.keys()) if isinstance(data, dict) else type(data)}")
            
            logger.info(f"成功加载 {len(images)} 张图像的标注")
            
            # 统计各split的图片数量
            split_counts = {}
            for img in images:
                split = img.get('split', 'unknown')
                split_counts[split] = split_counts.get(split, 0) + 1
            
            logger.info(f"各split图片数量: {split_counts}")
            
            return images
            
        except Exception as e:
            logger.error(f"加载标注文件失败: {str(e)}")
            raise
    
    def _generate_coco_format_annotations(self):
        """
        生成COCO格式的标注文件
        
        将Karpathy格式转换为COCO的JSON格式，以便复用COCO的数据加载逻辑。
        """
        logger.info("正在生成COCO格式的标注文件...")
        
        # 为每个split生成COCO格式的JSON文件
        splits = ['train', 'val', 'test']
        
        for split in splits:
            json_file = getattr(self, f'{split}_captions_file')
            
            # 检查文件是否已存在且格式正确
            need_regenerate = False
            if os.path.exists(json_file):
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
                
                # 收集该split的所有图片（Karpathy格式中split字段）
                split_images = []
                split_annotations = []
                image_id = 1
                annotation_id = 1
                
                for img_data in self.karpathy_data:
                    img_split = img_data.get('split', '')
                    
                    # 处理split映射
                    # Karpathy格式中：'train'对应训练集，'val'对应验证集，'test'对应测试集
                    # 'restval'是val2014中剩余的图像，我们可以在需要时使用
                    if split == 'train' and img_split == 'train':
                        pass  # 匹配
                    elif split == 'val' and img_split == 'val':
                        pass  # 匹配
                    elif split == 'test' and img_split == 'test':
                        pass  # 匹配
                    else:
                        continue  # 不匹配当前split
                    
                    # 获取图像信息
                    filename = img_data.get('filename', '')
                    filepath = img_data.get('filepath', '')
                    
                    # 构建图像信息
                    image_info = {
                        'id': image_id,
                        'file_name': filename,
                        'width': None,  # 将在加载图像时设置
                        'height': None,
                        'cocoid': img_data.get('cocoid', None)  # 保留原始COCO ID
                    }
                    split_images.append(image_info)
                    
                    # 获取sentences（caption列表）
                    sentences = img_data.get('sentences', [])
                    for sent_data in sentences:
                        # Karpathy格式的sentence有'raw'字段存储原始caption
                        caption = sent_data.get('raw', '').strip()
                        if not caption:
                            continue
                        
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
    
    def get_image_path(self, filename: str, split: str = None) -> str:
        """
        获取图片的完整路径
        
        Args:
            filename: 图片文件名
            split: 数据集分割（可选，用于确定在哪个目录查找）
            
        Returns:
            str: 图片的完整路径
        """
        # 根据文件名判断是在train2014还是val2014
        if 'train2014' in filename:
            return os.path.join(self.train_image_dir, filename)
        elif 'val2014' in filename:
            return os.path.join(self.val_image_dir, filename)
        else:
            # 默认在val2014中查找（因为test也在val2014目录）
            return os.path.join(self.val_image_dir, filename)
    
    def get_split_images(self, split: str) -> List[str]:
        """
        获取指定split的所有图片文件名
        
        Args:
            split: 数据集分割 ('train', 'val', 'test')
            
        Returns:
            List[str]: 图片文件名列表
        """
        split_images = []
        for img_data in self.karpathy_data:
            img_split = img_data.get('split', '')
            if split == 'train' and img_split == 'train':
                split_images.append(img_data.get('filename', ''))
            elif split == 'val' and img_split == 'val':
                split_images.append(img_data.get('filename', ''))
            elif split == 'test' and img_split == 'test':
                split_images.append(img_data.get('filename', ''))
        return split_images
    
    def get_image_captions(self, filename: str) -> List[str]:
        """
        获取指定图片的所有描述
        
        Args:
            filename: 图片文件名
            
        Returns:
            List[str]: 描述列表
        """
        for img_data in self.karpathy_data:
            if img_data.get('filename', '') == filename:
                sentences = img_data.get('sentences', [])
                return [sent.get('raw', '').strip() for sent in sentences if sent.get('raw', '').strip()]
        return []
    
    def print_dataset_info(self, split: str = "train"):
        """
        打印数据集详细信息（兼容COCODatasetConfig接口）
        
        Args:
            split: 数据集分割类型
        """
        print(f"\n{'='*60}")
        print(f"COCO Karpathy {split.upper()} 数据集分析报告")
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
        if len(split_images) > 0:
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

def create_coco_karpathy_config(data_root: str = "/root/autodl-tmp/coco2014") -> COCOKarpathyAdapter:
    """
    创建COCO Karpathy数据集配置的便捷函数
    
    Args:
        data_root: COCO 2014数据集根目录
    
    Returns:
        COCOKarpathyAdapter: 配置好的适配器对象
    """
    config = COCOKarpathyAdapter(data_root)
    if not config.validate_paths():
        raise FileNotFoundError("COCO Karpathy数据集路径验证失败，请检查数据集是否正确下载")
    return config

def analyze_coco_karpathy_dataset(data_root: str = "/root/autodl-tmp/coco2014"):
    """
    分析COCO Karpathy数据集的便捷函数
    
    Args:
        data_root: COCO 2014数据集根目录
    """
    config = create_coco_karpathy_config(data_root)
    
    # 分析各个数据集分割
    for split in ["train", "val", "test"]:
        config.print_dataset_info(split)


