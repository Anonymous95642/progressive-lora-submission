"""
=============================================================================
VizWiz-Captions 数据集适配器 (VizWiz Adapter)
=============================================================================

本模块实现了 VizWiz-Captions 数据集与 COCO 接口的适配，使其可以无缝接入
渐进式训练与评估流程（Progressive LoRA）。

【核心思路】
- VizWiz 标注本身已经是 COCO 风格：同时包含 "images" 和 "annotations" 两个列表，
  只是缺少 pycocotools.COCO 期望的 "category_id" 字段。
- 训练阶段的数据加载 (`COCOCaptionDataset` + `COCOCaptionLoader`) 不依赖 category_id，
  但评估阶段使用的 `pycocotools.COCO` 需要该字段。
- 因此，本适配器只做一件事：从原始 VizWiz JSON 读取数据，补充 `category_id=1`，
  并生成新的 COCO Caption 格式 JSON，供训练 / 评估 / 测试统一使用。

【目录结构假设】
默认假设 VizWiz-Captions 目录如下（与你给出的路径保持一致）::

    /root/autodl-tmp/VizWiz-Captions/
    ├── train/                      # 训练图片
    ├── val/                        # 验证图片
    ├── test/                       # 测试图片（如果有）
    └── annotations/
        ├── train.json              # 包含 images[] + annotations[]
        ├── val.json
        └── test.json               # 如果有测试标注

如果你的真实结构略有差异，只要 `data_root` 下能找到 `annotations/*.json`
以及对应的图片子目录，本适配器会进行简单的自动检测。

【生成的 COCO 格式缓存文件】
本模块会在 `data_root/.coco_format_cache/` 下生成以下文件：
- captions_train_vizwiz.json
- captions_val_vizwiz.json
- captions_test_vizwiz.json   （如果存在 test.json）

"""

import os
import json
import logging
from typing import Dict, List, Optional
from PIL import Image

logger = logging.getLogger(__name__)

class VizWizCaptionAdapter:
    """
    VizWiz-Captions 数据集适配器

    提供与 `COCODatasetConfig` 完全兼容的接口，让 VizWiz 可以无缝接入：
    - 渐进式训练 (`coco_trainer.COCOTrainer`)
    - 验证集评估 (`coco_evaluator.COCOCaptionEvaluator`)
    - 测试集预测 (`coco_test_predictor.COCOTestPredictor`)

    关键属性（与 COCODatasetConfig 对齐）:
    - self.train_image_dir / self.val_image_dir / self.test_image_dir
    - self.annotations_dir
    - self.train_captions_file / self.val_captions_file / self.test_captions_file
    - self.train_instances_file / self.val_instances_file / self.test_info_file
    """

    def __init__(self, data_root: str = "/root/autodl-tmp/VizWiz-Captions"):
        """
        初始化 VizWiz 适配器

        Args:
            data_root: VizWiz-Captions 数据集根目录
        """
        # 数据集根目录
        self.data_root = data_root

        # -------------------- 图片目录自动检测 --------------------
        # 常见结构 1：data_root/train, data_root/val, data_root/test
        train_dir = os.path.join(data_root, "train")
        val_dir = os.path.join(data_root, "val")
        test_dir = os.path.join(data_root, "test")

        # 也兼容 Images/train 之类的简单变体
        if not os.path.exists(train_dir) and os.path.exists(os.path.join(data_root, "Images", "train")):
            train_dir = os.path.join(data_root, "Images", "train")
        if not os.path.exists(val_dir) and os.path.exists(os.path.join(data_root, "Images", "val")):
            val_dir = os.path.join(data_root, "Images", "val")
        if not os.path.exists(test_dir) and os.path.exists(os.path.join(data_root, "Images", "test")):
            test_dir = os.path.join(data_root, "Images", "test")

        self.train_image_dir = train_dir
        self.val_image_dir = val_dir
        self.test_image_dir = test_dir

        # -------------------- 标注文件路径 --------------------
        # 原始 VizWiz 标注目录（与你项目中的 annotations/*.json 对应）
        self.annotations_dir = os.path.join(data_root, "annotations")
        self.vizwiz_train_file = os.path.join(self.annotations_dir, "train.json")
        self.vizwiz_val_file = os.path.join(self.annotations_dir, "val.json")
        self.vizwiz_test_file = os.path.join(self.annotations_dir, "test.json")

        # COCO 格式缓存目录
        self.coco_format_dir = os.path.join(data_root, ".coco_format_cache")
        os.makedirs(self.coco_format_dir, exist_ok=True)

        # 供 COCOCaptionDataset / COCOEvaluator 使用的 COCO Caption 文件
        self.train_captions_file = os.path.join(self.coco_format_dir, "captions_train_vizwiz.json")
        self.val_captions_file = os.path.join(self.coco_format_dir, "captions_val_vizwiz.json")
        self.test_captions_file = os.path.join(self.coco_format_dir, "captions_test_vizwiz.json")

        # 其他与 COCODatasetConfig 对齐的属性
        self.train_instances_file = None  # VizWiz-Captions 不涉及实例分割
        self.val_instances_file = None
        # 对于带有测试标注的情况，直接使用 captions 文件作为 test_info_file
        self.test_info_file = self.test_captions_file

        logger.info(f"VizWiz-Captions 适配器初始化完成，根目录: {data_root}")

        # 生成 COCO 格式标注文件（如果不存在或旧格式不含 category_id）
        self._generate_coco_format_annotations()

    # ------------------------------------------------------------------
    # 路径验证（供训练器 / 评估器在 setup_data 时调用）
    # ------------------------------------------------------------------
    def validate_paths(self) -> bool:
        """
        验证所有关键路径是否存在（兼容 COCODatasetConfig 接口）

        Returns:
            bool: 所有必需路径是否存在
        """
        required_paths = [
            self.data_root,
            self.train_image_dir,
            self.val_image_dir,
            self.annotations_dir,
            self.train_captions_file,
            self.val_captions_file,
        ]

        missing_paths = []
        for path in required_paths:
            if not os.path.exists(path):
                missing_paths.append(path)

        if missing_paths:
            logger.error(f"VizWiz 数据集缺失以下关键路径: {missing_paths}")
            return False

        # 测试集相关路径为可选
        optional_paths = {
            "测试集图像目录": self.test_image_dir,
            "测试集标注/信息文件": self.test_info_file,
        }
        for name, path in optional_paths.items():
            if path and os.path.exists(path):
                logger.info(f"✓ {name}存在: {path}")
            elif path:
                logger.warning(f"⚠ {name}不存在: {path}")
                if name == "测试集标注/信息文件":
                    logger.info("  → 测试预测阶段将仅生成描述，不计算 COCO 官方指标")

        logger.info("VizWiz-Captions 数据集路径验证完成")
        return True

    # ------------------------------------------------------------------
    # 核心：原始 VizWiz JSON -> COCO Caption JSON（补充 category_id）
    # ------------------------------------------------------------------
    def _convert_single_split(
        self,
        split: str,
        src_file: str,
        dst_file: str,
    ) -> None:
        """
        将单个 split 的 VizWiz JSON 转换为标准 COCO Caption JSON。

        Args:
            split: 'train' / 'val' / 'test'
            src_file: 原始 VizWiz JSON 路径
            dst_file: 目标 COCO Caption JSON 路径
        """
        if not os.path.exists(src_file):
            logger.warning(f"VizWiz {split} 标注文件不存在，跳过转换: {src_file}")
            return

        logger.info(f"正在转换 VizWiz {split} 标注为 COCO Caption 格式: {src_file}")

        try:
            with open(src_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"读取 VizWiz {split} 标注失败: {e}")
            return

        # 允许既支持完整 COCO-style 结构，也支持只有 annotations 的情况
        images = data.get("images", [])
        annotations = data.get("annotations", [])

        # 详细的诊断信息
        if not images:
            logger.error(
                f"VizWiz {split} 文件中未找到 'images' 字段，"
                f"当前 data keys: {list(data.keys())}"
            )
            logger.error(f"文件路径: {src_file}")
            logger.error(f"文件大小: {os.path.getsize(src_file) if os.path.exists(src_file) else 'N/A'} bytes")
            if isinstance(data, dict):
                logger.error(f"数据顶层键: {list(data.keys())}")
                # 尝试显示前几个键的内容类型
                for key in list(data.keys())[:3]:
                    val = data[key]
                    if isinstance(val, list):
                        logger.error(f"  - {key}: list, 长度={len(val)}")
                    elif isinstance(val, dict):
                        logger.error(f"  - {key}: dict, 键={list(val.keys())[:5]}")
                    else:
                        logger.error(f"  - {key}: {type(val).__name__}")
            return

        # 对于测试集，允许没有 annotations（测试集通常只有图像元数据，没有标注）
        # 对于训练集和验证集，必须要有 annotations
        if not annotations:
            if split == "test":
                logger.info(f"VizWiz test 集没有 annotations（这是正常的，测试集通常只有图像元数据），将只生成 images 列表")
            else:
                logger.error(
                    f"VizWiz {split} 文件中未找到 'annotations' 字段（训练集/验证集必须有标注），"
                    f"当前 data keys: {list(data.keys())}"
                )
                logger.error(f"文件路径: {src_file}")
                logger.error(f"images 数量: {len(images)}")
                if images:
                    logger.error(f"第一张图片的键: {list(images[0].keys()) if isinstance(images[0], dict) else 'N/A'}")
                return

        # 补充 / 检查 category_id（仅当有 annotations 时）
        new_annotations = []
        if annotations:
            for ann in annotations:
                # 仅保留标准 caption 相关字段，其他元数据（如 text_detected, is_precanned）可根据需要保留
                new_ann = dict(ann)
                if "caption" not in new_ann:
                    # 非 caption 标注，跳过
                    continue
                if "category_id" not in new_ann:
                    new_ann["category_id"] = 1  # 统一设为 1，便于 pycocotools 使用
                new_annotations.append(new_ann)

        # 补充 images 中的 width 和 height 字段（如果缺失）
        # 注意：必须从实际图像文件读取真实尺寸，不能使用默认值，否则可能导致训练/评估问题
        new_images = []
        image_dir = self.train_image_dir if split == "train" else (self.val_image_dir if split == "val" else self.test_image_dir)
        
        missing_size_count = 0
        for img in images:
            new_img = dict(img)  # 保留所有原始字段（file_name, vizwiz_url, id, text_detected等）
            
            # 如果缺少 width 或 height，必须从图像文件读取真实尺寸
            if "width" not in new_img or "height" not in new_img or new_img.get("width") is None or new_img.get("height") is None:
                file_name = new_img.get("file_name", "")
                if file_name:
                    img_path = os.path.join(image_dir, file_name)
                    if os.path.exists(img_path):
                        try:
                            with Image.open(img_path) as pil_img:
                                new_img["width"] = pil_img.width
                                new_img["height"] = pil_img.height
                        except Exception as e:
                            missing_size_count += 1
                            logger.error(f"❌ 无法读取图像尺寸 {img_path}: {e}")
                            logger.error(f"   这将导致训练/评估错误，请检查图像文件")
                            # 不设置默认值，让后续代码处理（coco_dataset会在加载时再次尝试）
                            new_img["width"] = None
                            new_img["height"] = None
                    else:
                        missing_size_count += 1
                        logger.error(f"❌ 图像文件不存在: {img_path}")
                        logger.error(f"   这将导致训练/评估错误，请检查文件路径")
                        new_img["width"] = None
                        new_img["height"] = None
                else:
                    missing_size_count += 1
                    logger.error(f"❌ 图像信息缺少 file_name 字段，图像ID: {new_img.get('id', 'unknown')}")
                    logger.error(f"   VizWiz标注格式可能不正确")
                    new_img["width"] = None
                    new_img["height"] = None
            new_images.append(new_img)
        
        if missing_size_count > 0:
            logger.warning(f"⚠️  警告：{missing_size_count} 张图像无法确定尺寸，可能影响训练/评估")

        # categories 字段：如果原始文件没有，就补一个最简单的
        categories = data.get("categories")
        if not categories:
            categories = [{"id": 1, "name": "image", "supercategory": "image"}]

        coco_format = {
            "images": new_images,
            "annotations": new_annotations,
            "categories": categories,
        }

        # 写入目标文件
        try:
            with open(dst_file, "w", encoding="utf-8") as f:
                json.dump(coco_format, f, indent=2, ensure_ascii=False)
            if split == "test" and not new_annotations:
                logger.info(
                    f"✓ 生成 VizWiz {split} COCO Caption 文件: {dst_file} "
                    f"({len(images)} 张图片, 无标注 - 测试集)"
                )
            else:
                logger.info(
                    f"✓ 生成 VizWiz {split} COCO Caption 文件: {dst_file} "
                    f"({len(images)} 张图片, {len(new_annotations)} 条描述)"
                )
        except Exception as e:
            logger.error(f"写入 COCO Caption 文件失败 ({split}): {e}")

    def _need_regenerate(self, json_file: str) -> bool:
        """
        判断现有 COCO Caption JSON 是否需要重新生成
        （例如旧版本缺少 category_id）。
        """
        if not os.path.exists(json_file):
            return True

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            anns = data.get("annotations", [])
            images = data.get("images", [])
            
            # 对于测试集，允许没有 annotations
            # 对于训练集/验证集，必须有 annotations
            is_test_file = "test" in json_file
            if not anns and not is_test_file:
                # 训练集/验证集必须有标注
                return True
            if anns and "category_id" not in anns[0]:
                logger.warning(
                    f"检测到旧格式的 VizWiz COCO Caption 文件（缺少 category_id），将重新生成: {json_file}"
                )
                return True
            # 检查 images 中是否有 width 和 height 字段
            images = data.get("images", [])
            if images:
                first_img = images[0]
                if "width" not in first_img or "height" not in first_img or first_img.get("width") is None or first_img.get("height") is None:
                    logger.warning(
                        f"检测到旧格式的 VizWiz COCO Caption 文件（缺少 width/height），将重新生成: {json_file}"
                    )
                    return True
            # 已包含 category_id 和 width/height，认为是新格式
            logger.info(f"VizWiz COCO Caption 文件已存在且格式正确: {json_file}")
            return False
        except Exception as e:
            logger.warning(f"读取 VizWiz COCO Caption 文件失败，将重新生成: {e}")
            return True

    def _generate_coco_format_annotations(self) -> None:
        """
        为 train/val/test 三个 split 生成（或更新） COCO Caption 格式标注文件。
        """
        logger.info("正在为 VizWiz-Captions 生成 COCO Caption 格式标注文件...")

        split_configs = [
            ("train", self.vizwiz_train_file, self.train_captions_file),
            ("val", self.vizwiz_val_file, self.val_captions_file),
            ("test", self.vizwiz_test_file, self.test_captions_file),
        ]

        for split, src, dst in split_configs:
            if self._need_regenerate(dst):
                self._convert_single_split(split, src, dst)

        logger.info("VizWiz-Captions COCO Caption 标注文件生成 / 校验完成")

    # ------------------------------------------------------------------
    # 可选的辅助方法（调试用，不在训练主流程中强依赖）
    # ------------------------------------------------------------------
    def get_image_path(self, filename: str, split: Optional[str] = None) -> str:
        """
        获取图片的完整路径（调试 / 可视化用）

        Args:
            filename: 图片文件名（如 "VizWiz_val_00007747.jpg"）
            split: 可选的 split 提示（'train' / 'val' / 'test'），
                   若未提供则根据前缀简单猜测。
        """
        if split is None:
            name_lower = filename.lower()
            if "train" in name_lower:
                split = "train"
            elif "val" in name_lower:
                split = "val"
            elif "test" in name_lower:
                split = "test"
            else:
                # 默认走 val 目录
                split = "val"

        if split == "train":
            base_dir = self.train_image_dir
        elif split == "val":
            base_dir = self.val_image_dir
        else:
            base_dir = self.test_image_dir

        return os.path.join(base_dir, filename)

    def get_split_images(self, split: str) -> List[str]:
        """
        获取指定 split 的所有图片文件名（基于 COCO Caption 文件）
        """
        if split == "train":
            ann_file = self.train_captions_file
        elif split == "val":
            ann_file = self.val_captions_file
        elif split == "test":
            ann_file = self.test_captions_file
        else:
            raise ValueError(f"不支持的 split: {split}")

        if not os.path.exists(ann_file):
            logger.warning(f"VizWiz {split} COCO Caption 文件不存在: {ann_file}")
            return []

        try:
            with open(ann_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [img.get("file_name", "") for img in data.get("images", [])]
        except Exception as e:
            logger.error(f"读取 VizWiz {split} COCO Caption 文件失败: {e}")
            return []

    def get_image_captions(self, image_id: int, split: str = "train") -> List[str]:
        """
        获取指定 image_id 在给定 split 下的所有描述（用于分析）
        """
        if split == "train":
            ann_file = self.train_captions_file
        elif split == "val":
            ann_file = self.val_captions_file
        elif split == "test":
            ann_file = self.test_captions_file
        else:
            raise ValueError(f"不支持的 split: {split}")

        if not os.path.exists(ann_file):
            logger.warning(f"VizWiz {split} COCO Caption 文件不存在: {ann_file}")
            return []

        try:
            with open(ann_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            captions = []
            for ann in data.get("annotations", []):
                if int(ann.get("image_id", -1)) == int(image_id) and ann.get("caption"):
                    captions.append(str(ann["caption"]))
            return captions
        except Exception as e:
            logger.error(f"读取 VizWiz {split} 描述失败: {e}")
            return []


