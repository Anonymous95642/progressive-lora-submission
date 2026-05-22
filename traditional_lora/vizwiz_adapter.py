"""
VizWiz-Captions 数据集适配器（Traditional LoRA 版本）

本文件是 `../vizwiz_adapter.py` 的轻量复制版本，目的是在 traditional_lora
子项目中提供与 `COCODatasetConfig` 兼容的 VizWiz 适配器，使 Traditional LoRA
训练 / 评估 / 测试可以直接使用 VizWiz-Captions。
"""

import os
import json
import logging
from typing import List, Optional
from PIL import Image

logger = logging.getLogger(__name__)

class VizWizCaptionAdapter:
    """与上级目录中的同名类保持接口一致，方便直接替换 COCODatasetConfig。"""

    def __init__(self, data_root: str = "/root/autodl-tmp/VizWiz-Captions"):
        self.data_root = data_root

        # 图片目录自动检测
        train_dir = os.path.join(data_root, "train")
        val_dir = os.path.join(data_root, "val")
        test_dir = os.path.join(data_root, "test")

        if not os.path.exists(train_dir) and os.path.exists(os.path.join(data_root, "Images", "train")):
            train_dir = os.path.join(data_root, "Images", "train")
        if not os.path.exists(val_dir) and os.path.exists(os.path.join(data_root, "Images", "val")):
            val_dir = os.path.join(data_root, "Images", "val")
        if not os.path.exists(test_dir) and os.path.exists(os.path.join(data_root, "Images", "test")):
            test_dir = os.path.join(data_root, "Images", "test")

        self.train_image_dir = train_dir
        self.val_image_dir = val_dir
        self.test_image_dir = test_dir

        # 标注与 COCO 格式缓存
        self.annotations_dir = os.path.join(data_root, "annotations")
        self.vizwiz_train_file = os.path.join(self.annotations_dir, "train.json")
        self.vizwiz_val_file = os.path.join(self.annotations_dir, "val.json")
        self.vizwiz_test_file = os.path.join(self.annotations_dir, "test.json")

        self.coco_format_dir = os.path.join(data_root, ".coco_format_cache")
        os.makedirs(self.coco_format_dir, exist_ok=True)

        self.train_captions_file = os.path.join(self.coco_format_dir, "captions_train_vizwiz.json")
        self.val_captions_file = os.path.join(self.coco_format_dir, "captions_val_vizwiz.json")
        self.test_captions_file = os.path.join(self.coco_format_dir, "captions_test_vizwiz.json")

        self.train_instances_file = None
        self.val_instances_file = None
        self.test_info_file = self.test_captions_file

        logger.info(f"[Traditional LoRA] VizWiz-Captions 适配器初始化完成，根目录: {data_root}")

        self._generate_coco_format_annotations()

    def validate_paths(self) -> bool:
        required_paths = [
            self.data_root,
            self.train_image_dir,
            self.val_image_dir,
            self.annotations_dir,
            self.train_captions_file,
            self.val_captions_file,
        ]
        missing = [p for p in required_paths if not os.path.exists(p)]
        if missing:
            logger.error(f"[Traditional LoRA] VizWiz 缺失关键路径: {missing}")
            return False

        optional = {
            "测试集图像目录": self.test_image_dir,
            "测试集标注/信息文件": self.test_info_file,
        }
        for name, path in optional.items():
            if path and os.path.exists(path):
                logger.info(f"✓ {name}存在: {path}")
            elif path:
                logger.warning(f"⚠ {name}不存在: {path}")

        logger.info("[Traditional LoRA] VizWiz-Captions 数据集路径验证完成")
        return True

    # ------- 内部：生成 COCO Caption JSON -------
    def _convert_single_split(self, split: str, src_file: str, dst_file: str) -> None:
        if not os.path.exists(src_file):
            logger.warning(f"[Traditional LoRA] VizWiz {split} 标注不存在，跳过: {src_file}")
            return
        try:
            with open(src_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"[Traditional LoRA] 读取 VizWiz {split} 标注失败: {e}")
            return

        images = data.get("images", [])
        annotations = data.get("annotations", [])
        
        if not images:
            logger.error(f"[Traditional LoRA] VizWiz {split} 缺少 images 字段")
            return

        # 对于测试集，允许没有 annotations（测试集通常只有图像元数据，没有标注）
        # 对于训练集和验证集，必须要有 annotations
        if not annotations:
            if split == "test":
                logger.info(f"[Traditional LoRA] VizWiz test 集没有 annotations（这是正常的，测试集通常只有图像元数据），将只生成 images 列表")
            else:
                logger.error(f"[Traditional LoRA] VizWiz {split} 缺少 annotations 字段（训练集/验证集必须有标注）")
                return

        # 补充 / 检查 category_id（仅当有 annotations 时）
        new_annotations = []
        if annotations:
            for ann in annotations:
                if "caption" not in ann:
                    continue
                new_ann = dict(ann)
                if "category_id" not in new_ann:
                    new_ann["category_id"] = 1
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
                            logger.error(f"[Traditional LoRA] ❌ 无法读取图像尺寸 {img_path}: {e}")
                            logger.error(f"[Traditional LoRA]    这将导致训练/评估错误，请检查图像文件")
                            # 不设置默认值，让后续代码处理（coco_dataset会在加载时再次尝试）
                            new_img["width"] = None
                            new_img["height"] = None
                    else:
                        missing_size_count += 1
                        logger.error(f"[Traditional LoRA] ❌ 图像文件不存在: {img_path}")
                        logger.error(f"[Traditional LoRA]    这将导致训练/评估错误，请检查文件路径")
                        new_img["width"] = None
                        new_img["height"] = None
                else:
                    missing_size_count += 1
                    logger.error(f"[Traditional LoRA] ❌ 图像信息缺少 file_name 字段，图像ID: {new_img.get('id', 'unknown')}")
                    logger.error(f"[Traditional LoRA]    VizWiz标注格式可能不正确")
                    new_img["width"] = None
                    new_img["height"] = None
            new_images.append(new_img)
        
        if missing_size_count > 0:
            logger.warning(f"[Traditional LoRA] ⚠️  警告：{missing_size_count} 张图像无法确定尺寸，可能影响训练/评估")

        categories = data.get("categories")
        if not categories:
            categories = [{"id": 1, "name": "image", "supercategory": "image"}]

        coco_format = {
            "images": new_images,
            "annotations": new_annotations,
            "categories": categories,
        }
        try:
            with open(dst_file, "w", encoding="utf-8") as f:
                json.dump(coco_format, f, indent=2, ensure_ascii=False)
            if split == "test" and not new_annotations:
                logger.info(
                    f"[Traditional LoRA] ✓ 生成 VizWiz {split} COCO Caption 文件: "
                    f"{dst_file} ({len(images)} 张图片, 无标注 - 测试集)"
                )
            else:
                logger.info(
                    f"[Traditional LoRA] ✓ 生成 VizWiz {split} COCO Caption 文件: "
                    f"{dst_file} ({len(images)} 张图片, {len(new_annotations)} 条描述)"
                )
        except Exception as e:
            logger.error(f"[Traditional LoRA] 写入 VizWiz {split} COCO Caption 失败: {e}")

    def _need_regenerate(self, json_file: str) -> bool:
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
                    f"[Traditional LoRA] 旧格式 VizWiz COCO Caption（缺少 category_id），重新生成: {json_file}"
                )
                return True
            # 检查 images 中是否有 width 和 height 字段
            images = data.get("images", [])
            if images:
                first_img = images[0]
                if "width" not in first_img or "height" not in first_img or first_img.get("width") is None or first_img.get("height") is None:
                    logger.warning(
                        f"[Traditional LoRA] 旧格式 VizWiz COCO Caption（缺少 width/height），重新生成: {json_file}"
                    )
                    return True
            logger.info(f"[Traditional LoRA] VizWiz COCO Caption 已存在且格式正确: {json_file}")
            return False
        except Exception as e:
            logger.warning(f"[Traditional LoRA] 读取 VizWiz COCO Caption 失败，将重新生成: {e}")
            return True

    def _generate_coco_format_annotations(self) -> None:
        logger.info("[Traditional LoRA] 正在生成 VizWiz-Captions COCO Caption 标注...")
        split_cfgs = [
            ("train", self.vizwiz_train_file, self.train_captions_file),
            ("val", self.vizwiz_val_file, self.val_captions_file),
            ("test", self.vizwiz_test_file, self.test_captions_file),
        ]
        for split, src, dst in split_cfgs:
            if self._need_regenerate(dst):
                self._convert_single_split(split, src, dst)
        logger.info("[Traditional LoRA] VizWiz-Captions COCO Caption 生成/校验完成")

    # ------- 可选调试接口 -------
    def get_image_path(self, filename: str, split: Optional[str] = None) -> str:
        if split is None:
            name = filename.lower()
            if "train" in name:
                split = "train"
            elif "val" in name:
                split = "val"
            elif "test" in name:
                split = "test"
            else:
                split = "val"
        if split == "train":
            base = self.train_image_dir
        elif split == "val":
            base = self.val_image_dir
        else:
            base = self.test_image_dir
        return os.path.join(base, filename)


