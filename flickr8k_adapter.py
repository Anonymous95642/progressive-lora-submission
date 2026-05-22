"""
Flickr8k 数据集适配器

将 Flickr8k 数据集转换为与 COCODatasetConfig 兼容的接口，
以便直接复用 COCO 训练 / 评估 / 测试流程。
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class Flickr8kAdapter:
    """提供与 COCODatasetConfig 一致的接口，方便替换使用。"""

    def __init__(self, data_root: str = "/root/autodl-tmp/Flickr8k"):
        # 根目录
        self.data_root = data_root

        # 图像目录（Flickr8k 图像直接位于根目录）
        self.image_dir = data_root
        self.train_image_dir = self.image_dir
        self.val_image_dir = self.image_dir
        self.test_image_dir = self.image_dir

        # 文本标注目录（兼容两种结构：有/无 Flickr8k_text 子目录）
        preferred_text_dir = os.path.join(data_root, "Flickr8k_text")
        tokens_in_preferred = os.path.join(preferred_text_dir, "Flickr8k.token.txt")
        tokens_in_root = os.path.join(data_root, "Flickr8k.token.txt")
        if os.path.exists(tokens_in_preferred):
            self.text_dir = preferred_text_dir
        else:
            self.text_dir = data_root

        self.tokens_file = os.path.join(self.text_dir, "Flickr8k.token.txt")
        self.train_list_file = os.path.join(self.text_dir, "Flickr_8k.trainImages.txt")
        self.val_list_file = os.path.join(self.text_dir, "Flickr_8k.devImages.txt")
        self.test_list_file = os.path.join(self.text_dir, "Flickr_8k.testImages.txt")

        # COCO 格式缓存目录
        self.coco_format_dir = os.path.join(data_root, ".coco_format_cache")
        os.makedirs(self.coco_format_dir, exist_ok=True)

        # COCO Caption 文件路径
        self.train_captions_file = os.path.join(self.coco_format_dir, "captions_train_flickr8k.json")
        self.val_captions_file = os.path.join(self.coco_format_dir, "captions_val_flickr8k.json")
        self.test_captions_file = os.path.join(self.coco_format_dir, "captions_test_flickr8k.json")

        # 兼容 COCODatasetConfig 的属性
        self.annotations_dir = self.coco_format_dir
        self.train_instances_file = None
        self.val_instances_file = None
        self.test_info_file = self.test_captions_file

        logger.info(f"Flickr8k 适配器初始化完成，根目录: {data_root}")

        # 解析原始 Flickr8k 文本标注与划分
        self.split_to_filenames, self.filename_to_captions = self._load_flickr8k_annotations()

        # 生成 COCO 格式标注
        self._generate_coco_format_annotations()

    def validate_paths(self) -> bool:
        """验证关键路径是否存在（兼容 COCODatasetConfig 接口）"""
        required_paths = [
            self.data_root,
            self.image_dir,
            self.text_dir,
            self.tokens_file,
            self.train_list_file,
            self.val_list_file,
            self.train_captions_file,
            self.val_captions_file,
        ]
        missing = [p for p in required_paths if not os.path.exists(p)]
        if missing:
            logger.error(f"Flickr8k 缺失关键路径: {missing}")
            return False

        optional = {
            "测试集图片列表": self.test_list_file,
            "测试集 COCO Caption 文件": self.test_captions_file,
        }
        for name, path in optional.items():
            if path and os.path.exists(path):
                logger.info(f"✓ {name}存在: {path}")
            elif path:
                logger.warning(f"⚠ {name}不存在: {path}")

        logger.info("Flickr8k 数据集路径验证完成")
        return True

    # ------- 内部：载入划分与标注 -------
    def _load_split_lists(self) -> Dict[str, List[str]]:
        split_to_file = {
            "train": self.train_list_file,
            "val": self.val_list_file,
            "test": self.test_list_file,
        }
        split_to_filenames: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

        for split, list_path in split_to_file.items():
            if not os.path.exists(list_path):
                logger.warning(f"Flickr8k {split} 划分文件不存在: {list_path}")
                continue
            try:
                with open(list_path, "r", encoding="utf-8") as f:
                    filenames = [line.strip() for line in f if line.strip()]
                split_to_filenames[split] = filenames
                logger.info(f"Flickr8k {split} 划分图片数: {len(filenames)}")
            except Exception as e:
                logger.error(f"读取 Flickr8k {split} 划分文件失败: {e}")

        return split_to_filenames

    def _load_tokens(self) -> Dict[str, List[str]]:
        filename_to_captions: Dict[str, List[str]] = {}
        if not os.path.exists(self.tokens_file):
            logger.error(f"Flickr8k 标注文件不存在: {self.tokens_file}")
            return filename_to_captions

        logger.info(f"正在解析 Flickr8k 标注文件: {self.tokens_file}")
        try:
            with open(self.tokens_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        left, caption = line.split("\t", 1)
                    except ValueError:
                        continue
                    if "#" in left:
                        filename, _ = left.split("#", 1)
                    else:
                        filename = left
                    caption = caption.strip()
                    if not caption:
                        continue
                    filename_to_captions.setdefault(filename, []).append(caption)
        except Exception as e:
            logger.error(f"解析 Flickr8k 标注文件失败: {e}")

        logger.info(f"共解析到 {len(filename_to_captions)} 张图片的描述")
        return filename_to_captions

    def _load_flickr8k_annotations(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        split_to_filenames = self._load_split_lists()
        filename_to_captions = self._load_tokens()
        return split_to_filenames, filename_to_captions

    # ------- COCO Caption 生成 -------
    def _need_regenerate(self, json_file: str) -> bool:
        if not os.path.exists(json_file):
            return True
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            anns = data.get("annotations", [])
            if not anns:
                return True
            if "category_id" not in anns[0]:
                logger.warning(
                    f"旧格式 Flickr8k COCO Caption（缺少 category_id），重新生成: {json_file}"
                )
                return True
            logger.info(f"Flickr8k COCO Caption 已存在且格式正确: {json_file}")
            return False
        except Exception as e:
            logger.warning(f"读取 Flickr8k COCO Caption 失败，将重新生成: {e}")
            return True

    def _generate_single_split(self, split: str, dst_file: str) -> None:
        filenames = self.split_to_filenames.get(split, [])
        if not filenames:
            logger.warning(f"Flickr8k {split} 划分为空，跳过生成: {dst_file}")
            return

        images: List[Dict] = []
        annotations: List[Dict] = []
        image_id = 1
        ann_id = 1

        for filename in filenames:
            captions = self.filename_to_captions.get(filename, [])
            if not captions:
                continue

            images.append(
                {
                    "id": image_id,
                    "file_name": filename,
                    "width": None,
                    "height": None,
                }
            )

            for caption in captions:
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "caption": caption,
                        "category_id": 1,
                    }
                )
                ann_id += 1

            image_id += 1

        if not images or not annotations:
            logger.warning(f"Flickr8k {split} 无有效数据，跳过写入: {dst_file}")
            return

        coco_format = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "image", "supercategory": "image"}],
        }

        try:
            with open(dst_file, "w", encoding="utf-8") as f:
                json.dump(coco_format, f, indent=2, ensure_ascii=False)
            logger.info(
                f"✓ 生成 Flickr8k {split} COCO Caption: "
                f"{dst_file} ({len(images)} 张图片, {len(annotations)} 条描述)"
            )
        except Exception as e:
            logger.error(f"写入 Flickr8k {split} COCO Caption 失败: {e}")

    def _generate_coco_format_annotations(self) -> None:
        logger.info("正在生成 Flickr8k COCO Caption 标注...")
        split_cfgs = [
            ("train", self.train_captions_file),
            ("val", self.val_captions_file),
            ("test", self.test_captions_file),
        ]
        for split, dst in split_cfgs:
            if self._need_regenerate(dst):
                self._generate_single_split(split, dst)
        logger.info("Flickr8k COCO Caption 生成/校验完成")

    # ------- 可选调试接口 -------
    def get_image_path(self, filename: str) -> str:
        return os.path.join(self.image_dir, filename)

    def get_split_images(self, split: str) -> List[str]:
        return list(self.split_to_filenames.get(split, []))

    def get_image_captions(self, filename: str) -> List[str]:
        return list(self.filename_to_captions.get(filename, []))


