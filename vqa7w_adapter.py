"""
Visual7W Telling (VQA7W) 数据集适配与评估工具

目录结构约定：
    data_root/
        images/                   # 原始图像目录（官方提供）
        dataset_v7w_telling/
            dataset_v7w_telling.json

本文件只新增功能，不修改任何现有逻辑，避免影响 COCO / Flickr / VizWiz 等流程。
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class VQA7WConfig:
    """VQA7W 路径配置（与现有 *Adapter 风格保持一致）"""

    def __init__(self, data_root: str = "/root/autodl-tmp/VQA7W"):
        self.data_root = data_root
        self.images_dir = os.path.join(data_root, "images")
        self.ann_dir = os.path.join(data_root, "dataset_v7w_telling")
        self.ann_file = os.path.join(self.ann_dir, "dataset_v7w_telling.json")

    def validate_paths(self) -> bool:
        ok = True
        if not os.path.exists(self.data_root):
            logger.error(f"VQA7W 数据根目录不存在: {self.data_root}")
            ok = False
        if not os.path.exists(self.images_dir):
            logger.error(f"VQA7W 图像目录不存在: {self.images_dir}")
            ok = False
        if not os.path.exists(self.ann_file):
            logger.error(f"VQA7W 标注文件不存在: {self.ann_file}")
            ok = False
        if ok:
            logger.info(f"VQA7W 路径检查通过: root={self.data_root}")
        return ok

class VQA7WAdapter:
    """
    VQA7W Telling 数据适配器

    将原始 JSON 结构：
        images -> qa_pairs
    展开为按 question (qa_id) 粒度的扁平样本列表。
    """

    def __init__(self, data_root: str = "/root/autodl-tmp/VQA7W"):
        self.config = VQA7WConfig(data_root)
        if not self.config.validate_paths():
            raise FileNotFoundError("VQA7W 数据集路径验证失败")

        self.samples: List[Dict[str, Any]] = []
        self._load_annotations()

    def _load_annotations(self) -> None:
        logger.info(f"加载 VQA7W Telling 标注: {self.config.ann_file}")
        with open(self.config.ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        images = data.get("images", [])
        for img in images:
            image_id = img.get("image_id")
            split = img.get("split", "train")
            filename = img.get("filename")
            img_path = os.path.join(self.config.images_dir, filename) if filename else None

            for qa in img.get("qa_pairs", []):
                qa_id = qa.get("qa_id")
                question = qa.get("question", "")
                answer = qa.get("answer", "")
                qtype = qa.get("type", "")
                choices = qa.get("multiple_choices", [])

                sample = {
                    "qa_id": qa_id,
                    "image_id": image_id,
                    "image_path": img_path,
                    "split": split,
                    "question": question,
                    "answer": answer,
                    "choices": choices,
                    "qtype": qtype,
                }
                self.samples.append(sample)

        logger.info(
            f"VQA7W 标注加载完成: {len(self.samples)} 个 QA 样本, "
            f"splits={{"
            f"train={sum(s['split']=='train' for s in self.samples)}, "
            f"val={sum(s['split']=='val' for s in self.samples)}, "
            f"test={sum(s['split']=='test' for s in self.samples)}}}"
        )

    def get_split(self, split: str) -> List[Dict[str, Any]]:
        split = split.lower()
        if split not in {"train", "val", "test"}:
            logger.warning(f"未知 split={split}，将返回全部样本")
            return list(self.samples)
        return [s for s in self.samples if s["split"] == split]

def build_vqa7w_questions_and_meta(
    samples: List[Dict[str, Any]]
) -> Tuple[Dict[int, List[str]], Dict[int, Dict[str, Any]]]:
    """
    将 VQA7W 样本构造为:
        - questions_dict: qa_id -> [question]
        - meta_map:       qa_id -> meta

    以便直接喂给 coco_dataset.calculate_vqa_complexity / compute_complexity_for_all_samples。
    """
    questions: Dict[int, List[str]] = {}
    meta_map: Dict[int, Dict[str, Any]] = {}

    for s in samples:
        qid = int(s["qa_id"])
        qtext = s.get("question", "") or ""
        questions[qid] = [qtext]

        answer = s.get("answer", "")
        qtype = (s.get("qtype") or "").lower()

        # 粗略推断 answer_type（与 VQAv2 标签保持类似接口）
        atype = "other"
        if qtype in {"yes", "no"} or answer.lower() in {"yes", "no"}:
            atype = "yes/no"
        elif any(ch.isdigit() for ch in answer):
            atype = "number"

        meta = {
            "answer": answer,
            "answer_type": atype,
            "question_type": qtype,
            # Visual7W Telling 没有多 annotator 信息，这里只提供单答案列表，
            # 让 _estimate_answer_count 至少能得到 1。
            "answers": [answer] if answer else [],
            # 没有 answer_freq / annotator_agreement 时可以留空，复杂度函数会回退到纯文本特征
        }
        meta_map[qid] = meta

    return questions, meta_map

class VQA7WDataset(Dataset):
    """
    简单的 VQA7W Dataset，用于训练/验证。

    为了避免影响现有逻辑，这里只提供最基础的接口，
    实际训练时可根据需要在外部封装 collate_fn / tokenizer 等。
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        transform=None,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        question = sample["question"]
        answer = sample["answer"]
        choices = sample.get("choices", [])

        image = Image.open(image_path).convert("RGB") if image_path and os.path.exists(image_path) else None
        if self.transform is not None and image is not None:
            image = self.transform(image)

        meta = {
            "qa_id": sample["qa_id"],
            "image_id": sample["image_id"],
            "answer": answer,
            "answer_type": None,
            "question_type": sample.get("qtype"),
            "choices": choices,
            "answers": [answer] if answer else [],
        }

        return {
            "image": image,
            "question": question,
            "answer": answer,
            "choices": choices,
            "meta": meta,
        }

def _normalize_answer_standard(text: str) -> str:
    """
    VQAv2官方标准答案归一化函数：
    
    参考标准：
    - VQAv2官方评估脚本（visualqa.org）
    - 完全符合VQAv2官方评估标准
    
    VQAv2官方归一化规则：
    1. 将所有字符转换为小写
    2. 删除句号，除非它出现在小数中
    3. 将数字单词转换为数字（例如，"two" → "2"）
    4. 去除冠词（a、an、the）
    5. 去除额外的空格
    6. 将连字符替换为空格
    7. 将撇号前后的空格去除
    8. 将"&"替换为"and"
    9. 将"n'"替换为"and"
    10. 添加撇号，如果缺少缩写（例如，将"dont"转换为"don't"）
    11. 用空格替换所有标点符号（除了撇号和冒号）
    """
    import re

    if not text:
        return ""
    
    # 1. 转换为小写并去除首尾空白
    text = text.strip().lower()
    
    # 2. 处理特殊字符：将"&"替换为"and"
    text = text.replace("&", "and")
    
    # 3. 先处理缩写（添加撇号），确保"can't"、"won't"等先有撇号
    # 这样后续的"n'"替换就不会影响这些缩写
    # 4. 添加撇号，如果缺少缩写（常见缩写）
    # 处理常见的缺少撇号的情况
    # 注意：使用单词边界确保精确匹配，避免误匹配
    contractions_map = {
        r"\bdont\b": "don't",
        r"\bcant\b": "can't",
        r"\bwont\b": "won't",
        r"\bisnt\b": "isn't",
        r"\barent\b": "aren't",
        r"\bwasnt\b": "wasn't",
        r"\bwerent\b": "weren't",
        r"\bhasnt\b": "hasn't",
        r"\bhavent\b": "haven't",
        r"\bhadnt\b": "hadn't",
        r"\bwouldnt\b": "wouldn't",
        r"\bcouldnt\b": "couldn't",
        r"\bshouldnt\b": "shouldn't",
        r"\bmustnt\b": "mustn't",
        r"\bdoesnt\b": "doesn't",
        r"\bdidnt\b": "didn't",
        r"\bim\b": "i'm",
        r"\byoure\b": "you're",
        r"\bhes\b": "he's",
        r"\bshes\b": "she's",
        r"\bits\b": "it's",
        r"\bwere\b": "we're",  # 注意：这可能与"were"（过去式）冲突，但VQAv2标准要求此转换
        r"\btheyre\b": "they're",
        r"\byouve\b": "you've",
        r"\bive\b": "i've",
        r"\bweve\b": "we've",
        r"\btheyve\b": "they've",
        r"\byoud\b": "you'd",
        r"\bhed\b": "he'd",
        r"\bshed\b": "she'd",
        r"\bwed\b": "we'd",
        r"\btheyd\b": "they'd",
        r"\byoull\b": "you'll",
        r"\bhell\b": "he'll",
        r"\bshell\b": "she'll",
        r"\bitll\b": "it'll",
        r"\bwell\b": "we'll",
        r"\btheyll\b": "they'll",
    }
    for pattern, replacement in contractions_map.items():
        text = re.sub(pattern, replacement, text)
    
    # 4. 处理"n'"：将独立的"n'"替换为"and"（如"rock 'n' roll"）
    # 注意：必须在处理缩写之后，只替换独立的"n'"，不能替换缩写中的"n'"（如"can't"中的"n'"）
    # 匹配模式：空格或开头 + 'n' + 空格或结尾（如"rock 'n' roll"）
    # 或者：空格或开头 + n' + 空格或结尾
    text = re.sub(r"(\s|^)'n'(\s|$)", r"\1and\2", text)
    text = re.sub(r"(\s|^)n'(\s|$)", r"\1and\2", text)
    
    # 5. 删除句号，除非它出现在小数中（保留小数点）
    # 先保护小数中的句号
    text = re.sub(r'\.(\d)', r'<DOT>\1', text)
    # 删除其他所有句号
    text = text.replace('.', '')
    # 恢复小数中的句号
    text = text.replace('<DOT>', '.')
    
    # 6. 将连字符替换为空格
    text = text.replace('-', ' ')
    
    # 7. 用空格替换所有标点符号（除了撇号和冒号）
    # 保留撇号和冒号，其他标点替换为空格
    text = re.sub(r"[^\w\s':]", " ", text)
    
    # 8. 将撇号前后的空格去除（确保撇号紧贴单词）
    text = re.sub(r"\s+'\s+", "'", text)
    text = re.sub(r"\s+'", "'", text)
    text = re.sub(r"'\s+", "'", text)
    
    # 9. 数字词映射（将数字单词转换为数字）
    num_map = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
        "thirteen": "13",
        "fourteen": "14",
        "fifteen": "15",
        "sixteen": "16",
        "seventeen": "17",
        "eighteen": "18",
        "nineteen": "19",
        "twenty": "20",
        "thirty": "30",
        "forty": "40",
        "fifty": "50",
        "sixty": "60",
        "seventy": "70",
        "eighty": "80",
        "ninety": "90",
        "hundred": "100",
        "thousand": "1000",
    }
    
    # 分词并处理数字词
    tokens = text.split()
    normalized_tokens = []
    for tok in tokens:
        # 检查是否是数字词
        if tok.lower() in num_map:
            normalized_tokens.append(num_map[tok.lower()])
        else:
            normalized_tokens.append(tok)
    
    text = " ".join(normalized_tokens)
    
    # 10. 去除冠词（a、an、the）
    tokens = text.split()
    tokens = [tok for tok in tokens if tok.lower() not in {"a", "an", "the"}]
    
    # 11. 去除额外的空格并重新组合
    text = " ".join(tokens)
    text = re.sub(r'\s+', ' ', text)  # 将多个空格替换为单个空格
    text = text.strip()
    
    return text

def _normalize_answer_moderate(text: str) -> str:
    """
    中等严格度的答案归一化函数（适用于VQA7W等数据集）：
    
    此函数在标准归一化和宽松归一化之间取得平衡：
    - 保留基本的文本归一化（小写、去标点、去冠词）
    - 只进行最必要的同义词映射（处理明显的变体，如tv/television）
    - 不进行数字词映射（保持数字的原始形式）
    
    适用场景：
    - VQA7W等数据集，其中答案可能有常见的变体表达
    - 需要平衡评估严格性和实用性
    - 当标准归一化过于严格时的折中方案
    """
    import re

    if not text:
        return ""
    
    # 转换为小写并去除首尾空白
    text = text.strip().lower()
    
    # 去除标点符号（保留字母、数字、连字符和空格）
    text = re.sub(r"[^\w\s-]", "", text)
    
    # 分词
    tokens = text.split()
    
    # 去除冠词
    tokens = [tok for tok in tokens if tok not in {"a", "an", "the"}]
    
    # 只进行最必要的同义词映射（处理明显的变体）
    # 注意：这个列表应该保持最小，只包含真正必要的映射
    minimal_syn_map = {
        "tv": "television",
        "t.v.": "television",
        "t v": "television",  # 处理空格变体
    }
    
    norm_tokens = []
    for tok in tokens:
        if tok in minimal_syn_map:
            norm_tokens.append(minimal_syn_map[tok])
        else:
            norm_tokens.append(tok)
    
    return " ".join(norm_tokens)

def _normalize_answer_legacy(text: str) -> str:
    """
    旧版答案归一化函数（包含同义词映射和数字词映射）：
    
    注意：此函数不符合标准VQA评估方法，因为：
    1. 同义词映射（tv↔television, bike↔bicycle等）会使评估过于宽松
    2. 数字词映射（one→1等）可能不符合标准做法
    
    保留此函数仅用于向后兼容或特殊需求。
    如需符合标准，请使用 _normalize_answer_standard。
    """
    import re

    if not text:
        return ""
    t = text.strip().lower()

    # 去掉简单标点
    t = re.sub(r"[.,!?;:]", " ", t)

    # 简单 tokenize
    tokens = t.split()

    # 去掉冠词
    tokens = [tok for tok in tokens if tok not in {"a", "an", "the"}]

    # 数字词映射
    num_map = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
    }

    # 简单同义词映射（覆盖常见多选项与口语变体）
    syn_map = {
        "tv": "television",
        "t.v.": "television",
        "cellphone": "phone",
        "cell": "phone",
        "cell phone": "phone",
        "mobile": "phone",
        "smartphone": "phone",
        "bike": "bicycle",
        "bikes": "bicycle",
        "cycle": "bicycle",
        "airplane": "plane",
        "aeroplane": "plane",
        "car": "automobile",
        "autos": "automobile",
        "sofa": "couch",
        "couch": "sofa",
    }

    norm_tokens = []
    for tok in tokens:
        if tok in num_map:
            norm_tokens.append(num_map[tok])
        elif tok in syn_map:
            norm_tokens.append(syn_map[tok])
        else:
            norm_tokens.append(tok)

    return " ".join(norm_tokens)

def evaluate_vqa7w_accuracy(
    predictions: Dict[int, str],
    samples: List[Dict[str, Any]],
    normalization_mode: str = "standard",
) -> float:
    """
    Visual7W Telling 的准确率评估函数
    
    Args:
        predictions: 预测答案字典，key为qa_id，value为预测答案字符串
        samples: 样本列表，每个样本包含qa_id和answer字段
        normalization_mode: 归一化模式（默认"standard"，符合VQAv2官方标准）
            - "standard": VQAv2官方标准归一化（完全符合VQAv2官方评估脚本，包含数字词映射等）
            - "moderate": 中等严格度（只进行必要的同义词映射，如tv↔television）
            - "legacy": 旧版归一化（最宽松，包含大量同义词映射和数字词映射）
    
    评估方法：
        - 每个 qa_id 只有一个标准答案 answer
        - 使用完全匹配（exact match）判定正确性
        - 比较前进行答案归一化
        
    注意：
        - 默认使用 "standard" 模式，完全符合VQAv2官方评估标准（包含数字词映射等）
        - "moderate" 模式可能对某些数据集更实用，但不符合VQAv2官方标准
        - "legacy" 模式过于宽松，不推荐用于论文发表
        - multiple_choices 是干扰项，不应作为正确答案的判断依据
    
    此函数不依赖任何训练逻辑，不会影响现有 COCO/Flickr/VizWiz evaluator。
    """
    # 选择归一化函数
    if normalization_mode == "standard":
        normalize_func = _normalize_answer_standard
        norm_type = "标准"
    elif normalization_mode == "moderate":
        normalize_func = _normalize_answer_moderate
        norm_type = "中等"
    elif normalization_mode == "legacy":
        normalize_func = _normalize_answer_legacy
        norm_type = "旧版"
    else:
        logger.warning(f"未知的归一化模式 {normalization_mode}，使用中等模式")
        normalize_func = _normalize_answer_moderate
        norm_type = "中等"
    
    correct = 0
    total = len(samples)

    if total == 0:
        logger.warning("VQA7W 评估时样本为空，返回 0.0")
        return 0.0

    for s in samples:
        qid = int(s["qa_id"])
        pred_raw = predictions.get(qid, "")

        gt = normalize_func(s.get("answer", "") or "")
        pred = normalize_func(pred_raw or "")

        # 判定规则（标准VQA做法）：
        # 1) 预测缺失视为错误
        # 2) 预测与标准答案（answer）完全匹配才算正确
        # 注意：multiple_choices 是干扰项，不应作为正确答案的判断依据
        is_correct = False
        if pred and gt and pred == gt:
            is_correct = True

        if is_correct:
            correct += 1

    acc = correct / total
    logger.info(f"VQA7W Accuracy ({norm_type}归一化): {acc:.4f} ({correct}/{total})")
    return float(acc)

def evaluate_vqa_accuracy_v2(
    predictions: Dict[int, str],
    samples: List[Dict[str, Any]],
    normalization_mode: str = "standard",
) -> float:
    """
    标准VQAv2评估方法（支持多标注者答案）
    
    此函数实现了VQAv2的标准评估方法：
    - 使用 min(1, num_agree/3) 计算每个问题的得分
    - 其中 num_agree 是同意预测答案的标注者数量
    - 如果至少3个标注者同意，得分为1.0
    
    Args:
        predictions: 预测答案字典，key为qa_id，value为预测答案字符串
        samples: 样本列表，每个样本应包含：
            - qa_id: 问题ID
            - answer: 单个标准答案（如果只有一个答案）
            - answers: 多个标注者的答案列表（如果有多个标注者）
        normalization_mode: 归一化模式（默认"standard"，符合VQAv2官方标准）
            - "standard": 标准归一化（符合VQAv2官方标准，主流论文使用）
            - "moderate": 中等严格度（只进行必要的同义词映射）
            - "legacy": 旧版归一化（最宽松，不推荐）
    
    Returns:
        float: 平均准确率（0.0-1.0）
    
    注意：
        - 如果样本有 answers 字段（列表），使用多标注者评估
        - 如果只有 answer 字段（字符串），回退到单答案评估
        - 此函数符合VQAv2官方评估标准
    """
    # 选择归一化函数
    if normalization_mode == "standard":
        normalize_func = _normalize_answer_standard
        norm_type = "标准"
    elif normalization_mode == "moderate":
        normalize_func = _normalize_answer_moderate
        norm_type = "中等"
    elif normalization_mode == "legacy":
        normalize_func = _normalize_answer_legacy
        norm_type = "旧版"
    else:
        logger.warning(f"未知的归一化模式 {normalization_mode}，使用中等模式")
        normalize_func = _normalize_answer_moderate
        norm_type = "中等"
    
    total_score = 0.0
    total = len(samples)

    if total == 0:
        logger.warning("VQA 评估时样本为空，返回 0.0")
        return 0.0

    for s in samples:
        qid = int(s["qa_id"])
        pred_raw = predictions.get(qid, "")
        pred_norm = normalize_func(pred_raw or "")
        
        # 获取所有标注者的答案
        answers = s.get("answers", [])
        if not answers:
            # 如果没有多标注者答案，回退到单个答案
            gt_answer = s.get("answer", "")
            if gt_answer:
                answers = [gt_answer]
        
        if not answers:
            # 如果没有答案，跳过（或视为错误）
            continue
        
        # 计算同意预测答案的标注者数量
        num_agree = 0
        for ans in answers:
            ans_norm = normalize_func(str(ans) if ans else "")
            if ans_norm == pred_norm:
                num_agree += 1
        
        # VQAv2标准：min(1, num_agree/3)
        # 如果至少3个标注者同意，得分为1.0
        score = min(1.0, num_agree / 3.0)
        total_score += score
    
    acc = total_score / total if total > 0 else 0.0
    logger.info(f"VQA Accuracy (VQAv2方法, {norm_type}归一化): {acc:.4f} (总分={total_score:.2f}/{total})")
    return float(acc)


