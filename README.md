# Progressive LoRA: Efficient Fine-tuning of Multimodal Large Models with Curriculum Learning

## 1. Overview

This repository provides an anonymous, submission-ready code release of Progressive LoRA for parameter-efficient fine-tuning experiments based on LLaVA-1.5-7B.

Progressive LoRA combines three-stage cumulative curriculum learning, progressive LoRA rank expansion, and SVD-based weight inheritance to improve the trade-off between training efficiency and downstream performance.

This submission version keeps only runnable code, configuration files, and essential documentation suitable for anonymous GitHub release.

## 2. Method Summary

### 2.1 Progressive LoRA

The core workflow is:

- Stage 1: Train with a low rank on the Easy subset.
- Stage 2: Expand to a medium rank and continue training on cumulative Easy + Medium samples.
- Stage 3: Expand to the target rank and finish training on the full dataset.

The main design choices are:

- Cumulative curriculum learning: samples grow from `Easy -> Easy+Medium -> All`.
- Progressive rank expansion: LoRA rank increases across stages instead of remaining fixed.
- SVD-based weight inheritance: rank expansion preserves useful update directions learned in previous stages.

### 2.2 Standard LoRA Baseline

The repository also includes a standard fixed-rank LoRA baseline for fair comparison against the progressive training strategy.

## 3. Tasks and Experimental Settings

### 3.1 Image Captioning

Supported datasets: `Flickr8K`, `Flickr30K`, and `COCO-Karpathy`.

| Dataset | Standard LoRA | Progressive-A | Progressive-B |
| --- | --- | --- | --- |
| Flickr8K | `r=32, 6 epochs` | `8->16->32, 1-2-4` | `16->24->32, 1-3-3` |
| Flickr30K | `r=32, 5 epochs` | `16->24->32, 1-2-3` | `8->16->32, 2-2-2` |
| COCO-Karpathy | `r=32, 5 epochs` | `8->16->32, 2-2-2` | `16->24->32, 1-2-3` |

Evaluation metrics: `CIDEr-D`, `BLEU-4`, `ROUGE-L`, and `METEOR`.

### 3.2 Visual Question Answering

Supported task: `Visual7W telling`.

| Method | Rank Config | Epochs |
| --- | --- | --- |
| Standard LoRA | `32(fixed)` | `4` |
| Progressive-A | `8->16->32` | `1-2-2` |
| Progressive-B | `16->24->32` | `1-2-2` |

Evaluation metric: normalized-text `Accuracy`.

## 4. Repository Structure

```text
llava_expe_submission/
├── README.md
├── Progressive_LoRA_Technical_Notes.md
├── requirements.txt
├── .gitignore
├── run_progressive_training.py
├── run_vqa7w_training.py
├── coco_trainer.py
├── coco_dataset.py
├── coco_evaluator.py
├── coco_test_predictor.py
├── coco_karpathy_adapter.py
├── flickr8k_adapter.py
├── flickr30k_adapter.py
├── vqa7w_adapter.py
├── vqa7w_trainer.py
├── lora_config.py
├── model_loader.py
└── traditional_lora/
    ├── run_lora_experiment.py
    ├── run_vqa7w_training.py
    ├── requirements.txt
    └── configs/
```

Key files:

- `run_progressive_training.py`: Progressive LoRA entry point for image captioning.
- `run_vqa7w_training.py`: Progressive LoRA entry point for Visual7W.
- `traditional_lora/run_lora_experiment.py`: standard LoRA baseline entry for image captioning.
- `traditional_lora/run_vqa7w_training.py`: standard LoRA baseline entry for Visual7W.
- `coco_trainer.py` / `vqa7w_trainer.py`: core training logic and stage transition implementation.

## 5. Environment

Recommended environment:

- Python `3.10+`
- PyTorch `2.0+`
- CUDA `11.8` or `12.1`
- Recommended GPU memory: `>= 24GB`

Install dependencies:

```bash
pip install -r requirements.txt
```

If needed, install PyTorch separately using the official wheel matching your local CUDA version.

## 6. Data Preparation

### 6.1 Base Model

Prepare a local path to LLaVA-1.5-7B before training.

```bash
git lfs install
git clone https://huggingface.co/liuhaotian/llava-v1.5-7b /path/to/llava-1.5-7b
```

### 6.2 Dataset Layout

#### Flickr8K

```text
/path/to/Flickr8k/
├── 1000268201_693b08cb0e.jpg
├── 1001773457_577c3a7d70.jpg
└── Flickr8k_text/
    ├── Flickr8k.token.txt
    ├── Flickr_8k.trainImages.txt
    ├── Flickr_8k.devImages.txt
    └── Flickr_8k.testImages.txt
```

#### Flickr30K

```text
/path/to/Flickr30K/
├── flickr30k-images/
│   └── flickr30k-images/
└── flickr_annotations_30k.csv
```

#### COCO-Karpathy

```text
/path/to/coco2014/
├── train2014/
├── val2014/
└── dataset_coco.json
```

#### Visual7W telling

```text
/path/to/VQA7W/
├── images/
└── dataset_v7w_telling/
    └── dataset_v7w_telling.json
```

## 7. Running Commands

Note: the entry scripts still keep some historical defaults for backward compatibility and quick environment checks. To match the submission settings, prefer the explicit commands below instead of relying on script defaults.

### 7.1 Progressive LoRA for Image Captioning

#### Flickr8K Progressive-A

```bash
python run_progressive_training.py \
  --model_path /path/to/llava-1.5-7b \
  --dataset flickr8k \
  --data_path /path/to/Flickr8k \
  --output_dir ./outputs/flickr8k_prog_a \
  --easy_epochs 1 \
  --medium_epochs 2 \
  --hard_epochs 4 \
  --easy_rank 8 \
  --medium_rank 16 \
  --hard_rank 32 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.1
```

#### Flickr30K Progressive-A

```bash
python run_progressive_training.py \
  --model_path /path/to/llava-1.5-7b \
  --dataset flickr30k \
  --data_path /path/to/Flickr30K \
  --output_dir ./outputs/flickr30k_prog_a \
  --easy_epochs 1 \
  --medium_epochs 2 \
  --hard_epochs 3 \
  --easy_rank 16 \
  --medium_rank 24 \
  --hard_rank 32 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.1
```

#### COCO-Karpathy Progressive-B

```bash
python run_progressive_training.py \
  --model_path /path/to/llava-1.5-7b \
  --dataset coco_karpathy \
  --data_path /path/to/coco2014 \
  --output_dir ./outputs/coco_karpathy_prog_b \
  --easy_epochs 1 \
  --medium_epochs 2 \
  --hard_epochs 3 \
  --easy_rank 16 \
  --medium_rank 24 \
  --hard_rank 32 \
  --batch_size 22 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.1
```

### 7.2 Standard LoRA for Image Captioning

```bash
python traditional_lora/run_lora_experiment.py \
  --model_path /path/to/llava-1.5-7b \
  --dataset flickr30k \
  --data_path /path/to/Flickr30K \
  --output_dir ./outputs/flickr30k_standard \
  --config_name traditional_lora_r32 \
  --epochs 5 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.1
```

### 7.3 Visual7W Progressive LoRA

```bash
nohup python run_vqa7w_training.py \
  --model_path /path/to/llava-1.5-7b \
  --data_path /path/to/VQA7W \
  --output_dir ./progressive_lora_vqa7w_standard \
  --easy_epochs 1 \
  --medium_epochs 2 \
  --hard_epochs 2 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --scheduler_type cosine \
  --cosine_num_cycles 0.5 \
  --max_new_tokens 48 \
  --temperature 0.7 \
  --complexity_thresholds 33.33 66.67 \
  --easy_lora_rank 16 \
  --medium_lora_rank 24 \
  --hard_lora_rank 32 \
  --lora_config_name progressive_lora
```

### 7.4 Visual7W Standard LoRA

```bash
nohup python traditional_lora/run_vqa7w_training.py \
  --model_path /path/to/llava-1.5-7b \
  --data_path /path/to/VQA7W \
  --output_dir ./traditional_lora_vqa7w_r32 \
  --num_epochs 4 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --scheduler_type cosine \
  --cosine_num_cycles 0.5 \
  --max_new_tokens 48 \
  --temperature 0.7 \
  --lora_config_name traditional_lora_r32
```

## 8. Result Summary

Representative results used in the current submission version, consistent with
the current tables and summary statements in `latex.txt`:

- Flickr8K: `Progressive-B` reaches `0.3295` on `BLEU-4`.
- Flickr30K: `Progressive-A` reaches `0.6711` on `CIDEr-D`.
- COCO-Karpathy: `Progressive-B` reaches `0.9797` on `CIDEr-D`.
- Visual7W: `Progressive-B` reaches `0.4036` on `Accuracy`.
- Relative computation reduction reaches up to `43.3%`.
- Cumulative parameter usage reduction reaches up to `43.3%`, depending on the dataset and rank schedule.

## 9. Metric Notes

- `R_param`: cumulative parameter usage ratio.
- `R_sample`: cumulative sample usage ratio.
- `Relative FLOPs`: relative estimate of LoRA-layer training cost.

```text
R_param = sum(rank_i * epochs_i) / (rank_baseline * epochs_baseline)
R_sample = sum(sample_count_i * epochs_i) / (sample_count_baseline * epochs_baseline)
Relative FLOPs = sum(rank_i * steps_i * epochs_i) /
                 (rank_baseline * steps_baseline * epochs_baseline)
```

## 10. Anonymity and Usage Notes

- This repository is prepared for anonymous code submission and does not include identity information.
- The documentation is intended to support reproducible commands, method understanding, and result alignment checks.
- For more implementation details, please refer to `Progressive_LoRA_Technical_Notes.md`.
