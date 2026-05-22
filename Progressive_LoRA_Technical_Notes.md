# Progressive LoRA Technical Notes

## 1. Document Scope

This document explains the method design, code mapping, and experimental conventions of Progressive LoRA. It is intended to serve as a technical note for the anonymous submission version.

This document focuses on three questions:

- What are the core mechanisms of Progressive LoRA?
- Which source files implement these mechanisms?
- How should the experimental settings and recommended commands in the repository be interpreted?

## 2. Method Overview

The goal of Progressive LoRA is to improve fine-tuning efficiency for multimodal tasks by gradually increasing adapter capacity without modifying the backbone model structure.

It consists of three main components:

1. Complexity estimation based on text features.
2. SVD-based LoRA weight inheritance and rank expansion.
3. A three-stage cumulative curriculum learning strategy.

## 3. Three-Stage Training Strategy

Training follows an easy-to-hard cumulative schedule.

### Stage 1

- Uses only Easy samples.
- Uses a low LoRA rank.
- Learns basic vision-language alignment first.

### Stage 2

- Uses cumulative `Easy + Medium` samples.
- Expands LoRA rank from low to medium.
- Preserves basic capability while learning richer patterns.

### Stage 3

- Uses the full training set.
- Expands LoRA rank to the final target value.
- Finishes final optimization on the full dataset.

This design differs from fixed-rank Standard LoRA because it changes both sample difficulty and adapter capacity over time.

## 4. Complexity Modeling

### 4.1 Image Caption Complexity

The image caption complexity score is mainly built from the following factors:

- Length complexity
- Lexical richness
- Semantic density
- Information entropy
- Structural complexity
- Repetition and grammar penalties

The main implementation is located in:

- `coco_dataset.py`
- `flickr8k_adapter.py`
- `flickr30k_adapter.py`
- `coco_karpathy_adapter.py`
- `vizwiz_adapter.py`

### 4.2 VQA Complexity

The complexity modeling for Visual7W differs from image captioning and places more emphasis on question text and answer format.

The main factors include:

- Question length and structural complexity
- Question-type bias
- Logical and causal density
- Answer complexity
- Confidence scaling

The main implementation is located in:

- `vqa7w_adapter.py`
- `vqa7w_trainer.py`
- `traditional_lora/vqa7w_trainer.py`
- `coco_dataset.py`

## 5. SVD Expansion and Weight Inheritance

Progressive LoRA does not simply increase the rank. During stage transition, it tries to preserve useful directions learned in previous stages.

The overall idea is:

- Reconstruct the update matrix from the current LoRA weights.
- Perform SVD on the update matrix.
- Keep learned principal directions and construct orthogonal complements for new dimensions.
- Regenerate higher-rank LoRA weights as initialization for the next stage.

The relevant implementation is mainly in:

- `coco_trainer.py`
- `vqa7w_trainer.py`

## 6. Code Mapping

### 6.1 Progressive LoRA Pipeline

- `run_progressive_training.py`: training entry for Progressive LoRA image captioning.
- `run_vqa7w_training.py`: training entry for Progressive LoRA Visual7W.
- `coco_trainer.py`: training, evaluation, and rank-expansion logic for captioning.
- `vqa7w_trainer.py`: training and evaluation logic for Visual7W.
- `model_loader.py`: backbone and LoRA model loading.
- `lora_config.py`: target modules and base LoRA configuration.

### 6.2 Standard LoRA Baseline

- `traditional_lora/run_lora_experiment.py`: fixed-rank LoRA entry for image captioning.
- `traditional_lora/run_vqa7w_training.py`: fixed-rank LoRA entry for Visual7W.
- `traditional_lora/configs/traditional_lora_r32.json`: main baseline configuration.

## 7. Experimental Conventions

### 7.1 Image Captioning Settings

| Dataset | Standard LoRA | Progressive-A | Progressive-B |
| --- | --- | --- | --- |
| Flickr8K | `r=32, 6 epochs` | `8->16->32, 1-2-4` | `16->24->32, 1-3-3` |
| Flickr30K | `r=32, 5 epochs` | `16->24->32, 1-2-3` | `8->16->32, 2-2-2` |
| COCO-Karpathy | `r=32, 5 epochs` | `8->16->32, 2-2-2` | `16->24->32, 1-2-3` |

### 7.2 Visual7W Settings

| Method | Rank Config | Epochs |
| --- | --- | --- |
| Standard LoRA | `32(fixed)` | `4` |
| Progressive-A | `8->16->32` | `1-2-2` |
| Progressive-B | `16->24->32` | `1-2-2` |

Note: some argparse defaults in the code preserve historical settings, so reproducible runs should prefer the explicit commands listed in `README.md`.

## 8. How to Interpret the Results

The repository uses the following core efficiency indicators:

- `R_param`: cumulative parameter usage ratio.
- `R_sample`: cumulative sample usage ratio.
- `Relative FLOPs`: relative estimate of LoRA-layer training computation.

These indicators are used to answer two central questions:

- Whether Progressive LoRA can reduce training cost while maintaining comparable or better performance than the baseline.
- Whether different rank schedules provide stable gains across datasets.

## 9. Suggested Reading Order

- Read `README.md` first for task entry points, environment requirements, and commands.
- Then read this document to understand the method design and code mapping.
- Finally inspect the source code for detailed training behavior.

## 10. Anonymity Note

This technical note does not include names, affiliations, or other identity markers and is prepared for anonymous code submission.
