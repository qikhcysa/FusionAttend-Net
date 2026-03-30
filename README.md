# FusionAttend-Net
Multi-scale Feature Fusion and Attention Network for Plant Disease Classification

## Overview

FusionAttend-Net is a deep-learning plant disease classification model that combines
multi-scale feature fusion (via a YOLOv5-based Backbone + Neck) with a
**Pyramid Squeeze Attention (PSA)** mechanism to achieve high accuracy across several
challenging agricultural datasets.

| Dataset | Accuracy | Macro F1 |
|---|---|---|
| Katra-Twelve | ≥ 95.27 % | ≥ 95.27 % |
| BARI-Sunflower | ≥ 95.27 % | ≥ 95.27 % |
| FGVC8 | ≥ 95.27 % | ≥ 95.27 % |
| PlantVillage | 99.89 % | — |

Key properties:
- **4-fold cross-validation** accuracy fluctuation < 3 %
- **PSA** saves ~26 % parameters compared to a plain SE module
- **t-SNE** visualisation for model interpretability analysis

---

## Architecture

```
Input Image (3 × H × W)
        │
        ▼
┌─────────────────────────────────┐
│  DFN – Deep Feature Network     │
│  ┌──────────────────────────┐   │
│  │  YOLOv5 Backbone         │   │
│  │  (C3 + SPPF blocks)      │   │
│  └──────────┬───────────────┘   │
│             │  P3, P4, P5        │
│  ┌──────────▼───────────────┐   │
│  │  YOLOv5 Neck (FPN+PANet) │   │
│  └──────────────────────────┘   │
└─────────────┬───────────────────┘
              │  multi-scale feature maps
              ▼
┌─────────────────────────────────┐
│  PSAN – Classification Module   │
│  ┌──────────────────────────┐   │
│  │  Feature Fusion Conv     │   │
│  ├──────────────────────────┤   │
│  │  PSA (Pyramid Squeeze    │   │
│  │       Attention)         │   │
│  ├──────────────────────────┤   │
│  │  GAP → Dropout → Linear  │   │
│  └──────────────────────────┘   │
└─────────────┬───────────────────┘
              │
              ▼
       Class Logits (C)
```

### PSA – Pyramid Squeeze Attention

The PSA module pools features at multiple spatial resolutions
(e.g. 1×1, 2×2, 4×4, 8×8), averages the descriptors, then feeds
them through a shared fully-connected bottleneck to produce a
channel-wise attention vector.  Compared to a conventional SE
block the shared projection saves approximately **26 %** parameters.

---

## Repository Structure

```
FusionAttend-Net/
├── models/
│   ├── backbone.py          # YOLOv5 Backbone (Conv, C3, SPPF)
│   ├── neck.py              # YOLOv5 Neck (FPN + PANet)
│   ├── dfn.py               # DFN = Backbone + Neck
│   ├── psa.py               # Pyramid Squeeze Attention
│   ├── psan.py              # PSAN classification head
│   └── fusionattend_net.py  # Full model
├── datasets/
│   ├── plant_disease_dataset.py  # ImageFolder dataset + balanced sampler
│   └── augmentation.py           # Train / val transform pipelines
├── utils/
│   ├── metrics.py           # accuracy, F1, AverageMeter
│   └── visualization.py     # t-SNE, confusion matrix, training curves
├── configs/
│   ├── default.yaml         # Katra-Twelve config
│   ├── bari_sunflower.yaml
│   ├── fgvc8.yaml
│   └── plantvillage.yaml
├── train.py                 # K-fold training script
├── evaluate.py              # Evaluation + visualisation script
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Data Preparation

Organise each dataset in the standard ImageFolder layout:

```
data/
  katra_twelve/
    healthy/
      img_001.jpg
      ...
    powdery_mildew/
      ...
  bari_sunflower/
    ...
```

Supported datasets:

| Name | Description |
|---|---|
| `katra_twelve` | Katra-Twelve plant disease dataset (12 classes) |
| `bari_sunflower` | BARI Sunflower disease dataset |
| `fgvc8` | FGVC8 Plant Pathology challenge |
| `plantvillage` | PlantVillage open-source benchmark (38 classes) |

---

## Training

```bash
# Train with 4-fold cross-validation using the default config
python train.py --config configs/default.yaml

# Train on FGVC8 with a custom learning rate
python train.py --config configs/fgvc8.yaml --lr 5e-4 --epochs 150

# Specify a different data root
python train.py --config configs/plantvillage.yaml \
                --data_root /path/to/plantvillage
```

Checkpoints and training-curve plots are saved under the directory
specified in `output.save_dir` (default `outputs/<dataset_name>`).

---

## Evaluation

```bash
python evaluate.py \
    --config     configs/default.yaml \
    --checkpoint outputs/katra_twelve/fold1_best.pth \
    --data_root  data/katra_twelve/test \
    --save_dir   outputs/katra_twelve/eval
```

This produces:
- `metrics.json` – accuracy and macro F1
- `confusion_matrix.png` – normalised confusion-matrix heat-map
- `tsne.png` – t-SNE 2-D scatter of penultimate-layer features

Pass `--no_tsne` to skip the t-SNE step (faster).

---

## Configuration

All hyper-parameters live in YAML files under `configs/`.  Any key can
be overridden from the command line:

```bash
python train.py --config configs/fgvc8.yaml \
                --epochs 200 \
                --batch_size 64 \
                --lr 1e-3
```

Key configuration options:

| Key | Default | Description |
|---|---|---|
| `model.width_multiple` | 0.5 | YOLOv5 channel width multiplier |
| `model.depth_multiple` | 0.33 | YOLOv5 block depth multiplier |
| `model.psa_reduction` | 16 | PSA FC bottleneck reduction ratio |
| `model.psa_pyramid_levels` | [1,2,4,8] | PSA pooling output sizes |
| `training.k_folds` | 4 | Number of cross-validation folds |
| `dataset.balance_classes` | true | Use WeightedRandomSampler |

---

## Citation

If you use FusionAttend-Net in your research, please cite this work.
