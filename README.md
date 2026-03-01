<div align="center">

# GuideTWSI

### A Diverse Tactile Walking Surface Indicator Dataset from Synthetic and Real-World Images for Blind and Low-Vision Navigation

**Hochul Hwang<sup>1</sup>, Soowan Yang<sup>2</sup>, Anh N. H. Nguyen<sup>1</sup>, Parth Goel<sup>1</sup>, Krisha Adhikari<sup>1</sup>, Sunghoon I. Lee<sup>1</sup>, Joydeep Biswas<sup>3</sup>, Nicholas A. Giudice<sup>4</sup>, Donghyun Kim<sup>1</sup>**

<sup>1</sup>University of Massachusetts Amherst &nbsp; <sup>2</sup>DGIST &nbsp; <sup>3</sup>UT Austin &nbsp; <sup>4</sup>University of Maine

[[Paper]](https://arxiv.org/) &nbsp; [[Project Page]](https://guidedogrobot-tactile.github.io/) &nbsp; [[Dataset (HuggingFace)]](https://huggingface.co/datasets/guidedogrobot-tactile/GuideTWSI) &nbsp; [[Pretrained Weights (HuggingFace)]](https://huggingface.co/guidedogrobot-tactile/GuideTWSI)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![ICRA 2026](https://img.shields.io/badge/ICRA-2026-green.svg)]()

</div>

<p align="center">
  <video src="https://github.com/user-attachments/assets/aa48c885-52f4-490c-af97-0c005d4e5144" width="90%" autoplay loop muted playsinline>
    Your browser does not support the video tag.
  </video>
</p>

## Overview

Tactile Walking Surface Indicators (TWSIs) — truncated domes and directional bars — are safety-critical landmarks that blind and low-vision (BLV) pedestrians use to locate crossings and hazard zones. Existing TWSI datasets are geographically biased toward East Asian directional bars, lack robot-relevant viewpoints, and rarely cover truncated domes.

**GuideTWSI** introduces the largest and most diverse TWSI dataset, combining:
- **39.5K images** across real-world compilations, photorealistic synthetic data, and robot-collected samples
- A **photorealistic UE4 + AirSim synthetic data pipeline** generating 15K+ annotated truncated dome images
- Consistent **+29 mIoU improvement** across state-of-the-art segmentation models when augmenting with synthetic data
- **96.15% stop success rate** in real-world robot experiments on a Unitree Go2 quadruped

## Highlights

- **Large-scale diverse dataset**: 39.5K images spanning 3 sub-datasets (RBar-22K, SDome-15K, RDome-2K) with real bars, synthetic domes, and robot-collected domes
- **Synthetic data pipeline**: Fully customizable UE4-based pipeline producing photorealistic truncated dome data with automatic ground truth across 10 environments
- **Cross-domain generalization**: Synthetic augmentation boosts segmentation performance across all tested models, with Mask2Former mIoU rising from 0.58 to 0.84
- **Real robot validation**: Fine-tuned YOLOv11-seg-N deployed on a Unitree Go2 achieves 96.15% stopping accuracy at truncated domes

## Dataset Overview

| Dataset | Scale | Type | Source | Modalities |
|---------|-------|------|--------|------------|
| **RBar-22K** | ~22K | Real/bars | SideGuide, Tenji10K, TP, 69 Roboflow repos | RGB, Seg |
| **SDome-15K** | 15K+ | Synthetic/domes | UE4 + AirSim pipeline (10 environments) | RGB+D, BBx, Seg |
| **RDome-2K** | 2.4K+ | Real/domes | Unitree Go2 robot collection | RGB, Seg |

For detailed dataset information, see [`docs/DATASET.md`](docs/DATASET.md).

## Installation

```bash
git clone https://github.com/DARoSLab/GuideTWSI
cd GuideTWSI
pip install -r requirements.txt
```

## Dataset Download

Download the GuideTWSI dataset and pretrained weights from HuggingFace:

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Download the full dataset
huggingface-cli download guidedogrobot-tactile/GuideTWSI --repo-type dataset --local-dir ./data

# Download pretrained weights
huggingface-cli download guidedogrobot-tactile/GuideTWSI-weights --local-dir ./checkpoints
```

## Model Training

We provide training notebooks for all models evaluated in the paper:

| Notebook | Model | Description |
|----------|-------|-------------|
| [`YOLOv11_Seg.ipynb`](notebooks/YOLOv11_Seg.ipynb) | YOLOv11-Seg-N/X | Instance segmentation via Ultralytics API |
| [`Mask2Former.ipynb`](notebooks/Mask2Former.ipynb) | Mask2Former | Transformer-based segmentation via Detectron2 |
| [`SAM2_UNet.ipynb`](notebooks/SAM2_UNet.ipynb) | SAM2.1+UNet | Frozen SAM2.1 backbone with custom UNet decoder |
| [`DINOv3_RegCls.ipynb`](notebooks/DINOv3_RegCls.ipynb) | DINOv3+RegCls | Patchwise logistic regression on DINOv3 features |
| [`DINOv3_EoMT.ipynb`](notebooks/DINOv3_EoMT.ipynb) | DINOv3+EoMT | Encoder-only Mask Transformer via lightly-train |

Model configs with exact paper hyperparameters are in [`configs/`](configs/).

## Evaluation

Run evaluation on any model checkpoint:

```bash
python evaluation/evaluate.py \
    --model yolov11_seg_n \
    --weights checkpoints/yolov11n_seg_best.pt \
    --data data/RDome-2K \
    --config configs/yolov11_seg_n.yaml \
    --reference
```

Supported models: `yolov11_seg_n`, `yolov11_seg_x`, `mask2former`, `sam2_unet`, `dinov3_regcls`, `dinov3_eomt`

## Results

Impact of synthetic data augmentation on truncated dome segmentation (RBar-train + SDome-15K → RDome-2K):

| Method | | Real Data Only | | | | Real + Synthetic Data | | | |
|--------|---|---|---|---|---|---|---|---|---|
| | **Prec.** | **Rec.** | **mAP50-95** | **mIoU** | **Prec.** | **Rec.** | **mAP50-95** | **mIoU** | **Δ mIoU** |
| YOLOv11-seg-N | 0.7958 | 0.6924 | 0.5934 | 0.6161 | 0.8718 | 0.8084 | 0.7288 | **0.7308** | +0.1147 |
| YOLOv11-seg-X | 0.8838 | 0.8204 | 0.7362 | 0.7389 | 0.9102 | 0.8588 | 0.8188 | **0.7887** | +0.0498 |
| Mask2Former | 0.9458 | 0.5975 | 0.4798 | 0.5777 | 0.9611 | 0.8669 | 0.7829 | **0.8375** | +0.2598 |
| SAM2.1+UNet | 0.8680 | 0.5165 | 0.3475 | 0.4789 | 0.9704 | 0.7031 | 0.5627 | **0.6883** | +0.2094 |
| DINOv3+RegCls | 0.9027 | 0.7804 | 0.6176 | 0.7322 | 0.8667 | 0.8924 | 0.6933 | **0.7926** | +0.0604 |
| DINOv3+EoMT | 0.8141 | 0.6237 | 0.4828 | 0.5804 | 0.9305 | 0.9197 | 0.8492 | **0.8756** | +0.2952 |

## Synthetic Data Generation

Our UE4 + AirSim pipeline generates photorealistic truncated dome data with automatic annotation. See the full tutorial:

- **Quick start**: [`data_generation/README.md`](data_generation/README.md)
- **Full pipeline guide**: [`docs/SYNTHETIC_PIPELINE.md`](docs/SYNTHETIC_PIPELINE.md)

```bash
python data_generation/orbit_navigator.py \
    --environment "CityPark" \
    --run 1 \
    --radius 2 \
    --altitude 0.3 \
    --speed 3 \
    --snapshots 30
```

## Repository Structure

```
GuideTWSI/
├── README.md                     # This file
├── LICENSE                       # MIT License
├── CITATION.cff                  # Citation metadata
├── requirements.txt              # Python dependencies
├── assets/                       # README images
├── configs/                      # YAML configs for each model
├── data_generation/              # Synthetic data pipeline (UE4 + AirSim)
├── data_utils/                   # Dataset processing tools
│   ├── format_converters.py      # YOLO ↔ COCO ↔ SAM2 ↔ RLE converter
│   ├── dataset_generator.py      # Train/val/test splitting
│   └── mask_generator.py         # Binary mask generation from RLE
├── notebooks/                    # Model training & evaluation notebooks
├── evaluation/                   # Unified evaluation metrics
│   ├── metrics.py                # Precision, Recall, F1, IoU, mAP
│   └── evaluate.py               # CLI evaluation script
└── docs/                         # Extended documentation
    ├── DATASET.md                # Dataset details & sources
    └── SYNTHETIC_PIPELINE.md     # Full synthetic pipeline tutorial
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{hwang2026guidetwsi,
  title={GuideTWSI: A Diverse Tactile Walking Surface Indicator Dataset from Synthetic and Real-World Images for Blind and Low-Vision Navigation},
  author={Hwang, Hochul and Yang, Soowan and Nguyen, Anh N. H. and Goel, Parth and Adhikari, Krisha and Lee, Sunghoon I. and Biswas, Joydeep and Giudice, Nicholas A. and Kim, Donghyun},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026}
}
```

## Acknowledgments

This work was supported by:
- National Institutes of Health (R21EY037411)
- National Science Foundation (2427788)
- NVIDIA Academic Grant Program

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
