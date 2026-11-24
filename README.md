# DisasterKD

PyTorch implementation of **DisasterKD: Frequency-guided cross-decoder knowledge distillation for UAV real-time disaster damage assessment**. A PSPNet-101 teacher guides a PSPNet-18 or DeepLabV3-18 student on FloodNet or RescueNet. During training, the teacher decoder maps the student's projected encoder features to logits. A Haar wavelet transform separates those logits into low- and high-frequency bands for distillation. Inference needs only the student. [Paper in *Pattern Recognition*](https://doi.org/10.1016/j.patcog.2025.112968)

## Reproduce on FloodNet

The commands below start with the public dataset and train both models. Run them from the repository root on a Linux machine with an NVIDIA GPU. The reference environment uses Python 3.10, PyTorch 2.1.2, torchvision 0.16.2 and CUDA 11.8 wheels. The first full training run downloads ImageNet initialization. [PyTorch installation options](https://pytorch.org/get-started/previous-versions/)

```bash
git clone https://github.com/SummerTide/DisasterKD.git
cd DisasterKD
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); assert torch.cuda.is_available()"
```

Download the official FloodNet-Supervised v1.0 archive (about 13 GB). Extract the indexed masks under `FloodNet-Supervised_v1.0`. The sibling `ColorMasks` directory contains visualizations. [Dataset and terms of use](https://github.com/BinaLab/FloodNet-Supervised_v1.0)

```bash
mkdir -p datasets/raw
curl -fL --retry 3 \
  'https://www.dropbox.com/scl/fo/k33qdif15ns2qv2jdxvhx/ANGaa8iPRhvlrvcKXjnmNRc?rlkey=ao2493wzl1cltonowjdbrnp7f&dl=1' \
  -o datasets/raw/floodnet.zip
unzip -q datasets/raw/floodnet.zip 'FloodNet-Supervised_v1.0/*' -d datasets/raw

bash scripts/reproduce.sh floodnet datasets/raw/FloodNet-Supervised_v1.0 pspnet18
```

The script prepares the data, trains a supervised teacher, distills the student and evaluates the final student on the test split. Repeating the command reuses prepared data and trains fresh weights in the selected run directory. Use `--run-root runs/experiment2` to keep a separate run. The main outputs are

```text
runs/floodnet/
├── teacher/teacher.pth
└── disasterkd/pspnet18/
    ├── student.pth        # inference weights
    ├── run.json           # resolved training settings and data paths
    ├── config.py          # selected experiment config
    ├── metrics.json       # per-class scores, mIoU, OA, kappa and valid pixel count
    ├── cm/                # NumPy confusion matrices, rows = ground truth
    └── vis_student.pth/   # paletted PNG predictions
```

**Quick installation check.** Run `python -m unittest discover -s tests -v` for an offline CPU test. On Apple Silicon, install `requirements.txt` directly in a Python 3.10 environment for the CPU check. It generates labeled images in a temporary directory and runs preparation, one teacher update, one student update and evaluation. To check the GPU training path on your prepared data, append `--steps 1 --batch-size 2 --crop-size 64 --no-pretrained --run-root runs/smoke` to the reproduction command. Evaluation still reads the selected test split.

## RescueNet V1

These experiments use the **original V1 annotations** with 3,595 training, 449 validation and 450 test images. Raw IDs 1–11 represent debris, water, four building-damage levels, vehicle, road, tree, pool and sand. ID 0 is ignored. The authors' historical V1 download link returned HTTP 404 when this guide was verified. Obtain the V1 archive from the dataset maintainers or use an existing V1 copy. The current release has separate clear/blocked road classes and different label IDs. [Original dataset description](https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation/blob/85ceda324bfbe92d8f7c24d98080fd8ce47d0ec0/README.md) · [Current label definitions](https://doi.org/10.1038/s41597-023-02799-4)

Point the command at the extracted folder containing `train/` and `test/`. For example, if the archive extracted to `datasets/raw/RescueNet`, train a DeepLabV3-18 student with

```bash
bash scripts/reproduce.sh rescuenet datasets/raw/RescueNet deeplabv3-18
```

Both datasets use this input layout, with matching numeric image IDs across each image/mask pair.

```text
RAW_ROOT/
├── train/train-org-img/ID.jpg
├── train/train-label-img/ID_lab.png
├── test/test-org-img/ID.jpg
└── test/test-label-img/ID_lab.png
```

FloodNet preparation crops 1024 × 1024 training tiles and shifts the final tile on each axis to the image boundary. RescueNet preparation resizes training images to 896 × 672 with bicubic interpolation and masks with nearest-neighbor interpolation. Both keep the original test resolution. Prepared pairs live in `datasets/<dataset>/{train,test}/{images,masks}/`. The recipes train on the supplied training split and evaluate the final checkpoint on the test split.

## Run individual stages

Each command resolves its inputs from the same data and run roots. Use these stages to reuse an existing teacher, train a second student, or repeat evaluation.

```bash
python -m disasterkd prepare --dataset floodnet --raw-root datasets/raw/FloodNet-Supervised_v1.0
python -m disasterkd teacher --dataset floodnet
python -m disasterkd distill --dataset floodnet --student deeplabv3-18
python -m disasterkd evaluate --dataset floodnet --student deeplabv3-18
```

Run `prepare` once per dataset and destination. An existing PSPNet-101 teacher can be supplied with `distill --teacher-checkpoint /path/to/teacher.pth`. To score a saved student, pass `evaluate --checkpoint /path/to/student.pth` with its dataset and architecture. Checkpoints are plain PyTorch state dictionaries loaded with `strict=True`. For shared storage, set `DISASTERKD_DATA_ROOT` and `DISASTERKD_RUN_ROOT`, or pass `--data-root` and `--run-root`.

### Released training settings

| Setting | FloodNet | RescueNet V1 |
| --- | --- | --- |
| Classes scored | 10, including background | 11, excluding raw ID 0 |
| Teacher / student updates in this pipeline | 120,000 / 120,000 | 60,000 / 60,000 |
| Batch size / training crop | 8 / 512 × 512 | 16 / 512 × 512 |
| Test inference | 1024 × 1024 sliding windows, stride 512 | Resize to 896 × 672, then project predictions back |
| Low-frequency weight, PSPNet-18 / DeepLabV3-18 | 0.4 / 0.4 | 0.0 / 0.4 |

Both recipes use SGD, initial LR 0.01, polynomial decay with power 0.9, momentum 0.9, weight decay 0.0001 and seed 2333. The released DisasterKD objective combines segmentation cross-entropy with a frequency loss weighted by 3.0. Its channel-wise KL temperature is 4.0 and the high-frequency weight is one minus the low-frequency weight. The table follows the executable configs, including the high-frequency-only RescueNet PSPNet-18 variant. The supervised teacher stage uses the same dataset schedule to produce a new checkpoint. A comparison against a particular historical teacher requires that teacher's weights. Batch size, step budget, backbone initialization and GPU kernels can affect the resulting scores.

### Comparison methods

`distill --method` accepts `disasterkd` (default), `cwd`, `ifvd`, `mgd`, `maskd` and `freekd`. CWD, IFVD and MGD use the trained teacher directly. MasKD and FreeKD first fit their own mask module. For example, after training the FloodNet teacher, run

```bash
python -m disasterkd mask --dataset floodnet --method maskd
python -m disasterkd distill --dataset floodnet --student pspnet18 --method maskd
python -m disasterkd evaluate --dataset floodnet --student pspnet18 --method maskd
```

## Verification

On a single NVIDIA RTX 3090, the pipeline completed a teacher update, a student update and full-image inference on real samples from each dataset. All five comparison methods also completed a FloodNet student update, including the mask pretraining stages. These checks cover the execution path. Reproducing benchmark scores requires the full splits and training budgets above.

## Code map

| Location | Contents |
| --- | --- |
| `disasterkd/` | Command-line interface, data preparation, training loop and distillation objectives |
| `configs/` | Dataset transforms, optimization schedules and experiment-specific loss weights |
| `data/` | Dataset readers, label maps, paired transforms and batch sampling |
| `module/baseline/` and `losses/` | Model implementations, teacher decoder and loss functions |
| `eval_floodnet.py`, `eval_rescuenet.py`, `utils/metrics.py` | Dataset-specific inference and pixel metrics |
| `scripts/`, `tests/` | End-to-end reproduction command and offline regression tests |

The original top-level training filenames remain as compatibility entry points. They delegate to the shared trainer and retain the experiment config's legacy `logs/` output paths. New runs can use `python -m disasterkd` throughout.

## Citation

```bibtex
@article{guo2026disasterkd,
  title   = {DisasterKD: Frequency-guided cross-decoder knowledge distillation for UAV real-time disaster damage assessment},
  author  = {Guo, Jianchong and Wan, Yuting and Wang, Junjue and Ma, Ailong and Zhong, Yanfei},
  journal = {Pattern Recognition},
  volume  = {174},
  pages   = {112968},
  year    = {2026},
  doi     = {10.1016/j.patcog.2025.112968}
}
```
