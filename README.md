# DisasterKD

Knowledge distillation for semantic segmentation on FloodNet and RescueNet. The training and evaluation scripts use PyTorch, torchvision, Albumentations and segmentation-models-pytorch. Existing PSPNet and DeepLabV3 state dictionaries can be loaded with `strict=True`.

## Installation

Use Python 3.10 and a matching PyTorch/torchvision build for your device. The training scripts use CUDA. From the repository root, run

```bash
python -m pip install -r requirements.txt
```

## Training

Set the image and mask directories in `configs/base/floodnet.py` or `configs/base/rescuenet.py`. Set `TEACHER_CKPT_PATH` in the selected distillation config to the trained teacher weights. The original server paths are the defaults. For example, to train a PSPNet-18 student on FloodNet, run

```bash
python floodnet_disasterkd_pspnet_r101_r18_train.py \
  --config_path distill_floodnet.testkd_pspnet_r101_r18
```

Each method keeps its own script and config. ImageNet initialization uses the local weights cache or downloads the backbone weights when enabled. The snapshot directory receives the student state dictionary and evaluation results.

## Evaluation

Select a config with the matching `STUDENT` or `MODEL` definition. For example, to evaluate the trained FloodNet student, run

```bash
python eval_floodnet.py \
  --config_path distill_floodnet.testkd_pspnet_r101_r18 \
  --checkpoint /path/to/floodnet-student.pth
```

Use `eval_rescuenet.py` with a RescueNet config for that dataset. Both evaluators report per-class IoU, F1, precision, recall, their class means, overall accuracy and Cohen's kappa. Confusion matrices are saved in `cm/` and prediction masks in `vis_*/` under the configured snapshot directory.

Run the migration regression checks with `python -m unittest discover -s tests`.
