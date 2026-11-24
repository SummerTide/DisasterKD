"""One training loop for supervised teachers, distillation and mask pretraining."""

import json
import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from tqdm import trange

from data.floodnet import FloodNetLoader
from data.loader import repeat_batches
from data.rescuenet import RescueNetLoader
from eval_floodnet import evaluate as evaluate_floodnet
from eval_rescuenet import evaluate as evaluate_rescuenet
from losses.freekd import MaskModule as WaveletMask
from losses.maskd import MaskModule
from losses.task import SegCrossEntropyLoss
from module.baseline.deeplabv3 import DeepLabV3
from module.baseline.pspnet import PSPNet
from utils.tools import adjust_learning_rate, get_console_file_logger
from .objectives import DistillationLoss


LOADERS = {'floodnet': FloodNetLoader, 'rescuenet': RescueNetLoader}
EVALUATORS = {'floodnet': evaluate_floodnet, 'rescuenet': evaluate_rescuenet}


def build_model(options, *, imagenet=True):
    options = dict(options)
    if 'encoder' in options:
        if not imagenet:
            options['encoder'] = dict(options['encoder'], pretrained=False)
        return PSPNet(options)
    if not imagenet:
        options['encoder_weights'] = None
    return DeepLabV3(options)


def save_weights(model, path):
    with NamedTemporaryFile(dir=path.parent, prefix='.weights-', suffix='.tmp', delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        torch.save(model.state_dict(), temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def train(args, cfg, checkpoint, device):
    loader = LOADERS[args.dataset](cfg.TRAIN_DATA_CONFIG)
    batches = repeat_batches(loader)
    criterion = SegCrossEntropyLoss(cfg.IGNORE_LABEL)
    teacher = objective = None
    if args.stage != 'teacher':
        teacher = build_model(cfg.TEACHER, imagenet=False)
        teacher.load_state_dict(torch.load(cfg.TEACHER_CKPT_PATH, map_location='cpu'), strict=True)
        teacher.requires_grad_(False).to(device).train(args.stage == 'mask')

    if args.stage == 'mask':
        mask_type = {'freekd': WaveletMask, 'maskd': MaskModule}[args.method]
        model = mask_type(channels=cfg.TEACHER['classes'], num_tokens=cfg.TOKENS)
    elif args.stage == 'teacher':
        model = build_model(cfg.TEACHER, imagenet=not args.no_pretrained)
    else:
        model = build_model(cfg.STUDENT, imagenet=not args.no_pretrained)
        objective = DistillationLoss(args.method, cfg).to(device)

    model.to(device).train()
    parameters = [{'params': model.parameters()}]
    if objective is not None:
        trainable = [p for p in objective.parameters() if p.requires_grad]
        if trainable:
            parameters.append({'params': trainable})
    optimizer = torch.optim.SGD(parameters, lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM,
                                weight_decay=cfg.WEIGHT_DECAY)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cfg.__file__, checkpoint.parent / 'config.py')
    logger = get_console_file_logger(f'{args.dataset}.{args.stage}', logdir=checkpoint.parent)
    settings = {**vars(args), 'config': cfg.__name__, 'steps': cfg.NUM_STEPS_STOP,
                'schedule_steps': cfg.NUM_STEPS, 'lr': cfg.LEARNING_RATE, 'momentum': cfg.MOMENTUM,
                'weight_decay': cfg.WEIGHT_DECAY, 'power': cfg.POWER, 'batch_size': loader.batch_size,
                'train_examples': len(loader.dataset), 'train_data': cfg.TRAIN_DATA_CONFIG,
                'recipe': {k: v for k, v in vars(cfg).items() if k.isupper()
                           and k not in ('TRAIN_DATA_CONFIG', 'TEST_DATA_CONFIG')},
                'teacher_weights': cfg.TEACHER_CKPT_PATH if teacher is not None else None,
                'device': str(device), 'torch': str(torch.__version__)}
    (checkpoint.parent / 'run.json').write_text(json.dumps(settings, default=str, indent=2) + '\n')
    for step in trange(cfg.NUM_STEPS_STOP):
        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, step, cfg)
        for _ in range(cfg.ITER_SIZE):
            images, labels = next(batches)
            images, target = images.to(device), labels['cls'].long().to(device)
            if teacher is not None:
                with torch.no_grad():
                    teacher_outputs = teacher(images)
            if args.stage == 'mask':
                logits, auxiliary = model(teacher_outputs[-1])
            else:
                outputs = model(images)
                logits = outputs[-1]
                auxiliary = 0 if objective is None else objective(outputs, teacher_outputs, target, teacher.forward_feature)
            task = criterion(logits, target)
            loss = task + auxiliary
            loss.backward()
            optimizer.step()
            if step == 0 or (step + 1) % 50 == 0 or step + 1 == cfg.NUM_STEPS_STOP:
                logger.info('step=%d loss=%.6f task=%.6f lr=%.7g', step + 1, loss.item(), task.item(), lr)
            if step + 1 == cfg.NUM_STEPS_STOP:
                break

    save_weights(model, checkpoint)
    logger.info('Saved %s', checkpoint)
    if args.legacy and args.stage == 'mask' and args.dataset == 'floodnet':
        EVALUATORS[args.dataset](teacher, cfg, ckpt_path=checkpoint, logger=logger, mask_module=model)
    elif args.legacy and args.stage != 'mask':
        EVALUATORS[args.dataset](model, cfg, ckpt_path=checkpoint, logger=logger)
