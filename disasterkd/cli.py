import argparse
import copy
import importlib
import os
from pathlib import Path
from types import SimpleNamespace

DATASETS = ('floodnet', 'rescuenet')
STUDENTS = ('pspnet18', 'deeplabv3-18')
METHODS = ('disasterkd', 'cwd', 'mgd', 'ifvd', 'maskd', 'freekd')


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def make_parser():
    parser = argparse.ArgumentParser(prog='python -m disasterkd')
    stages = parser.add_subparsers(dest='stage', required=True)
    for stage in ('prepare', 'teacher', 'mask', 'distill', 'evaluate'):
        sub = stages.add_parser(stage)
        sub.add_argument('--dataset', choices=DATASETS, required=True)
        sub.add_argument('--data-root', type=Path, help='parent of the prepared dataset directories')
        if stage == 'prepare':
            sub.add_argument('--raw-root', type=Path, required=True, help='extracted directory containing train/ and test/')
            continue
        sub.add_argument('--run-root', type=Path, default=Path(os.environ.get('DISASTERKD_RUN_ROOT', 'runs')))
        sub.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda'))
        sub.add_argument('--seed', type=int, default=2333)
        sub.add_argument('--config', '--config_path', help='experiment module under configs/')
        if stage in ('distill', 'evaluate'):
            sub.add_argument('--student', choices=STUDENTS, default='pspnet18')
            sub.add_argument('--method', choices=METHODS, default='disasterkd')
        if stage == 'mask':
            sub.add_argument('--method', choices=('maskd', 'freekd'), required=True)
        if stage in ('distill', 'mask'):
            sub.add_argument('--teacher-checkpoint', type=Path)
        if stage == 'distill':
            sub.add_argument('--mask-checkpoint', type=Path)
        if stage == 'evaluate':
            sub.add_argument('--checkpoint', type=Path, help='student weights (defaults to the selected run)')
        else:
            sub.add_argument('--steps', type=positive, help='override the number of updates and LR schedule length')
            sub.add_argument('--batch-size', type=positive)
            sub.add_argument('--workers', type=int)
            sub.add_argument('--crop-size', type=positive, help='training crop size (default 512)')
            if stage != 'mask':
                sub.add_argument('--no-pretrained', action='store_true', help='initialize the trained backbone offline')
    return parser


def run(args):
    if args.stage == 'prepare':
        from .prepare import prepare
        prepare(args.dataset, args.raw_root, args.data_root or os.environ.get('DISASTERKD_DATA_ROOT', 'datasets'))
        return

    import torch
    from albumentations import RandomCrop
    from data.paths import split_dirs
    from utils.tools import get_console_file_logger, seed_torch
    from .train import EVALUATORS, build_model, train

    method = getattr(args, 'method', 'disasterkd')
    student = getattr(args, 'student', 'pspnet18')
    suffix = 'r18' if student == 'pspnet18' else 'deeplabv3_r18'
    if args.stage == 'mask':
        suffix = 'mask_module'
    prefix = 'testkd' if method == 'disasterkd' else method
    config = args.config or f'distill_{args.dataset}.{prefix}_pspnet_r101_{suffix}'
    source = importlib.import_module('configs.' + config)
    cfg = SimpleNamespace(**vars(source))
    if args.stage == 'teacher' and not hasattr(cfg, 'TEACHER'):
        cfg.TEACHER = cfg.MODEL
    cfg.TRAIN_DATA_CONFIG = copy.deepcopy(cfg.TRAIN_DATA_CONFIG)
    cfg.TEST_DATA_CONFIG = copy.deepcopy(cfg.TEST_DATA_CONFIG)
    if args.data_root is not None:
        cfg.TRAIN_DATA_CONFIG.update(split_dirs(args.dataset, 'train', args.data_root))
        cfg.TEST_DATA_CONFIG.update(split_dirs(args.dataset, 'test', args.data_root))

    if getattr(args, 'steps', None):
        cfg.NUM_STEPS = cfg.NUM_STEPS_STOP = args.steps
    root = args.run_root / args.dataset
    teacher_path = root / 'teacher' / 'teacher.pth'
    mask_path = root / method / 'mask.pth'
    checkpoint = root / method / student / 'student.pth'
    if args.stage == 'teacher':
        checkpoint = teacher_path
    elif args.stage == 'mask':
        checkpoint = mask_path
    if args.legacy:
        checkpoint = Path(cfg.SNAPSHOT_DIR) / f'{cfg.TARGET_SET}_{cfg.NUM_STEPS_STOP}.pth'
    else:
        cfg.SNAPSHOT_DIR = str(checkpoint.parent)
    if args.stage in ('mask', 'distill'):
        cfg.TEACHER_CKPT_PATH = getattr(args, 'teacher_checkpoint', None) or (cfg.TEACHER_CKPT_PATH if args.legacy else teacher_path)
        if not Path(cfg.TEACHER_CKPT_PATH).is_file():
            raise FileNotFoundError(f"Missing teacher {cfg.TEACHER_CKPT_PATH}. Train it with the teacher command or pass --teacher-checkpoint.")
    if args.stage == 'distill' and method in ('maskd', 'freekd'):
        cfg.MASK_CKPT = args.mask_checkpoint or (cfg.MASK_CKPT if args.legacy else mask_path)
        if not Path(cfg.MASK_CKPT).is_file():
            raise FileNotFoundError(f'Missing mask {cfg.MASK_CKPT}. Run the mask command for {method} first.')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    seed_torch(args.seed)
    if args.stage == 'evaluate':
        checkpoint = args.checkpoint or checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = build_model(cfg.MODEL if hasattr(cfg, 'MODEL') else cfg.STUDENT, imagenet=False).to(device)
        Path(cfg.SNAPSHOT_DIR).mkdir(parents=True, exist_ok=True)
        logger = get_console_file_logger(cfg.TARGET_SET, logdir=cfg.SNAPSHOT_DIR)
        return EVALUATORS[args.dataset](model, cfg, ckpt_path=checkpoint, logger=logger)

    if args.batch_size:
        cfg.TRAIN_DATA_CONFIG['batch_size'] = args.batch_size
    if args.workers is not None:
        cfg.TRAIN_DATA_CONFIG['num_workers'] = args.workers
    if args.crop_size:
        if args.crop_size % 8:
            raise ValueError('--crop-size must be a multiple of the encoder output stride (8)')
        for i, transform in enumerate(cfg.TRAIN_DATA_CONFIG['transforms'].transforms):
            if isinstance(transform, RandomCrop):
                cfg.TRAIN_DATA_CONFIG['transforms'].transforms[i] = RandomCrop(args.crop_size, args.crop_size)
    train(args, cfg, checkpoint, device)


def main(argv=None, *, legacy=False):
    args = make_parser().parse_args(argv)
    args.legacy = legacy
    return run(args)
