"""Pixel-level segmentation scores accumulated across the entire test set."""

import json
import logging
import time
from pathlib import Path

import numpy as np


class PixelMetric:
    def __init__(self, num_classes, logdir=None, logger=None):
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.logdir = logdir
        self.logger = logger or logging.getLogger(__name__)

    def update(self, target, prediction):
        """Accumulate integer labels. The caller removes ignored pixels."""
        target = np.asarray(target, dtype=np.int64).ravel()
        prediction = np.asarray(prediction, dtype=np.int64).ravel()
        n = len(self.matrix)
        if target.shape != prediction.shape:
            raise ValueError("Target and prediction must have the same number of pixels")
        if target.size and (min(target.min(), prediction.min()) < 0 or max(target.max(), prediction.max()) >= n):
            raise ValueError("Pixel labels are outside the configured class range")
        self.matrix += np.bincount(target * n + prediction, minlength=n * n).reshape(n, n)

    def summary_all(self):
        cm = self.matrix
        actual, predicted = cm.sum(axis=1), cm.sum(axis=0)
        correct = cm.diagonal()
        eps = 1e-7
        precision = correct / (predicted + eps)
        recall = correct / (actual + eps)
        iou = correct / (actual + predicted - correct + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        scores = np.round(np.column_stack((iou, f1, precision, recall)), 5)
        total = cm.sum()
        oa = correct.sum() / (total + eps)
        # Expected disagreements under the empirical class marginals.
        expected = np.outer(actual.astype(float), predicted) / (total + eps)
        kappa = 1 - (total - correct.sum()) / (expected.sum() - expected.trace() + eps)

        rows = ['class       iou        f1  precision    recall']
        for label, values in zip([*range(len(cm)), 'mean'], [*scores, scores.mean(axis=0)]):
            rows.append(f'{label!s:>5} ' + ' '.join(f'{v:9.5f}' for v in values))
        rows.extend([f'OA    {oa:.5f}', f'Kappa {kappa:.5f}'])
        self.logger.info('\n%s', '\n'.join(rows))
        if self.logdir is not None:
            directory = Path(self.logdir) / 'cm'
            directory.mkdir(parents=True, exist_ok=True)
            np.save(directory / f'confusion_matrix-{time.time_ns()}.npy', cm)
        result = {'iou': scores[:, 0], 'f1': scores[:, 1], 'precision': scores[:, 2],
                  'recall': scores[:, 3], 'mIoU': round(scores[:, 0].mean(), 5),
                  'OA': round(oa, 5), 'Kappa': round(kappa, 5)}
        if self.logdir is not None:
            metrics = {key: value.tolist() for key, value in result.items()}
            metrics['evaluated_pixels'] = int(total)
            (Path(self.logdir) / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        return result
