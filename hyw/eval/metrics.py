from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class VideoMetrics:
    l1: float
    mse: float
    psnr: float


def compute_video_metrics(gt: np.ndarray, pred: np.ndarray) -> VideoMetrics:
    """
    Compute simple pixel-space metrics between GT and pred.

    Args:
        gt: (T,H,W,3) uint8 or float in [0,1]
        pred: (T,H,W,3) uint8 or float in [0,1]
    """
    if gt.dtype == np.uint8:
        gt_f = gt.astype(np.float32) / 255.0
    else:
        gt_f = gt.astype(np.float32)

    if pred.dtype == np.uint8:
        pred_f = pred.astype(np.float32) / 255.0
    else:
        pred_f = pred.astype(np.float32)

    T = min(gt_f.shape[0], pred_f.shape[0])
    gt_f = gt_f[:T]
    pred_f = pred_f[:T]

    diff = pred_f - gt_f
    l1 = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff * diff))
    psnr = float("inf") if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))
    return VideoMetrics(l1=l1, mse=mse, psnr=psnr)


