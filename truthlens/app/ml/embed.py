"""Pre-trained EfficientNet-Lite4 (ImageNet, ONNX model zoo, Apache-2.0) used as a frozen feature extractor.

Runs with OpenCV's DNN module - no PyTorch/TensorFlow needed. The 1280-d global-average-pool output is taken
for the whole image and averaged over five 224 px crops (fine texture), giving a 2560-d embedding.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "effnet_lite4.onnx"
LAYER = "onnx_node!efficientnet-lite4/model/head/AvgPool"
_NET = None


def available() -> bool:
    return MODEL_PATH.exists()


def _net():
    global _NET
    if _NET is None:
        _NET = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
    return _NET


def embed(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    views = [cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)]
    s = 448 / max(rgb.shape[:2])
    r2 = cv2.resize(rgb, (max(224, int(rgb.shape[1] * s)), max(224, int(rgb.shape[0] * s))), interpolation=cv2.INTER_AREA)
    H, W = r2.shape[:2]
    for yy, xx in [(0, 0), (0, W - 224), (H - 224, 0), (H - 224, W - 224), ((H - 224) // 2, (W - 224) // 2)]:
        views.append(r2[yy:yy + 224, xx:xx + 224])
    net = _net()
    out = []
    for v in views:
        net.setInput(((v.astype(np.float32) - 127.0) / 128.0)[None])
        out.append(net.forward(LAYER).reshape(-1))
    e = np.concatenate([out[0], np.mean(out[1:], axis=0)]).astype(np.float64)
    return e / (np.linalg.norm(e) + 1e-9)
