"""The pretrained trunk from net.py, run in plain NumPy.

This is what lets the website use the deep encoder. torch is several hundred
megabytes and needs more memory than the 512 MB machine the site runs on, but
the network is small enough that its forward pass is a handful of matrix
multiplies. train/export_numpy.py folds each BatchNorm into the convolution in
front of it and writes the result to an .npz, so inference here is only
convolutions, ReLUs, and two linear layers.

Layout is NHWC throughout, so every convolution is one matmul over
(positions * 64 squares) rows. A 3x3 convolution gathers the nine shifted
copies of the padded board into one row per square first.

The math is identical to net.py in eval mode. tests/test_npnet.py checks the
output against reference vectors taken from the torch model.
"""

from pathlib import Path

import numpy as np

DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "pos-v1.npz"


def _conv3x3(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """x is (B, 8, 8, C), w is (9 * C, C_out) in (dy, dx, c) order."""
    n, c = x.shape[0], x.shape[3]
    p = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    cols = np.concatenate([p[:, dy:dy + 8, dx:dx + 8, :] for dy in range(3) for dx in range(3)], axis=3)
    return (cols.reshape(n * 64, 9 * c) @ w + b).reshape(n, 8, 8, -1)


def _conv1x1(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    n, c = x.shape[0], x.shape[3]
    return (x.reshape(n * 64, c) @ w + b).reshape(n, 8, 8, -1)


def _flatten_chw(x: np.ndarray) -> np.ndarray:
    """torch's Flatten runs over (C, H, W), so the channel axis has to lead."""
    return x.transpose(0, 3, 1, 2).reshape(x.shape[0], -1)


def _log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


class NumpyNet:
    def __init__(self, path: str | Path = DEFAULT_WEIGHTS):
        blob = np.load(path)
        self.w = {k: blob[k].astype(np.float32) for k in blob.files if not k.startswith("meta_")}
        self.name = str(blob["meta_name"])
        self.fingerprint = str(blob["meta_fingerprint"])
        self.dim = int(blob["meta_dim"])
        self.blocks = int(blob["meta_blocks"])

    def _trunk(self, planes: np.ndarray) -> np.ndarray:
        """planes is (B, 18, 8, 8), the same input net.py takes."""
        w = self.w
        x = np.ascontiguousarray(planes.transpose(0, 2, 3, 1), dtype=np.float32)
        x = np.maximum(_conv3x3(x, w["stem_w"], w["stem_b"]), 0)
        for i in range(self.blocks):
            h = np.maximum(_conv3x3(x, w[f"b{i}_c1_w"], w[f"b{i}_c1_b"]), 0)
            x = np.maximum(x + _conv3x3(h, w[f"b{i}_c2_w"], w[f"b{i}_c2_b"]), 0)
        return x

    def embed(self, planes: np.ndarray) -> np.ndarray:
        w = self.w
        h = np.maximum(_conv1x1(self._trunk(planes), w["embed_conv_w"], w["embed_conv_b"]), 0)
        v = _flatten_chw(h) @ w["embed_fc_w"] + w["embed_fc_b"]
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(n, 1e-12)

    def policy(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = self.w
        p = _flatten_chw(np.maximum(_conv1x1(self._trunk(planes), w["policy_conv_w"], w["policy_conv_b"]), 0))
        return (_log_softmax(p @ w["from_fc_w"] + w["from_fc_b"]),
                _log_softmax(p @ w["to_fc_w"] + w["to_fc_b"]))
