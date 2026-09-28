"""Convert a torch checkpoint from pretrain.py into the .npz the website runs.

    python -m train.export_numpy models/pos-v1.pt backend/patterns/weights/pos-v1.npz

Each BatchNorm is folded into the convolution in front of it, using the running
statistics, which is exactly what eval mode computes. The result needs nothing
but NumPy to run (backend/patterns/npnet.py).
"""

import argparse

import numpy as np
import torch


def _fold(sd: dict, conv: str, bn: str, eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """Conv weight (out, in, kh, kw) plus BatchNorm into one (kh*kw*in, out)
    matrix and a bias, in the (dy, dx, c) order npnet gathers patches in."""
    w = sd[f"{conv}.weight"].double()
    scale = sd[f"{bn}.weight"].double() / torch.sqrt(sd[f"{bn}.running_var"].double() + eps)
    bias = sd[f"{bn}.bias"].double() - sd[f"{bn}.running_mean"].double() * scale
    w = w * scale[:, None, None, None]
    out, cin, kh, kw = w.shape
    mat = w.permute(2, 3, 1, 0).reshape(kh * kw * cin, out)
    return mat.float().numpy(), bias.float().numpy()


def export(src: str, dst: str) -> None:
    blob = torch.load(src, map_location="cpu", weights_only=True)
    sd, cfg = blob["model"], blob["config"]
    out: dict[str, np.ndarray] = {}

    out["stem_w"], out["stem_b"] = _fold(sd, "stem.0", "stem.1")
    for i in range(cfg["blocks"]):
        out[f"b{i}_c1_w"], out[f"b{i}_c1_b"] = _fold(sd, f"trunk.{i}.c1", f"trunk.{i}.b1")
        out[f"b{i}_c2_w"], out[f"b{i}_c2_b"] = _fold(sd, f"trunk.{i}.c2", f"trunk.{i}.b2")
    out["embed_conv_w"], out["embed_conv_b"] = _fold(sd, "embed_head.0", "embed_head.1")
    out["policy_conv_w"], out["policy_conv_b"] = _fold(sd, "policy_head.0", "policy_head.1")
    for name, key in (("embed_fc", "embed_head.4"), ("from_fc", "from_fc"), ("to_fc", "to_fc")):
        out[f"{name}_w"] = sd[f"{key}.weight"].T.contiguous().numpy()
        out[f"{name}_b"] = sd[f"{key}.bias"].numpy()

    np.savez_compressed(
        dst, **out,
        meta_name=np.array(cfg["name"]), meta_fingerprint=np.array(blob["fingerprint"]),
        meta_dim=np.array(cfg["dim"]), meta_blocks=np.array(cfg["blocks"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    a = ap.parse_args()
    export(a.checkpoint, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
