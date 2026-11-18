"""Verify the 5 firmware HIL benchmark scenarios in Python against exported folded weights."""

from __future__ import annotations

import pathlib
import numpy as np

from feature_extractor import CLASS_NAMES, extract_features_single
from models import numpy_forward_cnn, numpy_forward_mlp


def make_sim_scenarios() -> list[tuple[str, int, np.ndarray]]:
    def standing(noise_a: float, noise_g: float) -> np.ndarray:
        w = np.zeros((30, 6), dtype=np.float32)
        for t in range(30):
            jitter = ((t % 3) - 1) * noise_a
            gj = (1.0 if (t % 2 == 0) else -1.0) * noise_g
            w[t, 0] = jitter * 0.5
            w[t, 1] = jitter * 0.5
            w[t, 2] = 1.0 + jitter
            w[t, 3] = 1.2 + gj
            w[t, 4] = -0.8 + gj * 0.6
            w[t, 5] = 0.4 - gj * 0.5
        return w

    scenarios = []

    # 0. Normal Walking 1.4 Hz
    w0 = standing(0.18, 18.0)
    for t in range(30):
        w0[t, 2] = 1.0 + 0.28 * np.sin(t * 0.85)
        w0[t, 0] = 0.15 * np.cos(t * 0.85)
    scenarios.append(("1) Normal Walking 1.4Hz", 0, w0))

    # 1. Wrist Slap / Clap 5.2g (False Alert)
    w1 = standing(0.02, 1.5)
    w1[11, 0], w1[11, 2] = 4.9, 1.8
    w1[11, 3], w1[11, 4] = 145.0, -95.0
    for t in range(12, 30):
        w1[t, 0], w1[t, 2] = 0.21, 0.98
        w1[t, 3] = 1.2 + (1.8 if t % 2 == 0 else -1.4)
        w1[t, 4] = -0.8 + (1.5 if t % 3 == 0 else -1.2)
    scenarios.append(("2) Wrist Slap/Clap 5.2g", 1, w1))

    # 2. Dropped Band on Desk 5.8g (False Alert)
    w2 = standing(0.02, 15.0)
    for t in range(8, 11):
        w2[t, 0], w2[t, 1], w2[t, 2] = 0.08, -0.10, 0.12
        w2[t, 3], w2[t, 4], w2[t, 5] = 180.0, -140.0, 95.0
    w2[11, 0], w2[11, 1], w2[11, 2] = 4.2, 3.6, 1.6
    w2[11, 3], w2[11, 4] = 210.0, -110.0
    w2[12, 0], w2[12, 1], w2[12, 2] = 1.45, 0.45, 0.35
    w2[12, 3], w2[12, 4], w2[12, 5] = 45.0, -30.0, 15.0
    for t in range(13, 30):
        w2[t, 0], w2[t, 1], w2[t, 2] = 0.98, 0.12, 0.15
        w2[t, 3] = 1.2 + (0.06 if t % 2 == 0 else -0.06)
        w2[t, 4] = -0.8 + (-0.05 if t % 2 == 0 else 0.05)
        w2[t, 5] = 0.4
    scenarios.append(("3) Dropped Band on Desk 5.8g", 2, w2))

    # 3. Soft Fall + Recovery 4.4g
    w3 = standing(0.03, 5.0)
    for t in range(9, 11):
        w3[t, 0], w3[t, 2] = 0.25, 0.40
        w3[t, 3], w3[t, 4] = 110.0, 75.0
    w3[11, 0], w3[11, 2], w3[11, 3] = 3.6, 2.4, 155.0
    w3[12, 0], w3[12, 2], w3[12, 3] = 2.4, 1.3, 90.0
    for t in range(13, 30):
        rec = 1.0 if t >= 17 else 0.0
        w3[t, 0] = 0.85 - 0.35 * rec * np.sin((t - 17) * 0.5)
        w3[t, 2] = 0.50 + 0.45 * rec * np.cos((t - 17) * 0.5)
        w3[t, 3] = 1.2 + rec * 65.0 * np.sin((t - 17) * 0.8)
        w3[t, 4] = -0.8 + rec * 45.0 * np.cos((t - 17) * 0.8)
    scenarios.append(("4) Soft Fall + Recovery 4.4g", 3, w3))

    # 4. Critical Hard Fall 5.6g
    w4 = standing(0.03, 4.0)
    for t in range(8, 11):
        w4[t, 0], w4[t, 2] = 0.18, 0.28
        w4[t, 3], w4[t, 4] = 140.0, 95.0
    w4[11, 0], w4[11, 1], w4[11, 2], w4[11, 3] = 4.5, 1.8, 2.7, 175.0
    w4[12, 0], w4[12, 1], w4[12, 2], w4[12, 3] = 2.9, 1.2, 1.6, 105.0
    for t in range(14, 30):
        w4[t, 0] = 0.99 + 0.015 * np.sin(t * 0.4)
        w4[t, 1] = 0.08
        w4[t, 2] = 0.09
        w4[t, 3] = 1.2 + (2.1 if t % 2 == 0 else -1.9)
        w4[t, 4] = -0.8 + (1.7 if t % 3 == 0 else -1.5)
        w4[t, 5] = 0.4 + (0.9 if t % 2 == 0 else -0.8)
    scenarios.append(("5) Critical Hard Fall 5.6g", 4, w4))

    return scenarios


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def main() -> None:
    root = pathlib.Path(__file__).resolve().parent
    cnn_w = dict(np.load(root / "models" / "approach1_burst_cnn" / "weights_folded.npz"))
    mlp_npz = dict(np.load(root / "models" / "approach2_event_mlp" / "weights_folded.npz"))
    scaler_mean = mlp_npz.pop("scaler_mean")
    scaler_std = mlp_npz.pop("scaler_std")

    all_ok = True
    for name, expected, win in make_sim_scenarios():
        feats = extract_features_single(win)
        p_cnn = softmax(numpy_forward_cnn(win[None], cnn_w)[0])
        p_mlp = softmax(numpy_forward_mlp(feats[None], mlp_npz, scaler_mean, scaler_std)[0])
        p_ens = 0.30 * p_cnn + 0.70 * p_mlp
        c_cls, m_cls, e_cls = int(np.argmax(p_cnn)), int(np.argmax(p_mlp)), int(np.argmax(p_ens))
        ok = (m_cls == expected) and (e_cls == expected)
        all_ok = all_ok and ok
        print(
            f"{name:30s} | exp={CLASS_NAMES[expected]:23s} | "
            f"CNN={CLASS_NAMES[c_cls]:23s}({p_cnn[c_cls]*100:4.1f}%) | "
            f"MLP={CLASS_NAMES[m_cls]:23s}({p_mlp[m_cls]*100:4.1f}%) | "
            f"FUSED={CLASS_NAMES[e_cls]:23s}({p_ens[e_cls]*100:4.1f}%) | ok={ok}"
        )
    assert all_ok, "One or more firmware benchmark scenarios failed!"
    print("\nALL 5 FIRMWARE HIL BENCHMARK SCENARIOS PASSED 100% ON MLP + FUSED HYBRID!")


if __name__ == "__main__":
    main()
