"""Shared 6-axis IMU physics feature extraction for LumoHub.

Matches the C99 implementation in LUMOHUB/firmware/lumo-hub/main/ble/lumo_imu_ai.c
bit-for-bit (float32 arithmetic, exact slice indices, ddof=0 population std).

Window layout: T = 30 samples at 10 Hz (100 ms/sample, 3.0 seconds total)
  Channels: [ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps]
  - Pre-impact phase : t in [0..8]   (first 0.9 s)
  - Impact phase     : t in [9..14]  (0.6 s peak impact region)
  - Post-impact phase: t in [15..29] (1.5 s recovery / stillness check)
"""

from __future__ import annotations

import numpy as np

WINDOW_LEN = 30
NUM_CHANNELS = 6
NUM_FEATURES = 16
NUM_CLASSES = 5

CLASS_NAMES = [
    "NORMAL_ADL",
    "WRIST_SHOCK_FALSE_ALERT",
    "DEVICE_DROP_OFF_WRIST",
    "SOFT_FALL_RECOVERING",
    "CRITICAL_FALL_IMMOBILE",
]

FEATURE_NAMES = [
    "f0_pre_min_mag_g",
    "f1_impact_peak_g",
    "f2_impact_mean_g",
    "f3_impact_width_ratio",
    "f4_impact_peak_gyro_norm",
    "f5_impact_gyro_rms",
    "f6_tilt_angle_norm",
    "f7_gravity_vertical_drop",
    "f8_post_mean_mag_g",
    "f9_post_std_mag_g",
    "f10_post_mean_gyro_norm",
    "f11_post_std_gyro_norm",
    "f12_post_micro_tremor_score",
    "f13_peak_jerk_g",
    "f14_recovery_energy_ratio",
    "f15_post_stillness_ratio",
]


def extract_features_single(window: np.ndarray) -> np.ndarray:
    """Extract 16 physics features from a single [30, 6] float32 IMU window."""
    w = np.asarray(window, dtype=np.float32)
    assert w.shape == (WINDOW_LEN, NUM_CHANNELS), f"Expected (30, 6), got {w.shape}"

    ax, ay, az = w[:, 0], w[:, 1], w[:, 2]
    gx, gy, gz = w[:, 3], w[:, 4], w[:, 5]

    mag_a = np.sqrt(ax * ax + ay * ay + az * az, dtype=np.float32)
    mag_g = np.sqrt(gx * gx + gy * gy + gz * gz, dtype=np.float32)
    gyro_norm = (mag_g / np.float32(250.0)).astype(np.float32)

    feats = np.zeros(NUM_FEATURES, dtype=np.float32)

    # 0. Pre-impact free-fall depth [0..11]
    feats[0] = np.min(mag_a[0:12])
    # 1. Global impact peak [0..29]
    feats[1] = np.max(mag_a)
    # 2. Impact phase mean |a| [9..14]
    feats[2] = np.mean(mag_a[9:15])
    # 3. Pulse width ratio where |a| >= 2.2g in [8..16] (9 samples)
    feats[3] = np.float32(np.sum(mag_a[8:17] >= np.float32(2.2)) / 9.0)
    # 4. Normalized peak gyro in [8..16]
    feats[4] = np.max(gyro_norm[8:17])
    # 5. Impact gyro RMS in [9..14] (6 samples)
    feats[5] = np.sqrt(np.mean(gyro_norm[9:15] * gyro_norm[9:15]))

    # 6. Gravity vector tilt angle delta_theta / 180.0
    a_pre = np.mean(w[0:8, 0:3], axis=0, dtype=np.float32)
    a_post = np.mean(w[18:30, 0:3], axis=0, dtype=np.float32)
    n_pre = np.sqrt(np.sum(a_pre * a_pre), dtype=np.float32)
    n_post = np.sqrt(np.sum(a_post * a_post), dtype=np.float32)
    dot = np.sum(a_pre * a_post, dtype=np.float32)
    cos_theta = dot / (n_pre * n_post + np.float32(1e-6))
    cos_theta = np.clip(cos_theta, np.float32(-1.0), np.float32(1.0))
    tilt_deg = np.arccos(cos_theta) * np.float32(180.0 / np.pi)
    feats[6] = np.float32(tilt_deg / 180.0)

    # 7. Collapse of primary upright gravity axis
    abs_pre = np.abs(a_pre)
    primary_axis = int(np.argmax(abs_pre))
    feats[7] = np.float32(abs_pre[primary_axis] - np.abs(a_post[primary_axis]))

    # 8-11. Post-impact acceleration & gyro mean/std [16..29] (14 samples, ddof=0)
    post_a = mag_a[16:30]
    post_g_norm = gyro_norm[16:30]
    feats[8] = np.mean(post_a)
    feats[9] = np.std(post_a, ddof=0)
    feats[10] = np.mean(post_g_norm)
    feats[11] = np.std(post_g_norm, ddof=0)

    # 12. Post-impact micro-tremor log score [18..29] (12 samples, ddof=0)
    micro_std = np.std(mag_g[18:30], ddof=0)
    feats[12] = np.float32(np.log10(micro_std + np.float32(1e-4)) / 4.0 + 1.0)

    # 13. Peak jerk around impact [8..15]
    jerks = np.abs(mag_a[8:16] - mag_a[7:15])
    feats[13] = np.max(jerks)

    # 14. Post-impact vs impact energy ratio
    post_energy = np.sum(np.abs(mag_a[16:30] - np.float32(1.0)) + mag_g[16:30] / np.float32(100.0))
    impact_energy = np.sum(np.abs(mag_a[9:15] - np.float32(1.0)) + mag_g[9:15] / np.float32(100.0))
    feats[14] = np.float32(post_energy / (impact_energy + np.float32(0.1)))

    # 15. Post-impact stillness ratio [16..29] (14 samples)
    still_mask = (mag_a[16:30] >= np.float32(0.85)) & (mag_a[16:30] <= np.float32(1.15)) & (mag_g[16:30] < np.float32(12.0))
    feats[15] = np.float32(np.sum(still_mask) / 14.0)

    return feats


def extract_features_batch(windows: np.ndarray) -> np.ndarray:
    """Vectorized extraction of 16 physics features for [N, 30, 6] float32 array."""
    w = np.asarray(windows, dtype=np.float32)
    assert w.ndim == 3 and w.shape[1:] == (WINDOW_LEN, NUM_CHANNELS)
    n = w.shape[0]
    out = np.zeros((n, NUM_FEATURES), dtype=np.float32)

    ax, ay, az = w[:, :, 0], w[:, :, 1], w[:, :, 2]
    gx, gy, gz = w[:, :, 3], w[:, :, 4], w[:, :, 5]

    mag_a = np.sqrt(ax * ax + ay * ay + az * az, dtype=np.float32)
    mag_g = np.sqrt(gx * gx + gy * gy + gz * gz, dtype=np.float32)
    gyro_norm = (mag_g / np.float32(250.0)).astype(np.float32)

    out[:, 0] = np.min(mag_a[:, 0:12], axis=1)
    out[:, 1] = np.max(mag_a, axis=1)
    out[:, 2] = np.mean(mag_a[:, 9:15], axis=1)
    out[:, 3] = np.sum(mag_a[:, 8:17] >= np.float32(2.2), axis=1).astype(np.float32) / np.float32(9.0)
    out[:, 4] = np.max(gyro_norm[:, 8:17], axis=1)
    out[:, 5] = np.sqrt(np.mean(gyro_norm[:, 9:15] ** 2, axis=1))

    a_pre = np.mean(w[:, 0:8, 0:3], axis=1, dtype=np.float32)
    a_post = np.mean(w[:, 18:30, 0:3], axis=1, dtype=np.float32)
    n_pre = np.sqrt(np.sum(a_pre * a_pre, axis=1), dtype=np.float32)
    n_post = np.sqrt(np.sum(a_post * a_post, axis=1), dtype=np.float32)
    dot = np.sum(a_pre * a_post, axis=1, dtype=np.float32)
    cos_theta = np.clip(dot / (n_pre * n_post + np.float32(1e-6)), np.float32(-1.0), np.float32(1.0))
    tilt_deg = np.arccos(cos_theta) * np.float32(180.0 / np.pi)
    out[:, 6] = tilt_deg / np.float32(180.0)

    abs_pre = np.abs(a_pre)
    primary_axis = np.argmax(abs_pre, axis=1)
    idx = np.arange(n)
    out[:, 7] = abs_pre[idx, primary_axis] - np.abs(a_post[idx, primary_axis])

    post_a = mag_a[:, 16:30]
    post_g_norm = gyro_norm[:, 16:30]
    out[:, 8] = np.mean(post_a, axis=1)
    out[:, 9] = np.std(post_a, axis=1, ddof=0)
    out[:, 10] = np.mean(post_g_norm, axis=1)
    out[:, 11] = np.std(post_g_norm, axis=1, ddof=0)

    micro_std = np.std(mag_g[:, 18:30], axis=1, ddof=0)
    out[:, 12] = np.log10(micro_std + np.float32(1e-4)) / np.float32(4.0) + np.float32(1.0)

    out[:, 13] = np.max(np.abs(mag_a[:, 8:16] - mag_a[:, 7:15]), axis=1)

    post_energy = np.sum(np.abs(mag_a[:, 16:30] - np.float32(1.0)) + mag_g[:, 16:30] / np.float32(100.0), axis=1)
    impact_energy = np.sum(np.abs(mag_a[:, 9:15] - np.float32(1.0)) + mag_g[:, 9:15] / np.float32(100.0), axis=1)
    out[:, 14] = post_energy / (impact_energy + np.float32(0.1))

    still_mask = (mag_a[:, 16:30] >= np.float32(0.85)) & (mag_a[:, 16:30] <= np.float32(1.15)) & (mag_g[:, 16:30] < np.float32(12.0))
    out[:, 15] = np.sum(still_mask, axis=1).astype(np.float32) / np.float32(14.0)

    return out
