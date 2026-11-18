"""Build 5-class 6-axis IMU dataset combining real SisFall/dataMPU traces and 3D biomechanics simulation.

Target format:
  - Sampling rate : 10 Hz (100 ms/sample, matches LumoBand burst mode BLE packet 0x03)
  - Window shape  : [30, 6] float32 -> (ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps)
  - Dynamic range : +-8g (quantized to 1/4096 g), +-250 dps (quantized to 1/131 dps)
  - Classes (5)   :
      0: NORMAL_ADL
      1: WRIST_SHOCK_FALSE_ALERT
      2: DEVICE_DROP_OFF_WRIST
      3: SOFT_FALL_RECOVERING
      4: CRITICAL_FALL_IMMOBILE

Outputs:
  LUMOHUB/ml_pipeline/data/lumohub_imu_5class_dataset.npz
"""

from __future__ import annotations

import json
import os
import pathlib
import zipfile
import numpy as np

from feature_extractor import (
    CLASS_NAMES,
    FEATURE_NAMES,
    NUM_CHANNELS,
    NUM_CLASSES,
    WINDOW_LEN,
    extract_features_batch,
)

ACCEL_LSB = np.float32(1.0 / 4096.0)
GYRO_LSB = np.float32(1.0 / 131.0)


def quantize_mpu6050(win: np.ndarray) -> np.ndarray:
    """Clip and quantize [..., 6] IMU window to exact MPU6050 (+-8g, +-250dps) resolution."""
    out = win.astype(np.float32).copy()
    out[..., 0:3] = np.clip(out[..., 0:3], -8.0, 8.0)
    out[..., 3:6] = np.clip(out[..., 3:6], -250.0, 250.0)
    out[..., 0:3] = np.round(out[..., 0:3] / ACCEL_LSB) * ACCEL_LSB
    out[..., 3:6] = np.round(out[..., 3:6] / GYRO_LSB) * GYRO_LSB
    return out.astype(np.float32)


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    n = np.linalg.norm(v)
    return v / max(n, 1e-8)


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Sample uniformly from SO(3) via QR decomposition."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    d = np.diag(np.sign(np.diag(r)))
    q = q @ d
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def rotate_vector_rodrigues(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate 3D vector v around unit axis by angle_rad."""
    axis = axis / max(np.linalg.norm(axis), 1e-8)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)


def make_ortho_axis(rng: np.random.Generator, v: np.ndarray) -> np.ndarray:
    """Return a random unit axis approximately orthogonal to v (for realistic tipping)."""
    u = random_unit_vector(rng)
    perp = u - np.dot(u, v) * v
    n = np.linalg.norm(perp)
    if n < 1e-3:
        perp = np.cross(v, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(perp) < 1e-3:
            perp = np.cross(v, np.array([0.0, 1.0, 0.0]))
        n = np.linalg.norm(perp)
    return perp / n


def sisfall_subject_split(subject: str) -> str:
    """Strict subject-wise split with zero subject overlap across train/val/test."""
    if subject.startswith("SA"):
        num = int(subject[2:])
        if num <= 16:
            return "train"
        elif num <= 19:
            return "val"
        else:
            return "test"
    if subject.startswith("SE"):
        num = int(subject[2:])
        if num <= 10:
            return "train"
        elif num <= 12:
            return "val"
        else:
            return "test"
    return "train"


def load_sisfall_windows(cache_dir: pathlib.Path, rng: np.random.Generator) -> dict[str, tuple[list[np.ndarray], list[int]]]:
    """Extract real 10 Hz [30, 6] windows from SisFall_dataset.zip using MMA8451Q (+-8g) & ITG3200."""
    zp = cache_dir / "sisfall" / "SisFall_dataset.zip"
    splits: dict[str, tuple[list[np.ndarray], list[int]]] = {
        "train": ([], []),
        "val": ([], []),
        "test": ([], []),
    }
    if not zp.is_file():
        print("[SisFall] Zip not found, skipping.")
        return splits

    # MMA8451Q: 14-bit +-8g -> 16 / 16384; ITG3200: 16-bit +-2000dps -> 4000 / 65536
    acc_scale = 16.0 / 16384.0
    gyro_scale = 4000.0 / 65536.0

    with zipfile.ZipFile(zp) as zf:
        txt_files = sorted(
            n for n in zf.namelist()
            if n.endswith(".txt") and "__MACOSX" not in n and "readme" not in n.lower()
        )
        for name in txt_files:
            base = os.path.basename(name)
            parts = base.replace(".txt", "").split("_")
            if len(parts) < 3:
                continue
            code, subj = parts[0], parts[1]
            split = sisfall_subject_split(subj)

            try:
                raw_txt = zf.read(name).decode("utf-8", errors="ignore").replace(";", " ").replace(",", " ")
                flat = np.fromstring(raw_txt, sep=" ", dtype=np.float32)
                if flat.size % 9 != 0:
                    continue
                arr = flat.reshape(-1, 9)
            except Exception:
                continue
            if arr.shape[0] < 650:
                continue

            acc200 = arr[:, 6:9].astype(np.float32) * np.float32(acc_scale)
            gyro200 = np.clip(arr[:, 3:6].astype(np.float32) * np.float32(gyro_scale), -250.0, 250.0)

            if code.startswith("F"):
                mag200 = np.linalg.norm(acc200, axis=1)
                p200 = int(np.argmax(mag200))
                if mag200[p200] < 2.2:
                    continue
                # Extract up to 2 phase-shifted 10Hz views in train (1 clean in val/test)
                offsets = [p200 % 20] if split != "train" else [p200 % 20, (p200 + 3) % 20]
                for idx_off, off in enumerate(offsets):
                    acc10 = acc200[off::20]
                    gyro10 = gyro200[off::20]
                    p10 = int(np.argmax(np.linalg.norm(acc10, axis=1)))
                    target_pos = 11 if idx_off == 0 else int(rng.integers(10, 14))
                    s0 = p10 - target_pos
                    if s0 < 0 or s0 + WINDOW_LEN > len(acc10):
                        continue
                    win = np.concatenate([acc10[s0:s0 + WINDOW_LEN], gyro10[s0:s0 + WINDOW_LEN]], axis=1)
                    # Determine recovering vs immobile from actual post-impact motion in window
                    post_g = np.linalg.norm(win[18:30, 3:6], axis=1)
                    post_a = np.linalg.norm(win[18:30, 0:3], axis=1)
                    post_g_std = float(np.std(post_g))
                    post_a_std = float(np.std(post_a))
                    # F07 (dampened fall on table) or high post-impact recovery movement
                    if post_g_std > 14.0 or post_a_std > 0.16:
                        label = 3  # SOFT_FALL_RECOVERING
                    else:
                        label = 4  # CRITICAL_FALL_IMMOBILE
                    splits[split][0].append(quantize_mpu6050(win))
                    splits[split][1].append(label)

            elif code.startswith("D"):
                # D18 is stumble while walking; treat D01..D17, D19 as NORMAL_ADL
                if code == "D18":
                    continue
                acc10 = acc200[::20]
                gyro10 = gyro200[::20]
                step = 60 if len(acc10) > 600 else 30
                max_wins = 3 if split == "train" else 2
                taken = 0
                # Also prioritize active windows (sitting/jogging/bending)
                candidates = list(range(0, len(acc10) - WINDOW_LEN + 1, step))
                rng.shuffle(candidates)
                for s0 in candidates:
                    if taken >= max_wins:
                        break
                    win = np.concatenate([acc10[s0:s0 + WINDOW_LEN], gyro10[s0:s0 + WINDOW_LEN]], axis=1)
                    splits[split][0].append(quantize_mpu6050(win))
                    splits[split][1].append(0)  # NORMAL_ADL
                    taken += 1

    print(
        f"[SisFall] Extracted windows -> "
        f"train={len(splits['train'][1])}, val={len(splits['val'][1])}, test={len(splits['test'][1])}"
    )
    return splits


def load_datamapu_adl(cache_dir: pathlib.Path, rng: np.random.Generator) -> dict[str, tuple[list[np.ndarray], list[int]]]:
    """Extract unclipped real MPU6050 10 Hz ADL windows (idle/walk/stair/exercise) from Kaggle dataMPU."""
    base = cache_dir / "hnpawspham__mpu-fall-detect-classification" / "dataMPU"
    splits: dict[str, tuple[list[np.ndarray], list[int]]] = {
        "train": ([], []),
        "val": ([], []),
        "test": ([], []),
    }
    if not base.is_dir():
        return splits

    for cls in ["idle", "walk", "stair", "exercise"]:
        cls_dir = base / cls
        if not cls_dir.is_dir():
            continue
        for sub_split in ["train", "test"]:
            s_dir = cls_dir / sub_split
            if not s_dir.is_dir():
                continue
            for csv_path in sorted(s_dir.glob("*.csv")):
                try:
                    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=range(7))
                except Exception:
                    continue
                if arr.ndim != 2 or arr.shape[0] < WINDOW_LEN:
                    continue
                acc_g = arr[:, 1:4].astype(np.float32) / np.float32(9.80665)
                gyro_dps = arr[:, 4:7].astype(np.float32) * np.float32(180.0 / np.pi)
                # Keep only windows well inside +-2g so clipping artifacts don't appear
                for s0 in range(0, len(arr) - WINDOW_LEN + 1, 30):
                    win = np.concatenate([acc_g[s0:s0 + WINDOW_LEN], gyro_dps[s0:s0 + WINDOW_LEN]], axis=1)
                    if np.max(np.abs(win[:, 0:3])) < 1.85:
                        dst = "train" if sub_split == "train" else ("val" if rng.random() < 0.5 else "test")
                        splits[dst][0].append(quantize_mpu6050(win))
                        splits[dst][1].append(0)
    print(
        f"[dataMPU ADL] Extracted unclipped windows -> "
        f"train={len(splits['train'][1])}, val={len(splits['val'][1])}, test={len(splits['test'][1])}"
    )
    return splits


def simulate_window(cls_id: int, rng: np.random.Generator) -> np.ndarray:
    """Generate one physics-calibrated 30x6 IMU window (10 Hz) in (ax,ay,az,gx,gy,gz)."""
    t = np.arange(WINDOW_LEN, dtype=np.float64)
    acc = np.zeros((WINDOW_LEN, 3), dtype=np.float64)
    gyro = np.zeros((WINDOW_LEN, 3), dtype=np.float64)

    # Random initial 3D mounting/arm gravity direction + static device gyro bias
    g0 = random_unit_vector(rng)
    tip_axis = make_ortho_axis(rng, g0)
    spin_axis = random_unit_vector(rng)
    gyro_bias = rng.uniform(-2.6, 2.6, size=3)  # realistic uncalibrated MPU6050 zero-rate bias

    if cls_id == 0:
        # NORMAL_ADL: walking, brisk gesturing, sitting down firmly, or lying down smoothly
        mode = rng.choice(["walk", "sit", "lie_slow", "arm_gesture"], p=[0.35, 0.25, 0.20, 0.20])
        if mode == "walk":
            freq = rng.uniform(1.1, 2.2)
            amp = rng.uniform(0.15, 0.65)
            phase = rng.uniform(0, 2 * np.pi)
            for i in range(WINDOW_LEN):
                ang = 0.25 * np.sin(2 * np.pi * freq * i * 0.1 + phase)
                gi = rotate_vector_rodrigues(g0, tip_axis, ang)
                acc[i] = gi * (1.0 + amp * np.sin(2 * np.pi * freq * i * 0.1 + phase))
                gyro[i] = tip_axis * (ang * 45.0 * np.cos(2 * np.pi * freq * i * 0.1 + phase))
        elif mode == "sit":
            # Quick sit down: mild 1.6-2.7g seat contact, upright torso/wrist returns <30 deg
            t_imp = rng.integers(10, 14)
            peak_add = rng.uniform(0.6, 1.7)
            for i in range(WINDOW_LEN):
                ang = 0.35 * np.exp(-((i - t_imp) / 3.5) ** 2)
                gi = rotate_vector_rodrigues(g0, tip_axis, ang)
                bump = peak_add * np.exp(-((i - t_imp) / 1.4) ** 2)
                acc[i] = gi * (1.0 + bump)
                gyro[i] = tip_axis * (45.0 * np.sin((i - t_imp) * 0.7) * np.exp(-abs(i - t_imp) / 3.0))
        elif mode == "lie_slow":
            # Smooth transition from sitting/standing to lying down on bed (NO high-g shock)
            final_angle = np.radians(rng.uniform(55.0, 95.0))
            t_mid = rng.uniform(10.0, 16.0)
            slope = rng.uniform(0.35, 0.65)
            for i in range(WINDOW_LEN):
                s = 1.0 / (1.0 + np.exp(-slope * (i - t_mid)))
                ang = final_angle * s
                gi = rotate_vector_rodrigues(g0, tip_axis, ang)
                ds = slope * s * (1.0 - s) * 10.0  # rad-scaled rate
                acc[i] = gi * (1.0 + rng.normal(0.0, 0.04))
                gyro[i] = tip_axis * np.degrees(final_angle * ds)
        else:
            # Everyday arm reaching / picking up object
            t_mid = rng.integers(9, 15)
            for i in range(WINDOW_LEN):
                ang = 0.45 * np.sin(0.35 * (i - t_mid))
                gi = rotate_vector_rodrigues(g0, tip_axis, ang)
                acc[i] = gi * (1.0 + 0.35 * np.exp(-((i - t_mid) / 2.2) ** 2))
                gyro[i] = tip_axis * (60.0 * np.exp(-((i - t_mid) / 2.5) ** 2))
        acc += rng.normal(0.0, rng.uniform(0.012, 0.04), size=(WINDOW_LEN, 3))
        gyro += rng.normal(0.0, rng.uniform(1.0, 4.0), size=(WINDOW_LEN, 3))

    elif cls_id == 1:
        # WRIST_SHOCK_FALSE_ALERT: clapping, banging table, chopping, knocking, vigorous wave
        sub = rng.choice(["table_slap", "clap_repeat", "door_knock", "arm_whip_rest"], p=[0.30, 0.25, 0.20, 0.25])
        t_imp = int(rng.integers(10, 14))
        peak_g = rng.uniform(3.2, 7.6)
        # No sustained free-fall; at most brief pre-swing dip >= 0.72g
        pre_dip = rng.uniform(0.75, 1.05)
        post_tilt_deg = rng.uniform(2.0, 28.0) if sub != "arm_whip_rest" else rng.uniform(18.0, 48.0)
        final_angle = np.radians(post_tilt_deg)
        shock_dir = random_unit_vector(rng)
        human_tremor_dps = rng.uniform(0.65, 3.8)
        is_post_active = (sub in ["clap_repeat", "door_knock"]) or (rng.random() < 0.45)

        for i in range(WINDOW_LEN):
            if i < t_imp - 1:
                gi = rotate_vector_rodrigues(g0, tip_axis, rng.normal(0, 0.08))
                scale = 1.0 - (1.0 - pre_dip) * np.exp(-((i - (t_imp - 1.5)) / 1.0) ** 2)
                acc[i] = gi * scale
                gyro[i] = tip_axis * rng.uniform(5.0, 35.0) * np.exp(-((i - (t_imp - 1)) / 2.0) ** 2)
            elif i == t_imp:
                # Narrow sharp shock spike (1-2 samples wide at 10Hz)
                acc[i] = g0 + shock_dir * peak_g * rng.uniform(0.85, 1.0)
                gyro[i] = spin_axis * rng.uniform(70.0, 235.0)
            elif i == t_imp + 1 and rng.random() < 0.45:
                # Occasional 2nd sample rebound/follow-through
                acc[i] = g0 - shock_dir * rng.uniform(1.2, 2.6)
                gyro[i] = -spin_axis * rng.uniform(40.0, 140.0)
            else:
                progress = min(1.0, (i - t_imp) / 4.0)
                gi = rotate_vector_rodrigues(g0, tip_axis, final_angle * progress)
                if sub == "clap_repeat" and i > t_imp + 2 and rng.random() < 0.25:
                    acc[i] = gi * rng.uniform(2.0, 4.2)
                    gyro[i] = spin_axis * rng.uniform(50.0, 150.0)
                elif is_post_active:
                    wiggle = rng.uniform(0.08, 0.35) * np.sin(i * 1.1)
                    acc[i] = gi * (1.0 + wiggle)
                    gyro[i] = spin_axis * rng.uniform(12.0, 55.0) * np.cos(i * 0.9)
                else:
                    # Hard negative: hand rests calmly on table/lap after shock, WITH living micro-tremor
                    acc[i] = gi * (1.0 + rng.normal(0.0, 0.015))
                    gyro[i] = rng.normal(0.0, human_tremor_dps, size=3)

        acc += rng.normal(0.0, 0.012, size=(WINDOW_LEN, 3))
        gyro += rng.normal(0.0, 0.45, size=(WINDOW_LEN, 3))

    elif cls_id == 2:
        # DEVICE_DROP_OFF_WRIST: unbuckled & dropped onto floor/desk
        t_imp = int(rng.integers(10, 14))
        ff_len = int(rng.integers(2, 4))  # 200-400 ms free-fall
        peak_g = rng.uniform(3.6, 8.0)
        final_angle = np.radians(rng.uniform(25.0, 165.0))
        impact_dir = random_unit_vector(rng)
        bounce_dir = random_unit_vector(rng)
        g_post = rotate_vector_rodrigues(g0, tip_axis, final_angle)
        wobble_until = t_imp + (int(rng.integers(2, 5)) if rng.random() < 0.25 else 1)

        for i in range(WINDOW_LEN):
            if i < t_imp - ff_len:
                # Being removed from wrist
                acc[i] = g0 * rng.uniform(0.85, 1.25) + rng.normal(0, 0.05, size=3)
                gyro[i] = spin_axis * rng.uniform(10.0, 60.0)
            elif i < t_imp:
                # Parabolic free-fall + unconstrained mid-air spin
                acc[i] = rng.normal(0.0, 0.14, size=3)  # |a| ~ 0.05 - 0.38g
                gyro[i] = spin_axis * rng.uniform(130.0, 320.0)
            elif i == t_imp:
                # Sharp rigid surface impact
                acc[i] = impact_dir * peak_g
                gyro[i] = spin_axis * rng.uniform(110.0, 245.0)
            elif i <= wobble_until:
                # Short elastic bounce/clatter on hard desk/floor
                decay = np.exp(-(i - t_imp) * 0.9)
                acc[i] = g_post + bounce_dir * (peak_g * 0.45 * decay)
                gyro[i] = spin_axis * (rng.uniform(40.0, 160.0) * decay)
            else:
                # Dead-flat rigid surface: ONLY MPU6050 electronic noise floor (no living tremor!)
                acc[i] = g_post + rng.normal(0.0, rng.uniform(0.001, 0.004), size=3)
                gyro[i] = rng.normal(0.0, rng.uniform(0.03, 0.18), size=3)

    elif cls_id == 3:
        # SOFT_FALL_RECOVERING: stumble / fall with posture collapse then self-recovery within 1.5-3.0s
        t_imp = int(rng.integers(9, 13))
        ff_len = int(rng.integers(1, 4))
        peak_g = rng.uniform(3.0, 7.2)
        collapse_angle = np.radians(rng.uniform(45.0, 115.0))
        impact_dir = random_unit_vector(rng)
        g_floor = rotate_vector_rodrigues(g0, tip_axis, collapse_angle)
        rec_start = int(rng.integers(15, 22))  # starts pushing up / moving again
        rec_amp = rng.uniform(28.0, 135.0)
        rec_freq = rng.uniform(0.8, 2.0)

        for i in range(WINDOW_LEN):
            if i < t_imp - ff_len:
                acc[i] = g0 * rng.uniform(0.90, 1.25) + rng.normal(0, 0.06, size=3)
                gyro[i] = tip_axis * rng.uniform(-25.0, 25.0)
            elif i < t_imp:
                # Lunge / partial weightlessness
                ff_mag = rng.uniform(0.28, 0.72)
                gi = rotate_vector_rodrigues(g0, tip_axis, collapse_angle * 0.35)
                acc[i] = gi * ff_mag
                gyro[i] = tip_axis * rng.uniform(60.0, 180.0)
            elif t_imp <= i <= t_imp + 1:
                frac = 1.0 if i == t_imp else rng.uniform(0.48, 0.75)
                acc[i] = g_floor + impact_dir * (peak_g * frac)
                gyro[i] = tip_axis * rng.uniform(80.0, 220.0) * frac
            elif i < rec_start:
                # Brief stun before trying to get up
                acc[i] = g_floor * rng.uniform(0.92, 1.08) + rng.normal(0, 0.03, size=3)
                gyro[i] = rng.normal(0.0, rng.uniform(1.5, 6.0), size=3)
            else:
                # Active recovery: pushing on hands, rolling, lifting torso
                dt = (i - rec_start) * 0.1
                rec_ang = collapse_angle * max(0.2, 1.0 - 0.55 * min(1.0, dt / 1.2))
                gi = rotate_vector_rodrigues(g0, tip_axis, rec_ang)
                acc[i] = gi * (1.0 + rng.uniform(0.15, 0.55) * np.sin(2 * np.pi * rec_freq * dt + 0.4))
                gyro[i] = tip_axis * (rec_amp * np.cos(2 * np.pi * rec_freq * dt)) + rng.normal(0, 8.0, size=3)

        acc += rng.normal(0.0, 0.015, size=(WINDOW_LEN, 3))
        gyro += rng.normal(0.0, 0.6, size=(WINDOW_LEN, 3))

    else:
        # CRITICAL_FALL_IMMOBILE: severe fall + posture collapse + post-impact immobility with living tremor
        t_imp = int(rng.integers(9, 13))
        ff_len = int(rng.integers(2, 4))
        peak_g = rng.uniform(3.4, 8.2)
        collapse_angle = np.radians(rng.uniform(52.0, 125.0))
        impact_dir = random_unit_vector(rng)
        g_floor = rotate_vector_rodrigues(g0, tip_axis, collapse_angle)
        pulse_extra = int(rng.integers(1, 3))  # wide 2-3 sample body impact
        breath_freq = rng.uniform(0.20, 0.38)
        tremor_dps = rng.uniform(0.75, 4.2)

        for i in range(WINDOW_LEN):
            if i < t_imp - ff_len:
                acc[i] = g0 * rng.uniform(0.90, 1.20) + rng.normal(0, 0.04, size=3)
                gyro[i] = tip_axis * rng.uniform(-20.0, 20.0)
            elif i < t_imp:
                ff_mag = rng.uniform(0.15, 0.65)
                gi = rotate_vector_rodrigues(g0, tip_axis, collapse_angle * 0.45)
                acc[i] = gi * ff_mag
                gyro[i] = tip_axis * rng.uniform(75.0, 230.0)
            elif t_imp <= i <= t_imp + pulse_extra:
                decay = 1.0 if i == t_imp else np.exp(-0.75 * (i - t_imp))
                acc[i] = g_floor + impact_dir * (peak_g * decay)
                gyro[i] = tip_axis * rng.uniform(70.0, 225.0) * decay
            elif i <= t_imp + pulse_extra + 2:
                # Limb settling on floor right after impact
                acc[i] = g_floor + rng.normal(0.0, 0.10, size=3)
                gyro[i] = rng.normal(0.0, rng.uniform(6.0, 18.0), size=3)
            else:
                # Immobile on floor, WITH subtle living physiological breathing/tremor
                breath = 0.015 * np.sin(2 * np.pi * breath_freq * i * 0.1)
                acc[i] = g_floor * (1.0 + breath) + rng.normal(0.0, rng.uniform(0.008, 0.025), size=3)
                gyro[i] = rng.normal(0.0, tremor_dps, size=3)

    # Apply static device gyro zero-rate bias uniformly across ALL classes
    gyro += gyro_bias[None, :]
    win = np.concatenate([acc, gyro], axis=1)
    return quantize_mpu6050(win)


def build_split(
    split_name: str,
    target_per_class: int,
    real_sisfall: tuple[list[np.ndarray], list[int]],
    real_datamapu: tuple[list[np.ndarray], list[int]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    buckets_x: dict[int, list[np.ndarray]] = {c: [] for c in range(NUM_CLASSES)}
    buckets_src: dict[int, dict[str, int]] = {c: {"real": 0, "sim": 0} for c in range(NUM_CLASSES)}

    for x, y in zip(*real_sisfall):
        buckets_x[y].append(x)
        buckets_src[y]["real"] += 1
    for x, y in zip(*real_datamapu):
        buckets_x[y].append(x)
        buckets_src[y]["real"] += 1

    # Cap or top up each class cleanly to target_per_class
    xs_all: list[np.ndarray] = []
    ys_all: list[int] = []

    for c in range(NUM_CLASSES):
        real_list = buckets_x[c]
        if len(real_list) > target_per_class:
            idx = rng.choice(len(real_list), size=target_per_class, replace=False)
            selected = [real_list[i] for i in idx]
            n_real = target_per_class
            n_sim = 0
        else:
            selected = list(real_list)
            n_real = len(selected)
            # Keep a minimum healthy blend of simulator examples for full +-8g / SO(3) coverage
            n_sim = target_per_class - n_real
            for _ in range(n_sim):
                selected.append(simulate_window(c, rng))

        print(f"  [{split_name:5s}] Class {c} ({CLASS_NAMES[c]:24s}): real={n_real:4d}, sim={n_sim:4d}, total={len(selected):4d}")
        xs_all.extend(selected)
        ys_all.extend([c] * len(selected))

    X = np.stack(xs_all, axis=0).astype(np.float32)
    y = np.array(ys_all, dtype=np.int64)
    perm = rng.permutation(len(y))
    X = X[perm]
    y = y[perm]
    F = extract_features_batch(X)
    return X, y, F


def main() -> None:
    rng = np.random.default_rng(20261118)
    root = pathlib.Path(__file__).resolve().parent
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = pathlib.Path(os.environ.get("LOCALAPPDATA", str(data_dir))) / "lumo_public_cache"

    print("Loading real public IMU datasets from:", cache_dir)
    sisfall_splits = load_sisfall_windows(cache_dir, rng)
    datamapu_splits = load_datamapu_adl(cache_dir, rng)

    print("\nAssembling balanced 5-class splits (30,000 windows total):")
    X_train, y_train, F_train = build_split("train", 4200, sisfall_splits["train"], datamapu_splits["train"], rng)
    X_val, y_val, F_val = build_split("val", 900, sisfall_splits["val"], datamapu_splits["val"], rng)
    X_test, y_test, F_test = build_split("test", 900, sisfall_splits["test"], datamapu_splits["test"], rng)

    out_npz = data_dir / "lumohub_imu_5class_dataset.npz"
    np.savez_compressed(
        out_npz,
        X_train=X_train, y_train=y_train, F_train=F_train,
        X_val=X_val, y_val=y_val, F_val=F_val,
        X_test=X_test, y_test=y_test, F_test=F_test,
    )

    meta = {
        "window_len": WINDOW_LEN,
        "num_channels": NUM_CHANNELS,
        "num_features": len(FEATURE_NAMES),
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "feature_names": FEATURE_NAMES,
        "sampling_rate_hz": 10,
        "splits": {
            "train": int(len(y_train)),
            "val": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "sources": [
            "SisFall clinical IMU dataset (MMA8451Q +-8g + ITG3200, subject-split)",
            "Kaggle MPU6050 Fall Detection unclipped ADLs (hnpawspham/mpu-fall-detect-classification)",
            "LumoHub 3D Rigid-Body & Wrist Biomechanics Simulator (+-8g / +-250dps MPU6050 quantized)",
        ],
    }
    (data_dir / "dataset_info.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nSaved dataset to {out_npz} ({out_npz.stat().st_size / (1024 * 1024):.2f} MB)")


if __name__ == "__main__":
    main()
