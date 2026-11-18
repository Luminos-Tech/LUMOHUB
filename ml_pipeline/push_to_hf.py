"""Upload both trained LumoHub IMU models and the 5-class dataset to Hugging Face.

Target repositories under `fragilov`:
  1. fragilov/lumohub-imu-burst-1dcnn      (Approach 1: Raw [30x6] 1D-CNN)
  2. fragilov/lumohub-imu-event-mlp        (Approach 2: 16 Physics-Feature MLP + RF baseline)
  3. fragilov/lumohub-imu-fall-5class-dataset (Dataset + Dual Benchmark Report)
"""

from __future__ import annotations

import json
import os
import pathlib
from huggingface_hub import HfApi

from feature_extractor import CLASS_NAMES, FEATURE_NAMES

HF_TOKEN = os.environ.get("HF_TOKEN", "")
OWNER = "fragilov"


def build_cnn_readme(metrics: dict, cfg: dict) -> str:
    acc = metrics["test_accuracy"] * 100
    f1 = metrics["test_macro_f1"] * 100
    wfar = metrics["wrist_shock_false_alert_rate"] * 100
    dfar = metrics["device_drop_false_alert_rate"] * 100
    fsens = metrics["binary_fall_sensitivity"] * 100
    stress_rows = "\n".join(
        f"| {s['scenario']} | {s['cnn_acc']*100:.2f}% | {s['cnn_f1']*100:.2f}% |"
        for s in metrics.get("stress_tests", [])
    )
    return f"""---
license: mit
tags:
- tinyml
- esp32-s3
- imu
- fall-detection
- 1d-cnn
- mpu6050
- lumohub
metrics:
- accuracy
- f1
---

# LumoHub IMU Burst 1D-CNN (`LumoBurst1DCNN`) — Approach 1

End-to-end 1D Convolutional Neural Network running directly on the **LumoHub (`ESP32-S3`)** to eliminate wearable wrist false alerts and classify 3.0-second 6-axis IMU burst sequences (`30 samples x 6 channels` at `10 Hz`) received from the wearable over BLE (`0x03` MPU6050 diagnostic stream).

## Architecture
- **Input:** `[30, 6]` normalized `(ax/8g, ay/8g, az/8g, gx/250dps, gy/250dps, gz/250dps)`
- **Backbone:** `Conv1D(6->24, k=5)` + ReLU + `MaxPool(2)` -> `Conv1D(24->32, k=3)` + ReLU + `MaxPool(3)`
- **Dual Pooling:** Concatenated `GlobalAvgPool1D` + `GlobalMaxPool1D` (`64` dims)
- **Head:** `Linear(64->32)` + ReLU + `Linear(32->5)`
- **Footprint:** `{cfg['folded_parameters']:,}` parameters (`{cfg['flash_bytes']:,}` bytes Flash), `{cfg['macs_per_inference']:,}` MACs/inference

## 5-Class Taxonomy
0. `NORMAL_ADL` — Walking, sitting, lying down smoothly, everyday movement
1. `WRIST_SHOCK_FALSE_ALERT` — High-G wrist shock (clapping, table slap, knocking) without body collapse
2. `DEVICE_DROP_OFF_WRIST` — Unbuckled/dropped wearable onto a rigid surface (zero physiological tremor)
3. `SOFT_FALL_RECOVERING` — Stumble/fall with posture collapse followed by self-recovery motion
4. `CRITICAL_FALL_IMMOBILE` — Severe fall with posture collapse and prolonged immobility

## Benchmark Metrics (Held-out Test Split, N = 4,500)
- **Accuracy:** `{acc:.2f}%`
- **Macro F1:** `{f1:.2f}%`
- **Wrist-Shock False Alert Rate:** `{wfar:.2f}%`
- **Device Drop-Off False Alert Rate:** `{dfar:.2f}%`
- **Binary Fall Sensitivity (Classes 3 & 4):** `{fsens:.2f}%`

### BLE Packet Drop & Noise Stress Test
| Scenario | Accuracy | Macro F1 |
| :--- | :---: | :---: |
{stress_rows}

## Embedded Firmware Integration
Folded C99 weights are exported in `lumo_hub_cnn_model.h` for zero-dependency inference on `ESP32-S3` (`lumo_imu_ai.c`).
"""


def build_mlp_readme(metrics: dict, cfg: dict) -> str:
    acc = metrics["test_accuracy"] * 100
    f1 = metrics["test_macro_f1"] * 100
    wfar = metrics["wrist_shock_false_alert_rate"] * 100
    dfar = metrics["device_drop_false_alert_rate"] * 100
    fsens = metrics["binary_fall_sensitivity"] * 100
    feat_rows = "\n".join(
        f"| `{k}` | {v:.4f} |" for k, v in metrics.get("rf_feature_importances", {}).items()
    )
    stress_rows = "\n".join(
        f"| {s['scenario']} | {s['mlp_acc']*100:.2f}% | {s['mlp_f1']*100:.2f}% |"
        for s in metrics.get("stress_tests", [])
    )
    return f"""---
license: mit
tags:
- tinyml
- esp32-s3
- imu
- fall-detection
- physics-mlp
- mpu6050
- lumohub
metrics:
- accuracy
- f1
---

# LumoHub Physics-Feature Event MLP (`LumoEventMLP`) — Approach 2

Physics-informed Multi-Layer Perceptron running on **LumoHub (`ESP32-S3`)** using **16 biomechanical features** extracted from each 3.0-second 6-axis IMU burst window (`free-fall depth`, `impact pulse width`, `3D gravity tilt angle Δθ`, `upright axis collapse`, `post-impact micro-tremor log-score`, `peak jerk`, `recovery energy ratio`, etc.).

## Architecture
- **Input:** `16` standardized physics features (`(x - mean) / std`)
- **Head:** `Linear(16->64)` + ReLU -> `Linear(64->32)` + ReLU -> `Linear(32->5)` (BatchNorm folded into weights)
- **Footprint:** `{cfg['folded_parameters']:,}` floats (`{cfg['flash_bytes']:,}` bytes Flash), `{cfg['macs_per_inference']:,}` MACs/inference

## Benchmark Metrics (Held-out Test Split, N = 4,500)
- **Accuracy:** `{acc:.2f}%`
- **Macro F1:** `{f1:.2f}%`
- **Wrist-Shock False Alert Rate:** `{wfar:.2f}%`
- **Device Drop-Off False Alert Rate:** `{dfar:.2f}%`
- **Binary Fall Sensitivity (Classes 3 & 4):** `{fsens:.2f}%`

### BLE Packet Drop & Noise Stress Test
| Scenario | Accuracy | Macro F1 |
| :--- | :---: | :---: |
{stress_rows}

### Top Physics Features (Random Forest Baseline Importance)
| Feature | Importance |
| :--- | :---: |
{feat_rows}

## Embedded Firmware Integration
Folded C99 weights + scaler constants are exported in `lumo_hub_feature_model.h` (`lumo_imu_ai.c`).
"""


def build_dataset_readme(ds_info: dict, benchmark_md: str) -> str:
    return f"""---
license: mit
tags:
- imu
- fall-detection
- mpu6050
- sisfall
- wearable
- time-series
- lumohub
---

# LumoHub 5-Class 6-Axis IMU Fall & False-Alert Dataset

Curated 30,000-window (`900,000` 6-axis samples at `10 Hz`, `+-8g / +-250dps` MPU6050 quantized) dataset designed specifically for **eliminating false alerts** on wrist-worn IMU fall detectors paired with an intelligent Home Hub (`ESP32-S3`).

## Sources
- **SisFall Clinical IMU Dataset** (`MMA8451Q +-8g` + `ITG3200`, strictly split by subject ID across Train/Val/Test with zero subject leakage).
- **Kaggle MPU6050 Fall Detection** (`hnpawspham/mpu-fall-detect-classification` unclipped 10 Hz ADLs).
- **LumoHub 3D Rigid-Body & Wrist Biomechanics Simulator** (`SO(3)` 3D orientation invariance, realistic static gyro bias, physiological micro-tremor vs rigid-surface drop physics).

## Splits
- Train: `{ds_info['splits']['train']:,}` windows
- Val: `{ds_info['splits']['val']:,}` windows
- Test: `{ds_info['splits']['test']:,}` windows

{benchmark_md}
"""


def main() -> None:
    root = pathlib.Path(__file__).resolve().parent
    api = HfApi(token=HF_TOKEN)
    who = api.whoami()
    print(f"Authenticated to Hugging Face as: {who.get('name')} ({who.get('fullname')})")

    cnn_dir = root / "models" / "approach1_burst_cnn"
    mlp_dir = root / "models" / "approach2_event_mlp"
    data_dir = root / "data"

    # 1. Push Approach 1 (1D-CNN)
    cnn_repo = f"{OWNER}/lumohub-imu-burst-1dcnn"
    cnn_metrics = json.loads((cnn_dir / "evaluation_metrics.json").read_text(encoding="utf-8"))
    cnn_cfg = json.loads((cnn_dir / "config.json").read_text(encoding="utf-8"))
    (cnn_dir / "README.md").write_text(build_cnn_readme(cnn_metrics, cnn_cfg), encoding="utf-8")

    api.create_repo(repo_id=cnn_repo, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(cnn_dir),
        repo_id=cnn_repo,
        repo_type="model",
        commit_message="Upload LumoHub Approach 1 (LumoBurst1DCNN) weights, C99 header, and benchmarks",
    )
    print(f"[HF] Uploaded Approach 1 -> https://huggingface.co/{cnn_repo}")

    # 2. Push Approach 2 (Feature MLP)
    mlp_repo = f"{OWNER}/lumohub-imu-event-mlp"
    mlp_metrics = json.loads((mlp_dir / "evaluation_metrics.json").read_text(encoding="utf-8"))
    mlp_cfg = json.loads((mlp_dir / "config.json").read_text(encoding="utf-8"))
    (mlp_dir / "README.md").write_text(build_mlp_readme(mlp_metrics, mlp_cfg), encoding="utf-8")

    api.create_repo(repo_id=mlp_repo, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(mlp_dir),
        repo_id=mlp_repo,
        repo_type="model",
        commit_message="Upload LumoHub Approach 2 (LumoEventMLP) weights, scaler, C99 header, and benchmarks",
    )
    print(f"[HF] Uploaded Approach 2 -> https://huggingface.co/{mlp_repo}")

    # 3. Push Dataset + Benchmark Report
    ds_repo = f"{OWNER}/lumohub-imu-fall-5class-dataset"
    ds_info = json.loads((data_dir / "dataset_info.json").read_text(encoding="utf-8"))
    bench_md = (root / "BENCHMARK_REPORT.md").read_text(encoding="utf-8")
    (data_dir / "README.md").write_text(build_dataset_readme(ds_info, bench_md), encoding="utf-8")

    api.create_repo(repo_id=ds_repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(data_dir),
        repo_id=ds_repo,
        repo_type="dataset",
        commit_message="Upload LumoHub 5-class 6-axis IMU dataset and benchmark report",
    )
    api.upload_file(
        path_or_fileobj=str(root / "BENCHMARK_REPORT.md"),
        path_in_repo="BENCHMARK_REPORT.md",
        repo_id=ds_repo,
        repo_type="dataset",
    )
    print(f"[HF] Uploaded Dataset -> https://huggingface.co/datasets/{ds_repo}")

    # 4. Verify all 3 repos
    for rid, rtype in [(cnn_repo, "model"), (mlp_repo, "model"), (ds_repo, "dataset")]:
        files = api.list_repo_files(repo_id=rid, repo_type=rtype)
        print(f"Verified {rtype} repo '{rid}': {files}")


if __name__ == "__main__":
    main()
