# LumoHub Dual-Architecture IMU TinyML Benchmark Report

## 1. Test Set Comparison (N = 4,500 held-out windows: SisFall subjects SA20-23/SE13-15 + Biomechanics Test Split)

| Model | Accuracy | Macro F1 | Wrist-Shock False Alert | Drop-Off False Alert | Binary Fall Sensitivity | Flash Weights | MACs / Inf |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Approach 1: LumoBurst1DCNN** | **98.38%** | **98.38%** | 0.11% | 0.00% | 99.33% | 21,300 B | 58,368 |
| **Approach 2: LumoEventMLP** | **99.73%** | **99.73%** | 0.00% | 0.00% | 99.94% | 13,460 B | 3,232 |
| Baseline: RandomForest-120 | 99.89% | 99.89% | 0.00% | 0.00% | 100.00% | — | — |
| Dual Ensemble (CNN + MLP) | 99.49% | 99.49% | 0.00% | 0.00% | 99.78% | 34,760 B | 61,600 |

## 2. BLE Packet Loss & Noise Stress Test

| Stress Scenario | 1D-CNN Acc | 1D-CNN Macro F1 | Feature-MLP Acc | Feature-MLP Macro F1 |
| :--- | :---: | :---: | :---: | :---: |
| Clean 10Hz BLE (0% loss) | 98.38% | 98.38% | 99.73% | 99.73% |
| BLE Packet Drop 10% | 93.60% | 93.62% | 96.56% | 96.58% |
| BLE Packet Drop 25% | 85.18% | 85.28% | 89.49% | 89.73% |
| BLE Packet Drop 40% | 75.16% | 75.40% | 80.29% | 80.99% |
| High Sensor Noise (2x) | 98.44% | 98.44% | 99.73% | 99.73% |

## 3. Top Physics Feature Importances (Random Forest Analysis)

| Feature | Importance |
| :--- | :---: |
| `f12_post_micro_tremor_score` | 0.1733 |
| `f0_pre_min_mag_g` | 0.1425 |
| `f6_tilt_angle_norm` | 0.0904 |
| `f3_impact_width_ratio` | 0.0863 |
| `f14_recovery_energy_ratio` | 0.0858 |
| `f11_post_std_gyro_norm` | 0.0775 |
| `f9_post_std_mag_g` | 0.0563 |
| `f10_post_mean_gyro_norm` | 0.0550 |
| `f13_peak_jerk_g` | 0.0520 |
| `f2_impact_mean_g` | 0.0508 |
| `f1_impact_peak_g` | 0.0412 |
| `f15_post_stillness_ratio` | 0.0321 |
| `f4_impact_peak_gyro_norm` | 0.0192 |
| `f5_impact_gyro_rms` | 0.0162 |
| `f7_gravity_vertical_drop` | 0.0117 |
| `f8_post_mean_mag_g` | 0.0096 |
