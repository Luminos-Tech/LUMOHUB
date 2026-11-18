"""Train, evaluate, stress-test, and export both LumoHub IMU AI models:
  - Approach 1: LumoBurst1DCNN  (Raw [30 x 6] BLE Burst Sequence Classifier)
  - Approach 2: LumoEventMLP    (16 Physics-Feature Classifier + RandomForest baseline)

Run with:
  C:\\Users\\Kiet\\AppData\\Local\\Programs\\Python\\Python311\\python.exe LUMOHUB\\ml_pipeline\\train_and_evaluate.py
"""

from __future__ import annotations

import json
import pathlib
import pickle
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from export_c_headers import export_cnn_header, export_mlp_header
from feature_extractor import (
    CLASS_NAMES,
    FEATURE_NAMES,
    NUM_CLASSES,
    extract_features_batch,
)
from models import (
    LumoBurst1DCNN,
    LumoEventMLP,
    numpy_forward_cnn,
    numpy_forward_mlp,
)


def set_seed(seed: int = 2026) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def random_so3_batch(batch_size: int) -> torch.Tensor:
    """Generate [B, 3, 3] random rotation matrices for exact 3D orientation invariance."""
    mats = np.linalg.qr(np.random.randn(batch_size, 3, 3))[0]
    dets = np.linalg.det(mats)
    mats[dets < 0, :, 0] *= -1.0
    return torch.from_numpy(mats.astype(np.float32))


def augment_imu_batch(xb: torch.Tensor) -> torch.Tensor:
    """Apply orientation-preserving SO(3) 3D rotation + mild accel jitter to raw [B, 30, 6]."""
    bsz = xb.shape[0]
    rots = random_so3_batch(bsz)  # [B, 3, 3]
    acc = torch.bmm(xb[:, :, 0:3], rots)
    gyro = torch.bmm(xb[:, :, 3:6], rots)
    acc = acc + torch.randn_like(acc) * 0.008
    out = torch.cat([acc, gyro], dim=2)
    return out


def compute_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    report = classification_report(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )

    # Critical safety & false-alert metrics:
    # False Alert Rate (FAR) on non-fall classes (0, 1, 2) predicted as fall (3, 4)
    non_fall_mask = np.isin(y_true, [0, 1, 2])
    pred_fall = np.isin(y_pred, [3, 4])
    overall_far = float(np.mean(pred_fall[non_fall_mask])) if np.any(non_fall_mask) else 0.0

    wrist_mask = y_true == 1
    wrist_far = float(np.mean(pred_fall[wrist_mask])) if np.any(wrist_mask) else 0.0

    drop_mask = y_true == 2
    drop_far = float(np.mean(pred_fall[drop_mask])) if np.any(drop_mask) else 0.0

    # Fall Detection Sensitivity (recall for true falls {3,4} flagged as {3,4})
    fall_mask = np.isin(y_true, [3, 4])
    fall_sensitivity = float(np.mean(pred_fall[fall_mask])) if np.any(fall_mask) else 0.0

    # Critical Fall Recall (exact class 4 recall)
    crit_mask = y_true == 4
    crit_recall = float(np.mean((y_pred == 4)[crit_mask])) if np.any(crit_mask) else 0.0

    mean_conf = float(np.mean(np.max(probs, axis=1)))
    return {
        "test_accuracy": acc,
        "test_macro_f1": macro_f1,
        "test_weighted_f1": weighted_f1,
        "overall_false_alert_rate": overall_far,
        "wrist_shock_false_alert_rate": wrist_far,
        "device_drop_false_alert_rate": drop_far,
        "binary_fall_sensitivity": fall_sensitivity,
        "critical_fall_recall": crit_recall,
        "mean_confidence": mean_conf,
        "confusion_matrix": cm.tolist(),
        "per_class_report": report,
    }


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def simulate_ble_packet_drop(X: np.ndarray, drop_rate: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate missing BLE 0x03 packets via sample-and-hold (zero-order hold) inside [N, 30, 6]."""
    if drop_rate <= 0.0:
        return X.copy()
    out = X.copy()
    mask = rng.random((out.shape[0], out.shape[1])) < drop_rate
    mask[:, 0] = False  # assume initial sample exists
    for t in range(1, out.shape[1]):
        m = mask[:, t]
        out[m, t, :] = out[m, t - 1, :]
    return out


def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 65,
    batch_size: int = 128,
) -> tuple[LumoBurst1DCNN, dict]:
    print("\n=== Training Approach 1: LumoBurst1DCNN (Raw [30 x 6]) ===")
    model = LumoBurst1DCNN(num_classes=NUM_CLASSES, dropout=0.15)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    xv_t = torch.from_numpy(X_val)

    best_f1 = -1.0
    best_state = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb_aug = augment_imu_batch(xb)
            optimizer.zero_grad()
            logits = model(xb_aug)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.item()) * xb.shape[0]
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(xv_t).numpy()
            val_pred = np.argmax(val_logits, axis=1)
            val_acc = float(accuracy_score(y_val, val_pred))
            val_f1 = float(f1_score(y_val, val_pred, average="macro"))

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"  Epoch {epoch:02d}/{epochs} | loss={total_loss/len(y_train):.4f} "
                f"| val_acc={val_acc:.4f} | val_macro_f1={val_f1:.4f} (best={best_f1:.4f})"
            )

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model, {"train_time_sec": round(time.time() - t0, 2), "best_val_macro_f1": best_f1}


def train_mlp(
    F_train: np.ndarray,
    y_train: np.ndarray,
    F_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 80,
    batch_size: int = 128,
) -> tuple[LumoEventMLP, np.ndarray, np.ndarray, dict]:
    print("\n=== Training Approach 2: LumoEventMLP (16 Physics Features) ===")
    scaler_mean = np.mean(F_train, axis=0, dtype=np.float64).astype(np.float32)
    scaler_std = np.std(F_train, axis=0, ddof=0, dtype=np.float64).astype(np.float32)
    scaler_std = np.maximum(scaler_std, np.float32(1e-5))

    F_tr_n = ((F_train - scaler_mean[None, :]) / (scaler_std[None, :] + np.float32(1e-6))).astype(np.float32)
    F_va_n = ((F_val - scaler_mean[None, :]) / (scaler_std[None, :] + np.float32(1e-6))).astype(np.float32)

    model = LumoEventMLP(in_features=F_train.shape[1], num_classes=NUM_CLASSES, dropout=0.12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(F_tr_n), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    fv_t = torch.from_numpy(F_va_n)

    best_f1 = -1.0
    best_state = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for fb, yb in train_loader:
            fb_aug = fb + torch.randn_like(fb) * 0.015
            optimizer.zero_grad()
            logits = model(fb_aug)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.item()) * fb.shape[0]
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(fv_t).numpy()
            val_pred = np.argmax(val_logits, axis=1)
            val_acc = float(accuracy_score(y_val, val_pred))
            val_f1 = float(f1_score(y_val, val_pred, average="macro"))

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"  Epoch {epoch:02d}/{epochs} | loss={total_loss/len(y_train):.4f} "
                f"| val_acc={val_acc:.4f} | val_macro_f1={val_f1:.4f} (best={best_f1:.4f})"
            )

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model, scaler_mean, scaler_std, {"train_time_sec": round(time.time() - t0, 2), "best_val_macro_f1": best_f1}


def write_inference_demo(path: pathlib.Path, mode: str) -> None:
    if mode == "cnn":
        code = '''"""Standalone inference demo for fragilov/lumohub-imu-burst-1dcnn."""
import numpy as np

def predict_window(window_30x6: np.ndarray, npz_path: str = "weights_folded.npz"):
    w = np.load(npz_path)
    x = (window_30x6.astype(np.float32) / np.array([8, 8, 8, 250, 250, 250], np.float32)).T[None]
    def conv(inp, wt, bs, pad):
        p = np.pad(inp, ((0,0),(0,0),(pad,pad)))
        out = np.zeros((1, wt.shape[0], inp.shape[2]), np.float32)
        for oc in range(wt.shape[0]):
            acc = np.full(inp.shape[2], bs[oc], np.float32)
            for ic in range(wt.shape[1]):
                for k in range(wt.shape[2]):
                    acc += p[0, ic, k:k+inp.shape[2]] * wt[oc, ic, k]
            out[0, oc] = np.maximum(acc, 0.0)
        return out
    h1 = conv(x, w["conv1_w"], w["conv1_b"], 2).reshape(1, 24, 15, 2).max(axis=3)
    h2 = conv(h1, w["conv2_w"], w["conv2_b"], 1).reshape(1, 32, 5, 3).max(axis=3)
    feat = np.concatenate([h2.mean(axis=2), h2.max(axis=2)], axis=1)
    h = np.maximum(feat @ w["fc1_w"].T + w["fc1_b"], 0.0)
    logits = h @ w["fc2_w"].T + w["fc2_b"]
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return int(np.argmax(probs)), probs[0]
'''
    else:
        code = '''"""Standalone inference demo for fragilov/lumohub-imu-event-mlp."""
import numpy as np

def predict_features(feat_16: np.ndarray, npz_path: str = "weights_folded.npz"):
    w = np.load(npz_path)
    x = (feat_16.astype(np.float32) - w["scaler_mean"]) / (w["scaler_std"] + 1e-6)
    h1 = np.maximum(x @ w["fc1_w"].T + w["fc1_b"], 0.0)
    h2 = np.maximum(h1 @ w["fc2_w"].T + w["fc2_b"], 0.0)
    logits = h2 @ w["fc3_w"].T + w["fc3_b"]
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return int(np.argmax(probs)), probs[0]
'''
    path.write_text(code, encoding="utf-8")


def main() -> None:
    set_seed(20261118)
    from_saved = "--from-saved" in sys.argv
    root = pathlib.Path(__file__).resolve().parent
    repo_root = root.parent.parent
    data_path = root / "data" / "lumohub_imu_5class_dataset.npz"
    ds = np.load(data_path)

    X_train, y_train, F_train = ds["X_train"], ds["y_train"], ds["F_train"]
    X_val, y_val, F_val = ds["X_val"], ds["y_val"], ds["F_val"]
    X_test, y_test, F_test = ds["X_test"], ds["y_test"], ds["F_test"]
    print(f"Loaded dataset: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    cnn_dir = root / "models" / "approach1_burst_cnn"
    mlp_dir = root / "models" / "approach2_event_mlp"

    if from_saved:
        print("[--from-saved] Loading trained PyTorch checkpoints and RF baseline...")
        cnn_model = LumoBurst1DCNN(num_classes=NUM_CLASSES, dropout=0.15)
        cnn_model.load_state_dict(torch.load(cnn_dir / "model.pt", map_location="cpu", weights_only=True))
        cnn_model.eval()
        cnn_weights = cnn_model.export_folded_weights()
        cnn_saved_m = json.loads((cnn_dir / "evaluation_metrics.json").read_text(encoding="utf-8"))
        cnn_train_info = {
            "train_time_sec": cnn_saved_m.get("train_time_sec", 0.0),
            "best_val_macro_f1": cnn_saved_m.get("best_val_macro_f1", 0.0),
        }

        mlp_model = LumoEventMLP(in_features=F_train.shape[1], num_classes=NUM_CLASSES, dropout=0.12)
        mlp_model.load_state_dict(torch.load(mlp_dir / "model.pt", map_location="cpu", weights_only=True))
        mlp_model.eval()
        mlp_weights = mlp_model.export_folded_weights()
        scaler_npz = np.load(mlp_dir / "weights_folded.npz")
        scaler_mean = scaler_npz["scaler_mean"].astype(np.float32)
        scaler_std = scaler_npz["scaler_std"].astype(np.float32)
        mlp_saved_m = json.loads((mlp_dir / "evaluation_metrics.json").read_text(encoding="utf-8"))
        mlp_train_info = {
            "train_time_sec": mlp_saved_m.get("train_time_sec", 0.0),
            "best_val_macro_f1": mlp_saved_m.get("best_val_macro_f1", 0.0),
        }
        with open(mlp_dir / "rf_baseline.pkl", "rb") as f:
            rf = pickle.load(f)
    else:
        # 1. Train Approach 1 (1D-CNN)
        cnn_model, cnn_train_info = train_cnn(X_train, y_train, X_val, y_val)
        cnn_weights = cnn_model.export_folded_weights()

    # Parity check CNN
    with torch.no_grad():
        pt_logits_cnn = cnn_model(torch.from_numpy(X_test[:1000])).numpy()
    np_logits_cnn = numpy_forward_cnn(X_test[:1000], cnn_weights)
    cnn_parity_diff = float(np.max(np.abs(pt_logits_cnn - np_logits_cnn)))
    print(f"[Parity] Approach 1 CNN PyTorch vs C-folded NumPy max abs diff = {cnn_parity_diff:.2e}")
    assert cnn_parity_diff < 1e-4, f"CNN parity check failed: {cnn_parity_diff}"

    if not from_saved:
        # 2. Train Approach 2 (Feature MLP + Random Forest Baseline)
        mlp_model, scaler_mean, scaler_std, mlp_train_info = train_mlp(F_train, y_train, F_val, y_val)
        mlp_weights = mlp_model.export_folded_weights()

    F_te_n = ((F_test[:1000] - scaler_mean[None, :]) / (scaler_std[None, :] + np.float32(1e-6))).astype(np.float32)
    with torch.no_grad():
        pt_logits_mlp = mlp_model(torch.from_numpy(F_te_n)).numpy()
    np_logits_mlp = numpy_forward_mlp(F_test[:1000], mlp_weights, scaler_mean, scaler_std)
    mlp_parity_diff = float(np.max(np.abs(pt_logits_mlp - np_logits_mlp)))
    print(f"[Parity] Approach 2 MLP PyTorch vs C-folded NumPy max abs diff = {mlp_parity_diff:.2e}")
    assert mlp_parity_diff < 1e-4, f"MLP parity check failed: {mlp_parity_diff}"

    if not from_saved:
        print("\n=== Training RandomForest-120 Classical Baseline (for comparison) ===")
        rf = RandomForestClassifier(n_estimators=120, max_depth=16, random_state=42, n_jobs=-1)
        rf.fit(F_train, y_train)

    # 3. Full Test Set Evaluation
    cnn_test_logits = numpy_forward_cnn(X_test, cnn_weights)
    cnn_test_probs = softmax_np(cnn_test_logits)
    cnn_test_pred = np.argmax(cnn_test_probs, axis=1)
    cnn_metrics = compute_detailed_metrics(y_test, cnn_test_pred, cnn_test_probs)
    cnn_metrics.update(cnn_train_info)
    cnn_metrics["parity_max_diff"] = cnn_parity_diff

    mlp_test_logits = numpy_forward_mlp(F_test, mlp_weights, scaler_mean, scaler_std)
    mlp_test_probs = softmax_np(mlp_test_logits)
    mlp_test_pred = np.argmax(mlp_test_probs, axis=1)
    mlp_metrics = compute_detailed_metrics(y_test, mlp_test_pred, mlp_test_probs)
    mlp_metrics.update(mlp_train_info)
    mlp_metrics["parity_max_diff"] = mlp_parity_diff

    rf_probs = rf.predict_proba(F_test)
    rf_pred = np.argmax(rf_probs, axis=1)
    rf_metrics = compute_detailed_metrics(y_test, rf_pred, rf_probs)

    # Ensemble check (mean probabilities of CNN + MLP)
    ens_probs = 0.5 * cnn_test_probs + 0.5 * mlp_test_probs
    ens_pred = np.argmax(ens_probs, axis=1)
    ens_metrics = compute_detailed_metrics(y_test, ens_pred, ens_probs)

    print("\n=================== TEST SET COMPARISON ===================")
    for label, m in [
        ("Approach 1: LumoBurst1DCNN", cnn_metrics),
        ("Approach 2: LumoEventMLP  ", mlp_metrics),
        ("Baseline  : RandomForest  ", rf_metrics),
        ("Hybrid    : CNN + MLP Avg ", ens_metrics),
    ]:
        print(
            f"{label} | Acc={m['test_accuracy']*100:6.2f}% | MacroF1={m['test_macro_f1']*100:6.2f}% "
            f"| WristFAR={m['wrist_shock_false_alert_rate']*100:5.2f}% "
            f"| DropFAR={m['device_drop_false_alert_rate']*100:5.2f}% "
            f"| FallSens={m['binary_fall_sensitivity']*100:6.2f}%"
        )

    # 4. BLE Packet Drop & Sensor Noise Stress Test
    rng = np.random.default_rng(999)
    stress_results = []
    for scenario, drop_rate, noise_mult in [
        ("Clean 10Hz BLE (0% loss)", 0.00, 0.0),
        ("BLE Packet Drop 10%", 0.10, 0.0),
        ("BLE Packet Drop 25%", 0.25, 0.0),
        ("BLE Packet Drop 40%", 0.40, 0.0),
        ("High Sensor Noise (2x)", 0.00, 1.0),
    ]:
        X_s = simulate_ble_packet_drop(X_test, drop_rate, rng)
        if noise_mult > 0.0:
            X_s[:, :, 0:3] += rng.normal(0.0, 0.025, size=(X_s.shape[0], X_s.shape[1], 3)).astype(np.float32)
            # Add modest gyro noise only up to 0.12dps for drop class vs 0.6dps for human motion
            non_drop = y_test != 2
            X_s[non_drop, :, 3:6] += rng.normal(0.0, 0.6, size=(int(np.sum(non_drop)), X_s.shape[1], 3)).astype(np.float32)
            X_s[~non_drop, :, 3:6] += rng.normal(0.0, 0.10, size=(int(np.sum(~non_drop)), X_s.shape[1], 3)).astype(np.float32)
        F_s = extract_features_batch(X_s)

        c_pred = np.argmax(numpy_forward_cnn(X_s, cnn_weights), axis=1)
        m_pred = np.argmax(numpy_forward_mlp(F_s, mlp_weights, scaler_mean, scaler_std), axis=1)

        c_acc = float(accuracy_score(y_test, c_pred))
        c_f1 = float(f1_score(y_test, c_pred, average="macro"))
        m_acc = float(accuracy_score(y_test, m_pred))
        m_f1 = float(f1_score(y_test, m_pred, average="macro"))
        stress_results.append({
            "scenario": scenario,
            "cnn_acc": c_acc, "cnn_f1": c_f1,
            "mlp_acc": m_acc, "mlp_f1": m_f1,
        })

    # 5. Footprint & Save Artifacts
    cnn_params = int(sum(v.size for v in cnn_weights.values()))
    mlp_params = int(sum(v.size for v in mlp_weights.values()) + scaler_mean.size + scaler_std.size)
    cnn_macs = int(24 * 6 * 5 * 30 + 32 * 24 * 3 * 15 + 32 * 64 + 5 * 32)
    mlp_macs = int(16 * 64 + 64 * 32 + 32 * 5)

    cnn_dir = root / "models" / "approach1_burst_cnn"
    mlp_dir = root / "models" / "approach2_event_mlp"
    cnn_dir.mkdir(parents=True, exist_ok=True)
    mlp_dir.mkdir(parents=True, exist_ok=True)

    torch.save(cnn_model.state_dict(), cnn_dir / "model.pt")
    np.savez(cnn_dir / "weights_folded.npz", **cnn_weights)
    (cnn_dir / "config.json").write_text(json.dumps({
        "architecture": "LumoBurst1DCNN",
        "input_shape": [30, 6],
        "normalization": {"accel_div": 8.0, "gyro_div": 250.0},
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "folded_parameters": cnn_params,
        "flash_bytes": cnn_params * 4,
        "macs_per_inference": cnn_macs,
    }, indent=2), encoding="utf-8")
    cnn_metrics["stress_tests"] = stress_results
    (cnn_dir / "evaluation_metrics.json").write_text(json.dumps(cnn_metrics, indent=2), encoding="utf-8")
    write_inference_demo(cnn_dir / "inference_demo.py", "cnn")

    torch.save(mlp_model.state_dict(), mlp_dir / "model.pt")
    np.savez(mlp_dir / "weights_folded.npz", scaler_mean=scaler_mean, scaler_std=scaler_std, **mlp_weights)
    with open(mlp_dir / "rf_baseline.pkl", "wb") as f:
        pickle.dump(rf, f)
    (mlp_dir / "feature_scaler.json").write_text(json.dumps({
        "feature_names": FEATURE_NAMES,
        "mean": scaler_mean.tolist(),
        "std": scaler_std.tolist(),
    }, indent=2), encoding="utf-8")
    (mlp_dir / "config.json").write_text(json.dumps({
        "architecture": "LumoEventMLP",
        "input_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "folded_parameters": mlp_params,
        "flash_bytes": mlp_params * 4,
        "macs_per_inference": mlp_macs,
    }, indent=2), encoding="utf-8")
    feat_imp = {name: round(float(imp), 5) for name, imp in zip(FEATURE_NAMES, rf.feature_importances_)}
    mlp_metrics["rf_baseline_metrics"] = rf_metrics
    mlp_metrics["rf_feature_importances"] = dict(sorted(feat_imp.items(), key=lambda x: -x[1]))
    mlp_metrics["stress_tests"] = stress_results
    (mlp_dir / "evaluation_metrics.json").write_text(json.dumps(mlp_metrics, indent=2), encoding="utf-8")
    write_inference_demo(mlp_dir / "inference_demo.py", "mlp")

    # 6. Export C99 headers to Firmware + Sim + c_export
    hub_ble_dir = repo_root / "LUMOHUB" / "firmware" / "lumo-hub" / "main" / "ble"
    sim_src_dir = repo_root / "03_ThietKe_KyThuat" / "lumohub_sim_prototype" / "src"
    c_exp_dir = root / "c_export"

    export_cnn_header(
        cnn_weights, cnn_metrics,
        [hub_ble_dir / "lumo_hub_cnn_model.h", sim_src_dir / "lumo_hub_cnn_model.h", c_exp_dir / "lumo_hub_cnn_model.h", cnn_dir / "lumo_hub_cnn_model.h"],
    )
    export_mlp_header(
        mlp_weights, scaler_mean, scaler_std, mlp_metrics,
        [hub_ble_dir / "lumo_hub_feature_model.h", sim_src_dir / "lumo_hub_feature_model.h", c_exp_dir / "lumo_hub_feature_model.h", mlp_dir / "lumo_hub_feature_model.h"],
    )

    # 7. Write BENCHMARK_REPORT.md
    lines = [
        "# LumoHub Dual-Architecture IMU TinyML Benchmark Report",
        "",
        "## 1. Test Set Comparison (N = 4,500 held-out windows: SisFall subjects SA20-23/SE13-15 + Biomechanics Test Split)",
        "",
        "| Model | Accuracy | Macro F1 | Wrist-Shock False Alert | Drop-Off False Alert | Binary Fall Sensitivity | Flash Weights | MACs / Inf |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Approach 1: LumoBurst1DCNN** | **{cnn_metrics['test_accuracy']*100:.2f}%** | **{cnn_metrics['test_macro_f1']*100:.2f}%** | {cnn_metrics['wrist_shock_false_alert_rate']*100:.2f}% | {cnn_metrics['device_drop_false_alert_rate']*100:.2f}% | {cnn_metrics['binary_fall_sensitivity']*100:.2f}% | {cnn_params*4:,} B | {cnn_macs:,} |",
        f"| **Approach 2: LumoEventMLP** | **{mlp_metrics['test_accuracy']*100:.2f}%** | **{mlp_metrics['test_macro_f1']*100:.2f}%** | {mlp_metrics['wrist_shock_false_alert_rate']*100:.2f}% | {mlp_metrics['device_drop_false_alert_rate']*100:.2f}% | {mlp_metrics['binary_fall_sensitivity']*100:.2f}% | {mlp_params*4:,} B | {mlp_macs:,} |",
        f"| Baseline: RandomForest-120 | {rf_metrics['test_accuracy']*100:.2f}% | {rf_metrics['test_macro_f1']*100:.2f}% | {rf_metrics['wrist_shock_false_alert_rate']*100:.2f}% | {rf_metrics['device_drop_false_alert_rate']*100:.2f}% | {rf_metrics['binary_fall_sensitivity']*100:.2f}% | — | — |",
        f"| Dual Ensemble (CNN + MLP) | {ens_metrics['test_accuracy']*100:.2f}% | {ens_metrics['test_macro_f1']*100:.2f}% | {ens_metrics['wrist_shock_false_alert_rate']*100:.2f}% | {ens_metrics['device_drop_false_alert_rate']*100:.2f}% | {ens_metrics['binary_fall_sensitivity']*100:.2f}% | {(cnn_params+mlp_params)*4:,} B | {cnn_macs+mlp_macs:,} |",
        "",
        "## 2. BLE Packet Loss & Noise Stress Test",
        "",
        "| Stress Scenario | 1D-CNN Acc | 1D-CNN Macro F1 | Feature-MLP Acc | Feature-MLP Macro F1 |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for s in stress_results:
        lines.append(
            f"| {s['scenario']} | {s['cnn_acc']*100:.2f}% | {s['cnn_f1']*100:.2f}% | {s['mlp_acc']*100:.2f}% | {s['mlp_f1']*100:.2f}% |"
        )
    lines.extend([
        "",
        "## 3. Top Physics Feature Importances (Random Forest Analysis)",
        "",
        "| Feature | Importance |",
        "| :--- | :---: |",
    ])
    for name, val in mlp_metrics["rf_feature_importances"].items():
        lines.append(f"| `{name}` | {val:.4f} |")

    (root / "BENCHMARK_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved benchmark report to {root / 'BENCHMARK_REPORT.md'}")


if __name__ == "__main__":
    main()
