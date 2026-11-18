"""Standalone inference demo for fragilov/lumohub-imu-event-mlp."""
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
