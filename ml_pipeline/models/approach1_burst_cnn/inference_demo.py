"""Standalone inference demo for fragilov/lumohub-imu-burst-1dcnn."""
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
