"""PyTorch architectures & BatchNorm folding utilities for LumoHub dual IMU models.

Approach 1: LumoBurst1DCNN  -- End-to-end 1D-CNN on raw [30, 6] burst IMU windows
Approach 2: LumoEventMLP    -- Physics-informed MLP on [16] summary features
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def fold_conv1d_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d) -> tuple[np.ndarray, np.ndarray]:
    """Fold Conv1d + BatchNorm1d parameters into pure Conv1d (weight, bias) float32 arrays."""
    with torch.no_grad():
        w = conv.weight.detach().cpu().numpy().astype(np.float64)
        b = (
            conv.bias.detach().cpu().numpy().astype(np.float64)
            if conv.bias is not None
            else np.zeros(conv.out_channels, dtype=np.float64)
        )
        gamma = bn.weight.detach().cpu().numpy().astype(np.float64)
        beta = bn.bias.detach().cpu().numpy().astype(np.float64)
        mean = bn.running_mean.detach().cpu().numpy().astype(np.float64)
        var = bn.running_var.detach().cpu().numpy().astype(np.float64)
        eps = float(bn.eps)

        scale = gamma / np.sqrt(var + eps)
        w_folded = (w * scale[:, None, None]).astype(np.float32)
        b_folded = ((b - mean) * scale + beta).astype(np.float32)
        return w_folded, b_folded


def fold_linear_bn(linear: nn.Linear, bn: nn.BatchNorm1d | None) -> tuple[np.ndarray, np.ndarray]:
    """Fold Linear (+ optional BatchNorm1d) into pure Linear (weight, bias) float32 arrays."""
    with torch.no_grad():
        w = linear.weight.detach().cpu().numpy().astype(np.float64)
        b = (
            linear.bias.detach().cpu().numpy().astype(np.float64)
            if linear.bias is not None
            else np.zeros(linear.out_features, dtype=np.float64)
        )
        if bn is None:
            return w.astype(np.float32), b.astype(np.float32)

        gamma = bn.weight.detach().cpu().numpy().astype(np.float64)
        beta = bn.bias.detach().cpu().numpy().astype(np.float64)
        mean = bn.running_mean.detach().cpu().numpy().astype(np.float64)
        var = bn.running_var.detach().cpu().numpy().astype(np.float64)
        eps = float(bn.eps)

        scale = gamma / np.sqrt(var + eps)
        w_folded = (w * scale[:, None]).astype(np.float32)
        b_folded = ((b - mean) * scale + beta).astype(np.float32)
        return w_folded, b_folded


class LumoBurst1DCNN(nn.Module):
    """Approach 1: Compact 1D-CNN operating directly on [B, 30, 6] IMU windows."""

    def __init__(self, num_classes: int = 5, dropout: float = 0.15):
        super().__init__()
        self.conv1 = nn.Conv1d(6, 24, kernel_size=5, padding=2, bias=True)
        self.bn1 = nn.BatchNorm1d(24)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 30 -> 15

        self.conv2 = nn.Conv1d(24, 32, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)  # 15 -> 5

        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(64, 32, bias=True)
        self.fc2 = nn.Linear(32, num_classes, bias=True)

    @staticmethod
    def normalize_raw(x: torch.Tensor) -> torch.Tensor:
        # x: [B, 30, 6] -> normalize accel/8.0, gyro/250.0 -> [B, 6, 30]
        scale = torch.tensor([8.0, 8.0, 8.0, 250.0, 250.0, 250.0], dtype=x.dtype, device=x.device)
        x_norm = x / scale.view(1, 1, 6)
        return x_norm.transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.normalize_raw(x)
        h = self.pool1(self.relu(self.bn1(self.conv1(h))))
        h = self.pool2(self.relu(self.bn2(self.conv2(h))))
        gap = h.mean(dim=2)
        gmp = h.amax(dim=2)
        feat = torch.cat([gap, gmp], dim=1)
        feat = self.relu(self.fc1(self.drop(feat)))
        return self.fc2(feat)

    def export_folded_weights(self) -> dict[str, np.ndarray]:
        w1, b1 = fold_conv1d_bn(self.conv1, self.bn1)
        w2, b2 = fold_conv1d_bn(self.conv2, self.bn2)
        fw1, fb1 = fold_linear_bn(self.fc1, None)
        fw2, fb2 = fold_linear_bn(self.fc2, None)
        return {
            "conv1_w": w1, "conv1_b": b1,
            "conv2_w": w2, "conv2_b": b2,
            "fc1_w": fw1, "fc1_b": fb1,
            "fc2_w": fw2, "fc2_b": fb2,
        }


class LumoEventMLP(nn.Module):
    """Approach 2: Physics-informed MLP on [B, 16] standardized summary features."""

    def __init__(self, in_features: int = 16, num_classes: int = 5, dropout: float = 0.12):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 64, bias=True)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(p=dropout)

        self.fc2 = nn.Linear(64, 32, bias=True)
        self.bn2 = nn.BatchNorm1d(32)
        self.drop2 = nn.Dropout(p=dropout * 0.75)

        self.fc3 = nn.Linear(32, num_classes, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop1(self.relu(self.bn1(self.fc1(x))))
        h = self.drop2(self.relu(self.bn2(self.fc2(h))))
        return self.fc3(h)

    def export_folded_weights(self) -> dict[str, np.ndarray]:
        w1, b1 = fold_linear_bn(self.fc1, self.bn1)
        w2, b2 = fold_linear_bn(self.fc2, self.bn2)
        w3, b3 = fold_linear_bn(self.fc3, None)
        return {
            "fc1_w": w1, "fc1_b": b1,
            "fc2_w": w2, "fc2_b": b2,
            "fc3_w": w3, "fc3_b": b3,
        }


def numpy_forward_cnn(windows: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    """Pure float32 C-equivalent forward pass for LumoBurst1DCNN."""
    x = np.asarray(windows, dtype=np.float32)
    scale = np.array([8.0, 8.0, 8.0, 250.0, 250.0, 250.0], dtype=np.float32)
    x_norm = (x / scale[None, None, :]).transpose(0, 2, 1)  # [N, 6, 30]

    def conv1d_relu(inp: np.ndarray, w: np.ndarray, b: np.ndarray, pad: int) -> np.ndarray:
        n, c_in, t_in = inp.shape
        c_out, _, k = w.shape
        padded = np.pad(inp, ((0, 0), (0, 0), (pad, pad)), mode="constant")
        out = np.zeros((n, c_out, t_in), dtype=np.float32)
        for oc in range(c_out):
            acc = np.full((n, t_in), b[oc], dtype=np.float32)
            for ic in range(c_in):
                for ki in range(k):
                    acc += padded[:, ic, ki:ki + t_in] * w[oc, ic, ki]
            out[:, oc, :] = np.maximum(acc, np.float32(0.0))
        return out

    h1 = conv1d_relu(x_norm, weights["conv1_w"], weights["conv1_b"], pad=2)
    # MaxPool1d(2) -> [N, 24, 15]
    h1p = h1.reshape(h1.shape[0], h1.shape[1], 15, 2).max(axis=3)

    h2 = conv1d_relu(h1p, weights["conv2_w"], weights["conv2_b"], pad=1)
    # MaxPool1d(3) -> [N, 32, 5]
    h2p = h2.reshape(h2.shape[0], h2.shape[1], 5, 3).max(axis=3)

    gap = h2p.mean(axis=2)
    gmp = h2p.max(axis=2)
    feat = np.concatenate([gap, gmp], axis=1)  # [N, 64]

    fc1 = np.maximum(feat @ weights["fc1_w"].T + weights["fc1_b"][None, :], np.float32(0.0))
    logits = fc1 @ weights["fc2_w"].T + weights["fc2_b"][None, :]
    return logits.astype(np.float32)


def numpy_forward_mlp(
    features: np.ndarray,
    weights: dict[str, np.ndarray],
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
) -> np.ndarray:
    """Pure float32 C-equivalent forward pass for LumoEventMLP."""
    x = (np.asarray(features, dtype=np.float32) - scaler_mean[None, :]) / (scaler_std[None, :] + np.float32(1e-6))
    h1 = np.maximum(x @ weights["fc1_w"].T + weights["fc1_b"][None, :], np.float32(0.0))
    h2 = np.maximum(h1 @ weights["fc2_w"].T + weights["fc2_b"][None, :], np.float32(0.0))
    logits = h2 @ weights["fc3_w"].T + weights["fc3_b"][None, :]
    return logits.astype(np.float32)
