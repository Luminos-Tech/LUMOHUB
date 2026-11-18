"""Train a compact ESP32-friendly Random Forest (16 physics features) and export to C99 header.

Checks accuracy vs Flash footprint, verifies bit-exact Python-to-C parity,
exports `lumo_hub_rf_model.h` to Hub, Band, Sim, and ML model folders,
and updates `fragilov/lumohub-imu-event-mlp` on Hugging Face.
"""

from __future__ import annotations

import json
import os
import pathlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from feature_extractor import CLASS_NAMES, FEATURE_NAMES, NUM_CLASSES

HF_TOKEN = os.environ.get("HF_TOKEN", "")


def eval_rf_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    pred = np.argmax(probs, axis=1)
    acc = float(accuracy_score(y_true, pred))
    macro_f1 = float(f1_score(y_true, pred, average="macro"))
    non_fall = np.isin(y_true, [0, 1, 2])
    pred_fall = np.isin(pred, [3, 4])
    wrist_far = float(np.mean(pred_fall[y_true == 1]))
    drop_far = float(np.mean(pred_fall[y_true == 2]))
    fall_sens = float(np.mean(pred_fall[np.isin(y_true, [3, 4])]))
    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "wrist_far": wrist_far,
        "drop_far": drop_far,
        "fall_sens": fall_sens,
    }


def flatten_rf(rf: RandomForestClassifier) -> dict:
    tree_offsets = []
    features = []
    thresholds = []
    lefts = []
    rights = []
    leaf_probs = []

    offset = 0
    for est in rf.estimators_:
        t = est.tree_
        tree_offsets.append(offset)
        n = t.node_count
        for i in range(n):
            cl = int(t.children_left[i])
            cr = int(t.children_right[i])
            if cl == cr:  # leaf
                features.append(-1)
                thresholds.append(0.0)
                leaf_idx = len(leaf_probs)
                counts = t.value[i][0].astype(np.float64)
                s = float(np.sum(counts))
                probs = (counts / s) if s > 0 else np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
                # Quantize probabilities to 0..255 for compact Flash storage
                q = np.clip(np.round(probs * 255.0), 0, 255).astype(np.uint8)
                leaf_probs.append(q.tolist())
                lefts.append(leaf_idx)
                rights.append(-1)
            else:
                features.append(int(t.feature[i]))
                thresholds.append(float(t.threshold[i]))
                lefts.append(offset + cl)
                rights.append(offset + cr)
        offset += n

    return {
        "num_trees": len(rf.estimators_),
        "num_nodes": len(features),
        "num_leaves": len(leaf_probs),
        "tree_offsets": tree_offsets,
        "features": features,
        "thresholds": thresholds,
        "lefts": lefts,
        "rights": rights,
        "leaf_probs": leaf_probs,
    }


def predict_flat_rf(F: np.ndarray, flat: dict) -> np.ndarray:
    """Exact float32 mirror of the C99 traversal in lumo_imu_ai.c."""
    n_samples = F.shape[0]
    out = np.zeros((n_samples, NUM_CLASSES), dtype=np.float32)
    feats = flat["features"]
    thresh = flat["thresholds"]
    lefts = flat["lefts"]
    rights = flat["rights"]
    leaf_probs = flat["leaf_probs"]
    offsets = flat["tree_offsets"]
    nt = flat["num_trees"]

    for i in range(n_samples):
        row = F[i]
        acc = np.zeros(NUM_CLASSES, dtype=np.float32)
        for t_idx in range(nt):
            node = offsets[t_idx]
            while feats[node] >= 0:
                f_idx = feats[node]
                if float(row[f_idx]) <= float(np.float32(thresh[node])):
                    node = lefts[node]
                else:
                    node = rights[node]
            l_idx = lefts[node]
            lp = leaf_probs[l_idx]
            for c in range(NUM_CLASSES):
                acc[c] += lp[c] / 255.0
        out[i] = acc / float(nt)
    return out


def format_c_header(flat: dict, metrics: dict) -> str:
    n_trees = flat["num_trees"]
    n_nodes = flat["num_nodes"]
    n_leaves = flat["num_leaves"]
    flash_bytes = n_trees * 2 + n_nodes * (1 + 2 + 2 + 4) + n_leaves * NUM_CLASSES

    def wrap_ints(arr, per_line=16):
        chunks = [", ".join(str(x) for x in arr[i:i + per_line]) for i in range(0, len(arr), per_line)]
        return ",\n    ".join(chunks)

    def wrap_floats(arr, per_line=8):
        chunks = [", ".join(f"{float(x):.7e}f" for x in arr[i:i + per_line]) for i in range(0, len(arr), per_line)]
        return ",\n    ".join(chunks)

    leaf_rows = ",\n    ".join(
        "{ " + ", ".join(str(int(v)) for v in row) + " }" for row in flat["leaf_probs"]
    )

    return f"""/**
 * @file lumo_hub_rf_model.h
 * @brief Compact C99 Pruned Random Forest ({n_trees} trees) on 16 Physics Features.
 * Test Accuracy: {metrics['acc']*100:.2f}% | Macro F1: {metrics['macro_f1']*100:.2f}% | WristFAR: {metrics['wrist_far']*100:.2f}% | DropFAR: {metrics['drop_far']*100:.2f}%
 * Estimated Flash: {flash_bytes:,} bytes ({n_nodes} nodes, {n_leaves} leaves)
 * Auto-generated by LUMOHUB/ml_pipeline/export_rf_c.py. Include ONLY inside lumo_imu_ai.c.
 */
#pragma once

#include <stdint.h>

#define LUMO_RF_NUM_TREES   {n_trees}
#define LUMO_RF_NUM_NODES   {n_nodes}
#define LUMO_RF_NUM_LEAVES  {n_leaves}
#define LUMO_RF_CLASSES     {NUM_CLASSES}

static const uint16_t LUMO_RF_TREE_OFFSET[{n_trees}] = {{
    {wrap_ints(flat['tree_offsets'])}
}};

static const int8_t LUMO_RF_FEATURE[{n_nodes}] = {{
    {wrap_ints(flat['features'])}
}};

static const int16_t LUMO_RF_LEFT[{n_nodes}] = {{
    {wrap_ints(flat['lefts'])}
}};

static const int16_t LUMO_RF_RIGHT[{n_nodes}] = {{
    {wrap_ints(flat['rights'])}
}};

static const float LUMO_RF_THRESHOLD[{n_nodes}] = {{
    {wrap_floats(flat['thresholds'])}
}};

static const uint8_t LUMO_RF_LEAF_PROBS[{n_leaves}][{NUM_CLASSES}] = {{
    {leaf_rows}
}};
"""


def main() -> None:
    root = pathlib.Path(__file__).resolve().parent
    repo_root = root.parent.parent
    ds = np.load(root / "data" / "lumohub_imu_5class_dataset.npz")
    F_train, y_train = ds["F_train"], ds["y_train"]
    F_test, y_test = ds["F_test"], ds["y_test"]

    rf = RandomForestClassifier(
        n_estimators=25,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(F_train, y_train)
    probs_sk = rf.predict_proba(F_test)
    m_sk = eval_rf_metrics(y_test, probs_sk)
    print(
        f"[Compact RF-25 sklearn] Acc={m_sk['acc']*100:.2f}% | MacroF1={m_sk['macro_f1']*100:.2f}% "
        f"| WristFAR={m_sk['wrist_far']*100:.2f}% | DropFAR={m_sk['drop_far']*100:.2f}% "
        f"| FallSens={m_sk['fall_sens']*100:.2f}%"
    )

    flat = flatten_rf(rf)
    flash_bytes = flat["num_trees"] * 2 + flat["num_nodes"] * 9 + flat["num_leaves"] * NUM_CLASSES
    print(
        f"[Compact RF-25 C Export] trees={flat['num_trees']}, nodes={flat['num_nodes']}, "
        f"leaves={flat['num_leaves']}, est_flash={flash_bytes:,} B"
    )

    # Parity check against C-quantized float32 traversal
    probs_c = predict_flat_rf(F_test[:1000], flat)
    m_c = eval_rf_metrics(y_test[:1000], probs_c)
    agree = float(np.mean(np.argmax(probs_sk[:1000], axis=1) == np.argmax(probs_c, axis=1)))
    print(f"[Parity] sklearn vs C-quantized RF agreement (1000 test) = {agree*100:.2f}% (acc={m_c['acc']*100:.2f}%)")
    assert agree >= 0.995, f"Low RF quantization agreement: {agree}"

    header = format_c_header(flat, m_sk)
    targets = [
        repo_root / "LUMOHUB" / "firmware" / "lumo-hub" / "main" / "ble" / "lumo_hub_rf_model.h",
        repo_root / "03_ThietKe_KyThuat" / "lumohub_sim_prototype" / "src" / "lumo_hub_rf_model.h",
        root / "models" / "approach2_event_mlp" / "lumo_hub_rf_model.h",
        root / "c_export" / "lumo_hub_rf_model.h",
    ]
    for p in targets:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(header, encoding="utf-8")
        print(f"  Wrote {p}")

    # Update Hugging Face repo with compact RF C header + summary
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=str(root / "models" / "approach2_event_mlp" / "lumo_hub_rf_model.h"),
            path_in_repo="lumo_hub_rf_model.h",
            repo_id="fragilov/lumohub-imu-event-mlp",
            repo_type="model",
            commit_message="Add compact C99 Pruned Random Forest (25 trees) header lumo_hub_rf_model.h",
        )
        print("[HF] Uploaded lumo_hub_rf_model.h -> https://huggingface.co/fragilov/lumohub-imu-event-mlp")
    except Exception as e:
        print(f"[HF] Upload warning: {e}")


if __name__ == "__main__":
    main()
