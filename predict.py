from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config import get_settings
from utils import clean_text


logger = logging.getLogger(__name__)


def load_artifact(model_pkl: Path | None = None) -> dict[str, Any]:
    s = get_settings()
    model_pkl = model_pkl or s.MODEL_PKL
    if not model_pkl.exists():
        raise FileNotFoundError(
            f"Model not found at {model_pkl}. Train first via `python train.py` or the Admin Panel."
        )
    artifact = joblib.load(model_pkl)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError("Invalid model artifact format.")
    return artifact


def _row_from_template(template: dict[str, Any], ticket_text: str, overrides: dict[str, Any] | None) -> dict[str, Any]:
    row = dict(template)
    row["ticket_text"] = clean_text(ticket_text)
    if overrides:
        for k, v in overrides.items():
            row[k] = v

    # Recompute engineered features when base inputs change.
    def _safe_div(a: Any, b: Any) -> float:
        try:
            a = float(a)
            b = float(b)
            if b == 0:
                return float("nan")
            return a / b
        except Exception:
            return float("nan")

    row["cpu_mem_ratio"] = _safe_div(row.get("cpu_usage"), row.get("memory_usage"))
    row["net_per_exec_time"] = _safe_div(row.get("network_traffic"), row.get("execution_time"))
    row["power_per_instruction"] = _safe_div(row.get("power_consumption"), row.get("num_executed_instructions"))

    try:
        ts = pd.to_datetime(row.get("timestamp"), errors="coerce", utc=True)
        if pd.notna(ts):
            row["ts_hour"] = int(ts.hour)
            row["ts_dayofweek"] = int(ts.dayofweek)
            row["ts_month"] = int(ts.month)
    except Exception:
        pass
    return row


def predict_ticket_category(
    ticket_text: str,
    overrides: dict[str, Any] | None = None,
    model_pkl: Path | None = None,
) -> dict[str, Any]:
    artifact = load_artifact(model_pkl=model_pkl)
    pipe = artifact["pipeline"]
    le = artifact["label_encoder"]
    template = artifact.get("default_template", {})

    row = _row_from_template(template, ticket_text=ticket_text, overrides=overrides)
    X = pd.DataFrame([row])

    proba = None
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)
        proba = np.asarray(proba)[0]
        pred_idx = int(np.argmax(proba))
    else:
        pred_idx = int(pipe.predict(X)[0])

    label = str(le.inverse_transform([pred_idx])[0])
    confidence = float(proba[pred_idx]) if proba is not None else None

    topk = []
    if proba is not None:
        order = np.argsort(proba)[::-1][:5]
        for i in order:
            topk.append({"label": str(le.inverse_transform([int(i)])[0]), "score": float(proba[int(i)])})

    return {
        "predicted_category": label,
        "confidence": confidence,
        "top_predictions": topk,
    }

