from __future__ import annotations

import argparse
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier  # type: ignore

    _XGBOOST_AVAILABLE = True
except Exception:  # noqa: BLE001
    XGBClassifier = None  # type: ignore
    _XGBOOST_AVAILABLE = False

from config import get_settings
from utils import (
    compute_default_input_template,
    dump_json,
    ensure_ticket_columns,
    evaluate_classifier,
    extract_model_insights,
    safe_read_csv,
    setup_logging,
    validate_and_normalize_columns,
    build_preprocessor,
    make_splits,
)


logger = logging.getLogger(__name__)


def _dataset_fingerprint(df: pd.DataFrame) -> str:
    # Stable-enough fingerprint for admin UI display (not cryptographic).
    sample = df.head(200).to_csv(index=False).encode("utf-8", errors="ignore")
    return hashlib.sha256(sample).hexdigest()[:16]


def _build_models(random_state: int) -> dict[str, Any]:
    models: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            n_jobs=None,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
    }
    if _XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        logger.warning("xgboost is not installed; skipping XGBoost model.")

    return models


def train_and_save(
    dataset_csv: Path | None = None,
    model_pkl: Path | None = None,
    metrics_json: Path | None = None,
) -> dict[str, Any]:
    s = get_settings()
    setup_logging(s.LOGS_DIR / "train.log")

    dataset_csv = dataset_csv or s.DATASET_CSV
    model_pkl = model_pkl or s.MODEL_PKL
    metrics_json = metrics_json or s.METRICS_JSON

    df_raw = safe_read_csv(dataset_csv)
    df = validate_and_normalize_columns(df_raw)
    df = ensure_ticket_columns(df)

    target_col = s.TICKET_CATEGORY_COL

    numeric_cols = [
        "cpu_usage",
        "memory_usage",
        "network_traffic",
        "power_consumption",
        "num_executed_instructions",
        "execution_time",
        "energy_efficiency",
        "cpu_mem_ratio",
        "net_per_exec_time",
        "power_per_instruction",
        "ts_hour",
        "ts_dayofweek",
        "ts_month",
        "anomaly_status",
    ]
    categorical_cols = ["vm_id", "task_type", "task_priority", "task_status"]
    text_col = s.TICKET_TEXT_COL

    # Drop rows without target
    df = df.dropna(subset=[target_col]).copy()
    df[target_col] = df[target_col].astype("string").fillna("Unknown")

    label_encoder = LabelEncoder()
    label_encoder.fit(df[target_col].astype(str).to_numpy())

    splits = make_splits(df, target_col=target_col, test_size=s.TEST_SIZE, random_state=s.RANDOM_STATE)

    y_train_enc = label_encoder.transform(splits.y_train.astype(str).to_numpy())
    y_test_enc = label_encoder.transform(splits.y_test.astype(str).to_numpy())

    models = _build_models(random_state=s.RANDOM_STATE)

    results: dict[str, Any] = {
        "project": "AI Infrastructure Capacity Forecasting",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_csv),
        "dataset_rows": int(df.shape[0]),
        "dataset_fingerprint": _dataset_fingerprint(df),
        "target": target_col,
        "labels": label_encoder.classes_.tolist(),
        "model_results": {},
        "best_model": None,
    }

    best_name: str | None = None
    best_score = -np.inf
    best_pipeline: Pipeline | None = None
    best_eval: dict[str, Any] | None = None

    for name, clf in models.items():
        pre = build_preprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols, text_col=text_col)
        pipe = Pipeline(steps=[("preprocess", pre), ("model", clf)])
        logger.info("Training %s ...", name)
        pipe.fit(splits.X_train, y_train_enc)

        classes = label_encoder.classes_.tolist()
        eval_payload = evaluate_classifier(
            pipe,
            splits.X_test,
            y_test_enc,
            label_names=classes,
            label_order=list(range(len(classes))),
        )
        eval_payload["metric_used_for_selection"] = "f1_weighted"
        eval_payload["label_encoder_classes"] = label_encoder.classes_.tolist()

        results["model_results"][name] = eval_payload

        score = float(eval_payload["f1_weighted"])
        if score > best_score:
            best_score = score
            best_name = name
            best_pipeline = pipe
            best_eval = eval_payload

    if best_pipeline is None or best_name is None or best_eval is None:
        raise RuntimeError("Training failed: no model produced a valid result.")

    results["best_model"] = {
        "name": best_name,
        "accuracy": best_eval["accuracy"],
        "f1_weighted": best_eval["f1_weighted"],
    }

    insights = extract_model_insights(best_pipeline)
    results["best_model"]["insights"] = insights

    default_template = compute_default_input_template(df)

    artifact = {
        "pipeline": best_pipeline,
        "label_encoder": label_encoder,
        "default_template": default_template,
        "schema": {
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "text_col": text_col,
            "target_col": target_col,
        },
        "meta": {
            "trained_at_utc": results["trained_at_utc"],
            "dataset_fingerprint": results["dataset_fingerprint"],
            "dataset_rows": results["dataset_rows"],
        },
    }

    model_pkl.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_pkl)
    dump_json(metrics_json, results)

    logger.info("Saved best model to %s", model_pkl)
    logger.info("Saved metrics to %s", metrics_json)
    return results


def main() -> None:
    s = get_settings()
    parser = argparse.ArgumentParser(description="Train and save ML models.")
    parser.add_argument("--data", type=str, default=str(s.DATASET_CSV), help="Path to dataset CSV")
    parser.add_argument("--model-out", type=str, default=str(s.MODEL_PKL), help="Output .pkl path")
    parser.add_argument("--metrics-out", type=str, default=str(s.METRICS_JSON), help="Output metrics JSON path")
    args = parser.parse_args()

    train_and_save(
        dataset_csv=Path(args.data),
        model_pkl=Path(args.model_out),
        metrics_json=Path(args.metrics_out),
    )


if __name__ == "__main__":
    main()

