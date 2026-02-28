from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from config import get_settings


logger = logging.getLogger(__name__)


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )


_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s\-\_\/\.\:]+")


def clean_text(text: Any) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text).lower().strip()
    s = _NON_ALNUM_RE.sub(" ", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")
    return df


def validate_and_normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    s = get_settings()
    df = df.copy()

    # Normalize column names (common CSV issues)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required = set(s.REQUIRED_COLUMNS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Dataset missing required columns: "
            + ", ".join(missing)
            + ". Upload a CSV with the Cloud_Anomaly_Dataset schema."
        )

    # Timestamp parsing + feature extraction
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    if df["timestamp"].isna().mean() > 0.5:
        logger.warning("More than 50%% timestamps failed parsing; continuing with NaTs.")

    df["ts_hour"] = df["timestamp"].dt.hour.fillna(0).astype(int)
    df["ts_dayofweek"] = df["timestamp"].dt.dayofweek.fillna(0).astype(int)
    df["ts_month"] = df["timestamp"].dt.month.fillna(1).astype(int)

    # Numeric coercion
    numeric_cols = [
        "cpu_usage",
        "memory_usage",
        "network_traffic",
        "power_consumption",
        "num_executed_instructions",
        "execution_time",
        "energy_efficiency",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["anomaly_status"] = pd.to_numeric(df["anomaly_status"], errors="coerce").fillna(0).astype(int)
    df["anomaly_status"] = df["anomaly_status"].clip(0, 1)

    # Lightweight feature engineering (ratios)
    df["cpu_mem_ratio"] = df["cpu_usage"] / (df["memory_usage"].replace(0, np.nan))
    df["net_per_exec_time"] = df["network_traffic"] / (df["execution_time"].replace(0, np.nan))
    df["power_per_instruction"] = df["power_consumption"] / (df["num_executed_instructions"].replace(0, np.nan))

    return df


def _heuristic_ticket_text(df: pd.DataFrame) -> pd.Series:
    # Create a pseudo ticket narrative from telemetry for training / demo use.
    def _row_to_text(row: pd.Series) -> str:
        parts = [
            f"vm {row.get('vm_id', '')}",
            f"task {row.get('task_type', '')}",
            f"priority {row.get('task_priority', '')}",
            f"status {row.get('task_status', '')}",
            f"cpu {row.get('cpu_usage', '')}",
            f"memory {row.get('memory_usage', '')}",
            f"network {row.get('network_traffic', '')}",
            f"power {row.get('power_consumption', '')}",
            "anomaly detected" if int(row.get("anomaly_status", 0) or 0) == 1 else "no anomaly",
        ]
        return clean_text(" | ".join(map(str, parts)))

    return df.apply(_row_to_text, axis=1)


def _heuristic_ticket_category(df: pd.DataFrame) -> pd.Series:
    # Multi-class ticket categories derived from telemetry.
    # This lets the project be end-to-end even if the source dataset has no ticket labels.
    cpu = df["cpu_usage"].fillna(0)
    mem = df["memory_usage"].fillna(0)
    net = df["network_traffic"].fillna(0)
    power = df["power_consumption"].fillna(0)
    anomaly = df["anomaly_status"].fillna(0).astype(int)

    net_hi = net > net.quantile(0.85)
    power_hi = power > power.quantile(0.85)

    cat = np.where(anomaly == 1, "Anomaly", "General")
    cat = np.where((anomaly == 1) & (cpu >= 0.85), "Performance-CPU", cat)
    cat = np.where((anomaly == 1) & (mem >= 0.85), "Performance-Memory", cat)
    cat = np.where((anomaly == 1) & net_hi, "Network", cat)
    cat = np.where((anomaly == 1) & power_hi, "Power", cat)
    cat = np.where((anomaly == 0) & (cpu >= 0.8), "Capacity-Planning", cat)
    cat = np.where((anomaly == 0) & net_hi, "Network", cat)
    return pd.Series(cat, index=df.index, dtype="string")


def ensure_ticket_columns(df: pd.DataFrame) -> pd.DataFrame:
    s = get_settings()
    df = df.copy()
    if s.TICKET_TEXT_COL not in df.columns:
        df[s.TICKET_TEXT_COL] = _heuristic_ticket_text(df)
    else:
        df[s.TICKET_TEXT_COL] = df[s.TICKET_TEXT_COL].map(clean_text)

    if s.TICKET_CATEGORY_COL not in df.columns:
        df[s.TICKET_CATEGORY_COL] = _heuristic_ticket_category(df)
    else:
        df[s.TICKET_CATEGORY_COL] = df[s.TICKET_CATEGORY_COL].astype("string").fillna("Unknown")

    return df


@dataclass(frozen=True)
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def make_splits(df: pd.DataFrame, target_col: str, test_size: float, random_state: int) -> SplitData:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    stratify = None
    try:
        vc = y.value_counts(dropna=False)
        if y.nunique() > 1 and int(vc.min()) >= 2:
            stratify = y
    except Exception:
        stratify = None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def build_preprocessor(
    numeric_cols: Iterable[str],
    categorical_cols: Iterable[str],
    text_col: str,
) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    text_pipe = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    ngram_range=(1, 2),
                    max_features=25000,
                    min_df=2,
                ),
            )
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, list(numeric_cols)),
            ("cat", cat_pipe, list(categorical_cols)),
            ("txt", text_pipe, text_col),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def evaluate_classifier(
    model: Any,
    X_test: pd.DataFrame,
    y_test: Any,
    *,
    label_names: list[str] | None = None,
    label_order: list[int] | None = None,
) -> dict[str, Any]:
    """
    Evaluates a classifier and returns metrics suitable for dashboards.

    - If you train with encoded labels (0..K-1), pass `label_names` and `label_order`
      so the confusion matrix axes are readable.
    """
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1w = float(f1_score(y_test, y_pred, average="weighted"))

    if label_names is not None and label_order is not None:
        labels_for_cm = label_order
        labels_out: list[Any] = label_names
        report = classification_report(
            y_test,
            y_pred,
            labels=label_order,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )
    else:
        labels_for_cm = sorted(pd.unique(pd.Series(y_test)).tolist())
        labels_out = labels_for_cm
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=labels_for_cm)

    return {
        "accuracy": acc,
        "f1_weighted": f1w,
        "labels": labels_out,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _get_feature_names_from_column_transformer(ct: ColumnTransformer) -> list[str]:
    names: list[str] = []
    for name, transformer, cols in ct.transformers_:
        if name == "remainder" and transformer == "drop":
            continue
        if name == "txt":
            tfidf = transformer.named_steps["tfidf"]
            names.extend([f"txt__{t}" for t in tfidf.get_feature_names_out()])
            continue
        if hasattr(transformer, "named_steps"):
            last = list(transformer.named_steps.values())[-1]
        else:
            last = transformer

        if name == "cat":
            ohe = transformer.named_steps["onehot"]
            col_names = list(cols)
            ohe_names = ohe.get_feature_names_out(col_names)
            names.extend([f"cat__{n}" for n in ohe_names])
        else:
            names.extend([f"{name}__{c}" for c in cols])
    return names


def extract_model_insights(fitted_pipeline: Pipeline, top_k: int = 25) -> dict[str, Any]:
    """
    Returns best-effort feature importance / coefficients for the fitted classifier.
    Works for:
    - RandomForestClassifier, XGBClassifier: feature_importances_
    - LogisticRegression: coef_
    """
    insights: dict[str, Any] = {"top_features": []}
    try:
        pre: ColumnTransformer = fitted_pipeline.named_steps["preprocess"]
        clf = fitted_pipeline.named_steps["model"]
        feat_names = _get_feature_names_from_column_transformer(pre)

        scores = None
        if hasattr(clf, "feature_importances_"):
            scores = np.asarray(clf.feature_importances_, dtype=float)
        elif hasattr(clf, "coef_"):
            coef = np.asarray(clf.coef_, dtype=float)
            scores = np.mean(np.abs(coef), axis=0) if coef.ndim == 2 else np.abs(coef)

        if scores is None or len(scores) != len(feat_names):
            return insights

        idx = np.argsort(scores)[::-1][:top_k]
        insights["top_features"] = [
            {"feature": feat_names[i], "importance": float(scores[i])} for i in idx
        ]
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed extracting model insights: %s", e)
    return insights


def compute_default_input_template(df: pd.DataFrame) -> dict[str, Any]:
    """
    Used by the UI: if a user only provides ticket text, we fill other fields with
    training-set medians/modes so the pipeline can still run.
    """
    s = get_settings()
    template: dict[str, Any] = {}

    # Numeric defaults
    numeric_defaults = {
        "cpu_usage": float(df["cpu_usage"].median()),
        "memory_usage": float(df["memory_usage"].median()),
        "network_traffic": float(df["network_traffic"].median()),
        "power_consumption": float(df["power_consumption"].median()),
        "num_executed_instructions": float(df["num_executed_instructions"].median()),
        "execution_time": float(df["execution_time"].median()),
        "energy_efficiency": float(df["energy_efficiency"].median()),
        "cpu_mem_ratio": float(df["cpu_mem_ratio"].median(skipna=True)),
        "net_per_exec_time": float(df["net_per_exec_time"].median(skipna=True)),
        "power_per_instruction": float(df["power_per_instruction"].median(skipna=True)),
        "ts_hour": int(df["ts_hour"].mode().iloc[0]) if not df["ts_hour"].mode().empty else 0,
        "ts_dayofweek": int(df["ts_dayofweek"].mode().iloc[0]) if not df["ts_dayofweek"].mode().empty else 0,
        "ts_month": int(df["ts_month"].mode().iloc[0]) if not df["ts_month"].mode().empty else 1,
        "anomaly_status": 0,
    }
    template.update(numeric_defaults)

    # Categorical defaults
    for c in ["vm_id", "task_type", "task_priority", "task_status"]:
        template[c] = str(df[c].mode().iloc[0]) if c in df.columns and not df[c].mode().empty else "unknown"

    template["timestamp"] = datetime.utcnow().isoformat()
    template[s.TICKET_TEXT_COL] = ""
    return template

