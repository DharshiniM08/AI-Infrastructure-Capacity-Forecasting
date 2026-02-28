from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    ASSETS_DIR: Path = PROJECT_ROOT / "assets"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"

    DATASET_CSV: Path = DATA_DIR / "dataset.csv"
    MODEL_PKL: Path = MODELS_DIR / "model.pkl"
    METRICS_JSON: Path = MODELS_DIR / "metrics.json"

    # Columns (tabular)
    REQUIRED_COLUMNS: tuple[str, ...] = (
        "vm_id",
        "timestamp",
        "cpu_usage",
        "memory_usage",
        "network_traffic",
        "power_consumption",
        "num_executed_instructions",
        "execution_time",
        "energy_efficiency",
        "task_type",
        "task_priority",
        "task_status",
        "anomaly_status",
    )

    # Derived / optional columns
    TICKET_TEXT_COL: str = "ticket_text"
    TICKET_CATEGORY_COL: str = "ticket_category"

    # HuggingFace Router / Inference auth
    # Primary (per user requirement): HF_TOKEN
    # Backward compatible fallback: HF_API_TOKEN
    HF_TOKEN_ENV: str = "HF_TOKEN"
    HF_API_TOKEN_ENV: str = "HF_API_TOKEN"
    HF_MODEL_ID_ENV: str = "HF_MODEL_ID"
    HF_MODEL_ID_DEFAULT: str = "mistralai/Mistral-7B-Instruct-v0.2"
    HF_TIMEOUT_SECONDS: int = 60

    # Training defaults
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2


def get_settings() -> Settings:
    s = Settings()
    s.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    s.DATA_DIR.mkdir(parents=True, exist_ok=True)
    s.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    s.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    return s


def hf_token() -> str | None:
    s = get_settings()
    return os.getenv(s.HF_TOKEN_ENV) or os.getenv(s.HF_API_TOKEN_ENV)


def hf_model_id() -> str:
    s = get_settings()
    return os.getenv(s.HF_MODEL_ID_ENV, s.HF_MODEL_ID_DEFAULT)

