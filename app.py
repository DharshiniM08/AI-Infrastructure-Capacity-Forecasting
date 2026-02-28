from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import get_settings
from hf_api import generate_hf_llm_response
from predict import load_artifact, predict_ticket_category
from train import train_and_save
from utils import (
    load_json,
    safe_read_csv,
    setup_logging,
    validate_and_normalize_columns,
)


S = get_settings()
setup_logging(S.LOGS_DIR / "app.log")
logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="AI Infrastructure Capacity Forecasting",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_css(dark_mode: bool) -> None:
    css_path = S.ASSETS_DIR / "css.css"
    css = ""
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")

    if not dark_mode:
        css += """
        :root{
          --bg-0:#f6f7fb; --bg-1:#ffffff;
          --card: rgba(0,0,0,0.03);
          --card-2: rgba(0,0,0,0.04);
          --stroke: rgba(0,0,0,0.08);
          --text: rgba(0,0,0,0.88);
          --muted: rgba(0,0,0,0.60);
          --shadow: 0 10px 30px rgba(0,0,0,0.08);
        }
        section[data-testid="stSidebar"]{
          background: linear-gradient(180deg, rgba(0,0,0,0.02), rgba(0,0,0,0.01));
          border-right: 1px solid rgba(0,0,0,0.08);
        }
        """

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    df_raw = safe_read_csv(Path(path))
    df = validate_and_normalize_columns(df_raw)
    return df


@st.cache_data(show_spinner=False)
def load_metrics() -> dict[str, Any]:
    return load_json(S.METRICS_JSON)


@st.cache_resource(show_spinner=False)
def load_model_artifact() -> dict[str, Any]:
    return load_artifact(S.MODEL_PKL)


def section_title(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='section-sub'>{subtitle}</div>", unsafe_allow_html=True)


def kpi_card(title: str, value: str, sub: str | None = None) -> None:
    sub_html = f"<div class='kpi-sub'>{sub}</div>" if sub else ""
    st.markdown(
        f"""
        <div class="ai-card">
          <div class="kpi-title">{title}</div>
          <div class="kpi-value">{value}</div>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str) -> None:
    st.markdown(
        f"<div class='badge'><span class='badge-dot'></span><span>{text}</span></div>",
        unsafe_allow_html=True,
    )


def _dashboard_page(df: pd.DataFrame) -> None:
    section_title(
        "Dashboard",
        "End-to-end ML + LLM system for support ticket categorization and automated responses.",
    )

    metrics = load_metrics()
    best = (metrics or {}).get("best_model", {}) if isinstance(metrics, dict) else {}

    rows = int(df.shape[0])
    vms = int(df["vm_id"].nunique())
    anomaly_rate = float(df["anomaly_status"].mean() * 100.0)
    date_min = str(pd.to_datetime(df["timestamp"], utc=True).min())
    date_max = str(pd.to_datetime(df["timestamp"], utc=True).max())

    c1, c2, c3, c4, c5 = st.columns([1.1, 1.1, 1.1, 1.1, 1.2])
    with c1:
        kpi_card("Rows", f"{rows:,}", "Total records loaded")
    with c2:
        kpi_card("Unique VMs", f"{vms:,}", "Fleet coverage")
    with c3:
        kpi_card("Anomaly rate", f"{anomaly_rate:.1f}%", "From `anomaly_status`")
    with c4:
        kpi_card("Best model F1", f"{best.get('f1_weighted', 0):.3f}", best.get("name", "Train to populate"))
    with c5:
        kpi_card("Time range", "UTC", f"{date_min.split('+')[0]} → {date_max.split('+')[0]}")

    st.divider()
    badge("Live analytics • cached data • production UI")

    left, right = st.columns([1.35, 1.0])
    with left:
        section_title("Telemetry Overview", "CPU usage over time (sampled).")
        df_plot = df.sort_values("timestamp").copy()
        df_plot = df_plot.dropna(subset=["timestamp"])
        if len(df_plot) > 5000:
            df_plot = df_plot.sample(5000, random_state=42).sort_values("timestamp")
        fig = px.line(df_plot, x="timestamp", y="cpu_usage", color="anomaly_status", render_mode="webgl")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="anomaly")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        section_title("Dataset Stats", "Task types & anomalies.")
        tcounts = df["task_type"].astype(str).value_counts().head(8).reset_index()
        tcounts.columns = ["task_type", "count"]
        fig2 = px.bar(tcounts, x="task_type", y="count")
        fig2.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    c1, c2 = st.columns([1.0, 1.0])
    with c1:
        section_title("Anomaly Distribution")
        dist = df["anomaly_status"].value_counts().sort_index().reset_index()
        dist.columns = ["anomaly_status", "count"]
        fig3 = px.pie(dist, values="count", names="anomaly_status", hole=0.55)
        fig3.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)
    with c2:
        section_title("Resource Correlation", "CPU vs Memory (colored by anomaly).")
        d2 = df[["cpu_usage", "memory_usage", "anomaly_status"]].dropna()
        if len(d2) > 4000:
            d2 = d2.sample(4000, random_state=42)
        fig4 = px.scatter(d2, x="cpu_usage", y="memory_usage", color="anomaly_status", opacity=0.7)
        fig4.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig4, use_container_width=True)


def _ticket_classifier_page() -> None:
    section_title("Ticket Classifier", "Predict support ticket category with confidence scores.")

    try:
        _ = load_model_artifact()
        has_model = True
    except Exception:
        has_model = False

    if not has_model:
        st.warning("No trained model found. Go to **Admin Panel** and click **Retrain model**.")
        return

    with st.container():
        st.markdown("<div class='ai-card'>", unsafe_allow_html=True)
        ticket = st.text_area(
            "Customer ticket",
            placeholder="Example: Our VM is throttling during peak hours; latency spiked and CPU is pegged.",
            height=140,
        )
        with st.expander("Optional telemetry overrides (for higher accuracy)"):
            c1, c2, c3 = st.columns(3)
            cpu = c1.number_input("cpu_usage", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
            mem = c2.number_input("memory_usage", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
            net = c3.number_input("network_traffic", min_value=0.0, value=200.0, step=10.0)
            c4, c5, c6 = st.columns(3)
            power = c4.number_input("power_consumption", min_value=0.0, value=250.0, step=10.0)
            exec_time = c5.number_input("execution_time", min_value=0.0, value=3.5, step=0.1)
            anomaly = c6.selectbox("anomaly_status", options=[0, 1], index=0)

        go_btn = st.button("Predict category", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if not go_btn:
        return

    if not ticket.strip():
        st.error("Please enter a ticket.")
        return

    overrides = {
        "cpu_usage": cpu,
        "memory_usage": mem,
        "network_traffic": net,
        "power_consumption": power,
        "execution_time": exec_time,
        "anomaly_status": anomaly,
    }

    with st.spinner("Classifying ticket..."):
        pred = predict_ticket_category(ticket_text=ticket, overrides=overrides)

    st.toast("Prediction ready", icon="✅")
    c1, c2 = st.columns([1.0, 1.0])
    with c1:
        st.markdown("<div class='ai-card'>", unsafe_allow_html=True)
        st.metric("Predicted category", pred["predicted_category"])
        conf = pred.get("confidence")
        st.metric("Confidence", f"{conf:.2f}" if conf is not None else "n/a")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        top = pred.get("top_predictions") or []
        if top:
            dfp = pd.DataFrame(top)
            fig = px.bar(dfp.sort_values("score"), x="score", y="label", orientation="h")
            fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)


def _llm_generator_page() -> None:
    section_title("AI Response Generator", "Generate contextual support replies using HuggingFace Inference API.")
    st.markdown("<div class='dm-hint'>Tip: set `HF_TOKEN` in your environment for LLM responses.</div>", unsafe_allow_html=True)

    ticket = st.text_area(
        "Customer ticket",
        placeholder="Example: After today's deployment, our batch jobs are failing intermittently with high power draw and reduced energy efficiency. Please advise.",
        height=150,
    )
    include_model_context = st.toggle("Include ML model context (category + confidence)", value=True)

    generate_btn = st.button("Generate AI response", use_container_width=True)
    if not generate_btn:
        return

    if not ticket.strip():
        st.error("Please enter a ticket.")
        return

    predicted_category = None
    confidence = None
    if include_model_context:
        try:
            pred = predict_ticket_category(ticket_text=ticket, overrides=None)
            predicted_category = pred.get("predicted_category")
            confidence = pred.get("confidence")
        except Exception as e:
            logger.warning("Prediction unavailable for LLM context: %s", e)

    placeholder = st.empty()
    with st.spinner("Calling HuggingFace Inference API..."):
        placeholder.markdown("<div class='ai-card shimmer' style='height: 110px;'></div>", unsafe_allow_html=True)
        try:
            response = generate_hf_llm_response(
                ticket_text=ticket,
                predicted_category=predicted_category,
                confidence=confidence,
                context=None,
            )
        except Exception as e:
            placeholder.empty()
            st.error(str(e))
            return

    placeholder.empty()
    st.toast("LLM response generated", icon="✨")
    st.markdown("<div class='ai-card'>", unsafe_allow_html=True)
    st.write(response)
    st.markdown("</div>", unsafe_allow_html=True)


def _analytics_page() -> None:
    section_title("Model Analytics", "Compare models, inspect confusion matrix, and feature importance.")
    metrics = load_metrics()
    if not metrics:
        st.warning("No metrics found. Train a model in **Admin Panel**.")
        return

    model_results = metrics.get("model_results", {})
    best = metrics.get("best_model", {}) or {}

    if model_results:
        rows = []
        for name, m in model_results.items():
            rows.append({"Model": name, "Accuracy": m.get("accuracy", 0), "F1 (weighted)": m.get("f1_weighted", 0)})
        dfm = pd.DataFrame(rows).sort_values("F1 (weighted)", ascending=False)
        fig = px.bar(dfm, x="Model", y=["Accuracy", "F1 (weighted)"], barmode="group")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    labels = None
    cm = None
    if best and best.get("name") in model_results:
        m = model_results[best["name"]]
        labels = m.get("labels") or m.get("label_encoder_classes")
        cm = m.get("confusion_matrix")

    c1, c2 = st.columns([1.25, 1.0])
    with c1:
        section_title("Confusion Matrix", f"Best model: {best.get('name', 'n/a')}")
        if labels and cm:
            z = np.asarray(cm, dtype=float)
            fig2 = go.Figure(
                data=go.Heatmap(
                    z=z,
                    x=labels,
                    y=labels,
                    colorscale="Blues",
                    hovertemplate="true=%{y}<br>pred=%{x}<br>count=%{z}<extra></extra>",
                )
            )
            fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Confusion matrix unavailable. Retrain to populate.")

    with c2:
        section_title("Feature Importance", "Top drivers (best effort).")
        top = ((best.get("insights") or {}).get("top_features")) if isinstance(best, dict) else []
        if top:
            dfi = pd.DataFrame(top).head(18).sort_values("importance")
            fig3 = px.bar(dfi, x="importance", y="feature", orientation="h")
            fig3.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Feature importance not available for this model/pipeline.")


def _admin_page() -> None:
    section_title("Admin Panel", "Upload new CSV, validate schema, and retrain the best model.")

    st.markdown("<div class='ai-card'>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload new dataset CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            df_norm = validate_and_normalize_columns(df_up)
            st.success("CSV validated successfully.")
            st.dataframe(df_norm.head(25), use_container_width=True)
            if st.button("Save as active dataset", use_container_width=True):
                S.DATASET_CSV.write_text(uploaded.getvalue().decode("utf-8", errors="ignore"), encoding="utf-8")
                st.toast("Dataset saved to data/dataset.csv", icon="✅")
                st.cache_data.clear()
        except Exception as e:
            st.error(f"CSV validation failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        st.markdown("<div class='ai-card'>", unsafe_allow_html=True)
        st.write("Training will compare **Logistic Regression**, **Random Forest**, and **XGBoost** and save the best model to `models/model.pkl`.")
        retrain = st.button("Retrain model", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        m = load_metrics()
        st.markdown("<div class='ai-card'>", unsafe_allow_html=True)
        st.write("Current training metadata")
        if m:
            st.write(
                {
                    "trained_at_utc": m.get("trained_at_utc"),
                    "dataset_rows": m.get("dataset_rows"),
                    "dataset_fingerprint": m.get("dataset_fingerprint"),
                    "best_model": (m.get("best_model") or {}).get("name"),
                }
            )
        else:
            st.info("No metrics yet. Retrain to initialize.")
        st.markdown("</div>", unsafe_allow_html=True)

    if retrain:
        with st.spinner("Training models..."):
            try:
                res = train_and_save()
            except Exception as e:
                st.error(str(e))
                return
        st.toast(f"Training complete. Best: {res.get('best_model', {}).get('name')}", icon="🎉")
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()


def main() -> None:
    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = True

    with st.sidebar:
        st.markdown("<div class='section-title'>AI Infra Forecasting</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-sub'>ML ticket routing + LLM auto-responses</div>", unsafe_allow_html=True)
        st.session_state["dark_mode"] = st.toggle("Dark mode", value=st.session_state["dark_mode"])
        st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
        page = st.radio(
            "Navigation",
            options=["Dashboard", "Ticket Classifier", "AI Response Generator", "Model Analytics", "Admin Panel"],
            index=0,
        )
        st.markdown("---")
        st.caption("Artifacts: `data/dataset.csv`, `models/model.pkl`, `models/metrics.json`")

    _inject_css(dark_mode=bool(st.session_state["dark_mode"]))

    # Load dataset for pages that need it
    df = None
    if page in {"Dashboard", "Admin Panel"}:
        try:
            df = load_dataset(str(S.DATASET_CSV))
        except Exception as e:
            st.error(str(e))
            if page == "Dashboard":
                st.info("Upload a CSV in **Admin Panel** or use the provided sample dataset.")

    if page == "Dashboard":
        if df is not None:
            _dashboard_page(df)
    elif page == "Ticket Classifier":
        _ticket_classifier_page()
    elif page == "AI Response Generator":
        _llm_generator_page()
    elif page == "Model Analytics":
        _analytics_page()
    elif page == "Admin Panel":
        _admin_page()


if __name__ == "__main__":
    main()

