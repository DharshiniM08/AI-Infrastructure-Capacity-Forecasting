## AI Infrastructure Capacity Forecasting

> Production-grade ML + LLM application for cloud infra ticket classification and automated responses.

[![Streamlit](https://img.shields.io/badge/Streamlit-UI-ff4b4b)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Inference-yellow)](https://huggingface.co/docs/api-inference)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange)](https://scikit-learn.org/stable/)

---

### Logo

You can place a logo image at `assets/logo.png` and reference it from the UI or README:

```markdown
![AI Infra Forecasting Logo](assets/logo.png)
```

If no image is present, the app will still run; the logo is purely cosmetic.

---

## 1. Overview

- **Objective**:
  - Classify cloud infrastructure support tickets into operational categories.
  - Detect / explain anomalies based on telemetry (CPU, memory, network, power, etc.).
  - Generate contextual, human-like responses using a HuggingFace-hosted LLM.
- **Stack**:
  - **Python**, **scikit-learn**, **XGBoost** (optional), **pandas**, **numpy**
  - **Streamlit** for UI
  - **HuggingFace Router** (`router.huggingface.co`) for LLM responses via `requests.post`.

---

## 2. Folder Structure

```text
cursor_prjct/
  app.py            # Streamlit multi-page UI
  hf_api.py         # HuggingFace Router / Inference client
  train.py          # Model training & comparison
  predict.py        # Reusable prediction API
  utils.py          # Preprocessing, feature engineering, metrics, logging utilities
  config.py         # Central settings: paths, env vars, column names
  DOCS.md           # This documentation
  README.md         # Quickstart overview

  data/
    dataset.csv     # Cloud_Anomaly_Dataset-compatible CSV

  models/
    model.pkl       # Best trained model (Pipeline + LabelEncoder + defaults)
    metrics.json    # Training metrics + model comparison & insights

  assets/
    css.css         # Custom dark-mode stylesheet
    logo.png        # (optional) project logo

  logs/
    *.log           # App + training logs

  requirements.txt  # Python dependencies
```

---

## 3. Data Schema

### 3.1 Required Columns

The project expects a CSV with the following columns (case-insensitive; spaces are normalised):

- **Telemetry / infra**
  - `vm_id`
  - `timestamp`
  - `cpu_usage`
  - `memory_usage`
  - `network_traffic`
  - `power_consumption`
  - `num_executed_instructions`
  - `execution_time`
  - `energy_efficiency`
- **Task metadata**
  - `task_type`
  - `task_priority`
  - `task_status`
- **Target**
  - `anomaly_status` (0/1)

These are validated in `utils.validate_and_normalize_columns`.

### 3.2 Derived Columns

Created automatically during preprocessing:

- Time features:
  - `ts_hour`, `ts_dayofweek`, `ts_month`
- Ratio features:
  - `cpu_mem_ratio`
  - `net_per_exec_time`
  - `power_per_instruction`

### 3.3 Ticket Columns

If your dataset does **not** contain ticket text / categories, they are **heuristically derived**:

- `ticket_text`:
  - A concatenation of VM id, task metadata, resource utilisation, and anomaly flag.
- `ticket_category`:
  - Values like `"Anomaly"`, `"Performance-CPU"`, `"Performance-Memory"`, `"Network"`, `"Power"`, `"Capacity-Planning"`, `"General"`.

If your dataset **does** include:

- `ticket_text`
- `ticket_category`

then those are used directly for supervised learning.

---

## 4. ML Pipeline Details

Implementation lives primarily in:

- `utils.py`
- `train.py`

### 4.1 Preprocessing & Feature Engineering

- Column name normalisation (`lower()`, spaces → underscores).
- Timestamp parsing (`utc=True`) and extraction of:
  - Hour of day
  - Day of week
  - Month
- Numeric coercion for:
  - `cpu_usage`, `memory_usage`, `network_traffic`, `power_consumption`,
    `num_executed_instructions`, `execution_time`, `energy_efficiency`
- Additional ratios:
  - `cpu_mem_ratio = cpu_usage / memory_usage`
  - `net_per_exec_time = network_traffic / execution_time`
  - `power_per_instruction = power_consumption / num_executed_instructions`

### 4.2 Preprocessor

Defined in `utils.build_preprocessor` as a `ColumnTransformer`:

- **Numeric pipeline**:
  - `SimpleImputer(strategy="median")`
  - `StandardScaler(with_mean=False)`
- **Categorical pipeline**:
  - `SimpleImputer(strategy="most_frequent")`
  - `OneHotEncoder(handle_unknown="ignore")`
- **Text pipeline**:
  - `TfidfVectorizer` over `ticket_text` with:
    - `ngram_range=(1, 2)`
    - `max_features=25000`
    - `min_df=2`

### 4.3 Models

Defined in `train._build_models`:

- **Logistic Regression**
  - `max_iter=2000`
  - `class_weight="balanced"`
- **Random Forest**
  - `n_estimators=400`
  - `class_weight="balanced_subsample"`
- **XGBoost** (optional, only if `xgboost` is installed)
  - `n_estimators=500`, `max_depth=6`, `learning_rate=0.06`, etc.
  - If `xgboost` is missing, training logs a warning and continues with the other models.

### 4.4 Training & Selection

Executed by `train.train_and_save`:

1. Load & validate CSV.
2. Ensure `ticket_text` and `ticket_category` are present / derived.
3. Split into train/test (configurable `TEST_SIZE`, `RANDOM_STATE`).
4. Encode target labels using `LabelEncoder`.
5. For each model:
   - Build `Pipeline(preprocess → model)`.
   - Fit on training data.
   - Evaluate using:
     - Accuracy
     - Weighted F1-score
     - Confusion matrix
   - Track the best model using F1-weighted.
6. Persist artifacts to disk:
   - `models/model.pkl`: dict containing:
     - `pipeline` (preprocessor + classifier)
     - `label_encoder`
     - `default_template` (typical feature defaults for inference)
     - `schema` and `meta`
   - `models/metrics.json`: training metrics + model comparison + feature importance.

---

## 5. LLM Integration (HuggingFace Router)

### 5.1 Endpoint & Auth

- **File**: `hf_api.py`
- **Endpoint**:

```python
API_URL = "https://router.huggingface.co/featherless-ai/v1/completions"
```

- **Authentication**:

```python
headers = {
    "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
}
```

The token must be set **before** starting the app, e.g. in PowerShell:

```powershell
$env:HF_TOKEN = "<your_hf_token_here>"
```

You can also persist this:

```powershell
setx HF_TOKEN "<your_hf_token_here>"
```

### 5.2 Low-level Client

```python
def query(payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(API_URL, headers=headers, json=payload, timeout=s.HF_TIMEOUT_SECONDS)
    response.raise_for_status()  # wrapped as a user-friendly error in code
    return response.json()
```

### 5.3 Prompt Template & Wrapper

`generate_hf_llm_response`:

- Accepts:
  - `ticket_text`
  - Optional `predicted_category`, `confidence`
  - Optional telemetry `context`
- Builds a structured prompt that instructs the LLM to:
  - Acknowledge the user.
  - Provide diagnosis hypotheses.
  - Suggest next steps.
  - Ask clarifying questions.
- Sends the prompt through `query({...})` using an `inputs` field and returns the generated text.

---

## 6. Streamlit UI

### 6.1 Global Layout & Theme

- Single-page multi-view app driven by a sidebar radio.
- Dark-mode first UI with:
  - Gradient background
  - Card-based layout (`.ai-card` from `assets/css.css`)
  - Animated buttons
  - Minimal chrome (header/footer hidden)

### 6.2 Pages

1. **Dashboard**
   - KPI cards: row count, unique VMs, anomaly rate, best model metrics, time window.
   - Plotly charts:
     - CPU usage over time vs anomaly status.
     - Task type counts.
     - Anomaly distribution pie.
     - CPU vs memory scatter.

2. **Ticket Classifier**
   - Text area for ticket description.
   - Optional telemetry overrides (CPU, memory, traffic, power, execution time, anomaly flag).
   - On submit:
     - Calls `predict_ticket_category`.
     - Displays predicted category and confidence.
     - Shows top-k predictions as a horizontal bar chart.

3. **AI Response Generator**
   - Text area for ticket.
   - Toggle to include model-predicted category as context.
   - On submit:
     - Optionally predicts category.
     - Calls `generate_hf_llm_response`.
     - Displays generated response in a card with shimmer-loading animation.

4. **Model Analytics**
   - Model comparison chart (Accuracy & F1).
   - Confusion matrix heatmap for best model.
   - Feature importance chart (top drivers).

5. **Admin Panel**
   - CSV uploader with schema validation.
   - Option to save uploaded CSV as `data/dataset.csv`.
   - Button to retrain models and update `model.pkl` and `metrics.json`.

---

## 7. Setup & Usage

### 7.1 Install Dependencies

```bash
cd c:\Users\skyli\Downloads\pyspiders\cursor_prjct
pip install -r requirements.txt
```

### 7.2 Train the Model

CLI:

```bash
python train.py --data data/dataset.csv
```

Or via UI:

- Open Streamlit → **Admin Panel** → **Retrain model**.

### 7.3 Run the App

1. Set `HF_TOKEN` in the same terminal session:

```powershell
$env:HF_TOKEN = "<your_hf_token_here>"
```

2. Launch Streamlit:

```powershell
cd "c:\Users\skyli\Downloads\pyspiders\cursor_prjct"
streamlit run app.py
```

3. Navigate using the sidebar to the desired page.

---

## 8. Extending the Project

- **Add models**: update `_build_models` in `train.py`.
- **Custom prompts**: edit `_build_prompt` in `hf_api.py`.
- **Additional features**: extend feature engineering in `utils.validate_and_normalize_columns`.
- **New pages**: add view functions in `app.py` and wire them into the sidebar radio.

For general best practices on ML system design, see:

- [Machine Learning Engineering (Chip Huyen)](https://huyenchip.com/machine-learning-systems-design/)
- [MLOps: Continuous Delivery and Automation Pipelines in ML](https://martinfowler.com/articles/mlops.html)

