## AI Infrastructure Capacity Forecasting

Production-style ML + LLM app that:

- Trains and compares **Logistic Regression**, **Random Forest**, (**XGBoost if installed**) for **ticket category classification**
- Persists the best model to `models/model.pkl`
- Generates automated responses via **HuggingFace Inference API** using `requests.post()` (no local transformers)
- Serves a modern multi-page **Streamlit** UI with dashboards + admin retraining

### Folder structure

```
AI-Infrastructure-Capacity-Forecasting/
  app.py
  hf_api.py
  train.py
  predict.py
  utils.py
  config.py
  models/
    model.pkl
    metrics.json
  data/
    dataset.csv
  assets/
    css.css
  requirements.txt
```

### Setup

```bash
pip install -r requirements.txt
```

Optional (for the LLM page):

- Set `HF_TOKEN` (HuggingFace access token)
- Optionally set `HF_MODEL_ID` (defaults to `mistralai/Mistral-7B-Instruct-v0.2`)

### Train

```bash
python train.py --data data/dataset.csv
```

### Run the UI

```bash
streamlit run app.py
```

