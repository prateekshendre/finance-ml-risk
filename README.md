# 🤖 Machine Learning for Financial Risk
End-to-end **credit default and counterparty risk modeling pipeline** with explainability and governance.  
This repo contains a reproducible workflow from ingestion through model explainability and a minimal deployment path.  

---
## 🚀 What This Project Does
* **Ingests** raw loan or counterparty data and produces cleaned feature sets  
* **Encodes & scales** variables using reproducible preprocessing pipelines  
* **Trains** logistic regression, random forest, and gradient boosting models  
* **Validates** performance with time-based splits and cross-validation  
* **Explains** model behavior with SHAP for global and local interpretability  
* **Generates** model cards and validation reports for governance  

---
## 📂 Data
* **Sample datasets** live in `data/examples/` for quick testing  
* Real data should be ingested via `src/ingestion.py` and stored securely in `data/feature_store/`  
* Clean CSV/parquet snapshots enable reproducible backtests and validations  

---
## 🛠 Methods & Tools
* **pandas, numpy** – data pipelines and transforms  
* **scikit-learn** – preprocessing, baseline models, validation  
* **xgboost / lightgbm** – boosted trees  
* **shap** – explainability  
* **mlflow** – experiment tracking  
* **pytest** – testing  
* **Docker** – packaging  

---
## 📁 Repository Layout

```

finance-ml-risk/  
├── data/  
│   ├── examples/                 # sample CSVs  
│   └── feature_store/            # engineered features  
├── configs/  
│   └── train.yaml                # training config  
├── notebooks/  
│   └── eda_feature_engineering.ipynb  
├── src/  
│   ├── ingestion.py              # raw to cleaned  
│   ├── features.py               # feature engineering  
│   ├── pipeline.py               # preprocessing pipeline  
│   ├── train.py                  # training entry  
│   ├── evaluate.py               # validation metrics  
│   ├── explainability.py         # SHAP reports  
│   └── predict.py                # inference API  
├── experiments/                  # MLflow logs  
├── outputs/  
│   ├── models/  
│   └── reports/  
├── reports/  
│   ├── model_card_template.md  
│   └── validation_report_template.md  
├── scripts/  
│   └── run_training.sh  
├── tests/  
│   ├── test_pipeline.py  
│   └── test_features.py  
├── .github/  
│   └── workflows/ci.yml  
├── requirements.txt  
├── Dockerfile  
├── README.md  
├── LICENSE  
└── .gitignore

```

---
## 🖼 Key Outputs
1. **Performance Metrics** – ROC AUC, precision-recall, calibration  
2. **SHAP Explainability Reports** – global & local explanations  
3. **Feature Stability** – PSI across time buckets  
4. **Model Card** – intended use, limitations, risks  
5. **Validation Report** – compliance-ready documentation  

---
## ⚡ How to Run Locally
1. Clone repo  
   ```
   git clone https://github.com/your-username/finance-ml-risk.git  
   cd finance-ml-risk
   ```

2. Create virtual environment  
   ```
   python -m venv venv  
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```
   
3. Install dependencies  
   ```
   pip install -r requirements.txt
   ```

4. Prepare data  
   ```
   python src/ingestion.py --input data/examples/raw.csv --out data/feature_store/clean.parquet
   ```

5. Train model  
   ```
   python src/train.py --config configs/train.yaml
   ```

6. Evaluate & explain  
   ```
   python src/evaluate.py --model outputs/models/model.pkl --data data/feature_store/clean.parquet  
   python src/explainability.py --model outputs/models/model.pkl --data data/feature_store/clean.parquet --out outputs/reports/shap_summary.html
   ```

7. Serve model (Docker)  
   ```
   docker build -t finance-ml-risk .  
   docker run --rm -p 8080:8080 finance-ml-risk
   ```

---
## 📌 Example Training Config
```
train.yaml  
seed: 42  
model:  
  type: xgboost  
  params:  
    n_estimators: 200  
    learning_rate: 0.05  
    max_depth: 6  
data:  
  train_start: 2018-01-01  
  train_end: 2021-12-31  
  val_start: 2022-01-01  
  val_end: 2022-12-31  
paths:  
  raw_input: data/examples/raw.csv  
  feature_store: data/feature_store/clean.parquet  
  outputs: outputs/  
features:  
  categorical: [sector, region]  
  numeric: [loan_amount, ltv, dsr]
``` 

---
## 🔮 Next Steps
* Add **drift monitoring** with PSI  
* Automate retraining with CI/CD gates  
* Build lightweight monitoring dashboard  
* Integrate with online feature store  

---
## 📜 License
MIT License
