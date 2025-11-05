Here is your **updated README.md** in **Medium GitHub style**, fully aligned with the **final version of your project** (clean data pipeline + ML model + prediction output).
It is formatted and ready to copy-paste.

---

```markdown
# Energy Consumption Prediction in Electric Arc Furnace (EAF)

## Overview
This project predicts the **energy consumption (in MWh)** of an Electric Arc Furnace (EAF) using real industrial process data.  
It combines multiple furnace data sources (Transformer, Gas Lance, Temperature) into a unified dataset and trains machine learning models to estimate energy usage per heat (batch).

The final model helps in optimizing energy planning, cost reduction, and furnace operation efficiency.

---

## Dataset Source
The raw dataset used in this project is publicly available on Kaggle:

https://www.kaggle.com/datasets/yuriykatser/industrial-data-from-the-arc-furnace

It contains three CSV files:

| File | Description |
|-------|-------------|
| `eaf_transformer.csv` | Power, duration, heat start time |
| `eaf_temp.csv` | Temperature readings and O₂ ppm values |
| `eaf_gaslance_mat.csv` | Oxygen & gas flow and amount data |

---

## Project Objectives
- Clean and preprocess raw industrial data
- Merge multiple furnace logs using `HEATID`
- Engineer features such as duration, average temperature, oxygen/gas totals, etc.
- Create new target variable: **Energy consumption (MWh)**
- Train and evaluate multiple ML models (Linear Regression, RandomForest, GradientBoosting)
- Select and export the best model for future predictions

---

## Why This Project Matters
Electric Arc Furnaces consume large amounts of electricity during steelmaking.  
Predicting energy usage helps in:

- Reducing cost of operations
- Planning power allocation efficiently
- Identifying abnormal energy spikes
- Improving furnace control and automation

---

## Tech Stack
| Category           | Tools / Libraries|
|--------------------|------------------|
| Language           | Python 3         |
| Data Handling      | Pandas, NumPy    |
| ML Models          | Scikit-learn     |
| Visualization      | Matplotlib       |
| Model Export       | Joblib           |
| Dev Environment    | Jupyter Notebook |

---

## Folder Structure
```

Energy_prediction/
│
├── data/
│   ├── raw/                # Original CSVs
│   └── processed/          # Cleaned merged dataset
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   └── 02_model_training.ipynb
│
├── models/
│   └── energy_model_gb.pkl        # Saved GradientBoosting model
│
├── outputs/
│   └── predictions.csv            # Actual vs predicted energy values
│
├── README.md
└── .gitignore

```

---

## Model Results
| Model | MAE | RMSE | R² |
|--------|------|------|------|
| Linear Regression | High error | Not suitable |
| Random Forest | ~0.87 | 0.992 R² |
| **Gradient Boosting (Final Model)** | ~1.57 | **0.994 R²** |

📌 Final exported model: `models/energy_model_gb.pkl`

---

## How to Run

### 1. Open Notebook 1 (Data Cleaning)
```

notebooks/01_data_cleaning.ipynb

```
Generates: `data/processed/merged_cleaned.csv`

### 2. Open Notebook 2 (Model Training)
```

notebooks/02_model_training.ipynb

````
Trains models, evaluates, and saves final model + predictions.

### 3. Load model manually (example)
```python
import joblib
model = joblib.load("models/energy_model_gb.pkl")
y_pred = model.predict([[...feature_values...]])
````

---

## Outputs Generated

| File                             | Purpose                             |
| -------------------------------- | ----------------------------------- |
| `merged_cleaned.csv`             | Final dataset for ML                |
| `energy_model_gb.pkl`            | Saved trained model                 |
| `predictions.csv`                | Test set actual vs predicted values |
| Scatter/feature importance plots | Generated inside Notebook 2         |

---

## Future Enhancements

* Deploy as REST API (FastAPI / Flask)
* Add interactive dashboard (Streamlit / Dash)
* Train LSTM/GRU model for time-series predictions
* Real-time industrial integration (SCADA / PLC feed)

---

## Author

Name: *Prajwal Kavishwar*
Course: B.Tech Mathematics and computing engineering branch

---

