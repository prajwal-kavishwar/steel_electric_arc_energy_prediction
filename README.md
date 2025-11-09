
#  Energy Consumption Prediction in Electric Arc Furnace (EAF)

## 1. Overview
This project predicts the **energy consumption (in MWh)** of an Electric Arc Furnace (EAF) heat using industrial process data such as furnace duration, power, gas/oxygen usage, and temperature.  
The system processes data, trains a machine learning model, and provides a **Streamlit-based GUI** to allow instant prediction.

Unlike static ML deployments, this project **trains the model automatically at startup** (no `.pkl` dependency), making it reusable, portable, and fully open-source.

---

## 2. Dataset Source
Raw industrial furnace data was taken from Kaggle:

https://www.kaggle.com/datasets/yuriykatser/industrial-data-from-the-arc-furnace

Files included:

| File                   | Description             |
|------------------------|-------------------------|
| `eaf_transformer.csv`  | Power & duration log    |
| `eaf_temp.csv`         | Temperature & O₂ ppm log|
| `eaf_gaslance_mat.csv` | Gas & oxygen flow log   |

These datasets were merged using `HEATID` into:  
 `data/processed/merged_cleaned.csv`

---

## 3. Key Features of the System
- Full data cleaning & merging pipeline
- Auto feature engineering (`duration_hours`, `Energy_MWh`)
- Self-training ML model (Gradient Boosting Regressor)
- No stored model file required (`.pkl` not needed)
- GUI for manual prediction input
- Graph output for visual interpretation
- Optional CSV batch prediction (removable if unused)

---

## 4. Tech Stack

| Category       | Tools                                    |
|----------------|------------------------------------------|
| Language       | Python 3                                 |
| Data Handling  | Pandas, NumPy                            |
| ML Model       | Scikit-learn (GradientBoostingRegressor) |
| GUI App        | Streamlit                                |
| Visualization  | Matplotlib                               |
| Dev Environment| Jupyter Notebook                         |

---

## 5. Folder Structure
```

Energy_prediction/
│
├── app.py                       # Streamlit GUI app (auto-trains model)
│
├── data/
│   ├── raw/                     # Original CSVs
│   └── processed/
│       └── merged_cleaned.csv   # Final training dataset
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   └── 02_model_training_v2.ipynb
│
├── outputs/                     # (Optional) generated prediction files
│
├── README.md
└── .gitignore

````

---

## 6. How to Run

### Option A — Run the GUI App
```bash
pip install -r requirements.txt
streamlit run app.py
````

➡ The app will:

1. Load `merged_cleaned.csv`
2. Train model automatically
3. Allow manual prediction input
4. Show visual output (graph)

### Option B — Run notebooks manually

1. Run `01_data_cleaning.ipynb` → generates processed CSV
2. Run `02_model_training_v2.ipynb` → trains & evaluates model

---

## 7. ML Model Used

| Model                       | Notes                        |
| --------------------------- | ---------------------------- |
| Gradient Boosting Regressor | Final selected model         |
| Auto trains at app startup  | No `.pkl` storage needed     |
| Feature count               | 11 final engineered features |

Target variable:

```
Energy_MWh = MW_mean × duration_hours
```

---

## 8. Future Improvements

 Deploy as web API (FastAPI)
 Add real furnace sensor input
 Add time-series deep learning model (LSTM)
 Add alert system for excess energy usage
 Industrial HMI/SCADA integration

---

## 9. Author

**Prajwal Kavishwar**
B.Tech – Mathematics & Computing Engineering
Project: Energy Optimization using Machine Learning

---

