
---

## Energy Consumption Prediction in Electric Arc Furnace (EAF)

### Project Overview

This project focuses on analyzing and predicting energy consumption patterns in an Electric Arc Furnace (EAF) using real-world industrial process data.
The goal is to combine power (Transformer), gas, and temperature (Temp) data to understand how factors such as power (MW), gas usage, and oxygen levels affect temperature and overall energy efficiency.

---

### Dataset Source

The dataset used in this project is publicly available on Kaggle:
https://www.kaggle.com/datasets/yuriykatser/industrial-data-from-the-arc-furnace

The dataset includes logs of transformer, gas lance, and temperature readings used for energy analysis.

---

### Objectives

* Clean and preprocess industrial datasets (raw CSV files)
* Merge Transformer, Gas, and Temperature datasets by HEATID
* Analyze correlations between variables such as MW, O2_AMOUNT, GAS_AMOUNT, and TEMP
* Develop a basic AI/ML model to predict temperature or energy consumption based on process parameters
* Provide useful visualizations for better process understanding

---

### Why This Project Matters

Energy optimization in EAF operations can save significant cost and reduce environmental impact.
By predicting temperature and energy usage trends, this system can:

* Recommend optimal energy input settings
* Help minimize energy wastage
* Provide insights for better furnace control and planning

---

### Technologies Used

* Python 3
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn (for machine learning prediction)
* Jupyter Notebook

---

### Folder Structure

```
Energy_prediction/
│
├── data/
│   ├── raw/              # Original unprocessed CSV files
│   │   ├── eaf_transformer.csv
│   │   ├── eaf_temp.csv
│   │   └── eaf_gaslance_mat.csv
│   │
│   └── cleaned/          # Automatically generated cleaned dataset
│       └── merged_cleaned.csv
│
├── energy_prediction.ipynb   # Main notebook (data cleaning + ML + analysis)
└── README.md
```

### How to Run Locally

1. Install dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

2. Open the notebook:

   ```bash
   jupyter notebook energy_prediction.ipynb
   ```

3. Run all cells in order to:

   * Load and clean data
   * Merge datasets
   * Generate cleaned output in `data/cleaned/merged_cleaned.csv`
   * Run visual analysis and prediction model

### Outputs

* Correlation plots between MW, O2_AMOUNT, GAS_AMOUNT, and TEMP
* Energy–Temperature prediction model
* Cleaned dataset ready for advanced analytics or AI optimization

### Future Enhancements

* Add deep learning (LSTM) for time-based prediction
* Develop an interactive dashboard (e.g., Streamlit or Dash)
* Integrate recommendation logic for optimal energy usage

