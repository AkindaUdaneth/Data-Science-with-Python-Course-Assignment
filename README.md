# 🛢️ Middle East Oil and Economic Analysis
**Data Science with Python - Group D Final Project**

## 📌 Project Overview
This project investigates the intersection of Middle Eastern macroeconomic health and global energy markets. The primary objective is to build a Machine Learning regression model capable of accurately predicting **Brent Crude Oil Prices (USD per barrel)** using local economic indicators (GDP, Inflation, FDI, etc.) from Middle Eastern countries spanning from 1990 to 2024.

By analyzing this data, we aim to determine if regional economic health can successfully forecast a highly volatile global energy commodity.

## 📊 Dataset
* **Source:** Middle East Economic Data (1990 - 2024)
* **Target Variable:** `Brent_Oil_Price_USD_per_barrel`
* **Key Input Features:**
  * GDP Growth (Annual %)
  * Inflation (Consumer prices %)
  * Exports (% of GDP)
  * Foreign Direct Investment (FDI net inflows % GDP)
  * Unemployment (Total %)
  * Life Expectancy (Years)

## ⚙️ Project Pipeline

### 1. Data Pre-Processing & Feature Engineering
* **Cleaning:** Dropped non-numeric string identifiers (`Country`, `Country_Code`) to ensure machine learning compatibility.
* **Imputation:** Applied column mean imputation to fill missing values (`NaN`s) without losing valuable historical rows.
* **Feature Engineering:** Created an `Oil_Price_Lag1` feature using `.shift(1)`. This allows the model to learn from previous year trends without causing data leakage.

### 2. Exploratory Data Analysis (EDA)
* Visualized the historical 34-year trend of Brent Oil to identify major volatility spikes (e.g., 2008 financial crisis, 2020 pandemic).
* Generated a Correlation Heatmap to establish the mathematical relationships between local economic indicators and global oil prices.

### 3. Machine Learning Models
We utilized an 80/20 Train-Test split and progressed through the following algorithms:
* **Baseline Model:** `LinearRegression` (Established baseline accuracy).
* **Advanced Model I:** `RandomForestRegressor` (Captured complex, non-linear economic interactions).
* **Advanced Model II:** `XGBoost Regressor` (Selected as the final optimized gradient boosting model).

### 4. Hyperparameter Tuning & Post-Processing
* **Pruning:** Utilized `RandomizedSearchCV` on the XGBoost model to restrict `max_depth` and tune learning rates, effectively filtering out noisy rules.
* **Symbolic Filtering:** Applied a mathematical filter (`np.clip`) to the final prediction array to ensure the model logically never predicted an impossible negative oil price.

## 🏆 Results & Evaluation

The models were evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the R² Score. The tuned XGBoost model significantly outperformed the baseline:

| Model | MAE | RMSE | R² Score |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | $7.78 | 11.83 | 0.7828 |
| **Random Forest (Default)** | $5.07 | 8.87 | 0.8781 |
| **Random Forest (Pruned)** | $4.91 | 8.64 | 0.8844 |
| **XGBoost (Tuned)** | **$4.83** | **6.72** | **0.9391** |

### 💡 Key Findings
By extracting feature importances from our XGBoost model, we identified the strongest economic drivers. The **Previous Year's Oil Price (Lag1)**, **Unemployment**, and **Exports (% of GDP)** held the most predictive weight over global Brent Crude prices.

## 🚀 How to Run the Project
1. Clone this repository:
   ```bash
   https://github.com/AkindaUdaneth/Data-Science-with-Python-Course-Assignment.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Data-Science-with-Python-Course-Assignment
   ```

3. Install the required dependencies:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```

4. Open the Jupyter Notebook:
   ```
   jupyter notebook groupD.ipynb
   ```

## 👥 Group D Members

- R.M.M. Kithnula
- M.S.A. Omindu Kumara
- M.A.M. Aafiq
- J.M. Visvani Ishanka
- K.H.G.A. Udaneth
