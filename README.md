# Credit Default Risk Prediction

A machine learning project for predicting credit default risk using Lending Club loan data combined with macroeconomic indicators. This project includes exploratory data analysis (EDA), feature engineering, model development with LightGBM, and an interactive Streamlit dashboard for inference and visualization.

## Project Overview

This project aims to build an end-to-end credit risk prediction pipeline capable of identifying borrowers with a higher probability of default. The workflow integrates:

* Lending Club borrower-level loan data
* Quantitative macroeconomic indicators
* Qualitative macroeconomic and sentiment-related variables
* Machine learning classification models
* Interactive dashboard deployment using Streamlit

The final production model is based on **LightGBM (LGBM)** and is serialized for deployment and inference.

---

## Repository Structure

```bash
.
├── artifacts/
│   └── lgbm_model.pkl
├── app.py
├── eda.ipynb
├── macro_eda_v1.ipynb
├── main.ipynb
├── qualitative.ipynb
└── README.md
```

---

## File Descriptions

### `artifacts/`

Contains serialized machine learning artifacts used for deployment.

#### Contents

* **Serialized LightGBM model**

  * Trained production-ready credit default prediction model
  * Used by the Streamlit dashboard for real-time inference

---

### `app.py`

Streamlit dashboard application for interactive model deployment and prediction.

#### Features

* User input interface for loan/application data
* Real-time credit default prediction
* Probability score visualization
* Model inference using the serialized LightGBM model
* Interactive dashboard components for demonstration purposes

#### Run the dashboard

```bash
streamlit run app.py
```

---

### `eda.ipynb`

Exploratory Data Analysis (EDA) notebook for the Lending Club dataset.

#### Includes

* Dataset overview and cleaning
* Missing value analysis
* Feature distributions
* Correlation analysis
* Class imbalance inspection
* Loan default behavior analysis
* Visualizations for borrower characteristics and risk patterns

#### Purpose

Used to understand borrower-level loan data and identify important predictive features.

---

### `macro_eda_v1.ipynb`

EDA notebook for quantitative macroeconomic datasets.

#### Includes

* Time-series analysis of macroeconomic indicators
* Inflation, unemployment, interest rate, and GDP-related analysis
* Trend and seasonality visualization
* Correlation between macroeconomic conditions and default behavior

#### Purpose

Explores how macroeconomic variables may influence credit default risk.

---

### `main.ipynb`

Main end-to-end machine learning workflow notebook.

#### Includes

* Data preprocessing
* Feature engineering
* Dataset merging and transformation
* Train-validation-test split
* Model training and hyperparameter tuning
* LightGBM model development
* Model evaluation and performance metrics
* Visualization of results
* Feature importance analysis
* Final model serialization

#### Purpose

Core notebook containing the complete modeling pipeline and experimentation workflow.

---

### `qualitative.ipynb`

EDA notebook for qualitative macroeconomic data.

#### Includes

* Analysis of qualitative economic indicators
* Sentiment-oriented or categorical macroeconomic variables
* Feature transformation and encoding
* Visualization of qualitative economic trends

#### Purpose

Investigates non-quantitative macroeconomic signals relevant to credit risk prediction.

---

## Model

### Primary Model

* **LightGBM (LGBMClassifier)**

### Why LightGBM?

* High performance on tabular financial datasets
* Handles missing values efficiently
* Strong predictive capability on imbalanced classification problems
* Fast training and inference

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* LightGBM
* Matplotlib
* Seaborn
* Streamlit

---

## Workflow

1. Perform EDA on Lending Club loan data
2. Analyze macroeconomic datasets
3. Preprocess and engineer features
4. Train and evaluate machine learning models
5. Serialize best-performing model
6. Deploy interactive dashboard with Streamlit

---

## Installation

Clone the repository:

```bash
git clone https://github.com/clarencemarvin/comp4501-Credit-Default-Risk-Prediction.git
cd comp4501-Credit-Default-Risk-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

---

## Live Dashboard

Access the deployed Streamlit dashboard here:

https://comp4501-credit-default-risk-prediction-6azflezfolyjdfwyscgaud.streamlit.app/

## Future Improvements

* Add model explainability using SHAP
* Implement automated hyperparameter tuning
* Add API deployment support
* Improve macroeconomic feature engineering
* Add model monitoring and drift detection

---

## License

This project is intended for academic and educational purposes.
