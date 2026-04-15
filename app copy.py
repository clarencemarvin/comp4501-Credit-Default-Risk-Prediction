import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Macro-Aware Credit Risk Dashboard",
    page_icon="📉",
    layout="wide"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "model.txt"
PREPROCESSOR_PATH = ARTIFACT_DIR / "preprocessor.pkl"
SCHEMA_PATH = ARTIFACT_DIR / "feature_schema.json"
TRAIN_REF_PATH = ARTIFACT_DIR / "training_reference.csv"

# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f7f7f2 0%, #eef7f2 100%);
    }

    html, body, [class*="css"] {
        color: black;
    }

    label, p, span, div, h1, h2, h3, h4, h5, h6 {
        color: black !important;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMarkdownContainer"],
    [data-testid="stText"],
    [data-testid="stCaptionContainer"] {
        color: black !important;
    }

    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #1f2a2e !important;
        margin-bottom: 4px;
    }

    .sub-title {
        font-size: 15px;
        color: #5b6b6f !important;
        margin-bottom: 20px;
    }

    .card {
        background: rgba(255,255,255,0.92);
        padding: 18px 20px;
        border-radius: 18px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.04);
        margin-bottom: 14px;
    }

    .metric-label {
        font-size: 14px;
        color: #6b7c80 !important;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 700;
        color: #1f2a2e !important;
    }

    .risk-low {
        color: #1b8f5a !important;
        font-weight: 700;
    }

    .risk-moderate {
        color: #b8860b !important;
        font-weight: 700;
    }

    .risk-elevated {
        color: #d97706 !important;
        font-weight: 700;
    }

    .risk-high {
        color: #c0392b !important;
        font-weight: 700;
    }

    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #1f2a2e !important;
        margin-bottom: 10px;
    }

    .small-note {
        font-size: 13px;
        color: #6b7c80 !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.92);
        padding: 14px 16px;
        border-radius: 18px;
        border: 1px solid rgba(0,0,0,0.04);
        box-shadow: 0 6px 20px rgba(0,0,0,0.05);
    }

    div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.92);
        border-radius: 18px;
        border: 1px solid rgba(0,0,0,0.04);
        overflow: hidden;
    }

    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.78);
        border-right: 1px solid rgba(0,0,0,0.05);
    }

    section[data-testid="stSidebar"] * {
        color: black !important;
    }

    .stTextInput input,
    .stNumberInput input,
    .stDateInput input,
    .stTextArea textarea {
        background-color: rgba(255,255,255,0.95) !important;
        color: black !important;
        border-radius: 12px !important;
    }

    .stSlider label,
    .stNumberInput label,
    .stSelectbox label,
    .stDateInput label,
    .stTextInput label {
        color: black !important;
        font-weight: 600;
    }

    /* Number input */
    [data-testid="stNumberInput"] div[data-baseweb="input"] {
        background: rgba(255,255,255,0.95) !important;
        border-radius: 14px !important;
    }

    [data-testid="stNumberInput"] button {
        background: transparent !important;
        color: black !important;
        border: none !important;
        box-shadow: none !important;
    }

    [data-testid="stNumberInput"] button:hover {
        background: rgba(0,0,0,0.04) !important;
        color: black !important;
    }

    [data-testid="stNumberInput"] button svg {
        fill: black !important;
    }

    [data-testid="stNumberInput"] div[data-baseweb="input"] > div {
        background: transparent !important;
    }

    /* Date input */
    [data-testid="stDateInput"] div[data-baseweb="input"] {
        background: rgba(255,255,255,0.95) !important;
        border-radius: 14px !important;
    }

    [data-testid="stDateInput"] div[data-baseweb="input"] > div {
        background: transparent !important;
    }

    [data-testid="stDateInput"] button {
        background: transparent !important;
        color: black !important;
        border: none !important;
        box-shadow: none !important;
    }

    [data-testid="stDateInput"] button:hover {
        background: rgba(0,0,0,0.04) !important;
        color: black !important;
    }

    [data-testid="stDateInput"] button svg {
        fill: black !important;
    }

    /* Selectbox closed state */
    .stSelectbox div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.98) !important;
        color: black !important;
        border-radius: 14px !important;
        border: 1px solid rgba(0,0,0,0.08) !important;
        box-shadow: none !important;
    }

    .stSelectbox div[data-baseweb="select"] * {
        color: black !important;
    }

    /* Dropdown popover */
    div[data-baseweb="popover"] {
        background: transparent !important;
    }

    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] [role="listbox"] {
        background: rgba(255,255,255,0.98) !important;
        border: 1px solid rgba(0,0,0,0.08) !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 24px rgba(0,0,0,0.08) !important;
        padding: 6px !important;
    }

    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] [role="option"] {
        background: transparent !important;
        color: black !important;
        border-radius: 12px !important;
    }

    div[data-baseweb="popover"] li *,
    div[data-baseweb="popover"] [role="option"] * {
        color: black !important;
        background: transparent !important;
        fill: black !important;
    }

    div[data-baseweb="popover"] li:hover,
    div[data-baseweb="popover"] [role="option"]:hover {
        background: rgba(69,198,168,0.12) !important;
        color: black !important;
    }

    div[data-baseweb="popover"] li[aria-selected="true"],
    div[data-baseweb="popover"] [role="option"][aria-selected="true"] {
        background: rgba(69,198,168,0.18) !important;
        color: black !important;
    }

    div[data-baseweb="popover"] li > div,
    div[data-baseweb="popover"] [role="option"] > div,
    div[data-baseweb="popover"] li[aria-selected="true"] > div,
    div[data-baseweb="popover"] [role="option"][aria-selected="true"] > div {
        background: transparent !important;
        color: black !important;
    }

    div[data-baseweb="popover"] div {
        color: black !important;
    }

    div[data-baseweb="popover"]::-webkit-scrollbar,
    div[data-baseweb="popover"] ul::-webkit-scrollbar {
        width: 8px;
    }

    /* Sidebar buttons */
    section[data-testid="stSidebar"] form {
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    section[data-testid="stSidebar"] .stButton,
    section[data-testid="stSidebar"] div[data-testid="stFormSubmitButton"] {
        width: 100% !important;
        margin: 0 0 12px 0 !important;
        padding: 0 !important;
    }

    section[data-testid="stSidebar"] .stButton > button,
    section[data-testid="stSidebar"] div[data-testid="stFormSubmitButton"] > button {
        width: 100% !important;
        min-height: 54px !important;
        padding: 0 16px !important;
        margin: 0 !important;
        border-radius: 16px !important;
        border: none !important;
        background: linear-gradient(135deg, #45c6a8 0%, #6dd7bf 100%) !important;
        color: black !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        box-shadow: 0 6px 16px rgba(69,198,168,0.24) !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover,
    section[data-testid="stSidebar"] div[data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(135deg, #3ab596 0%, #5dcdb3 100%) !important;
        color: black !important;
    }

    [data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(0,0,0,0.06);
    }

    [data-testid="stDataFrame"] * {
        color: black !important;
    }

    [data-testid="stDataFrame"] div[role="table"] {
        background: rgba(255,255,255,0.96) !important;
    }

    button[data-baseweb="tab"] {
        color: black !important;
    }

    [data-testid="stPlotlyChart"] {
        background: transparent !important;
    }

    [data-testid="stInfo"] {
        background: rgba(255,255,255,0.85) !important;
        color: black !important;
        border: 1px solid rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOAD ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():
    missing = [p.name for p in [MODEL_PATH, PREPROCESSOR_PATH, SCHEMA_PATH, TRAIN_REF_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifact files: {missing}")

    model = lgb.Booster(model_file=str(MODEL_PATH))
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)

    train_ref = pd.read_csv(TRAIN_REF_PATH)
    return model, preprocessor, schema, train_ref

model, preprocessor, schema, train_ref = load_artifacts()

cat_cols = schema["cat_cols"]
num_cols = schema["num_cols"]
feature_cols = schema["feature_cols"]
feature_names_transformed = preprocessor.get_feature_names_out()

# ============================================================
# FRED HELPERS
# ============================================================

@st.cache_data(ttl=3600)
def fetch_fred_series_csv(series_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_code}"
    s = pd.read_csv(url)

    if s.shape[1] < 2:
        raise ValueError(f"FRED response for {series_code} does not have 2 columns. Columns found: {list(s.columns)}")

    s = s.rename(columns={s.columns[0]: "DATE", s.columns[1]: series_code})
    s["DATE"] = pd.to_datetime(s["DATE"], errors="coerce")
    s[series_code] = pd.to_numeric(s[series_code], errors="coerce")

    s = s[(s["DATE"] >= pd.Timestamp(start_date)) & (s["DATE"] <= pd.Timestamp(end_date))].copy()
    s = s.dropna(subset=["DATE"])
    s = s.set_index("DATE")
    s.index = s.index.to_period("M").to_timestamp()
    return s[[series_code]]

@st.cache_data(ttl=3600)
def get_macro_features_for_date(application_date) -> dict:
    app_month = pd.Timestamp(application_date).to_period("M").to_timestamp()

    start_date = (app_month - pd.DateOffset(months=18)).strftime("%Y-%m-%d")
    end_date = app_month.strftime("%Y-%m-%d")

    fred_codes = {
        "Inflation": "CPIAUCSL",
        "FedFunds": "FEDFUNDS",
        "HomePrices": "CSUSHPISA",
        "UNRATE": "UNRATE"
    }

    macro = pd.DataFrame()
    for name, code in fred_codes.items():
        s = fetch_fred_series_csv(code, start_date, end_date)
        macro[name] = s[code]

    macro = macro.sort_index().ffill()

    return {
        "Inflation_L6": float(macro["Inflation"].shift(6).dropna().iloc[-1]),
        "FedFunds_L3": float(macro["FedFunds"].shift(3).dropna().iloc[-1]),
        "HomePrices_L12": float(macro["HomePrices"].shift(12).dropna().iloc[-1]),
        "UNRATE_L6": float(macro["UNRATE"].shift(6).dropna().iloc[-1]),
    }

# ============================================================
# MODEL HELPERS
# ============================================================

def build_single_applicant_row(applicant_dict: dict, application_date):
    macro_dict = get_macro_features_for_date(application_date)
    row = {**applicant_dict, **macro_dict}
    row_df = pd.DataFrame([row])

    for col in feature_cols:
        if col not in row_df.columns:
            row_df[col] = np.nan

    row_df = row_df[feature_cols]
    return row_df, macro_dict

def predict_single_applicant(applicant_dict: dict, application_date):
    row_df, macro_dict = build_single_applicant_row(applicant_dict, application_date)
    X_row = preprocessor.transform(row_df)
    pd_hat = float(model.predict(X_row)[0])
    return pd_hat, row_df, X_row, macro_dict

def get_risk_bucket(pd_hat: float):
    if pd_hat < 0.05:
        return "Low Risk", "risk-low"
    elif pd_hat < 0.10:
        return "Moderate Risk", "risk-moderate"
    elif pd_hat < 0.20:
        return "Elevated Risk", "risk-elevated"
    return "High Risk", "risk-high"

def get_recommendation(pd_hat: float):
    if pd_hat < 0.05:
        return "Approve"
    elif pd_hat < 0.12:
        return "Review"
    return "Manual Review"

def make_gauge(pd_hat: float):
    value = pd_hat * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"size": 34, "color": "black"}},
        title={"text": "Predicted Default Probability", "font": {"size": 18, "color": "black"}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": "black"}},
            "bar": {"color": "#2e8b75"},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 5], "color": "#d9f2e6"},
                {"range": [5, 10], "color": "#cdecc8"},
                {"range": [10, 20], "color": "#f8dfb0"},
                {"range": [20, 100], "color": "#f4c4bd"},
            ],
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="black")
    )
    return fig

# ============================================================
# EXPLANATION HELPERS
# ============================================================

pretty_raw_name_map = {
    "fico_range_low": "FICO score",
    "loan_amnt": "Loan amount",
    "dti": "Debt-to-income ratio",
    "revol_util": "Revolving utilization",
    "annual_inc": "Annual income",
    "open_acc": "Open accounts",
    "delinq_2yrs": "Recent delinquencies",
    "pub_rec": "Public records",
    "inq_last_6mths": "Recent credit inquiries",
    "mort_acc": "Mortgage accounts",
    "revol_bal": "Revolving balance",
    "Inflation_L6": "Inflation backdrop",
    "FedFunds_L3": "Fed funds backdrop",
    "HomePrices_L12": "Home price backdrop",
    "UNRATE_L6": "Unemployment backdrop",
    "term": "Loan term",
    "home_ownership": "Home ownership",
    "purpose": "Loan purpose",
    "emp_length": "Employment length"
}

def get_raw_feature_name(transformed_name: str, cat_cols: list, num_cols: list):
    if transformed_name.startswith("num__"):
        return transformed_name.replace("num__", "")

    if transformed_name.startswith("cat__"):
        cat_part = transformed_name.replace("cat__", "")
        for c in cat_cols:
            prefix = c + "_"
            if cat_part.startswith(prefix):
                return c
        return cat_part

    return transformed_name

def group_contributions(contrib_df: pd.DataFrame, cat_cols: list, num_cols: list):
    tmp = contrib_df.copy()
    tmp["raw_feature"] = tmp["feature"].apply(lambda x: get_raw_feature_name(x, cat_cols, num_cols))

    grouped = (
        tmp.groupby("raw_feature", as_index=False)
        .agg({"shap_value": "sum"})
    )

    grouped["abs_shap"] = grouped["shap_value"].abs()
    grouped["pretty_feature"] = grouped["raw_feature"].map(pretty_raw_name_map).fillna(grouped["raw_feature"])
    grouped = grouped.sort_values("abs_shap", ascending=False).reset_index(drop=True)
    return grouped

def classify_reason_groups(grouped_df: pd.DataFrame, top_positive=4, top_negative=3):
    pos_df = grouped_df[grouped_df["shap_value"] > 0].sort_values("shap_value", ascending=False).copy()
    neg_df = grouped_df[grouped_df["shap_value"] < 0].sort_values("shap_value", ascending=True).copy()

    high_risk = pos_df.head(2)["pretty_feature"].tolist()
    medium_risk = pos_df.iloc[2:top_positive]["pretty_feature"].tolist()
    reducing = neg_df.head(top_negative)["pretty_feature"].tolist()

    return {
        "high_risk_factors": high_risk,
        "medium_risk_factors": medium_risk,
        "risk_reducing_factors": reducing
    }

def join_nicely(items: list):
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"

def make_explanation_paragraph(reason_groups: dict, predicted_pd=None, risk_bucket=None):
    high_risk = reason_groups.get("high_risk_factors", [])
    medium_risk = reason_groups.get("medium_risk_factors", [])
    reducing = reason_groups.get("risk_reducing_factors", [])

    intro = ""
    if predicted_pd is not None and risk_bucket is not None:
        intro = f"This applicant is classified as {risk_bucket} with an estimated probability of default of {predicted_pd:.1%}. "

    para = intro
    if high_risk:
        para += f"The strongest contributors to higher risk are {join_nicely(high_risk)}. "
    if medium_risk:
        para += f"Additional moderate risk pressure comes from {join_nicely(medium_risk)}. "
    if reducing:
        para += f"However, some factors help reduce the predicted risk, particularly {join_nicely(reducing)}."
    return para.strip()

def get_prediction_explanation(X_row):
    contrib = model.predict(X_row, pred_contrib=True)
    sv = np.array(contrib)[0]

    if len(sv) == len(feature_names_transformed) + 1:
        sv = sv[:-1]

    contrib_df = pd.DataFrame({
        "feature": feature_names_transformed,
        "shap_value": sv
    })

    contrib_df["abs_shap"] = contrib_df["shap_value"].abs()
    contrib_df = contrib_df.sort_values("abs_shap", ascending=False).reset_index(drop=True)

    grouped = group_contributions(contrib_df, cat_cols, num_cols)
    return contrib_df, grouped

def make_contribution_chart(grouped_df: pd.DataFrame, top_n=8):
    plot_df = grouped_df.head(top_n).copy()
    plot_df = plot_df.sort_values("shap_value", ascending=True)
    plot_df["direction"] = np.where(plot_df["shap_value"] >= 0, "Increase Risk", "Reduce Risk")

    fig = px.bar(
        plot_df,
        x="shap_value",
        y="pretty_feature",
        orientation="h",
        color="direction",
        color_discrete_map={
            "Increase Risk": "#ef7d73",
            "Reduce Risk": "#45c6a8"
        },
        labels={"shap_value": "Contribution", "pretty_feature": "Feature"},
    )

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="black"),
        legend=dict(font=dict(color="black"))
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=True,
        zerolinecolor="rgba(0,0,0,0.15)",
        tickfont=dict(color="black"),
        title_font=dict(color="black")
    )

    fig.update_yaxes(
        showgrid=False,
        tickfont=dict(color="black"),
        title_font=dict(color="black")
    )
    return fig

# ============================================================
# SESSION STATE
# ============================================================

if "prediction_ready" not in st.session_state:
    st.session_state.prediction_ready = False
if "saved_applicant" not in st.session_state:
    st.session_state.saved_applicant = None
if "saved_application_date" not in st.session_state:
    st.session_state.saved_application_date = None
if "saved_pd_hat" not in st.session_state:
    st.session_state.saved_pd_hat = None
if "saved_row_df" not in st.session_state:
    st.session_state.saved_row_df = None
if "saved_macro_dict" not in st.session_state:
    st.session_state.saved_macro_dict = None
if "saved_grouped_contrib_df" not in st.session_state:
    st.session_state.saved_grouped_contrib_df = None
if "saved_risk_bucket" not in st.session_state:
    st.session_state.saved_risk_bucket = None
if "saved_risk_class" not in st.session_state:
    st.session_state.saved_risk_class = None
if "saved_recommendation" not in st.session_state:
    st.session_state.saved_recommendation = None
if "saved_reason_groups" not in st.session_state:
    st.session_state.saved_reason_groups = None
if "saved_explanation_paragraph" not in st.session_state:
    st.session_state.saved_explanation_paragraph = None

# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="main-title">Macro-Aware Credit Risk Scoring</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Enter borrower information, fetch macro context automatically, and estimate probability of default.</div>',
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR INPUT
# ============================================================

st.sidebar.header("Borrower Profile")

with st.sidebar.form("borrower_form", clear_on_submit=False):
    application_date = st.date_input(
        "Application date",
        value=st.session_state.saved_application_date if st.session_state.saved_application_date else pd.Timestamp("2018-01-01").date()
    )

    term_options = ["36 months", "60 months"]
    home_ownership_options = ["RENT", "MORTGAGE", "OWN", "OTHER"]
    purpose_options = [
        "debt_consolidation", "credit_card", "home_improvement", "small_business",
        "major_purchase", "car", "medical", "vacation", "moving", "house", "wedding", "other"
    ]
    emp_length_options = [
        "< 1 year", "1 year", "2 years", "3 years", "4 years",
        "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"
    ]

    saved = st.session_state.saved_applicant

    term = st.selectbox(
        "Loan term",
        term_options,
        index=0 if not saved else term_options.index(saved["term"])
    )

    home_ownership = st.selectbox(
        "Home ownership",
        home_ownership_options,
        index=0 if not saved else home_ownership_options.index(saved["home_ownership"])
    )

    purpose = st.selectbox(
        "Loan purpose",
        purpose_options,
        index=0 if not saved else purpose_options.index(saved["purpose"])
    )

    emp_length = st.selectbox(
        "Employment length",
        emp_length_options,
        index=3 if not saved else emp_length_options.index(saved["emp_length"])
    )

    fico_range_low = st.number_input(
        "FICO score",
        min_value=300,
        max_value=850,
        value=680 if not saved else int(saved["fico_range_low"]),
        step=1
    )

    loan_amnt = st.number_input(
        "Loan amount",
        min_value=500,
        max_value=100000,
        value=15000 if not saved else int(saved["loan_amnt"]),
        step=500
    )

    dti = st.number_input(
        "Debt-to-income ratio",
        min_value=0.0,
        max_value=100.0,
        value=18.0 if not saved else float(saved["dti"]),
        step=0.1
    )

    revol_util = st.number_input(
        "Revolving utilization",
        min_value=0.0,
        max_value=150.0,
        value=45.0 if not saved else float(saved["revol_util"]),
        step=0.1
    )

    annual_inc = st.number_input(
        "Annual income",
        min_value=0.0,
        max_value=1000000.0,
        value=70000.0 if not saved else float(saved["annual_inc"]),
        step=1000.0
    )

    open_acc = st.number_input(
        "Open accounts",
        min_value=0,
        max_value=100,
        value=8 if not saved else int(saved["open_acc"]),
        step=1
    )

    delinq_2yrs = st.number_input(
        "Recent delinquencies",
        min_value=0,
        max_value=20,
        value=0 if not saved else int(saved["delinq_2yrs"]),
        step=1
    )

    pub_rec = st.number_input(
        "Public records",
        min_value=0,
        max_value=20,
        value=0 if not saved else int(saved["pub_rec"]),
        step=1
    )

    inq_last_6mths = st.number_input(
        "Recent credit inquiries (6m)",
        min_value=0,
        max_value=50,
        value=1 if not saved else int(saved["inq_last_6mths"]),
        step=1
    )

    mort_acc = st.number_input(
        "Mortgage accounts",
        min_value=0,
        max_value=50,
        value=0 if not saved else int(saved["mort_acc"]),
        step=1
    )

    revol_bal = st.number_input(
        "Revolving balance",
        min_value=0.0,
        max_value=500000.0,
        value=12000.0 if not saved else float(saved["revol_bal"]),
        step=500.0
    )

    run_btn = st.form_submit_button("Run Prediction", use_container_width=True)

if st.sidebar.button("Clear Saved Prediction", use_container_width=True):
    keys_to_clear = [
        "prediction_ready", "saved_applicant", "saved_application_date",
        "saved_pd_hat", "saved_row_df", "saved_macro_dict", "saved_grouped_contrib_df",
        "saved_risk_bucket", "saved_risk_class", "saved_recommendation",
        "saved_reason_groups", "saved_explanation_paragraph"
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            st.session_state[k] = None if k != "prediction_ready" else False
    st.rerun()

# ============================================================
# MAIN
# ============================================================

if run_btn:
    applicant = {
        "term": term,
        "home_ownership": home_ownership,
        "purpose": purpose,
        "emp_length": emp_length,
        "fico_range_low": fico_range_low,
        "loan_amnt": loan_amnt,
        "dti": dti,
        "revol_util": revol_util,
        "annual_inc": annual_inc,
        "open_acc": open_acc,
        "delinq_2yrs": delinq_2yrs,
        "pub_rec": pub_rec,
        "inq_last_6mths": inq_last_6mths,
        "mort_acc": mort_acc,
        "revol_bal": revol_bal,
    }

    try:
        pd_hat, row_df, X_row, macro_dict = predict_single_applicant(applicant, application_date)
        _, grouped_contrib_df = get_prediction_explanation(X_row)

        risk_bucket, risk_class = get_risk_bucket(pd_hat)
        recommendation = get_recommendation(pd_hat)
        reason_groups = classify_reason_groups(grouped_contrib_df)
        explanation_paragraph = make_explanation_paragraph(reason_groups, pd_hat, risk_bucket)

        st.session_state.prediction_ready = True
        st.session_state.saved_applicant = applicant
        st.session_state.saved_application_date = application_date
        st.session_state.saved_pd_hat = pd_hat
        st.session_state.saved_row_df = row_df
        st.session_state.saved_macro_dict = macro_dict
        st.session_state.saved_grouped_contrib_df = grouped_contrib_df
        st.session_state.saved_risk_bucket = risk_bucket
        st.session_state.saved_risk_class = risk_class
        st.session_state.saved_recommendation = recommendation
        st.session_state.saved_reason_groups = reason_groups
        st.session_state.saved_explanation_paragraph = explanation_paragraph

    except Exception as e:
        st.error(f"Prediction failed: {e}")

if st.session_state.prediction_ready:
    applicant = st.session_state.saved_applicant
    application_date = st.session_state.saved_application_date
    pd_hat = st.session_state.saved_pd_hat
    row_df = st.session_state.saved_row_df
    macro_dict = st.session_state.saved_macro_dict
    grouped_contrib_df = st.session_state.saved_grouped_contrib_df
    risk_bucket = st.session_state.saved_risk_bucket
    risk_class = st.session_state.saved_risk_class
    recommendation = st.session_state.saved_recommendation
    reason_groups = st.session_state.saved_reason_groups
    explanation_paragraph = st.session_state.saved_explanation_paragraph

    portfolio_default_rate = float(train_ref["default"].mean()) if "default" in train_ref.columns else np.nan

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f'<div class="card"><div class="metric-label">Predicted PD</div><div class="metric-value">{pd_hat:.2%}</div></div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f'<div class="card"><div class="metric-label">Risk Bucket</div><div class="metric-value {risk_class}">{risk_bucket}</div></div>',
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f'<div class="card"><div class="metric-label">Recommendation</div><div class="metric-value" style="font-size:24px;">{recommendation}</div></div>',
            unsafe_allow_html=True
        )
    with c4:
        base_rate_text = f"{portfolio_default_rate:.2%}" if pd.notna(portfolio_default_rate) else "N/A"
        st.markdown(
            f'<div class="card"><div class="metric-label">Training Base Default Rate</div><div class="metric-value">{base_rate_text}</div></div>',
            unsafe_allow_html=True
        )

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Default Probability</div>', unsafe_allow_html=True)
        st.plotly_chart(make_gauge(pd_hat), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Macro Context Used</div>', unsafe_allow_html=True)

        macro_name_map = {
            "Inflation_L6": "Inflation",
            "FedFunds_L3": "Federal Funds Rate",
            "HomePrices_L12": "Home Price Index",
            "UNRATE_L6": "Unemployment Rate"
        }

        macro_value_map = {
            "Inflation_L6": lambda x: f"{x:,.3f}",
            "FedFunds_L3": lambda x: f"{x:.2f}%",
            "HomePrices_L12": lambda x: f"{x:,.3f}",
            "UNRATE_L6": lambda x: f"{x:.1f}%"
        }

        macro_df = pd.DataFrame(
            {
                "Macro Variable": [macro_name_map.get(k, k) for k in macro_dict.keys()],
                "Value": [macro_value_map.get(k, lambda v: v)(v) for k, v in macro_dict.items()]
            }
        )

        st.dataframe(macro_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Why this result?</div>', unsafe_allow_html=True)
        st.write(explanation_paragraph)

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.markdown("**High risk**")
            for x in reason_groups["high_risk_factors"]:
                st.write(f"- {x}")
        with rc2:
            st.markdown("**Moderate risk**")
            for x in reason_groups["medium_risk_factors"]:
                st.write(f"- {x}")
        with rc3:
            st.markdown("**Risk reducing**")
            for x in reason_groups["risk_reducing_factors"]:
                st.write(f"- {x}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Top Feature Contributions</div>', unsafe_allow_html=True)
        st.plotly_chart(make_contribution_chart(grouped_contrib_df), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">What-if Simulation</div>', unsafe_allow_html=True)

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        sim_loan = st.slider("Loan amount", 500, 100000, int(applicant["loan_amnt"]), step=500)
    with r1c2:
        sim_dti = st.slider("DTI", 0.0, 100.0, float(applicant["dti"]), step=0.1)
    with r1c3:
        sim_income = st.slider("Annual income", 1000, 1000000, int(applicant["annual_inc"]), step=1000)
    with r1c4:
        sim_fico = st.slider("FICO score", 300, 850, int(applicant["fico_range_low"]), step=1)

    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        sim_inq = st.slider("Recent inquiries", 0, 50, int(applicant["inq_last_6mths"]), step=1)
    with r2c2:
        sim_mort = st.slider("Mortgage accounts", 0, 50, int(applicant["mort_acc"]), step=1)
    with r2c3:
        sim_revol_bal = st.slider("Revolving balance", 0, 500000, int(applicant["revol_bal"]), step=500)

    sim_applicant = applicant.copy()
    sim_applicant["loan_amnt"] = sim_loan
    sim_applicant["dti"] = sim_dti
    sim_applicant["annual_inc"] = sim_income
    sim_applicant["fico_range_low"] = sim_fico
    sim_applicant["inq_last_6mths"] = sim_inq
    sim_applicant["mort_acc"] = sim_mort
    sim_applicant["revol_bal"] = sim_revol_bal

    sim_pd, _, _, _ = predict_single_applicant(sim_applicant, application_date)

    ss1, ss2 = st.columns(2)
    with ss1:
        st.metric("Current PD", f"{pd_hat:.2%}")
    with ss2:
        st.metric("Simulated PD", f"{sim_pd:.2%}", delta=f"{(sim_pd - pd_hat):.2%}")

    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Show applicant input and processed values"):
        st.dataframe(row_df, use_container_width=True)

    st.caption("This tool is for analytical and educational use and should support, not replace, human credit judgment.")

else:
    st.info("Fill in the borrower profile on the left and click Run Prediction.")