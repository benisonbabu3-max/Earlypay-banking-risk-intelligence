import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ============================================================
# EARLYPAY — BANKING RISK INTELLIGENCE
# Research prototype: UCI Default of Credit Card Clients (Taiwan)
# ============================================================

st.set_page_config(
    page_title="EarlyPay | Banking Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "processed" / "earlypay_clustered.csv"
MODEL_PATH = BASE_DIR / "models" / "earlypay_final_classifier.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "earlypay_threshold.pkl"
CLASSIFICATION_RESULTS_PATH = BASE_DIR / "models" / "classification_results.csv"

# -----------------------------
# VISUAL THEME
# -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background: #f4f7fb;
        color: #17243a;
    }
    [data-testid="stSidebar"] {
        background: #101d35;
    }
    [data-testid="stSidebar"] * {
        color: #eef4ff !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 7px 4px;
    }
    .main-title {
        color: #14233d;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }
    .sub-title {
        color: #687991;
        font-size: 1rem;
        margin-bottom: 1.4rem;
    }
    .eyebrow {
        color: #4776c7;
        text-transform: uppercase;
        letter-spacing: 0.11em;
        font-size: 0.72rem;
        font-weight: 700;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5ebf3;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 3px 12px rgba(20, 35, 61, 0.035);
    }
    div[data-testid="stMetricLabel"] {
        color: #687991;
    }
    div[data-testid="stMetricValue"] {
        color: #14233d;
    }
    .section-card {
        background: #ffffff;
        border: 1px solid #e5ebf3;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }
    .risk-high {
        background: #fff1f0;
        border: 1px solid #ffd0cc;
        color: #a52a25;
        border-radius: 12px;
        padding: 16px 18px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .risk-low {
        background: #edf8f2;
        border: 1px solid #c9ead7;
        color: #176b43;
        border-radius: 12px;
        padding: 16px 18px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .small-note {
        color: #718096;
        font-size: 0.82rem;
    }
    div.stButton > button[kind="primary"],
    div.stFormSubmitButton > button {
        background: #2459a6;
        border: 0;
        border-radius: 9px;
        color: white;
        font-weight: 600;
        min-height: 2.7rem;
    }
    div.stButton > button[kind="primary"]:hover,
    div.stFormSubmitButton > button:hover {
        background: #194783;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# LOAD DATA AND MODEL
# -----------------------------
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_models():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Classifier not found: {MODEL_PATH}")
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"Threshold file not found: {THRESHOLD_PATH}")

    loaded_model = joblib.load(MODEL_PATH)
    loaded_threshold = float(joblib.load(THRESHOLD_PATH))
    return loaded_model, loaded_threshold


try:
    df = load_data()
    model, threshold = load_models()
except Exception as exc:
    st.error("EarlyPay could not load its data or model files.")
    st.exception(exc)
    st.stop()


# -----------------------------
# FEATURE ENGINEERING + PREDICTION
# -----------------------------
REPAYMENT_COLS = [f"repayment_status_{i}" for i in range(1, 7)]
BILL_COLS = [f"bill_amount_{i}" for i in range(1, 7)]
PAYMENT_COLS = [f"payment_amount_{i}" for i in range(1, 7)]


def prepare_customer_features(customer):
    """Create the engineered variables used by the trained model."""
    customer = customer.copy()

    repayment = np.array([customer[c] for c in REPAYMENT_COLS], dtype=float)
    bills = np.array([customer[c] for c in BILL_COLS], dtype=float)
    payments = np.array([customer[c] for c in PAYMENT_COLS], dtype=float)

    bill_1 = float(customer["bill_amount_1"])
    payment_1 = float(customer["payment_amount_1"])
    credit_limit = float(customer["credit_limit"])

    ratio = payment_1 / bill_1 if bill_1 > 0 else np.nan

    customer.update(
        {
            "payment_bill_ratio": ratio,
            "delay_months_count": int((repayment > 0).sum()),
            "max_delay_severity": float(repayment.max()),
            "avg_delay_severity": float(np.maximum(repayment, 0).mean()),
            "avg_bill_amount": float(bills.mean()),
            "payment_to_bill_ratio": ratio,
            "credit_utilization": (
                bill_1 / credit_limit if credit_limit > 0 else np.nan
            ),
            "bill_trend": float(bills[0] - bills[1]),
            "payment_trend": float(payments[0] - payments[1]),
        }
    )

    # Always use the exact feature names and order from model training.
    feature_order = list(model.feature_names_in_)
    return pd.DataFrame([[customer[col] for col in feature_order]], columns=feature_order)


def predict_customer_risk(customer):
    features = prepare_customer_features(customer)
    probability = float(model.predict_proba(features)[0][1])

    return {
        "probability": probability,
        "risk": "HIGH RISK" if probability >= threshold else "LOWER RISK",
        "features": features,
    }


def get_dataset_predictions():
    """Return cached model probabilities for the historical dataset."""
    feature_order = list(model.feature_names_in_)
    missing = [col for col in feature_order if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing model features: {missing}")
    return model.predict_proba(df[feature_order])[:, 1]


# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.markdown("## 🏦 EarlyPay")
st.sidebar.caption("BANKING RISK INTELLIGENCE")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Customer Analytics",
        "Risk Prediction",
        "Segmentation",
        "Model Performance",
    ],
)

st.sidebar.divider()
st.sidebar.caption("ML-powered repayment stress research prototype")
st.sidebar.caption("Dataset: UCI • Taiwan")


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================
if page == "Overview":
    st.markdown('<div class="eyebrow">BANKING RISK INTELLIGENCE</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">EarlyPay Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">An early warning view of customer repayment behaviour and potential default risk.</div>',
        unsafe_allow_html=True,
    )

    probabilities = get_dataset_predictions()
    flagged_count = int((probabilities >= threshold).sum())
    total_customers = len(df)
    historical_default_rate = float(df["default_payment"].mean() * 100)
    flagged_rate = flagged_count / total_customers * 100 if total_customers else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customers in dataset", f"{total_customers:,}")
    m2.metric("Historical default rate", f"{historical_default_rate:.2f}%")
    m3.metric("Customers flagged by model", f"{flagged_count:,}")
    m4.metric("Model-flagged share", f"{flagged_rate:.1f}%")

    st.divider()
    left, right = st.columns([1.05, 1])

    with left:
        st.subheader("Historical default distribution")
        status_counts = (
            df["default_payment"]
            .value_counts()
            .reindex([0, 1], fill_value=0)
            .rename(index={0: "No historical default", 1: "Historical default"})
            .rename_axis("Customer outcome")
            .reset_index(name="Customers")
        )
        fig = px.pie(
            status_counts,
            names="Customer outcome",
            values="Customers",
            hole=0.62,
            color_discrete_sequence=["#3c78c8", "#e28a65"],
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=20, b=10),
            legend_title_text="",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Model risk classification")
        risk_counts = pd.DataFrame(
            {
                "Risk classification": ["Flagged by model", "Below threshold"],
                "Customers": [flagged_count, total_customers - flagged_count],
            }
        )
        fig = px.bar(
            risk_counts,
            x="Risk classification",
            y="Customers",
            color="Risk classification",
            text="Customers",
            color_discrete_sequence=["#e28a65", "#3c78c8"],
        )
        fig.update_layout(
            showlegend=False,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"The model flags a customer when the estimated probability is at least "
        f"{threshold:.4f}. This threshold was selected on validation data with a "
        f"target recall of 70%; it is a research setting, not a bank policy."
    )
    st.caption(
        "Dataset note: this prototype uses the UCI Default of Credit Card Clients "
        "dataset from Taiwan. Results should not be assumed to represent Indian bank customers."
    )


# ============================================================
# PAGE 2 — CUSTOMER ANALYTICS
# ============================================================
elif page == "Customer Analytics":
    st.markdown('<div class="eyebrow">PORTFOLIO EXPLORATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Customer Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Explore historical customer characteristics and repayment patterns.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Age distribution by historical outcome")
        age_df = df.copy()
        age_df["Historical outcome"] = age_df["default_payment"].map(
            {0: "No default", 1: "Default"}
        )
        fig = px.histogram(
            age_df,
            x="age",
            color="Historical outcome",
            barmode="overlay",
            opacity=0.78,
            nbins=25,
            color_discrete_sequence=["#3c78c8", "#e28a65"],
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Credit limit by historical outcome")
        box_df = df.copy()
        box_df["Historical outcome"] = box_df["default_payment"].map(
            {0: "No default", 1: "Default"}
        )
        fig = px.box(
            box_df,
            x="Historical outcome",
            y="credit_limit",
            color="Historical outcome",
            color_discrete_sequence=["#3c78c8", "#e28a65"],
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Repayment delay patterns")
    delay_cols = [f"repayment_status_{i}" for i in range(1, 7)]
    delay_summary = (
        df.groupby("default_payment")[delay_cols]
        .apply(lambda x: (x > 0).mean() * 100)
        .T.reset_index()
        .rename(columns={"index": "Month"})
    )
    delay_summary = delay_summary.rename(
        columns={0: "No default (%)", 1: "Default (%)"}
    )
    delay_summary["Month"] = [
        f"Month {i}" for i in range(1, 7)
    ]
    delay_long = delay_summary.melt(
        id_vars="Month",
        var_name="Historical outcome",
        value_name="Customers with delay (%)",
    )
    fig = px.bar(
        delay_long,
        x="Month",
        y="Customers with delay (%)",
        color="Historical outcome",
        barmode="group",
        color_discrete_sequence=["#3c78c8", "#e28a65"],
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "These charts describe associations in the historical dataset. They do not establish causation."
    )


# ============================================================
# PAGE 3 — RISK PREDICTION
# ============================================================
elif page == "Risk Prediction":
    st.markdown('<div class="eyebrow">INDIVIDUAL CUSTOMER ASSESSMENT</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Customer Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Enter customer profile, repayment history, bills and payments to estimate repayment risk.</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Complete all sections using the same six-month order: Month 1 is the most recent month "
        "in this dataset; Month 6 is the oldest month."
    )

    repayment_options = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    repayment_labels = {
        -2: "-2 — No consumption",
        -1: "-1 — Paid duly",
        0: "0 — Revolving credit",
        1: "1 — 1 month delay",
        2: "2 — 2 months delay",
        3: "3 — 3 months delay",
        4: "4 — 4 months delay",
        5: "5 — 5 months delay",
        6: "6 — 6 months delay",
        7: "7 — 7 months delay",
        8: "8 — 8 months delay",
        9: "9 — 9+ months delay",
    }

    with st.form("earlypay_customer_risk_form"):
        st.subheader("1. Customer profile")
        p1, p2, p3 = st.columns(3)

        with p1:
            credit_limit = st.number_input(
                "Credit limit (NT$)",
                min_value=0.0,
                max_value=10_000_000.0,
                value=50_000.0,
                step=5_000.0,
            )
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=30,
                step=1,
            )

        with p2:
            gender = st.selectbox(
                "Gender code",
                options=[1, 2],
                format_func=lambda x: "1 — Male" if x == 1 else "2 — Female",
            )
            education = st.selectbox(
                "Education code",
                options=[1, 2, 3, 4, 5, 6, 0],
                format_func=lambda x: {
                    1: "1 — Graduate school",
                    2: "2 — University",
                    3: "3 — High school",
                    4: "4 — Others",
                    5: "5 — Unknown",
                    6: "6 — Unknown",
                    0: "0 — Unknown",
                }[x],
            )

        with p3:
            marital_status = st.selectbox(
                "Marital status code",
                options=[1, 2, 3, 0],
                format_func=lambda x: {
                    1: "1 — Married",
                    2: "2 — Single",
                    3: "3 — Other",
                    0: "0 — Unknown",
                }[x],
            )

        st.divider()
        st.subheader("2. Six-month repayment status")
        st.caption(
            "Use the dataset's status codes. Positive values represent months of payment delay."
        )

        repayment_values = {}
        repayment_columns = st.columns(3)
        for i in range(1, 7):
            with repayment_columns[(i - 1) % 3]:
                repayment_values[f"repayment_status_{i}"] = st.selectbox(
                    f"Month {i} repayment status",
                    options=repayment_options,
                    index=2,
                    format_func=lambda x: repayment_labels[x],
                    key=f"repayment_status_input_{i}",
                )

        st.divider()
        st.subheader("3. Monthly bill amounts")
        st.caption("Enter statement bill amounts in New Taiwan dollars (NT$). Negative values are allowed because they can occur in the source data.")

        bill_values = {}
        bill_columns = st.columns(3)
        for i in range(1, 7):
            with bill_columns[(i - 1) % 3]:
                bill_values[f"bill_amount_{i}"] = st.number_input(
                    f"Month {i} bill amount (NT$)",
                    min_value=-1_000_000.0,
                    max_value=10_000_000.0,
                    value=10_000.0 if i == 1 else 8_000.0,
                    step=1_000.0,
                    key=f"bill_amount_input_{i}",
                )

        st.divider()
        st.subheader("4. Monthly payment amounts")
        payment_values = {}
        payment_columns = st.columns(3)
        for i in range(1, 7):
            with payment_columns[(i - 1) % 3]:
                payment_values[f"payment_amount_{i}"] = st.number_input(
                    f"Month {i} payment amount (NT$)",
                    min_value=0.0,
                    max_value=10_000_000.0,
                    value=2_000.0,
                    step=500.0,
                    key=f"payment_amount_input_{i}",
                )

        st.divider()
        submitted = st.form_submit_button(
            "Assess customer risk",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        customer_input = {
            "credit_limit": float(credit_limit),
            "gender": int(gender),
            "education": int(education),
            "marital_status": int(marital_status),
            "age": int(age),
            **repayment_values,
            **bill_values,
            **payment_values,
        }

        try:
            result = predict_customer_risk(customer_input)
            probability = result["probability"]
            risk = result["risk"]

            st.divider()
            st.subheader("Risk assessment result")

            result_col1, result_col2 = st.columns([1, 1.3])
            with result_col1:
                st.metric("Estimated default probability", f"{probability * 100:.2f}%")
                st.metric("Decision threshold", f"{threshold * 100:.2f}%")
                if risk == "HIGH RISK":
                    st.markdown(
                        '<div class="risk-high">HIGH RISK — Model score is at or above the threshold.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="risk-low">LOWER RISK — Model score is below the threshold.</div>',
                        unsafe_allow_html=True,
                    )

            with result_col2:
                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=probability * 100,
                        number={"suffix": "%", "font": {"size": 34, "color": "#17243a"}},
                        title={"text": "Estimated probability", "font": {"size": 16}},
                        gauge={
                            "axis": {"range": [0, 100], "ticksuffix": "%"},
                            "bar": {"color": "#2459a6"},
                            "steps": [
                                {"range": [0, threshold * 100], "color": "#e7f3ec"},
                                {"range": [threshold * 100, 100], "color": "#fde9e6"},
                            ],
                            "threshold": {
                                "line": {"color": "#b84a43", "width": 3},
                                "thickness": 0.8,
                                "value": threshold * 100,
                            },
                        },
                    )
                )
                gauge.update_layout(
                    height=260,
                    margin=dict(l=25, r=25, t=45, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(gauge, use_container_width=True)

            with st.expander("View engineered model inputs"):
                st.dataframe(result["features"].T.rename(columns={0: "Value"}), use_container_width=True)

            st.warning(
                "This is an academic research prototype trained on historical Taiwan credit-card data. "
                "The probability is a model estimate, not a guarantee of default, a credit score, "
                "or a recommendation to approve or reject a real customer."
            )

        except Exception as exc:
            st.error("The prediction could not be generated. Check the entered values and model files.")
            st.exception(exc)


# ============================================================
# PAGE 4 — SEGMENTATION
# ============================================================
elif page == "Segmentation":
    st.markdown('<div class="eyebrow">CUSTOMER BEHAVIOUR GROUPS</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Customer Segmentation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Compare the repayment behaviour of the clusters generated during model development.</div>',
        unsafe_allow_html=True,
    )

    if "customer_segment" not in df.columns:
        st.error("The clustered dataset does not contain customer_segment.")
    else:
        segment_summary = (
            df.groupby("customer_segment")
            .agg(
                Customers=("default_payment", "count"),
                Historical_Default_Rate=("default_payment", "mean"),
                Average_Delayed_Months=("delay_months_count", "mean"),
                Average_Delay_Severity=("avg_delay_severity", "mean"),
                Median_Payment_Bill_Ratio=("payment_to_bill_ratio", "median"),
            )
            .reset_index()
        )
        segment_summary["Historical_Default_Rate"] *= 100

        st.dataframe(
            segment_summary.style.format(
                {
                    "Historical_Default_Rate": "{:.2f}%",
                    "Average_Delayed_Months": "{:.2f}",
                    "Average_Delay_Severity": "{:.2f}",
                    "Median_Payment_Bill_Ratio": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                segment_summary,
                x="customer_segment",
                y="Historical_Default_Rate",
                text_auto=".1f",
                title="Historical default rate by segment",
                labels={
                    "customer_segment": "Cluster",
                    "Historical_Default_Rate": "Historical default rate (%)",
                },
                color="customer_segment",
                color_discrete_sequence=["#3c78c8", "#e28a65"],
            )
            fig.update_layout(
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.bar(
                segment_summary,
                x="customer_segment",
                y="Average_Delayed_Months",
                text_auto=".2f",
                title="Average delayed months by segment",
                labels={
                    "customer_segment": "Cluster",
                    "Average_Delayed_Months": "Average delayed months",
                },
                color="customer_segment",
                color_discrete_sequence=["#3c78c8", "#e28a65"],
            )
            fig.update_layout(
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Cluster numbers are algorithm-generated labels, not rankings. Historical default rate is shown "
            "after clustering to help profile the groups; it was not used as a clustering input."
        )


# ============================================================
# PAGE 5 — MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":
    st.markdown('<div class="eyebrow">MODEL EVALUATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Review the classification model comparison and the operating threshold used by EarlyPay.</div>',
        unsafe_allow_html=True,
    )

    st.metric("Validated decision threshold", f"{threshold:.4f}")

    if CLASSIFICATION_RESULTS_PATH.exists():
        results = pd.read_csv(CLASSIFICATION_RESULTS_PATH)
        st.subheader("Classification model comparison")
        st.dataframe(results, use_container_width=True, hide_index=True)

        numeric_columns = results.select_dtypes(include="number").columns.tolist()
        metric_candidates = [
            col for col in numeric_columns
            if any(term in col.lower() for term in ["accuracy", "precision", "recall", "f1", "auc", "roc"])
        ]

        if metric_candidates and len(results) > 0:
            model_name_col = next(
                (col for col in results.columns if col.lower() in ["model", "model_name", "classifier"]),
                None,
            )
            if model_name_col:
                metric_to_plot = st.selectbox(
                    "Compare metric",
                    metric_candidates,
                    index=metric_candidates.index(metric_candidates[0]),
                )
                fig = px.bar(
                    results,
                    x=model_name_col,
                    y=metric_to_plot,
                    text_auto=".3f",
                    title=f"Model comparison — {metric_to_plot}",
                    color=model_name_col,
                    color_discrete_sequence=["#3c78c8", "#e28a65", "#6d91c7"],
                )
                fig.update_layout(
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("classification_results.csv was not found in the models folder.")

    st.info(
        "The decision threshold was selected using a validation split from the training data, "
        "with a target recall of at least 70%. The held-out test set is used for final evaluation. "
        "A threshold is a research operating point and should not be treated as a production banking policy."
    )

    st.caption(
        "Interpret precision, recall, F1-score and ROC-AUC together. Accuracy alone can be misleading "
        "when default and non-default cases are imbalanced."
    )


# -----------------------------
# GLOBAL FOOTER
# -----------------------------
st.divider()
st.caption(
    "EarlyPay • Academic data science prototype • UCI Default of Credit Card Clients (Taiwan). "
    "Not validated for live banking decisions."
)