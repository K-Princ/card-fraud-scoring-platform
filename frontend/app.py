
import streamlit as st
import requests
import pandas as pd

from pathlib import Path

# ==================================================
# 1. CONFIGURATION
# ==================================================

API_URL = "http://localhost:8000"

BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports"

st.set_page_config(
    page_title="Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 2. API FUNCTIONS
# ==================================================

def get_api_data(endpoint):
    """Fetch data from the FastAPI backend."""

    try:
        response = requests.get(
            f"{API_URL}{endpoint}",
            timeout=10
        )

        response.raise_for_status()

        return response.json()

    except requests.RequestException:
        return None


def get_model_status():
    """Check whether the model-status endpoint responds."""

    return get_api_data("/model-status")


def submit_transaction(payload):
    """Send transaction details to the fraud prediction API."""

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        return response.json(), None

    except requests.RequestException as error:
        return None, str(error)


# ==================================================
# 3. MONITORING REPORT FUNCTIONS
# ==================================================

@st.cache_data
def load_performance_report():
    path = REPORTS_DIR / "model_performance.csv"

    if path.exists():
        return pd.read_csv(path)

    return None


@st.cache_data
def load_drift_report():
    path = REPORTS_DIR / "prediction_drift_summary.csv"

    if path.exists():
        return pd.read_csv(path, index_col=0)

    return None


@st.cache_data
def load_monitoring_summary():
    path = REPORTS_DIR / "monitoring_summary.md"

    if path.exists():
        return path.read_text(encoding="utf-8")

    return None


# ==================================================
# 4. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("🛡️ Fraud Detection")

st.sidebar.caption(
    "Real-Time Card Fraud Scoring"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Fraud Scoring",
        "Transaction History",
        "Model Monitoring"
    ]
)

st.sidebar.divider()

# Refresh button

if st.sidebar.button(
    "🔄 Refresh Dashboard",
    use_container_width=True
):
    st.cache_data.clear()
    st.rerun()


# System status

st.sidebar.subheader("System Status")

model_status = get_model_status()

if model_status is not None:

    st.sidebar.success("API Connected")

    st.sidebar.success("Model Service Available")

else:

    st.sidebar.error("API Unavailable")

    st.sidebar.warning("Model Status Unknown")

st.sidebar.divider()

st.sidebar.caption(
    "ML Fraud Detection Platform | v1.0.0"
)


# ==================================================
# 5. OVERVIEW DASHBOARD
# ==================================================

if page == "Overview":

    st.title("🛡️ Fraud Detection Platform")

    st.caption(
        "Real-Time Card Fraud Scoring & Monitoring"
    )

    st.divider()

    st.header("Dashboard Overview")

    predictions = get_api_data("/predictions")

    if predictions is None:

        st.error(
            "Unable to retrieve dashboard data. "
            "Check whether FastAPI is running."
        )

    elif not predictions:

        st.info(
            "No transactions have been analyzed yet. "
            "Visit Fraud Scoring to analyze your first transaction."
        )

    else:

        df = pd.DataFrame(predictions)

        # ------------------------------------------
        # KPI CALCULATIONS
        # ------------------------------------------

        total_transactions = len(df)

        fraud_count = int(
            (df["prediction"] == 1).sum()
        )

        legitimate_count = (
            total_transactions - fraud_count
        )

        fraud_rate = (
            fraud_count / total_transactions * 100
        )

        high_risk_count = int(
            (df["risk_level"] == "HIGH").sum()
        )

        # ------------------------------------------
        # KPI CARDS
        # ------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Transactions",
            f"{total_transactions:,}"
        )

        col2.metric(
            "Fraud Detected",
            f"{fraud_count:,}"
        )

        col3.metric(
            "Fraud Rate",
            f"{fraud_rate:.2f}%"
        )

        col4.metric(
            "High Risk Alerts",
            f"{high_risk_count:,}"
        )

        st.divider()

        # ------------------------------------------
        # TRANSACTION DISTRIBUTION
        # ------------------------------------------

        st.subheader("Transaction Distribution")

        col1, col2 = st.columns(2)

        with col1:

            st.write("Fraud vs Legitimate")

            transaction_counts = pd.DataFrame({
                "Transaction Type": [
                    "Legitimate",
                    "Fraud"
                ],
                "Count": [
                    legitimate_count,
                    fraud_count
                ]
            }).set_index("Transaction Type")

            st.bar_chart(
                transaction_counts,
                horizontal=True
            )

        with col2:

            st.write("Risk Level Distribution")

            risk_counts = (
                df["risk_level"]
                .value_counts()
                .reindex(
                    ["LOW", "MEDIUM", "HIGH"],
                    fill_value=0
                )
            )

            risk_chart = pd.DataFrame({
                "Transactions": risk_counts
            })

            st.bar_chart(risk_chart)

        st.divider()

        # ------------------------------------------
        # RECENT PREDICTION ACTIVITY
        # ------------------------------------------

        st.subheader("Recent Prediction Activity")

        recent_df = (
            df.head(10)
            .copy()
            .sort_values("prediction_id")
        )

        recent_df["Prediction ID"] = (
            recent_df["prediction_id"].astype(str)
        )

        probability_chart = recent_df[
            ["Prediction ID", "fraud_probability"]
        ].set_index("Prediction ID")

        st.line_chart(
            probability_chart,
            y="fraud_probability"
        )

        st.caption(
            "Fraud probabilities for the 10 most recent stored predictions."
        )

        st.divider()

        # ------------------------------------------
        # RECENT TRANSACTIONS
        # ------------------------------------------

        st.subheader("Recent Transactions")

        display_columns = [
            "prediction_id",
            "transaction_id",
            "transaction_amt",
            "fraud_probability",
            "prediction",
            "risk_level"
        ]

        st.dataframe(
            df[display_columns].head(10),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Overview metrics represent stored API predictions, "
            "not the complete IEEE-CIS dataset."
        )


# ==================================================
# 6. FRAUD SCORING
# ==================================================

elif page == "Fraud Scoring":

    st.title("Transaction Fraud Scoring")

    st.caption(
        "Analyze transactions using the trained LightGBM model."
    )

    st.divider()

    st.subheader("Transaction Details")

    with st.form("fraud_scoring_form"):

        col1, col2 = st.columns(2)

        with col1:

            transaction_amt = st.number_input(
                "Transaction Amount",
                min_value=0.01,
                value=150.0,
                step=10.0
            )

            dist2 = st.number_input(
                "Distance 2",
                value=38.0
            )

            d7 = st.number_input(
                "D7",
                value=0.0
            )

            d13 = st.number_input(
                "D13",
                value=0.0
            )

        with col2:

            product_cd = st.selectbox(
                "Product Category",
                ["W", "C", "H", "R", "S"]
            )

            card4 = st.selectbox(
                "Card Network",
                [
                    "Unknown",
                    "visa",
                    "mastercard",
                    "discover",
                    "american express"
                ]
            )

            card6 = st.selectbox(
                "Card Type",
                [
                    "Unknown",
                    "debit",
                    "credit",
                    "charge card",
                    "debit or credit"
                ]
            )

        submitted = st.form_submit_button(
            "Analyze Transaction",
            use_container_width=True
        )

    # ----------------------------------------------
    # PREDICTION RESULT
    # ----------------------------------------------

    if submitted:

        payload = {
            "TransactionAmt": transaction_amt,
            "dist2": dist2,
            "D7": d7,
            "D13": d13,
            "ProductCD": product_cd,
            "card4": card4,
            "card6": card6
        }

        with st.spinner("Analyzing transaction..."):

            result, error = submit_transaction(payload)

        if error:

            st.error(
                f"Prediction request failed: {error}"
            )

        elif result:

            probability = result["fraud_probability"]

            risk_level = result["risk_level"]

            prediction = result["prediction"]

            transaction_id = result["transaction_id"]

            st.divider()

            st.subheader("Prediction Results")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Fraud Probability",
                f"{probability:.2%}"
            )

            col2.metric(
                "Risk Level",
                risk_level
            )

            col3.metric(
                "Transaction ID",
                transaction_id
            )

            st.progress(
                min(max(probability, 0.0), 1.0)
            )

            if prediction == 1:

                st.error(
                    "⚠️ Transaction classified as potentially fraudulent."
                )

            else:

                st.success(
                    "✅ Transaction classified as legitimate."
                )

            st.caption(
                "Risk classification is based on the configured "
                "demo prediction threshold and risk bands."
            )


# ==================================================
# 7. TRANSACTION HISTORY
# ==================================================

elif page == "Transaction History":

    st.title("Transaction History")

    st.caption(
        "Previously analyzed transactions stored in PostgreSQL."
    )

    st.divider()

    predictions = get_api_data("/predictions")

    if predictions is None:

        st.error(
            "Unable to retrieve transaction history."
        )

    elif not predictions:

        st.info(
            "No transactions have been analyzed yet."
        )

    else:

        df = pd.DataFrame(predictions)

        st.metric(
            "Stored Predictions",
            f"{len(df):,}"
        )

        # Risk level filter

        risk_options = ["ALL", "LOW", "MEDIUM", "HIGH"]

        selected_risk = st.selectbox(
            "Filter by Risk Level",
            risk_options
        )

        if selected_risk != "ALL":

            df = df[
                df["risk_level"] == selected_risk
            ]

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Records are retrieved from the FastAPI backend."
        )


# ==================================================
# 8. MODEL MONITORING
# ==================================================

elif page == "Model Monitoring":

    st.title("Model Monitoring")

    st.caption(
        "Fraud model performance and prediction drift analysis."
    )

    st.divider()

    # ----------------------------------------------
    # MODEL PERFORMANCE
    # ----------------------------------------------

    st.subheader("Model Performance")

    performance = load_performance_report()

    if performance is not None and not performance.empty:

        metrics = performance.iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Precision",
            f"{metrics['precision']:.2%}"
        )

        col2.metric(
            "Recall",
            f"{metrics['recall']:.2%}"
        )

        col3.metric(
            "F1 Score",
            f"{metrics['f1_score']:.2%}"
        )

        col4.metric(
            "Average Precision",
            f"{metrics['average_precision']:.2%}"
        )

    else:

        st.warning(
            "Model performance report not found."
        )

    st.divider()

    # ----------------------------------------------
    # PREDICTION DRIFT
    # ----------------------------------------------

    st.subheader("Prediction Drift Analysis")

    drift = load_drift_report()

    if drift is not None:

        col1, col2 = st.columns(2)

        reference_mean = drift.loc[
            "mean", "Reference"
        ]

        current_mean = drift.loc[
            "mean", "Current"
        ]

        col1.metric(
            "Reference Mean Probability",
            f"{reference_mean:.2%}"
        )

        col2.metric(
            "Current Mean Probability",
            f"{current_mean:.2%}",
            delta=(
                f"{(current_mean - reference_mean) * 100:.2f} "
                "percentage points"
            )
        )

        st.write(
            "Prediction Distribution Comparison"
        )

        st.bar_chart(
            drift.loc[
                ["mean", "50%", "75%"]
            ].T
        )

        st.write(
            "Full Prediction Distribution Statistics"
        )

        st.dataframe(
            drift,
            use_container_width=True
        )

    else:

        st.warning(
            "Prediction drift report not found."
        )

    st.divider()

    # ----------------------------------------------
    # MONITORING SUMMARY
    # ----------------------------------------------

    st.subheader("Monitoring Summary")

    summary = load_monitoring_summary()

    if summary:

        st.markdown(summary)

    else:

        st.info(
            "Monitoring summary file not found."
        )

    st.divider()

    # ----------------------------------------------
    # EVIDENTLY REPORT LINKS
    # ----------------------------------------------

    st.subheader("Detailed Monitoring Reports")

    data_drift_path = (
        REPORTS_DIR / "data_drift_report.html"
    )

    prediction_drift_path = (
        REPORTS_DIR / "prediction_drift_report.html"
    )

    col1, col2 = st.columns(2)

    with col1:

        if data_drift_path.exists():

            with open(data_drift_path, "rb") as file:

                st.download_button(
                    "Download Data Drift Report",
                    data=file,
                    file_name="data_drift_report.html",
                    mime="text/html",
                    use_container_width=True
                )

        else:

            st.info("Data drift report unavailable.")

    with col2:

        if prediction_drift_path.exists():

            with open(prediction_drift_path, "rb") as file:

                st.download_button(
                    "Download Prediction Drift Report",
                    data=file,
                    file_name="prediction_drift_report.html",
                    mime="text/html",
                    use_container_width=True
                )

        else:

            st.info("Prediction drift report unavailable.")

    st.caption(
        "Performance and drift figures are historical offline "
        "evaluation results, not live monitoring measurements."
    )
