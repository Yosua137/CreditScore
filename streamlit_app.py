import json
import os

import boto3
import pandas as pd
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "credit-score-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-1")

# Copied from CreditPreprocessor.ALL_LOAN_TYPES (pipeline/preprocessing.py) —
# duplicated on purpose so this lightweight host never needs to import sklearn.
ALL_LOAN_TYPES = [
    "Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan",
    "Home Equity Loan", "Mortgage Loan", "No Loan", "Not Specified",
    "Payday Loan", "Personal Loan", "Student Loan",
]

BADGE = {"Good": "🟢", "Standard": "🟡", "Poor": "🔴"}


st.set_page_config(
    page_title="Credit Score Predictor",
    page_icon="💳",
    layout="wide",
)


@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)


def invoke_endpoint(instance: dict) -> dict:
    runtime = get_runtime_client()
    payload = {"instances": [instance]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


# Header
st.title("💳 Credit Score Prediction")
st.caption(f"Endpoint: `{ENDPOINT_NAME}` · Region: `{REGION}`")

# Input Form
st.subheader("🔎 Masukkan Data Nasabah")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📅 Profil Umum**")
    Month = st.selectbox(
        "Month",
        ["January", "February", "March", "April", "May",
         "June", "July", "August", "September", "October",
         "November", "December"],
        index=0,
    )
    Age                     = st.number_input("Age",                        18,  90,  32, 1)
    Occupation              = st.selectbox("Occupation", [
        "Accountant","Architect","Developer","Doctor","Engineer",
        "Entrepreneur","Journalist","Lawyer","Manager","Mechanic",
        "Media_Manager","Musician","Scientist","Teacher","Unknown","Writer",
    ])
    Annual_Income           = st.number_input("Annual Income (USD)",        0.0, 300_000.0, 37_443.0, 500.0)
    Monthly_Inhand_Salary   = st.number_input("Monthly Inhand Salary (USD)", 0.0, 20_000.0, 3_080.0,  100.0)
    Monthly_Balance         = st.number_input("Monthly Balance (USD)",      -5_000.0, 50_000.0, 2_000.0, 100.0)

with col2:
    st.markdown("**🏦 Akun & Kredit**")
    Num_Bank_Accounts        = st.number_input("Num Bank Accounts",          0, 20, 5, 1)
    Num_Credit_Card          = st.number_input("Num Credit Cards",           0, 20, 5, 1)
    Interest_Rate            = st.number_input("Interest Rate (%)",          0, 34, 13, 1)
    Num_of_Loan              = st.number_input("Num of Loans",               0, 20,  4, 1)
    Outstanding_Debt         = st.number_input("Outstanding Debt (USD)",     0.0, 5_000.0, 1_426.0, 50.0)
    Credit_Utilization_Ratio = st.number_input("Credit Utilization Ratio (%)", 0.0, 100.0, 32.0, 0.5)
    Credit_Mix               = st.selectbox("Credit Mix",
                                            ["Bad", "Standard", "Good", "Unknown"])
    Changed_Credit_Limit     = st.number_input("Changed Credit Limit",      -20.0, 30.0, 5.0, 0.5)

with col3:
    st.markdown("**⏱️ Riwayat & Pembayaran**")
    st.markdown("**Credit History Age**")
    cha_years  = st.number_input("  → Years",  0, 40, 10, 1, key="cha_y")
    cha_months = st.number_input("  → Months", 0, 11,  0, 1, key="cha_m")
    Credit_History_Months   = cha_years * 12 + cha_months

    Delay_from_due_date      = st.number_input("Avg Delay from Due Date (days)", 0, 60, 20, 1)
    Num_of_Delayed_Payment   = st.number_input("Num of Delayed Payments",        0, 30, 14, 1)
    Num_Credit_Inquiries     = st.number_input("Num Credit Inquiries",           0, 20,  5, 1)
    Total_EMI_per_month      = st.number_input("Total EMI per Month (USD)",      0.0, 5_000.0, 100.0, 10.0)
    Amount_invested_monthly  = st.number_input("Amount Invested Monthly (USD)",  0.0, 2_000.0, 150.0, 10.0)
    Payment_of_Min_Amount    = st.selectbox("Payment of Min Amount", ["No", "Yes", "NM"])
    Payment_Behaviour        = st.selectbox("Payment Behaviour", [
        "Low_spent_Small_value_payments",
        "Low_spent_Medium_value_payments",
        "Low_spent_Large_value_payments",
        "High_spent_Small_value_payments",
        "High_spent_Medium_value_payments",
        "High_spent_Large_value_payments",
        "Unknown",
    ])

st.markdown("**🏷️ Type of Loan** *(pilih semua yang berlaku)*")
selected_loans = st.multiselect(
    "Loan Types",
    options=ALL_LOAN_TYPES,
    default=["Personal Loan"],
    label_visibility="collapsed",
)
if not selected_loans:
    selected_loans = ["No Loan"]
Clean_Loans = list(set(selected_loans))  # JSON has no tuple type -> plain list

st.divider()


# Predict Button
if st.button("🔍 Predict Credit Score", type="primary", use_container_width=True):
    instance = {
        "Month":                    Month,
        "Age":                      float(Age),
        "Occupation":               Occupation,
        "Annual_Income":            float(Annual_Income),
        "Monthly_Inhand_Salary":    float(Monthly_Inhand_Salary),
        "Num_Bank_Accounts":        float(Num_Bank_Accounts),
        "Num_Credit_Card":          float(Num_Credit_Card),
        "Interest_Rate":            float(Interest_Rate),
        "Num_of_Loan":              float(Num_of_Loan),
        "Delay_from_due_date":      float(Delay_from_due_date),
        "Num_of_Delayed_Payment":   float(Num_of_Delayed_Payment),
        "Changed_Credit_Limit":     float(Changed_Credit_Limit),
        "Num_Credit_Inquiries":     float(Num_Credit_Inquiries),
        "Credit_Mix":               Credit_Mix,
        "Outstanding_Debt":         float(Outstanding_Debt),
        "Credit_Utilization_Ratio": float(Credit_Utilization_Ratio),
        "Payment_of_Min_Amount":    Payment_of_Min_Amount,
        "Total_EMI_per_month":      float(Total_EMI_per_month),
        "Amount_invested_monthly":  float(Amount_invested_monthly),
        "Payment_Behaviour":        Payment_Behaviour,
        "Monthly_Balance":          float(Monthly_Balance),
        "Credit_History_Months":    float(Credit_History_Months),
        "Clean_Loans":              Clean_Loans,
    }

    try:
        result = invoke_endpoint(instance)
    except NoCredentialsError:
        st.error(
            "❌ No AWS credentials found. If running on EC2, attach LabInstanceProfile. "
            "If running locally, configure ~/.aws/credentials."
        )
    except ClientError as e:
        st.error(f"❌ AWS error: {e.response['Error'].get('Message', str(e))}")
    else:
        pred_label = result["labels"][0]
        pred_proba = result["probabilities"][0]
        target_names = result["classes"]

        proba_df = pd.DataFrame({
            "Credit Score": target_names,
            "Probability (%)": [round(p * 100, 2) for p in pred_proba],
        }).set_index("Credit Score")

        badge = BADGE.get(pred_label, "⚪")
        st.markdown(f"## {badge} Prediksi: **{pred_label}**")

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.metric("Predicted Class",  pred_label)
            st.metric("Confidence",       f"{max(pred_proba) * 100:.1f}%")
            st.metric("Credit Hist. Age", f"{Credit_History_Months} months")

        with col_b:
            st.markdown("**Probabilitas per kelas:**")
            st.dataframe(
                proba_df.style.bar(subset=["Probability (%)"], color="#4CAF50"),
                use_container_width=True,
            )
