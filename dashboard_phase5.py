"""
Phase 5 - Streamlit dashboard (simple version).
Run with:  streamlit run dashboard_phase5.py
It only READS the files made in Phases 1-4. Nothing is retrained here.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from rag_chatbot import show_chatbot

BASE = Path(__file__).parent
st.set_page_config(page_title="Customer Churn Dashboard", page_icon="📊", layout="wide")


# ---------- load the saved files (cached so they load only once) ----------
@st.cache_data
def load_scores():
    return pd.read_csv(BASE / "churn_risk_scores.csv")       # data + churn probability + risk level


@st.cache_data
def load_json(name):
    return json.loads((BASE / name).read_text(encoding="utf-8"))


@st.cache_resource
def load_model():
    return joblib.load(BASE / "churn_model.pkl")             # the saved Random Forest


df = load_scores()
metrics = load_json("model_metrics.json")
features = load_json("model_features.json")
model = load_model()


def churn_rate_by(column):
    """Churn % for each group of a column."""
    return (df.groupby(column)["Churn"].mean() * 100).round(1)


# ---------- sidebar menu ----------
st.sidebar.title("Churn Project")
page = st.sidebar.radio("Go to", [
    "Overview",
    "EDA Findings",
    "Statistical Tests",
    "Model Results",
    "Predict a Customer",
    "High-Risk Customers",
    "Ask the Chatbot",
])


# ---------- 1. Overview ----------
if page == "Overview":
    st.title("Customer Churn Prediction Dashboard")
    st.write("Goal: find which customers are likely to leave, so the company can act early.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total customers", f"{len(df):,}")
    c2.metric("Churn rate", f"{df['Churn'].mean() * 100:.1f}%")
    c3.metric("High-risk customers", f"{(df['Risk_Level'] == 'High').sum():,}")
    c4.metric("Model accuracy", f"{metrics['test_scores']['Accuracy'] * 100:.1f}%")

    st.subheader("Customers by risk level")
    st.bar_chart(df["Risk_Level"].value_counts())

    st.subheader("Project phases")
    st.write("1. Understand data  →  2. Clean data  →  3. EDA and statistics  →  4. Modeling  →  5. Dashboard and chatbot")


# ---------- 2. EDA ----------
elif page == "EDA Findings":
    st.title("EDA Findings: who churns?")

    st.subheader("Churn % by category")
    col = st.selectbox("Choose a column", ["Gender", "Subscription Type", "Contract Length"])
    st.bar_chart(churn_rate_by(col))

    st.subheader("Churn % by number")
    col1, col2 = st.columns(2)

    delay = pd.cut(df["Payment Delay"], [-1, 5, 10, 15, 20, 100],
                   labels=["00-05", "06-10", "11-15", "16-20", "21+"])
    col1.write("Payment delay (days)")
    col1.bar_chart((df.groupby(delay, observed=True)["Churn"].mean() * 100).round(1))

    col2.write("Support calls")
    col2.bar_chart(churn_rate_by("Support Calls"))

    tenure = pd.cut(df["Tenure"], [-1, 12, 24, 36, 48, 100],
                    labels=["00-12", "13-24", "25-36", "37-48", "49+"])
    st.write("Tenure (months)")
    st.bar_chart((df.groupby(tenure, observed=True)["Churn"].mean() * 100).round(1))

    st.info("Churn jumps after about 15 days of payment delay and after 4 support calls. "
            "Longer tenure goes with MORE churn here, which is the opposite of what we expected.")


# ---------- 3. Statistics ----------
elif page == "Statistical Tests":
    st.title("Statistical Tests and Hypotheses")
    st.write("The data is not normal, so we used Mann-Whitney U for numbers and Chi-square for categories. "
             "We report effect size too, because with so much data every p-value looks significant.")

    # take the verdict table straight from the README so numbers always match
    readme = (BASE / "README.md").read_text(encoding="utf-8")
    verdicts = readme.split("### Hypothesis verdicts")[1].split("### 30-second")[0]
    st.markdown(verdicts)


# ---------- 4. Model results ----------
elif page == "Model Results":
    st.title("Model Results")
    st.write("Final model: **" + metrics["final_model"] + "**")

    scores = metrics["test_scores"]
    cols = st.columns(5)
    for box, name in zip(cols, scores):
        box.metric(name, scores[name])

    st.subheader("All models compared (test set)")
    st.dataframe(pd.DataFrame(metrics["all_models"]).set_index("Model"), use_container_width=True)

    st.subheader("Top features (importance)")
    st.bar_chart(pd.Series(metrics["top_features"]))

    st.subheader("Risk levels")
    st.table(pd.DataFrame(metrics["risk_bands"], index=["Churn probability"]).T)

    st.info("Scores are very high because this dataset follows sharp rules, not because of data leakage. "
            "5-fold cross-validation F1 = " + str(metrics["cv_mean_f1"]) + " (std " + str(metrics["cv_std_f1"]) + ").")


# ---------- 5. Predict ----------
elif page == "Predict a Customer":
    st.title("Predict a Customer")
    st.write("Enter customer details and the saved model gives the churn probability.")

    def number(label):
        """Slider that uses the min, max and median of that column."""
        return st.slider(label, int(df[label].min()), int(df[label].max()), int(df[label].median()))

    left, right = st.columns(2)
    with left:
        age = number("Age")
        tenure = number("Tenure")
        usage = number("Usage Frequency")
        calls = number("Support Calls")
        delay = number("Payment Delay")
    with right:
        spend = number("Total Spend")
        last = number("Last Interaction")
        gender = st.selectbox("Gender", ["Female", "Male"])
        sub = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
        contract = st.selectbox("Contract Length", ["Annual", "Quarterly", "Monthly"])

    if st.button("Predict"):
        # one row, in the same column order the model was trained on
        row = {
            "Age": age, "Tenure": tenure, "Usage Frequency": usage,
            "Support Calls": calls, "Payment Delay": delay,
            "Total Spend": spend, "Last Interaction": last,
            "Gender_Male": int(gender == "Male"),
            "Subscription Type_Premium": int(sub == "Premium"),
            "Subscription Type_Standard": int(sub == "Standard"),
            "Contract Length_Monthly": int(contract == "Monthly"),
            "Contract Length_Quarterly": int(contract == "Quarterly"),
        }
        X = pd.DataFrame([row])[features]
        prob = model.predict_proba(X)[0][1]

        if prob < 0.30:
            level = "Low"
        elif prob < 0.70:
            level = "Medium"
        else:
            level = "High"

        st.metric("Churn probability", f"{prob * 100:.1f}%")
        st.progress(float(prob))
        if level == "Low":
            st.success("Low risk")
        elif level == "Medium":
            st.warning("Medium risk")
        else:
            st.error("High risk: consider a retention offer")


# ---------- 6. High-risk customers ----------
elif page == "High-Risk Customers":
    st.title("Customers by Risk")

    levels = st.multiselect("Risk level", ["High", "Medium", "Low"], default=["High"])
    min_prob = st.slider("Minimum churn probability", 0.0, 1.0, 0.0)

    shown = df[df["Risk_Level"].isin(levels) & (df["Churn_Probability"] >= min_prob)]
    shown = shown.sort_values("Churn_Probability", ascending=False)

    st.write(f"{len(shown):,} customers match")
    st.dataframe(shown.head(500), use_container_width=True)
    st.caption("Showing the top 500 rows.")


# ---------- 7. Chatbot ----------
else:
    show_chatbot()
