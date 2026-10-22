"""Customer Churn Dashboard
Run with:  streamlit run streamlit_dashboard.py
Pages: Overview/EDA, Statistical Tests, Model Fitting, Prediction
"""
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.multitest import multipletests

DATA_PATH = "C:/Users/Hp/Downloads/customer_churn_dataset-testing-master.csv"
TARGET = "Churn"
NUMERIC = ["Age", "Tenure", "Usage Frequency", "Support Calls",
           "Payment Delay", "Total Spend", "Last Interaction"]
CATEGORICAL = ["Gender", "Subscription Type", "Contract Length"]

st.set_page_config(page_title="Customer Churn Dashboard", page_icon="📉", layout="wide")


# ----------------------------------------------------------------- data
@st.cache_data
def load_data(source):
    df = pd.read_csv(source)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)
    return df


st.sidebar.title("📉 Churn Dashboard")
uploaded = st.sidebar.file_uploader("Use a different CSV (optional)", type="csv")
if uploaded is not None:
    df = load_data(uploaded)
elif os.path.exists(DATA_PATH):
    df = load_data(DATA_PATH)
else:
    st.error(f"Dataset not found at {DATA_PATH}. Upload the CSV from the sidebar.")
    st.stop()

page = st.sidebar.radio("Go to", ["Overview & EDA", "Statistical Tests",
                                  "Model Fitting", "Predict Churn"])


# ----------------------------------------------------------------- model
@st.cache_resource
def train_model(model_name, test_size, _df):
    X = _df[NUMERIC + CATEGORICAL]
    y = _df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)

    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
    ])
    if model_name == "Logistic Regression":
        clf = LogisticRegression(max_iter=1000)
    else:
        clf = RandomForestClassifier(n_estimators=150, max_depth=12,
                                     n_jobs=-1, random_state=42)
    pipe = Pipeline([("pre", pre), ("clf", clf)]).fit(X_train, y_train)

    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    }
    return pipe, metrics, confusion_matrix(y_test, pred)


# ================================================================= PAGE 1
if page == "Overview & EDA":
    st.title("Overview & Exploratory Data Analysis")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Features", len(NUMERIC) + len(CATEGORICAL))
    c3.metric("Churn rate", f"{df[TARGET].mean() * 100:.1f}%")
    c4.metric("Missing values", int(df.isnull().sum().sum()))

    with st.expander("Preview data & summary statistics"):
        st.dataframe(df.head(20), use_container_width=True)
        st.dataframe(df[NUMERIC].describe().T.round(2), use_container_width=True)

    st.subheader("Churn distribution")
    counts = df[TARGET].map({0: "Retained", 1: "Churned"}).value_counts().reset_index()
    counts.columns = ["Status", "Customers"]
    st.plotly_chart(px.pie(counts, names="Status", values="Customers", hole=0.4),
                    use_container_width=True)

    st.subheader("Numeric features")
    col = st.selectbox("Choose a numeric feature", NUMERIC)
    l, r = st.columns(2)
    l.plotly_chart(px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}"),
                   use_container_width=True)
    plot_df = df.assign(Status=df[TARGET].map({0: "Retained", 1: "Churned"}))
    r.plotly_chart(px.box(plot_df, x="Status", y=col, color="Status",
                          title=f"{col}: Churned vs Retained"),
                   use_container_width=True)

    st.subheader("Categorical features")
    cat = st.selectbox("Choose a categorical feature", CATEGORICAL)
    rate = df.groupby(cat)[TARGET].mean().mul(100).round(2).reset_index()
    l, r = st.columns(2)
    l.plotly_chart(px.histogram(df, x=cat, title=f"{cat} counts"),
                   use_container_width=True)
    r.plotly_chart(px.bar(rate, x=cat, y=TARGET, title=f"Churn rate (%) by {cat}"),
                   use_container_width=True)

    st.subheader("Correlation heatmap")
    corr = df[NUMERIC + [TARGET]].corr()
    st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1), use_container_width=True)


# ================================================================= PAGE 2
elif page == "Statistical Tests":
    st.title("Statistical Testing")
    st.caption("Same tests as Statistical_Testing.ipynb; p-values corrected with Benjamini-Hochberg (FDR).")

    t1, t2, t3 = st.tabs(["Welch's t-test", "Chi-square", "One-way ANOVA"])

    with t1:
        st.write("Do numeric features differ between churned and retained customers?")
        rows = []
        for c in NUMERIC:
            ret = df.loc[df[TARGET] == 0, c].dropna()
            chu = df.loc[df[TARGET] == 1, c].dropna()
            t, p = stats.ttest_ind(chu, ret, equal_var=False)
            rows.append({"Feature": c, "Retained Mean": ret.mean(),
                         "Churned Mean": chu.mean(),
                         "Difference": chu.mean() - ret.mean(),
                         "T Statistic": t, "P Value": p})
        tt = pd.DataFrame(rows)
        tt["Adjusted P Value"] = multipletests(tt["P Value"], method="fdr_bh")[1]
        tt["Significant"] = np.where(tt["Adjusted P Value"] < 0.05, "Yes", "No")
        st.dataframe(tt.round(4), use_container_width=True, hide_index=True)
        fig = px.bar(tt.sort_values("T Statistic"), x="T Statistic", y="Feature",
                     orientation="h", color="Significant",
                     title="T statistic by feature (churned - retained)")
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.write("Is each categorical feature associated with churn?")
        rows = []
        tables = {}
        for c in CATEGORICAL:
            ct = pd.crosstab(df[c], df[TARGET])
            tables[c] = ct
            chi2, p, dof, exp = chi2_contingency(ct)
            n, k = ct.to_numpy().sum(), min(ct.shape)
            rows.append({"Feature": c, "Chi-Square": chi2, "DoF": dof,
                         "P Value": p,
                         "Cramer's V": np.sqrt(chi2 / (n * (k - 1))) if k > 1 else np.nan})
        ch = pd.DataFrame(rows)
        ch["Adjusted P Value"] = multipletests(ch["P Value"], method="fdr_bh")[1]
        ch["Significant"] = np.where(ch["Adjusted P Value"] < 0.05, "Yes", "No")
        st.dataframe(ch.round(4), use_container_width=True, hide_index=True)
        pick = st.selectbox("Show contingency table for", CATEGORICAL)
        st.dataframe(tables[pick].rename(columns={0: "Retained", 1: "Churned"}),
                     use_container_width=True)

    with t3:
        c1, c2 = st.columns(2)
        value = c1.selectbox("Numeric feature", NUMERIC, index=1)
        group = c2.selectbox("Grouping feature", CATEGORICAL, index=1)
        data = df[[value, group]].dropna()
        groups = [g[value].values for _, g in data.groupby(group)]
        st.dataframe(data.groupby(group)[value].agg(["count", "mean", "std"]).round(3),
                     use_container_width=True)
        lev_stat, lev_p = stats.levene(*groups)
        f_stat, p = stats.f_oneway(*groups)
        m1, m2, m3 = st.columns(3)
        m1.metric("F statistic", f"{f_stat:.4f}")
        m2.metric("ANOVA p-value", f"{p:.4g}")
        m3.metric("Levene p-value", f"{lev_p:.4g}")
        if p < 0.05:
            st.success("At least one group mean differs significantly.")
        else:
            st.info("Insufficient evidence that group means differ.")
        if lev_p < 0.05:
            st.warning("Group variances differ (Levene p < 0.05); consider Welch's ANOVA.")
        st.plotly_chart(px.box(data, x=group, y=value, color=group),
                        use_container_width=True)


# ================================================================= PAGE 3
elif page == "Model Fitting":
    st.title("Model Fitting")
    c1, c2 = st.columns(2)
    model_name = c1.selectbox("Model", ["Random Forest", "Logistic Regression"])
    test_size = c2.slider("Test set size", 0.1, 0.4, 0.2, 0.05)

    with st.spinner("Training..."):
        pipe, metrics, cm = train_model(model_name, test_size, df)

    cols = st.columns(len(metrics))
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, f"{v:.3f}")

    l, r = st.columns(2)
    cm_df = pd.DataFrame(cm, index=["Actual Retained", "Actual Churned"],
                         columns=["Pred Retained", "Pred Churned"])
    l.subheader("Confusion matrix")
    l.plotly_chart(px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues"),
                   use_container_width=True)

    r.subheader("Feature importance")
    names = pipe.named_steps["pre"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    imp = clf.feature_importances_ if hasattr(clf, "feature_importances_") else np.abs(clf.coef_[0])
    imp_df = pd.DataFrame({"Feature": names, "Importance": imp}).sort_values("Importance")
    r.plotly_chart(px.bar(imp_df, x="Importance", y="Feature", orientation="h"),
                   use_container_width=True)


# ================================================================= PAGE 4
else:
    st.title("Predict Customer Churn")
    model_name = st.sidebar.selectbox("Model", ["Random Forest", "Logistic Regression"])
    pipe, metrics, _ = train_model(model_name, 0.2, df)

    st.write("Enter customer details:")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", 18, 100, 35)
    tenure = c1.number_input("Tenure (months)", 0, 120, 24)
    usage = c1.number_input("Usage Frequency", 0, 60, 15)
    calls = c2.number_input("Support Calls", 0, 20, 3)
    delay = c2.number_input("Payment Delay (days)", 0, 60, 10)
    spend = c2.number_input("Total Spend", 0.0, 5000.0, 600.0, step=10.0)
    last = c3.number_input("Last Interaction (days ago)", 0, 60, 15)
    gender = c3.selectbox("Gender", sorted(df["Gender"].unique()))
    sub = c3.selectbox("Subscription Type", sorted(df["Subscription Type"].unique()))
    contract = c3.selectbox("Contract Length", sorted(df["Contract Length"].unique()))

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([{
            "Age": age, "Tenure": tenure, "Usage Frequency": usage,
            "Support Calls": calls, "Payment Delay": delay, "Total Spend": spend,
            "Last Interaction": last, "Gender": gender,
            "Subscription Type": sub, "Contract Length": contract}])
        prob = pipe.predict_proba(row)[0, 1]
        if prob >= 0.5:
            st.error(f"Likely to CHURN - probability {prob:.1%}")
        else:
            st.success(f"Likely to STAY - churn probability {prob:.1%}")
        st.progress(float(prob))
        st.caption(f"Model: {model_name} (test accuracy {metrics['Accuracy']:.3f})")
