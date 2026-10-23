import numpy as np
import pandas as pd
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from scipy import stats
from scipy.stats import chi2_contingency

MODEL = "gemini-3.5-flash-lite"   # ~1s replies (gemini-3.8-flash was timing out)
T = "Churn"
NUM = ["Age", "Tenure", "Usage Frequency", "Support Calls", "Payment Delay",
       "Total Spend", "Last Interaction"]
CAT = ["Gender", "Subscription Type", "Contract Length"]

PROJECT_INFO = """
Project: Customer Churn Analysis (DataCore Solution). Goal: understand why customers
leave (churn) and predict who is likely to churn. Target = Churn (1 = churned, 0 = retained).
Numeric features: Age, Tenure, Usage Frequency, Support Calls, Payment Delay, Total Spend,
Last Interaction. Categorical: Gender, Subscription Type, Contract Length.
Files: Churn.ipynb (EDA, encoding, scaling, Logistic Regression), Statistical_Testing.ipynb
(hypothesis tests), streamlit_dashboard.py (dashboard), chatbot.py (this assistant).
Dashboard pages: Overview & EDA, Statistical Tests, Model Fitting, Predict Churn.
Tests: Welch's t-test (numeric vs churn), Chi-square + Cramer's V (categorical vs churn),
one-way ANOVA + Levene's test; p-values corrected with Benjamini-Hochberg (FDR).
Models: Logistic Regression and Random Forest (150 trees, max depth 12) in a scikit-learn
Pipeline (StandardScaler + OneHotEncoder), 80/20 stratified split, random_state=42.
Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are the assistant inside a customer churn dashboard. Answer briefly and "
               "simply. For project/metric questions use ONLY the facts below and quote the exact "
               "numbers. If a detail is not there, say so. You may also explain basic data "
               "science and statistics.\n{info}\n\nRecent chat:\n{history}"),
    ("human", "{question}"),
])


@st.cache_data(show_spinner=False)
def live_facts(df, _train):
    """Real numbers computed from the data + the dashboard's own models."""
    out = [f"\nLIVE RESULTS: {len(df):,} customers, {df.isnull().sum().sum()} missing values, "
           f"overall churn rate {df[T].mean():.1%}.",
           "Mean of each numeric feature (0 = retained, 1 = churned):",
           df.groupby(T)[NUM].mean().round(2).T.to_string(),
           "Welch t-test (churned vs retained):"]
    for c in NUM:
        t, p = stats.ttest_ind(df.loc[df[T] == 1, c], df.loc[df[T] == 0, c], equal_var=False)
        out.append(f"  {c}: t={t:.1f}, p={p:.3g}")
    out.append("Chi-square (categorical vs churn) and churn rate per category:")
    for c in CAT:
        ct = pd.crosstab(df[c], df[T])
        chi2, p, _, _ = chi2_contingency(ct)
        v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
        out.append(f"  {c}: p={p:.3g}, Cramer's V={v:.3f}; churn rate: "
                   + ", ".join(f"{k}={r:.1%}" for k, r in df.groupby(c)[T].mean().items()))
    for name in ["Logistic Regression", "Random Forest"]:
        pipe, m, cm = _train(name, 0.2, df)
        out.append(f"{name} (test 20%): " + ", ".join(f"{k}={v:.3f}" for k, v in m.items())
                   + f"; confusion matrix [[TN,FP],[FN,TP]]={cm.tolist()}")
    imp = pd.Series(pipe.named_steps["clf"].feature_importances_,
                    index=pipe.named_steps["pre"].get_feature_names_out()).nlargest(8)
    out.append("Random Forest top features: " + ", ".join(f"{k}={v:.3f}" for k, v in imp.items()))
    return "\n".join(out)


def render_chatbot(df, train_model):
    with st.sidebar.expander("\U0001F4AC Ask the chatbot"):
        with st.form("chat_form", clear_on_submit=True):
            question = st.text_input("Your question")
            ask = st.form_submit_button("Ask")
        hist = st.session_state.setdefault("chat_hist", [])
        if ask and question:
            with st.spinner("Loading project context..."):
                facts = live_facts(df, train_model)
            llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0.2, max_retries=1,
                                         timeout=30, google_api_key=st.secrets["GOOGLE_API_KEY"])
            chain = prompt | llm | StrOutputParser()
            history = "\n".join(f"Q: {q}\nA: {a}" for q, a in hist[-3:]) or "(none)"
            ans = st.write_stream(chain.stream(
                {"info": PROJECT_INFO + facts, "history": history, "question": question}))
            hist.append((question, ans))
        elif hist:
            st.info(hist[-1][1])