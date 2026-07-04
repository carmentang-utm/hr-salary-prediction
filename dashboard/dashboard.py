import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(page_title="HR Salary Dashboard", layout="wide")
st.title("HR Salary Analytics & Prediction Dashboard")
st.write(
    "This dashboard supports HR managers, department heads, and the finance "
    "team with salary benchmarking, workforce analysis, and data-driven "
    "salary estimation."
)

# ---------- Load data and model (deployment-safe absolute paths) ----------
BASE_DIR = Path(__file__).resolve().parent.parent

@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "data" / "cleaned_employee_dataset.csv")

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "models" / "salary_model.pkl")

df = load_data()
model = load_model()

# ---------- Sidebar: interactive filters ----------
st.sidebar.header("Filter Options")

# Interactive feature 1: Department dropdown
department = st.sidebar.selectbox(
    "Department", ["All"] + sorted(df["Department"].unique().tolist()))
if department != "All":
    df = df[df["Department"] == department]

# Interactive feature 2: Region dropdown
region = st.sidebar.selectbox(
    "Region", ["All"] + sorted(df["Region"].unique().tolist()))
if region != "All":
    df = df[df["Region"] == region]

# Interactive feature 3: Salary range slider
if not df.empty:
    min_sal, max_sal = int(df["Salary"].min()), int(df["Salary"].max())
    if min_sal < max_sal:
        sal_range = st.sidebar.slider(
            "Salary Range", min_sal, max_sal, value=(min_sal, max_sal))
        df = df[df["Salary"].between(sal_range[0], sal_range[1])]

# ---------- Summary metrics ----------
st.subheader("Workforce Summary")
c1, c2, c3 = st.columns(3)
c1.metric("Employees (filtered)", len(df))
c2.metric("Average Salary", f"{df['Salary'].mean():,.0f}" if not df.empty else "N/A")
c3.metric("Median Salary", f"{df['Salary'].median():,.0f}" if not df.empty else "N/A")

if df.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# ---------- Visualization 1: Salary distribution ----------
st.subheader("1. Salary Distribution")
fig1, ax1 = plt.subplots(figsize=(8, 3.5))
sns.histplot(df["Salary"], bins=30, kde=True, ax=ax1)
ax1.set_xlabel("Salary")
st.pyplot(fig1)

# ---------- Visualization 2: Average salary by department ----------
st.subheader("2. Average Salary by Department")
dept_salary = df.groupby("Department")["Salary"].mean().sort_values()
st.bar_chart(dept_salary)

# ---------- Visualization 3: Salary by performance score ----------
st.subheader("3. Salary by Performance Score")
fig3, ax3 = plt.subplots(figsize=(8, 3.5))
order = ["Poor", "Average", "Good", "Excellent"]
sns.boxplot(x="Performance_Score", y="Salary", data=df, order=order, ax=ax3)
st.pyplot(fig3)

# ---------- Predictive output: salary estimation ----------
st.subheader("4. Salary Prediction (Machine Learning Model)")
st.write("Estimate the expected salary for an employee profile using the "
         "Gradient Boosting model developed in Sprint 3.")

p1, p2, p3 = st.columns(3)
with p1:
    in_age = st.slider("Age", 18, 65, 30)
    in_dept = st.selectbox("Department (profile)",
                           sorted(load_data()["Department"].unique()))
with p2:
    in_region = st.selectbox("Region (profile)",
                             sorted(load_data()["Region"].unique()))
    in_status = st.selectbox("Status", ["Active", "Pending", "Inactive"])
with p3:
    in_perf = st.selectbox("Performance", order, index=2)
    in_remote = st.checkbox("Remote Work", value=True)
    in_tenure = st.slider("Tenure (years)", 0.0, 10.0, 3.0, step=0.5)

if st.button("Predict Salary"):
    profile = pd.DataFrame({
        "Age": [in_age],
        "Department": [in_dept],
        "Region": [in_region],
        "Status": [in_status],
        "Performance_Level": [order.index(in_perf) + 1],
        "Remote_Work": [int(in_remote)],
        "Tenure_Years": [in_tenure],
    })
    predicted = model.predict(profile)[0]
    st.metric("Predicted Salary", f"{predicted:,.2f}")
    st.caption("Note: the model has a mean absolute error of approximately "
               "±17,000, reflecting the limited predictive signal in the "
               "current dataset. Predictions should be used as indicative "
               "benchmarks, not exact figures.")

# ---------- Filtered data table ----------
st.subheader("Filtered Employee Records")
st.dataframe(df[["Employee_ID", "Department", "Region", "Status",
                 "Performance_Score", "Remote_Work", "Tenure_Years", "Salary"]])
