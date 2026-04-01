import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# ✅ CONFIG
st.set_page_config(page_title="Creditcard Fraud Detection", layout="wide")

# Load model
model = pickle.load(open("model.pkl", "rb"))

# ---------------- LOGIN ----------------
if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    st.title("Login Page")

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username == "admin" and password == "1234":
                st.session_state.login = True
                st.rerun()
            else:
                st.error("Invalid credentials")

    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigations")

page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Analytics", "About"]
)

# ---------------- DASHBOARD ----------------
if page == "Dashboard":

    st.title("Creditcard Fraud Detection Dashboard")

    file = st.file_uploader("Upload CSV File Here", type=["csv"])

    if file:
        df = pd.read_csv(file)

        st.dataframe(df, use_container_width=True)

        if st.button("Run Detection"):

            preds = model.predict(df)
            df["Prediction"] = preds

            # save for analytics
            st.session_state["data"] = df

            fraud = int(preds.sum())
            total = len(df)
            normal = total - fraud

            st.subheader("Summary")

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Transactions", total)
            col2.metric("Fraud", fraud)
            col3.metric("Normal ✅", normal)

            st.markdown("---")

            fig = px.bar(
                x=["Normal", "Fraud"],
                y=[normal, fraud],
                title="Fraud Analysis"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            st.subheader("Results")
            st.dataframe(df, use_container_width=True)

            if fraud > 0:
                st.error(f"{fraud} Fraud Transactions Detected")
            else:
                st.success("✅ All Transactions are Safe")

# ---------------- ANALYTICS ----------------
elif page == "Analytics":

    st.title("Analytics Panel")

    df = st.session_state.get("data", None)

    if df is None:
        st.warning("⚠️ Please run detection in Dashboard first")
    else:
        fraud = int(df["Prediction"].sum())
        total = len(df)
        normal = total - fraud

        st.subheader("Insights")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total", total)
        col2.metric("Fraud", fraud)
        col3.metric("Fraud Rate", f"{(fraud/total)*100:.2f}%")

        st.markdown("---")

        fig1 = px.pie(
            names=["Normal", "Fraud"],
            values=[normal, fraud],
            title="Fraud Distribution"
        )
        st.plotly_chart(fig1, use_container_width=True)

        if "Amount" in df.columns:
            st.markdown("### Amount Distribution")
            fig2 = px.histogram(df, x="Amount")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")

        st.subheader("Fraud Transactions")
        st.dataframe(df[df["Prediction"] == 1], use_container_width=True)

# ---------------- ABOUT ----------------
elif page == "About":

    st.title("About")

    st.write("""
    This is a Fraud Detection System using Machine Learning.

    Features:
    - Login System
    - CSV Upload
    - Real-time Prediction
    - Analytics Dashboard
    - Responsive UI
    """)
