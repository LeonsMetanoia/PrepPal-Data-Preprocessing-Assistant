import streamlit as st
import pandas as pd
from preprocessing import eda, cleaner

st.set_page_config(page_title="PrepPal", layout="wide")
st.title("🤖 PrepPal: Teman Preprocessing Berbasis AI")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    st.subheader("📈 Exploratory Data Analysis")
    eda.show_basic_eda(df)

    st.subheader("🧹 Saran Pembersihan Data")
    cleaner.show_cleaning_suggestions(df)

    if st.button("Lakukan Preprocessing Otomatis"):
        df_cleaned = cleaner.auto_clean(df)
        st.success("Preprocessing selesai!")
        st.dataframe(df_cleaned.head())

        # Optional download
        st.download_button("📥 Download Dataset Bersih", df_cleaned.to_csv(index=False), file_name="cleaned_data.csv")
