import streamlit as st
import pandas as pd
import requests  # ← ini tetap digunakan
from preprocessing import eda, cleaner

st.set_page_config(page_title="PrepPal", layout="wide")
st.title("🤖 PrepPal: Teman Preprocessing Berbasis AI")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    st.subheader("📈 Exploratory Data Analysis")
    eda.show_basic_eda(df)  # Fungsi ini akan menampilkan laporan Sweetviz

    st.subheader("🧹 Saran Pembersihan Data")
    cleaner.show_cleaning_suggestions(df)

    if st.button("Lakukan Preprocessing Otomatis"):
        df_cleaned = cleaner.auto_clean(df)
        st.success("Preprocessing selesai!")
        st.dataframe(df_cleaned.head())

        st.download_button("📥 Download Dataset Bersih", df_cleaned.to_csv(index=False), file_name="cleaned_data.csv")

# Chatbot AI Lokal dengan model Mistral dari Ollama
st.markdown("---")
st.subheader("🧠 Tanya AI tentang Preprocessing atau Data")

with st.form(key="chat_form"):
    user_prompt = st.text_area("Masukkan pertanyaan ke AI (misal: 'Apa itu missing value?')", height=100)
    submit = st.form_submit_button("Kirim ke AI")

if submit and user_prompt:
    with st.spinner("Sedang berpikir..."):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "mistral",  # pastikan nama model sama persis dengan yang kamu jalankan
                    "prompt": user_prompt,
                    "stream": False
                }
            )
            data = response.json()
            st.markdown("**🧠 Jawaban AI:**")
            st.write(data["response"])
        except Exception as e:
            st.error(f"Gagal menghubungi model Mistral dari Ollama. Error: {str(e)}")