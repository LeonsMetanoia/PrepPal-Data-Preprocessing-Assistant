import streamlit as st
import pandas as pd
import io
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#  Multicollinearity Check (VIF)  Variance Inflation Factor
from statsmodels.stats.outliers_influence import variance_inflation_factor


# LangChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

# File lokal
from preprocessing import eda, cleaner

# Setup halaman
st.set_page_config(page_title="PrepPal", layout="wide")
st.title("🤖 PrepPal: Teman Preprocessing Berbasis AI (LLM(Mistral))")
st.markdown("> **Kenapa AI Nggak Pernah CurHat?**  \n> Soalnya dia cuma bisa *Deep Learning*, bukan *Deep Feeling*. 😂 WKWKWK")
# Upload file
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

df = None
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

        st.download_button("📥 Download Dataset Bersih", df_cleaned.to_csv(index=False), file_name="cleaned_data.csv")


   # =====================
    # 📌 Outlier Detection
    # =====================
    st.subheader("📊 Outlier Detection (Z-score Method)")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    zscore_threshold = st.slider("Z-score Threshold", 0.0, 5.0, 3.0)

    if st.button("Detect Outliers"):
        df_outliers = df[numeric_cols].copy()
        z_scores = (df_outliers - df_outliers.mean()) / df_outliers.std()
        outlier_mask = (np.abs(z_scores) > zscore_threshold).any(axis=1)
        st.write(f"Jumlah outlier terdeteksi: {outlier_mask.sum()} baris")
        st.dataframe(df[outlier_mask])

    # ===============================
    # 🚨 Outlier Detection & Handling
    # ===============================
    st.subheader("🚨 Outlier Detection & Handling")

    selected_col = st.selectbox("Pilih kolom numerik untuk deteksi outlier:", numeric_cols)

    method = st.selectbox("Pilih metode penanganan outlier:", [
        "Tampilkan Outlier", "Hapus Outlier", "Set Outlier sebagai NaN", "Winsorizing"
    ])

    if selected_col:
        Q1 = df[selected_col].quantile(0.25)
        Q3 = df[selected_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]

        if method == "Tampilkan Outlier":
            st.write(f"Jumlah outlier pada kolom **{selected_col}**: {outliers.shape[0]}")
            st.dataframe(outliers)

        elif method == "Hapus Outlier":
            df = df[~((df[selected_col] < lower_bound) | (df[selected_col] > upper_bound))]
            st.success("Outlier berhasil dihapus.")
            st.dataframe(df)

        elif method == "Set Outlier sebagai NaN":
            df[selected_col] = df[selected_col].apply(
                lambda x: np.nan if x < lower_bound or x > upper_bound else x
            )
            st.success("Outlier telah di-set sebagai NaN.")
            st.dataframe(df)

        elif method == "Winsorizing":
            df[selected_col] = np.where(df[selected_col] < lower_bound, lower_bound, df[selected_col])
            df[selected_col] = np.where(df[selected_col] > upper_bound, upper_bound, df[selected_col])
            st.success("Outlier telah ditangani dengan Winsorizing.")
            st.dataframe(df)


    # ============================
    # 🔍 Multicollinearity Check (VIF) and Correlation Matrix and it's HeatMap :
    # ============================
st.subheader("🔍 Multicollinearity Check (VIF) + Correlation Matrix")

if st.button("Check Multicollinearity"):
    df_vif = df[numeric_cols].dropna()

    if df_vif.shape[1] < 2:
        st.warning("Minimal 2 kolom numerik diperlukan untuk menghitung VIF.")
    else:
        # VIF Calculation
        vif_data = pd.DataFrame()
        vif_data["feature"] = df_vif.columns
        vif_data["VIF"] = [variance_inflation_factor(df_vif.values, i) for i in range(df_vif.shape[1])]
        st.write("### Variance Inflation Factor (VIF)")
        st.dataframe(vif_data)

        # Correlation Matrix
        st.write("### Correlation Matrix")
        corr_matrix = df_vif.corr()
        st.dataframe(corr_matrix)

        # Heatmap
        st.write("### Heatmap Visualisasi Korelasi")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        st.pyplot(fig)


# ------------------------------
# 🔗 Integrasi LangChain Chatbot
# ------------------------------
st.markdown("---")
st.subheader("🙋‍♂️ Tanya AI tentang Preprocessing atau Statistik Dataset yang kita uda upload!!")

with st.form(key="chat_form"):
    user_prompt = st.text_area("Masukkan pertanyaan ke AI (misal: 'Kolom mana yang memiliki missing value terbanyak?', 'Berapa nilai maksimum kolom umur?')", height=100)
    submit = st.form_submit_button("Kirim ke AI")

if submit and user_prompt:
    with st.spinner("Sedang berpikir..."):
        try:
            if df is not None:
                # Data preview (10 baris awal)
                preview_buffer = io.StringIO()
                df.head(10).to_string(buf=preview_buffer)
                preview = preview_buffer.getvalue()

                # Statistik ringkasan (.describe)
                stats_buffer = io.StringIO()
                df.describe(include='all').to_string(buf=stats_buffer)
                stats = stats_buffer.getvalue()

                # Missing values
                missing_buffer = io.StringIO()
                df.isnull().sum().to_string(buf=missing_buffer)
                missing = missing_buffer.getvalue()
            else:
                preview = "Tidak ada data."
                stats = ""
                missing = ""

            # Template Prompt LangChain
            template = """
Kamu adalah asisten data pintar. Jawablah pertanyaan pengguna berdasarkan informasi dataset yang diberikan.

Contoh potongan data (10 baris pertama):
{preview}

Statistik ringkasan dataset:
{stats}

Jumlah missing value per kolom:
{missing}

Pertanyaan:
{question}

Jawaban:
"""
            prompt = PromptTemplate.from_template(template)

            # Setup LLM dari Ollama
            llm = Ollama(model="mistral")

            chain = prompt | llm | StrOutputParser()

            result = chain.invoke({
                "preview": preview,
                "stats": stats,
                "missing": missing,
                "question": user_prompt
            })

            st.markdown("**🧠 Jawaban AI:**")
            st.write(result)

        except Exception as e:
            st.error(f"Gagal menjalankan LLM dengan LangChain. Error: {str(e)}")
