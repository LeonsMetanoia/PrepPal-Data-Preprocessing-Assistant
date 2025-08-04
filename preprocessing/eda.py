import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show_basic_eda(df):
    st.write("Jumlah Baris dan Kolom:", df.shape)
    st.write("Tipe Data per Kolom:")
    st.write(df.dtypes)
    
    st.write("Statistik Deskriptif:")
    st.write(df.describe())

    st.write("Jumlah Missing Values:")
    st.write(df.isnull().sum())

    st.write("Distribusi Beberapa Fitur:")
    numeric_cols = df.select_dtypes(include='number').columns[:3]
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)
