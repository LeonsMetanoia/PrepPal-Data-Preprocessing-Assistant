import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import sweetviz as sv
import streamlit.components.v1 as components
import tempfile
    
def show_basic_eda(df):
    st.write("Jumlah Baris dan Kolom:", df.shape)
    st.write("Tipe Data per Kolom:")
    st.write(df.dtypes)

    st.write("Statistik Deskriptif:")
    st.write(df.describe())

    st.write("Jumlah Missing Values:")
    st.write(df.isnull().sum())

    numeric_cols = df.select_dtypes(include='number').columns[:3]
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

def show_sweetviz_report(df):
    report = sv.analyze(df)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
        report.show_html(filepath=f.name, open_browser=False)
        components.html(open(f.name, 'r').read(), height=1000, scrolling=True)
