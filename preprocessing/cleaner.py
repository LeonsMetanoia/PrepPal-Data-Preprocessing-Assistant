import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def show_cleaning_suggestions(df):
    if df.isnull().values.any():
        st.warning("Terdapat nilai kosong (missing values). Disarankan untuk imputasi atau drop.")
    
    dupes = df.duplicated().sum()
    if dupes > 0:
        st.warning(f"Terdapat {dupes} baris duplikat.")

    obj_cols = df.select_dtypes(include='object').columns
    if len(obj_cols) > 0:
        st.info(f"Terdapat kolom kategori: {list(obj_cols)}. Disarankan untuk encoding.")

def auto_clean(df):
    df = df.drop_duplicates()

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    # Encode categorical
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Scale numeric
    numeric_cols = df.select_dtypes(include='number').columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df
