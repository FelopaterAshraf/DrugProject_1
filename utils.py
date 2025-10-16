import os, json, joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder



def load_data(path="data/drug_consumption.csv"):
    df = pd.read_csv(path)
    return df


def explore_data(df):
    print("\n Dataset Info: \n")
    print(df.info())

    print("\n Missing Values: \n")
    print(df.isna().sum())

    print("\n Duplicate Values: \n")
    print(df.duplicated().sum())

    print("\n Statistical Summary: \n")
    print(df.describe())

def outlier_detection(df, factor=1.5):
    num_cols = df.select_dtypes(include="number").columns
    print("\n=== Outlier Detection (IQR method) ===")
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        count = ((df[col] < lower) | (df[col] > upper)).sum()
        print(f"{col}: {count} outliers")

def unique_values(df):
    print("\n Unique Values:")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")

def handle_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"✅ Removed {before - after} duplicate rows.")
    return df

def handle_outliers(df, factor=1.5):
    df = df.copy()
    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        df[col] = np.clip(df[col], lower, upper)
    print("Outliers clipped using IQR method.")
    return df


def handle_nulls(df):
    df = df.copy()
    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(exclude="number").columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("✅ Filled missing values (median for numeric, mode for categorical).")
    return df

def split_xy(df, target, drop_cols=None):
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')

    X = df.drop(columns=[target])
    y = df[target]

    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    return X, y

def split_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler, X.columns.tolist()

OUTPUT_DIR = "models"


def save_model(model, name="model"):
    import os, joblib
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(model, f"{OUTPUT_DIR}/{name}.pkl")