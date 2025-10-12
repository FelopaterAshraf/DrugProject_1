import os, json, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder



def load_data(path="data/drug_consumption.csv"):
    df = pd.read_csv(path)
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