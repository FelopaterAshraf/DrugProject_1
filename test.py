import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from utils import load_data, split_xy, split_scale


MODEL_NAME = "Cannabis_randomforest"   
TARGET = "Cannabis"


model = joblib.load(f"models/{MODEL_NAME}.pkl")

df = load_data()
df[TARGET] = df[TARGET].apply(lambda x: 0 if x == "CL0" else 1)
X, y = split_xy(df, TARGET, drop_cols=["ID", "Unnamed: 0"])
X = X.select_dtypes(include="number")

X_train, X_test, y_train, y_test, scaler, cols = split_scale(X, y)
X_test = scaler.transform(X_test)


y_pred = model.predict(X_test)

print(f"Results for {MODEL_NAME}:\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))
