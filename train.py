import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report,confusion_matrix
from utils import load_data, split_xy, split_scale, save_model

TARGET = "Cannabis"


df = load_data()

df[TARGET] = df[TARGET].apply(lambda x: 0 if x == "CL0" else 1)


X, y = split_xy(df, TARGET, drop_cols=["ID", "Unnamed: 0"])
X = X.select_dtypes(include="number")
X_train, X_test, y_train, y_test, scaler, cols = split_scale(X, y)


log_model = LogisticRegression(max_iter=2000, random_state=42)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

print("Logistic Regression Results:\n")
print("Accuracy:", accuracy_score(y_test, log_pred))
print("F1 Score:", f1_score(y_test, log_pred))
print("\nReport:\n", classification_report(y_test, log_pred))
print("Logistic Confusion:\n", confusion_matrix(y_test, log_pred))
save_model(log_model,name="Cannabis_logistic")



print("Saved Logistic Regression model\n")


rf_model = RandomForestClassifier(n_estimators=1000, n_jobs=-1, min_samples_leaf=2,random_state=42,class_weight="balanced_subsample")
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("Random Forest Results:\n")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("F1 Score:", f1_score(y_test, rf_pred))
print("\nReport:\n", classification_report(y_test, rf_pred))
print("RF Confusion:\n", confusion_matrix(y_test, rf_pred))
save_model(rf_model, name="Cannabis_randomforest")

print("Saved Random Forest model\n")


