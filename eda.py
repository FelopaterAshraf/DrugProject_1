import pandas as pd
import matplotlib.pyplot as plt
from utils import load_data
import os

os.makedirs("plots", exist_ok=True)

df = load_data()

df = df.drop(columns=["ID", "Unnamed: 0"], errors="ignore")


df.hist(figsize=(12,10), bins=20, color="skyblue", edgecolor="black")
plt.tight_layout()
plt.savefig("plots/histograms.png")
plt.close()
print("Saved histograms.png")


num_cols = df.select_dtypes(include="number").columns

plt.figure(figsize=(14,8))
df[num_cols].boxplot()
plt.title("Boxplots for All Numeric Columns")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/boxplots_all_numeric.png")
plt.close()
print("Saved boxplots_all_numeric.png")
