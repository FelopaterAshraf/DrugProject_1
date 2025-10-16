import pandas as pd
import matplotlib.pyplot as plt
from utils import load_data , explore_data,handle_duplicates,handle_nulls,handle_outliers,outlier_detection,unique_values
import os
import seaborn as sns

os.makedirs("plots", exist_ok=True)

df = load_data()

df = df.drop(columns=["ID", "Unnamed: 0"], errors="ignore")

explore_data(df)
outlier_detection(df, factor=1.5)
unique_values(df)

# df = handle_duplicates(df)
# df = handle_nulls(df)          
# df = handle_outliers(df, factor=1.5)


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

plt.figure(figsize=(12, 10))
sns.heatmap(df[num_cols].corr(), cmap="coolwarm", center=0, annot=True, linewidths=0.5, fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png")
plt.close()
print("Saved correlation_heatmap.png")
