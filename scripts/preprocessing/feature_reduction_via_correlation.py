import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("../../data/cleaned/Balanced_Applied_Science_Journals.csv")
features = df.drop(columns=["Title", "Publisher", "Labels"], errors='ignore')
numeric_features = features.select_dtypes(include=["float64", "int64"])
correlation_matrix = numeric_features.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, linewidths=.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()
