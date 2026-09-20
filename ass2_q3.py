import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)

corr = df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Correlation Heatmap of Wine Dataset")
plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.show()

corr_matrix = corr.copy()

for i in range(len(corr_matrix)):
    corr_matrix.iloc[i, i] = 0

max_pair = corr_matrix.stack().idxmax()
max_value = corr_matrix.stack().max()

print("Strongest Positive Correlation:")
print(max_pair[0], "and", max_pair[1])
print("Correlation:", round(max_value, 2))