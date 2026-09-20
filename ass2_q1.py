import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target


print("\nFirst Five Rows")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

print("\n Missing Values")
print(df.isnull().sum())

print("\nDuplicate Rows ")
print(df.duplicated().sum())

print("\nTarget Value Counts ")
print(df["target"].value_counts())

print("\n Correlation Matrix")
print(df.corr(numeric_only=True))

plt.figure(figsize=(7, 5))

sns.histplot(
    df["alcohol"],
    kde=True
)

plt.title("Distribution of Alcohol")


plt.figure(figsize=(7, 5))

sns.scatterplot(
    data=df,
    x="alcohol",
    y="color_intensity",
    hue="target"
)

plt.title("Alcohol vs Color Intensity by Class")

plt.show()