import pandas as pd
import numpy as np

data = {
    "age": [22, 25, 28, np.nan, 35],
    "salary": [25000, 30000, np.nan, 40000, 50000],
    "department": ["it", "hr", "it", "sales", "hr"],
    "years_of_experience": [1, 2, 4, np.nan, 8]
}

df = pd.DataFrame(data)

print(df)


df["age"] = df["age"].fillna(df["age"].mean())
df["salary"] = df["salary"].fillna(df["salary"].mean())
df["years_of_experience"] = df["years_of_experience"].fillna(
    df["years_of_experience"].mean()
)

print(df)