import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

df = pd.DataFrame({
    'Age': [22, 25, 28, 30, 35],
    'Salary': [25000, 30000, 40000, 45000, 55000],
    'Years_of_Experience': [1, 2, 4, 5, 10]
})

pipeline = Pipeline([
    ('scaler', MinMaxScaler())
])

scaled_data = pipeline.fit_transform(df)

scaled_df = pd.DataFrame(
    scaled_data,
    columns=df.columns
)

print("Dataset after MinMax Scaling:")
print(scaled_df)
