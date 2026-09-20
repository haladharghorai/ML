import pandas as pd

data = {
    "Student Name": ["Rahul", "Amit", "Priya", "Neha", "Riya"],
    "Roll Number": [1, 2, 3, 4, 5],
    "Marks": [85, 72, 91, 78, 88],
    "Attendance": [90, 85, 95, 80, 92]
}

df = pd.DataFrame(data)

print("Student Data:")
print(df)

result = df[df["Marks"] > 80]

print("\nStudents who scored above 80:")
print(result)