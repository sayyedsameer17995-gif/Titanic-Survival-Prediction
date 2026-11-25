
# ==========================================
# TITANIC SURVIVAL PREDICTION
# Machine Learning Project
# Author: Sameer Sayyed
# ==========================================


# 1. Import Libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# 2. Load Dataset

titanic = fetch_openml("titanic", version=1, as_frame=True)

df = titanic.frame


# 3. Explore Dataset

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSurvival Counts:")
print(df["survived"].value_counts())


# 4. Visualize Survival Count

sns.countplot(data=df, x="survived")

plt.title("Titanic Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")

plt.show()


# 5. Visualize Survival by Gender

sns.countplot(data=df, x="sex", hue="survived")

plt.title("Survival Count by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")

plt.show()


# 6. Visualize Age vs Fare

sns.scatterplot(data=df, x="age", y="fare", hue="survived")

plt.title("Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")

plt.show()


# 7. Select Useful Features

df = df[
    ["survived", "pclass", "sex", "age", "fare", "sibsp", "parch"]
].copy()


# 8. Feature Engineering

df["FamilySize"] = df["sibsp"] + df["parch"] + 1


# 9. Handle Missing Values

df["age"] = df["age"].fillna(df["age"].mean())

df["fare"] = df["fare"].fillna(df["fare"].mean())


# 10. Convert Gender into Numerical Values

le = LabelEncoder()

df["sex"] = le.fit_transform(df["sex"])


# 11. Convert Target into Integer Values

df["survived"] = df["survived"].astype(int)


# 12. Separate Features and Target

X = df.drop("survived", axis=1)

y = df["survived"]


# 13. Split Dataset into Training and Testing Sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 14. Create Decision Tree Model

model = DecisionTreeClassifier(random_state=42)


# 15. Train the Model

model.fit(X_train, y_train)


# 16. Make Predictions

y_pred = model.predict(X_test)

print("\nFirst 10 Predictions:")
print(y_pred[:10])


# 17. Calculate Model Accuracy

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100, "%")


# 18. Compare Actual and Predicted Values

results = pd.DataFrame({
    "Actual": y_test.iloc[:10].values,
    "Predicted": y_pred[:10]
})

print("\nActual vs Predicted Results:")
print(results)
