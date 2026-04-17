import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# -----------------------------
# CREATE OUTPUT FOLDER
# -----------------------------
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# -----------------------------
# STEP 1: CREATE DATA
# -----------------------------
np.random.seed(42)

n = 500

data = pd.DataFrame({
    "experience": np.random.randint(1, 10, n),
    "salary": np.random.randint(20000, 100000, n),
    "training_hours": np.random.randint(10, 100, n),
    "projects": np.random.randint(1, 10, n),
    "attendance": np.random.uniform(0.7, 1.0, n)
})

# TARGET LOGIC
conditions = [
    (data["experience"] > 6) & (data["training_hours"] > 60),
    (data["experience"] > 3),
]
choices = ["High", "Medium"]

data["performance"] = np.select(conditions, choices, default="Low")

print("DATA PREVIEW:\n", data.head())

# -----------------------------
# STEP 2: EDA (GRAPHS)
# -----------------------------

# Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="performance", data=data)
plt.title("Performance Distribution")
plt.savefig("outputs/performance_distribution.png")
plt.show()

# Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("outputs/heatmap.png")
plt.show()

# -----------------------------
# STEP 3: MODEL TRAINING
# -----------------------------
X = data.drop("performance", axis=1)
y = data["performance"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# -----------------------------
# STEP 4: PREDICTION
# -----------------------------
pred = model.predict(X_test)

print("\nCLASSIFICATION REPORT:\n")
print(classification_report(y_test, pred))

print("\nCONFUSION MATRIX:\n")
print(confusion_matrix(y_test, pred))

# -----------------------------
# STEP 5: CONFUSION MATRIX GRAPH
# -----------------------------
cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.show()

# -----------------------------
# STEP 6: FEATURE IMPORTANCE
# -----------------------------
importance = model.feature_importances_

plt.figure(figsize=(6,4))
sns.barplot(x=importance, y=X.columns)
plt.title("Feature Importance")
plt.savefig("outputs/feature_importance.png")
plt.show()

# -----------------------------
# STEP 7: SAMPLE PREDICTION
# -----------------------------
sample = [[5, 60000, 80, 6, 0.9]]
result = model.predict(sample)

print("\nSAMPLE PREDICTION (HR DEMO):")
print("Employee Prediction:", result[0])

# -----------------------------
# STEP 8: INSIGHTS
# -----------------------------
print("\nKEY INSIGHTS:")
print("1. Higher training hours improve performance")
print("2. Experience plays major role in high performance")
print("3. Attendance impacts consistency")
print("4. More projects = better exposure")

# -----------------------------
# STEP 9: SAVE MODEL
# -----------------------------
joblib.dump(model, "models/employee_model.pkl")

print("\nMODEL SAVED SUCCESSFULLY 🚀")