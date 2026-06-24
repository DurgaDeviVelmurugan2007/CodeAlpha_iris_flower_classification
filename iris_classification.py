# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
data = pd.read_csv("Iris.csv")

# Display Dataset Information
print("First 5 rows of the dataset:")
print(data.head())

print("\nDataset Shape:", data.shape)

print("\nMissing Values:")
print(data.isnull().sum())

print("\nStatistical Summary:")
print(data.describe())

# Features (Input)
X = data[['SepalLengthCm',
          'SepalWidthCm',
          'PetalLengthCm',
          'PetalWidthCm']]

# Target (Output)
y = data['Species']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Decision Tree Model
model = DecisionTreeClassifier(random_state=42)

# Train Model
model.fit(X_train, y_train)

# Predict Test Data
y_pred = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# ==========================
# Confusion Matrix Heatmap
# ==========================
plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

# Save image
plt.savefig("images/confusion_matrix.png")
plt.show()

# ==========================
# Accuracy Plot
# ==========================
plt.figure(figsize=(5, 5))

plt.bar(
    ['Decision Tree'],
    [accuracy],
    color='skyblue',
    edgecolor='black'
)

plt.ylim(0, 1)
plt.xlabel("Model")
plt.ylabel("Accuracy Score")
plt.title("Model Accuracy")

# Display accuracy value
plt.text(
    0,
    accuracy + 0.02,
    f'{accuracy:.2f}',
    ha='center',
    fontsize=12
)

# Save image
plt.savefig("images/accuracy_plot.png")
plt.show()

# ==========================
# Feature Importance Plot
# ==========================
importance = model.feature_importances_

plt.figure(figsize=(8, 5))

sns.barplot(
    x=importance,
    y=['SepalLengthCm',
       'SepalWidthCm',
       'PetalLengthCm',
       'PetalWidthCm']
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance")

# Save image
plt.savefig("images/feature_importance.png")
plt.show()

# ==========================
# Test with New Flower
# ==========================
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])

prediction = model.predict(new_flower)

print("\nPredicted Species:", prediction[0])
