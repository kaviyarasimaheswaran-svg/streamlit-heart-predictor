# predict.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Step 1: Load dataset
data = pd.read_csv("heart.csv")

# Step 2: Split features and target
X = data.drop("target", axis=1)
y = data["target"]

# Step 3: Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)  # you can tune this value
knn.fit(x_train, y_train)

# Step 5: Save model
with open("heart_knn_model.pkl", "wb") as file:
    pickle.dump(knn, file)
print("✅ KNN Model Saved")

# Step 6: Load model
with open("heart_knn_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)
print("✅ KNN Model Loaded")

# Step 7: Evaluate accuracy
y_pred = loaded_model.predict(x_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 8: Test prediction on one sample
sample = x_test.sample(1)
prediction = loaded_model.predict(sample)
print("Prediction for sample:", prediction)
