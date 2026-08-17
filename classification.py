from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score
)
import matplotlib.pyplot as plt


# Load the Iris dataset
iris = load_iris()

print("=== Iris Dataset ===")
print("Number of samples:", len(iris.data))
print("Number of features:", iris.data.shape[1])
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)

# Separate features and target
X = iris.data
y = iris.target

print("\nFeature data shape:", X.shape)
print("Target data shape:", y.shape)

# Step 9: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Step 10: Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed.")

# Step 11: Test different K values
k_values = range(1, 16)
k_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)

    predictions = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)

    k_accuracies.append(accuracy)

best_k = k_values[list(k_accuracies).index(max(k_accuracies))]

print("\nBest K value:", best_k)
print("Best accuracy:", round(max(k_accuracies) * 100, 2), "%")

# Step 12: Train final KNN model
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train_scaled, y_train)

print("\nFinal KNN model trained successfully.")

# Step 13: Make predictions
y_pred = model.predict(X_test_scaled)

print("\nPredictions completed.")
print("Predicted classes:", y_pred)

# Step 14: Evaluate model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n=== Model Evaluation ===")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Weighted F1 Score:", round(f1, 2))

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)

# Step 15: Plot K values
plt.figure(figsize=(8, 5))
plt.plot(k_values, k_accuracies, marker="o")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy for Different K Values")
plt.xticks(list(k_values))
plt.grid(True)
plt.show()