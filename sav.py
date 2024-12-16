import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Load Data
data = load_iris()
X = data.data
y = data.target

# 2. Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train a Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Save the Model to a .pkl File
with open("random_forest_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved as random_forest_model.pkl")
