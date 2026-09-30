import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load the student dataset
data = pd.read_csv("student_data.csv")

# Input features
X = data[
    [
        "study_hours",
        "attendance",
        "previous_score",
        "assignments_completed",
        "sleep_hours"
    ]
]

# Target value we want to predict
y = data["final_score"]

# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create the AI model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

# Calculate prediction error
error = mean_absolute_error(y_test, predictions)

print("AI model trained successfully!")
print("Mean Absolute Error:", round(error, 2))

# Save the trained model
joblib.dump(model, "student_performance_model.pkl")

print("Model saved as student_performance_model.pkl")