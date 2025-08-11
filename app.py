import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# PART 1: Pickle Student Data
# -----------------------------
student = {
    "id": 101,
    "name": "Vyshnavi",
    "age": 21,
    "gender": "Female",
    "grades": {"Math": 90, "Science": 88, "English": 85}
}

# Save student data
with open("student_record.pkl", "wb") as file:
    pickle.dump(student, file)
print(" Student data saved to 'student_record.pkl'.")

# Load student data
with open("student_record.pkl", "rb") as file:
    loaded_student = pickle.load(file)

# Display student data
print("\n Loaded Student Data:")
for key, value in loaded_student.items():
    print(f"{key}: {value}")

# -----------------------------
# PART 2: Create Dataset for Training
# -----------------------------
data = {
    "Math": [50, 80, 65, 90, 78, 92, 55, 47, 99, 73],
    "Science": [60, 90, 75, 88, 85, 90, 52, 50, 95, 79],
    "English": [70, 100, 85, 85, 80, 91, 50, 48, 96, 81],
    "Performance": [0, 1, 1, 1, 1, 1, 0, 0, 1, 1]  # 1=Excellent, 0=Needs Improvement
}
df = pd.DataFrame(data)

X = df[["Math", "Science", "English"]]
y = df["Performance"]

# -----------------------------
# PART 3: Train-Test Split and Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Save scaler
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)
print("\n Scaler saved to 'scaler.pkl'.")

# -----------------------------
# PART 4: Train Model
# -----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
with open("student_model.pkl", "wb") as file:
    pickle.dump(model, file)
print(" Model saved to 'student_model.pkl'.")

# -----------------------------
# PART 5: Test Model
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n Model Accuracy on Test Data: {accuracy:.2f}")

# -----------------------------
# PART 6: Predict Using Saved Model and Scaler
# -----------------------------
# Extract marks from loaded_student
marks = list(loaded_student["grades"].values())  # [Math, Science, English]

# Load scaler and model
loaded_scaler = pickle.load(open("scaler.pkl", "rb"))
loaded_model = pickle.load(open("student_model.pkl", "rb"))

# Scale input and predict
scaled_input = loaded_scaler.transform([marks])
prediction = loaded_model.predict(scaled_input)

# Show result
print("\n Prediction for student (", loaded_student["name"], "):")
print(" Excellent" if prediction[0] == 1 else "Needs Improvement")