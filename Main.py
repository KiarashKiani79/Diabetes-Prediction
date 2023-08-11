import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("diabetes_data.csv")

# Preprocess the data
# Replace zero values with NaN for relevant columns
data["Glucose"] = data["Glucose"].replace(0, np.nan)
data["BloodPressure"] = data["BloodPressure"].replace(0, np.nan)
data["SkinThickness"] = data["SkinThickness"].replace(0, np.nan)
data["Insulin"] = data["Insulin"].replace(0, np.nan)
data["BMI"] = data["BMI"].replace(0, np.nan)
data["DiabetesPedigreeFunction"] = data["DiabetesPedigreeFunction"].replace(0, np.nan)
data["Age"] = data["Age"].replace(0, np.nan)

# Drop rows with missing values
data.dropna(inplace=True)

# Split the data into features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Implement the logistic regression algorithm from scratch
class LogisticRegression:
    def __init__(self, learning_rate=0.001, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def _sigmoid(self, z):
        z = np.clip(z, -709, 709)  # Limit the range of z to prevent overflow
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Initialize weights
        self.weights = np.zeros(X.shape[1])

        # Gradient descent
        for _ in range(self.num_iterations):
            # Compute predictions
            z = np.dot(X, self.weights)
            y_pred = self._sigmoid(z)

            # Update weights
            gradient = np.dot(X.T, (y_pred - y))
            self.weights -= self.learning_rate * gradient

    def predict_proba(self, X):
        z = np.dot(X, self.weights)
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

 
# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model's performance
y_pred = model.predict(X_val)

# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy:.2f}")

def get_user_input():
    # Ask the user for input and return it as a pandas DataFrame
    pregnancies = float(input("+ Number of pregnancies: "))
    glucose = float(input("+ Glucose level: "))
    blood_pressure = float(input("+ Blood pressure: "))
    skin_thickness = float(input("+ Skin thickness: "))
    insulin = float(input("+ Insulin level: "))
    bmi = float(input("+ Body mass index (BMI): "))
    diabetes_pedigree = float(input("+ Diabetes pedigree function: "))
    age = float(input("+ Age: "))

    user_input = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness,
          insulin, bmi, diabetes_pedigree, age]],
        columns=X.columns
    )

    # Preprocess the user input
    user_input = scaler.transform(user_input)

    return user_input

def main():
    while True:
        print("==================================================\n")
        print("Enter the following information about the patient:")
        user_input = get_user_input()

        # Predict the outcome
        prediction = model.predict(user_input)

        print(f"Prediction: {'Diabetic' if prediction[0] == 1 else 'Non-diabetic'}")

        choice = input("Would you like to make another prediction? (y/n): ")
        if choice.lower() != "y":
            print("\nTake care!\n")
            break

print("\n==================================================")
print("=== Welcome to the diabetes prediction tool! ===")
if __name__ == "__main__":
    main()
