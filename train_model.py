import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

print("1. Loading dataset...")
# Load the data you just collected
df = pd.read_csv('sign_dataset.csv')

# Separate the coordinates (X) from the answers/labels (y)
X = df.drop('label', axis=1)
y = df['label']

print("2. Splitting data into training and testing sets...")
# Keep 80% of data for training, and 20% for testing the AI like a pop quiz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("3. Training the Random Forest model...")
# Create the algorithm and teach it
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("4. Testing the model...")
# Give the AI the test data and see how many it guesses correctly
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"--> Model Accuracy: {accuracy * 100:.2f}%\n")

# Save the trained brain to a file
with open('sign_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
print("Success! Model saved as 'sign_model.pkl'.")