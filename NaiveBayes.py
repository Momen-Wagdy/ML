import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the preprocessed dataset
df = pd.read_csv('preprocessed_train.csv')

# Split the dataset into features (X) and target variable (y)
X = df.drop(columns=['Survived'])
y = df['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implement Naive Bayes algorithm (Gaussian Naive Bayes)
naive_bayes = GaussianNB()

# Train the model
naive_bayes.fit(X_train, y_train)

# Evaluate the model
y_pred = naive_bayes.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print performance metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

# Apply the trained model to the testing set for predictions
y_pred_test = naive_bayes.predict(X_test)
