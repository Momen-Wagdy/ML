import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the preprocessed dataset
df = pd.read_csv('preprocessed_train.csv')

# Split the dataset into features (X) and target variable (y)
X = df.drop(columns=['Survived'])
y = df['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Design the architecture of the neural network
# Here, we'll use a simple feedforward neural network with 2 hidden layers
# You can adjust the number of neurons in each layer, activation functions, etc. based on experimentation
mlp = MLPClassifier(hidden_layer_sizes=(50, 30), activation='relu', solver='adam', random_state=42)

# Train the model
mlp.fit(X_train, y_train)

# Evaluate the model
y_pred = mlp.predict(X_test)
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
y_pred_test = mlp.predict(X_test)
