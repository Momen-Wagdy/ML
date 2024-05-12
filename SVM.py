import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the preprocessed dataset
df = pd.read_csv('preprocessed_train.csv')

# Split the dataset into features (X) and target variable (y)
X = df.drop(columns=['Survived'])
y = df['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implement SVM algorithm with various hyperparameters
# Define the parameter grid for grid search
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.1, 1, 10],
    'kernel': ['linear', 'poly', 'rbf']
}

# Initialize SVM classifier
svm = SVC()

# Perform grid search to find the best combination of hyperparameters
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters from grid search
best_params = grid_search.best_params_

# Train the model with the best hyperparameters
svm_best = SVC(**best_params)
svm_best.fit(X_train, y_train)

# Evaluate the model
y_pred = svm_best.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print performance metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

# Apply the tuned model to the testing set for predictions
y_pred_test = svm_best.predict(X_test)
