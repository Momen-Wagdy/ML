import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the preprocessed dataset
df = pd.read_csv('preprocessed_train.csv')

# Pairplot for visualizing relationships between variables
sns.pairplot(df)
plt.show()

# Correlation heatmap
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Split the dataset into features (X) and target variable (y)
X = df.drop(columns=['Survived'])
y = df['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implement KNN algorithm with a range of k values
k_values = [3, 5, 7, 9, 11]
distance_metrics = ['euclidean', 'manhattan', 'chebyshev']

best_k = None
best_score = 0

for k in k_values:
    for metric in distance_metrics:
        # Initialize KNN classifier with current k value and distance metric
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        
        # Train the model
        knn.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Print performance metrics
        print(f"K={k}, Metric={metric}: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1-score={f1}")
        
        # Update best k value if current performance is better
        if accuracy > best_score:
            best_score = accuracy
            best_k = k

# Select the best k value
print(f"\nBest k value: {best_k} (Accuracy={best_score})")

# Train the model with the best k value
best_knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
best_knn.fit(X_train, y_train)

# Apply the model to the testing set for predictions
y_pred_test = best_knn.predict(X_test)

# Evaluate the model performance on testing set
accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)

print("\nPerformance on Testing Set:")
print(f"Accuracy={accuracy_test}, Precision={precision_test}, Recall={recall_test}, F1-score={f1_test}")
