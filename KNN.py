import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


df = pd.read_csv('preprocessed_train.csv')


sns.pairplot(df)
plt.show()

correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


X = df.drop(columns=['Survived'])
y = df['Survived']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


k_values = [3, 5, 7, 9, 11]
distance_metrics = ['euclidean', 'manhattan', 'chebyshev']

best_k = None
best_score = 0

for k in k_values:
    for metric in distance_metrics:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        

        knn.fit(X_train, y_train)
        

        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        

        print(f"K={k}, Metric={metric}: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1-score={f1}")
        

        if accuracy > best_score:
            best_score = accuracy
            best_k = k


print(f"\nBest k value: {best_k} (Accuracy={best_score})")


best_knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
best_knn.fit(X_train, y_train)


y_pred_test = best_knn.predict(X_test)


accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)

print("\nPerformance on Testing Set:")
print(f"Accuracy={accuracy_test}, Precision={precision_test}, Recall={recall_test}, F1-score={f1_test}")
