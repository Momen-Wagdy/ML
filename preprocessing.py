import pandas as pd

df = pd.read_csv('train.csv')

print("Original dataset:")
print(df.head())

columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
df = df.drop(columns=columns_to_drop)

df['Age'].fillna(df['Age'].mean(), inplace=True)

df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

df = pd.get_dummies(df, columns=['Embarked'])


print("\nPreprocessed dataset:")
print(df.head())

df.to_csv('preprocessed_train.csv', index=False)
