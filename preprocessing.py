import pandas as pd

# Load the dataset
df = pd.read_csv('train.csv')

# Display the first few rows of the dataset
print("Original dataset:")
print(df.head())

# Drop unnecessary columns
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
df = df.drop(columns=columns_to_drop)

# Handling missing values
# For numerical columns, fill missing values with mean
df['Age'].fillna(df['Age'].mean(), inplace=True)
# For categorical column 'Embarked', fill missing values with the most frequent value
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encoding categorical variables
# Convert 'Sex' column to numerical values (1 for male, 0 for female)
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
# One-hot encode 'Embarked' column
df = pd.get_dummies(df, columns=['Embarked'])

# Feature Scaling (optional)
# You may want to scale numerical features like 'Age' and 'Fare' if needed

# Display the preprocessed dataset
print("\nPreprocessed dataset:")
print(df.head())

# Save the preprocessed dataset to a new CSV file
df.to_csv('preprocessed_train.csv', index=False)
