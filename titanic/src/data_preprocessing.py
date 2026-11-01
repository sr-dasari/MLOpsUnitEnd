# Task 1: Working with Titanic Data
# 1.1 Import path utilities and pandas for reading the dataset.
# %%
print(f"1. Importing path utilities and pandas for reading the dataset.")
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Announce that the preprocessing script has started.
print("Starting Titanic Dataset Preprocessing Pipeline... \n")

# Locate the input and output CSV files in the project's data directory.
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
raw_file = DATA_DIR / 'titanic.csv'
processed_file = DATA_DIR / 'processed_titanic.csv'

# %%
# 1.2. Read the raw Titanic dataset and report its dimensions.
print(f"2. Reading raw data from: {raw_file}...")
df = pd.read_csv(raw_file)
print(f"Loaded raw dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print()

# %%
# 1.3. Inspect data types and identify categorical and numerical features.
print("3. Inspecting data types and identifying feature categories...")
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"   • Categorical features: {categorical_features}")
print(f"   • Numerical features: {numerical_features}")
print()

# %%
# 1.4. Explore data distribution and identify missing or inconsistent values.
print("4. Exploring data distribution and identifying missing/inconsistent values...")
print(f"   • Missing values per column:\n{df.isnull().sum()}")
print(f"   • Unique values per column:\n{df.nunique()}")
print()

# %%
# 1.4a. Check for Categorical Inconsistencies
print("Sex values:", df['Sex'].unique())
print("Embarked values:", df['Embarked'].unique())

# %%
# 1.4b. Logical Inconsistencies
# Check for impossible combinations (e.g., infants travelling alone)
infant_alone = df[(df['Age'] < 1) & (df['SibSp'] == 0) & (df['Parch'] == 0)]
print(f"Logical errors (Infants travelling alone): {len(infant_alone)}")

# %%
# 1.4c. Check for Numerical Inconsistencies (Outliers)
print("Checking for numerical inconsistencies...")
print(df[['Age', 'Fare', 'SibSp', 'Parch']].describe().loc[['min', 'max']])

# %%
# 1.5. Perform grouping and pivot analysis on key variables such as Pclass and Sex.
print("\n5. Performing grouping and pivot analysis...")
grouped_data = df.groupby(['Pclass', 'Sex']).size()
print(grouped_data)

# %%
# 1.6. Generate summary statistics to understand survival patterns.
print("\n6. Generating summary statistics...")
summary_stats = df.groupby('Survived').agg({
    'Age': 'mean',
    'Fare': 'mean',
    'Pclass': 'mean',
    'SibSp': 'mean',
    'Parch': 'mean'
})
print(summary_stats)

# %%
# 1.7. Prepare insights for model training and reporting.
print("\n7. Preparing insights for model training and reporting...")
print("\n--- Synthesized Insights for Reporting & Training ---")
overall_survival = df["Survived"].mean()
print(f"1. Overall Survival Rate: {overall_survival:.2%}")

# Highlight class disparities
class_survival = df.groupby("Pclass")["Survived"].mean()
for cls, rate in class_survival.items():
    print(f"   - Survival Rate in Class {cls}: {rate:.2%}")

# Highlight gender disparities
gender_survival = df.groupby("Sex")["Survived"].mean()
for sex, rate in gender_survival.items():
    print(f"2. Gender Impact: Survival rate for {sex}: {rate:.2%}")

# Age impact
child_survival = df[df["Age"] < 16]["Survived"].mean()
adult_survival = df[df["Age"] >= 16]["Survived"].mean()
print(f"3. Age Impact: Survival rate for children (<16) is {child_survival:.2%}, "
      f"compared to {adult_survival:.2%} for adults.")

print("\n--- Modeling Recommendations ---")
print("- Drop high-cardinality non-informative columns: 'PassengerId', 'Name', 'Ticket', 'Cabin'")
print(f"- Encode categorical variables {['Sex', 'Embarked']} using One-Hot Encoding.")
print(f"- Fill missing Age values with median ({df['Age'].median():.0f}).")
print(f"- Fill missing Embarked values with mode ('{df['Embarked'].mode()[0]}').")
print(f"- Address class imbalance: {overall_survival:.2%} survived vs {1 - overall_survival:.2%} did not.")

# %%
# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
print(f"\nFilled missing values: Age ({df['Age'].isnull().sum()} remaining), Embarked ({df['Embarked'].isnull().sum()} remaining)")

# %%
# Drop high-cardinality non-informative columns
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

print("Remaining columns in DataFrame:")
print(df.columns.tolist())

# %%
# One-hot encode categorical columns
categorical_cols = ['Sex', 'Embarked']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("Columns after One-Hot Encoding:")
print(df_encoded.columns.tolist())

# %%
# Save Processed Data
print(f"\nSaving processed dataset to: {processed_file}")
processed_file.parent.mkdir(parents=True, exist_ok=True)
df_encoded.to_csv(processed_file, index=False)

print(f"\nPreprocessing completed successfully!")
print(f"   • Final dataset shape: {df_encoded.shape[0]} rows, {df_encoded.shape[1]} columns")

# %%
