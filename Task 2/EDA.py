import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df = sns.load_dataset('titanic')

print("Shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nStatistical Summary:\n", df.describe(include='all'))

df.drop(['deck'], axis=1, inplace=True)

df['age'].fillna(df['age'].median(), inplace=True)

df['embark_town'].fillna(df['embark_town'].mode()[0], inplace=True)

df.dropna(inplace=True)

plt.hist(df['age'], bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

sns.countplot(x='survived', data=df)
plt.title('Survival Count')
plt.show()

sns.countplot(x='sex', hue='survived', data=df)
plt.title('Survival by Gender')
plt.show()

sns.scatterplot(x='age', y='fare', hue='survived', data=df)
plt.title('Age vs Fare (colored by Survival)')
plt.show()

numeric_df = df.select_dtypes(include=['number'])  # only numeric columns
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

sns.boxplot(x=df['fare'])
plt.title('Boxplot for Fare')
plt.show()

survivors = df[df['survived'] == 1]['age']
non_survivors = df[df['survived'] == 0]['age']

t_stat, p_val = stats.ttest_ind(survivors, non_survivors)
print(f"T-test on Age of Survivors vs Non-Survivors: t={t_stat:.2f}, p={p_val:.4f}")

df.to_csv("titanic_cleaned.csv", index=False)
print("Cleaned dataset saved as titanic_cleaned.csv")
