# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Step 2: Load the Iris dataset
try:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("✅ Dataset loaded successfully!\n")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")

# Step 3: Explore the dataset
print("👀 First 5 rows of the dataset:")
print(df.head())

print("\n📋 Dataset info:")
print(df.info())

print("\n❓ Missing values check:")
print(df.isnull().sum())

# Step 4: Analyze data
print("\n📊 Descriptive statistics:")
print(df.describe())

print("\n📈 Mean values grouped by species:")
print(df.groupby('species').mean())

# Step 5: Visualize the data
sns.set(style="whitegrid")

# Line chart: Sepal length over index by species
sns.lineplot(data=df, x=df.index, y='sepal length (cm)', hue='species')
plt.title("Sepal Length Over Index")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

# Bar chart: Average petal length per species
sns.barplot(data=df, x='species', y='petal length (cm)', estimator='mean')
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# Histogram: Distribution of sepal width
sns.histplot(df['sepal width (cm)'], bins=10, kde=True)
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# Scatter plot: Sepal length vs petal length
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.show()

# My Observations:
# - Setosa has the shortest petals.
# - Virginica has the longest petals on average.
# - Sepal width is fairly normally distributed.
# - There's a positive relationship between sepal length and petal length.
