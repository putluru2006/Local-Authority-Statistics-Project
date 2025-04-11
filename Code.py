import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Load the dataset
df = pd.read_csv("C:\\Users\\putlu\\Downloads\\local-authority-statistics-december-2024-quarter (1).csv")

# Clean column names
df.columns = df.columns.str.strip()

# 1. Load & Explore the Dataset
print("=== First 5 Rows ===")
print(df.head())

print("\n=== Dataset Info ===")
print(df.info())

print("\n=== Summary Statistics ===")
print(df.describe(include='all'))

print("\n=== Missing Values ===")
print(df.isnull().sum())

# 2. Data Cleaning & Preprocessing
df.fillna(df.mean(numeric_only=True), inplace=True)


for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\n=== Missing Values After Cleaning ===")
print(df.isnull().sum())

# 3. Statistical Analysis
correlation_matrix = df.corr(numeric_only=True)
print("\n=== Correlation Matrix ===")
print(correlation_matrix)

numeric_df = df.select_dtypes(include=np.number)
filtered_df = numeric_df.loc[:, numeric_df.std() > 1e-6]
z_scores = np.abs(zscore(filtered_df))
outliers = (z_scores > 3).sum()
print("\n=== Outliers Count (Z-Score > 3) ===")
print(outliers)
# 4. Data Visualization

 #Boxplot
df.select_dtypes(include='number').plot(
    kind='box',
    subplots=True,
    layout=(3, 3),
    figsize=(15, 10),
    sharex=False,
    color=dict(boxes='darkblue', whiskers='black', medians='red', caps='gray')
)
plt.suptitle("Boxplots for Numerical Features", fontsize=16)
plt.tight_layout()
plt.show()


# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Numerical Columns", fontsize=14)
plt.show()

# 5. Trend Analysis
col = df.select_dtypes(include='number').columns[1]
plt.plot(df.index, df[col], marker='.')
plt.title("Line Graph of " + col, fontsize=16)
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Pie chart
df.columns = df.columns.str.strip()
top_groups = df["Group"].value_counts().head(5)
plt.pie(
    top_groups,
    labels=top_groups.index,
    autopct='%1.0f%%',
    startangle=45,
    explode=[0.05]*5,
    shadow=True,
    colors=['yellow', 'green', 'blue', 'pink', 'violet']
)
plt.title("Top 5 Groups Distribution", fontsize=14)
plt.tight_layout()
plt.show()

# 7. Bar Chart
df["Group"].value_counts().plot(
    kind='bar',
    figsize=(8, 6),
    title="Bar Chart of Group Distribution",
    color=['skyblue', 'lightgreen', 'orange', 'lightcoral', 'violet', 'gold', 'turquoise']
)
plt.xlabel("Group")
plt.ylabel("Count")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 8. Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x="Period", y="Data_value", data=df, color=['green'], marker='*')
plt.title("Scatter Plot of Data Value vs. Period")
plt.xlabel("Period")
plt.ylabel("Data Value")
plt.tight_layout()
plt.show()

# 9. Pair Plot
sns.pairplot(df.select_dtypes(include=np.number),
             plot_kws={'color':'blue','marker':'.'},height=2.2,aspect=1  )
plt.suptitle("Pair Plot of Numerical Features", y=1, fontsize=16, fontweight='bold')
plt.show()
