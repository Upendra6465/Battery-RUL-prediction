import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Load dataset
data = pd.read_csv("Battery_RUL.csv")

# ✅ Check basic info
print("Dataset Info:")
print(data.info())  # Check data types & missing values

# ✅ Check feature distributions
print("\nFeature Statistics:")
print(data.describe())

# ✅ Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# ✅ Count unique cycle counts per battery (Assuming 'Cycle_Index' resets per battery)
cycle_counts = data.groupby('Cycle_Index').size()
print("\nCycle Counts Per Battery:")
print(cycle_counts.value_counts())

# ✅ Plot Cycle Index Distribution
plt.figure(figsize=(10, 5))
sns.histplot(data['Cycle_Index'], bins=50, kde=True)
plt.title("Cycle Count Distribution")
plt.xlabel("Cycle Index")
plt.ylabel("Frequency")
plt.show()

# ✅ Plot correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
