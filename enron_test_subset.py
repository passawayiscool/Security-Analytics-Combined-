import pandas as pd

# Load the full Enron dataset
df = pd.read_csv("Enron.csv")

# Drop rows with missing values in required columns
df_clean = df.dropna(subset=["subject", "body", "label"])

if len(df_clean) < 500:
	raise ValueError(f"Not enough valid rows after cleaning: {len(df_clean)} found, need 500.")

# Randomly sample 500 valid rows
subset = df_clean.sample(n=500, random_state=42)

# Save to new CSV
subset.to_csv("enron_test_subset(500).csv", index=False)

print("Subset of 500 valid emails saved to enron_test_subset.csv")