import pandas as pd

# Load the full Enron dataset
df = pd.read_csv("Enron.csv")

# Randomly sample 500 rows
subset = df.sample(n=500, random_state=42)

# Save to new CSV
subset.to_csv("enron_test_subset.csv", index=False)

print("Subset of 500 emails saved to enron_test_subset.csv")