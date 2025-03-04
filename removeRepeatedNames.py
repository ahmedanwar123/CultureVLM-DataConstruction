import pandas as pd

# Load the CSV file
df = pd.read_csv("gldv2_info.csv")

# Extract unique names
unique_names = df["name"].drop_duplicates()

# Save to a new CSV file
unique_names.to_csv("unique_names.csv", index=False, header=["name"])

print("Unique names saved to unique_names.csv")
