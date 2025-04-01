import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

# Load the dataset
df = pd.read_csv('data/by_country/can.csv')

# Basic data exploration
print(f"Original dataset shape: {df.shape}")
print(f"Original columns: {len(df.columns)}")

# Convert columns to numeric where possible, handling errors
for col in df.columns:
    if col not in ['geo', 'time']:  # Skip non-numeric columns
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Count non-NA values in each column
value_counts = df.count()

# Remove columns with fewer than 5 values
min_values_required = 5
columns_to_keep = value_counts[value_counts >= min_values_required].index.tolist()
columns_to_keep = ['geo', 'time'] + [col for col in columns_to_keep if col not in ['geo', 'time']]
df_filtered = df[columns_to_keep]

# Print information about removed columns
removed_insufficient_columns = [col for col in df.columns if col not in columns_to_keep]
print(f"Removed {len(removed_insufficient_columns)} columns with fewer than {min_values_required} values")
print(f"Filtered dataset shape: {df_filtered.shape}")

# Separate non-numeric and numeric columns
non_numeric_cols = ['geo', 'time']
numeric_cols = [col for col in df_filtered.columns if col not in non_numeric_cols]

# Detect and remove duplicate columns (columns with exactly the same values)
print("Detecting duplicate columns...")
duplicates = {}
duplicate_columns = []

# We'll compare each column with every other column
for i, col1 in enumerate(numeric_cols):
    for col2 in numeric_cols[i+1:]:
        # Check if columns are identical (ignoring NaN values)
        # Two columns are identical if they have the same non-NaN values in the same rows
        # And if they have NaN values in the same rows
        is_duplicate = True
        for idx, (val1, val2) in enumerate(zip(df_filtered[col1], df_filtered[col2])):
            # If one is NaN and the other isn't, they're not identical
            if pd.isna(val1) != pd.isna(val2):
                is_duplicate = False
                break
            # If both are not NaN and they're not equal, they're not identical
            if not pd.isna(val1) and not pd.isna(val2) and val1 != val2:
                is_duplicate = False
                break
        
        if is_duplicate:
            if col1 not in duplicates:
                duplicates[col1] = []
            duplicates[col1].append(col2)
            duplicate_columns.append(col2)

# Remove duplicate columns
unique_duplicate_columns = list(set(duplicate_columns))
df_no_duplicates = df_filtered.drop(columns=unique_duplicate_columns)

print(f"Found {len(unique_duplicate_columns)} duplicate columns")
print("Duplicate column groups:")
for col, dups in duplicates.items():
    if dups:  # Only print if there are duplicates
        print(f"  {col} has duplicates: {', '.join(dups)}")

# Save the list of duplicate columns
duplicate_cols_df = pd.DataFrame(unique_duplicate_columns, columns=['Duplicate_Column'])
duplicate_cols_df.to_csv('duplicate_columns.csv', index=False)
print(f"Saved list of {len(unique_duplicate_columns)} duplicate columns to 'duplicate_columns.csv'")

# Update numeric columns list after removing duplicates
numeric_cols = [col for col in df_no_duplicates.columns if col not in non_numeric_cols]

# Extract numeric data for imputation
numeric_data = df_no_duplicates[numeric_cols]
print(f"Dataset shape after removing duplicate columns: {df_no_duplicates.shape}")

# Fill missing values using MLE-based approach (Iterative Imputer with Bayesian Ridge Regression)
print("Imputing missing values using MLE-based approach...")
imputer = IterativeImputer(
    estimator=BayesianRidge(),
    max_iter=100,
    random_state=42,
    min_value=numeric_data.min().min(),  # Respect minimum values in the dataset
    max_value=numeric_data.max().max()   # Respect maximum values in the dataset
)
imputed_data = imputer.fit_transform(numeric_data)

# Create a new DataFrame with imputed values
imputed_df = pd.DataFrame(imputed_data, columns=numeric_cols)

# Add back non-numeric columns
for col in non_numeric_cols:
    imputed_df[col] = df_no_duplicates[col].values

# Reorder columns to match original
all_cols = non_numeric_cols + numeric_cols
imputed_df = imputed_df[all_cols]

print(f"Imputed dataset shape: {imputed_df.shape}")

# Identify constant columns (columns with only one unique value)
constant_columns = []
for col in numeric_cols:
    if imputed_df[col].nunique() <= 2:
        constant_columns.append(col)

print(f"Found {len(constant_columns)} constant columns")

# Remove constant columns
df_without_constants = imputed_df.drop(columns=constant_columns)
print(f"Dataset shape after removing constant columns: {df_without_constants.shape}")

# Calculate basic statistics for remaining columns
remaining_numeric_cols = [col for col in numeric_cols if col not in constant_columns]
stats_df = df_without_constants[remaining_numeric_cols].describe().T
stats_df['column_name'] = stats_df.index
stats_df = stats_df[['column_name', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]

# Save the cleaned dataset (without constant columns)
df_without_constants.to_csv('cleaned_dataset.csv', index=False)
print(f"Saved cleaned dataset with {df_without_constants.shape[1]} columns to 'cleaned_dataset.csv'")

# Save the list of constant columns
constant_cols_df = pd.DataFrame(constant_columns, columns=['Constant_Column'])
constant_cols_df.to_csv('constant_columns.csv', index=False)
print(f"Saved list of {len(constant_columns)} constant columns to 'constant_columns.csv'")

# Also save the list of columns removed due to insufficient data
insufficient_data_cols = pd.DataFrame(removed_insufficient_columns, columns=['Removed_Due_To_Insufficient_Data'])
insufficient_data_cols.to_csv('columns_with_insufficient_data.csv', index=False)
print(f"Saved list of {len(removed_insufficient_columns)} columns removed due to insufficient data to 'columns_with_insufficient_data.csv'")

# Save statistics of remaining columns
stats_df.to_csv('column_statistics.csv', index=False)
print(f"Saved statistics of remaining columns to 'column_statistics.csv'")

# Visualize the distribution of non-constant columns (for first 10 columns)
plot_cols = [col for col in numeric_cols if col not in constant_columns][:10]  # Take first 10 for visualization
if plot_cols:
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(plot_cols):
        plt.subplot(2, 5, i+1)
        sns.histplot(df_without_constants[col], kde=True)
        plt.title(col)
        plt.tight_layout()
    plt.savefig('column_distributions.png')
    print("Saved distribution plots for first 10 non-constant columns to 'column_distributions.png'")