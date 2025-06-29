import pandas as pd
import numpy as np
from tabulate import tabulate

# Load the datasets
# df_fixed = pd.read_csv("fixed_4Q_experiments_dataframe.csv")
# df_trainable = pd.read_csv("trainable_2R_experiments_dataframe.csv")
df_rbf = pd.read_csv("rbf_learns_quantum_dataframe.csv")

# --- Standardization ---
# Create a common 'kernel_type' column for easy grouping and comparison.
# df_fixed["kernel_type"] = "Fixed (Pauli Search)"
# df_trainable["kernel_type"] = "Trainable (Re-uploading)"
df_rbf["kernel_type"] = "RBF"

# Combine the two dataframes.
# `pd.concat` handles the different columns automatically, filling missing values with NaN.
# df_combined = pd.concat([df_fixed, df_trainable], ignore_index=True)

# Select and reorder columns for clarity. This drops the experiment-specific columns.
common_columns = [
    "kernel_type",
    "model_name",
    "quantum_data_entanglment",
    "noise_percent",
    "train_size",
    "test_accuracy",
    "generalization_gap",
    "training_time_sec",
]
# df = df_combined[common_columns].copy()
df = df_rbf.copy()
print(df)

print("Data successfully loaded and merged.")
print(f"Total rows in combined dataframe: {len(df)}")
print("\nUnique values for key parameters:")
print(f"Kernel Types: {df['kernel_type'].unique()}")
# print(f"RBF Gamma values: {sorted(df['rbf_gamma'].unique())}")
print(f"Noise levels: {sorted(df['noise_percent'].unique())}")

# df_combined.to_csv("combined.csv", index=False)

# Filter for the final performance at the largest training size
df_final = df[df["train_size"] == 4000].copy()

# We only care about the performance of the quantum models for this comparison
# df_final_qsvc = df_final[df_final["model_name"] == "Quantum_SVM"].copy()
df_final_qsvc = df_final[df_final["model_name"] == "RBF_SVM"].copy()

# Group by the experimental conditions and calculate the mean and std dev
accuracy_summary = (
    df_final_qsvc.groupby(["quantum_data_entanglment", "noise_percent", "kernel_type"])[
        "test_accuracy"
    ]
    .agg(["mean", "std"])
    .unstack()
)


# Helper function to format the output nicely
def format_mean_std(mean, std):
    return f"{mean:.3f} ± {std:.3f}"


# Create an empty DataFrame to store our final, formatted strings
formatted_summary = pd.DataFrame(index=accuracy_summary.index)

# Get the unique model names (e.g., 'Fixed Kernel', 'Trainable Kernel')
# from the second level of the column MultiIndex
model_names = accuracy_summary.columns.get_level_values(1).unique()

# Loop through each model type
for model in model_names:
    # Select the Series of mean values for the current model
    means = accuracy_summary[("mean", model)]

    # Select the Series of standard deviation values for the current model
    stds = accuracy_summary[("std", model)]

    # Use the formatting function to combine the mean and std into a new column
    # in our formatted_summary DataFrame.
    formatted_summary[model] = np.vectorize(format_mean_std)(means, stds)

print("\n--- Aggregated Final Test Accuracy ---")
print(formatted_summary.to_markdown())


# Step 1: Restructure the DataFrame for Hierarchical Display
presentation_df = formatted_summary.reset_index()

# Replace redundant gamma values with empty strings for a 'merged cell' look
for i in range(1, len(presentation_df)):
    if (
        presentation_df.loc[i, "quantum_data_entanglment"]
        == presentation_df.loc[i - 1, "quantum_data_entanglment"]
    ):
        presentation_df.loc[i, "quantum_data_entanglment"] = ""

# Rename columns for the final table header
presentation_df.rename(
    columns={
        "quantum_data_entanglment": "Entaglement Strategy",
        "noise_percent": "Noise (%)",
    },
    inplace=True,
)

# # Step 2: Add bold formatting for the "winner" in each row
# for index, row in presentation_df.iterrows():
#     try:
#         # Extract numerical mean values for comparison, stripping LaTeX characters
#         fixed_mean = float(row["Fixed (Pauli Search)"].split(" ")[0])
#         trainable_mean = float(row["Trainable (Re-uploading)"].split(" ")[0])

#         if trainable_mean > fixed_mean:
#             presentation_df.loc[index, "Trainable (Re-uploading)"] = (
#                 f"\\textbf{{{row['Trainable (Re-uploading)']}}}"
#             )
#         else:
#             presentation_df.loc[index, "Fixed (Pauli Search)"] = (
#                 f"\\textbf{{{row['Fixed (Pauli Search)']}}}"
#             )
#     except (ValueError, IndexError) as e:
#         print(f"Could not parse row {index} for bolding: {e}")

# Step 3: Generate and print the LaTeX table using tabulate
print("\n--- LaTeX Code for Publication-Ready Table ---")
latex_table = tabulate(
    presentation_df,
    headers="keys",
    tablefmt="latex_booktabs",  # Use the booktabs format for professional tables
    showindex=False,
    colalign=("center",) * len(presentation_df.columns),
)
print(latex_table)

# --- End of New/Modified Section ---
