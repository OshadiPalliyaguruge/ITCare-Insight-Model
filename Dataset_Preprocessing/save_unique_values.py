import pandas as pd

dataset="dataset\\New dataset\\incident_report_preprocessed_final_98000_cleaned.csv"
output_file = "Dataset_Summary.csv"

# Read the csv
df = pd.read_csv(dataset, encoding='latin1')

# Initialize an empty list to store the rows
rows = []

# Loop through each column in the DataFrame
for column in df.columns:
    unique_values = df[column].unique()
    unique_count = df[column].nunique()
    
    # Print the column information
    print(f"Column: {column}")
    print(f"Total Unique Values: {unique_count}")
    if unique_count > 20:  # Limit to first 20 unique values if too many
        unique_values_str = ', '.join(map(str, unique_values[:20])) + ", ... (and more)"
    else:
        unique_values_str = ', '.join(map(str, unique_values))
    
    print(f"Unique Values: {unique_values_str}")
    print("-" * 40)
    
    # Add the information to the rows list
    rows.append({
        'Column': column,
        'Total Unique Values': unique_count,
        'Unique Values': unique_values_str
    })

# Convert the list of rows into a DataFrame
summary_df = pd.DataFrame(rows)

# Save the DataFrame to a CSV file
summary_df.to_csv(output_file, index=False)

print(f"Summary has been saved to '{output_file}'.")
