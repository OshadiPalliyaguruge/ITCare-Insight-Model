import pandas as pd


dataset="d:\\User Data\\Oshadi\\USJ\\Acedemic\\3rd Year\\Sem 6\\Project\\Test1\\clean_and_encoding\\dataset\\New dataset\\incident_report_preprocessed_final_98000_cleaned.csv"

# Read the csv 
df = pd.read_csv(dataset, encoding='latin1') # if csv file in different directory add with correct path


# Print first 5 rows  
print(df.head(),"\n")
#see last 5 rows 
print(df.tail(),"\n")

print(df.shape) #see number of rows and columns
num_rows = df.shape[0]
print("No of Rows: " + str(df.shape[0])+"\n")
num_columns = df.shape[1]
print("Number of columns in the dataset:", num_columns,"\n")

# Print column names 
print(df.columns,"\n")

# Print info on data types, non-nulls, memory usage etc.
print(df.info(),"\n")  

# Display summary statistics  
# Set display to show all rows and columns  
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Print full summary statistics description
print(df.describe()) 


# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())   

# Check duplicates
print("\nNumber of duplicate rows:")   
print(df.duplicated().sum())


# Loop through each column in the DataFrame
for column in df.columns:
    unique_values = df[column].unique()
    unique_count = df[column].nunique()
    
    print(f"Column: {column}")
    print(f"Total Unique Values: {unique_count}")
    print("-" * 40)


    # Print all unique values or limit to first N values if too many
    if unique_count > 20:  # You can change this limit based on your needs
        print(unique_values[:20])  # Print first 20 unique values only
        print("... (and more)")
    else:
        print(unique_values)
    
    print("-" * 40)