import pandas as pd

# Load the CSV files
csv_file1 = pd.read_csv("d:\\User Data\\Oshadi\\USJ\\Acedemic\\3rd Year\\Sem 6\\Project\\Test1\\clean_and_encoding\\dataset\\2018_dataset.csv",encoding='latin1')
csv_file2 = pd.read_csv("d:\\User Data\\Oshadi\\USJ\\Acedemic\\3rd Year\\Sem 6\\Project\\Test1\\clean_and_encoding\\dataset\\2019_dataset.csv",encoding='latin1')
csv_file3 = pd.read_csv("d:\\User Data\\Oshadi\\USJ\\Acedemic\\3rd Year\\Sem 6\\Project\\Test1\\clean_and_encoding\\dataset\\2020_dataset.csv",encoding='latin1')
csv_file4 = pd.read_csv("d:\\User Data\\Oshadi\\USJ\\Acedemic\\3rd Year\\Sem 6\\Project\\Test1\\clean_and_encoding\\dataset\\2021_dataset.csv",encoding='latin1')
csv_file5 = pd.read_csv("d:\\User Data\\Oshadi\\USJ\\Acedemic\\3rd Year\\Sem 6\\Project\\Test1\\clean_and_encoding\\dataset\\2022_dataset.csv",encoding='latin1')
csv_file6 = pd.read_csv("d:\\User Data\\Oshadi\\USJ\\Acedemic\\3rd Year\\Sem 6\\Project\\Test1\\clean_and_encoding\\dataset\\2023_dataset.csv",encoding='latin1')

# Concatenate the files
combined_csv = pd.concat([csv_file1, csv_file2,csv_file3, csv_file4,csv_file5, csv_file6], ignore_index=True)

# Save to a new CSV file
combined_csv.to_csv("d:\\User Data\\Oshadi\\USJ\\Acedemic\\3rd Year\\Sem 6\\Project\\Test1\\clean_and_encoding\\dataset\\newdataset.csv", index=False)

print("CSV files have been combined successfully!")
