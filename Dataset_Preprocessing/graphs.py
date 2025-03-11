import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = 'D:\\User Data\\Oshadi\\USJ\\Acedemic\\3rd Year\\Sem 6\\Project\\Test1\\clean_and_encoding\\dataset\\New dataset\\incident_report_preprocessed_final_98000_cleaned.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Convert "Submit Date" to datetime, handling potential 12-hour and 24-hour formats
data['Submit Date'] = pd.to_datetime(data['Submit Date'], errors='coerce', format='%d/%m/%Y %I:%M:%S %p')
data['Submit Date'] = data['Submit Date'].fillna(pd.to_datetime(data['Submit Date'], errors='coerce', format='%d/%m/%Y %H:%M'))

# Drop rows with unparsed dates if any remain
data = data.dropna(subset=['Submit Date'])

# Extract date parts for analysis
data['Year'] = data['Submit Date'].dt.year
data['Month'] = data['Submit Date'].dt.month_name()
data['Weekday'] = data['Submit Date'].dt.day_name()

# Count total issues and issues reported through "Remedy" in the "Submitter" column
total_issues = len(data)
remedy_issues = data[data['Submitter'].str.contains("Remedy", case=False, na=False)].shape[0]

# Calculate percentage of Remedy issues
remedy_percentage = (remedy_issues / total_issues) * 100
other_percentage = 100 - remedy_percentage

# Plotting
labels = ['Remedy', 'Other']
sizes = [remedy_percentage, other_percentage]
colors = ['#66b3ff', '#ff9999']
explode = (0.1, 0)  # "explode" the first slice

# 1. Yearly Trend Analysis
plt.figure(figsize=(10, 6))
yearly_counts = data['Year'].value_counts().sort_index()
sns.lineplot(x=yearly_counts.index, y=yearly_counts.values, marker='o')
plt.title("Yearly Trend of Incident Submissions")
plt.xlabel("Year")
plt.ylabel("Number of Incidents")
plt.xticks(yearly_counts.index, rotation=45)
plt.tight_layout()
plt.savefig('yearly_trend.png', transparent=True)  # Save with transparent background
plt.show()

# 2. Monthly Trend Analysis (across all years)
plt.figure(figsize=(12, 6))
monthly_counts = data['Month'].value_counts().reindex([
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
])
sns.barplot(x=monthly_counts.index, y=monthly_counts.values, palette="coolwarm")
plt.title("Monthly Distribution of Incident Submissions")
plt.xlabel("Month")
plt.ylabel("Number of Incidents")
plt.tight_layout()
plt.savefig('monthly_trend.png', transparent=True)  # Save with transparent background
plt.show()

# 3. Day of the Week Analysis
plt.figure(figsize=(10, 6))
weekday_counts = data['Weekday'].value_counts().reindex([
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
])
sns.barplot(x=weekday_counts.index, y=weekday_counts.values, palette="magma")
plt.title("Day of the Week Distribution of Incident Submissions")
plt.xlabel("Day of the Week")
plt.ylabel("Number of Incidents")
plt.tight_layout()
plt.savefig('weekday_distribution.png', transparent=True)  # Save with transparent background
plt.show()

# Pie chart for Remedy vs Other issues
plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=140)
plt.title('Percentage of Issues Reported by Remedy Application vs. Other Methods')
plt.savefig('remedy_vs_other_issues.png', transparent=True)  # Save with transparent background
plt.show()
