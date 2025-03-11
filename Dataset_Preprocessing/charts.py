import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('D:\\User Data\\Oshadi\\USJ\\Acedemic\\3rd Year\\Sem 6\\Project\\Test1\\clean_and_encoding\\dataset\\New dataset\\incident_report_preprocessed_final_98000_cleaned.csv')

plt.style.use('ggplot')  # or 'default', 'classic', etc.
sns.set_palette("Set2")

# Descriptive Analysis Outputs
# Plot for Priority Distribution
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Priority')
plt.title('Distribution of Incidents by Priority')
plt.xlabel('Priority')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot for Status Distribution
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Status')
plt.title('Distribution of Incidents by Status')
plt.xlabel('Status')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Trend Analysis Outputs
# Convert 'Submit Date' to datetime
df['Submit Date'] = pd.to_datetime(df['Submit Date'], errors='coerce')
# Filter out rows with invalid dates
df = df.dropna(subset=['Submit Date'])
# Grouping by month and year
df['Month'] = df['Submit Date'].dt.to_period('M')
monthly_trends = df.groupby('Month').size()

# Plotting trend over time
plt.figure(figsize=(14, 7))
monthly_trends.plot()
plt.title('Incident Trends Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Incidents')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Categorical Analysis Outputs
# Incident Types Distribution
plt.figure(figsize=(14, 6))
sns.countplot(y='Operational Categorization Tier 1', data=df, 
              order=df['Operational Categorization Tier 1'].value_counts().index)
plt.title('Most Common Incident Types')
plt.xlabel('Count')
plt.ylabel('Incident Type')
plt.tight_layout()
plt.show()

# Resolution Time Analysis Outputs
# Convert 'Resolution Time' if 'Resolved Date' column is available
if 'Resolved Date' in df.columns:
    df['Resolved Date'] = pd.to_datetime(df['Resolved Date'], errors='coerce')
    df['Resolution Time'] = (df['Resolved Date'] - df['Submit Date']).dt.days

    # Average resolution time by Priority
    resolution_time_priority = df.groupby('Priority')['Resolution Time'].mean().dropna()

    # Plotting
    plt.figure(figsize=(10, 6))
    resolution_time_priority.plot(kind='bar')
    plt.title('Average Resolution Time by Priority')
    plt.xlabel('Priority')
    plt.ylabel('Resolution Time (days)')
    plt.tight_layout()
    plt.show()

# Heatmap of Incident Status by Department Outputs
# Pivot table for heatmap
heatmap_data = df.pivot_table(index='Department', columns='Status', aggfunc='size', fill_value=0)

# Heatmap plot
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu")
plt.title('Incident Status by Department')
plt.xlabel('Status')
plt.ylabel('Department')
plt.tight_layout()
plt.show()

