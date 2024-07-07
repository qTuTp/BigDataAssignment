import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d  # For smoothing data
from Database import getDatabase

# Load and preprocess data for age categories
dbName = getDatabase()

# dataCaseM = pd.read_csv("dataset/cases_malaysia.csv")

colCaseMalaysia = dbName["caseMalaysia"]
dataCaseM = pd.DataFrame(list(colCaseMalaysia.find()))

# Renaming columns for easier access
dataCaseM.rename(columns={
    'cases_child': 'child',
    'cases_adolescent': 'adolescent',
    'cases_adult': 'adult',
    'cases_elderly': 'elderly'
}, inplace=True)

# Calculate total cases for each category
totals = {
    'child': dataCaseM['child'].sum(),
    'adolescent': dataCaseM['adolescent'].sum(),
    'adult': dataCaseM['adult'].sum(),
    'elderly': dataCaseM['elderly'].sum()
}

# Calculate total active cases
total_active_cases = dataCaseM['cases_active'].sum()

# Calculate proportions of active cases in each category
proportions = {
    'child': dataCaseM['cases_active'].sum() / totals['child'],
    'adolescent': dataCaseM['cases_active'].sum() / totals['adolescent'],
    'adult': dataCaseM['cases_active'].sum() / totals['adult'],
    'elderly': dataCaseM['cases_active'].sum() / totals['elderly']
}

# Load and preprocess data for active cases by state
# dataCaseS = pd.read_csv("dataset/cases_state.csv")

colCaseState = dbName["caseState"]
dataCaseS = pd.DataFrame(list(colCaseState.find()))
dataCaseS['cases_active'] = pd.to_numeric(dataCaseS['cases_active'], errors='coerce')
data_agg = dataCaseS.groupby('state')['cases_active'].sum().reset_index()
data_agg.sort_values(by='cases_active', ascending=False, inplace=True)
smoothed_cases = gaussian_filter1d(data_agg['cases_active'], sigma=2)

# Load and preprocess data for recovery cases by state
dataRecovery = pd.DataFrame(list(colCaseState.find()))
dataRecovery['date'] = pd.to_datetime(dataRecovery['date'])
dataRecovery['year'] = dataRecovery['date'].dt.year
pivot_data = dataRecovery.pivot_table(index='year', columns='state', values='cases_recovered', aggfunc='sum')
state_order = pivot_data.sum().sort_values().index
pivot_data = pivot_data[state_order]

# Create combined figure with multiple subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot total cases by age category (ax1)
categories = list(totals.keys())
totals_values = list(totals.values())
ax1.bar(categories, totals_values, color=['blue', 'green', 'orange', 'red'])
ax1.set_xlabel('Age Categories')
ax1.set_ylabel('Total Number of Cases')
ax1.set_title('Total Cases by Age Category')
ax1.grid(axis='y', linestyle='--', alpha=0.7)


# Plot proportion of active cases by age category (ax2)
labels = list(proportions.keys())
sizes = list(proportions.values())
ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
ax2.set_title('Proportion of Active COVID-19 Cases by Age Category')
ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.


# Plot smoothed active cases by state (ax3)
ax3.plot(data_agg['state'], smoothed_cases, marker='o', linestyle='-', color='b')
ax3.set_xticks(data_agg.index)
ax3.set_xticklabels(data_agg['state'], rotation=90)
ax3.set_xlabel('State')
ax3.set_ylabel('Smoothed Active Cases')
ax3.set_title('Active COVID-19 Cases by State')
ax3.grid(True)


# Plot stacked area for recovery cases by state (ax4)
colors = plt.cm.tab20c(range(len(state_order)))
pivot_data.plot(kind='area', stacked=True, figsize=(12, 8), color=colors, ax=ax4)
ax4.set_title('State Recovery Cases Over Time (2020-2024)')
ax4.set_xlabel('Year')
ax4.set_ylabel('Recovery Cases (millions)')
ax4.legend(title='State', loc='upper left', bbox_to_anchor=(1, 1))
ax4.set_xticks([2020, 2021, 2022, 2023, 2024])

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
