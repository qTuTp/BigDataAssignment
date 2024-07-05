import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d  # For smoothing data
from matplotlib.ticker import FuncFormatter  # Import FuncFormatter for custom formatting

# Load data for total cases by age category
dataCaseM = pd.read_csv("dataset/cases_malaysia.csv")
dataCaseS = pd.read_csv("dataset/cases_state.csv")

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

# Calculate total cases by state
dataCaseS['cases_new'] = pd.to_numeric(dataCaseS['cases_new'], errors='coerce')
state_total_cases = dataCaseS.groupby('state')['cases_new'].sum().reset_index()
state_total_cases = state_total_cases.sort_values(by='cases_new', ascending=False)

data_agg = dataCaseS.groupby('state')['cases_active'].sum().reset_index()
data_agg.sort_values(by='cases_active', ascending=False, inplace=True)
smoothed_cases = gaussian_filter1d(data_agg['cases_active'], sigma=2)

# Extracting data for plotting
states = state_total_cases['state']
total_cases = state_total_cases['cases_new']

# Plotting combined charts
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

# Bar chart on ax1
categories = list(totals.keys())
totals_values = list(totals.values())
ax1.bar(categories, totals_values, color=['blue', 'green', 'orange', 'red'])
ax1.set_xlabel('Age Categories')
ax1.set_ylabel('Total Number of Cases (in millions)')
ax1.set_title('COVID-19 Case Distribution Across Age Groups')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Function to format y-axis values in millions
def millions_formatter(x, pos):
    return f'{x / 1e6:.0f}M'

ax1.yaxis.set_major_formatter(FuncFormatter(millions_formatter))

# Pie chart on ax2
labels = list(proportions.keys())
sizes = list(proportions.values())
ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
ax2.set_title('Proportion of Active COVID-19 Cases by Age Category')
ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Line chart on ax3
smoothed_cases = gaussian_filter1d(data_agg['cases_active'], sigma=2)
ax3.plot(states, smoothed_cases, marker='o', linestyle='-', color='b', linewidth=2)
ax3.set_title('Active COVID-19 Cases by State')
ax3.set_xlabel('State')
ax3.set_ylabel('Active Cases (in millions)')
ax3.grid(True)
ax3.tick_params(axis='x', rotation=45)

ax3.yaxis.set_major_formatter(FuncFormatter(millions_formatter))

# Heatmap on ax4
heatmap = ax4.imshow([total_cases], cmap='YlOrRd', aspect='auto')
ax4.set_xticks(np.arange(len(states)))
ax4.set_xticklabels(states, rotation=45)
ax4.set_xlabel('State')
ax4.set_ylabel('Total Cases (in millions)')
ax4.set_title('Total COVID-19 Cases by State')
fig.colorbar(heatmap, ax=ax4, label='Total Cases (in millions)')

ax4.yaxis.set_major_formatter(FuncFormatter(millions_formatter))

plt.tight_layout()
plt.show()
