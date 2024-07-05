import matplotlib.pyplot as plt
import pandas as pd
from Database import getDatabase

# Connect to the database and retrieve the collections
db = getDatabase()
collCaseS = db["caseState"]

# Retrieve data from MongoDB collections and load into pandas DataFrame
dataCaseS = pd.DataFrame(list(collCaseS.find()))

# Group the data by state and sum the cases for each age group
state_cases = dataCaseS.groupby('state')[['cases_child', 'cases_adolescent', 'cases_adult', 'cases_elderly']].sum().reset_index()

# Rename columns for clarity
state_cases.rename(columns={
    'cases_child': 'totalCases_child',
    'cases_adolescent': 'totalCases_adolescent',
    'cases_adult': 'totalCases_adult',
    'cases_elderly': 'totalCases_elderly'
}, inplace=True)

# Plot the data
plt.figure(figsize=(12, 8))

# Define the positions of the bars on the x-axis
states = state_cases['state']
bar_width = 0.5

# Plot each age group as a bar stack
p1 = plt.bar(states, state_cases['totalCases_child'], bar_width, label='Child', color='skyblue')
p2 = plt.bar(states, state_cases['totalCases_adolescent'], bar_width, bottom=state_cases['totalCases_child'], label='Adolescent', color='lightgreen')
p3 = plt.bar(states, state_cases['totalCases_adult'], bar_width, bottom=state_cases['totalCases_child'] + state_cases['totalCases_adolescent'], label='Adult', color='orange')
p4 = plt.bar(states, state_cases['totalCases_elderly'], bar_width, bottom=state_cases['totalCases_child'] + state_cases['totalCases_adolescent'] + state_cases['totalCases_adult'], label='Elderly', color='red')

# Add titles and labels
plt.title('Total Number of COVID-19 Cases in Each State by Age Group')
plt.xlabel('State')
plt.ylabel('Total Cases (in millions)')
plt.xticks(rotation=90)  # Rotate state names for better readability
plt.legend()

# Customize the y-axis tick labels to show values in millions
ax = plt.gca()
ax.set_yticklabels(['{:.1f}M'.format(x / 1e6) for x in ax.get_yticks()])

# Adjust layout to fit labels
plt.tight_layout()

# Show plot
plt.show()
