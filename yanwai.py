import matplotlib.pyplot as plt
import pandas as pd
from Database import getDatabase

# Function to format y-axis labels in millions
def millions_formatter(x, pos):
    return '{:.1f}M'.format(x)

# Get database connection
db = getDatabase()

# Fetch data for state vaccinations
collVaxS = db["vaxState"]
dataVaxS = pd.DataFrame(list(collVaxS.find()))
dataVaxS['date'] = pd.to_datetime(dataVaxS['date'])
dataVaxS = dataVaxS.sort_values(by='date')
state_vaccinations = dataVaxS.groupby('state')['daily'].sum().reset_index()
state_vaccinations['daily_millions'] = state_vaccinations['daily'] / 1e6
colors = ['skyblue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# Plotting the bar chart with custom colors
plt.figure(figsize=(12, 8))
bars = plt.bar(state_vaccinations['state'], state_vaccinations['daily_millions'], color=colors[:len(state_vaccinations)])
plt.xlabel('State')
plt.ylabel('Total Daily Vaccinations (Millions)')
plt.title('Total Daily Vaccinations by State')
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(millions_formatter))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Fetch data for COVID-19 cases and vaccinations in Malaysia
collCaseM = db["caseMalaysia"]
collVaxM = db["vaxMalaysia"]
dataCaseM = pd.DataFrame(list(collCaseM.find()))
dataVaxM = pd.DataFrame(list(collVaxM.find()))
dataCaseM['date'] = pd.to_datetime(dataCaseM['date'])
dataVaxM['date'] = pd.to_datetime(dataVaxM['date'])
data = pd.merge(dataCaseM, dataVaxM, on='date')
data = data.sort_values(by='date')

# Create smoothed line chart
data_interpolated = data.interpolate(method='linear')
dates = data_interpolated['date']
new_cases_caseM = data_interpolated['cases_new']
recoveries_vaxM = data_interpolated['cases_recovered']
plt.figure(figsize=(10, 6))
plt.plot(dates, new_cases_caseM, label='New Cases (caseMalaysia)', color='blue')
plt.plot(dates, recoveries_vaxM, label='Recovered Cases (vaxMalaysia)', color='green')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.title('Smoothed Line Chart: COVID-19 Cases Over Time')
plt.legend()
plt.tight_layout()
plt.savefig('smoothed_cases_chart.png')
plt.show()

# Create scatter plot for daily vaccinations vs new COVID-19 cases
x = data['daily'].values
y = data['cases_new'].values
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', alpha=0.7)
plt.xlabel('Daily Vaccinations')
plt.ylabel('New COVID-19 Cases')
plt.title('Scatter Plot: New COVID-19 Cases vs Daily Vaccinations')
plt.tight_layout()
plt.show()

# Create plots for each year (2021-2024) showing COVID-19 cases, recoveries, and vaccination rates
data_2021 = data[data['date'].dt.year == 2021]
data_2022 = data[data['date'].dt.year == 2022]
data_2023 = data[data['date'].dt.year == 2023]
data_2024 = data[data['date'].dt.year == 2024]

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(data_2021['date'], data_2021['cases_new'], label='New Cases', color='blue')
plt.plot(data_2021['date'], data_2021['cases_recovered'], label='Recovered Cases', color='green')
plt.plot(data_2021['date'], data_2021['daily'], label='Daily Vaccination', color='red')
plt.xlabel('Date')
plt.ylabel('Number of Cases / Vaccination Rate')
plt.title('2021: COVID-19 Cases, Recoveries, and Vaccination Rate in Malaysia')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(data_2022['date'], data_2022['cases_new'], label='New Cases', color='blue')
plt.plot(data_2022['date'], data_2022['cases_recovered'], label='Recovered Cases', color='green')
plt.plot(data_2022['date'], data_2022['daily'], label='Daily Vaccination', color='red')
plt.xlabel('Date')
plt.ylabel('Number of Cases / Vaccination Rate')
plt.title('2022: COVID-19 Cases, Recoveries, and Vaccination Rate in Malaysia')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(data_2023['date'], data_2023['cases_new'], label='New Cases', color='blue')
plt.plot(data_2023['date'], data_2023['cases_recovered'], label='Recovered Cases', color='green')
plt.plot(data_2023['date'], data_2023['daily'], label='Daily Vaccination', color='red')
plt.xlabel('Date')
plt.ylabel('Number of Cases / Vaccination Rate')
plt.title('2023: COVID-19 Cases, Recoveries, and Vaccination Rate in Malaysia')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(data_2024['date'], data_2024['cases_new'], label='New Cases', color='blue')
plt.plot(data_2024['date'], data_2024['cases_recovered'], label='Recovered Cases', color='green')
plt.plot(data_2024['date'], data_2024['daily'], label='Daily Vaccination', color='red')
plt.xlabel('Date')
plt.ylabel('Number of Cases / Vaccination Rate')
plt.title('2024: COVID-19 Cases, Recoveries, and Vaccination Rate in Malaysia')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
