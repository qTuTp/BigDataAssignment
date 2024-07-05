import pandas as pd
import matplotlib.pyplot as plt
from Database import getDatabase

# Connect to the database and retrieve the collections
db = getDatabase()
collVaxM = db["vaxMalaysia"]

# Retrieve data from MongoDB collections and load into pandas DataFrame
dataVaxM = pd.DataFrame(list(collVaxM.find()))

# Create cumulative attributes for each type of vaccination
dataVaxM['cumulative_pfizer'] = (dataVaxM['pfizer1'] +
                                 dataVaxM['pfizer2'] +
                                 dataVaxM['pfizer3'] +
                                 dataVaxM['pfizer4']).cumsum()

dataVaxM['cumulative_sinovac'] = (dataVaxM['sinovac1'] +
                                  dataVaxM['sinovac2'] +
                                  dataVaxM['sinovac3'] +
                                  dataVaxM['sinovac4']).cumsum()

dataVaxM['cumulative_astra'] = (dataVaxM['astra1'] +
                                dataVaxM['astra2'] +
                                dataVaxM['astra3'] +
                                dataVaxM['astra4']).cumsum()

dataVaxM['cumulative_sinopharm'] = (dataVaxM['sinopharm1'] +
                                    dataVaxM['sinopharm2'] +
                                    dataVaxM['sinopharm3'] +
                                    dataVaxM['sinopharm4']).cumsum()

dataVaxM['cumulative_cansino'] = (dataVaxM['cansino'] +
                                  dataVaxM['cansino3'] +
                                  dataVaxM['cansino4']).cumsum()

# Aggregate the total cumulative vaccinations for each type
total_vax = dataVaxM[['cumulative_pfizer',
                      'cumulative_sinovac',
                      'cumulative_astra',
                      'cumulative_sinopharm',
                      'cumulative_cansino']].max()

# Define the labels and colors for the pie chart
labels = ['Pfizer', 'Sinovac', 'AstraZeneca', 'Sinopharm', 'CanSino']
colors = ['#8BC1F7', '#BDE2B9', '#A2D9D9', '#F0AB00', '#B2B0EA']

# Plot the pie chart
plt.figure(figsize=(10, 8))
wedges, texts = plt.pie(total_vax, startangle=140, colors=colors)

# Add a custom legend with names and percentages
percentages = [f'{label} ({value/sum(total_vax)*100:.1f}%)' for label, value in zip(labels, total_vax)]
plt.legend(wedges, percentages, title="Vaccine Types", loc="best")

plt.title('Distribution of COVID-19 Vaccinations by Type in Malaysia')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.tight_layout()
plt.show()
