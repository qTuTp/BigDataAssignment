import pandas as pd
import matplotlib.pyplot as plt
from Database import getDatabase

# Connect to the database and retrieve the collections
db = getDatabase()
collVaxM = db["vaxMalaysia"]

# Retrieve data from MongoDB collections and load into pandas DataFrame
dataVaxM = pd.DataFrame(list(collVaxM.find()))

# Extract relevant data for the stages of vaccination for adolescents and children
stages = ['Partial', 'Full', 'Booster', 'Booster2']
cumul_adol = [
    dataVaxM['cumul_partial_adol'].max(),
    dataVaxM['cumul_full_adol'].max(),
    dataVaxM['cumul_booster_adol'].max(),
    dataVaxM['cumul_booster2_adol'].max()
]

cumul_child = [
    dataVaxM['cumul_partial_child'].max(),
    dataVaxM['cumul_full_child'].max(),
    dataVaxM['cumul_booster_child'].max(),
    dataVaxM['cumul_booster2_child'].max()
]

# Plot the line chart
plt.figure(figsize=(12, 8))
plt.plot(stages, cumul_adol, marker='o', label='Adolescents')
plt.plot(stages, cumul_child, marker='o', label='Children')
plt.title('Total Number of Each Vaccination Stage for Adolescents and Children')
plt.xlabel('Vaccination Stage')
plt.ylabel('Number of Vaccinations (in millions)')
plt.legend()
plt.grid(True)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
plt.show()
