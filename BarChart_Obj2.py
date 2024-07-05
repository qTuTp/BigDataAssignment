import pandas as pd
import matplotlib.pyplot as plt
from Database import getDatabase

# Connect to the database and retrieve the collections
db = getDatabase()
collVaxM = db["vaxMalaysia"]

# Retrieve data from MongoDB collections and load into pandas DataFrame
dataVaxM = pd.DataFrame(list(collVaxM.find()))

# Preprocess the data
# Convert dates to datetime format
dataVaxM['date'] = pd.to_datetime(dataVaxM['date'])

# Create new attributes for cumulative vaccinations for adolescents and children
dataVaxM['cumulative_adol'] = (dataVaxM['daily_partial_adol'] +
                               dataVaxM['daily_full_adol'] +
                               dataVaxM['daily_booster_adol'] +
                               dataVaxM['daily_booster2_adol']).cumsum()

dataVaxM['cumulative_child'] = (dataVaxM['daily_partial_child'] +
                                dataVaxM['daily_full_child'] +
                                dataVaxM['daily_booster_child'] +
                                dataVaxM['daily_booster2_child']).cumsum()

# Aggregate cumulative vaccinations for the final count
total_vax = dataVaxM[['cumulative_adol', 'cumulative_child']].max()

# Plot the data
plt.figure(figsize=(8, 6))
bar_width = 0.35
index = ['Adolescents', 'Children']

bars = plt.bar(index, total_vax, bar_width, color=['skyblue', 'lightgreen'])

# plt.xlabel('Age Group')
plt.ylabel('Total Cumulative Vaccinations (in millions)')
plt.title('Total Cumulative COVID-19 Vaccinations by Adolescents and Children in Malaysia')
plt.tight_layout()

# Customize the y-axis tick labels to show values in millions
ax = plt.gca()
ax.set_yticklabels(['{:.1f}M'.format(x / 1e6) for x in ax.get_yticks()])

# Annotate bars with cumulative values
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:,}', ha='center', va='bottom', fontsize=12, color='black')

plt.show()
