#%% Import
import matplotlib.pyplot as plt
import pandas as pd
from Database import getDatabase

#%%
# Connect to the database and retrieve the collections
dbName = getDatabase()
collection = dbName["vaxState"]

# Retrieve data from MongoDB collections and load into pandas DataFrame
vaxStateDF = pd.DataFrame(list(collection.find()))
vaxStateDF = vaxStateDF.drop(columns="_id")


#%% Print Detail
vaxStateDF.info()
vaxStateDF.describe()

#%%
# Group the data by state and sum the cases for each age group
stateVax = vaxStateDF.groupby('state').sum()

# Extract the data and sum
adol_data = stateVax[['daily_partial_adol', 'daily_full_adol', 'daily_booster_adol', 'daily_booster2_adol']]
child_data = stateVax[['daily_partial_child', 'daily_full_child', 'daily_booster_child', 'daily_booster2_child']]

adol_data['total_adol'] = adol_data.sum(axis=1)
child_data['total_child'] = child_data.sum(axis=1)

plot_data = pd.DataFrame({
    'total_adol': adol_data['total_adol'],
    'total_child': child_data['total_child']
})

#%% Print detail
print(adol_data)

#%%
# Plot the data
plot_data.plot(kind='bar', stacked=True, figsize=(10, 7))

# Add labels and title
plt.xlabel('State')
plt.ylabel('Number of Vaccinations(millions)')
plt.title('Vaccinations by State - Adolescent and Child')
plt.legend(['Adolescent', 'Child'])
plt.xticks(rotation=90)  # Rotate state labels for better readability

# Show the plot
plt.tight_layout()
plt.show()

# %%
