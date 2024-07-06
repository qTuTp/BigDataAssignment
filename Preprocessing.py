# %% Import
import pandas as pd
from Database import getDatabase
import matplotlib.pyplot as plt
import seaborn as sns
import json

# %% Get Case Malaysia
dbName = getDatabase()

collection = dbName["caseMalaysia"]
cursor = collection.find({})
documents = list(cursor)

caseMalaysiaDF = pd.DataFrame(documents)
caseMalaysiaDF = caseMalaysiaDF.drop(columns=['_id'])

# %% Print Detail
print("CASE MALAYSIA INFO: ")
caseMalaysiaDF.info()
print("CASE MALAYSIA DESCRIBE: ")
caseMalaysiaDF.describe()

print(caseMalaysiaDF.isnull().sum())

# %%Remove unuse column
caseMalaysiaDF = caseMalaysiaDF.drop(columns=['cluster_import'])
caseMalaysiaDF = caseMalaysiaDF.drop(columns=['cluster_religious'])
caseMalaysiaDF = caseMalaysiaDF.drop(columns=['cluster_community'])
caseMalaysiaDF = caseMalaysiaDF.drop(columns=['cluster_highRisk'])
caseMalaysiaDF = caseMalaysiaDF.drop(columns=['cluster_education'])
caseMalaysiaDF = caseMalaysiaDF.drop(columns=['cluster_detentionCentre'])
caseMalaysiaDF = caseMalaysiaDF.drop(columns=['cluster_workplace'])
caseMalaysiaDF = caseMalaysiaDF.drop(columns=['cases_import'])
caseMalaysiaDF = caseMalaysiaDF.drop(columns=['cases_cluster'])


# %% Print Detail
print("CASE MALAYSIA INFO: ")
caseMalaysiaDF.info()
print("CASE MALAYSIA DESCRIBE: ")
caseMalaysiaDF.describe()

# %% Plot box plot
num_columns = len(caseMalaysiaDF.columns)
num_rows = (num_columns + 2) // 3

# Plot boxplot for each column
plt.figure(figsize=(15, num_rows * 5))
skipDate = True
for i, column in enumerate(caseMalaysiaDF.columns, 1):
    if skipDate:
        skipDate = False
        continue
    plt.subplot(num_rows, 3, i)
    sns.boxplot(y=caseMalaysiaDF[column])
    plt.title(f'Boxplot of {column}')
    plt.ylabel('Value')
plt.tight_layout()
plt.show()

# %% Replace outlier with mean
skipDate = True
for column in caseMalaysiaDF.columns:
    if skipDate:
        skipDate = False
        continue
    mean_value = int(caseMalaysiaDF[column].mean())
    std_value = caseMalaysiaDF[column].std()
    upper_bound = mean_value + 3 * std_value
    
    caseMalaysiaDF.loc[caseMalaysiaDF[column] > upper_bound, column] = mean_value


# %%
print(caseMalaysiaDF)
caseMalaysiaDF.info()
caseMalaysiaDF.describe()

# %% Add sequential date

caseMalaysiaDF['sequentialDay'] = caseMalaysiaDF.index + 1

# %% Save processed case malatsia into mongoDB
collectionCaseMalaysiaClean = dbName["caseMalaysiaClean"]
caseMalaysiaJSON = json.loads(caseMalaysiaDF.to_json(orient="records"))
collectionCaseMalaysiaClean.delete_many({})
collectionCaseMalaysiaClean.insert_many(caseMalaysiaJSON)

#######################################################################################

# %%Preprocessing for vax malaysia
# Get Vax Malaysia
collection = dbName["vaxMalaysia"]
cursor = collection.find({})
documents = list(cursor)

vaxMalaysiaDF = pd.DataFrame(documents)
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['_id'])

# %% Print Detail
print("VAX MALAYSIA INFO: ")
vaxMalaysiaDF.info()
print("VAX MALAYSIA DESCRIBE: ")
vaxMalaysiaDF.describe()

print(vaxMalaysiaDF.isnull().sum())

# %%Remove unuse column
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['pfizer1'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['pfizer2'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['pfizer3'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['pfizer4'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['sinovac1'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['sinovac2'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['sinovac3'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['sinovac4'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['astra1'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['astra2'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['astra3'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['astra4'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['sinopharm1'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['sinopharm2'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['sinopharm3'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['sinopharm4'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['cansino'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['cansino3'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['cansino4'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['pending1'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['pending2'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['pending3'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['pending4'])




# %% Print Detail
print("CASE MALAYSIA INFO: ")
vaxMalaysiaDF.info()
print("CASE MALAYSIA DESCRIBE: ")
vaxMalaysiaDF.describe()

# %% Plot box plot
num_columns = len(vaxMalaysiaDF.columns)
num_rows = (num_columns + 2) // 3

# Plot boxplot for each column
plt.figure(figsize=(15, num_rows * 5))
skipDate = True
for i, column in enumerate(vaxMalaysiaDF.columns, 1):
    if skipDate:
        skipDate = False
        continue
    plt.subplot(num_rows, 3, i)
    sns.boxplot(y=vaxMalaysiaDF[column])
    plt.title(f'Boxplot of {column}')
    plt.ylabel('Value')
plt.tight_layout()
plt.show()

# %% Replace outlier with mean
skipDate = True
for column in vaxMalaysiaDF.columns:
    if skipDate:
        skipDate = False
        continue
    mean_value = int(vaxMalaysiaDF[column].mean())
    std_value = vaxMalaysiaDF[column].std()
    upper_bound = mean_value + 3 * std_value
    
    vaxMalaysiaDF.loc[vaxMalaysiaDF[column] > upper_bound, column] = mean_value


# %%
print(vaxMalaysiaDF)
vaxMalaysiaDF.info()
vaxMalaysiaDF.describe()

# %% Add sequential date

vaxMalaysiaDF['sequentialDay'] = vaxMalaysiaDF.index + 1

# %% Save processed vax malaysia into mongoDB
collectionVaxMalaysiaClean = dbName["vaxMalaysiaClean"]
vaxMalaysiaJSON = json.loads(vaxMalaysiaDF.to_json(orient="records"))
collectionVaxMalaysiaClean.delete_many({})
collectionVaxMalaysiaClean.insert_many(vaxMalaysiaJSON)

#######################################################################################
# %% Preprocessing for cases state
# Get Case State
collection = dbName["caseState"]
cursor = collection.find({})
documents = list(cursor)

caseStateDF = pd.DataFrame(documents)
caseStateDF = caseStateDF.drop(columns=['_id'])

# %% Print Detail
print("CASE MALAYSIA INFO: ")
caseStateDF.info()
print("CASE MALAYSIA DESCRIBE: ")
caseStateDF.describe()

print(caseStateDF.isnull().sum())

# %%Remove unuse column
caseStateDF = caseStateDF.drop(columns=['cases_import'])
caseStateDF = caseStateDF.drop(columns=['cases_cluster'])

# %% Print Detail
print("CASE MALAYSIA INFO: ")
caseStateDF.info()
print("CASE MALAYSIA DESCRIBE: ")
caseStateDF.describe()

# %% Plot box plot
num_columns = len(caseStateDF.columns)
num_rows = (num_columns + 2) // 3

# Plot boxplot for each column
plt.figure(figsize=(15, num_rows * 5))
skipDate = True
skipState = True
for i, column in enumerate(caseStateDF.columns, 1):
    if skipDate:
        skipDate = False
        continue
    if skipState:
        skipState = False
        continue
    plt.subplot(num_rows, 3, i)
    sns.boxplot(y=caseStateDF[column])
    plt.title(f'Boxplot of {column}')
    plt.ylabel('Value')
plt.tight_layout()
plt.show()

# %% Replace outlier with mean
skipDate = True
skipState = True
for column in caseStateDF.columns:
    if skipDate:
        skipDate = False
        continue
    if skipState:
        skipState = False
        continue
    mean_value = int(caseStateDF[column].mean())
    std_value = caseStateDF[column].std()
    upper_bound = mean_value + 3 * std_value
    
    caseStateDF.loc[caseStateDF[column] > upper_bound, column] = mean_value


# %%
print(caseStateDF)
caseStateDF.info()
caseStateDF.describe()

# %% Save processed vax malaysia into mongoDB
collectionCaseStateClean = dbName["caseStateClean"]
caseStateJSON = json.loads(caseStateDF.to_json(orient="records"))
collectionCaseStateClean.delete_many({})
collectionCaseStateClean.insert_many(caseStateJSON)

#######################################################################################

# %% Preprocessing for vax state
# Get Vax State
collection = dbName["vaxState"]
cursor = collection.find({})
documents = list(cursor)

vaxStateDF = pd.DataFrame(documents)
vaxStateDF = vaxStateDF.drop(columns=['_id'])

# %% Print Detail
print("CASE MALAYSIA INFO: ")
vaxStateDF.info()
print("CASE MALAYSIA DESCRIBE: ")
vaxStateDF.describe()

print(vaxStateDF.isnull().sum())

# %%Remove unuse column
vaxStateDF = vaxStateDF.drop(columns=['pfizer1'])
vaxStateDF = vaxStateDF.drop(columns=['pfizer2'])
vaxStateDF = vaxStateDF.drop(columns=['pfizer3'])
vaxStateDF = vaxStateDF.drop(columns=['pfizer4'])
vaxStateDF = vaxStateDF.drop(columns=['sinovac1'])
vaxStateDF = vaxStateDF.drop(columns=['sinovac2'])
vaxStateDF = vaxStateDF.drop(columns=['sinovac3'])
vaxStateDF = vaxStateDF.drop(columns=['sinovac4'])
vaxStateDF = vaxStateDF.drop(columns=['astra1'])
vaxStateDF = vaxStateDF.drop(columns=['astra2'])
vaxStateDF = vaxStateDF.drop(columns=['astra3'])
vaxStateDF = vaxStateDF.drop(columns=['astra4'])
vaxStateDF = vaxStateDF.drop(columns=['sinopharm1'])
vaxStateDF = vaxStateDF.drop(columns=['sinopharm2'])
vaxStateDF = vaxStateDF.drop(columns=['sinopharm3'])
vaxStateDF = vaxStateDF.drop(columns=['sinopharm4'])
vaxStateDF = vaxStateDF.drop(columns=['cansino'])
vaxStateDF = vaxStateDF.drop(columns=['cansino3'])
vaxStateDF = vaxStateDF.drop(columns=['cansino4'])
vaxStateDF = vaxStateDF.drop(columns=['pending1'])
vaxStateDF = vaxStateDF.drop(columns=['pending2'])
vaxStateDF = vaxStateDF.drop(columns=['pending3'])
vaxStateDF = vaxStateDF.drop(columns=['pending4'])


# %% Print Detail
print("VAX STATE INFO: ")
vaxStateDF.info()
print("VAX STATE DESCRIBE: ")
vaxStateDF.describe()

# %% Plot box plot
num_columns = len(vaxStateDF.columns)
num_rows = (num_columns + 2) // 3

# Plot boxplot for each column
plt.figure(figsize=(15, num_rows * 5))
skipDate = True
skipState = True
for i, column in enumerate(vaxStateDF.columns, 1):
    if skipDate:
        skipDate = False
        continue
    if skipState:
        skipState = False
        continue
    plt.subplot(num_rows, 3, i)
    sns.boxplot(y=vaxStateDF[column])
    plt.title(f'Boxplot of {column}')
    plt.ylabel('Value')
plt.tight_layout()
plt.show()

# %% Replace outlier with mean 
skipDate = True
skipState = True
for column in vaxStateDF.columns:
    if skipDate:
        skipDate = False
        continue
    if skipState:
        skipState = False
        continue
    mean_value = int(vaxStateDF[column].mean())
    std_value = vaxStateDF[column].std()
    upper_bound = mean_value + 3 * std_value
    
    vaxStateDF.loc[vaxStateDF[column] > upper_bound, column] = mean_value


# %%
print(vaxStateDF)
vaxStateDF.info()
vaxStateDF.describe()

# %% Save processed vax malaysia into mongoDB
collectionVaxStateClean = dbName["vaxStateClean"]
vaxStateJSON = json.loads(vaxStateDF.to_json(orient="records"))
collectionVaxStateClean.delete_many({})
collectionVaxStateClean.insert_many(vaxStateJSON)

#######################################################################################

# %% Merge Case Malaysia and Vax Malaysia
dbName = getDatabase()

colCaseMalaysia = dbName["caseMalaysiaClean"]
cursor = colCaseMalaysia.find({})
documents = list(cursor)
caseMalaysia = pd.DataFrame(documents)
caseMalaysia = caseMalaysia.drop(columns=['_id'])
caseMalaysia = caseMalaysia.drop(columns=['sequentialDay'])


colVaxMalaysia = dbName["vaxMalaysiaClean"]
cursor = colVaxMalaysia.find({})
documents = list(cursor)
vaxMalaysia = pd.DataFrame(documents)
vaxMalaysia = vaxMalaysia.drop(columns=["_id"])
vaxMalaysia = vaxMalaysia.drop(columns=["sequentialDay"])


#%%Print Case
print("Case Malaysia: ")
caseMalaysia.describe()
caseMalaysia.info()
print(caseMalaysia)

#%%Print Vax
print("\nVax Malaysia: ")
vaxMalaysia.describe()
vaxMalaysia.info()
print(vaxMalaysia)

# %% Merge dataset
caseVaxMalaysia = pd.merge(caseMalaysia, vaxMalaysia, on='date', how='inner')
caseVaxMalaysia['sequentialDay'] = caseVaxMalaysia.index + 1

#%%
print("Inner Join:")
print(caseVaxMalaysia)

# %% Save processed case vax malaysia into mongoDB
colCaseVax = dbName["caseVaxMalaysia"]
caseVaxJSON = json.loads(caseVaxMalaysia.to_json(orient="records"))
colCaseVax.delete_many({})
colCaseVax.insert_many(caseVaxJSON)

# %%
