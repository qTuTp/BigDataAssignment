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
# Reset index to start from 0
caseMalaysiaDF = caseMalaysiaDF.reset_index(drop=True)

caseMalaysiaDF['sequentialDay'] = caseMalaysiaDF.index + 1

# %% Save processed case malatsia into mongoDB
collectionCaseMalaysiaClean = dbName["caseMalaysiaClean"]
caseMalaysiaJSON = json.loads(caseMalaysiaDF.to_json(orient="records"))
collectionCaseMalaysiaClean.delete_many({})
collectionCaseMalaysiaClean.insert_many(caseMalaysiaJSON)

#######################################################################################

# %%Preprocessing for vax malaysia
# Get Case Malaysia
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
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['cumul_partial'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['cumul_full'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['cumul_booster'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['cumul_booster2'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['cumul'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['cumul_partial_adol'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['cumul_full_adol'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['cumul_booster_adol'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['cumul_booster2_adol'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['cumul_partial_child'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['cumul_full_child'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['cumul_booster_child'])
vaxMalaysiaDF = vaxMalaysiaDF.drop(columns=['cumul_booster2_child'])



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
# Reset index to start from 0
vaxMalaysiaDF = vaxMalaysiaDF.reset_index(drop=True)

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

