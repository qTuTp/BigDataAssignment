from Database import getDatabase
import pandas as pd
import json

dbName = getDatabase()

"""
collCaseM: Collection for cases in whole malaysia
collCaseS: Collection for cases in state
collVaxM: Collection for Vaccination in Malaysia
collVaxS: Collection for Vaccination in state
collPop: Collection for Population of Malaysia
"""

collCaseM = dbName["caseMalaysia"]
collCaseS = dbName["caseState"]
collVaxM = dbName["vaxMalaysia"]
collVaxS = dbName["vaxState"]
collPop = dbName["population"]

dataCaseM = pd.read_csv("dataset/cases_malaysia.csv")
dataCaseS = pd.read_csv("dataset/cases_state.csv")
dataVaxM = pd.read_csv("dataset/vax_malaysia.csv")
dataVaxS = pd.read_csv("dataset/vax_state.csv")
dataPop = pd.read_csv("dataset/population.csv")

caseM = json.loads(dataCaseM.to_json(orient='records'))  
caseS = json.loads(dataCaseS.to_json(orient='records'))  
vaxM = json.loads(dataVaxM.to_json(orient='records'))  
vaxS = json.loads(dataVaxS.to_json(orient='records'))  
pop = json.loads(dataPop.to_json(orient='records'))  

collCaseM.delete_many({})
collCaseS.delete_many({})
collVaxM.delete_many({})
collVaxS.delete_many({})
collPop.delete_many({})

collCaseM.insert_many(caseM)
collCaseS.insert_many(caseS)
collVaxM.insert_many(vaxM)
collVaxS.insert_many(vaxS)
collPop.insert_many(pop)