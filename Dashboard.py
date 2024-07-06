import streamlit as st
import pandas as pd
from Database import getDatabase
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import logging
from scipy.ndimage import gaussian_filter1d  # For smoothing data
import numpy as np
from matplotlib.ticker import FuncFormatter  # Import FuncFormatter for custom formatting

def millions_formatter(x, pos):
    return f'{x / 1e6:.0f}M'

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,handlers=[
        logging.FileHandler("dashboard.log"),  # Log to this file
        logging.StreamHandler()          # Optional: log to the console too
    ])

    # Get Database and collection
    dbName = getDatabase()
    logging.info("Connected to the database")
    colVaxState = dbName["vaxState"]
    colVaxMalaysia = dbName["vaxMalaysia"]
    colCaseState = dbName["caseState"]
    colCaseMalaysia = dbName["caseMalaysia"]
    colCaseVaxMalaysia = dbName["caseVaxMalaysia"]
    
    vaxStateDF = pd.DataFrame(list(colVaxState.find()))
    vaxMalaysiaDF = pd.DataFrame(list(colVaxMalaysia.find()))
    caseStateDF = pd.DataFrame(list(colCaseState.find()))
    caseMalaysiaDF = pd.DataFrame(list(colCaseMalaysia.find()))
    caseVaxMalaysiaDF = pd.DataFrame(list(colCaseVaxMalaysia.find()))
    
    st.title("Covid-19 Analysis")
    st.markdown("Below is the analysis result on covid-19")
    st.sidebar.title("Select Visual Charts")
    st.sidebar.markdown("Please select the graph accordingly")
    
    chartDetail = {
        "Line Chart":["Active Covid-19 By State", "New Covid-19 Cases, Recoveries and Vaccination in 2021",
                      "New Covid-19 Cases, Recoveries and Vaccination in 2022","New Covid-19 Cases, Recoveries and Vaccination in 2023",
                      "New Covid-19 Cases, Recoveries and Vaccination in 2024", "New Covid-19 Cases and Recoveries Overall",
                      "Vaccination Stage for adolescent and child", "Covid-19 Active Cases and Cummulative Vaccination in Overall"],
        "Bar Chart":["Covid-19 Case Distribution across age group", "Daily Vaccination by State", "Covid-19 Cases by state in age group", 
                     "Vaccination by State for adolescent and child"],
        "Scatter Plot":["Covid-19 Cases vs Daily Vaccination","Linear Regression for Daily Covid-19 Cases and Amount of Vaccination - Testing", 
                        "Linear Regression for Daily Covid-19 Cases and Amount of Vaccination - Training"],
        "Pie Chart":["Active Covid-19 Case by age group", "Distribution of Vaccine Type"],
        "Heat Map":["Total Case by State"]
    }
    
    chartVisual = st.sidebar.selectbox('Select Charts/Plot type', ('Line Chart', 'Bar Chart', 'Scatter Plot', "Pie Chart", "Heat Map"))
    logging.info(chartVisual)
    selectedChart = st.sidebar.selectbox("Select Chart",chartDetail[chartVisual])
    st.header(selectedChart)
    
    
    if chartVisual == "Line Chart":
        if selectedChart == "Active Covid-19 By State":
            # Renaming columns for easier access
            caseMalaysiaDF.rename(columns={
                'cases_child': 'child',
                'cases_adolescent': 'adolescent',
                'cases_adult': 'adult',
                'cases_elderly': 'elderly'
            }, inplace=True)

            # Calculate total cases for each category
            totals = {
                'child': caseMalaysiaDF['child'].sum(),
                'adolescent': caseMalaysiaDF['adolescent'].sum(),
                'adult': caseMalaysiaDF['adult'].sum(),
                'elderly': caseMalaysiaDF['elderly'].sum()
            }

            # Calculate total active cases
            total_active_cases = caseMalaysiaDF['cases_active'].sum()

            # Calculate proportions of active cases in each category
            proportions = {
                'child': caseMalaysiaDF['cases_active'].sum() / totals['child'],
                'adolescent': caseMalaysiaDF['cases_active'].sum() / totals['adolescent'],
                'adult': caseMalaysiaDF['cases_active'].sum() / totals['adult'],
                'elderly': caseMalaysiaDF['cases_active'].sum() / totals['elderly']
            }

            # Calculate total cases by state
            caseStateDF['cases_new'] = pd.to_numeric(caseStateDF['cases_new'], errors='coerce')
            state_total_cases = caseStateDF.groupby('state')['cases_new'].sum().reset_index()
            state_total_cases = state_total_cases.sort_values(by='cases_new', ascending=False)

            data_agg = caseStateDF.groupby('state')['cases_active'].sum().reset_index()
            data_agg.sort_values(by='cases_active', ascending=False, inplace=True)
            smoothed_cases = gaussian_filter1d(data_agg['cases_active'], sigma=2)

            # Extracting data for plotting
            states = state_total_cases['state']
            total_cases = state_total_cases['cases_new']

            # Line chart on ax
            # fig, ax = plt.subplots()
            smoothed_cases = gaussian_filter1d(data_agg['cases_active'], sigma=2)
            # ax.plot(states, smoothed_cases, marker='o', linestyle='-', color='b', linewidth=2)
            # ax.set_title('Active COVID-19 Cases by State')
            # ax.set_xlabel('State')
            # ax.set_ylabel('Active Cases (in millions)')
            # ax.grid(True)
            # ax.tick_params(axis='x', rotation=45)
            
            # st.pyplot(fig)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=states, y=smoothed_cases,
                                 mode='lines', name='Active Cases'))
            
            fig.update_layout(title=selectedChart,
                      xaxis_title='States',
                      yaxis_title='Active Cases')
            
            st.plotly_chart(fig)



        elif selectedChart == "New Covid-19 Cases, Recoveries and Vaccination in 2021":
            caseMalaysiaDF['date'] = pd.to_datetime(caseMalaysiaDF['date'])
            vaxMalaysiaDF['date'] = pd.to_datetime(vaxMalaysiaDF['date'])
            data = pd.merge(caseMalaysiaDF, vaxMalaysiaDF, on='date')
            data = data.sort_values(by='date')
            
            data_2021 = data[data['date'].dt.year == 2021]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=data_2021["date"], y=data_2021["cases_new"],
                                 mode='lines', name='New Cases'))
            fig.add_trace(go.Scatter(x=data_2021["date"], y=data_2021["cases_recovered"],
                                 mode='lines', name='Recovered Cases'))
            fig.add_trace(go.Scatter(x=data_2021["date"], y=data_2021["daily"],
                                 mode='lines', name='Vaccination'))
            fig.update_layout(title=selectedChart,
                      xaxis_title='Date',
                      yaxis_title='Number')
            
            st.plotly_chart(fig)
        elif selectedChart == "New Covid-19 Cases, Recoveries and Vaccination in 2022":
            caseMalaysiaDF['date'] = pd.to_datetime(caseMalaysiaDF['date'])
            vaxMalaysiaDF['date'] = pd.to_datetime(vaxMalaysiaDF['date'])
            data = pd.merge(caseMalaysiaDF, vaxMalaysiaDF, on='date')
            data = data.sort_values(by='date')
            
            data_2022 = data[data['date'].dt.year == 2022]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=data_2022["date"], y=data_2022["cases_new"],
                                 mode='lines', name='New Cases'))
            fig.add_trace(go.Scatter(x=data_2022["date"], y=data_2022["cases_recovered"],
                                 mode='lines', name='Recovered Cases'))
            fig.add_trace(go.Scatter(x=data_2022["date"], y=data_2022["daily"],
                                 mode='lines', name='Vaccination'))
            fig.update_layout(title=selectedChart,
                      xaxis_title='Date',
                      yaxis_title='Number')
            
            st.plotly_chart(fig)
        elif selectedChart == "New Covid-19 Cases, Recoveries and Vaccination in 2023":
            caseMalaysiaDF['date'] = pd.to_datetime(caseMalaysiaDF['date'])
            vaxMalaysiaDF['date'] = pd.to_datetime(vaxMalaysiaDF['date'])
            data = pd.merge(caseMalaysiaDF, vaxMalaysiaDF, on='date')
            data = data.sort_values(by='date')
            
            data_2023 = data[data['date'].dt.year == 2023]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=data_2023["date"], y=data_2023["cases_new"],
                                 mode='lines', name='New Cases'))
            fig.add_trace(go.Scatter(x=data_2023["date"], y=data_2023["cases_recovered"],
                                 mode='lines', name='Recovered Cases'))
            fig.add_trace(go.Scatter(x=data_2023["date"], y=data_2023["daily"],
                                 mode='lines', name='Vaccination'))
            fig.update_layout(title=selectedChart,
                      xaxis_title='Date',
                      yaxis_title='Number')
            
            st.plotly_chart(fig)
        elif selectedChart == "New Covid-19 Cases, Recoveries and Vaccination in 2024":
            caseMalaysiaDF['date'] = pd.to_datetime(caseMalaysiaDF['date'])
            vaxMalaysiaDF['date'] = pd.to_datetime(vaxMalaysiaDF['date'])
            data = pd.merge(caseMalaysiaDF, vaxMalaysiaDF, on='date')
            data = data.sort_values(by='date')
            
            data_2024 = data[data['date'].dt.year == 2024]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=data_2024["date"], y=data_2024["cases_new"],
                                 mode='lines', name='New Cases'))
            fig.add_trace(go.Scatter(x=data_2024["date"], y=data_2024["cases_recovered"],
                                 mode='lines', name='Recovered Cases'))
            fig.add_trace(go.Scatter(x=data_2024["date"], y=data_2024["daily"],
                                 mode='lines', name='Vaccination'))
            fig.update_layout(title=selectedChart,
                      xaxis_title='Date',
                      yaxis_title='Number')
            
            st.plotly_chart(fig)
            
        elif selectedChart == "Covid-19 Active Cases and Cummulative Vaccination in Overall":
            caseMalaysiaDF['date'] = pd.to_datetime(caseMalaysiaDF['date'])
            vaxMalaysiaDF['date'] = pd.to_datetime(vaxMalaysiaDF['date'])
            data = pd.merge(caseMalaysiaDF, vaxMalaysiaDF, on='date')
            data = data.sort_values(by='date')
                        
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=data["date"], y=data["cases_active"],
                                 mode='lines', name='Active Cases'))
            fig.add_trace(go.Scatter(x=data["date"], y=data["cumul_partial"],
                                 mode='lines', name='Cummulative Partial Vaccination'))
            fig.add_trace(go.Scatter(x=data["date"], y=data["cumul_full"],
                                 mode='lines', name='Cummulative Full Vaccination'))
            fig.add_trace(go.Scatter(x=data["date"], y=data["cumul_booster"],
                                 mode='lines', name='Cummulative Booster Vaccination'))
            fig.add_trace(go.Scatter(x=data["date"], y=data["cumul_booster2"],
                                 mode='lines', name='Cummulative Booster 2 Vaccination'))

            fig.update_layout(title=selectedChart,
                      xaxis_title='Date',
                      yaxis_title='Number')
            
            st.plotly_chart(fig)
        elif selectedChart == "New Covid-19 Cases and Recoveries Overall":
            caseMalaysiaDF['date'] = pd.to_datetime(caseMalaysiaDF['date'])
            vaxMalaysiaDF['date'] = pd.to_datetime(vaxMalaysiaDF['date'])
            data = pd.merge(caseMalaysiaDF, vaxMalaysiaDF, on='date')
            data = data.sort_values(by='date')
            data_interpolated = data.interpolate(method='linear')
            dates = data_interpolated['date']
            new_cases_caseM = data_interpolated['cases_new']
            recoveries_vaxM = data_interpolated['cases_recovered']
            # vaccination_vaxM = data_interpolated['daily']
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=dates, y=new_cases_caseM,
                                 mode='lines', name='New Cases'))
            fig.add_trace(go.Scatter(x=dates, y=recoveries_vaxM,
                                 mode='lines', name='Recovered Cases'))
            # fig.add_trace(go.Scatter(x=dates, y=vaccination_vaxM,
            #                      mode='lines', name='Vaccination'))
            fig.update_layout(title=selectedChart,
                      xaxis_title='Date',
                      yaxis_title='Number')
            
            st.plotly_chart(fig)
            # fig, ax = plt.subplots()
            
            # ax.plot(dates, new_cases_caseM, label='New Cases (caseMalaysia)', color='blue')
            # ax.plot(dates, recoveries_vaxM, label='Recovered Cases (vaxMalaysia)', color='green')
            # ax.set_xlabel('Date')
            # ax.set_ylabel('Cases')
            # ax.set_title('Smoothed Line Chart: COVID-19 Cases Over Time')
            # ax.legend()
            # ax.tight_layout()
            
            # st.pyplot(fig)
        elif selectedChart == "Vaccination Stage for adolescent and child":
            fig = go.Figure()
            # Extract relevant data for the stages of vaccination for adolescents and children
            stages = ['Partial', 'Full', 'Booster', 'Booster2']
            cumul_adol = [
                vaxMalaysiaDF['cumul_partial_adol'].max(),
                vaxMalaysiaDF['cumul_full_adol'].max(),
                vaxMalaysiaDF['cumul_booster_adol'].max(),
                vaxMalaysiaDF['cumul_booster2_adol'].max()
            ]

            cumul_child = [
                vaxMalaysiaDF['cumul_partial_child'].max(),
                vaxMalaysiaDF['cumul_full_child'].max(),
                vaxMalaysiaDF['cumul_booster_child'].max(),
                vaxMalaysiaDF['cumul_booster2_child'].max()
            ]

            plot_data = pd.DataFrame({
                'Stage': stages,
                'Adolescents': cumul_adol,
                'Children': cumul_child
            })
            
            fig.add_trace(go.Scatter(x=plot_data['Stage'], y=plot_data['Adolescents'],
                                 mode='lines', name='Adolescents'))
            fig.add_trace(go.Scatter(x=plot_data['Stage'], y=plot_data['Children'],
                                 mode='lines', name='Children'))
            fig.update_layout(title=selectedChart,
                      xaxis_title='Vaccination Stage',
                      yaxis_title='Number of Vaccinations')
            
            st.plotly_chart(fig)
    elif chartVisual == "Bar Chart":
        pass
    elif chartVisual == "Scatter Plot":
        pass
    elif chartVisual == "Pie Chart":
        pass
    elif chartVisual == "Heat Map":
        if selectedChart == "Total Case by State":
            caseStateDF['cases_new'] = pd.to_numeric(caseStateDF['cases_new'], errors='coerce')
            state_total_cases = caseStateDF.groupby('state')['cases_new'].sum().reset_index()
            state_total_cases = state_total_cases.sort_values(by='cases_new', ascending=False)

            states = state_total_cases['state']
            total_cases = state_total_cases['cases_new']
            
            fig, ax = plt.subplots(figsize=(18, 12))
            
            heatmap = ax.imshow([total_cases], cmap='YlOrRd', aspect='auto')
            ax.set_xticks(np.arange(len(states)))
            ax.set_xticklabels(states, rotation=45)
            ax.set_xlabel('State')
            ax.set_ylabel('Total Cases (in millions)')
            ax.set_title('Total COVID-19 Cases by State')
            fig.colorbar(heatmap, ax=ax, label='Total Cases (in millions)')
            
            ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))

            
            st.pyplot(fig)
        
    
