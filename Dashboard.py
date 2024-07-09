import streamlit as st
import pandas as pd
from Database import getDatabase
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import logging
from scipy.ndimage import gaussian_filter1d  # For smoothing data
import numpy as np
from matplotlib.ticker import FuncFormatter  # Import FuncFormatter for custom formatting
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

toRead="""
#################################TO READ!!!##############################
-To run the dashboard, please type 'streamlit run ./Dashboard.py' in the terminal and enter

-In case you encountered issues related to uninstall library, please type 'pip install -r ./requirements.txt' and run in the terminal 
to install the required dependencies.

#########################################################################

"""
del toRead

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
    population = dbName["population"]
    
    vaxStateDF = pd.DataFrame(list(colVaxState.find()))
    vaxMalaysiaDF = pd.DataFrame(list(colVaxMalaysia.find()))
    caseStateDF = pd.DataFrame(list(colCaseState.find()))
    caseMalaysiaDF = pd.DataFrame(list(colCaseMalaysia.find()))
    caseVaxMalaysiaDF = pd.DataFrame(list(colCaseVaxMalaysia.find()))
    popDF = pd.DataFrame(list(population.find()))
    
    st.title("Covid-19 Analysis")
    st.markdown("Below is the analysis result on covid-19")
    st.sidebar.title("Select Visual Charts")
    st.sidebar.markdown("Please select the graph accordingly")
    
    chartDetail = {
        "Line Chart":["Active Covid-19 By State", "New Covid-19 Cases, Recoveries and Vaccination in 2021",
                      "New Covid-19 Cases, Recoveries and Vaccination in 2022","New Covid-19 Cases, Recoveries and Vaccination in 2023",
                      "New Covid-19 Cases, Recoveries and Vaccination in 2024", "New Covid-19 Cases and Recoveries Overall",
                      "Vaccination Stage for adolescent and child", "Covid-19 Active Cases and Cummulative Vaccination in Overall"],
        "Bar Chart":["Covid-19 Case Distribution across age group", "Total Vaccination by State", "Covid-19 Cases by state in age group", 
                     "Vaccination by State for adolescent and child", "Population For each state"],
        "Scatter Plot":["Covid-19 Cases vs Daily Vaccination"],
        "Pie Chart":["Active Covid-19 Case by age group", "Distribution of Vaccine Type", "Population For Each State", "Population of age group"],
        "Area Chart":["State Recovery Cases Over Time (2020-2024)"],
        "Linear Regression":["Linear Regression for Daily Covid-19 Cases and Cummulative Vaccination"]
    }
    
    chartVisual = st.sidebar.selectbox('Select Charts/Plot type', ('Line Chart', 'Bar Chart', 'Scatter Plot', "Pie Chart", "Area Chart", "Linear Regression"))
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
            
            st.markdown("The Active COVID-19 by State line chart provides a thorough visual depiction of the current COVID-19 case distribution and trends throughout Malaysia's states. Understanding the regional dynamics of the pandemic and determining the intensity of outbreaks in various geographic locations depend heavily on this graph.")



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
            
            st.markdown("The first year's vaccine effort reaches its peaks in July 2021 with 600,000 doses provided. This peak indicates that an aggressive first vaccination programme to stop the spread of COVID-19 is required. Simultaneously, the annual total of new and recovered COVID-19 cases stays low and constant at less than 5,000. ")
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
            
            st.markdown("The second year has a peak of 250,000 vaccinations in March 2022, followed by a sharp decrease to less than 5,000 by July 2022. March 2022 shows a little increase in new and recovered cases at this time, but they still fall short of 10,000. This slight increase in incidence in spite of the vaccination campaign's decrease indicates that, even if the vaccination programme slowed, a serious outbreak was still avoided.")
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
            
            st.markdown("The number of vaccinations in 2023 peaks at 12,000 in January and falls to less than 1,000 by May. November 2023 shows an increase in new and recovered cases throughout the year, eventually reaching 6,000. There may be a relationship between the decrease in vaccination efforts and the rise in new infections and recoveries, as evidenced by the coincidence between the increase in cases and the decline in vaccination rates towards the end of the year.")
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
            
            st.markdown("In January 2024, there are 3,000 new and recovered cases. However, at February 2024 the total has dropped dramatically to less than 500. In 2024, vaccination rates stay below 500 throughout the entire year. Despite the low vaccination rates, the decrease in cases at the start of the year implies that efficient control measures were reinstated. ")
            
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
            
            st.markdown("From the chart, it clear that majority of people has token partial and full vaccination, while only half amount of them has taken booster vaccination, and booster 2 vaccination haven’t taken by a lot of people. The vaccination process was initiated by the partial vaccination and full vaccination, then follow with booster and then booster 2 vaccination. From this graph, it shows that there a peak of active cases around September 2021 and the active cases drops, this is likely due to the amount of people has taken the partial and full vaccination, helping to reduce the number of active cases. There is another peak of active cases in around March 2022, but the peak also drop quickly as there is also a rise of number of people who has taken partial vaccination and full vaccination just before the peak. This shows the effect of vaccination has start to develop. As Majority of people in Malaysia has taken partial and full vaccination, there is no more sudden rise in active in the dataset.")
            
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
            
            st.markdown("The Trends in New and Recovered COVID-19 Cases line chart offers an in-depth analysis of the changes in officially reported COVID-19 cases and recoveries between July 2020 and January 2024. The x-axis tracks trends over the period July 2020 to January 2024. The y-axis displays the total number of  cases, with separate lines for newly identified COVID-19 cases and recovered cases. ")
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
            
            st.markdown("The line chart above depicts the cumulative vaccinations in various states for adolescents and children based on the vaxMalaysia dataset. The x-axis represents the stages of vaccination: First, Second, First booster dose and Second booster dose for people are on the x-axis. The number of vaccinations is shown in millions on the y-axis. ")
    elif chartVisual == "Bar Chart":
        if selectedChart == "Covid-19 Case Distribution across age group":
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
            
            categories = list(totals.keys())
            totals_values = list(totals.values())
            fig1 = px.bar(x=categories, y=totals_values, labels={'x': 'Age Categories', 'y': 'Total Number of Cases'},
                        title='COVID-19 Case Distribution Across Age Groups', color=categories)
            fig1.update_yaxes(tickformat=',.1f', title='Total Number of Cases')
            st.plotly_chart(fig1)
            
            st.markdown("The Covid-19 Cases Distribution Across Age Groups bar chart provides a thorough overview of the distribution of COVID-19 cases among various age groups in the Malaysian population. The four main age groups that are the subject of this visualization are children, adolescents, adults, and the elderly. ")

            
        elif selectedChart == "Total Vaccination by State":
            vaxStateDF['date'] = pd.to_datetime(vaxStateDF['date'])
            vaxStateDF = vaxStateDF.sort_values(by='date')
            state_vaccinations = vaxStateDF.groupby('state')['daily'].sum().reset_index()
            state_vaccinations['daily_millions'] = state_vaccinations['daily']

            # Streamlit layout for state vaccinations bar chart
            st.subheader('Total Vaccinations by State')
            state_chart = px.bar(state_vaccinations, x='state', y='daily_millions', 
                                labels={'daily_millions': 'Total Vaccinations'}, 
                                title='Total Vaccinations by State')
            st.plotly_chart(state_chart)
            
            st.markdown("The Total COVID-19 Vaccinations by State in Malaysia bar chart offers a comparison of the vaccinations programmes in Malaysia's various states. The states of Malaysia are presented on the x-axis, while the total number of vaccines given is displayed on the y-axis. ")
            
        elif selectedChart == "Covid-19 Cases by state in age group":
            caseStateDF['date'] = pd.to_datetime(caseStateDF['date'])
            state_cases = caseStateDF.groupby('state')[['cases_child', 'cases_adolescent', 'cases_adult', 'cases_elderly']].sum().reset_index()
            state_cases.rename(columns={
                'cases_child': 'totalCases_child',
                'cases_adolescent': 'totalCases_adolescent',
                'cases_adult': 'totalCases_adult',
                'cases_elderly': 'totalCases_elderly'
            }, inplace=True)
            
            # fig, ax = plt.subplots(figsize=(12, 8))
            # # Define the positions of the bars on the x-axis
            # states = state_cases['state']
            # bar_width = 0.5

            # # Plot each age group as a bar stack
            # p1 = plt.bar(states, state_cases['totalCases_child'], bar_width, label='Child', color='skyblue')
            # p2 = plt.bar(states, state_cases['totalCases_adolescent'], bar_width, bottom=state_cases['totalCases_child'], label='Adolescent', color='lightgreen')
            # p3 = plt.bar(states, state_cases['totalCases_adult'], bar_width, bottom=state_cases['totalCases_child'] + state_cases['totalCases_adolescent'], label='Adult', color='orange')
            # p4 = plt.bar(states, state_cases['totalCases_elderly'], bar_width, bottom=state_cases['totalCases_child'] + state_cases['totalCases_adolescent'] + state_cases['totalCases_adult'], label='Elderly', color='red')
            
            # ax.set_title('Total Number of COVID-19 Cases in Each State by Age Group')
            # ax.set_xlabel('State')
            # ax.set_ylabel('Total Cases (in millions)')
            # ax.set_xticklabels(states, rotation=90)  # Rotate state names for better readability
            # ax.legend()
            # # Customize the y-axis tick labels to show values in millions
            # ax.set_yticklabels(['{:.1f}M'.format(x / 1e6) for x in ax.get_yticks()])
            # fig.tight_layout()
            
            # st.pyplot(fig)
            # Melt the dataframe for easier plotting with Plotly
            state_cases_melted = state_cases.melt(id_vars='state', value_vars=['totalCases_child', 'totalCases_adolescent', 'totalCases_adult', 'totalCases_elderly'],
                                                var_name='Age Group', value_name='Total Cases')

            # Create the plotly bar chart
            fig = px.bar(state_cases_melted, 
                        x='state', 
                        y='Total Cases', 
                        color='Age Group', 
                        title='Total Number of COVID-19 Cases in Each State by Age Group',
                        labels={'state': 'State', 'Total Cases': 'Total Cases'},
                        height=600)

            # Customize the y-axis to show values in millions
            fig.update_yaxes(tickformat=',.1f')

            # Rotate x-axis labels for better readability
            fig.update_layout(xaxis={'categoryorder':'total descending', 'tickangle':-90})

            # Show the plot in Streamlit
            st.plotly_chart(fig)
            
            st.markdown("The stacked bar chart above visualizes the total number of COVID-19 cases in each state of Malaysia, categorized by four different age groups. The age groups are child, adolescent, adult, and elderly. The x-axis shows the states, while the y-axis shows the total number of COVID-19 cases. The age groups are represented by different colors such as blue for children, light blue for adolescents, red for adults, and light red for elderly people. ")
            
        elif selectedChart == "Vaccination by State for adolescent and child":
            # Group the data by state and sum the cases for each age group
            stateVax = vaxStateDF.groupby('state')[['daily_partial_adol', 'daily_full_adol', 'daily_booster_adol', 'daily_booster2_adol','daily_partial_child', 'daily_full_child', 'daily_booster_child', 'daily_booster2_child']].sum().reset_index()
            
            # Extract the data and sum
            adol_data = stateVax[['daily_partial_adol', 'daily_full_adol', 'daily_booster_adol', 'daily_booster2_adol']]
            child_data = stateVax[['daily_partial_child', 'daily_full_child', 'daily_booster_child', 'daily_booster2_child']]

            adol_data['total_adol'] = adol_data.sum(axis=1)
            child_data['total_child'] = child_data.sum(axis=1)
            

            # # Extract the data and sum
            # adol_data = stateVax[['daily_partial_adol', 'daily_full_adol', 'daily_booster_adol', 'daily_booster2_adol']]
            # child_data = stateVax[['daily_partial_child', 'daily_full_child', 'daily_booster_child', 'daily_booster2_child']]

            # adol_data['total_adol'] = adol_data.sum(axis=1)
            # child_data['total_child'] = child_data.sum(axis=1)

            plot_data = pd.DataFrame({
                'State': stateVax["state"],
                'Adolescent': adol_data['total_adol'],
                'Child': child_data['total_child']
            })
            
            state_vax_melted = plot_data.melt(id_vars='State', value_vars=['Adolescent', 'Child'],
                                                var_name='Age Group', value_name='Total Vaccination')

            # Plot using Plotly
            st.subheader('Vaccinations by State - Adolescent and Child')
            fig = px.bar(state_vax_melted, x='State', y="Total Vaccination", 
                        labels={'Total Vaccination': 'Number of Vaccinations'}, 
                        color='Age Group',
                        title='Vaccinations by State - Adolescent and Child',
                        barmode='stack')
            st.plotly_chart(fig)
            
            st.markdown("In some state such as Johor, Sarawak and Selangor have almost half amount of vaccination is taken by child even when referencing to Pie Chart Population For each State, the children population only take around 20%. This could be due to children is more vulnerable toward covid-19, hence they are prioritized for vaccination.")
        elif selectedChart == "Population For each state":
            
            
            # Remove Malaysia and non state
            popDF = popDF[popDF['state'] != 'Malaysia']
            popDF = popDF[popDF['state'] != 'W.P. Labuan']
            popDF = popDF[popDF['state'] != 'W.P. Putrajaya']
            popDF = popDF[popDF['state'] != 'Klang Valley']
            
            # Melt the DataFrame to have age categories as a single column
            popMelt = popDF.melt(id_vars=["state", "pop"], value_vars=["pop_18", "pop_60", "pop_12", "pop_5"],
                                var_name="age_category", value_name="population")
            
            ageCatMapping = {
                "pop_60": "60+",
                "pop_18": "18+",
                "pop_12": "12-17",
                "pop_5": "5-11"
            }
            
            popMelt["age_category"] = popMelt["age_category"].map(ageCatMapping)
            
            # Create the stacked bar chart using Plotly
            fig = px.bar(popMelt, x="state", y="population", color="age_category", title="Population by Age Category in Each State",
                        labels={"population": "Population", "state": "State", "age_category": "Age Category"})
            
            st.plotly_chart(fig)
            
            st.markdown("From the chart, the state that have the highest population in Malaysia is Selangor while the state with the lowest population is Perlis. It is clear that the age of majority of the population in every state is 18 – 59, while 60+, 12-17 and 5-11 is the minority age group.")
            
    elif chartVisual == "Scatter Plot":
        if selectedChart == "Covid-19 Cases vs Daily Vaccination":
            data = pd.merge(caseMalaysiaDF, vaxMalaysiaDF, on='date')
            data = data.sort_values(by='date')
            
            # Create scatter plot for daily vaccinations vs new COVID-19 cases
            # Streamlit layout for scatter plot
            st.subheader('New COVID-19 Cases vs Daily Vaccinations')
            scatter_chart = px.scatter(data, x='daily', y='cases_new', 
                                    labels={'daily': 'Daily Vaccinations', 'cases_new': 'New COVID-19 Cases'}, 
                                    title='New COVID-19 Cases vs Daily Vaccinations')
            st.plotly_chart(scatter_chart)
            
            st.markdown("The scatter plot New COVID-19 Cases and Daily Vaccinations shows the relationship between the daily count of new COVID-19 cases show at the y-axis and the number of COVID-19 vaccination doses shows at the x-axis. Higher values indicate more comprehensive vaccination campaigns or higher vaccination rates during a particular period. Each point on the x-axis represents a certain day's vaccination efforts. The daily total of new reported COVID-19 cases is represented by each point on the y-axis greater values relate to more new infections reported on those days.")
            
    elif chartVisual == "Pie Chart":
        if selectedChart == "Active Covid-19 Case by age group":
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
            
            proportions = {
                'child': total_active_cases / totals['child'],
                'adolescent': total_active_cases / totals['adolescent'],
                'adult': total_active_cases / totals['adult'],
                'elderly': total_active_cases / totals['elderly']
            }

            # Plot pie chart for proportions of active cases by age category
            fig = px.pie(values=list(proportions.values()), names=list(proportions.keys()), 
                        title='Proportion of Active COVID-19 Cases by Age Category', hole=0.3)
            st.plotly_chart(fig)
            
            st.markdown("The Proportion of Active COVID-19 Cases by Age Group pie chart provides a clear and informative illustration of the distribution of COVID-19 infections among four age groups: children (22.6%), adolescents (44.4%), adults (3.92%), and the elderly (29.1%). ")
            
        elif selectedChart == "Distribution of Vaccine Type":
            # Create cumulative attributes for each type of vaccination
            vaxMalaysiaDF['cumulative_pfizer'] = (vaxMalaysiaDF['pfizer1'] +
                                            vaxMalaysiaDF['pfizer2'] +
                                            vaxMalaysiaDF['pfizer3'] +
                                            vaxMalaysiaDF['pfizer4']).cumsum()

            vaxMalaysiaDF['cumulative_sinovac'] = (vaxMalaysiaDF['sinovac1'] +
                                            vaxMalaysiaDF['sinovac2'] +
                                            vaxMalaysiaDF['sinovac3'] +
                                            vaxMalaysiaDF['sinovac4']).cumsum()

            vaxMalaysiaDF['cumulative_astra'] = (vaxMalaysiaDF['astra1'] +
                                            vaxMalaysiaDF['astra2'] +
                                            vaxMalaysiaDF['astra3'] +
                                            vaxMalaysiaDF['astra4']).cumsum()

            vaxMalaysiaDF['cumulative_sinopharm'] = (vaxMalaysiaDF['sinopharm1'] +
                                                vaxMalaysiaDF['sinopharm2'] +
                                                vaxMalaysiaDF['sinopharm3'] +
                                                vaxMalaysiaDF['sinopharm4']).cumsum()

            vaxMalaysiaDF['cumulative_cansino'] = (vaxMalaysiaDF['cansino'] +
                                            vaxMalaysiaDF['cansino3'] +
                                            vaxMalaysiaDF['cansino4']).cumsum()
            
            
            # Aggregate the total cumulative vaccinations for each type
            total_vax = vaxMalaysiaDF[['cumulative_pfizer',
                                'cumulative_sinovac',
                                'cumulative_astra',
                                'cumulative_sinopharm',
                                'cumulative_cansino']].max()
            
            # Define the labels and colors for the pie chart
            labels = ['Pfizer', 'Sinovac', 'AstraZeneca', 'Sinopharm', 'CanSino']
            
            # Plot the pie chart using Plotly
            fig = px.pie(values=total_vax, names=labels, title='Distribution of COVID-19 Vaccinations by Type in Malaysia',
                        color_discrete_sequence=px.colors.qualitative.Pastel)

            # Show the plot in Streamlit
            st.plotly_chart(fig)
            
            st.markdown("The pie chart presenting the types of COVID-19 vaccines administered in Malaysia. The chart employs the use of cumulative frequencies of different vaccines to depict their proportions within the overall vaccination campaign. The available types of vaccines are Pfizer, Sinovac, AstraZeneca, CanSino, and Sinopharm.")
            
        elif selectedChart == "Population For Each State":
            # Remove Malaysia and non state
            popDF = popDF[popDF['state'] != 'W.P. Labuan']
            popDF = popDF[popDF['state'] != 'W.P. Putrajaya']
            popDF = popDF[popDF['state'] != 'Klang Valley']
            popDF = popDF[popDF['state'] != 'Malaysia']
            fig = px.pie(popDF, values="pop", names="state", title="Population For Each State")
            
            st.plotly_chart(fig)
            
            st.markdown("From this pie chart, it can clearly show the population comparison between state, allow us to know the precedence of population for each state. The order of population of state from the least to the most is:")
            
            totalPop = popDF['pop'].sum()
            popDF['percentage'] = (popDF['pop'] / totalPop) * 100
            
            
            
            # table = popDF[['state', 'percentage', 'pop']]
            
            # table.rename(columns={
            #     'state': 'State',
            #     'pop': 'Population',
            #     'percentage': 'Percentage'
            # }, inplace=True)
            popDF = popDF.sort_values(by='pop', ascending=True)
            
            table = go.Table(
                header=dict(values=['State', 'Percentage (%)', 'Population'],
                            align='left'),
                cells=dict(values=[popDF['state'], popDF['percentage'], popDF['pop']],
                        align='left')
            )
            
            fig = go.Figure(data=[table])
            
            st.plotly_chart(fig)
            
            st.markdown("The majority of population in Malaysia is from Selangor which is 20.206% with 6,555,100 people while the state with the least population is from Perlis with only 0.788%, 255,500.")
            
        elif selectedChart == "Population of age group":
            popDF = popDF[popDF['state'] == 'Malaysia']
            age_group_values = popDF[['pop_60', 'pop_18', 'pop_12', 'pop_5']].values[0]
            age_group_labels = ['60+', '18+', '12-17', '5-11']

            fig = px.pie(names=age_group_labels, values=age_group_values, title='Age Group Distribution in Malaysia')
            
            st.plotly_chart(fig)
            
            st.markdown("This chart shows that the 69.5% of people in Malaysia is between the ages of 18 – 59. This large range also result in more instances being categorised into this category. From the chart, the age group in range 5-11 is the least age group in Malaysia, with only 9.19%.")
            
        elif selectedChart == "Transmission Rate For each Age Group":
            popDF = popDF[popDF['state'] == 'Malaysia']
            
            caseMalaysiaDF['cases_18+'] = caseMalaysiaDF[['cases_18_29', 'cases_30_39', 'cases_40_49', 'cases_50_59']].sum(axis=1)
            caseMalaysiaDF['cases_60+'] = caseMalaysiaDF[['cases_60_69', 'cases_70_79', 'cases_80']].sum(axis=1)
            
            print(caseMalaysiaDF)
            print(popDF)
            # case5_11 = pd.DataFrame({
            #     "Status": ["Infected", "Not Infected"],
            #     "Amount": [caseMalaysiaDF['cases_5_11'].sum(), (popDF['pop_5'] - caseMalaysiaDF['cases_5_11'].sum())]
            # })
            # cases = {
            #     '5-11': caseMalaysiaDF['cases_5_11'].sum(),
            #     '12-17': caseMalaysiaDF['cases_12_17'].sum(),
            #     '18+': caseMalaysiaDF['cases_18+'].sum(),
            #     '60+': caseMalaysiaDF['cases_60+'].sum()
            # }
            
            # plot_data = pd.DataFrame({
            #     'Age Group': list(cases.keys()),
            #     'Cases': list(cases.values()),
            # })
            # Get the total cases per new age group
            cases = {
                '5-11': caseMalaysiaDF['cases_5_11'].sum(),
                '12-17': caseMalaysiaDF['cases_12_17'].sum(),
                '18+': caseMalaysiaDF['cases_18+'].sum(),
                '60+': caseMalaysiaDF['cases_60+'].sum(),
            }
            
            # Population data for each age group (adjust column names accordingly)
            pop_data = {
                '5-11': popDF['pop_5'].sum(),
                '12-17': popDF['pop_12'].sum(),
                '18+': popDF[['pop_18']].sum(),
                '60+': popDF[['pop_60']].sum()
            }

            # Calculate the number of not infected individuals per age group
            not_infected = {
                '5-11': popDF['pop_5'].sum() - cases['5-11'],
                '12-17': popDF['pop_12'].sum() - cases['12-17'],
                '18+': popDF['pop_18'].sum() - caseMalaysiaDF['cases_18+'].sum(),
                '60+': popDF['pop_60'].sum() - caseMalaysiaDF['cases_60+'].sum()
            }
            
            print(cases)
            print(not_infected)

            # Create DataFrames for each age group for plotting
            plot_data_5_11 = pd.DataFrame({
                'Status': ['Infected', 'Not Infected'],
                'Count': [cases['5-11'], not_infected['5-11']]
            })

            plot_data_12_17 = pd.DataFrame({
                'Status': ['Infected', 'Not Infected'],
                'Count': [cases['12-17'], not_infected['12-17']]
            })

            plot_data_18_plus = pd.DataFrame({
                'Status': ['Infected', 'Not Infected'],
                'Count': [cases['18+'], not_infected['18+']]
            })

            plot_data_60_plus = pd.DataFrame({
                'Status': ['Infected', 'Not Infected'],
                'Count': [cases['60+'], not_infected['60+']]
            })
            

            # Plotting the pie charts
            st.subheader('Age Group 5-11')
            fig_5_11 = px.pie(plot_data_5_11, values='Count', names='Status', title='COVID-19 Cases in Age Group 5-11')
            st.plotly_chart(fig_5_11)

            st.subheader('Age Group 12-17')
            fig_12_17 = px.pie(plot_data_12_17, values='Count', names='Status', title='COVID-19 Cases in Age Group 12-17')
            st.plotly_chart(fig_12_17)

            st.subheader('Age Group 18+')
            fig_18_plus = px.pie(plot_data_18_plus, values='Count', names='Status', title='COVID-19 Cases in Age Group 18+')
            st.plotly_chart(fig_18_plus)

            st.subheader('Age Group 60+')
            fig_60_plus = px.pie(plot_data_60_plus, values='Count', names='Status', title='COVID-19 Cases in Age Group 60+')
            st.plotly_chart(fig_60_plus)
            
            
    elif chartVisual == "Area Chart":
        if selectedChart == "State Recovery Cases Over Time (2020-2024)":
            # caseStateDF['cases_new'] = pd.to_numeric(caseStateDF['cases_new'], errors='coerce')
            # state_total_cases = caseStateDF.groupby('state')['cases_new'].sum().reset_index()
            # state_total_cases = state_total_cases.sort_values(by='cases_new', ascending=False)

            # states = state_total_cases['state']
            # total_cases = state_total_cases['cases_new']
            
            # fig, ax = plt.subplots(figsize=(18, 12))
            
            # heatmap = ax.imshow([total_cases], cmap='YlOrRd', aspect='auto')
            # ax.set_xticks(np.arange(len(states)))
            # ax.set_xticklabels(states, rotation=45)
            # ax.set_xlabel('State')
            # ax.set_ylabel('Total Cases (in millions)')
            # ax.set_title('Total COVID-19 Cases by State')
            # fig.colorbar(heatmap, ax=ax, label='Total Cases (in millions)')
            
            # ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
  
            # st.pyplot(fig)
            
            dataRecovery = pd.DataFrame(list(colCaseState.find()))
            dataRecovery['date'] = pd.to_datetime(dataRecovery['date'])
            dataRecovery['year'] = dataRecovery['date'].dt.year
            pivot_data = dataRecovery.pivot_table(index='year', columns='state', values='cases_recovered', aggfunc='sum')
            state_order = pivot_data.sum().sort_values().index
            pivot_data = pivot_data[state_order]

            pivot_data = dataRecovery.pivot_table(index='year', columns='state', values='cases_recovered', aggfunc='sum')
            state_order = pivot_data.sum().sort_values().index
            pivot_data = pivot_data[state_order]
            
            fig = go.Figure()
            for state in state_order:
                fig.add_trace(go.Scatter(x=pivot_data.index, y=pivot_data[state], mode='lines', stackgroup='one', name=state))

            fig.update_layout(title='State Recovery Cases Over Time (2020-2024)', xaxis_title='Year', yaxis_title='Recovery Cases')
            
            st.plotly_chart(fig)
            st.markdown("The Recovery Cases Over Time 2020 - 2024 region chart provides a thorough overview of the development of COVID-19 recovery cases throughout the states of Malaysia for five years. ")


            
    elif chartVisual == "Linear Regression":
        caseVaxMalaysiaDF['cumulative_vaccinations'] = caseVaxMalaysiaDF['daily'].cumsum()

        # X: Cumulative Vaccinations, Y: Daily New Cases
        X = caseVaxMalaysiaDF[['cumulative_vaccinations']].values
        Y = caseVaxMalaysiaDF[['cases_new']].values
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

        # Create a linear regression model
        model = LinearRegression()

        # Fit the model to the training data
        model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = model.predict(X_test)

        Y_train_pred = model.predict(X_train)
        
        # Plot the training data and the regression line using Plotly
        fig = go.Figure()

        # Scatter plot for training data
        fig.add_trace(go.Scatter(x=X_train.flatten(), y=y_train.flatten(),
                                mode='markers',
                                marker=dict(color='blue'),
                                name='Training Data'))

        # Regression line
        fig.add_trace(go.Scatter(x=X_train.flatten(), y=model.predict(X_train).flatten(),
                                mode='lines',
                                line=dict(color='red', width=3),
                                name='Regression Line'))

        # Layout
        fig.update_layout(title='Linear Regression - Training Set',
                        xaxis_title='Cumulative Vaccinations',
                        yaxis_title='Daily New Cases',
                        legend=dict(x=0, y=1, traceorder='normal'),
                        showlegend=True)

        # Show plot
        st.plotly_chart(fig)
        
        # Plot the test data and the regression line using Plotly
        fig = go.Figure()

        # Scatter plot for test data
        fig.add_trace(go.Scatter(x=X_test.flatten(), y=y_test.flatten(),
                                mode='markers',
                                marker=dict(color='blue'),
                                name='Test Data'))

        # Regression line (using training data prediction for consistency)
        fig.add_trace(go.Scatter(x=X_train.flatten(), y=model.predict(X_train).flatten(),
                                mode='lines',
                                line=dict(color='red', width=3),
                                name='Regression Line'))

        # Layout
        fig.update_layout(title='Linear Regression - Test Set',
                        xaxis_title='Cumulative Vaccinations (in millions)',
                        yaxis_title='Daily New Cases',
                        legend=dict(x=0, y=1, traceorder='normal'),
                        showlegend=True)

        # Show plot
        st.plotly_chart(fig)
        
        # Calculate and print evaluation metrics
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        st.markdown("The linear regressions above show the linear regression model of training and test sets. The regression line depicts the general trend as the cumulative of the vaccinations rises, the new cases of COVID-19 reduce. This inverse relationship is anticipated and suggests that the model is correctly picking up on how vaccinations decrease new cases.")
        
        # Create a table trace for the metrics
        metric_table = go.Table(
            header=dict(values=['Metric', 'Value'],
                        align='left'),
            cells=dict(values=[['Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)'],
                            [mae, mse, rmse]],
                    align='left')
        )

        fig = go.Figure(data=[metric_table])
        fig.update_layout(title='Performance Metrics',
                        height=300,
                        margin=dict(l=20, r=20, t=50, b=20))
        
        st.plotly_chart(fig)

        
        

    
