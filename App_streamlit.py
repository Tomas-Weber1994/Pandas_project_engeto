import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from scipy.spatial.distance import squareform, pdist
from scipy import stats
import calendar
from haversine import haversine
import datetime
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# loading dataset
df = pd.read_csv("edinburgh_bikes.csv")

df.drop(["start_station_description", "end_station_description", "index", "start_station_id", "end_station_id", "ended_at"], axis = 1, inplace = True) # Nonrelevant columns to my analysis at the moment
df["started_at"] = pd.to_datetime(df["started_at"]).dt.date.astype("datetime64") # converting from object to date
df.rename(columns = {"started_at" : "date", "duration" : "rental_period"}, inplace = True)

df.groupby("date")["rental_period"].count().tail() # detection of last dates in df
help_condition = ((df["date"] >= "2021-06-01") & (df["date"] <= "2021-06-30"))

# Lists of active and inactive start stations in last month
active_start_stations = df[help_condition]["start_station_name"].unique().tolist() 
active_end_stations = df[help_condition]["end_station_name"].unique().tolist()

# check - inactive start station is also not active as end station
set(active_start_stations).symmetric_difference(set(active_end_stations))

df = df.assign(station_status = np.where((df.start_station_name.isin(active_start_stations)) & (df.end_station_name.isin(active_end_stations)), "active", "inactive"))
inactive_stations = df[df["station_status"] == "inactive"]["start_station_name"].unique().tolist()

# Rentals and returns per active station
condition = df["station_status"] == "active"
df_active = df[condition]
# df_active.groupby("date").count() - 1220 days in df

df_rentals_per_station = df_active.groupby("start_station_name")[["rental_period"]].count().reset_index().rename(columns = {"start_station_name" : "station_name", "rental_period" : "num_of_rentals"})
df_returns_per_station = df_active.groupby("end_station_name")[["rental_period"]].count().reset_index().rename(columns = {"end_station_name" : "station_name", "rental_period" : "num_of_returns"})

df_mobility = df_rentals_per_station.merge(df_returns_per_station).sort_values("num_of_rentals", ascending = False)

df_mobility["num_of_rentals_per_day"] = (df_mobility["num_of_rentals"] / 1220).round(2) # mean per day during 2018 - 2021
df_mobility["num_of_returns_per_day"] = (df_mobility["num_of_returns"] / 1220).round(2)
df_mobility["bike_mobility_per_day"] = (df_mobility["num_of_rentals_per_day"] - df_mobility["num_of_returns_per_day"]).round(2)

st.sidebar.markdown("## Bike rentals in Edinburgh")
page = st.sidebar.radio('Page', ['Rentals per station', 'Map + distance calculator', 'Rentals during 2018-2021', 'Causes of fluctuation in demand', 'Prediction'])

# Weather table for visualisation later
df_weather = pd.read_csv('edinburgh_weather.csv') # Loading of table containing weather data from Edinburgh
df_weather["vis"] = df_weather["vis"].map({"Excellent" : 4, "Good" : 3, "Average" : 2,"Poor" : 1}) 
df_weather.drop(["feels", "gust", "time"], axis = 1, inplace = True) # One information about wind and temperature is enough, concrete time is for me at the moment also not relevant

for column in df_weather[["rain", "pressure", "temp", "wind"]].columns:
    df_weather[column] = df_weather[column].apply(lambda x: x.split(" ")[0]) # Clearing values from string

for column in df_weather[["humidity","cloud"]].columns:
    df_weather[column] = df_weather[column].apply(lambda x: x.split("%")[0]) 

df_weather = df_weather.astype({"date" : "datetime64", "temp" : "int64", "wind" : "int64", "rain" : "float64", "humidity" : "int64", "cloud" : "int64", "pressure" : "int64"}) # Converting from object 
df_weather_avg = df_weather.groupby("date").mean().round(2).reset_index().rename(columns = {"temp" : "temperature", "rain" : "rain_mm", "wind" : "wind_km/h", "humidity": "humidity_in_pct", "cloud" : "clouds_in_pct", "pressure": "pressure_in_mb", "vis" : "visibility"}) # Grouping by date, calculation of average values per day

# Covid table for visualisation later
# loading of covid data - applied country filter in SQL (United Kingdom)
# Data are for whole UK, we do not have covid cases specific for Edinburg
df_covid_UK = pd.read_csv('covid19_UK.csv').dropna() # deleting NaN values in the first row
df_covid_UK["date"] = pd.to_datetime(df_covid_UK["date"]).dt.date.astype("datetime64")

# Bike rentals per day for future visualisation, grouped per station and date
df_rentals_per_day = df.groupby("date")["rental_period"].count().reset_index().rename(columns = {"rental_period" : "rentals_per_day"})
df_rentals_per_day["formated_date"] = df_rentals_per_day["date"].dt.strftime("%Y-%m")

# Merging data; using inner join - covid dataframe has only 487 records - shorter observed period
df_bikes_covid = df_rentals_per_day.drop('formated_date', axis = 1).merge(df_covid_UK, how = 'inner').rename(columns = {"confirmed": "tested_positive", "deaths" : "covid_deaths"})

condition = df_bikes_covid["tested_positive"] >= 0  # Error in dataframe - two negative values in column tested_positive --> filtered out
df_bikes_covid = df_bikes_covid[condition]

if page == 'Rentals per station':

    current_stations = df_mobility['station_name'].unique()
    station_filter = st.sidebar.multiselect('Select multiple stations:', current_stations)
    if station_filter:
        df_mobility = df_mobility[df_mobility['station_name'].isin(station_filter)]
    st.markdown("### Bike rentals per active station and day")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_mobility["station_name"], 
        y = df_mobility["num_of_rentals_per_day"],
        marker_color="blue", 
        name = "rentals"))
                        
    fig.add_trace(go.Bar(
        x=df_mobility["station_name"], 
        y = df_mobility["num_of_returns_per_day"],
        marker_color="green", 
        name = "returns"))                     
                
    fig.update_layout(
        xaxis_tickfont_size=12,
        xaxis=dict(title = 'Station name'),
        yaxis=dict(
            title='Rented bikes per day in average',
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend=dict(
            x=1,
            y=0.5,
        ),
        barmode='group',
        bargap=0.15, 
        bargroupgap=0.1,
        height = 500
    )
    fig.update_xaxes(tickangle=-45)
    # Chart is interactive, can be zoomed in

    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_mobility["station_name"], 
        y = df_mobility["num_of_rentals"], 
        marker_color="blue", 
        name = "rentals"))
                        
    fig.add_trace(go.Bar(
        x=df_mobility["station_name"], 
        y = df_mobility["num_of_returns"], 
        marker_color="green", 
        name = "returns"))                     
                
    fig.update_layout(
        xaxis_tickfont_size=12,
        xaxis=dict(title = 'Station name'),
        yaxis=dict(
            title='Rented bikes in total',
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend=dict(
            x=1,
            y=0.5,
        ),
        barmode='group',
        bargap=0.15, 
        bargroupgap=0.1,
        height = 500
    )
    fig.update_xaxes(tickangle=-45)

    # Chart is interactive, can be zoomed in
    st.markdown("### Bike rentals per active station during 2018 - 2021")
    st.plotly_chart(fig, use_container_width=True)
    st.write("")
    st.write("Please check if wide mode is enabled - click on the right corner --> settings --> wide mode.")

elif page == 'Map + distance calculator':
    df_locations = df_mobility.merge(df_active.groupby("start_station_name")[["start_station_longitude", "start_station_latitude"]].mean().reset_index().rename(columns = {"start_station_name" : "station_name", "start_station_longitude" : "longitude", "start_station_latitude" : "latitude"}))

    df_array = df_locations[["latitude", "longitude"]].to_numpy()
    dist_mat = squareform(pdist(df_array,metric=haversine))
    df_distance_matrix = pd.DataFrame(dist_mat, columns = df_locations["station_name"], index = df_locations["station_name"])

    df_distance_matrix = df_distance_matrix.round(2)

    np.fill_diagonal(df_distance_matrix.values, np.nan) 

    df_distance_matrix["closest_station"] = df_distance_matrix.loc[df_distance_matrix.index, df_distance_matrix.columns].idxmin()
    df_distance_matrix["distance_to_closest_station_in_km"] = df_distance_matrix.loc[df_distance_matrix.index, df_distance_matrix.columns].min()

    # generating a new column for each station - closest station and distance
    df_locations = df_locations.set_index("station_name").join(df_distance_matrix[["closest_station", "distance_to_closest_station_in_km"]])

    # Map printing
    df_map = df_locations.reset_index().sort_values(by='num_of_rentals_per_day', ascending=False).rename(
    columns = {'num_of_rentals_per_day' : 'Bike rentals per day',
              'num_of_returns_per_day' : 'Bike returns per day'})

    fig = px.scatter_mapbox(df_map,lat='latitude', 
                            lon='longitude', 
                            color = 'Bike rentals per day', 
                            zoom=11, 
                            height=600, 
                            hover_name='station_name', 
                            hover_data=['Bike rentals per day', 'Bike returns per day', 'closest_station', 'distance_to_closest_station_in_km'], 
                            size="Bike rentals per day", 
                            size_max = 17,
                            color_continuous_scale=px.colors.sequential.Hot)
    fig.update_layout(mapbox_style="open-street-map")
    st.markdown("## Map visualisation - Bike rental stations in Edinburgh")
    st.plotly_chart(fig, use_container_width=True)

    # Distance calculator
    stations = df_mobility['station_name'].unique()
    st.markdown('## Distance calculator')
    location1 = st.multiselect('Select station 1:', stations)
    location2 = st.multiselect('Select station 2:', stations)
    
    button = st.button("Print distance",disabled=False)

    if button :
        if len(location1) <= 1 and len(location2) <= 1:
            location1 = ' '.join(location1) # From list to string
            location2 = ' '.join(location2)
            dist = df_distance_matrix.loc[location1, location2]
            st.markdown(f"#### Distance between {location1} and {location2}: {dist} km ")
        else:
            st.warning("You have to select only 1 location.")

    # st.write(f"Closest station is...: ") Can be added 

elif page == 'Rentals during 2018-2021':
    # Station filter
    stations = df['start_station_name'].unique()
    station_filter = st.sidebar.multiselect('Select station or multiple stations:', stations)
    if station_filter:
        df = df[df['start_station_name'].isin(station_filter)]

    # Bike rentals per day for future visualisation, grouped per station and date
    df_rentals_per_day = df.groupby(["date", "start_station_name"])["rental_period"].count().reset_index().rename(columns = {"rental_period" : "rentals_per_day"})

    #  Bike rentals in total per month
    df_rentals_per_day["formated_date"] = df_rentals_per_day["date"].dt.strftime("%Y-%m")
    df_rentals_per_month = df_rentals_per_day.drop("date", axis = 1)
    df_rentals_per_month = df_rentals_per_month.groupby(["formated_date", "start_station_name"]).sum().reset_index().rename(columns = {"rentals_per_day" : "rentals_per_month", "formated_date" : "date"})

    condition = df_rentals_per_month["date"] == "2018-09" # incomplete month
    df_rentals_per_month = df_rentals_per_month[~condition] 

    # Visualisation - bike rentals per month
    if not station_filter:
        message = "For comparison between stations please use filter on the left side."
        title = "## Bike rentals in 2018-2021 per month"
        df_rentals_per_day = df.groupby(["date"])["rental_period"].count().reset_index().rename(columns = {"rental_period" : "rentals_per_day"})

        # Bike rentals in total per month - now without grouping per station
        df_rentals_per_day["formated_date"] = df_rentals_per_day["date"].dt.strftime("%Y-%m")
        df_rentals_per_month = df_rentals_per_day.drop("date", axis = 1)
        df_rentals_per_month = df_rentals_per_month.groupby("formated_date").sum().reset_index().rename(columns = {"rentals_per_day" : "rentals_per_month", "formated_date" : "date"})
        condition = df_rentals_per_month["date"] == "2018-09" # incomplete month
        df_rentals_per_month  = df_rentals_per_month[~condition] 

        chart1 = alt.Chart(df_rentals_per_month).mark_line(opacity=0.6).encode(
            x = alt.X(
                "date", 
                axis=alt.Axis(title = "Date", labelAngle = -45, labelFontSize=13, titleFontSize=15, titleColor = "grey")),
            y = alt.Y(
                "rentals_per_month",
                axis=alt.Axis(title ="Bike rentals", labelFontSize=13, titleFontSize=15, titleColor = "grey")),
            tooltip= ["date", "rentals_per_month"],
            strokeWidth = alt.value(8)
        ).properties(
            width = 800,
        ).configure_axis(
            domainWidth=0.8
        ).interactive()

        chart1 = chart1.configure_title(
            fontSize=20,
            font='Courier',
            align = "center",
            color='gray')
        
    
    else: 
        title = "## Comparison between selected stations in 2018 - 2021"
        message = ""
        chart1 = alt.Chart(df_rentals_per_month).mark_line(opacity=0.6).encode(
            x = alt.X(
                "date", 
                axis=alt.Axis(title = "Date", labelAngle = -45, labelFontSize=13, titleFontSize=15, titleColor = "grey")),
            y = alt.Y(
                "rentals_per_month",
                scale=alt.Scale(domain=[df_rentals_per_month["rentals_per_month"].min(), df_rentals_per_month["rentals_per_month"].max() + 100]),
                axis=alt.Axis(title ="Bike rentals", labelFontSize=13, titleFontSize=15, titleColor = "grey")),
            tooltip= ["date", "rentals_per_month"],
            color = "start_station_name",
            strokeWidth = alt.value(8)
        ).properties(
            width = 800,
        ).configure_axis(
            domainWidth=0.8
        ).interactive()

        chart1 = chart1.configure_title(
            fontSize=20,
            font='Courier',
            align = "center",
            color='gray')

    st.markdown(title)
    st.altair_chart(chart1, use_container_width=True)
    st.write(message)

elif page == 'Causes of fluctuation in demand':

    st.markdown("# Causes of fluctuation in demand")
    st.write("")
    # Wihout filters - # TO DO - replace with function 
    df_rentals_per_month = df_rentals_per_day.drop("date", axis = 1)
    df_rentals_per_month = df_rentals_per_month.groupby("formated_date").sum().reset_index().rename(columns = {"rentals_per_day" : "rentals_per_month", "formated_date" : "date"})
    condition = df_rentals_per_month["date"] == "2018-09" # incomplete month
    df_rentals_per_month = df_rentals_per_month[~condition] 

    # Converting to month name and grouping
    df_rentals_per_month['month'] = pd.DatetimeIndex(df_rentals_per_month['date']).month
    df_rentals_per_month_name = df_rentals_per_month.groupby("month")["rentals_per_month"].mean().astype(int).reset_index().rename(columns = {"rentals_per_month" : "avg_rentals_per_month"})

    d = dict(enumerate(calendar.month_name))
    df_rentals_per_month_name['month'] = df_rentals_per_month_name['month'].map(d)
    df_rentals_per_month_name.index = range(1,13)

    chart2 = alt.Chart(df_rentals_per_month_name).mark_line(opacity=0.6).encode(
        x = alt.X(
            "month", 
            sort = None,
            axis=alt.Axis(title = "Month", labelAngle = -45, labelFontSize=13, titleFontSize=15, titleColor = "grey")),
        y = alt.Y(
            "avg_rentals_per_month",
            scale=alt.Scale(domain=[0, 30000]),
            axis=alt.Axis(title ="Bike rentals", labelFontSize=13, titleFontSize=15, titleColor = "grey")),
        tooltip= ["month", "avg_rentals_per_month"],
        strokeWidth = alt.value(8)
    ).properties(
        width = 800,
        title='Average bike rentals in 2018 - 2021'
    ).configure_axis(
        domainWidth=0.8
    ).interactive()

    chart2 = chart2.configure_title(
        fontSize=20,
        font='Courier',
        align = "center",
        color='gray')
    
    st.markdown("## Season")
    st.altair_chart(chart2, use_container_width=True)

    # Covid

    # Correlation between rentals per day and covid cases
    corr = df_bikes_covid.corr(method ='pearson').round(2).loc[["rentals_per_day"], ["tested_positive", "covid_deaths"]]

    # There is a negative relationship between rentals per day and confirmed covid cases (and deaths)
    # Hypothesis: During covid cases explosion, more employees worked from home --> decrease in rented bikes
    # ! Correlation is not causality

    # Displaying relationship with scatterplot + regression line

    chart3 = alt.Chart(df_bikes_covid).mark_circle(size=60).encode(
        x = alt.X(
            "tested_positive", 
            axis=alt.Axis(title = "Confirmed covid cases", labelAngle = -45, labelFontSize=13, titleFontSize=15, titleColor = "grey")),
        y = alt.Y(
            "rentals_per_day",
            axis=alt.Axis(title ="Bike rentals per day", labelFontSize=13, titleFontSize=15, titleColor = "grey")),
        tooltip= ["date", "rentals_per_day", "tested_positive", "covid_deaths"],
        strokeWidth = alt.value(4)
    ).properties(
        width = 800,
        title="Relationship between bike rentals and positive covid tests"
    )

    # Seems like exponential decrease --> fitting an exponential regression

    chart_regression_line = chart3.transform_regression("tested_positive", "rentals_per_day", method = "exp").mark_line(color = "red")

    # Top level graph object
    chart3 = alt.layer(chart3, chart_regression_line).configure_view(
        stroke = "transparent"
    ).configure_axis(
        domainWidth=0.8
    ).configure_title(
        fontSize=20,
        font="Courier",
        align = "center",
        color="gray"
    ).interactive()

    # The relationship seems not to be linear - data suggests high covid values have greater impact on bike rentals (lockdown?).
    # Low number of covid cases = large variance in bike rentals - it is obvious that there is a greater influence of season / temperature

    st.markdown("## Covid")
    st.altair_chart(chart3, use_container_width = True)
    st.markdown("##### Correlation matrix")
    st.write(corr.round(2))

    # Weather
    st.markdown("## Weather")

    df_weather_final  = df_rentals_per_day.merge(df_weather_avg).drop("formated_date", axis = 1)

    # Correlation matrix

    correlation_matrix = df_weather_final.corr(method ='pearson').round(2).sort_values("rentals_per_day")[["rentals_per_day"]]
    temperature_rentals_correlation = correlation_matrix.loc["temperature", "rentals_per_day"]
  
    chart4 = alt.Chart(df_weather_final).mark_circle(size=60).encode(
        x = alt.X(
            "temperature", 
            axis=alt.Axis(title = "Temperature", labelAngle = -45, labelFontSize=13, titleFontSize=15, titleColor = "grey")),
        y = alt.Y(
            "rentals_per_day",
            axis=alt.Axis(title ="Bike rentals per day", labelFontSize=13, titleFontSize=15, titleColor = "grey")),
        tooltip= ["date", "rentals_per_day", "temperature"],
        strokeWidth = alt.value(4)
    ).properties(
        title = "Relationship between bike rentals and temperature",
        width = 800
    )

    chart_regression_line_2 = chart4.transform_regression("temperature", "rentals_per_day", method = "exp").mark_line(color = "red")
        
    chart4 = alt.layer(chart4, chart_regression_line_2).configure_view(
        stroke = "transparent"
    ).configure_axis(
        domainWidth=0.8
    ).configure_title(
        fontSize=20,
        font="Courier",
        align = "center",
        color="gray"
    ).interactive()
            
    st.altair_chart(chart4, use_container_width = True)
    st.write(f"Temperature and number of rented bikes per day seems to be strongly corelated: r = {temperature_rentals_correlation}")
    st.markdown("##### Correlation matrix - other significant correlations")
    st.write(correlation_matrix.loc[["humidity_in_pct", "wind_km/h"], "rentals_per_day"])

    # Business day vs Weekend
    st.markdown("## Weekend")

    # New column - distinction of business day vs weekend

    business_days = pd.date_range(df_rentals_per_day["date"].min(), df_rentals_per_day["date"].max(), freq = "B") # Range of dates excluding weekends
    df_business_days = pd.DataFrame(business_days).rename(columns = {0: "date"}) # Converting to df

    list_of_business_days = df_business_days["date"].tolist()
    df_rentals_per_day["day"] = np.where(df_rentals_per_day.date.isin(list_of_business_days), "business day", "weekend")
                            
    df_weekday = df_rentals_per_day.groupby("day")[["rentals_per_day"]].mean().round().reset_index()
    df_weekday["pct_increase"] = df_weekday[["rentals_per_day"]].pct_change().round(2)

    pct_increase = int(df_weekday.loc[1, "pct_increase"] * 100)

    chart5 = alt.Chart(df_weekday).mark_bar(size = 125,opacity=0.6).encode(
    x = alt.X(
        "day", 
        axis=alt.Axis(title = "", labelAngle = 0, labelFontSize=17)),
    y = alt.Y(
        "rentals_per_day",
        axis=alt.Axis(title ="Bike rentals", labelFontSize=17, titleFontSize=17, titleColor = "grey")),
    tooltip= "rentals_per_day",
    strokeWidth = alt.value(4)
    ).properties(
        title='Average bike rentals at weekends and on business days',
        width = 50
    ).configure_title(
        fontSize=20,
        font='Courier',
        align = "center",
        color='gray')

    st.altair_chart(chart5, use_container_width = True)
    st.write(f"During the weekend, there is an average increase of {pct_increase} % rented bikes compared to business day")

elif page == "Prediction":
    df_final = df_bikes_covid.merge(df_weather_avg, how = "inner")

    # New columns for prediction purposes - month number and disctinction of business day / weekend
    df_final["month_number"] = pd.DatetimeIndex(df_final['date']).month
    business_days = pd.date_range(df_final["date"].min(), df_final["date"].max(), freq = "B") # Range of dates excluding weekends
    df_business_days = pd.DataFrame(business_days).rename(columns = {0: "date"}) # Converting to df
    list_of_business_days = df_business_days["date"].tolist()
    df_final["business_day_boolean"] = np.where(df_final.date.isin(list_of_business_days), True, False)

    # Regression using train test split
    # Using table df_final, where there are all possible predictors
    # Disadvantage - when merging above, we are loosing valuable data - (e.g. covid table and weather table do not overlap so much)
    # -- Left joining covid table, rentals table and weather table is on my openion not a suitable option, 
    # --resulting e.g. in lower correlation between covid and bike rentals (when filling covid cases with 0) 
    # -- Covid appeared no sooner then in 2019 --

    # By using these selected columns the explanatory power seems to be the highest

    reg_cols = ["rentals_per_day", "tested_positive", "covid_deaths", "temperature","humidity_in_pct","wind_km/h", "rain_mm", "month_number", "business_day_boolean"]
    df_train, df_test = train_test_split(df_final[reg_cols], train_size=0.7, test_size=0.3, random_state=100)

    y_train = df_train.pop('rentals_per_day')
    x_train = df_train

    y_test = df_test.pop('rentals_per_day')
    x_test = df_test

    ## scaling and training
    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    x_train_df = pd.DataFrame(x_train_scaled, index=x_train.index, columns=x_train.columns)

    reg = linear_model.LinearRegression()
    reg.fit(x_train_df, y_train)

    ## scaling test and predict

    x_test_scaled = scaler.transform(x_test)
    x_test_df = pd.DataFrame(x_test_scaled, index=x_test.index, columns=x_test.columns)

    predicted = reg.predict(x_test_df)

    predicted = pd.Series(predicted)
    y_test = y_test.reset_index().drop(columns='index')
    comparison = pd.concat([predicted, y_test], axis=1)

    st.markdown("## Prediction of bike rentals in Edinburg per day") 
    months = list(calendar.month_name)[1:] # Month names for prediction

    c1, c2,c3 = st.columns((1,4,1))
    with c2.form("Bike rentals prediction"):
        confirmed_positive_tests = st.slider('Confirmed positive Covid cases in UK', min_value=0, max_value=100000)
        confirmed_deaths = st.slider('Covid deaths in UK', min_value=0, max_value=2000)
        temp = st.slider('Temperature', min_value=0, max_value=30)
        humidity = st.slider('Humidity percentage', min_value=50, max_value=100)
        wind = st.slider('Wind in km/h', min_value=0, max_value=40)
        rain = st.slider('Rain in mm', min_value=0, max_value=10)
        month = st.select_slider('Month', months)
        business_boolean = st.select_slider("Choose Business day or Weekend", options:=["Business day", "Weekend"])
        
        # Converting back to desired types
        month_number = {"January" : 1, "February" : 2, "March": 3, "April": 4, "May": 5, "June": 6, "July": 7, "August": 8, "September" : 9, "October": 10, "November": 11, "December": 12}
        month = month_number[month]
        if business_boolean == "Business day":
            business_boolean = 1
        else:
            business_boolean = 0

        submitted = st.form_submit_button("Predict!")

        if submitted:
            prediction_features = [confirmed_positive_tests, confirmed_deaths, temp, humidity, wind, rain, month, business_boolean]

            regressor_prep = pd.DataFrame(prediction_features, index=x_test.columns).T
            reg_prep_scaled = pd.DataFrame(scaler.transform(regressor_prep), columns=x_test.columns)
            
            # final_prediction = round(reg.predict(reg_prep_scaled)[0])
            final_prediction = reg.predict(reg_prep_scaled).round()
        
            if final_prediction <= 0:
                c2.markdown(f"#### Given these conditions, number of rented bikes would be close to zero.")
            else:
                c2.markdown(f"#### Given these conditions, {int(final_prediction)} bikes would be rented on such day in Edinburgh.")

            c2.write("Mean squared error: %.2f" % mean_squared_error(y_test, predicted))
            c2.write("R2 Coefficient: %.2f" % r2_score(y_test, predicted))