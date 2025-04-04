import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.express as px
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from streamlit_option_menu import option_menu

#Streamlit instellingen
st.set_page_config(layout="wide")

with st.sidebar:
    pagina = option_menu(
        "Navigatie", 
        ["Introductie", "Data verkenning", "Correlatie", "Analyse fietstochten", "Voorspellend model", "Conclusie"],
        icons=["house", "bar-chart", "graph-up", "bicycle", "cpu", "check2-circle"],
        menu_icon="cast",
        default_index=0
    )

###########################################################################################################################

@st.cache_data
def load_weather_data_raw():
    df_weer = pd.read_csv('weather_london.csv') 
    df_weer.rename(columns={'Unnamed: 0': 'Datum'}, inplace=True)
    df_weer['Datum'] = pd.to_datetime(df_weer['Datum'])
    return df_weer

###########################################################################################################################

@st.cache_data
def load_weather_data():
    df_weer = pd.read_csv('weather_london.csv') 
    df_weer.rename(columns={'Unnamed: 0': 'Datum'}, inplace=True)
    df_weer['Datum'] = pd.to_datetime(df_weer['Datum'])
    df_weer.drop(columns=['tsun'])
    df_weer['neerslag_totaal'] = df_weer['prcp'].fillna(0) + df_weer['snow'].fillna(0)

    df_weer['windrichting'] = pd.cut(df_weer['wdir'], 
                                    bins=[0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5,360], 
                                    labels=['N', 'NO', 'O', 'ZO', 'Z', 'ZW', 'W', 'NW','N'], 
                                    right=False,ordered=False)
    richting_map = {
    'N': 0,
    'NO': 45,
    'O': 90,
    'ZO': 135,
    'Z': 180,
    'ZW': 225,
    'W': 270,
    'NW': 315}

    df_weer['windrichting_num'] = df_weer['windrichting'].map(richting_map)
    
    return df_weer

###########################################################################################################################

@st.cache_data
def load_weather_data_2019():
    df_weer = pd.read_csv('weather_london.csv') 
    df_weer.rename(columns={'Unnamed: 0': 'Datum'}, inplace=True)
    df_weer['Datum'] = pd.to_datetime(df_weer['Datum'])
    df_weer.drop(columns=['tsun'])
    df_weer['neerslag_totaal'] = df_weer['prcp'].fillna(0) + df_weer['snow'].fillna(0)

    df_weer['windrichting'] = pd.cut(df_weer['wdir'], 
                                    bins=[0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5,360], 
                                    labels=['N', 'NO', 'O', 'ZO', 'Z', 'ZW', 'W', 'NW','N'], 
                                    right=False,ordered=False)
    richting_map = {
    'N': 0,
    'NO': 45,
    'O': 90,
    'ZO': 135,
    'Z': 180,
    'ZW': 225,
    'W': 270,
    'NW': 315}

    df_weer['windrichting_num'] = df_weer['windrichting'].map(richting_map)

    df_weer= df_weer[df_weer['Datum'].dt.year == 2019]
    df_weer['snow'] = df_weer['snow'].fillna(0)
    kolommen_voor_ffill = df_weer.columns.drop(['Datum', 'snow']) 
    df_weer[kolommen_voor_ffill] = df_weer[kolommen_voor_ffill].fillna(method='ffill')
    return df_weer

###########################################################################################################################

@st.cache_data
def load_metro_data():
    metro = pd.read_csv('AnnualisedEntryExit_2019.csv', sep=';')
    stations = gpd.read_file("London stations.json")
    metro_stations = metro.merge(stations, how='left', left_on='Station', right_on='name')
    metro_stations = metro_stations.dropna(subset=['geometry'])
    metro_stations['Annualised_en/ex'] = metro_stations['Annualised_en/ex'].str.replace('.', '').astype('float')
    return gpd.GeoDataFrame(metro_stations, geometry="geometry")

###########################################################################################################################

@st.cache_data
def load_fiets_data():
    F4_dec = pd.read_csv('Koud redelijk droog 191JourneyDataExtract04Dec2019-10Dec2019.csv')
    F4_dec['week_soort'] = 'Koud & droog'
    F4_dec['regenval'] = 'Droog'
    F6_mrt = pd.read_csv('Nat gemiddeld 152JourneyDataExtract06Mar2019-12Mar2019.csv')
    F6_mrt['week_soort'] = "Gemiddeld & nat"
    F6_mrt['regenval'] = 'Nat'
    F26_jun = pd.read_csv('Warm droog 168JourneyDataExtract26Jun2019-02Jul2019.csv')
    F26_jun['week_soort'] = 'Warm & droog'
    F26_jun['regenval'] = 'Droog'
    F11_dec = pd.read_csv('Nat koud 192JourneyDataExtract11Dec2019-17Dec2019.csv')
    F11_dec['week_soort'] = 'Koud & nat'
    F11_dec['regenval'] = 'Nat'
    F08_aug = pd.read_csv('Nat warm 174JourneyDataExtract07Aug2019-13Aug2019.csv')
    F08_aug['week_soort'] = 'Warm & nat'
    F08_aug['regenval'] = 'Nat'
    F20_mrt = pd.read_csv('Droog gemiddeld 154JourneyDataExtract20Mar2019-26Mar2019.csv')
    F20_mrt['week_soort'] = "Gemiddeld & droog"
    F20_mrt['regenval'] = 'Droog'


    fiets_df = pd.concat([F4_dec, F6_mrt, F26_jun, F11_dec, F08_aug, F20_mrt, ])
    fiets_df.drop(columns=['Rental Id', 'Bike Id'], inplace=True)
    fiets_df['End Date']=pd.to_datetime(fiets_df['End Date'], format='%d/%m/%Y %H:%M')
    fiets_df['End'] = fiets_df['End Date'].dt.date
    fiets_df['Start Date']=pd.to_datetime(fiets_df['Start Date'], format='%d/%m/%Y %H:%M')
    fiets_df['Start'] = fiets_df['Start Date'].dt.date
 
    parkeer = pd.read_csv('cycle_stations.csv')
    parkeer['id'] = parkeer['id'].replace({865: 32, 857: 174, 867: 512, 868: 270, 870: 347, 864: 120, 869: 100, 872: 339, 852: 639, 866: 777})
 
    fiets_start = fiets_df.groupby(['Start', 'StartStation Name', 'StartStation Id','week_soort','regenval']).size().reset_index(name="Vertrek_Count")
    fiets_eind = fiets_df.groupby(['End', 'EndStation Name', 'EndStation Id','week_soort', 'regenval']).size().reset_index(name="Aankomst_Count")
    fiets_count = fiets_start.merge(fiets_eind, left_on=['Start', 'StartStation Name'], right_on=['End', 'EndStation Name'], how='outer').fillna(0)
    fiets_count.rename(columns={'Start': 'datum', 'StartStation Name': 'Station', 'StartStation Id': 'Id','week_soort_x':'Week_soort', 'regenval_x':'regenval'}, inplace=True)
    fiets_count['datum'] = pd.to_datetime(fiets_count['datum'])
    fiets_count['total_users'] = fiets_count['Vertrek_Count'] + fiets_count['Aankomst_Count']
    fiets_count['dag_van_de_week'] = pd.to_datetime(fiets_count['datum']).dt.dayofweek
    weekelijkse_gebruikers = fiets_count.groupby(['Station', 'Week_soort'])['total_users'].sum().reset_index()
    weekelijkse_gebruikers.rename(columns={'total_users': 'weekly_total_users'}, inplace=True)
    fiets_count = fiets_count.merge(weekelijkse_gebruikers, on=['Station', 'Week_soort'], how='left')
    fiets_count = fiets_count.merge(parkeer[['id', 'lat', 'long']], how='left', left_on='Id', right_on='id')
    fiets_count = fiets_count.dropna(subset=['lat'])
    fiets_count["geometry"] = fiets_count.apply(lambda row: Point(row["long"], row["lat"]), axis=1)
    return gpd.GeoDataFrame(fiets_count, geometry="geometry")
###########################################################################################################################

@st.cache_data
def load_ritten_per_dag():
    df = pd.read_csv('ritten_per_dag.csv') 
    df['Start Date'] = pd.to_datetime(df['Start Date'])

    # Voeg kolom toe met de dag van de week (in het Nederlands)
    df['dag_index'] = df['Start Date'].dt.dayofweek
    dagen_nl = ['maandag', 'dinsdag', 'woensdag', 'donderdag', 'vrijdag', 'zaterdag', 'zondag']
    df['dag'] = df['Start Date'].dt.dayofweek.map(lambda x: dagen_nl[x])

    # Voeg kolom toe voor weekend: 0 = doordeweeks, 1 = weekend
    df['weekdag_of_weekend'] = df['Start Date'].dt.dayofweek.map(lambda x: 1 if x >= 5 else 0)

    return df

###########################################################################################################################

@st.cache_data
def load_ritten_per_dag_janfeb_2020():
    df = pd.read_csv('ritten_per_dag_jan_feb_2020.csv') 
    df['Start Date'] = pd.to_datetime(df['Start Date'])
    return df
###########################################################################################################################

@st.cache_data
def load_tijsduur_ritten():
    df=pd.read_csv('huurfietsen_per_dag.csv')
    df['Huurtijd']=df['Huurtijd']/df['Aantal Ritten']
    df['Start Date'] = pd.to_datetime(df['Start Date'])
    return df

###########################################################################################################################
@st.cache_data
def load_start_stop():
    df=pd.read_csv('top_10_samengevoegd.csv')
    return df

###########################################################################################################################

# Data laden
weather_raw=load_weather_data_raw()
weather = load_weather_data()
weather2019=load_weather_data_2019()
metro_stations_geo = load_metro_data()
fiets_count_geo = load_fiets_data()
ritten_per_dag_fiets = load_ritten_per_dag()
ritten_per_dag_fiets_janfeb_2020 = load_ritten_per_dag_janfeb_2020()
tijdsduur_ritten_2019=load_tijsduur_ritten()
start_stop_stations=load_start_stop()

jaren = sorted(weather['Datum'].dt.year.unique())





custom_labels = {
    "tmax": "Maximale temperatuur (°C)",
    "tavg": "Gemiddelde temperatuur (°C)",
    "tmin": "Minimale temperatuur (°C)",
    "Huurtijd": "Gemmidelde huurtijd (minuten)",
    "Aantal Ritten": "Aantal fietsritten",
    "neerslag_totaal": "Neerslag totaal (mm)",
    "pres": "Luchtdruk (hPa)",
    "wpgt": "Max windvlaag (km/h)",
    "wspd": "Windsnelheid (km/h)",
    "weekdag_of_weekend": "Weekdag of weekend",
    "windrichting_num": "Windrichting (graden)",
    "dag": "Dag",
    "dag_index": "Dag",
    "tsun":"Aantal minuten zon",
    "wdir":"Windrichting (graden)",
    "snow":"Sneeuwval",
    "prcp":"Regenval (mm)"}


###########################################################################################################################
###########################################################################################################################
###########################################################################################################################



if pagina == 'Introductie':
    st.title('🚲 Heen en weer in Londen – Deelfietsen & Weer')
    st.subheader("Team 5 - Nadine Glas & Pol Nieuwint")
    st.markdown("""
    Welkom bij onze interactieve data-analyse over het gebruik van **deelfietsen in Londen**! In dit dashboard onderzoeken we hoe **weersomstandigheden** zoals temperatuur, neerslag en wind invloed hebben op **het gebruik van deelfietsen**.

    🌤️ **Onderzoeksvraag:**  
    _"Welke weersfactoren beïnvloeden het aantal ritten en de duur van ritten met deelfietsen in Londen?"_

    ---  

    ### 🎯 Wat waren onze doelen bij deze herhaling van de case?
    - 🔍 **Beter inzicht krijgen in de data** door uitgebreidere verkenning en voorbereiding  
    - 📆 **Weekdag** en **gemiddelde huurtijd** meenemen als verklarende factoren  
    - 📈 **Visualisaties verbeteren** en duidelijker maken voor de gebruiker  
    - 🤖 Een krachtiger en betrouwbaarder **voorspellend model** bouwen op basis van meerdere variabelen""")  









#######################################################################################################################################
#######################################################################################################################################
###########################################################################################################################

# Pagina: Weerdata
if pagina == 'Data verkenning':
    
    # Streamlit App
    st.title("Verkenning van de weer en deelfietsdata")

    



    # VISUALISATIE MISSENDE WAARDEN ##############################################################################################
    # ▼ 1. Selectbox voor keuze uit dataframe
    keuze_df = st.selectbox(
        "📂 Kies een dataset om missende waarden te bekijken:",
        options=["Weer 2000-2022", "Weer 2019", "Aantal fietsritten per dag 2019", "Tijdsduur fietsritten 2019", 
                "2019 Entry/Exit per metro station",
                "Aantal fietsritten per dag Jan/Feb 2020", "Begin- en eindlocatie fietsritten"]
    )

    # ▼ 2. Data laden op basis van keuze
    if keuze_df == "Weer 2000-2022":
        df = weather_raw
    elif keuze_df == "Weer 2019":
        df = weather_raw[weather_raw['Datum'].dt.year == 2019]  
    elif keuze_df == "Aantal fietsritten per dag 2019":
        df = ritten_per_dag_fiets
    elif keuze_df == "Tijdsduur fietsritten 2019":
        df = tijdsduur_ritten_2019
    elif keuze_df == "2019 Entry/Exit per metro station":
        df = metro_stations_geo
    elif keuze_df == "Aantal fietsritten per dag Jan/Feb 2020":
        df = ritten_per_dag_fiets_janfeb_2020
    elif keuze_df == "Begin- en eindlocatie fietsritten":
        df = start_stop_stations 


    # ▼ 3. Missende waarden tellen
    missing = df.isna().sum()
    missing = missing[missing > 0]
    missing_percent = (missing / len(df)) * 100

    missing_df = pd.DataFrame({
        'Kolom': missing.index,
        'Aantal missend': missing.values,
        'Percentage missend': missing_percent.values
    })

    # 🏷️ ▼ 3a. Pas kolomnamen aan met custom labels (indien beschikbaar)
    missing_df['Label'] = missing_df['Kolom'].map(custom_labels).fillna(missing_df['Kolom'])

    # ▼ 4. Visualiseer als dual y-axis
    if not missing_df.empty:
        fig = go.Figure()

        # Y1: aantal missende waarden
        fig.add_trace(go.Bar(
            x=missing_df['Label'],
            y=missing_df['Aantal missend'],
            name='Aantal missende waarden',
            marker_color='indianred',
            yaxis='y1',
        ))

        # Y2: percentage
        fig.add_trace(go.Scatter(
            x=missing_df['Label'],
            y=missing_df['Percentage missend'],
            name='Percentage missend (%)',
            yaxis='y2',
            mode='lines+markers',
            marker=dict(color='gold')
        ))

        fig.update_layout(
            title=f"🔍 Missende waarden in dataset: {keuze_df}",
            xaxis=dict(title="Kolom"),
            yaxis=dict(title="Aantal missende waarden"),
            yaxis2=dict(
                title="Percentage (%)",
                overlaying='y',
                side='right'
            ),
            legend=dict(x=1.05, y=1),
            template='plotly_dark',
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success(f"✅ Geen missende waarden gevonden in `{keuze_df}`.")
    

    # VISUALISATIE WEERDATASET LONDON 2019 ######################################################################################################

    st.subheader('Weeromstandigheden in Londen 2019')
    jaar = st.slider('Selecteer het jaar', min_value=min(jaren), max_value=2022, value=2019, step=1)

    # ▼ Kopieer jaar subset
    df_jaar = weather[weather['Datum'].dt.year == jaar].copy()

    # ▼ 3-laagse subplot
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            f"{custom_labels['tmax']}, {custom_labels['tavg']} & {custom_labels['tmin']}",
            f"{custom_labels['wspd']} & {custom_labels['wpgt']}",
            f"{custom_labels['prcp']} & sneeuw"
        ],
        row_heights=[0.33, 0.33, 0.34],
        specs=[[{}], [{}], [{}]]
    )

    # 🔴 Bovenste plot - Temperatuur
    fig.add_trace(go.Scatter(x=df_jaar['Datum'], y=df_jaar['tmax'], name=custom_labels['tmax'], line=dict(color='red', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_jaar['Datum'], y=df_jaar['tavg'], name=custom_labels['tavg'], line=dict(color='green', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_jaar['Datum'], y=df_jaar['tmin'], name=custom_labels['tmin'], line=dict(color='blue', width=1)), row=1, col=1)

    # 🟠 Middelste plot - Wind
    fig.add_trace(go.Scatter(x=df_jaar['Datum'], y=df_jaar['wspd'], name=custom_labels['wspd'], line=dict(color='sandybrown', width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_jaar['Datum'], y=df_jaar['wpgt'], name=custom_labels['wpgt'], line=dict(color='orange', width=1)), row=2, col=1)

    # 🔵 Onderste plot - Neerslag
    fig.add_trace(go.Bar(x=df_jaar['Datum'], y=df_jaar['prcp'], name='Regenval (mm)', marker_color='royalblue'), row=3, col=1)
    fig.add_trace(go.Bar(x=df_jaar['Datum'], y=df_jaar['snow'], name='Sneeuwval (mm)', marker_color='white'), row=3, col=1)

    # 🛠️ Layout
    fig.update_layout(
        template='plotly_dark',
        height=850,
        width=1500,
        barmode='overlay',
        bargap=0.1,
        showlegend=True,
        legend=dict(x=1.05, y=1)
    )

    # 🗓️ X-as
    fig.update_xaxes(
        title_text="Datum",
        tickmode='linear',
        dtick='M1',
        row=3, col=1
    )

    # Y-as labels
    fig.update_yaxes(title_text="Temperatuur (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Wind (km/h & °)", row=2, col=1)
    fig.update_yaxes(title_text="Neerslag (mm)", row=3, col=1)

    # Toon de grafiek
    st.plotly_chart(fig, use_container_width=True)

    
    
    
    
    # BOXPLOTS TEMPERATUREN ######################################################################################################################

    # 📌 Create subplots with shared y-axis
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            f"🔴 {custom_labels['tmax']}",
            f"🟢 {custom_labels['tavg']}",
            f"🔵 {custom_labels['tmin']}"
        ],
        shared_yaxes=True
    )

    fig.add_trace(go.Box(y=weather2019["tmax"], name=custom_labels['tmax'], marker_color="red"), row=1, col=1)
    fig.add_trace(go.Box(y=weather2019["tavg"], name=custom_labels['tavg'], marker_color="green"), row=1, col=2)
    fig.add_trace(go.Box(y=weather2019["tmin"], name=custom_labels['tmin'], marker_color="blue"), row=1, col=3)

    fig.update_layout(
        title="📊 Boxplots van de temperatuur variabelen (2019)",
        showlegend=True,
        height=500,
        width=900,
        yaxis_title="Temperatuur (°C)"
    )

    # Remove x-axis labels and set y-axis range
    for col in range(1, 4):
        fig.update_xaxes(showticklabels=False, row=1, col=col)
        fig.update_yaxes(range=[-10, 40], row=1, col=col)

    st.plotly_chart(fig, use_container_width=True)




    # BAR+LIJN PLOT AANTAL FIETSRITTEN EN WEERDATA ######################################################################################################################################

    # Combine the data first
    df_combined = ritten_per_dag_fiets.merge(
        tijdsduur_ritten_2019[['Start Date', 'Huurtijd']],
        on='Start Date',
        how='inner'
    )

    # Setup subplot with secondary y-axis enabled
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]]
    )

    # ▶ Bar trace: Aantal Ritten
    fig.add_trace(go.Bar(
        x=df_combined['Start Date'],
        y=df_combined['Aantal Ritten'],
        name='Fietsritten per dag',
        marker=dict(color='white'),
        opacity=0.7,
        customdata=np.stack((df_combined['Huurtijd'],), axis=-1),
        hovertemplate=
            '📅 Datum: %{x|%d-%m-%Y}<br>' +
            '🚲 Aantal ritten: %{y}<br>' +
            '⏱ Gem. huurtijd: %{customdata[0]:.1f} min<br>' +
            '<extra></extra>'
    ), secondary_y=False)

    # ▶ Line trace: Gemiddelde huurtijd
    fig.add_trace(go.Scatter(
        x=df_combined['Start Date'],
        y=df_combined['Huurtijd'],
        mode='lines',
        name='Gem. huurtijd',
        line=dict(color='violet', width=2),
        customdata=np.stack((df_combined['Aantal Ritten'],), axis=-1),
        hovertemplate=
            '📅 Datum: %{x|%d-%m-%Y}<br>' +
            '⏱ Gem. huurtijd: %{y:.1f} min<br>' +
            '🚲 Aantal ritten: %{customdata[0]}<br>' +
            '<extra></extra>'
    ), secondary_y=True)

    # Layout and axes
    fig.update_layout(
        title='📊 Aantal fietsritten & gemiddelde huurtijd per dag (Londen 2019)',
        template='plotly_dark',
        barmode='overlay',
        height=500,
        showlegend=True,
        legend=dict(x=1.05, y=1),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    fig.update_xaxes(
        title_text="Datum",
        tickvals=pd.date_range(start="2019-01-01", end="2019-12-31", freq="MS"),
        ticktext=[d.strftime("%b") for d in pd.date_range(start="2019-01-01", end="2019-12-31", freq="MS")],
        tickangle=0,
        showgrid=True
    )

    fig.update_yaxes(
        title_text="Aantal fietsritten per dag", secondary_y=False
    )
    fig.update_yaxes(
        title_text="Gemiddelde huurtijd (min)", secondary_y=True
    )

    # Show it
    st.plotly_chart(fig, use_container_width=True)





#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################









if pagina == 'Correlatie':
    st.title('De correlatie van verschillende variabelen')

    # CORRELATIEMATRIX AANTAL RITTEN #######################################################################################################################################
    tijdsduur_ritten_correlatie = tijdsduur_ritten_2019.drop(columns=['Aantal Ritten'])
    df_merged = weather.merge(ritten_per_dag_fiets, left_on="Datum", right_on="Start Date")
    df_merged = df_merged.merge(tijdsduur_ritten_correlatie, on='Start Date')

    # Drop non-numeric columns
    df_corr = df_merged.drop(columns=["Datum", "Start Date",'wdir','windrichting','dag']) 

    # Compute correlation matrix
    correlatie_matrix = df_corr.corr()
    
    # Sorteer op absolute correlatie met 'Aantal Ritten'
    correlatie_bike = correlatie_matrix[['Aantal Ritten']].dropna()
    correlatie_bike['abs_corr'] = correlatie_bike['Aantal Ritten'].abs()  
    
    # Classificeer de correlatie volgens de tabel
    def classificatie(r):
        if abs(r) > 0.5:
            return "Sterk"
        elif abs(r) > 0.3:
            return "Matig"
        elif abs(r) > 0:
            return "Zwak"
        else:
            return "Geen"
    
    # Voeg de classificatie toe
    correlatie_bike['Sterkte'] = correlatie_bike['Aantal Ritten'].apply(classificatie)
    
    # Sorteer en verwijder helper kolom
    correlatie_bike = correlatie_bike.sort_values(by='abs_corr', ascending=False).drop(columns=['abs_corr'])
    
    # Maak aangepaste annotaties met zowel de correlatie als de classificatie
    annotaties = correlatie_bike.apply(lambda row: f"{row['Aantal Ritten']:.2f}\n({row['Sterkte']})", axis=1)

    st.subheader("📊 Correlatie heatmap aantal fietsritten 2019")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        correlatie_bike[['Aantal Ritten']], 
        annot=annotaties.values.reshape(-1,1),  # Gebruik aangepaste annotaties
        cmap="coolwarm", fmt="", linewidths=0.5, center=0, ax=ax,
        yticklabels = [custom_labels.get(var, var) for var in correlatie_bike.index]
    )
    ax.set_title("Correlatie tussen aantal fietsritten en variabelen")
    
    # Display plot in Streamlit
    st.pyplot(fig)


    # SCATTERPLOT ONDER CORRELATIE 1 #######################################################################################################################################
    # Load Data
    jaar_data = pd.read_csv('ritten_per_dag.csv')
    jaar_data['Start Date'] = pd.to_datetime(jaar_data['Start Date'])

    df_weather = pd.read_csv('weather_london.csv')
    df_weather = df_weather.rename(columns={'Unnamed: 0': 'Date'})
    df_weather['Date'] = pd.to_datetime(df_weather['Date'], errors='coerce')

    # Filter only for the year 2019
    weather_2019 = df_weather[df_weather['Date'].dt.year == 2019]

    # Count unique tmax values and their occurrences in 2019
    tmax_counts_2019 = weather_2019['tmax'].value_counts().reset_index()
    tmax_counts_2019.columns = ['tmax', 'count']
    tmax_counts_2019 = tmax_counts_2019.sort_values(by='tmax', ascending=True)

     # ▼ 1. Maak merge van ritten met weerdata
    merged_df = pd.merge(jaar_data, df_weather, left_on='Start Date', right_on='Date', how='inner')

    # Voeg huurtijd toe aan de merge
    tijdsduur_ritten_2019['Start Date'] = pd.to_datetime(tijdsduur_ritten_2019['Start Date'])
    merged_df = pd.merge(merged_df, tijdsduur_ritten_2019[['Start Date', 'Huurtijd']], on='Start Date', how='left')

    keuze_mapping = {
        "Maximale temperatuur (tmax)": "tmax",
        "Gemiddelde temperatuur (tavg)": "tavg",
        "Minimale temperatuur (tmin)": "tmin",
        "Huurtijd (in minuten)": "Huurtijd"
    }

    # ▼ 2. Keuzemenu met nette labels
    label_keuze = st.selectbox(
        "📊 Kies een variabele voor de x-as:",
        options=list(keuze_mapping.keys()),
        index=0
    )

    # ▼ 3. Haal de kolomnaam op uit de mapping
    keuze_x = keuze_mapping[label_keuze]

# ▼ 3. Maak een samenvattende tabel per gekozen variabele
    if keuze_x in merged_df.columns:
        summary_table = merged_df.groupby(keuze_x)['Aantal Ritten'].agg(
            mean='mean',
            median='median',
            mode=lambda x: stats.mode(x, keepdims=True)[0][0]
        ).reset_index()

        # Voeg telling toe (hoe vaak kwam deze waarde voor)
        value_counts = merged_df[keuze_x].value_counts().reset_index()
        value_counts.columns = [keuze_x, 'count']
        new_df = pd.merge(summary_table, value_counts, on=keuze_x)

        # ▼ 4. Lineaire regressie
        X = new_df[keuze_x].values
        y = new_df['mean'].values
        coeff = np.polyfit(X, y, 1)
        poly_fn = np.poly1d(coeff)
        regressie_formule = f"y = {coeff[0]:.2f}x + {coeff[1]:.2f}"

        # ▼ 5. Plot
        fig = px.scatter(
            new_df,
            x=keuze_x,
            y="mean",
            color=keuze_x,
            color_continuous_scale="RdBu_r",
            size="count",
            hover_data={keuze_x: True, "mean": True, "count": True},
            labels={keuze_x: keuze_x, "mean": "Gemiddeld aantal fietsritten"},
            title=f"Aantal fietsritten per dag t.o.v. {keuze_x}"
        )

        fig.add_trace(go.Scatter(
            x=new_df[keuze_x],
            y=poly_fn(new_df[keuze_x]),
            mode='lines',
            name=f"Lineaire regressie: {regressie_formule}",
            line=dict(color="gold", width=2)
        ))

        fig.update_traces(marker=dict(line=dict(width=1, color="black")))
        fig.update_layout(coloraxis_colorbar=dict(title=keuze_x))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("❌ Gekozen variabele is niet beschikbaar in de dataset.")
    

    # CORRELATIEMATRIX HUURTIJD #######################################################################################################################################
    # Compute correlation matrix
    correlatie_matrix = df_corr.corr()

    # Select correlation with 'Huurtijd'
    correlatie_bike = correlatie_matrix[['Huurtijd']].dropna()
    correlatie_bike['abs_corr'] = correlatie_bike['Huurtijd'].abs()  # absolute correlation

    # Classificatie functie
    def classificatie(r):
        if abs(r) > 0.5:
            return "Sterk"
        elif abs(r) > 0.3:
            return "Matig"
        elif abs(r) > 0:
            return "Zwak"
        else:
            return "Geen"

    # Voeg classificatie toe
    correlatie_bike['Sterkte'] = correlatie_bike['Huurtijd'].apply(classificatie)

    # Sorteer en drop helper kolom
    correlatie_bike = correlatie_bike.sort_values(by='abs_corr', ascending=False).drop(columns=['abs_corr'])

    # Annotaties
    annotaties = correlatie_bike.apply(lambda row: f"{row['Huurtijd']:.2f}\n({row['Sterkte']})", axis=1)

    # Plot
    st.subheader("📊 Correlatie heatmap huurtijd fietsritten 2019")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        correlatie_bike[['Huurtijd']], 
        annot=annotaties.values.reshape(-1, 1),
        cmap="coolwarm", fmt="", linewidths=0.5, center=0, ax=ax,
        yticklabels = [custom_labels.get(var, var) for var in correlatie_bike.index]
    )
    ax.set_title("Correlatie tussen huurtijd en variabelen")
    st.pyplot(fig)


    # SCATTERPLOT ONDER CORRELATIEMATRIX 2 #######################################################################################################################################
    keuze_mapping = {
        "Maximale temperatuur (tmax)": "tmax",
        "Gemiddelde temperatuur (tavg)": "tavg",
        "Minimale temperatuur (tmin)": "tmin",
        "Aantal Ritten": "Aantal Ritten"
    }

    # ▼ 2. Keuzemenu met nette labels
    label_keuze = st.selectbox(
        "📊 Kies een variabele voor de x-as:",
        options=list(keuze_mapping.keys()),
        index=0,
        key="xas_select"
    )

    # ▼ 3. Haal de kolomnaam op uit de mapping
    keuze_x = keuze_mapping[label_keuze]

    # ▼ 3. Maak een samenvattende tabel per gekozen variabele
    if keuze_x in merged_df.columns:
        summary_table = merged_df.groupby(keuze_x)['Huurtijd'].agg(
            mean='mean',
            median='median',
            mode=lambda x: stats.mode(x, keepdims=True)[0][0]
        ).reset_index()

        # Voeg telling toe (hoe vaak kwam deze waarde voor)
        value_counts = merged_df[keuze_x].value_counts().reset_index()
        value_counts.columns = [keuze_x, 'count']
        new_df = pd.merge(summary_table, value_counts, on=keuze_x)

        # ▼ 4. Lineaire regressie
        X = new_df[keuze_x].values
        y = new_df['mean'].values
        coeff = np.polyfit(X, y, 1)
        poly_fn = np.poly1d(coeff)
        regressie_formule = f"y = {coeff[0]:.2f}x + {coeff[1]:.2f}"

        # ▼ 5. Plot
        fig = px.scatter(
            new_df,
            x=keuze_x,
            y="mean",
            color=keuze_x,
            color_continuous_scale="RdBu_r",
            size="count",
            hover_data={keuze_x: True, "mean": True, "count": True},
            labels={keuze_x: keuze_x, "mean": "Tijdsduur van fietsritten (in min.)"},
            title=f"Tijdsduur van fietsritten per dag t.o.v. {keuze_x}"
        )

        fig.add_trace(go.Scatter(
            x=new_df[keuze_x],
            y=poly_fn(new_df[keuze_x]),
            mode='lines',
            name=f"Lineaire regressie: {regressie_formule}",
            line=dict(color="gold", width=2)
        ))

        fig.update_traces(marker=dict(line=dict(width=1, color="black")))
        fig.update_layout(coloraxis_colorbar=dict(title=keuze_x))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("❌ Gekozen variabele is niet beschikbaar in de dataset.")




#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################




if pagina == 'Analyse fietstochten':
    st.title('Analyse van de fietstochten in Londen')
    st.subheader('Deelfietsgebruik tijdens verschillende weersomstandigheden')

    # BARPLOT WEEKSOORTEN ##########################################################################################################################
    # Zorg voor weekend en doordeweeks
    fiets_count_geo['is_weekend'] = fiets_count_geo['dag_van_de_week'].isin([5, 6])  # 5=Saturday, 6=Sunday
 
    # Groeperen per week, is_weekend en het type dag
    week_data = fiets_count_geo.groupby(['Week_soort', 'is_weekend']).agg({'total_users': 'sum'}).reset_index()
 
    # Voor barchart willen we een kolom per week per type dag (doordeweeks vs weekend)
    week_data_pivot = week_data.pivot_table(index='Week_soort', columns='is_weekend', values='total_users', aggfunc='sum')
 
    # Gemiddelde per dag berekenen voor week- en weekenddagen
    week_data_pivot['Doordeweekse ritten'] = week_data_pivot[False]
    week_data_pivot['Weekend ritten'] = week_data_pivot[True]
    week_data_pivot = week_data_pivot.reset_index()

    # Maak barplot met correcte gemiddelden
    fig = px.bar(
        week_data_pivot,
        x='Week_soort',
        y=['Doordeweekse ritten', 'Weekend ritten'],
        labels={'value': 'Totaal aantal gebruikers', 'Week_soort': 'Week soort', 'variable': 'Type dag'},
        title='Totaal aantal ritten per weeksoort', 
        barmode='stack',
        height=400
    )
    # Toon de plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


    # KAART FIETS+METRO+BEZIENSWAARDIGHEDEN ######################################################################################################################################
    st.subheader('Heatmap fietstochten in warme, koude, natte, droge weken')   
     # Voorbereiden dataframes
    fiets_plot = fiets_count_geo.copy()
    fiets_plot['lat'] = fiets_plot.geometry.y
    fiets_plot['lon'] = fiets_plot.geometry.x

    # Locaties voor dropdown
    map_center_dropdown = {
        'Londen Centrum 🏙️': {"lat": 51.5074, "lon": -0.1278, "zoom": 11},
        'The British Museum 🏛️': {"lat": 51.5194, "lon": -0.1270, "zoom": 14},
        'The Tower of London 🏰': {"lat": 51.5081, "lon": -0.0759, "zoom": 14},
        'Buckingham Palace 👑': {"lat": 51.5014, "lon": -0.1419, "zoom": 14},
        'The Houses of Parliament & Big Ben 🏛️': {"lat": 51.5007, "lon": -0.1246, "zoom": 14},
        'The London Eye 🎡': {"lat": 51.5033, "lon": -0.1195, "zoom": 14}
    }

    location = st.selectbox('Kies een locatie om in te zoomen:', options=list(map_center_dropdown.keys()), index=0)
    selected_location = map_center_dropdown[location]

    # ▸ Filters boven de kaart
    col1, col2 = st.columns(2)
    with col1:
        gekozen_temp = st.radio("🌡️ Temperatuurtype", ["Warm", "Gemiddeld", "Koud"], horizontal=True)
    with col2:
        gekozen_regen = st.radio("🌧️ Regenval", ["Droog", "Nat"], horizontal=True)

    # ▸ Filter op de combinaties
    filtered_fiets_plot = fiets_plot[
        fiets_plot['Week_soort'].str.contains(gekozen_temp, case=False) &
        (fiets_plot['regenval'].str.lower() == gekozen_regen.lower())
    ]

    # ▸ Beperk tot slechts één unieke week per type (automatisch kiezen)
    unieke_weken = sorted(filtered_fiets_plot['Week_soort'].unique())
    if len(unieke_weken) > 0:
        gekozen_week = unieke_weken[0]
        filtered_fiets_plot = filtered_fiets_plot[filtered_fiets_plot['Week_soort'] == gekozen_week]
        st.caption(f"Geselecteerde week: **{gekozen_week}**")
    else:
        st.warning("⚠️ Geen gegevens beschikbaar voor deze combinatie.")
        st.stop()

    # 🔁 Reset slider value only if it's not set or out of bounds
    slider_key = "min_fietsritten_slider"

    max_slider = int(filtered_fiets_plot['weekly_total_users'].max())

    # Initial or invalid state: reset to 0
    if slider_key not in st.session_state or st.session_state[slider_key] > max_slider:
        st.session_state[slider_key] = 0

    min_fietsritten = st.slider(
        'Minimaal aantal fietsritten',
        0,
        max_slider,
        st.session_state[slider_key],
        key=slider_key
    )

    # Filter op basis van sliderwaarde
    filtered_fiets_plot = filtered_fiets_plot[filtered_fiets_plot['weekly_total_users'] >= min_fietsritten]

    # Maak interactieve kaart zonder kleurbar
    fig = px.scatter_mapbox(
        filtered_fiets_plot,
        lat="lat", lon="lon",
        color_discrete_sequence=["green"],
        size='weekly_total_users',
        hover_name="Station",
        size_max=25,
        labels={"weekly_total_users": "Aantal fietsritten"}
    )

    fig.update_traces(name="Fietsritten", showlegend=True)

    # Heatmap laag toevoegen zonder colourbar
    fig.add_trace(go.Densitymapbox(
        lat=filtered_fiets_plot['lat'],
        lon=filtered_fiets_plot['lon'],
        z=filtered_fiets_plot['weekly_total_users'],
        radius=5,
        colorscale="Viridis",
        opacity=0.5,
        showscale=False,
        name='Heatmap fietsritten'
    ))

    # Bezienswaardigheden altijd toevoegen
    markers_data = [
        {"name": "The British Museum 🏛️", "lat": 51.5194, "lon": -0.1270},
        {"name": "The Tower of London 🏰", "lat": 51.5081, "lon": -0.0759},
        {"name": "Buckingham Palace 👑", "lat": 51.5014, "lon": -0.1419},
        {"name": "The Houses of Parliament 🏛️", "lat": 51.5007, "lon": -0.1246},
        {"name": "The London Eye 🎡", "lat": 51.5033, "lon": -0.1195}
    ]

    fig.add_trace(go.Scattermapbox(
        lat=[m['lat'] for m in markers_data],
        lon=[m['lon'] for m in markers_data],
        mode='markers+text',
        text=[m['name'] for m in markers_data],
        marker=dict(size=14, color='red'),
        textposition='top right',
        name='Bezienswaardigheden'
    ))

    center_lat = np.mean([m['lat'] for m in markers_data])
    center_lon = np.mean([m['lon'] for m in markers_data])+0.009
    max_radius = max(np.sqrt((np.array([m['lat'] for m in markers_data]) - center_lat)**2 +
                             (np.array([m['lon'] for m in markers_data]) - center_lon)**2))

    theta = np.linspace(0, 2*np.pi, 100)
    circle_lat = center_lat + max_radius * np.cos(theta)
    circle_lon = center_lon + max_radius * np.sin(theta)

    fig.add_trace(go.Scattermapbox(
        lat=circle_lat,
        lon=circle_lon,
        mode='lines',
        line=dict(color='orange', width=2),
        name='Cirkel rondom bezienswaardigheden'
    ))

    fig.update_layout(
        mapbox_style="carto-darkmatter",
        height=600,
        mapbox=dict(center={"lat": selected_location["lat"], "lon": selected_location["lon"]}, zoom=selected_location["zoom"]),
        legend=dict(y=0.99, x=0.01, bgcolor="rgba(0,0,0,0.6)", font=dict(color="white")),
        margin={"r":0,"t":0,"l":0,"b":0}
    )

    st.plotly_chart(fig, use_container_width=True)





    # fietsritten en huurtijd per dag bar+lijn ####################################################################################
    st.subheader('Visualisatie van de gemiddelde huurtijd per dag')


    df_combined = ritten_per_dag_fiets.merge(
        tijdsduur_ritten_2019[['Start Date', 'Huurtijd']],
        on='Start Date',
        how='inner')
    
    # Groepeer per dag van de week
    df_gemiddeld = df_combined.groupby(['dag_index', 'dag']).agg({
        'Aantal Ritten': 'mean',
        'Huurtijd': 'mean'
    }).reset_index().sort_values('dag_index')

    # ▶ Voeg customdata toe (zodat hover beide waarden kan tonen)
    custom_data = np.stack((df_gemiddeld['Aantal Ritten'], df_gemiddeld['Huurtijd']), axis=-1)

    fig = go.Figure()

    # ▶ Balk: Aantal Ritten
    fig.add_trace(go.Bar(
        x=df_gemiddeld['dag'],
        y=df_gemiddeld['Aantal Ritten'],
        name='Gemiddeld aantal ritten',
        marker_color='white',
        opacity=0.9,
        customdata=custom_data,
        hovertemplate=(
            "<b>%{x}</b><br>" +
            "Gem. ritten: %{y:.0f}<br>" +
            "Gem. huurtijd: %{customdata[1]:.1f} min"
        )
    ))

    # ▶ Lijn: Huurtijd
    fig.add_trace(go.Scatter(
        x=df_gemiddeld['dag'],
        y=df_gemiddeld['Huurtijd'],
        name='Gemiddelde huurtijd (min)',
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='violet', width=3),
        customdata=custom_data,
        hovertemplate=(
            "<b>%{x}</b><br>" +
            "Gem. huurtijd: %{y:.1f} min<br>" +
            "Gem. ritten: %{customdata[0]:.0f}"
        )
    ))

    # ▶ Layout
    fig.update_layout(
        title='📊 Gemiddeld aantal fietsritten en huurtijd per weekdag',
        xaxis=dict(title='Weekdag'),
        yaxis=dict(title='Gemiddeld aantal ritten'),
        yaxis2=dict(
            title='Gemiddelde huurtijd (minuten)',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        template='plotly_dark',
        legend=dict(x=1.02, y=1),
        margin=dict(t=50, b=40)
    )

    # ▶ Plot tonen
    st.plotly_chart(fig, use_container_width=True)


#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################





elif pagina == 'Voorspellend model':
    st.title('Voorspelling aantal fietsritten jan/feb 2020')
    # Regressiecoefficienten tabel ######################################################################################################################################    
    # Load Data
    jaar_data = pd.read_csv('ritten_per_dag.csv')
    jaar_data['Start Date'] = pd.to_datetime(jaar_data['Start Date'])

    df_weather = pd.read_csv('weather_london.csv')
    df_weather = df_weather.rename(columns={'Unnamed: 0': 'Date'})
    df_weather['Date'] = pd.to_datetime(df_weather['Date'], errors='coerce')

    # Filter only for the year 2019
    weather_2019 = df_weather[df_weather['Date'].dt.year == 2019]

    # Count unique tmax values and their occurrences in 2019
    tmax_counts_2019 = weather_2019['tmax'].value_counts().reset_index()
    tmax_counts_2019.columns = ['tmax', 'count']
    tmax_counts_2019 = tmax_counts_2019.sort_values(by='tmax', ascending=True)

    # Merge the weather and bike data on the date column
    merged_df = pd.merge(jaar_data, df_weather, left_on='Start Date', right_on='Date', how='inner')

    # Group by 'tmax' and calculate statistics
    summary_table = merged_df.groupby('tmax')['Aantal Ritten'].agg(
        mean='mean',
        median='median',
        mode=lambda x: stats.mode(x, keepdims=True)[0][0]  # Get the first mode
    ).reset_index()

    # Merge counts
    new_df = pd.merge(summary_table, tmax_counts_2019)


    # **Train Model Using `tmax` Instead of `tavg`**
    X_train = new_df[['tmax']]  
    y_train = new_df['mean']  

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Select weather data for Jan & Feb 2020
    weer_2020 = df_weather[(df_weather['Date'].dt.year == 2020) & (df_weather['Date'].dt.month.isin([1, 2]))]

    # Compute daily max temperature in Jan & Feb 2020
    weer_2020_avg = weer_2020.groupby('Date')['tmax'].max().reset_index()

    # Predict bike rides based on max temperature
    weer_2020_avg['Voorspelde Fietsritten'] = model.predict(weer_2020_avg[['tmax']])


    from sklearn.linear_model import LinearRegression
    from sklearn.impute import SimpleImputer

    # Load Data
    jaar_data = pd.read_csv('ritten_per_dag.csv')
    jaar_data['Start Date'] = pd.to_datetime(jaar_data['Start Date'])
    jaar_data['weekdag_of_weekend'] = jaar_data['Start Date'].dt.dayofweek.map(lambda x: 1 if x >= 5 else 0)

    # Merge weather and bike data for training
    df_weer = pd.read_csv('weather_london.csv') 
    df_weer.rename(columns={'Unnamed: 0': 'Datum'}, inplace=True)
    df_weer['Datum'] = pd.to_datetime(df_weer['Datum'])
    df_weer.drop(columns=['tsun'])
    df_weer['neerslag_totaal'] = df_weer['prcp'].fillna(0) + df_weer['snow'].fillna(0)
    df_weer['windrichting'] = pd.cut(df_weer['wdir'], 
                                    bins=[0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5,360], 
                                    labels=['N', 'NO', 'O', 'ZO', 'Z', 'ZW', 'W', 'NW','N'], 
                                    right=False,ordered=False)
    richting_map = {
    'N': 0,
    'NO': 45,
    'O': 90,
    'ZO': 135,
    'Z': 180,
    'ZW': 225,
    'W': 270,
    'NW': 315}

    df_weer['windrichting_num'] = df_weer['windrichting'].map(richting_map)

    merged_df = pd.merge(jaar_data, df_weer, left_on='Start Date', right_on='Datum', how='inner')

    # Use multiple features
    features = ['tmax', 'tmin', 'tavg', 'neerslag_totaal', 'wspd', 'pres','windrichting_num','weekdag_of_weekend','wpgt']
    X_train = merged_df[features]
    y_train = merged_df['Aantal Ritten']

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)

    # Train the model
    model = LinearRegression()
    model.fit(X_train_imputed, y_train)

    # --- Predict for Jan-Feb 2020 ---
    weer_2020 = df_weer[(df_weer['Datum'].dt.year == 2020) & (df_weer['Datum'].dt.month.isin([1, 2]))].copy()
    weer_2020['weekdag_of_weekend'] = weer_2020['Datum'].dt.dayofweek.map(lambda x: 1 if x >= 5 else 0)
    weer_2020['neerslag_totaal'] = weer_2020['prcp'].fillna(0) + weer_2020['snow'].fillna(0)

    X_test = weer_2020[features]

    if X_test.isna().all(axis=None):
        st.error("❌ Geen voorspelling mogelijk: alle waarden in de weerdata van 2020 bevatten ontbrekende gegevens.")
        st.dataframe(weer_2020[features].isna().sum().to_frame())
    else:
        # Impute test set
        X_test_imputed = imputer.transform(X_test)
        weer_2020['Voorspelde Fietsritten'] = model.predict(X_test_imputed)
        
    # Toon gesorteerde regressiecoëfficiënten
    coeff_df = pd.DataFrame({
        'Variabele': features,
        'Coëfficiënt': model.coef_
    })

    # Sorteer op absolute invloed (van groot naar klein)
    coeff_df['Absolute'] = coeff_df['Coëfficiënt'].abs()
    coeff_df = coeff_df.sort_values(by='Absolute', ascending=False).drop(columns='Absolute')

    st.subheader("📈 Gesorteerde regressiecoëfficiënten (invloed op voorspelde fietsritten)")

    coeff_df['Variabele'] = coeff_df['Variabele'].map(custom_labels)
    coeff_df = coeff_df.reset_index(drop=True)
    coeff_df.index = coeff_df.index + 1
    coeff_df.index.name = "#"
    st.dataframe(coeff_df)

    # VOORSPELLING PLOT ######################################################################################################################################
    # 📊 Aangepaste voorspellinggrafiek met extra info
    fig_pred = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            "🌡️ Temperatuur & 🌧️ Neerslag",
            "💨 Windsnelheid & Windvlaag",
             "🚲 Voorspelde Fietsritten",
        ),
        specs=[[{"secondary_y": True}], [{}], [{}]]
    )

    # 🌡️ Temperatuurtracés
    fig_pred.add_trace(go.Scatter(
        x=weer_2020['Datum'], y=weer_2020['tmax'],
        name="Max. Temp (°C)", mode="lines",
        line=dict(color="red", dash="dash")
    ), row=1, col=1, secondary_y=False)

    fig_pred.add_trace(go.Scatter(
        x=weer_2020['Datum'], y=weer_2020['tavg'],
        name="Gem. Temp (°C)", mode="lines",
        line=dict(color="green", dash="dash")
    ), row=1, col=1, secondary_y=False)

    fig_pred.add_trace(go.Scatter(
        x=weer_2020['Datum'], y=weer_2020['tmin'],
        name="Min. Temp (°C)", mode="lines",
        line=dict(color="blue", dash="dash")
    ), row=1, col=1, secondary_y=False)

    # 🌧️ Neerslag op y2-as (zelfde subplot)
    fig_pred.add_trace(go.Bar(
        x=weer_2020['Datum'], y=weer_2020['neerslag_totaal'],
        name="Neerslag (mm)", marker_color="lightskyblue", opacity=0.6
    ), row=1, col=1, secondary_y=True)

    # 🚲 Voorspelling
    fig_pred.add_trace(go.Bar(
        x=weer_2020['Datum'], y=weer_2020['Voorspelde Fietsritten'],
        name="Voorspelde Fietsritten", marker_color="limegreen", opacity=0.7
    ), row=3, col=1)

    # 💨 Wind
    fig_pred.add_trace(go.Scatter(
        x=weer_2020['Datum'], y=weer_2020['wspd'],
        name="Gem. Windsnelheid (km/h)", mode="lines+markers",
        line=dict(color="orange")
    ), row=2, col=1)

    fig_pred.add_trace(go.Scatter(
        x=weer_2020['Datum'], y=weer_2020['wpgt'],
        name="Windvlaag (km/h)", mode="lines+markers",
        line=dict(color="violet", dash="dot")
    ), row=2, col=1)

    # 🛠️ Layout
    fig_pred.update_layout(
        title="📈 Voorspelling van Fietsritten o.b.v. weerdata (Jan/Feb 2020)",
        height=900,
        template="plotly_dark",
        showlegend=True,
        legend=dict(x=1.02, y=1),
        margin=dict(t=60, b=40)
    )

    fig_pred.update_yaxes(title_text="Temperatuur (°C)", row=1, col=1, secondary_y=False)
    fig_pred.update_yaxes(title_text="Neerslag (mm)", row=1, col=1, secondary_y=True)
    fig_pred.update_yaxes(title_text="Aantal fietsritten", row=3, col=1)
    fig_pred.update_yaxes(title_text="Windsnelheid (km/h)", row=2, col=1)

    # Toon in Streamlit
    st.plotly_chart(fig_pred, use_container_width=True)


    # VOORSPELDNIEUW+OUD+WERKELIJK ######################################################################################################################################
    fig = go.Figure()

    # Plot predicted bike rides (green line)
    fig.add_trace(go.Scatter(
        x=weer_2020['Datum'],
        y=weer_2020['Voorspelde Fietsritten'],
        mode='lines',
        name='Voorspelde Fietsritten',
        line=dict(color='green', width=3)
    ))

    # plot old prediction (red line)
    fig.add_trace(go.Scatter(
        x=weer_2020_avg['Date'],
        y=weer_2020_avg['Voorspelde Fietsritten'],
        mode='lines',
        name='Oude voorspelde Fietsritten',
        line=dict(color='red', width=3)
    ))

    # Plot actual bike rides (blue line)
    fig.add_trace(go.Bar(
        x=ritten_per_dag_fiets_janfeb_2020['Start Date'],
        y=ritten_per_dag_fiets_janfeb_2020['Aantal Ritten'],
        name='fietsritten per dag',
        marker=dict(color='white', opacity=0.6),
  # Koppel deze bars aan de tweede y-as
    ))

    # Update layout
    fig.update_layout(
        title='Vergelijking van Voorspelde en Werkelijke Fietsritten (Jan/Feb 2020)',
        xaxis_title='Datum',
        yaxis_title='Aantal fietsritten',
        template='plotly_dark',
        showlegend=True,
        legend=dict(x=1.1, y=1),
        height=500,
        width=1000
    )

    # Display plot in Streamlit
    st.subheader("📊 Voorspelde vs Werkelijke Fietsritten")
    st.plotly_chart(fig, use_container_width=True)



#######################################################################################################################################
#######################################################################################################################################
###########################################################################################################################





if pagina == 'Conclusie':
    st.title('✅ Conclusie')

    st.subheader('📌 Belangrijkste bevindingen')
    st.markdown("""
    - **Weersinvloeden** spelen een grote rol in het gebruik van deelfietsen in Londen.  
    - **Temperatuur** blijkt de grootste invloed te hebben op het **aantal ritten** per dag.  
    - Voor de **duur van een rit** is vooral de **dag van de week** (weekdag vs weekend) bepalend.  
    - Op basis van historische weer- en fietsdata kunnen we betrouwbare **voorspellingen doen** over toekomstig fietsgedrag.
    """)

    "---"  

    st.subheader('🚲 Toepassing van deze inzichten')
    st.markdown("""
    - **Stadsplanners & beleidsmakers** kunnen hiermee slimmere keuzes maken rondom fietsinfrastructuur, vooral tijdens drukke of rustige perioden.  
    - **Fietsverhuurders** kunnen efficiënter plannen voor **onderhoud en herverdeling van fietsen**, door te voorspellen wanneer vraag hoog of laag is — dit minimaliseert stilstand en maximaliseert beschikbaarheid.  
    """)

    "---"  

    st.subheader('🔄 Verbeteringen in het vernieuwde dashboard')
    st.markdown("""
    #### 🔹 Introductie
    - Duidelijke toelichting op de onderzoeksvragen en doelen.

    #### 🔹 Data verkenning
    - Visualisatie van **ontbrekende waarden** per dataset.
    - Gedetailleerde plots van **temperatuur, neerslag en wind** in 2019.
    - Toevoeging van de **gemiddelde huurtijd** per dag.

    #### 🔹 Correlatie-analyse
    - Zowel **aantal ritten** als **huurtijd** zijn geanalyseerd.
    - Correlaties zijn geclassificeerd als **sterk, matig of zwak**.
    - Toegevoegde interactieve scatterplots maken verbanden visueel inzichtelijk.

    #### 🔹 Analyse fietstochten
    - **Barplot per weeksoort**: toont ritten in koude, warme, natte of droge weken.
    - **Interactieve heatmap van Londen** met filtermogelijkheden op temperatuur en regen.
    - Nieuwe grafiek met **gemiddeld aantal ritten en huurtijd per weekdag**.

    #### 🔹 Voorspellend model
    - In plaats van enkel de maximale temperatuur is nu een **multivariabel regressiemodel** gebruikt.
    - Voorspellingen van fietsgedrag voor jan/feb 2020 o.b.v. **9 weer- en tijdvariabelen**.
    - Vergelijking tussen **oude vs nieuwe voorspelling** én de werkelijke cijfers.
    """)