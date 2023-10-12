# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 12:33:34 2023

@author: bartd
"""

############################################################################################################################################################
##Inladen van de packages.
############################################################################################################################################################


import numpy as np
import pandas as pd
import datetime as dt

import streamlit as st
import folium
from streamlit_folium import folium_static
from folium import plugins
from folium.plugins import MarkerCluster
from scipy.stats import norm
from ipywidgets import interact, widgets

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from branca.element import Template, MacroElement


############################################################################################################################################################
##Inladen van de dataframes autoinfo, laadpaaldata en opencharge.
############################################################################################################################################################


path = 'C:/Users/bartd/OneDrive/Bureaublad/Data Science (Minor)/' #Path moet aangepast worden.

autoinfo = pd.read_csv(f'{path}autoinfo_opgeschoont.csv')
laadpaaldata = pd.read_csv(f'{path}laadpaaldata_opgeschoont.csv')
opencharge = pd.read_csv(f'{path}opencharge_opgeschoont.csv')
benzineinfo = pd.read_csv(f'{path}benzineinfo_opgeschoont.csv')


############################################################################################################################################################
##De data waar aanpassen naar datetimes i.p.v. floats/objects.
############################################################################################################################################################


#Autoinfo datetime aanpassen.
kolommen = autoinfo.keys()
kolom_datum = []
for kolom in kolommen:
    sub = 'datum'
    if sub in kolom:
        kolom_datum.append(kolom)
for kolom in kolom_datum:
    autoinfo[kolom] = pd.to_datetime(autoinfo[kolom]).dt.strftime('%Y-%m-%d')
    autoinfo[kolom] = pd.to_datetime(autoinfo[kolom])   

#Laadpaaldata datetime aanpassen.
laadpaaldata['Started'] = pd.to_datetime(laadpaaldata['Started'], format='%Y-%m-%d %H:%M:%S')
laadpaaldata['Ended'] = pd.to_datetime(laadpaaldata['Ended'], format='%Y-%m-%d %H:%M:%S')

#Opencharge datetime aanpassen.
kolommen = opencharge.keys()
kolom_date = [] 
for kolom in kolommen:
    sub = 'Date'
    if sub in kolom:
        kolom_date.append(kolom)
for kolom in kolom_date:
    opencharge[kolom] = pd.to_datetime(opencharge[kolom]).dt.strftime('%Y-%m-%d')
    opencharge[kolom] = pd.to_datetime(opencharge[kolom])

#Benzineinfo datetime aanpassen.
benzineinfo['datum_eerste_toelating'] = pd.to_datetime(benzineinfo['datum_eerste_toelating'], format='%Y-%m-%d', errors='coerce')

############################################################################################################################################################
############################################################################################################################################################
##Beginnen met het maken in streamlit.
############################################################################################################################################################
############################################################################################################################################################


#Hier komt de titel en overige toevoeging....
st.title("Elektrische auto's in Nederland")
st.markdown('''Deze dashboard gaat over informatie over elektrische auto's in Nederland. Dit heeft betrekking tot de laadpalen door Nederland heen, hoeveel 
            elektriciteit wordt verbruikt bij een laadpaal en tot slot hoeveel van de auto's in Nederland elektrisch zijn.
            ''')


############################################################################################################################################################
##Sidebar Streamlit
############################################################################################################################################################

@st.cache_data()
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv_laadpaaldata = convert_df(laadpaaldata)
csv_autoinfo = convert_df(autoinfo)
csv_opencharge = convert_df(opencharge)
csv_benzineinfo = convert_df(benzineinfo)


st.sidebar.title('Referenties')
st.sidebar.write("Voor deze dashboard zijn verschillende datasets en API's gebruikt. Hieronder staan alle referenties.")

st.sidebar.divider()
st.sidebar.subheader('Laadpaaldata')
st.sidebar.write('De laadpaaldata is afkomstig van uit 2018.')
st.sidebar.download_button(label="Laadpaaldata.csv", data=csv_laadpaaldata, file_name='df_laadpaaldata.csv', mime='text/csv')


st.sidebar.divider()
st.sidebar.subheader("Informatie elektrische auto's")
st.sidebar.write("De informatie van de elektrische auto's is afkomstig van de openbare dataset van het RDW (https://opendata.rdw.nl/Voertuigen/Elektrische-voertuigen/w4rt-e856 ).")
st.sidebar.download_button(label="Autoinfo.csv", data=csv_autoinfo, file_name='df_autoinfo.csv', mime='text/csv')


st.sidebar.divider()
st.sidebar.subheader("Informatie brandstof")
st.sidebar.write("Deze dataset is een opgeschoonde samengevoegde versie van 2 andere datasets, beiden afkomstig van het RDW. De eerste is een dataset die het soort brandstof weergeeft(https://opendata.rdw.nl/Voertuigen/Elektrische-auto-s/vsxf-rq7p ), en de andere is de kentekeninformatie, waarbij ook de tenaamstelling staat (https://opendata.rdw.nl/Voertuigen/Kenteken-tenaamstelling/db8s-mw3u ).")
st.sidebar.download_button(label="Benzineinfo.csv", data=csv_benzineinfo, file_name='df_benzineinfo.csv', mime='text/csv')


st.sidebar.divider()
st.sidebar.subheader("Informatie locatie laadpalen")
st.sidebar.write('De informatie van de locatie van de laadpalen is afkomstig van de openchargemap API (https://openchargemap.org/site ). Hier staat informatie over 7907 laadpalen binnen Nederland, met daarbij de locatie en overige informatie.')
st.sidebar.download_button(label="Opencharge.csv", data=csv_opencharge, file_name='df_opencharge.csv', mime='text/csv')


############################################################################################################################################################
##Tabs maken.
############################################################################################################################################################


tab_opencharge, tab_laadpaaldata, tab_autoinfo, tab_benzineinfo = st.tabs(["Opencharge kaart", "Laadpaaldata", "Auto informatie", "Benzine informatie"])


############################################################################################################################################################
##Plaatjes benzineinfo.
############################################################################################################################################################


brandstof_counts = benzineinfo['brandstof_omschrijving'].value_counts()
percentage = (brandstof_counts / brandstof_counts.sum()) * 100

data = {'Brandstof': percentage.index, 'Percentage': percentage.values}
df = pd.DataFrame(data)
brandstof_pie = px.pie(df, names='Brandstof', values='Percentage', title='Percentage van brandstof_omschrijving')
#brandstof_pie.show()


with tab_benzineinfo:
    st.subheader("Cumulatieve plot voor het totale aantal auto's tussen 1877 en 2023 op tijdstip x.")
    # Add a sidebar with a slider for price range
    year_range = st.slider("Selecteer een prijs-interval (in $)", 1877, 2023, (1877, 2023))


benzine_groepen = benzineinfo.groupby(by=["brandstof_omschrijving", "jaar"]).size()
benzine_groepen = pd.DataFrame(benzine_groepen)
benzine_groepen = benzine_groepen.reset_index()
benzine_groepen = benzine_groepen.rename(columns = {0: 'aantal'})
benzine_groepen['totaal_tot_dan'] = 0


for i in range(len(benzine_groepen)):
    if benzine_groepen['brandstof_omschrijving'].iloc[i] == benzine_groepen['brandstof_omschrijving'].iloc[i - 1]:
        benzine_groepen.loc[i, 'totaal_tot_dan'] = benzine_groepen.loc[i-1 , 'totaal_tot_dan'] + benzine_groepen.loc[i, 'aantal'] 
    else:
        benzine_groepen.loc[i, 'totaal_tot_dan'] = benzine_groepen.loc[i , 'aantal']
        
brandstof_line = px.line(benzine_groepen, x="jaar", y="totaal_tot_dan", color = 'brandstof_omschrijving', 
              title="Cumulatieve lijndiagram voor aantal het auto's sinds 1877.", 
              labels = {'brandstof_omschrijving': 'Brandstof', 'jaar':'Jaar','totaal_tot_dan':"Aantal auto's tot dan"})
brandstof_line.update_xaxes(range=[year_range[0], year_range[1]]) 
#brandstof_line.show()


with tab_benzineinfo:
    st.plotly_chart(brandstof_line)
    st.divider()
    st.plotly_chart(brandstof_pie)
    


############################################################################################################################################################
##Plaatjes laadpaaldata.
############################################################################################################################################################


contime_box = px.histogram(laadpaaldata, x='ConnectedTime', marginal = 'box', nbins = 50)
contime_box.update_layout(title='Histogram en boxplot verbonden aan laadpaal (in uren)',
                          xaxis_title = 'Verbonden tijd', yaxis_title = 'Aantal')
#contime_box.show()



chargetime_box = px.histogram(laadpaaldata, x='ChargeTime', marginal = 'box', nbins = 50)
chargetime_box.update_layout(title='Histogram en boxplot aan het laden (in uren)',
                             xaxis_title = 'Tijd aan het laden', yaxis_title = 'Aantal')




toten_box = px.histogram(laadpaaldata, x='TotalEnergy', marginal = 'box', nbins = 50)
toten_box.update_layout(title='Histogram en boxplot totaal verbruikte energie in Wh',
                        xaxis_title = 'Verbruikte energie', yaxis_title = 'Aantal')
#toten_box.show()






# Laadtijden ophalen uit de laadpaaldata (verondersteld dat laadpaaldata ergens is gedefinieerd)
laadtijden = laadpaaldata["ChargeTime"]

# Maak het histogram
fig, ax = plt.subplots()
n, bins, patches = ax.hist(laadtijden, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')

# Voeg een titel en labels toe
ax.set_title('Histogram van Laadtijden')
ax.set_xlabel('Laadtijd (in uren)')
ax.set_ylabel('Kansdichtheid')

# Voeg annotaties toe voor het gemiddelde en de mediaan
gemiddelde = np.mean(laadtijden)
mediaan = np.median(laadtijden)
ax.axvline(gemiddelde, color='red', linestyle='dashed', linewidth=2, label=f'Gemiddelde: {gemiddelde:.2f}')
ax.axvline(mediaan, color='green', linestyle='dashed', linewidth=2, label=f'Mediaan: {mediaan:.2f}')

# Voeg een benadering van de kansdichtheidsfunctie toe
xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, np.mean(laadtijden), np.std(laadtijden))
ax.plot(x, p, 'k', linewidth=2, label='Kansdichtheidsfunctie')

# Voeg een legenda toe
ax.legend()

laadpaaldata['uur_fig5'] = laadpaaldata['Started'].dt.hour

uur_box = px.histogram(laadpaaldata, x='uur_fig5', nbins = 24)
uur_box.update_layout(title="Hoeveel auto's aan de lader gaan per uur.",
                          xaxis_title = 'Uur', yaxis_title = 'Aantal')
#uur_box.show()





with tab_laadpaaldata:
    st.plotly_chart(contime_box)
    st.divider()
    
    st.pyplot(fig)
    st.divider()
    
    st.plotly_chart(chargetime_box)
    st.divider()
    
    st.plotly_chart(toten_box)  
    st.divider()
    
    st.plotly_chart(uur_box)
    
    st.write('''Wanneer er wordt gekeken naar de dagen en maanden zijn er geen grote verschillen. Alleen op de 31e zijn er minder waarnemingen, 
             wat logisch is omdat maar de helft van de maanden een 31e dag hebben. Bij de maanden zijn er 's zomers alleen net iets minder waarnemingen.
             Maar dat komt ook omdat als het warm is de batterijduur van elektrische auto langer meegaat. 
             ''')


############################################################################################################################################################
##Plaatjes autoinfo.
############################################################################################################################################################


plot_merk = autoinfo.groupby(['merk']).size().reset_index(name='aantal')
plot_merk = plot_merk[plot_merk['aantal'] > 1000]
plot_merk = plot_merk.sort_values(by='aantal', ascending=False)

automerk = px.histogram(plot_merk, x = 'merk', y='aantal', 
                        labels = {'merk': 'Merk'})
automerk.update_layout(title="Aantal verkochte auto's per merk.",
                       yaxis_title = 'Aantal')

with tab_autoinfo:
    st.plotly_chart(automerk)


############################################################################################################################################################
##Plaatjes openchargemap.
############################################################################################################################################################

with tab_opencharge:


# Assuming you have already read the CSV file into opencharge_schoon DataFrame
# opencharge_schoon = pd.read_csv('opencharge_opgeschoont.csv')

# Convert the dataframe to a Plotly box plot
    fig3333 = px.box(opencharge, x='NumberOfPoints')

# Voeg titels toe aan de x- en y-assen
    fig3333.update_layout(
        title_text='Boxplot van aantal laadpalen per oplaadpunt',
        xaxis_title='Aantal laadpalen',  
        )

# Show the interactive plot
#fig3333.show()

    st.plotly_chart(fig3333)
    st.write('''Te zien is dat ruim 50% van de laadpalen maar een connectiepaal heeft. Daarom is er voor gekozen om in de in kaart gebrachte laadpalen in 
             Nederland te categoriseren onder 2 delen, meer dan 1 laadpaal (groen), of minder dan 1 laadpaal (rood).
             ''')


    df = opencharge

# Define coordinates and zoom levels for the center of each province
    province_centers = {
        "Nederland": (52.1326, 5.2913, 7),
        'Groningen': (53.2194, 6.5683, 8.5),
        'Friesland': (53.1642, 5.7814, 9.5),
        'Drenthe': (52.8105, 6.5907, 9),
        'Overijssel': (52.4992, 6.1276, 9),
        'Flevoland': (52.5522, 5.4261, 9.5),
        'Gelderland': (52.0614, 5.9285, 9),
        'Utrecht': (52.0907, 5.1214, 10),
        'Noord-Holland': (52.3874, 4.8998, 9),
        'Zuid-Holland': (51.9225, 4.47917, 9.3),
        'Zeeland': (51.4599, 3.6984, 9.2),
        'Noord-Brabant': (51.6369, 5.3181, 9),
        'Limburg': (51.2093, 5.9526, 9)
        }

# Streamlit app
    st.subheader('Laadpalen in Nederland')
    st.write('Laadpalen in Nederland, eventueel onderverdeeld in provincie.')

# Create a dropdown widget
    selected_province = st.selectbox('Selecteer een provincie:', list(province_centers.keys()))

# Retrieve coordinates and zoom level for the selected province
    selected_coords = province_centers[selected_province]

# Create a base map with the selected coordinates and zoom level
    mymap = folium.Map(location=(selected_coords[0], selected_coords[1]), zoom_start=selected_coords[2])

# Create a MarkerCluster layer
    marker_cluster = MarkerCluster().add_to(mymap)

# Add points to the map
    for index, row in df.iterrows():
        color = 'green' if row['NumberOfPoints'] > 1 else 'red'
        folium.CircleMarker(location=[row['AddressInfo.Latitude'], row['AddressInfo.Longitude']],
                            radius=5,
                            color=color,
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.7,
                            popup=f"Aantal palen: {row['NumberOfPoints']}").add_to(marker_cluster)

# Display the map using folium_static
    #folium_static(mymap)

# Define legend HTML
    legend_html = '''
    <div style="
            position: absolute;
            top: 50%;  
            left: 50%; 
            transform: translate(50%, 50%);
            bottom: 50px; 
            left: 50px; 
            width: 200px; 
            height: 100px; 
            background-color: white;
            border: 2px solid grey; 
            z-index: 9999; 
            font-size: 14px;
            padding: 10px;
            ">
            <b>Aantal Laadpalen</b> <br>
            Meer dan 1 laadpaal: <span style="color: green;">&#11044;</span> Groen<br>
            1 laadpaal: <span style="color: red;">&#11044;</span> Rood
            </div>
            '''
            
    col1, col2 = st.columns(2)
    legend = MacroElement()
    legend._template = Template(legend_html)

# Add the custom legend to the map
    mymap.get_root().add_child(legend)
    with col1:
        folium_static(mymap)
# Display the legend box
    with col2:
        st.markdown(legend_html, unsafe_allow_html=True)


############################################################################################################################################################
##Voorspelling doen.
############################################################################################################################################################


linear_reg = pd.read_pickle(f'{path}linear_reg.plk')

st.divider()
st.subheader('Autoprijs voorspellen')
st.write('''Er is een model gemaakt waarme de prijs van een auto voorspeld kan worden. De input hiervoor is het automerk, het aantal zitplaatsen en het
         aantal deuren, de lengte en tot slot het bouwjaar. 
         ''')


def predict_price(aantal_zitplaatsen, aantal_deuren, lengte, merk, jaar):
    # Convert the brand name to its corresponding numeric value using the brand_mapping dictionary
    merk_numeric = brand_mapping.get(merk)
    
    if merk_numeric is None:
        st.error("Please select a valid brand from the dropdown.")
        return None

    # Create a numpy array with the input values, including the brand as a numeric value
    input_data = np.array([[aantal_zitplaatsen, aantal_deuren, lengte, merk_numeric, jaar]])

    # Use the trained linear regression model to make predictions
    predicted_price = linear_reg.predict(input_data)

    return predicted_price[0]


# Your array of categorical values
original_array = np.array(['MITSUBISHI', 'NISSAN', 'CITROEN', 'MAZDA', 'PEUGEOT', 'N.S.U.', 'RENAULT', 'FORD', 'VOLKSWAGEN', 'TESLA', 'PORSCHE', 'THINK', 'SMART', 'MERCEDES-BENZ', 'JAGUAR', 'MIA', 'TESLA MOTORS', 'BMW I', 'FORD-CNG-TECHNIK', 'KIA', 'SAAB', 'ROVER', 'VOLVO', 'MICRO COMPACT CAR SMART', 'LOTUS', 'FIAT', 'ZIE BIJZONDERHEDEN', 'MG', 'TOYOTA', 'JAGUAR CARS', 'ZOTYE', 'BMW', 'BYD', 'MINI', 'AUSTIN', 'TRIUMPH', 'DAIHATSU', 'MATRA', 'WEINSBERG', 'ROLLS ROYCE', 'KEWET', 'MORRIS', 'JEEP', 'SKODA', 'DAF', 'HYUNDAI', 'AUDI', 'OPEL', 'CHEVROLET', 'TRABANT', 'DS', 'VW', 'JAC', 'POESSL', 'SEAT', 'ASTON MARTIN', 'CECOMP', 'MAXUS', 'HONDA', 'POLESTAR', 'MW MOTORS SRO', 'LEXUS', 'E GO', 'AIWAYS', 'SERES', 'DFSK', 'DACIA', 'MAN', 'DAIMLER BENZ', 'GERMAN E-CARS', 'M.A.N.', 'CUPRA', 'SSANGYONG', 'SUZUKI', 'XPENG', 'STELLA', 'SUBARU', 'DETROIT ELECTRIC', 'LAND ROVER', 'ALFA ROMEO', 'NIO', 'FAW', 'ORA', 'LUCID MOTORS', 'VOLKSWAGEN/ZIMNY', 'POSSL', 'GENESIS', 'ABARTH', 'VOYAH', 'UAZ', 'INDUSAUTO', 'BLUECAR'])

# Create a mapping dictionary
brand_mapping = {
    'MITSUBISHI': 0, 'NISSAN': 1, 'CITROEN': 2, 'MAZDA': 3, 'PEUGEOT': 4, 'N.S.U.': 5, 'RENAULT': 6, 'FORD': 7,
    'VOLKSWAGEN': 8, 'TESLA': 9, 'PORSCHE': 10, 'THINK': 11, 'SMART': 12, 'MERCEDES-BENZ': 13, 'JAGUAR': 14,
    'MIA': 15, 'TESLA MOTORS': 16, 'BMW I': 17, 'FORD-CNG-TECHNIK': 18, 'KIA': 19, 'SAAB': 20, 'ROVER': 21, 'VOLVO': 22,
    'MICRO COMPACT CAR SMART': 23, 'LOTUS': 24, 'FIAT': 25, 'ZIE BIJZONDERHEDEN': 26, 'MG': 27, 'TOYOTA': 28,
    'JAGUAR CARS': 29, 'ZOTYE': 30, 'BMW': 31, 'BYD': 32, 'MINI': 33, 'AUSTIN': 34, 'TRIUMPH': 35, 'DAIHATSU': 36,
    'MATRA': 37, 'WEINSBERG': 38, 'ROLLS ROYCE': 39, 'KEWET': 40, 'MORRIS': 41, 'JEEP': 42, 'SKODA': 43, 'DAF': 44,
    'HYUNDAI': 45, 'AUDI': 46, 'OPEL': 47, 'CHEVROLET': 48, 'TRABANT': 49, 'DS': 50, 'VW': 51, 'JAC': 52, 'POESSL': 53,
    'SEAT': 54, 'ASTON MARTIN': 55, 'CECOMP': 56, 'MAXUS': 57, 'HONDA': 58, 'POLESTAR': 59, 'MW MOTORS SRO': 60,
    'LEXUS': 61, 'E GO': 62, 'AIWAYS': 63, 'SERES': 64, 'DFSK': 65, 'DACIA': 66, 'MAN': 67, 'DAIMLER BENZ': 68,
    'GERMAN E-CARS': 69, 'M.A.N.': 70, 'CUPRA': 71, 'SSANGYONG': 72, 'SUZUKI': 73, 'XPENG': 74, 'STELLA': 75,
    'SUBARU': 76, 'DETROIT ELECTRIC': 77, 'LAND ROVER': 78, 'ALFA ROMEO': 79, 'NIO': 80, 'FAW': 81, 'ORA': 82,
    'LUCID MOTORS': 83, 'VOLKSWAGEN/ZIMNY': 84, 'POSSL': 85, 'GENESIS': 86, 'ABARTH': 87, 'VOYAH': 88, 'UAZ': 89,
    'INDUSAUTO': 90, 'BLUECAR': 91
}


# Dropdown to select a brand
merk = st.selectbox("Select a brand:", original_array)
# Convert the selected brand to its numeric value
numeric_value = brand_mapping.get(merk)

aantal_zitplaatsen = int(st.number_input('Kies het aantal zitplaatsen: ', min_value = 0, max_value = 8))
aantal_deuren = int(st.number_input('Kies het aantal deuren: ', min_value = 2, max_value = 6))
lengte = int(st.number_input('Kies de lengte van de auto: ', min_value = 200, max_value = 600))
jaar = int(st.number_input('Kies bouwjaar van de auto: ', min_value = 1990, max_value = 2023))


predicted_price = predict_price(aantal_zitplaatsen, aantal_deuren, lengte, merk, jaar)
st.write(f'De voorspelde prijs is €{round(predicted_price,2)}.')



############################################################################################################################################################
##Dataframe.
############################################################################################################################################################

st.divider()
st.subheader('Preview dataframes')
st.write('Een preview van de verschillende gebruikte dataframes, deze zijn downloadbaar bij de referenties. Alleen de benzineinfo dataframe staat er niet tussen, aangezien dit bestand te groot is voor de preview.')

tab_opencharge1, tab_laadpaaldata2, tab_autoinfo3 = st.tabs(["Opencharge", "Laadpaaldata", "Autoinfo"])

with tab_opencharge1:
    st.dataframe(opencharge)
    
with tab_laadpaaldata2:
    st.dataframe(laadpaaldata)
    
with tab_autoinfo3:
    st.dataframe(autoinfo)