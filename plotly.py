


kdata["cluster"] = y_pred_pca
kdata["cluster"] = kdata["cluster"].astype(str)
kdata['country'] = kdata['country'].str.replace('Cape Verde', 'Cabo Verde')
kdata['country'] = kdata['country'].str.replace('Congo, Dem. Rep.', 'Congo, The Democratic Republic of the')
kdata['country'] = kdata['country'].str.replace('Congo, Rep.', 'Republic of the Congo')
kdata['country'] = kdata['country'].str.replace('Macedonia, FYR', 'North Macedonia')
kdata['country'] = kdata['country'].str.replace('Micronesia, Fed. Sts.', 'Micronesia, Federated States of')
kdata['country'] = kdata['country'].str.replace('South Korea', 'Korea, Republic of')
kdata['country'] = kdata['country'].str.replace('St. Vincent and the Grenadines', 'Saint Vincent and the Grenadines')

import pycountry
import plotly.express as px
import plotly.io as pio

pio.renderers.default='browser'

list_countries = kdata['country'].unique().tolist()
"""
citation : https://www.kaggle.com/fran77/clustering-the-countries
"""
d_country_code = {}  # To hold the country names and their ISO
for country in list_countries:
    try:
        country_data = pycountry.countries.search_fuzzy(country)
        country_code = country_data[0].alpha_3
        d_country_code.update({country: country_code})
    except:
        print('could not add ISO 3 code for ->', country)
        # If could not find country, make ISO code ' '
        d_country_code.update({country: ' '})

# create a new column iso_alpha in the data
# and fill it with appropriate iso 3 code
for k, v in d_country_code.items():
    kdata.loc[(kdata.country == k), 'iso_alpha'] = v
kdata[kdata['iso_alpha'].duplicated(keep=False)]
kdata.loc[112,'iso_alpha'] = 'NER'
fig = px.choropleth(data_frame = kdata,
                    locations= "iso_alpha",
                    color= "cluster",  # value in column 'Confirmed' determines color
                    hover_name= "country",
                    )
fig.update_layout(
    title={
        'text': "Cluster Map based on country, child_mort, Exports, health, imports, Income, Inflation, life_expec, total_fer, gdpp",
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
                    
fig.show()
#hierarchical clustering map
#4clusters
kdata["clusterhr"] = y_pred_hc
kdata["clusterhr"] = kdata["clusterhr"].astype(str)
fig = px.choropleth(data_frame = kdata,
                    locations= "iso_alpha",
                    color= "clusterhr",  # value in column 'Confirmed' determines color
                    hover_name= "country",
                    )
fig.update_layout(
    title={
        'text': "Hierarchical Cluster Map based on country, child_mort, Exports, health, imports, Income, Inflation, life_expec, total_fer, gdpp",
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
                    
fig.show()
#2clusters
kdata["clusterhr2"] = y_pred_hc
kdata["clusterhr2"] = kdata["clusterhr2"].astype(str)
fig = px.choropleth(data_frame = kdata,
                    locations= "iso_alpha",
                    color= "clusterhr2",  # value in column 'Confirmed' determines color
                    hover_name= "country",
                    )
fig.update_layout(
    title={
        'text': "Hierarchical Cluster Map(2 clusters) based on country, child_mort, Exports, health, imports, Income, Inflation, life_expec, total_fer, gdpp",
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
                    
fig.show()
#child mortality 
fig = px.choropleth(data_frame = kdata,
                    locations= "iso_alpha",
                    color= "child_mort",  # value in column 'Confirmed' determines color
                    hover_name= "country")
fig.update_layout(
    title={
        'text': "Death of children under 5 years of age per 1000 live births",
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
                    
fig.show()

#Income
fig = px.choropleth(data_frame = kdata,
                    locations= "iso_alpha",
                    color= "income",  # value in column 'Confirmed' determines color
                    hover_name= "country")
fig.update_layout(
    title={
        'text': "Net income per person",
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
                    
fig.show()
#gdp
fig = px.choropleth(data_frame = kdata,
                    locations= "iso_alpha",
                    color= "gdpp",  # value in column 'Confirmed' determines color
                    hover_name= "country",
                    color_continuous_scale = "Aggrnyl")

fig.update_layout(
    title={
        'text': "The GDP per capita. Calculated as the Total GDP divided by the total population.",
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
                    
fig.show()
#Inflation
fig = px.choropleth(data_frame = kdata,
                    locations= "iso_alpha",
                    color= "inflation",  # value in column 'Confirmed' determines color
                    hover_name= "country",
                    color_continuous_scale = "RdBu")

fig.update_layout(
    title={
        'text': "The measurement of the annual growth rate of the Total GDP",
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
                    
fig.show()