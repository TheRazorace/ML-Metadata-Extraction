#!/usr/bin/env python
# coding: utf-8

# <a class="anchor" id="0"></a>
# 
# # EDA of the week forecast trends of 70 countries (2022.02.21 - 2022.02.27)
# ## Data plotting and mapping
# ### Data from output of the notebook [COVID-19 in 70 countries: daily Prophet forecast](https://www.kaggle.com/vbmokin/covid-19-in-70-countries-daily-prophet-forecast?scriptVersionId=88418297)

# ## Acknowledgements
# * dataset with COVID data excluding data for Ukraine [COVID-19: Forecast trends for the many countries](https://www.kaggle.com/vbmokin/covid19-forecast-trends-for-the-many-countries)
# * dataset with correct COVID data for Ukraine [COVID-19 in Ukraine: daily data](https://www.kaggle.com/vbmokin/covid19-in-ukraine-daily-data)
# * dataset with holidays [COVID-19: Holidays of countries](https://www.kaggle.com/vbmokin/covid19-holidays-of-countries)
# * dataset with coordinates of the countries capitals [World Cities Database](https://www.kaggle.com/abhijithchandradas/world-cities-database)
# * notebook for data generation [COVID-19 in 70 countries: daily Prophet forecast](https://www.kaggle.com/vbmokin/covid-19-in-70-countries-daily-prophet-forecast)
# * notebook for the section 4.1 [FE - Feature Importance - Advanced Visualization](https://www.kaggle.com/vbmokin/fe-feature-importance-advanced-visualization)
# * notebook for the section 4.2 [Wuhan Coronavirus : A geographical analysis](https://www.kaggle.com/parulpandey/wuhan-coronavirus-a-geographical-analysis)
# 

# <a class="anchor" id="0.1"></a>
# 
# ## Table of Contents
# 
# 1. [Import libraries](#1)
# 1. [Download datasets](#2)
# 1. [FE](#3)
# 1. [EDA](#4)
#     -  [Plotting country trends in parallel coordinates](#4.1)
#     -  [Mapping the slopes of country trends](#4.2)    

# ## 1. Import libraries <a class="anchor" id="1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
import openpyxl
import plotly.express as px
import plotly.graph_objects as go

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import linregress

import folium 
from folium import plugins


# In[ ]:


# Graphics in retina format 
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

# Increase the default plot size and set the color scheme
plt.rcParams['figure.figsize'] = 8, 5
#plt.rcParams['image.cmap'] = 'viridis'

pd.set_option('max_columns',100)
pd.set_option('max_rows',100)


# ## 2. Download datasets <a class="anchor" id="2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


# Download data
df = pd.read_csv('../input/covid19-forecast-trends-for-the-many-countries/forecast_future_dfs_2022-02-22.csv')
df


# In[ ]:


forecast_dates = list(np.array(df.tail(7).ds.tolist()))
forecast_dates


# In[ ]:


df.info()


# ## 3. FE<a class="anchor" id="3"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


df = df[['ds', 'trend', 'country']]
df


# In[ ]:


df2 = pd.pivot_table(df, values='trend', index='ds', columns='country')
df_cols = df2.columns
df2


# In[ ]:


# Coutries with delay 1 day
#df2_add = df2[['Belgium', 'Burundi', 'Iceland', 'Luxembourg', 'Peru', 'Sweden', 'Switzerland', 'Thailand']]
df2_day_delay_countries_list = ['Nigeria', 'Paraguay']
df2_add = df2[df2_day_delay_countries_list]
df2_add


# In[ ]:


# Coutries with delay many days (more than 1 day)
df2_more_day_delay_countries_list = ['Belgium', 'Burundi', 'Finland', 'Honduras', 
                                     'Hungary', 'Iceland', 'Ireland', 'Luxembourg', 
                                     'Nicaragua', 'Spain', 'Sweden', 'Switzerland']
df2_more_day_delay_countries = df2[df2_more_day_delay_countries_list]
df2_more_day_delay_countries


# In[ ]:


df2_add = df2_add[-8:-1]
forecast_dates_add = df2_add.index.tolist()
df2_add


# In[ ]:


ds_start_add = df2_add.index.tolist()[0]
ds_start_add


# In[ ]:


ds_end_add = df2_add.index.tolist()[-1]
ds_end_add


# In[ ]:


df2_without_days_delay_countries_list = list(set(df2.columns.tolist()).difference(set(df2_more_day_delay_countries_list+df2_day_delay_countries_list)))
df2 = df2[df2_without_days_delay_countries_list]
df2 = df2[-7:]
df2


# In[ ]:


ds_start = df2.index.tolist()[0]
ds_start


# In[ ]:


ds_end = df2.index.tolist()[-1]
ds_end


# ## 4. EDA <a class="anchor" id="4"></a>
# 
# [Back to Table of Contents](#0.1)

# ### 4.1. Plotting country trends in parallel coordinates <a class="anchor" id="4.1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


# EDA for countries
df3 = df2.T
df3.columns = forecast_dates
df3


# In[ ]:


def plot_feature_parallel(df, title):
    # Draw sns.parallel_coordinates for features of the given df
    
    plt.figure(figsize=(15,6))
    parallel_coordinates(df, 'country', colormap=plt.get_cmap("tab20c"), lw=3)
    plt.title(title)
    plt.xlabel("Dates")
    plt.ylabel("New cases")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig('graph.png')
    plt.show()


# In[ ]:


df3 = df3.dropna()
df3 = df3.reset_index(drop=False)


# In[ ]:


# Countries with delay 1 day
df3_add = df2_add.T
df3_add.columns = forecast_dates_add
df3_add = df3_add.dropna().reset_index(drop=False)
df3_add


# In[ ]:


# Calculation slope of all trends
df3['slope_n'] = 0
df3['slope'] = 0
x = np.array(range(7))
for i in range(len(df3)):
    y = df3.iloc[i, 1:8].astype(float).values
    df3.loc[i,'slope'] = linregress(x, y)[0]
    df3.loc[i,'slope_n'] = df3.loc[i,ds_end] - df3.loc[i,ds_start]
df3


# In[ ]:


# Calculation slope of trends for countries with delay 1 day
df3_add['slope_n'] = 0
df3_add['slope'] = 0
x = np.array(range(7))
for i in range(len(df3_add)):
    y = df3_add.iloc[i, 1:8].astype(float).values
    df3_add.loc[i,'slope'] = linregress(x, y)[0]
    df3_add.loc[i,'slope_n'] = df3_add.loc[i,ds_end_add] - df3_add.loc[i,ds_start_add]
df3_add


# In[ ]:


df3.sort_values(by=['country'])


# In[ ]:


# Turkey is abnormal (Prophet model is not adequate)
counties_non_adequate = ['Paraguay', 'Turkey', 'China', 'Peru', 'Nigeria']
#df3 = df3[df3['country'] != 'Turkey']
df3 = df3[~(df3['country'].isin(counties_non_adequate + df2_more_day_delay_countries_list))]
df3.country.tolist()


# In[ ]:


# Selection in df3 only European countries
country_eu = ['Albania', 'Austria', 'Belarus', 'Belgium', 'Bulgaria', 'Croatia', 'Czechia', 
              'Denmark', 'Estonia', 'Finland', 'France', 'Georgia', 'Germany', 'Greece',
              'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Moldova', 'Netherlands',
              'Norway', 'Poland', 'Portugal', 'Romania', 'Serbia',
              'Slovakia', 'Slovenia', 'Turkey', 'Ukraine', 'United Kingdom']


# In[ ]:


df3_eu = df3[df3['country'].isin(country_eu)].reset_index(drop=True)
df3_eu[['country', 'slope']].sort_values(by=['slope'], ascending=False)


# In[ ]:


title = f"COVID-19 trends of the most of European countries with negative slope for {ds_start} - {ds_end}"
plot_feature_parallel(df3_eu[df3_eu['slope'] < 0][['country'] + forecast_dates], title)


# In[ ]:


df3_for_UA = df3_eu[(df3_eu['slope'] > -100) & (df3_eu['slope'] < -1)]
df3_for_UA.sort_values(by=['slope'], ascending=False)


# In[ ]:


title = f"COVID-19 trends of several European countries with negative slope for {ds_start} - {ds_end}"
plot_feature_parallel(df3_for_UA[['country'] + forecast_dates], title)


# ### 4.2. Mapping the slopes of country trends <a class="anchor" id="4.2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


df4 = pd.concat([df3[['country', 'slope']], df3_add[['country', 'slope']]]).reset_index(drop=True)
df4.sort_values(by=['slope'], ascending=False).style.set_caption('The slope of all countries trends')


# In[ ]:


df4_eu = df4[df4['country'].isin(country_eu)].reset_index(drop=True)
df4_eu[['country', 'slope']].sort_values(by=['slope'], ascending=False).style.set_caption('The slope of European countries trends')


# In[ ]:


# Download coordinates of the capitals of countries
world_coordinates = pd.read_csv('../input/world-cities-database/worldcities.csv')
world_coordinates = world_coordinates[world_coordinates['capital'] == 'primary']
non_main_capital_now = ['Bujumbura', 'The Hague', 'Cape Town', 'Bloemfontein']
world_coordinates = world_coordinates[~world_coordinates['city_ascii'].isin(non_main_capital_now)]
world_coordinates = world_coordinates[['city_ascii', 'lat', 'lng', 'country']]
world_coordinates


# In[ ]:


# Merging the coordinates dataframe with original dataframe
world_data = pd.merge(df4, world_coordinates, on='country', how='left').dropna().reset_index(drop=True)
world_data


# In[ ]:


world_data_grow = world_data[world_data['slope'] >= 0]
world_data_reduce = world_data[world_data['slope'] < 0]
world_data_reduce['slope'] = world_data_reduce['slope'].abs()


# In[ ]:


print(f'Creation and display map with layers control window for screenshots with a small scale of this map (COVID-19 new cases forecasting: {ds_start} - {ds_end})')


# In[ ]:


# Create map and display it
# Big circle - for screenshots of the map on a small scale with control window
world_map = folium.Map(location=[10, -20], zoom_start=2.3, tiles='Stamen Toner')
r0 = 0.75
r1 = 0.005  # 0.02
group0 = folium.FeatureGroup(name='<span style=\\"color: red;\\">red circles</span>')
for lat, lon, value, name in zip(world_data_grow['lat'], world_data_grow['lng'], world_data_grow['slope'], world_data_grow['country']):
    folium.CircleMarker([lat, lon],
                        radius=r0 + value*r1,
                        popup = ('<strong>country</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>slope trend</strong>: ' + str(value) + '<br>'),
                        color='red',
                        
                        fill_color='red',
                        fill_opacity=0.7 ).add_to(group0)
group0.add_to(world_map)

group1 = folium.FeatureGroup(name='<span style=\\"color: red;\\">blue circles</span>')
for lat, lon, value, name in zip(world_data_reduce['lat'], world_data_reduce['lng'], world_data_reduce['slope'], world_data_reduce['country']):
    folium.CircleMarker([lat, lon],
                        radius=r0 + value*r1,
                        popup = ('<strong>country</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>slope trend</strong>: ' + str(value) + '<br>'),
                        color='blue',
                        
                        fill_color='blue',
                        fill_opacity=0.7 ).add_to(group1)
group1.add_to(world_map)
folium.map.LayerControl('topright', collapsed=False).add_to(world_map)


world_map


# In[ ]:


print(f'Creation and display map with layers control window for screenshots with a big scale of this map (COVID-19 new cases forecasting: {ds_start} - {ds_end})')


# In[ ]:


# Create map and display it
# Big circle - for screenshots of the map on a big scale
world_map = folium.Map(location=[10, -20], zoom_start=2.3,tiles='Stamen Toner')
r0 = 2
r1 = 0.004  # 0.03
for lat, lon, value, name in zip(world_data_grow['lat'], world_data_grow['lng'], world_data_grow['slope'], world_data_grow['country']):
    folium.CircleMarker([lat, lon],
                        radius=r0 + value*r1,
                        popup = ('<strong>country</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>slope trend</strong>: ' + str(value) + '<br>'),
                        color='red',
                        
                        fill_color='red',
                        fill_opacity=0.7 ).add_to(world_map)
    
for lat, lon, value, name in zip(world_data_reduce['lat'], world_data_reduce['lng'], world_data_reduce['slope'], world_data_reduce['country']):
    folium.CircleMarker([lat, lon],
                        radius=r0 + value*r1,
                        popup = ('<strong>country</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>slope trend</strong>: ' + str(value) + '<br>'),
                        color='blue',
                        
                        fill_color='blue',
                        fill_opacity=0.7 ).add_to(world_map)

world_map


# I hope you find this notebook and datasets useful and enjoyable.

# Your comments and feedback are most welcome.

# [Go to Top](#0)
