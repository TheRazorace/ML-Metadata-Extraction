#!/usr/bin/env python
# coding: utf-8

#  **DATA SCIENCE WITH COUNTRIES OF WORLD**
#     

# ![](http://peoplemagazine.com.pk/wp-content/uploads/2017/11/o-WORLD-facebook.jpg)

# **Importing Libraries**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# import warnings
import warnings
# ignore warnings
warnings.filterwarnings("ignore")
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Reading Countries of World Data**

# In[ ]:


world = pd.read_csv("../input/countries of the world.csv")
world.head(10)


# **First of all , We have to know our data types.As you know There will be some problems with our data types and we will change some of them in order to use them effectively.**

# In[ ]:


world.info()
world.dtypes


# **There are problems with our column names.We have to fix them.**

# In[ ]:


world.columns = (["country","region","population","area","density","coastline","migration","infant_mortality","gdp","literacy","phones","arable","crops","other","climate","birthrate","deathrate","agriculture","industry","service"])


# **We have to change our datatype to "category" and "float" to use easily**

# In[ ]:


world.country = world.country.astype('category')
world.region = world.region.astype('category')
world.density = world.density.str.replace(",",".").astype(float)
world.coastline = world.coastline.str.replace(",",".").astype(float)
world.migration = world.migration.str.replace(",",".").astype(float)
world.infant_mortality = world.infant_mortality.str.replace(",",".").astype(float)
world.literacy = world.literacy.str.replace(",",".").astype(float)
world.phones = world.phones.str.replace(",",".").astype(float)
world.arable = world.arable.str.replace(",",".").astype(float)
world.crops = world.crops.str.replace(",",".").astype(float)
world.other = world.other.str.replace(",",".").astype(float)
world.climate = world.climate.str.replace(",",".").astype(float)
world.birthrate = world.birthrate.str.replace(",",".").astype(float)
world.deathrate = world.deathrate.str.replace(",",".").astype(float)
world.agriculture = world.agriculture.str.replace(",",".").astype(float)
world.industry = world.industry.str.replace(",",".").astype(float)
world.service = world.service.str.replace(",",".").astype(float)


# In[ ]:


world.info()


# **Yes our datatypes are perfect now. We can go on.**

# **MISSING VALUES**

# **There are some Nan values and these will be a problem for us. So we have fix them. Now Let's learn how many Nan values in our dataset.**

# In[ ]:


missing = world.isnull().sum()
missing


# **Let's fill the missing values by using fillna method.We will fill Nan values with mean of columns**

# In[ ]:


world.fillna(world.mean(),inplace=True)


# **In the region column , There are some spaces before and after of the regions.We can fix this issue with "strip()" method**

# In[ ]:


world.region = world.region.str.strip()


# **We can group our data by regions and let's look at the mean of each region**

# In[ ]:


group = world.groupby("region")
group.mean()


# **Now our data is ready , Data preprocessing is finished. Our data is clean and let's look at World data**

# In[ ]:


world.head(10)


# **Data visualization is the presentation of data in a pictorial or graphical format. It enables decision makers to see analytics presented visually, so they can grasp difficult concepts or identify new patterns. NOW Let's make some visualization**

# In[ ]:


region = world.region.value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=region.index,y=region.values)
plt.xticks(rotation=45)
plt.ylabel('Number of countries')
plt.xlabel('Region')
plt.title('Number of Countries by REGİON',color = 'red',fontsize=20)


# **As you see on graph , number one region is Sub-saharan Africa.Let's look at GDP, infant mortality , birthrate and deathrate of countries**

# In[ ]:


sns.set(style="darkgrid",font_scale=1.5)
f, axes = plt.subplots(2,2,figsize=(15,10))

sns.distplot(world.infant_mortality,bins=20,kde=False,color="y",ax=axes[0,0])
sns.distplot(world.gdp,hist=False,rug=True,color="r",ax=axes[0,1])
sns.distplot(world.birthrate,hist=False,color="g",kde_kws={"shade":True},ax=axes[1,0])
sns.distplot(world.deathrate,color="m",ax=axes[1,1])


# In[ ]:


sns.boxplot(x="region",y="gdp",data=world,width=0.7,palette="Set3",fliersize=5)
plt.xticks(rotation=90)
plt.title("GDP BY REGİON",color="red")


# **Now ,We will look at correlation of features.Correlation is a statistical technique that can show whether and how strongly pairs of variables are related. For example, height and weight are related; taller people tend to be heavier than shorter people. When two sets of data are strongly linked together we say they have a High Correlation.**

# In[ ]:


world.corr()


# **Let's visualize the correlations with heatmap**

# In[ ]:


f,ax = plt.subplots(figsize=(18, 16))
sns.heatmap(world.corr(), annot=True, linewidths=.8, fmt= '.1f',ax=ax)


# **There are relationships with these features (gdp,infant mortality,birthrate,phones,literacy,service)**

# In[ ]:


x = world.loc[:,["region","gdp","infant_mortality","birthrate","phones","literacy","service"]]
sns.pairplot(x,hue="region",palette="inferno")


# **GDP is key value. As you see, there are positive correlation between "Phones" and "Service". Let's look these two features deeply**

# In[ ]:


sns.lmplot(x="gdp",y="phones",data=world,height=10)
sns.lmplot(x="gdp",y="service",data=world,height=10)


# **More money more phones :)** **, Let's make some interactive visualization with plotly. I like plotly**

# In[ ]:


gdp=world.sort_values(["gdp"],ascending=False)


# **Let's look at birthrate and deathrate of top 100 countries and last 100 countries interactively**

# In[ ]:


# prepare data frame
df = gdp.iloc[:100,:]

# Creating trace1
trace1 = go.Scatter(
                    x = df.gdp,
                    y = df.birthrate,
                    mode = "lines",
                    name = "Birthrate",
                    marker = dict(color = 'rgba(235,66,30, 0.8)'),
                    text= df.country)
# Creating trace2
trace2 = go.Scatter(
                    x = df.gdp,
                    y = df.deathrate,
                    mode = "lines+markers",
                    name = "Deathrate",
                    marker = dict(color = 'rgba(10,10,180, 0.8)'),
                    text= df.country)
z = [trace1, trace2]
layout = dict(title = 'Birthrate and Deathrate of World Countries (Top 100)',
              xaxis= dict(title= 'GDP',ticklen= 5,zeroline= False)
             )
fig = dict(data = z, layout = layout)
iplot(fig)


# In[ ]:


# prepare data frame
df = gdp.iloc[77:177,:]

# Creating trace1
trace1 = go.Scatter(
                    x = df.gdp,
                    y = df.birthrate,
                    mode = "lines",
                    name = "Birthrate",
                    marker = dict(color = 'rgba(235,66,30, 0.8)'),
                    text= df.country)
# Creating trace2
trace2 = go.Scatter(
                    x = df.gdp,
                    y = df.deathrate,
                    mode = "lines+markers",
                    name = "Deathrate",
                    marker = dict(color = 'rgba(10,10,180, 0.8)'),
                    text= df.country)
z = [trace1, trace2]
layout = dict(title = 'Birthrate and Deathrate Percentage of World Countries (Last 100)',
              xaxis= dict(title= 'GDP',ticklen= 5,zeroline= False)
             )
fig = dict(data = z, layout = layout)
iplot(fig)


# **Let's look at percentage of agriculture , industry and service of top 100 and last 100 countries interactively**

# In[ ]:


# prepare data frame
df = gdp.iloc[:100,:]

# Creating trace1
trace1 = go.Scatter(
                    x = df.gdp,
                    y = df.agriculture,
                    mode = "lines+markers",
                    name = "AGRICULTURE",
                    marker = dict(color = 'rgba(235,66,30, 0.8)'),
                    text= df.country)
# Creating trace2
trace2 = go.Scatter(
                    x = df.gdp,
                    y = df.industry,
                    mode = "lines+markers",
                    name = "INDUSTRY",
                    marker = dict(color = 'rgba(10,10,180, 0.8)'),
                    text= df.country)
# Creating trace3
trace3 = go.Scatter(
                    x = df.gdp,
                    y = df.service,
                    mode = "lines+markers",
                    name = "SERVICE",
                    marker = dict(color = 'rgba(10,250,60, 0.8)'),
                    text= df.country)


z = [trace1, trace2,trace3]
layout = dict(title = 'Service , Industry and Agriculture Percentage of World Countries (TOP 100)',
              xaxis= dict(title= 'GDP',ticklen= 5,zeroline= False)
             )
fig = dict(data = z, layout = layout)
iplot(fig)


# In[ ]:


# prepare data frame
df = gdp.iloc[77:,:]

# Creating trace1
trace1 = go.Scatter(
                    x = df.gdp,
                    y = df.agriculture,
                    mode = "lines+markers",
                    name = "AGRICULTURE",
                    marker = dict(color = 'rgba(235,66,30, 0.8)'),
                    text= df.country)
# Creating trace2
trace2 = go.Scatter(
                    x = df.gdp,
                    y = df.industry,
                    mode = "lines+markers",
                    name = "INDUSTRY",
                    marker = dict(color = 'rgba(10,10,180, 0.8)'),
                    text= df.country)
# Creating trace3
trace3 = go.Scatter(
                    x = df.gdp,
                    y = df.service,
                    mode = "lines+markers",
                    name = "SERVICE",
                    marker = dict(color = 'rgba(10,250,60, 0.8)'),
                    text= df.country)


z = [trace1, trace2,trace3]
layout = dict(title = 'Service , Industry and Agriculture Percentage of World Countries (LAST 100)',
              xaxis= dict(title= 'GDP',ticklen= 5,zeroline= False)
             )
fig = dict(data = z, layout = layout)
iplot(fig)


# **Let's look at agriculture and service features of top 7 countries (literacy)**

# In[ ]:


lit = world.sort_values("literacy",ascending=False).head(7)


# In[ ]:


trace1 = go.Bar(
                x = lit.country,
                y = lit.agriculture,
                name = "agriculture",
                marker = dict(color = 'rgba(255, 20, 20, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = lit.gdp)
trace2 = go.Bar(
                x = lit.country,
                y = lit.service,
                name = "service",
                marker = dict(color = 'rgba(20, 20, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = lit.gdp)
data = [trace1, trace2]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# **Let's look at industry and service features of top 7 countries (literacy)**

# In[ ]:


x = lit.country

trace1 = {
  'x': x,
  'y': lit.industry,
  'name': 'industry',
  'type': 'bar'
};
trace2 = {
  'x': x,
  'y': lit.service,
  'name': 'service',
  'type': 'bar'
};
data = [trace1, trace2];
layout = {
  'xaxis': {'title': 'Top 7 country'},
  'barmode': 'relative',
  'title': 'industry and service percentage of top 7 country (literacy)'
};
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# In[ ]:


fig = {
  "data": [
    {
      "values": lit.gdp,
      "labels": lit.country,
      "domain": {"x": [0, .5]},
      "name": "GDP percentage of",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"GDP of top 7 country(literacy)",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "GDP",
                "x": 0.22,
                "y": 0.5
            },
        ]
    }
}
iplot(fig)


# **Let's make bubble chart with service and literacy of countries**

# In[ ]:


lite = world.sort_values("literacy",ascending=False).head(15)
data = [
    {
        'y': lite.service,
        'x': lite.index,
        'mode': 'markers',
        'marker': {
            'color': lite.service,
            'size': lite.literacy,
            'showscale': True
        },
        "text" :  lite.country    
    }
]
iplot(data)


# **Let's make world map with plotly interactively**

# In[ ]:


#Population per country
data = dict(type='choropleth',
locations = world.country,
locationmode = 'country names', z = world.population,
text = world.country, colorbar = {'title':'Population'},
colorscale = 'Blackbody', reversescale = True)
layout = dict(title='Population per country',
geo = dict(showframe=False,projection={'type':'natural earth'}))
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)


# In[ ]:


#Population per country
data = dict(type='choropleth',
locations = world.country,
locationmode = 'country names', z = world.infant_mortality,
text = world.country, colorbar = {'title':'Infant Mortality'},
colorscale = 'YlOrRd', reversescale = True)
layout = dict(title='Infant Mortality per Country',
geo = dict(showframe=False,projection={'type':'natural earth'}))
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)


# In[ ]:


#Population per country
data = dict(type='choropleth',
locations = world.country,
locationmode = 'country names', z = world.gdp,
text = world.country, colorbar = {'title':'GDP'},
colorscale = 'Hot', reversescale = True)
layout = dict(title='GDP of World Countries',
geo = dict(showframe=False,projection={'type':'natural earth'}))
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)


# Please **Upvote** if you like this kernel

# In[ ]:




