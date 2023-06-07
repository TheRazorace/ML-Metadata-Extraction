#!/usr/bin/env python
# coding: utf-8

# # Happiness Data Analysis

# # INTRODUCTION
# * [About this Datasets](#15)
# * [Import Library and Data](#16)
# * [Data Cleaning](#1)
# * [Correlation Map](#2)
# * [Ranking For Economy](#3)
# * [Score Level According to Region](#4)
# * [Created Level for Classification](#5)
# * [Pie Charts Which is Created Level](#6)
# * [Swarm Plot Example](#7)
# * [Joint Plot](#8)
# * [Table for Country Levels](#9)
# * [Swarm Visualization for Levels](#10)
# * [Filter Which is According to Average](#11)
# * [Comparisons](#12)
# * [Joint Plot for Level](#13)
# * [Boxen Plot Examples](#14)

# <a id="15"></a> 
# ## About this Datasets

# **Context**
# 
# The World Happiness Report is a landmark survey of the state of global happiness. The first report was published in 2012, the second in 2013, the third in 2015, and the fourth in the 2016 Update. The World Happiness 2017, which ranks 155 countries by their happiness levels, was released at the United Nations at an event celebrating International Day of Happiness on March 20th. The report continues to gain global recognition as governments, organizations and civil society increasingly use happiness indicators to inform their policy-making decisions. Leading experts across fields – economics, psychology, survey analysis, national statistics, health, public policy and more – describe how measurements of well-being can be used effectively to assess the progress of nations. The reports review the state of happiness in the world today and show how the new science of happiness explains personal and national variations in happiness.
# 
# **Content**
# 
# The happiness scores and rankings use data from the Gallup World Poll. The scores are based on answers to the main life evaluation question asked in the poll. This question, known as the Cantril ladder, asks respondents to think of a ladder with the best possible life for them being a 10 and the worst possible life being a 0 and to rate their own current lives on that scale. The scores are from nationally representative samples for the years 2013-2016 and use the Gallup weights to make the estimates representative. The columns following the happiness score estimate the extent to which each of six factors – economic production, social support, life expectancy, freedom, absence of corruption, and generosity – contribute to making life evaluations higher in each country than they are in Dystopia, a hypothetical country that has values equal to the world’s lowest national averages for each of the six factors. They have no impact on the total score reported for each country, but they do explain why some countries rank higher than others.
# 
# **Inspiration**
# 
# What countries or regions rank the highest in overall happiness and each of the six factors contributing to happiness? How did country ranks or scores change between the 2015 and 2016 as well as the 2016 and 2017 reports? Did any country experience a significant increase or decrease in happiness?
# 
# **What is Dystopia?**
# 
# Dystopia is an imaginary country that has the world’s least-happy people. The purpose in establishing Dystopia is to have a benchmark against which all countries can be favorably compared (no country performs more poorly than Dystopia) in terms of each of the six key variables, thus allowing each sub-bar to be of positive width. The lowest scores observed for the six key variables, therefore, characterize Dystopia. Since life would be very unpleasant in a country with the world’s lowest incomes, lowest life expectancy, lowest generosity, most corruption, least freedom and least social support, it is referred to as “Dystopia,” in contrast to Utopia.
# 
# **What are the residuals?**
# 
# 
# The residuals, or unexplained components, differ for each country, reflecting the extent to which the six variables either over- or under-explain average 2014-2016 life evaluations. These residuals have an average value of approximately zero over the whole set of countries. Figure 2.2 shows the average residual for each country when the equation in Table 2.1 is applied to average 2014- 2016 data for the six variables in that country. We combine these residuals with the estimate for life evaluations in Dystopia so that the combined bar will always have positive values. As can be seen in Figure 2.2, although some life evaluation residuals are quite large, occasionally exceeding one point on the scale from 0 to 10, they are always much smaller than the calculated value in Dystopia, where the average life is rated at 1.85 on the 0 to 10 scale.
# 
# What do the columns succeeding the Happiness Score(like Family, Generosity, etc.) describe?
# 
# The following columns: GDP per Capita, Family, Life Expectancy, Freedom, Generosity, Trust Government Corruption describe the extent to which these factors contribute in evaluating the happiness in each country. The Dystopia Residual metric actually is the Dystopia Happiness Score(1.85) + the Residual value or the unexplained value for each country as stated in the previous answer.
# 
# If you add all these factors up, you get the happiness score so it might be un-reliable to model them to predict Happiness Scores.

# <a id="16"></a> 
# # Import  Library

# In[ ]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sbr
import numpy as np
import os
import warnings
print(os.listdir("../input"))


# # Import Datasets

# In[ ]:


d2015=pd.read_csv("../input/2015.csv")
d2016=pd.read_csv("../input/2016.csv")
d2017=pd.read_csv("../input/2017.csv")


# Three data sets aren't including same data. Let's look those data sets.

# ### 2015 Year Overview

# In[ ]:


d2015.info()


# ### 2016 Year Overview

# In[ ]:


d2016.info()


# ### 2017 Year Overview

# In[ ]:


d2017.info()


# <a id="1"></a> 
# # Data Cleaning

# ## I deleted some columns for to obtain data that is common to 3 years

# My first goal is that the three data sets contain the same data.

# In[ ]:


del d2017["Whisker.high"]


# In[ ]:


del d2017["Whisker.low"]


# In[ ]:


d2015.drop(columns="Standard Error",inplace=True,errors="ignore")


# In[ ]:


d2016.drop(columns="Lower Confidence Interval",inplace=True,errors="ignore")
d2016.drop(columns="Upper Confidence Interval",inplace=True,errors="ignore")


# ## I changed columns name for be to same in three data.

# In[ ]:


d2015=d2015.rename(columns={"Happiness Rank":"Rank","Happiness Score":"Score","Economy (GDP per Capita)":"Economy","Health (Life Expectancy)":"Health","Trust (Government Corruption)":"Trust","Dystopia Residual":"Dystopia_Residual"})
d2016=d2016.rename(columns={"Happiness Rank":"Rank","Happiness Score":"Score","Economy (GDP per Capita)":"Economy","Health (Life Expectancy)":"Health","Trust (Government Corruption)":"Trust","Dystopia Residual":"Dystopia_Residual"})
d2017=d2017.rename(columns={"Happiness.Rank":"Rank","Happiness.Score":"Score","Economy..GDP.per.Capita.":"Economy","Health..Life.Expectancy.":"Health","Trust..Government.Corruption.":"Trust","Dystopia.Residual":"Dystopia_Residual"})


# ## Columns name same for three years data, now.

# In[ ]:


d2015.head()


# In[ ]:


d2016.head()


# In[ ]:


d2017.head()


# ## I deleted some countries for be same data in three years.

# I deleted different countries in 2017 year data.

# In[ ]:


d2017.drop(d2017.index[154],inplace=True)
d2017.drop(d2017.index[146],inplace=True)
d2017.drop(d2017.index[138],inplace=True)
d2017.drop(d2017.index[112],inplace=True)
d2017.drop(d2017.index[110],inplace=True)
d2017.drop(d2017.index[49],inplace=True)
d2017.sort_values(by="Country").head()


# I deleted different countries in 2015 year data.

# In[ ]:


d2015.drop(d2015.index[147],inplace=True)
d2015.drop(d2015.index[139],inplace=True)
d2015.drop(d2015.index[125],inplace=True)
d2015.drop(d2015.index[100],inplace=True)
d2015.drop(d2015.index[98],inplace=True)
d2015.drop(d2015.index[96],inplace=True)
d2015.drop(d2015.index[93],inplace=True)
d2015.drop(d2015.index[39],inplace=True)
d2015.drop(d2015.index[21],inplace=True)
d2015.sort_values(by="Country").head()


# I deleted different countries in 2016 year data.

# In[ ]:


d2016.drop(d2016.index[142],inplace=True)
d2016.drop(d2016.index[137],inplace=True)
d2016.drop(d2016.index[112],inplace=True)
d2016.drop(d2016.index[101],inplace=True)
d2016.drop(d2016.index[75],inplace=True)
d2016.drop(d2016.index[51],inplace=True)
d2016.drop(d2016.index[39],inplace=True)
d2016.drop(d2016.index[14],inplace=True)
d2016.sort_values(by="Country").head()


# ### All of data same for three years, now. Let's Look.

# In[ ]:


d2015.info()


# In[ ]:


d2016.info()


# In[ ]:


d2017.info()


# In[ ]:


new2015=d2015.sort_values(by="Country").copy()
new2016=d2016.sort_values(by="Country").copy()
new2017=d2017.sort_values(by="Country").copy()


# In[ ]:


new2017.head()


# In[ ]:


new2015.index=range(len(new2015))
new2016.index=range(len(new2016))
new2017.index=range(len(new2017))
new2015.head(3)


# ## I created a new data. Includes sum of three years data.

# uni_data is new data that sum all columns.

# In[ ]:


uni_data=pd.DataFrame()
uni_data["Country"]=new2015.Country
uni_data["Region"]=new2015.Region
uni_data["Rank"]=new2015.Rank
uni_data["Score"]=new2015.Score+new2016.Score+new2017.Score
uni_data["Economy"]=new2015.Economy+new2016.Economy+new2017.Economy
uni_data["Family"]=new2015.Family+new2016.Family+new2017.Family
uni_data["Health"]=new2015.Health+new2016.Health+new2017.Health
uni_data["Freedom"]=new2015.Freedom+new2016.Freedom+new2017.Freedom
uni_data["Trust"]=new2015.Trust+new2016.Trust+new2017.Trust
uni_data["Generosity"]=new2015.Generosity+new2016.Generosity+new2017.Generosity
uni_data["Dystopia_Residual"]=new2015["Dystopia_Residual"]+new2016["Dystopia_Residual"]+new2017["Dystopia_Residual"]
uni_data.head(10)


# In[ ]:


uni_data.tail(5)


# In[ ]:


uni_data.info()


# In[ ]:


uni_data.describe()


# In[ ]:


uni_data.corr()


# <a id="2"></a> 
# # Correlation Map Visualization

# In[ ]:


f,ax=plt.subplots(figsize=(18,18))
sbr.heatmap(uni_data.corr(),annot=True,fmt=".1f",linewidths=.2,cmap="Spectral",ax=ax,linecolor="black")
plt.show()


# <a id="3"></a> 
# # Ranking for Economy

# Top 5 economy

# In[ ]:


uni_data.sort_values(by="Economy",ascending=False).head()


# Lowest 5 Economy

# In[ ]:


uni_data.sort_values(by="Economy").head()


# Top 5 Score

# In[ ]:


uni_data.sort_values(by="Score",ascending=False).head()


# **Melted Data for Region and Country**

# In[ ]:


melted=pd.melt(frame=uni_data,id_vars="Country",value_vars=["Region"])
melted


# <a id="4"></a> 
# # Score Level according to Region

# In[ ]:


f,ax=plt.subplots(figsize=(15,20))
sbr.boxplot(x=uni_data.Score,y=uni_data.Region,data=uni_data)
sbr.swarmplot(x=uni_data.Score,y=uni_data.Region,data=uni_data,color=".10",size=8)
warnings.filterwarnings("ignore")


# In[ ]:


uni_data.set_index("Rank").sort_values(by="Rank",ascending="False").head(10)


# In[ ]:


uni_data[["Country","Economy","Trust"]].head(20)


# In[ ]:


uni_data["Economy"].describe()


# <a id="5"></a> 
# # Created Level for Classification

# ### I used to Averages and Standart Deviations for create levels.

# In[ ]:


stdscore=uni_data.Score.std()
scoremean=sum(uni_data.Score)/len(uni_data.Score)
print("Score Average: ",scoremean)
print("Score Standart Deviation: ",stdscore)


# In[ ]:


stdeco=uni_data["Economy"].std()
ecomean=sum(uni_data.Economy)/len(uni_data.Economy)
print("Economy Average: " ,ecomean)
print("Economy Standard Deviation: ",stdeco)


# In[ ]:


stdhealth=uni_data["Health"].std()
healthmean=sum(uni_data.Health)/len(uni_data.Health)
print("Health Average: ",healthmean)
print("Health Standart Deviation: ",stdhealth)


# In[ ]:


stdfamily=uni_data["Family"].std()
familymean=sum(uni_data.Family)/len(uni_data.Family)
print("Family Average: ",familymean)
print("Family Standart Deviation: ",stdfamily)


# ### I created new level for four columns includes "High","Normal" and "Low".

# In[ ]:


datamean=pd.DataFrame()
datamean["Country"]=uni_data.Country
datamean["Region"]=uni_data.Region
datamean["Score"]=uni_data.Score
datamean["Score_Level"]=["High" if i>scoremean+stdscore else "Normal" if (scoremean-stdscore)<i<(scoremean+stdscore) else "Low" for i in uni_data.Score]
datamean["Economy"]=uni_data.Economy
datamean["Economic_Level"]=["High" if i>ecomean+stdeco else "Normal" if (ecomean-stdeco)<i<(ecomean+stdeco) else "Low" for i in uni_data.Economy]
datamean["Health"]=uni_data.Health
datamean["Health_Level"]=["High" if i>healthmean+stdhealth else "Normal" if (healthmean-stdhealth)<i<(healthmean+stdhealth) else "Low" for i in uni_data.Health]
datamean["Family"]=uni_data.Family
datamean["Family_Level"]=["High" if i>familymean+stdfamily else "Normal" if (familymean-stdfamily)<i<(familymean+stdfamily) else "Low" for i in uni_data.Family]
datamean.head(10)


# In[ ]:


datamean.tail(10)


# <a id="6"></a> 
# # Pie Charts

# ## Pie Chart About Country Levels

# In[ ]:


labels=datamean.Score_Level.value_counts().index
colors=("gold","Green","Red")
explode=[0,0.1,0.15]
sizes=datamean.Score_Level.value_counts().values

plt.figure(figsize=(10,10))
plt.pie(sizes,explode=explode,colors=colors,labels=labels,shadow=True,autopct='%1.1f%%')
plt.title("Pie Chart According to Score Level",color="Black",fontsize=15)
warnings.filterwarnings("ignore")


# In[ ]:


labels=datamean.Economic_Level.value_counts().index
colors=("lightyellow","red","Yellowgreen")
explode=[0,0.1,0.15]
sizes=datamean.Economic_Level.value_counts().values

plt.figure(figsize=(10,10))
plt.pie(sizes,explode=explode,colors=colors,labels=labels,shadow=True,autopct='%1.1f%%')
plt.title("Pie Chart According to Economic Level",color="Black",fontsize=15)
warnings.filterwarnings("ignore")


# In[ ]:


labels=datamean.Health_Level.value_counts().index
colors=["lightgreen","red","yellowgreen"]
explode=[0,0.1,0.15]
sizes=datamean.Health_Level.value_counts().values

plt.figure(figsize=(10,10))
plt.pie(sizes,explode=explode,colors=colors,labels=labels,shadow=True,autopct="%1.1f%%")
plt.title("Pie Chart According to Health Level",color="Black",fontsize=15)
warnings.filterwarnings("ignore")


# In[ ]:


labels=datamean.Family_Level.value_counts().index
colors=["lightblue","red","yellowgreen"]
explode=[0,0.1,0.15]
sizes=datamean.Health_Level.value_counts().values

plt.figure(figsize=(10,10))
plt.pie(sizes,explode=explode,colors=colors,labels=labels,shadow=True,autopct="%1.1f%%")
plt.title("Pie Chart According to Family Level",color="Black",fontsize=15)
warnings.filterwarnings("ignore")


# In[ ]:


datamean.head()


# <a id="7"></a> 
# # Swarm Plot Examples

# ## Swarm plot for Score Level

# In[ ]:


f,ax=plt.subplots(figsize=(15,15))
sbr.swarmplot(x=datamean.Score_Level,y=datamean.Score,hue=datamean.Region,size=12)
warnings.filterwarnings("ignore")


# ## Swarm plot for Economic Level

# In[ ]:


f,ax=plt.subplots(figsize=(15,15))
sbr.swarmplot(x=datamean.Economic_Level,y=datamean.Economy,hue=datamean.Region,size=12)
warnings.filterwarnings("ignore")


# ## Swarm plot for Healt Level

# In[ ]:


f,ax=plt.subplots(figsize=(15,15))
sbr.swarmplot(x=datamean["Health_Level"],y=datamean.Health,hue=datamean.Region,size=12)
warnings.filterwarnings("ignore")


# ## Swarm plot for Family Level

# In[ ]:


f,ax=plt.subplots(figsize=(15,15))
sbr.swarmplot(x=datamean["Family_Level"],y=datamean.Family,hue=datamean.Region,size=12)
warnings.filterwarnings("ignore")


# <a id="8"></a> 
# # Joint Plot

# In[ ]:


sbr.jointplot(x=datamean.Health,y=datamean.Economy,data=datamean,kind="kde",space=0,color="g")
warnings.filterwarnings("ignore")


# <a id="9"></a> 
# # Table for Country Levels

# ## High Level Countries

# These countries are all values High Level

# In[ ]:


grpdata=datamean.set_index(["Score_Level","Economic_Level","Health_Level","Family_Level"])
grpdata.loc["High","High","High","High"]


# ## Normal Level Countries

# These Countries are all values Normal Level

# In[ ]:


grpdata.loc["Normal","Normal","Normal","Normal"]


# ## Low Level Countries

# These countries are all values Low Level

# In[ ]:


grpdata.loc["Low","Low","Low","Low"]


# <a id="10"></a> 
# # Swarm Visualization for Levels

# ### High Levels Countries

# In[ ]:


f,ax=plt.subplots(figsize=(10,8))
sbr.swarmplot(x=grpdata.loc["High","High","High","High"].Score,y=grpdata.loc["High","High","High","High"].Country,size=10,linewidth=1)
warnings.filterwarnings("ignore")


# ### Low Levels Countries

# In[ ]:


f,ax=plt.subplots(figsize=(10,6))
sbr.swarmplot(x=grpdata.loc["Low","Low","Low","Low"].Score,y=grpdata.loc["Low","Low","Low","Low"].Country,size=10,linewidth=1)
warnings.filterwarnings("ignore")


# <a id="11"></a> 
# # Filter Which is According to Average

# ## Filter for upper to average countries.

# In[ ]:


filter_eco=uni_data.Economy>sum(uni_data.Economy)/len(uni_data.Economy)
filter_health=uni_data.Health>sum(uni_data.Health)/len(uni_data.Health)
filter_trust=uni_data.Trust>sum(uni_data.Trust)/len(uni_data.Trust)
filter_family=uni_data.Family>sum(uni_data.Family)/len(uni_data.Family)
uni_data[filter_eco & filter_health & filter_trust & filter_family]


# ## Economic distribution of countries for above average.

# In[ ]:


f,ax=plt.subplots(figsize=(15,15))
sbr.barplot(x="Economy",y="Country",data=uni_data[filter_eco & filter_health & filter_trust & filter_family].sort_values(by="Economy",ascending=False))
warnings.filterwarnings("ignore")


# ## Filter for under to average countries.

# In[ ]:


filter_eco2=uni_data.Economy<sum(uni_data.Economy)/len(uni_data.Economy)
filter_health2=uni_data.Health<sum(uni_data.Health)/len(uni_data.Health)
filter_trust2=uni_data.Trust<sum(uni_data.Trust)/len(uni_data.Trust)
filter_family2=uni_data.Family<sum(uni_data.Family)/len(uni_data.Family)
uni_data[filter_eco2 & filter_health2 & filter_trust2 & filter_family2]


# ## Economic distribution of countries for under average.

# In[ ]:


f,ax=plt.subplots(figsize=(15,15))
sbr.barplot(x="Economy",y="Country",data=uni_data[filter_eco2 & filter_health2 & filter_trust2 & filter_family2].sort_values(by="Economy"))
warnings.filterwarnings("ignore")


# <a id="12"></a> 
# # Comparisons

# ## Comparison for two levels.

# In[ ]:


f,ax=plt.subplots(figsize=(10,10))
p1=sbr.kdeplot(uni_data[filter_eco & filter_health & filter_trust & filter_family].Economy,shade=True,color="g")
p1=sbr.kdeplot(uni_data[filter_eco2 & filter_health2 & filter_family2 & filter_trust2].Economy,shade=True,color="r")
warnings.filterwarnings("ignore")


# ## Comparison for three levels according to Score.

# In[ ]:


f,ax=plt.subplots(figsize=(10,10))
p2=sbr.kdeplot(grpdata.loc["High","High","High","High"].Score,color="g",shade=True)
p2=sbr.kdeplot(grpdata.loc["Normal","Normal","Normal","Normal"].Score,color="y",shade=True)
p2=sbr.kdeplot(grpdata.loc["Low","Low","Low","Low"].Score,color="r",shade=True)
warnings.filterwarnings("ignore")


# ## Comparison for three levels according to Family.

# In[ ]:


f,ax=plt.subplots(figsize=(10,10))
plt3=sbr.kdeplot(grpdata.loc["High","High","High","High"].Family,shade=True)
plt3=sbr.kdeplot(grpdata.loc["Normal","Normal","Normal","Normal"].Family,shade=True)
plt3=sbr.kdeplot(grpdata.loc["Low","Low","Low","Low"].Family,shade=True)
warnings.filterwarnings("ignore")


# <a id="13"></a> 
# # Joint Plot for Level

# In[ ]:


sbr.jointplot(x=grpdata.loc["High","High","High","High"].Health,y=grpdata.loc["High","High","High","High"].Economy,kind="kde",color="g")
sbr.jointplot(x=grpdata.loc["Normal","Normal","Normal","Normal"].Health,y=grpdata.loc["Normal","Normal","Normal","Normal"].Economy,kind="kde",color="y")
sbr.jointplot(x=grpdata.loc["Low","Low","Low","Low"].Health,y=grpdata.loc["Low","Low","Low","Low"].Economy,kind="kde",color="r")
warnings.filterwarnings("ignore")


# ### Filter for High Level Columns

# In[ ]:


f,ax=plt.subplots(figsize=(15,15))
sbr.swarmplot(x="Economy",y="Score_Level",hue="Country",data=datamean[filter_eco & filter_health & filter_family & filter_trust],size=15)
warnings.filterwarnings("ignore")


# <a id="14"></a> 
# # Boxen Plot Visualization

# In[ ]:


f,ax=plt.subplots(figsize=(17,12))
sbr.boxenplot(x="Economic_Level",y="Health",data=datamean,hue="Region")
warnings.filterwarnings("ignore")


# In[ ]:


f,ax=plt.subplots(figsize=(17,10))
sbr.boxenplot(x="Score_Level",y="Family",data=datamean,hue="Region")
warnings.filterwarnings("ignore")

