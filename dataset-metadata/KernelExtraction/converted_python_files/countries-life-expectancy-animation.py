#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:844f1af6-5579-4cbd-85fb-d657d88b308b.png)
# 
# Animation is very fun and very cool to do to analyzing for your data!

# Thank you [Naman Manchanda](https://www.kaggle.com/namanmanchanda) for helping me. Be sure to check out some of his notebooks! Don't forget to check out [Countries Life Expectancy](https://www.kaggle.com/brendan45774/countries-life-expectancy) df and give it an upvote!

# # Installing Environment

# In[ ]:


pip install pandas-alive


# In[ ]:


pip install bar_chart_race


# # Importing Libraries

# In[ ]:


import numpy as np
import pandas as pd
import pandas_alive

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'notebook')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv("../input/countries-life-expectancy/Life expectancy.csv")


# In[ ]:


import pandas_profiling
report = pandas_profiling.ProfileReport(df)
from IPython.display import display
display(report)


# In[ ]:


df = pd.read_csv("../input/countries-life-expectancy/Life expectancy.csv")
df.head()


# # Bar Graph Race

# 2 different ways to do a bar graph

# In[ ]:


fig = px.bar(df, x="Entity", y="Life expectancy", color="Entity",
  animation_frame="Year", animation_group="Entity", range_y=[0,210])
fig.show()


# In[ ]:


fig = px.bar(df, x="Entity", y="Life expectancy", color="Entity",
  animation_frame="Year", animation_group="Entity")
fig.show()


# In[ ]:


dfw = pd.read_csv("../input/countries-life-expectancy/Life expectancy.csv",index_col=0,parse_dates=[0],thousands=',')

dfw.fillna(0).plot_animated(period_fmt="%Y",title='Countries Life Expectancy 1800-2017')


# # Scatter Plot Timeline

# In[ ]:


px.scatter(df, x="Year", y="Life expectancy", animation_frame="Year", animation_group="Entity",
           size="Life expectancy", color="Entity", hover_name="Entity",
           log_x=True, size_max=55, range_x=[1800,2017], range_y=[1,110])


# # Creating Animated Plots with Pandas_Alive (Look at the output part to see the charts!)

# In[ ]:


df = pd.read_csv('../input/countries-life-expectancy/Life expectancy.csv',parse_dates=['Year'])

# Only years from 1800 onwards
df = df[df['Year'].astype(int) >= 1800]

# Convert Year column to datetime
df['Year'] = pd.to_datetime(df['Year'])

display(df)


# **Bar Race Chart!**

# In[ ]:


# Pivot data to turn from `long` to `wide` format
pivoted_df = df.pivot(index='Year',columns='Entity',values='Life expectancy').fillna(0)

display(pivoted_df.head(5))


# * n_visible - Change the number of visible bars on the plot
# * period_fmt - Change the way the date is represented on the plot (eg, '%d/%m/%Y')
# * title - Set a title for the plot
# * fixed_max - Set the x-axis to be fixed from the lowest - biggest number

# In[ ]:


pivoted_df.plot_animated(filename='population-over-time-bar-chart-race.gif',n_visible=10,period_fmt="%Y",title='Top 10 Life Expectancy Countries 1800-2016',fixed_max=True)


# **Pie Chart**

# In[ ]:


pivoted_df.plot_animated(filename='example-pie-chart.gif',kind="pie",rotatelabels=True,period_label={'x':0,'y':0})


# **Animated Line Chart!**

# In[ ]:


total_df = pivoted_df.sum(axis=1)

display(total_df)


# In[ ]:


pivoted_df.diff().fillna(0).plot_animated(filename='different-countries-line-chart.gif',kind='line',period_label={'x':0.25,'y':0.9})


# In[ ]:


total_df.plot_animated(kind='line',filename="total-population-over-time-line.gif",period_fmt="%Y",title="Total Population Over Time")


# In[ ]:


import matplotlib.pyplot as plt

with plt.xkcd():
    animated_line_chart = total_df.plot_animated(filename='xkcd-line-plot.gif',kind='line',period_label=False,title="Total Population Over Time")


# **Combined Animation Chart!**

# In[ ]:


bar_chart_race = pivoted_df.plot_animated(n_visible=10,period_fmt="%Y",title='Top 10 Populous Countries 1800-2016')
animated_line_chart = total_df.plot_animated(kind='line',period_label=False,title="Total Population Over Time")

pandas_alive.animate_multiple_plots('population-combined-charts.gif',[bar_chart_race,animated_line_chart])


# # If you like this notebook, please give an Upvote! Don't forget to check out my other notebooks too!
# 
# * [ConnectX Baseline](https://www.kaggle.com/brendan45774/connectx-baseline)
# * [Countries Life Expectancy Animation](https://www.kaggle.com/brendan45774/countries-life-expectancy-animation)
# * [Data Visuals - Matplotlib](http://www.kaggle.com/brendan45774/data-visuals-matplotlib)
# * [Digit Recognizer Solution](http://www.kaggle.com/brendan45774/digit-recognizer-solution)
# * [Dictionary and Pandas Cheat sheet](https://www.kaggle.com/brendan45774/dictionary-and-pandas-cheat-sheet)
# * [EDA Tutorial Hollywood Movies](https://www.kaggle.com/brendan45774/eda-tutorial-hollywood-movies)
# * [Getting started with Matplotlib](http://www.kaggle.com/brendan45774/getting-started-with-matplotlib)
# * [Guide to Matplotlib Image](https://www.kaggle.com/brendan45774/guide-to-matplotlib-image)
# * [HOG features - Histogram of Oriented Gradients](https://www.kaggle.com/brendan45774/hog-features-histogram-of-oriented-gradients)
# * [How to get the lowest score](https://www.kaggle.com/brendan45774/how-to-get-the-lowest-score)
# * [House predict solution](http://www.kaggle.com/brendan45774/house-predict-solution)
# * [K-Means Clustering (Image Compression)](https://www.kaggle.com/brendan45774/k-means-clustering-image-compression)
# * [Kuzushiji-MNIST Panda](http://www.kaggle.com/brendan45774/kuzushiji-mnist-panda)
# * [Plotly Coronavirus (Covid-19)](https://www.kaggle.com/brendan45774/plotly-coronavirus-covid-19)
# * [Titanic Top Solution](http://www.kaggle.com/brendan45774/titanic-top-solution)
# * [Titanic Data Solution](http://www.kaggle.com/brendan45774/titanic-data-solution)
# * [Word Cloud - Analyzing Names](https://www.kaggle.com/brendan45774/word-cloud-analyzing-names)
