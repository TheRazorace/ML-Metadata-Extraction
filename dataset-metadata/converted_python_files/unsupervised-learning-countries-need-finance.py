#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # What is Clustering
# Cluster analysis or clustering is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense) to each other than to those in other groups (clusters). It is a main task of exploratory data analysis, and a common technique for statistical data analysis, used in many fields, including pattern recognition, image analysis, information retrieval, bioinformatics, data compression, computer graphics and machine learning. 
# 
# Clustering is the task of dividing the population or data points into a number of groups such that data points in the same groups are more similar to other data points in the same group and dissimilar to the data points in other groups. It is basically a collection of objects on the basis of similarity and dissimilarity between them.
# 
# ![](https://media.geeksforgeeks.org/wp-content/uploads/merge3cluster.jpg)
# 
# Ref: Wikipedia & https://www.geeksforgeeks.org/clustering-in-machine-learning/
# 

# # Different type of Clustering Alogrithm
# 1. Affinity Propagation
# 2. Agglomerative Clustering
# 3. BIRCH
# 4. DBSCAN
# 5. K-Means
# 6. Mini-Batch K-Means
# 7. Mean Shift
# 8. OPTICS
# 9. Spectral Clustering
# 10. Gaussian Mixture Model
# 
# ### 1. Affinity Propagation
# Affinity Propagation involves finding a set of exemplars that best summarize the data.
# 
# ### 2. Agglomerative clustering
# Agglomerative clustering involves merging examples until the desired number of clusters is achieved.
# 
# ### 3. BIRCH Clustering 
# (BIRCH is short for Balanced Iterative Reducing and Clustering using Hierarchies) involves constructing a tree structure from which cluster centroids are extracted.
# 
# ### 4. DBSCAN Clustering 
# where DBSCAN is short for Density-Based Spatial Clustering of Applications with Noise involves finding high-density areas in the domain and expanding those areas of the feature space around them as clusters.
# 
# ### 5. K-Means
# K-Means Clustering may be the most widely known clustering algorithm and involves assigning examples to clusters in an effort to minimize the variance within each cluster.
# 
# ### 6. Mini-Batch K-Means
# Mini-Batch K-Means is a modified version of k-means that makes updates to the cluster centroids using mini-batches of samples rather than the entire dataset, which can make it faster for large datasets, and perhaps more robust to statistical noise.
# 
# ### 7. Mean Shift
# Mean shift clustering involves finding and adapting centroids based on the density of examples in the feature space.
# 
# ### 8. OPTICS
# OPTICS clustering (where OPTICS is short for Ordering Points To Identify the Clustering Structure) is a modified version of DBSCAN described above.
# 
# ### 9. Spectral Clustering
# Spectral Clustering is a general class of clustering methods, drawn from linear algebra.
# 
# ### 10. Gaussian Mixture Model
# A Gaussian mixture model summarizes a multivariate probability density function with a mixture of Gaussian probability distributions as its name suggests.
# 
# ![](https://scikit-learn.org/stable/_images/sphx_glr_plot_cluster_comparison_0011.png)
# 
# Ref: https://machinelearningmastery.com/clustering-algorithms-with-python/
# Ref: https://scikit-learn.org/stable/modules/clustering.html
# 

# # <center> <font size=20 color='Blue'> Unsupervised learning </font> </center>
# Clustering the Countries by using Unsupervised Learning for HELP International
# 
# ### Objective: 
# To categorise the countries using socio-economic and health factors that determine the overall development of the country.
# 
# ### Problem Statement:
# HELP International have been able to raise around $ 10 million. Now the CEO of the NGO needs to decide how to use this money strategically and effectively. So, CEO has to make decision to choose the countries that are in the direst need of aid. Hence, your Job as a Data scientist is to categorise the countries using some socio-economic and health factors that determine the overall development of the country. Then you need to suggest the countries which the CEO needs to focus on the most.

# # Read Data

# In[ ]:


pd.set_option("display.max_colwidth",180)
df = pd.read_csv('../input/unsupervised-learning-on-country-data/Country-data.csv')
data_dict = pd.read_csv('../input/unsupervised-learning-on-country-data/data-dictionary.csv')


# # Dataset feature details

# In[ ]:


data_dict


# ### Dataframe details

# In[ ]:


df.info()


# ***Except country feature, other features are either float or integer. there is no text data in the dataframe***

# ### Statistical Analysis

# In[ ]:


df.shape


# ***There are 167 rows and 10 columns(features)***

# In[ ]:


df.describe().T


# ***Observations:***
# 1. child_mort, exports, imports, income, inflation, gdpp - seems to have large difference between 75% percentile and max value. it looks like these features are right scewed.

# In[ ]:


df['country'].value_counts()


# ***Observations***
# 1. country feature is identical value, cant be considered as categorical as there is no multiple entries. so, this particular feature might not be helpful for the modeling. but, we shall use for EDA.

# ### Check if there is null or na values

# In[ ]:


df.isnull().sum()


# In[ ]:


df.isna().sum()


# ***Fortunately there is no null value identified.***

# ## Exploratory Data Analysis

# In[ ]:


#Import ploting libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
colors = ['#DB1C18','#DBDB3B','#51A2DB']
sns.set(palette=colors, font='Serif', style='white', rc={'axes.facecolor':'whitesmoke', 'figure.facecolor':'whitesmoke'})


# ### Univariated data analysis

# In[ ]:


df['country'].count()


# ***Each row in the dataset belongs to each country data***

# In[ ]:


fig, ax = plt.subplots(nrows=3,ncols=3, figsize=(15,8), constrained_layout=True)
plt.suptitle("Univariated Data Analyis")
ax=ax.flatten()
int_cols= df.select_dtypes(exclude='object').columns
for x, i in enumerate(int_cols):
    sns.histplot(df[i], ax=ax[x], kde=True, color=colors[2])


# In[ ]:


fig, ax = plt.subplots(nrows=3,ncols=3, figsize=(15,8), constrained_layout=True)
plt.suptitle("Univariated Data Analyis")
ax=ax.flatten()
int_cols= df.select_dtypes(exclude='object').columns
for x, i in enumerate(int_cols):
    sns.boxplot(x=df[i], ax=ax[x], color=colors[2])


# ***Observations***
# 1. Both Histogram and the boxplot clearly shows that the numerical features are contineous or discreate values. there are no features with categorical values. 
# 2. Box plot shows us there are clear outliers in child_mort, exports, imports, income, gdpp features. however, these informations are belongs to each country. so, we can't expect the values to be normaly distributed wihtout outliers. 
# 3. Also, the problem statement clearly describes the we need to cluster the countries that need help. so, there are clustering algorithms like ***Manhaten distance*** are less sensible to outliers.

# ### Bivariated Data Analysis

# In[ ]:


px.scatter(data_frame=df, x='exports', y='imports',size='gdpp', text='country', color='gdpp', title='Countries by Export & Import and corresponding GDP')


# In[ ]:


for i in int_cols:
    fig=px.choropleth(data_frame=df, locationmode='country names', locations='country', color=i, title=f'{i} rate by countries')
    fig.show()


# ***From the above Graphs we can clearly see that there are 2 clusters. Aftican and south Asian countires and rest of the world countries. however, further exploration would help us to learn better***

# In[ ]:


sns.pairplot(df, corner =True)


# ***Observations***:
# 1. Child_mort has negative relationship with GDP as the child mortality is less the GDP also increases and vice versa.
# 2. Export, Income, Income has clear postivite relationship with GDP. 
# 3. Total_fer and child_mort has postive relationship. 
# 4. total_fer and life_expec has negative relationship. 
# 5. life_expec and childe_mort has negative relationship.

# In[ ]:


fig=plt.figure(figsize=(15,8))
sns.heatmap(df.corr(), annot=True, square=True)


# ## Data Modeling

# In[ ]:


from sklearn.preprocessing import StandardScaler
df_scaled = StandardScaler().fit_transform(df.drop(['country'], axis=1))


# ### PCA - Principal component analysis
# PCA is used in exploratory data analysis and for making predictive models. It is commonly used for dimensionality reduction by projecting each data point onto only the first few principal components to obtain lower-dimensional data while preserving as much of the data's variation as possible. The first principal component can equivalently be defined as a direction that maximizes the variance of the projected data. 

# In[ ]:


from sklearn.decomposition import PCA
decom = PCA(svd_solver='auto')
decom.fit(df_scaled)


# In[ ]:


cum_exp_ratio = np.cumsum(np.round(decom.explained_variance_ratio_,2))
print(cum_exp_ratio)
fig=plt.figure(figsize=(10,8))
ax=sns.lineplot(y=cum_exp_ratio, x=np.arange(0,len(cum_exp_ratio)))
ax=sns.scatterplot(y=cum_exp_ratio, x=np.arange(0,len(cum_exp_ratio)))
ax.set_xlabel('No of components')
ax.set_ylabel('explaned variance ratio')


# ***PCA with number of clusters 3 and 4 as deviation in the variance ratio. even 5 also can be considerd as the difference is less. so, lets us try to use the 3,4,5 cluster combination in K_Mean clustering***

# ### Hierarachial Clustering

# In[ ]:


import scipy.cluster.hierarchy as sch
fig=plt.figure(figsize=(15,8))
dendrogram = sch.dendrogram(sch.linkage(df_scaled, method = 'ward'))
plt.suptitle('Hierarchial clustering - Dendrogram')
plt.xlabel('Countries')
plt.ylabel('Euclidean Distances')
plt.show()


# ***We can clearly see that there 3 cluster***

# ### K_Mean Clustering

# Kmeans Algorithm is an Iterative algorithm that divides a group of n datasets into k subgroups /clusters based on the similarity and their mean distance from the centroid of that particular subgroup/ formed. KMean is mostly commonly used clustering algorithm

# In[ ]:


from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
visualize = KElbowVisualizer(model, k=(1,10))
visualize.fit(df_scaled)
visualize.poof()


# ***Elbow method is common method used to validate the clustering algorithm. here we can see the K value 3 with relatively good distortion score.***

# In[ ]:


model = KMeans(n_clusters=3, random_state=1)
model.fit(df_scaled)
df['KMean_labels']=model.labels_
fig,ax=plt.subplots(nrows=1,ncols=3,figsize=(18,8))
sns.scatterplot(data=df, x='exports', y='income', hue='KMean_labels', ax=ax[0])
sns.scatterplot(data=df, x='exports', y='gdpp', hue='KMean_labels', ax=ax[1])
sns.scatterplot(data=df, x='child_mort', y='health', hue='KMean_labels', ax=ax[2])


# In[ ]:


df.groupby(['KMean_labels','country']).mean()


# In[ ]:


from sklearn.metrics import silhouette_score
silhouette_score(df_scaled,labels=model.labels_)


# In[ ]:


#df['KMean_labels']=df['KMean_labels'].astype('category')
cat = {0:'Need Help',1:'Might need help',2:'No Help needed'}
df['KMean_labels']=df['KMean_labels'].map(cat)

px.choropleth(data_frame=df, locationmode='country names', locations='country', color=df['KMean_labels'], title='Countries by category that need help',
              color_discrete_map={'Need Help':'#DB1C18','Might need help':'#DBDB3B','No Help needed':'#51A2DB'} ,projection='equirectangular')


# In[ ]:


px.choropleth(data_frame=df, locationmode='country names', locations='country', color=df['KMean_labels'], title='African Countries by category that need help',
              color_discrete_map={'Need Help':'#DB1C18','Might need help':'#DBDB3B','No Help needed':'#51A2DB'} ,projection='equirectangular', scope='africa')


# In[ ]:


px.choropleth(data_frame=df, locationmode='country names', locations='country', color=df['KMean_labels'], title='Asian Countries by category that need help',
              color_discrete_map={'Need Help':'#DB1C18','Might need help':'#DBDB3B','No Help needed':'#51A2DB'} ,projection='equirectangular', scope='asia')


# ***Observations:***
# I have clustred the countries in 3 categories. 
# 1. Need Help
# 2. Might need help
# 3. No Help needed
# 
# ***Conclusion:***
# 1. Most African countries and Pakistan, Afganistan, Iraq, Yemen, Lao etc falls in the category of "Help Needed" based on the GDP, Income, Health rate etc
# 2. Most Asian countires fall in 2nd category
# 3. American, Australian countires, Canada & Europian may not need help. 

# In[ ]:


df[df['KMean_labels']=='Need Help']['country']


# ### Please revewi the Kernel and provide your input for further improvements. Appriciate your feedback and comments
