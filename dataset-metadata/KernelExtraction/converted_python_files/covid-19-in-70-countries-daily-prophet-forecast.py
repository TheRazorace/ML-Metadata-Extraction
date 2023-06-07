#!/usr/bin/env python
# coding: utf-8

# # Dataset with holidays data [COVID-19: Holidays of countries](https://www.kaggle.com/vbmokin/covid19-holidays-of-countries) (holidays for 2020-2022)

# # COVID-19: EDA & Forecasting with holidays impact for confirmed cases. Prophet with prior_scale optimization. Forecasting for the next week.

# * Version 32 - only from 01.12.2020
# * Versions 33-63 - all data from 2020
# * Version 64 - from 01.09.2021

# # Acknowledgements
# 
# - dataset [COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University](https://github.com/CSSEGISandData/COVID-19)
# - my dataset with holidays data [COVID-19: Holidays of countries](https://www.kaggle.com/vbmokin/covid19-holidays-of-countries) - it is recommended to follow the updates
# - and others datasets
# - notebook with the code to read the data [COVID-19: current situation on August](https://www.kaggle.com/corochann/covid-19-current-situation-on-august)
# - notebook [COVID-19 Novel Coronavirus EDA & Forecasting Cases](https://www.kaggle.com/khoongweihao/covid-19-novel-coronavirus-eda-forecasting-cases) from [@Wei Hao Khoong](https://www.kaggle.com/khoongweihao)
# - notebook with code for parameters tuning [COVID-19-in-Ukraine: Prophet & holidays tuning](https://www.kaggle.com/vbmokin/covid-19-in-ukraine-prophet-holidays-tuning)
# - https://facebook.github.io/prophet/
# - https://facebook.github.io/prophet/docs/
# - https://github.com/facebook/prophet
# - https://github.com/dr-prodigy/python-holidays

# There are many studies in the field of coronavirus forecasting. Many researchers use **Prophet** (from Facebook). But for some reason, no one takes into account the holidays impact. After all, despite all the prohibitions, it is difficult for people to stay at home and they still somehow celebrate the **holidays** to which they are accustomed. The desire to celebrate is especially strong when people are sitting at home all the time looking for something to do. In our opinion, the impact of the holidays is manifested in the fact that within 4-10 days after these holidays there may be a jump in the number of confirmed cases, due to the fact that people went shopping, and even visiting each other, perhaps even in violation of quarantine requirements. 
# 
# The Prophet uses the library [holidays](https://github.com/dr-prodigy/python-holidays) with information about the main holidays of 67 countries, but and its package has some disadvantages. That's why I created a more perfect own dataset and plan to update it periodically. My graduate students ( ) help me fill it. Now my dataset [COVID-19: Holidays of countries](https://www.kaggle.com/vbmokin/covid19-holidays-of-countries) has holidays for 70 countries and more adapted for use in the prediction of coronavirus diseases. 
# 
# We will limit myself to forecasting only those countries for which there is information about holidays for Prophet.
# 
# A multiplicative Prophet model is built taking into account the weekly and triply (3 days) seasonality, first without taking into account the holidays, and then taking into account the holidays. Then the model is tuned according to the parameter "prior_scale" that gets the value from the list of the user and is searching the one that provides the best forecast of the 14 last values of the data. The optimal model is determined for each country (with or without holidays, the best value of parameter "prior_scale" ). 
# 
# The model with optimal parameters is used to predict future data for the next 3 days. The data is taken from [COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University](https://github.com/CSSEGISandData/COVID-19) (usually this dataset are updated there daily and are available as of yesterday), so the next 3 days are counted from the date of the last committee of this notebook. 
# 
# For the best models, plots with forecast values and with all components of the model are built.
# 
# There is a version of this notebook with model validation, there are versions for some countries, where the model is tuned simultaneously for all three parameters of the holidays (lower_window, upper_window, prior_scale) - see among [notebooks of the dataset **"COVID-19: Holidays of countries"**](https://www.kaggle.com/vbmokin/covid19-holidays-of-countries/notebooks)
# 
# Added holidays in 70 countries in **2021** to those that were in 2020.

# The forecast for all countries with the tuning of only one parameter (prior_scale) gives low accuracy unfortunately. The purpose of this notebook is to show how to use a holiday [dataset](https://www.kaggle.com/vbmokin/covid19-holidays-of-countries) for 70 countries.
# 
# Higher forecasting accuracy (up to 2-6% for 7 days) is achieved by tuning 11 parameters for each country individually - see their list in the [dataset](https://www.kaggle.com/vbmokin/covid19-holidays-of-countries). For example, see notebooks for Ukraine.

# <a class="anchor" id="0.1"></a>
# ## Table of Contents
# 
# 1. [Import libraries](#1)
# 1. [Download data](#2)
# 1. [Selection of countries with data on holidays](#3)
# 1. [EDA](#4)
# 1. [Model training, forecasting and evaluation](#5)
# 

# ## 1. Import libraries<a class="anchor" id="1"></a>
# 
# [Back to Table of Contents](#0.1)

# Import libraries

# In[ ]:


import os
import pandas as pd
import numpy as np
import requests
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objects as go

from datetime import date, timedelta, datetime
from fbprophet import Prophet
from fbprophet.make_holidays import make_holidays_df
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric
import holidays

import pycountry
import plotly.express as px
from collections import namedtuple

import warnings
warnings.simplefilter('ignore')


# ## 2. Download data<a class="anchor" id="2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks https://github.com/CSSEGISandData/COVID-19
myfile = requests.get('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
open('data', 'wb').write(myfile.content)
confirmed_global_df = pd.read_csv('data')
confirmed_global_df


# In[ ]:


# Thanks to https://www.kaggle.com/corochann/covid-19-current-situation-on-august
def _convert_date_str(df):
    try:
        df.columns = list(df.columns[:4]) + [datetime.strptime(d, "%m/%d/%y").date().strftime("%Y-%m-%d") for d in df.columns[4:]]
    except:
        print('_convert_date_str failed with %y, try %Y')
        df.columns = list(df.columns[:4]) + [datetime.strptime(d, "%m/%d/%Y").date().strftime("%Y-%m-%d") for d in df.columns[4:]]

_convert_date_str(confirmed_global_df)
confirmed_global_df


# In[ ]:


# Thanks to https://www.kaggle.com/corochann/covid-19-current-situation-on-august
df = confirmed_global_df.melt(
    id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], value_vars=confirmed_global_df.columns[4:], var_name='Date', value_name='ConfirmedCases')


# In[ ]:


df


# In[ ]:


df["Country/Region"].unique()


# It would be more correct to call this list "Country/Regions"

# In[ ]:


# Convert name of countries to ISO 3166
df["Country/Region"].replace({'Korea, South': 'Korea, Republic of'}, inplace=True)
df["Country/Region"].replace({'Russia': 'Russian Federation'}, inplace=True)
df["Country/Region"].replace({'US': 'United States'}, inplace=True)


# In[ ]:


df['Date'] = pd.to_datetime(df['Date'], format = "%Y-%m-%d")
df


# In[ ]:


# Version 32 - selection dates from 2020-12-01
# Version 33-63 - without this filter
# Version 64 - selection dates from 2021-09-01
df = df[df['Date'] >= datetime.strptime('2021-09-01', "%Y-%m-%d")]
df


# In[ ]:


df2 = df.groupby(["Date", "Country/Region"])[['Date', 'Country/Region', 'ConfirmedCases']].sum().reset_index()


# In[ ]:


df_countries = df2['Country/Region'].unique()
df_countries


# In[ ]:


latest_date = df2['Date'].max()
latest_date


# ## 3. Selection of countries with data on holidays<a class="anchor" id="3"></a>
# 
# [Back to Table of Contents](#0.1)

# ### Thank to dataset [COVID-19: Holidays of countries](https://www.kaggle.com/vbmokin/covid19-holidays-of-countries)

# In[ ]:


# Thanks to dataset https://www.kaggle.com/vbmokin/covid19-holidays-of-countries
holidays_df = pd.read_csv('../input/covid19-holidays-of-countries/holidays_df_of_70_countries_for_covid_19_2022_Temporary.csv')
holidays_df


# In[ ]:


holidays_df['country'].unique()


# In[ ]:


holidays_df_code_countries = holidays_df['code'].unique()
holidays_df_code_countries


# In[ ]:


def dict_code_countries_with_holidays(list_name_countries: list,
                                      holidays_df: pd.DataFrame()):
    """
    Defines a dictionary with the names of user countries and their two-letter codes (ISO 3166) 
    in the dataset "COVID-19: Holidays of countries" 
    
    Returns: 
    - countries: dictionary with the names of user countries and their two-letter codes (ISO 3166) 
    - holidays_df_identificated: DataFrame with holidays data for countries from dictionary 'countries'
    
    Args: 
    - list_name_countries: list of the name of countries (name or common_name or official_name or alha2 or alpha3 codes from ISO 3166)
    - holidays_df: DataFrame with holidays "COVID-19: Holidays of countries"
    """
    
    import pycountry
    
    # Identification of countries for which there are names according to ISO
    countries = {}
    dataset_all_countries = list(holidays_df['code'].unique())
    list_name_countries_identificated = []
    list_name_countries_not_identificated = []
    for country in list_name_countries:
        try: 
            country_id = pycountry.countries.get(alpha_2=country)
            if country_id.alpha_2 in dataset_all_countries:
                countries[country] = country_id.alpha_2
        except AttributeError:
            try: 
                country_id = pycountry.countries.get(name=country)
                if country_id.alpha_2 in dataset_all_countries:
                    countries[country] = country_id.alpha_2
            except AttributeError:
                try: 
                    country_id = pycountry.countries.get(official_name=country)
                    if country_id.alpha_2 in dataset_all_countries:
                        countries[country] = country_id.alpha_2
                except AttributeError:
                    try: 
                        country_id = pycountry.countries.get(common_name=country)
                        if country_id.alpha_2 in dataset_all_countries:
                            countries[country] = country_id.alpha_2
                    except AttributeError:
                        try: 
                            country_id = pycountry.countries.get(alpha_3=country)
                            if country_id.alpha_2 in dataset_all_countries:
                                countries[country] = country_id.alpha_2
                        except AttributeError:
                            list_name_countries_not_identificated.append(country)
    holidays_df_identificated = holidays_df[holidays_df['code'].isin(countries.values())]
    
    print(f'Thus, the dataset has holidays in {len(countries)} countries from your list with {len(list_name_countries)} countries')
    if len(countries) == len(dataset_all_countries):
        print('All available in this dataset holiday data is used')
    else:
        print("Holidays are available in the dataset for such countries (if there are countries from your list, then it's recommended making changes to the list)")
        print(np.array(holidays_df[~holidays_df['code'].isin(countries.values())].country_official_name.unique()))
        
    return countries, holidays_df_identificated


# In[ ]:


countries_dict, holidays_df = dict_code_countries_with_holidays(df_countries,holidays_df)


# In[ ]:


def adaption_df_to_holidays_df_for_prophet(df, col, countries_dict):
    # Adaptation the dataframe df (by column=col) to holidays_df by list of countries in dictionary countries_dict
    
    # Filter df for countries which there are in the dataset with holidays
    df = df[df[col].isin(list(countries_dict.keys()))].reset_index(drop=True)
    
    # Add alpha_2 (code from ISO 3166) for each country
    df['iso_alpha'] = None
    for key, value in countries_dict.items():
        df.loc[df[col] == key, 'iso_alpha'] = value    
    
    return df


# In[ ]:


holidays_df


# In[ ]:


df2 = adaption_df_to_holidays_df_for_prophet(df2, 'Country/Region', countries_dict)
df2.columns = ['Date', 'Country', 'Confirmed', 'iso_alpha']
df2


# In[ ]:


print("Number of countries/regions with data: " + str(len(df2.Country.unique())))


# ## 4. EDA<a class="anchor" id="4"></a>
# 
# [Back to Table of Contents](#0.1)

# ## Earliest Cases

# In[ ]:


df2.describe()


# In[ ]:


df2.head()


# ## Latest Cases

# In[ ]:


df2.tail()


# ## 5. Model training, forecasting and evaluation<a class="anchor" id="5"></a>
# 
# [Back to Table of Contents](#0.1)

# Forecasting Confirmed Cases Worldwide with Prophet by Country

# In[ ]:


lower_window_list = [0, -1, -2, -3] # must be exactly 4 values (identical allowed)
upper_window_list = [0, 1, 2, 3] # must be exactly 4 values (identical allowed)
prior_scale_list = [0.05, 0.5, 1, 15] # must be exactly 4 values (identical allowed)


# In[ ]:


def convert10_base4(n):
    # convert decimal to base 4
    alphabet = "0123"
    if n < 4:
        return alphabet[n]
    else:
        return (convert10_base4(n // 4) + alphabet[n % 4]).format('4f')


# In[ ]:


days_to_forecast = 7 # in future (after training data)
days_to_forecast_for_evalution = 7 # on the latest training data - for model training
first_forecasted_date = sorted(list(set(df2['Date'].values)))[-days_to_forecast]

print('The first date to perform forecasts for evaluation is: ' + str(first_forecasted_date))


# In[ ]:


print('The end date to perform forecasts in future for is: ' + (df2['Date'].max() + timedelta(days = days_to_forecast)).strftime("%Y-%m-%d"))


# In[ ]:


confirmed_df = df2[['Date', 'Country', 'Confirmed', 'iso_alpha']]
confirmed_df


# In[ ]:


all_countries = confirmed_df['Country'].unique()
all_countries


# In[ ]:


n = 64 # number of combination of parameters lower_window / upper_window / prior_scale


# In[ ]:


def make_forecasts(all_countries, confirmed_df, holidays_df, days_to_forecast, days_to_forecast_for_evalution, first_forecasted_date):
    # Thanks to https://www.kaggle.com/vbmokin/covid-19-in-ukraine-prophet-holidays-tuning
    
    def eval_error(forecast_df, country_df_val, first_forecasted_date, title):
        # Evaluate forecasts with validation set val_df and calculaction and printing with title the relative error
        forecast_df[forecast_df['yhat'] < 0]['yhat'] = 0
        result_df = forecast_df[(forecast_df['ds'] >= pd.to_datetime(first_forecasted_date))]
        result_val_df = result_df.merge(country_df_val, on=['ds'])
        result_val_df['rel_diff'] = (result_val_df['y'] - result_val_df['yhat']).round().abs()
        relative_error = [sum(result_val_df['rel_diff'].values)*100/result_val_df['y'].sum()]
        return relative_error
    
    def model_training_forecasting(df, forecast_days, holidays_df=None):
        # Prophet model training and forecasting
        
        model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False, 
                        holidays=holidays_df, changepoint_range=1, changepoint_prior_scale = 0.25)
        model.add_seasonality(name='weekly', period=7, fourier_order=8, mode = 'multiplicative', prior_scale = 0.3)
        #model.add_seasonality(name='triply', period=3, fourier_order=2, mode = 'multiplicative', prior_scale = 0.5)
        model.fit(df)
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        forecast[forecast['yhat'] < 0]['yhat'] = 0
        forecast['yhat_lower'] = forecast['yhat_lower'].round().astype('int')
        forecast['yhat'] = forecast['yhat'].round().astype('int')
        forecast['yhat_upper'] = forecast['yhat_upper'].round().astype('int')
    
        return model, forecast
    
    forecast_dfs = []
    relative_errors = []
    cols_w = ['ds', 'trend', 'yhat', 'yhat_lower', 'yhat_upper', 'trend_lower', 'trend_upper', 'additive_terms', 'additive_terms_lower', 'additive_terms_upper',
              'multiplicative_terms','multiplicative_terms_lower', 'multiplicative_terms_upper', 'weekly', 'weekly_lower', 'weekly_upper']
    cols_h = ['ds', 'trend', 'yhat', 'yhat_lower', 'yhat_upper', 'trend_lower', 'trend_upper', 'additive_terms', 'additive_terms_lower', 'additive_terms_upper',
              'holidays', 'holidays_lower', 'holidays_upper', 'multiplicative_terms','multiplicative_terms_lower', 'multiplicative_terms_upper', 'weekly',
              'weekly_lower', 'weekly_upper']
    relative_errors_holidays = []
    #forecast_holidays_dfs = []
    #forecast_future_dfs = []
    counter = 0
    results = pd.DataFrame(columns=['Country', 'Country_code', 'Conf_real', 'Conf_pred', 'Conf_pred_h', 'n_h', 'err', 'err_h', 'lower_window', 'upper_window', 'prior_scale', 'how_less, %'])    
    w = 0
    for j in range(len(all_countries)):
        country = all_countries[j]
        if country in confirmed_df['Country'].values:
            print(f'Country {str(country)} is listed')
            country_df = confirmed_df[confirmed_df['Country'] == country].reset_index(drop=True)
            country_iso_alpha = country_df['iso_alpha'][0]
            
            # Calc daily values and ignoring zero daily total cases
            country_df['Confirmed'] = country_df['Confirmed'].diff()
            country_df.loc[0,'Confirmed'] = 0
            country_df = country_df[country_df['Confirmed'] > 0].reset_index(drop=True)
            
            # Selection holidays of country
            country_holidays_df = holidays_df[holidays_df['code'] == country_iso_alpha][['ds', 'holiday', 'lower_window', 'upper_window', 'prior_scale']].reset_index(drop=True)
            country_dfs = []            
            
            # Data preparation for forecast with Prophet
            country_df = country_df[['Date', 'Confirmed']]
            country_df.columns = ['ds','y']            

            # Set training and validation datasets
            country_df_future = country_df.copy()
            country_df_val = country_df[(country_df['ds'] >= pd.to_datetime(first_forecasted_date))].copy()
            if country_df_val['y'].sum() > 0:
                # There is data of the last week

                country_df = country_df[(country_df['ds'] < pd.to_datetime(first_forecasted_date))]            

                # Without holidays
                # Model training and forecasting without holidays
                model, forecast = model_training_forecasting(country_df, days_to_forecast_for_evalution)
                #fig1 = model.plot_components(forecast)

                # Evaluate forecasts with validation set val_df and calculaction and printing the relative error
                forecast_df = forecast[['ds', 'yhat']].copy()
                relative_errors += eval_error(forecast_df, country_df_val, first_forecasted_date, 'without holidays')

                # With holidays
                # Model training with tuning prior_scale and forecasting
                relative_error_holidays_min = relative_errors[-1]
                number_holidays = len(country_holidays_df[(country_holidays_df['ds'] > '2020-01-21')])
                for i in range(n):
                    parameters_iter = convert10_base4(i).zfill(3)
                    lower_window_i = lower_window_list[int(parameters_iter[0])]
                    upper_window_i = upper_window_list[int(parameters_iter[1])]
                    prior_scale_i = prior_scale_list[int(parameters_iter[2])]
                    country_holidays_df['lower_window'] = lower_window_i
                    country_holidays_df['upper_window'] = upper_window_i
                    country_holidays_df['prior_scale'] = prior_scale_i
                    model_holidays, forecast_holidays = model_training_forecasting(country_df, days_to_forecast_for_evalution, country_holidays_df)

                    # Evaluate forecasts with validation set val_df and calculaction and printing the relative error
                    forecast_holidays_df = forecast_holidays[['ds', 'yhat']].copy()
                    relative_error_holidays = eval_error(forecast_holidays_df, country_df_val, first_forecasted_date, 'with holidays impact')

                    # Save results
                    if i == 0:
                        relative_error_holidays_min = relative_error_holidays
                        forecast_holidays_df_best = forecast_holidays[cols_h]
                        model_holidays_best = model_holidays
                        lower_window_best = lower_window_i
                        upper_window_best = upper_window_i
                        prior_scale_best = prior_scale_i
                    elif (relative_error_holidays[0] < relative_error_holidays_min[0]):
                        relative_error_holidays_min = relative_error_holidays
                        forecast_holidays_df_best = forecast_holidays[cols_h]
                        model_holidays_best = model_holidays
                        lower_window_best = lower_window_i
                        upper_window_best = upper_window_i
                        prior_scale_best = prior_scale_i
                    print('i =',i,' from',n,':  lower_window =', lower_window_i, 'upper_window =',upper_window_i, 'prior_scale =', prior_scale_i)
                    print('error_holidays =',relative_error_holidays[0], 'err_holidays_min (WAPE)',relative_error_holidays_min[0], '\n')

                # Results visualization
                print('The best errors of model with holidays is', relative_error_holidays_min[0], 'with lower_window =', str(lower_window_best),
                  ' upper_window =', str(upper_window_best), ' prior_scale =', str(prior_scale_best))
                print('The best errors WAPE of model with holidays is', relative_error_holidays_min[0], '\n')
                relative_errors_holidays += relative_error_holidays_min            

                # Save results to dataframe with all dates
                forecast_holidays_df_best['country'] = country
                forecast_holidays_df_best.rename(columns={'yhat':'confirmed'}, inplace=True)
                if w == 0:                
                    forecast_holidays_dfs = forecast_holidays_df_best.tail(days_to_forecast_for_evalution)
                else:
                    forecast_holidays_dfs = pd.concat([forecast_future_dfs, forecast_holidays_df_best.tail(days_to_forecast_for_evalution)])

                # Forecasting the future
                if relative_errors[-1] < relative_errors_holidays[-1]:
                    # The forecast without taking into account the holidays is the best
                    model_future_best, forecast_future_best = model_training_forecasting(country_df_future, days_to_forecast)
                    forecast_plot = model_future_best.plot(forecast_future_best, ylabel='Confirmed in '+ country + ' (forecasting without holidays)')
                    cols = cols_w
                else:
                    # The forecast taking into account the holidays is the best
                    country_holidays_df['prior_scale'] = prior_scale_best
                    model_future_best, forecast_future_best = model_training_forecasting(country_df_future, days_to_forecast, country_holidays_df)
                    forecast_plot = model_future_best.plot(forecast_future_best, ylabel='Confirmed in '+ country + ' (forecasting with holidays)')
                    cols = cols_h
                # Save forecasting results 
                forecast_future_df_best = forecast_future_best[cols]
                forecast_future_df_best['country'] = country
                forecast_future_df_best.rename(columns={'yhat':'confirmed'}, inplace=True)
                if w == 0:                
                    forecast_future_dfs = forecast_future_df_best.tail(days_to_forecast)
                else:
                    forecast_future_dfs = pd.concat([forecast_future_dfs, forecast_future_df_best.tail(days_to_forecast)])

                # Save results to dataframe with result for the last date
                results.loc[w,'Country'] = country
                results.loc[w,'Country_code'] = country_iso_alpha
                confirmed_real_last = country_df_val.tail(1)['y'].values[0].astype('int')
                results.loc[w,'Conf_real'] = confirmed_real_last if confirmed_real_last > 0 else 0
                confirmed_pred_last = round(forecast_df.tail(1)['yhat'].values[0]).astype('int')
                results.loc[w,'Conf_pred'] = confirmed_pred_last if confirmed_pred_last > 0 else 0
                confirmed_pred_holidays_last = round(forecast_holidays_df_best.tail(1)['confirmed'].values[0],0).astype('int')
                results.loc[w,'Conf_pred_h'] = confirmed_pred_holidays_last if confirmed_pred_holidays_last > 0 else 0
                results.loc[w,'n_h'] = number_holidays
                results.loc[w,'err'] = relative_errors[-1]
                results.loc[w,'err_h'] = relative_errors_holidays[-1]
                results.loc[w,'lower_window'] = lower_window_best
                results.loc[w,'upper_window'] = upper_window_best
                results.loc[w,'prior_scale'] = prior_scale_best
                results.loc[w,'how_less, %'] = round((relative_errors[-1]-relative_errors_holidays[-1])*100/relative_errors[-1],1)
                w += 1

                # Output forecasting data
                forecast_future_opt_future = forecast_future_df_best[['ds', 'yhat_lower', 'confirmed', 'yhat_upper']].copy().tail(days_to_forecast)
                forecast_future_opt_future.columns = ['ds', 'confirmed_lower', 'confirmed', 'confirmed_upper']
                display(forecast_future_opt_future)

                # Output plot
                model_future_best.plot_components(forecast_future_best)
            else:
                print(f'Data for country {str(country)} for the last week are missing!\n')
                continue

        else:
            print(f'Country {str(country)} is not listed!')
            continue
    
    return forecast_holidays_dfs, relative_errors_holidays, forecast_future_dfs, results


# In[ ]:


forecast_holidays_dfs, relative_errors_holidays, forecast_future_dfs, results = make_forecasts(all_countries, confirmed_df, holidays_df, 
                                                                                               days_to_forecast, days_to_forecast_for_evalution, first_forecasted_date)


# In[ ]:


forecast_future_dfs


# In[ ]:


forecast_holidays_dfs


# In[ ]:


pd.set_option('max_rows', 75)


# In[ ]:


print('Forecasting results')
display(results.sort_values(by=['err_h'], ascending=True))


# In[ ]:


df_h_impact = results[results['how_less, %'] > 1]
if len(df_h_impact) > 0:
    print('Countries with the impact of holidays')
    display(df_h_impact.sort_values(by=['how_less, %'], ascending=False))
    print('Number of these countries is', len(df_h_impact))


# In[ ]:


df_h_non_impact = results[results['how_less, %'] < -10]
if len(df_h_non_impact) > 0:
    print('Countries without the impact of holidays')
    display(df_h_non_impact.sort_values(by=['how_less, %'], ascending=False))
    print('Number of these countries is', len(df_h_non_impact))


# In[ ]:


df_neutral = results[(results['how_less, %'] <= 1) & (results['how_less, %'] >= -10)]
if len(df_neutral) > 0:
    print('Others countries')
    display(df_neutral.sort_values(by=['how_less, %'], ascending=False))
    print('Number of these countries is', len(df_neutral))


# In[ ]:


if len(forecast_holidays_dfs) > 0:
    forecast_holidays_dfs.to_csv('forecast_holidays_dfs.csv', index=False)

if len(forecast_future_dfs) > 0:
    forecast_future_dfs.to_csv('forecast_future_dfs.csv', index=False)

if len(results) > 0:
    results.to_csv('results.csv', index=False)
    results[['Country', 'Country_code', 'lower_window', 'upper_window', 'prior_scale']].to_csv('holidays_params.csv', index=False)


# I hope you find this notebook useful and enjoyable.

# Your comments and feedback are most welcome.

# [Go to Top](#0)
