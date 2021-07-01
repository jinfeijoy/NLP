#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
sys.path.append('C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\python_functions\\jl_nlp_pkg')
sys.path.append('C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\python_functions\\jl_model_explain_pkg')
import nlpbasic.textClean as textClean
import nlpbasic.docVectors as DocVector
import nlpbasic.dataExploration as DataExploration
import nlpbasic.lda as lda
import nlpbasic.tfidf as tfidf

import model_explain.plot as meplot
import model_explain.shap as meshap

import data_visualization.distribution_plot as dbplot
import data_visualization.plotly_plot as ppt
import data_visualization.map_plot as mppt

from numpy import array,asarray,zeros
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Flatten,Embedding
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import re
import pycountry
from datetime import datetime, timedelta
import bar_chart_race as bcr

import plotly.express as px
root_path = 'C:\\Users\\luoyan011\\Desktop\\PersonalLearning\\GitHub\\NLP_data'


# Reference:
# * https://www.kaggle.com/gpreda/covid-19-vaccination-progress
# * https://www.kaggle.com/ivannatarov/covid-19-a-fall-of-darkness-eda-plotly
# * https://www.kaggle.com/andreshg/covid-progression-a-plotly-animated-guide
# * https://jackmckew.dev/creating-animated-plots-with-pandas_alive.html
# 
# Dataset:
# https://www.kaggle.com/josephassaker/covid19-global-dataset
# 
# Structure:
# * Global Situation: 
#     * cumulative case in N and %
#     * cumulative vacc in N and %
# * Specific Countries: US, Canada, China, India, Israel, others
#     * cumulative case in N and %
#     * cumulative vacc in N and %   
# * Vacc brand:
#     * vacc usage in countries
#     * diff vacc brand production volume per day
# * Map:
#     * current infection rate 
#     * death rate for covid
#     * current vacc rate
# 

# ## Load Data

# In[2]:


country_vacc = pd.read_csv(os.path.join(root_path, "country_vaccinations.csv"))
country_vacc_manu = pd.read_csv(os.path.join(root_path, "country_vaccinations_by_manufacturer.csv"))
worldometer_smr = pd.read_csv(os.path.join(root_path, "worldometer_coronavirus_summary_data.csv"))
worldometer_daily = pd.read_csv(os.path.join(root_path, "worldometer_coronavirus_daily_data.csv"))

worldometer_daily['date']=pd.to_datetime(worldometer_daily['date'], errors='coerce').dt.date
country_vacc['date']=pd.to_datetime(country_vacc['date'], errors='coerce').dt.strftime('%m-%d-%Y')
country_vacc_manu['date']=pd.to_datetime(country_vacc_manu['date'], errors='coerce')#.dt.date
worldometer_smr['infection_rate'] = worldometer_smr.total_confirmed/worldometer_smr.population
worldometer_smr['death_rate'] = worldometer_smr.total_deaths/worldometer_smr.total_confirmed


# In[17]:


country_vacc.dtypes


# In[3]:


worldometer_smr.head(3)


# In[6]:


worldometer_daily.head(3)


# In[9]:


country_vacc.head(3)


# In[10]:


country_vacc_manu.head(3)


# In[69]:


country_vacc_manu.location.unique()


# ## Cumulative case and vaccination

# In[3]:


vacc_global_count = country_vacc[['total_vaccinations','people_vaccinated','people_fully_vaccinated','date']].groupby('date').agg('sum')
vacc_global_count['vaccination_rate'] = vacc_global_count.people_vaccinated/(worldometer_smr.population.sum())
vacc_global_count['fully_vaccination_rate'] = vacc_global_count.people_fully_vaccinated/(worldometer_smr.population.sum())
vacc_global_count['date'] = vacc_global_count.index
vacc_global_count = vacc_global_count.reset_index(drop=True)
# vacc_global_count.head(3)
covid_global_count = worldometer_daily[['date','cumulative_total_cases', 'cumulative_total_deaths']].groupby('date').agg('sum')
covid_global_count['infection_rate'] = covid_global_count.cumulative_total_cases/(worldometer_smr.population.sum())
covid_global_count['death_rate'] = covid_global_count.cumulative_total_deaths/(covid_global_count.cumulative_total_cases)
covid_global_count['date'] = covid_global_count.index
covid_global_count = covid_global_count.reset_index(drop=True)
# covid_global_count.head(3)
global_data = covid_global_count.merge(vacc_global_count, how = 'left', on = 'date')

# global_data.head(3)


# In[5]:


colors = ['#DC143C', '#fa9b98', '#0d8c68', '#95c5b1', '#4169E1', '#adc3ff', '#595959', '#c4c4c4']
sb.palplot(sb.color_palette(colors))
plt.axis('off')
plt.box(on=None)


# In[6]:


ppt.viz_scatter(number_of_graphs=4, 
            df = global_data[['date','infection_rate', 'vaccination_rate','fully_vaccination_rate','death_rate']], 
            fill = 'tozeroy', 
            hovertemplate_name= ['infection_rate', 'vaccination_rate','fully_vaccination_rate','death_rate'], 
            color_list= [colors[0], colors[2],colors[4], colors[6]], 
            title_one= 'Death Rate and Infection Rate', 
            title_two= 'Daily Cumulative Rate', 
            y_title= 'Rate')


# In[5]:


from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
query = """
select A.date, A.country, 
    B.iso_code,
    A.cumulative_total_cases, A.cumulative_total_deaths, 
    B.total_vaccinations, B.people_vaccinated, B.people_fully_vaccinated,
    A.cumulative_total_cases / C.population as infection_rate,
    A.cumulative_total_deaths / A.cumulative_total_cases as death_rate,
    B.people_vaccinated / C.population as vaccination_rate,
    B.people_fully_vaccinated / C.population as fully_vaccination_rate
from worldometer_daily A
left join country_vacc B on A.date = B.date and A.country = B.country
left join worldometer_smr C on A.country = C.country
"""
country_dataset = pysqldf(query)


# In[8]:


country = 'Canada'
ppt.viz_scatter(number_of_graphs=4, 
            df = country_dataset[country_dataset.country==country][['date','death_rate' ,'vaccination_rate' ,'fully_vaccination_rate' ,'infection_rate']], 
            fill = 'tozeroy', 
            hovertemplate_name= ['infection_rate', 'vaccination_rate','fully_vaccination_rate','death_rate'], 
            color_list= [colors[0], colors[2],colors[4], colors[6]], 
            title_one= 'Vaccination Rate and Infection Rate', 
            title_two= 'Daily Cumulative Rate', 
            y_title= 'Rate')


# In[9]:


country = 'Canada'
ppt.viz_scatter(number_of_graphs=3, 
            df = country_dataset[country_dataset.country==country][['date','cumulative_total_cases' ,'people_vaccinated' ,'people_fully_vaccinated']], 
            fill = 'none', 
            hovertemplate_name= ['cumulative_total_cases' ,'people_vaccinated' ,'people_fully_vaccinated'], 
            color_list= [colors[0], colors[2],colors[4]], 
            title_one= 'Vaccination and Infection', 
            title_two= 'Daily Cumulative Count', 
            y_title= 'Count')


# ## Vaccine Production

# In[11]:


vacc_production_country = ['Chile', 'Czechia', 'France', 'Germany', 'Iceland', 'Italy', 
                           'Latvia', 'Lithuania', 'Romania', 'Switzerland', 'United States', 'Hungary']
dbplot.generate_bar_proportion(country_vacc_manu, 'vaccine', 'location', color = 0, order = True, topn = 13)


# In[6]:


country_id = 10
dbplot.generate_bar_proportion(country_vacc_manu[country_vacc_manu.location==vacc_production_country[country_id]], 
                               'vaccine', 
                               'date', 
                               color = 0, 
                               order = False, 
                               topn = 10,
                              subtitle = 'Country: '+vacc_production_country[country_id])


# In[12]:


country_id = 11
country_vacc_manuT = country_vacc_manu[country_vacc_manu.location == vacc_production_country[country_id]].pivot(index='date', columns='vaccine', values='total_vaccinations')
country_vacc_manuT.insert(0, 'date', country_vacc_manuT.index)
ppt.viz_scatter(number_of_graphs=(len(country_vacc_manuT.columns)-1), 
            df = country_vacc_manuT, 
            fill = 'tozeroy', 
            hovertemplate_name= country_vacc_manuT.columns, 
            color_list= [colors[i] for i in range(len(country_vacc_manuT.columns)-1)], 
            title_one= 'Total Vaccinations Daily', 
            title_two= 'Country: ' + vacc_production_country[country_id], 
            y_title= 'Count')


# In[5]:


vacc_counts_per_brand = country_vacc_manu.groupby('vaccine').agg({'total_vaccinations':'sum'})
vacc_counts_per_brand['vaccine'] = vacc_counts_per_brand.index
dbplot.bar_plot(x="vaccine", y="total_vaccinations", df=vacc_counts_per_brand, title = 'test', size=2, ordered=True, horizontally=True)


# In[20]:


len(vacc_counts_per_brand)


# In[20]:


vacc_counts_per_brand = country_vacc_manu.groupby(['vaccine', 'date']).agg({'total_vaccinations':'sum'})
vacc_counts_per_brand.reset_index(inplace = True)
# vacc_counts_per_brand=vacc_counts_per_brand.set_index(['date'])
vacc_counts_per_brand = vacc_counts_per_brand.pivot(index='date', columns='vaccine', values='total_vaccinations')
vacc_counts_per_brand = vacc_counts_per_brand.reset_index(drop=False)
vacc_counts_per_brand['date']=pd.to_datetime(vacc_counts_per_brand['date'], errors='coerce').dt.date
vacc_counts_per_brand.set_index("date", inplace = True)
vacc_counts_per_brand = vacc_counts_per_brand.rename_axis(None, axis=1)
vacc_counts_per_brand.head(3)


# ### Generate animated bar plot by using library panda_alive or bar_chart_race.

# In[56]:


bcr.bar_chart_race(df=vacc_counts_per_brand,
#                    filename='vaccine.mp4',
                   title= "Vaccine Production")


# In[29]:


import pandas_alive
def current_total(values):
    total = values.sum()
    s = f'Total Population : {int(total):,}'
    return {'x': .85, 'y': .2, 's': s, 'ha': 'right', 'size': 11}

# Generate bar chart race
vacc_counts_per_brand.fillna(0).plot_animated(filename='vaccine.gif',
                                     n_visible=10,
#                                      period_fmt="%Y",
                                     title='vaccine distribution',
                                     fixed_max=True,
                                     perpendicular_bar_func='mean',
                                     period_summary_func=current_total)


# ## Map

# In[6]:


mppt.static_scatter_map(worldometer_smr, 'infection_rate', 'country', 'continent', 'Infection Rate', 'Total number of infection divided by the population')


# In[8]:


mppt.static_choro_map(worldometer_smr, 'infection_rate', 'country', 'Infection Rate', 'Total number of infection divided by the population', colors = 'reds')


# In[10]:


mppt.static_choro_map(worldometer_smr, 'death_rate', 'country', 'Death Rate', 'Total number of death divided by the total infecaction', colors = 'reds')


# In[17]:


mppt.dynamic_choro_map(country_vacc, 'iso_code', 'country', 'total_vaccinations_per_hundred', 'date', 'Vaccination per hundred per country')


# In[16]:


mppt.dynamic_choro_map(country_vacc, 'iso_code', 'country', 'total_vaccinations', 'date', 'Number of Vaccination per country')

