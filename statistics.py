#%%
import mlflow
import pandas as pd
import plotly.express as px
import seaborn as sns
import functools as ft
import matplotlib.pyplot as plt

def min_max_scaling(series):
    return (series - series.min()) / (series.max() - series.min())

df = pd.DataFrame()

for i in range(48):

    experiment_id = f'{i}'

    runs = mlflow.search_runs(experiment_ids=experiment_id)
    
    df = df.append(runs)

df = df[df.status == 'FINISHED']

#%%

# Use starttime endtime to calculate the duration and combine it with dt size to make plots

df = df[['run_id', 'experiment_id', 'metrics.Partition density', 'params.Threshold', 'params.Method']]

link_clust = df.loc[df['params.Method'] == 'Link Clustering', ['metrics.Partition density', 'experiment_id']].reset_index(drop=True).rename(columns={'metrics.Partition density': 'Link Clustering'})
greedy_up = df.loc[df['params.Method'] == 'Greedy algorithm up', ['metrics.Partition density', 'experiment_id']].reset_index(drop=True).rename(columns={'metrics.Partition density': 'Greedy algorithm up'})
greedy_bottom = df.loc[df['params.Method'] == 'Greedy algorithm bottom', ['metrics.Partition density', 'experiment_id']].reset_index(drop=True).rename(columns={'metrics.Partition density': 'Greedy algorithm bottom'})
tune = df.loc[df['params.Method'] == 'Tuning dendrogram cut', ['metrics.Partition density', 'experiment_id']].reset_index(drop=True).rename(columns={'metrics.Partition density': 'Tuning dendrogram cut'})

dfs = [link_clust, greedy_up, greedy_bottom, tune]
df_final = ft.reduce(lambda left, right: pd.merge(left, right, on='experiment_id'), dfs)
#%%

# Plot the distribution of the normalized partition density
df_tmp = df_final[['Greedy algorithm bottom', 'Greedy algorithm up', 'Tuning dendrogram cut', 'Link Clustering']]
df_tmp = df_tmp.melt()
df_tmp['value'] = min_max_scaling(df_tmp['value'])

sns.kdeplot(df_tmp.loc[(df_tmp['variable']=='Greedy algorithm bottom'),
            'value'], color='r', shade=True, label='Greedy algorithm bottom')
  
sns.kdeplot(df_tmp.loc[(df_tmp['variable']=='Greedy algorithm up'),
            'value'], color='b', shade=True, label='Greedy algorithm up')

sns.kdeplot(df_tmp.loc[(df_tmp['variable']=='Tuning dendrogram cut'),
            'value'], color='g', shade=True, label='Tuning dendrogram cut')

sns.kdeplot(df_tmp.loc[(df_tmp['variable']=='Link Clustering'),
            'value'], color='y', shade=True, label='Link Clustering')
  
plt.xlabel('Partition density')
plt.ylabel('Probability Density')
#%%

df_final['Greedy bottom - LC'] = df_final['Greedy algorithm bottom'] - df_final['Link Clustering']
df_final['Greedy up - LC'] = df_final['Greedy algorithm up'] - df_final['Link Clustering']
df_final['Tuning - LC'] = df_final['Tuning dendrogram cut'] - df_final['Link Clustering'] 
df_final = df_final[['Greedy bottom - LC', 'Greedy up - LC', 'Tuning - LC']]

# Boxplot
df_final.boxplot()

# Plot the distribution of the difference between each method and the baseline model
df_melt = df_final.melt()

sns.kdeplot(df_melt.loc[(df_melt['variable']=='Greedy bottom - LC'),
            'value'], color='r', shade=True, label='Greedy bottom - LC')
  
plt.xlabel('Improvement')
plt.ylabel('Probability Density')

sns.kdeplot(df_melt.loc[(df_melt['variable']=='Greedy up - LC'),
            'value'], color='b', shade=True, label='Greedy up - LC')
  
plt.xlabel('Improvement')
plt.ylabel('Probability Density')

sns.kdeplot(df_melt.loc[(df_melt['variable']=='Tuning - LC'),
            'value'], color='g', shade=True, label='Tuning - LC')
  
plt.xlabel('Improvement')
plt.ylabel('Probability Density')

#%%

dt = {'method': ['Greedy bottom', 'Greedy up', 'Tuning'], 
    'freq of improvement': [len(df_final[df_final['Greedy bottom - LC'] > 0])/df_final.shape[0],
                            len(df_final[df_final['Greedy up - LC'] > 0])/df_final.shape[0],
                            len(df_final[df_final['Tuning - LC'] > 0])/df_final.shape[0]],
    'freq of decrease': [len(df_final[df_final['Greedy bottom - LC'] < 0])/df_final.shape[0],
                            len(df_final[df_final['Greedy up - LC'] < 0])/df_final.shape[0],
                            len(df_final[df_final['Tuning - LC'] < 0])/df_final.shape[0]]}

freq = pd.DataFrame(dt)

ax = freq.plot(x="method", y="freq of improvement", kind="bar")

freq.plot(x="method", y="freq of decrease", kind="bar", ax=ax, color='red')

# %%
