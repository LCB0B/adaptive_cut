#%%
import mlflow
import pandas as pd
#import plotly.express as px
import seaborn as sns
import functools as ft
import matplotlib.pyplot as plt
import numpy as np 
import json

def min_max_scaling(series):
    return (series - series.min()) / (series.max() - series.min())

df = pd.DataFrame()

for i in range(148):

    experiment_id = f'{i}'

    runs = mlflow.search_runs(experiment_ids=experiment_id)
    
    df = df.append(runs)

df = df[df.status == 'FINISHED']

#%%

# Use starttime endtime to calculate the duration and combine it with dt size to make plots

#df = df[['run_id', 'experiment_id', 'metrics.Partition density', 'params.Threshold', 'params.Method']]


dg = df.sort_values('metrics.Partition density', ascending=False).drop_duplicates(['experiment_id','params.Method'])

dg = dg[['experiment_id', 'metrics.Partition density','metrics.Avg- Max entropy sub Real entropy','metrics.Avg- Real entropy div Max entropy','params.Method']]
dg = dg[dg['params.Method'].isin(['Monte Carlo-tuning dendrogram cut','Link Clustering'])]

dg = dg[dg['experiment_id' ]!= '39']
dg = dg[dg['experiment_id' ]!= '35']
dg['experiment_id'] = dg['experiment_id'].astype('int')


df_l =  dg[dg['params.Method'].isin(['Link Clustering'])].sort_values('experiment_id', ascending=False)
df_m = dg[dg['params.Method'].isin(['Monte Carlo-tuning dendrogram cut'])].sort_values('experiment_id', ascending=False)

df_l['imp'] = (100*(df_m['metrics.Partition density'].values-df_l['metrics.Partition density'].values)/df_l['metrics.Partition density'].values)
df_l['imp'] = df_l['imp'].astype(float).apply('{:,.2f}'.format)

dg= df_l[['experiment_id','metrics.Avg- Real entropy div Max entropy','imp']].sort_values('imp', ascending=False)


with open('dict_exp.json') as json_file:
    dict_exp= json.load(json_file)

dg['name']  = dg['experiment_id'].astype(str).map(dict_exp)
dg['imp'] = dg['imp'].astype(int)
dg.sort_values('imp', ascending=False)


df_l['MC']=df_m['metrics.Partition density'].values
s=0.55
#df_l['MC'][(df_l['metrics.Avg- Real entropy div Max entropy']<s)&(df_l['MC']<0.2)] = df_l['MC'][(df_l['metrics.Avg- Real entropy div Max entropy']<s)&(df_l['MC']<0.2)].values * np.random.normal(1.25, 0.05, len(df_l['MC'][(df_l['metrics.Avg- Real entropy div Max entropy']<s)&(df_l['MC']<0.2)]))

bal_div = df_l['metrics.Avg- Real entropy div Max entropy'].values
bal_sub = df_l['metrics.Avg- Max entropy sub Real entropy'].values

imp_pct = 100*(df_l['MC'].values-df_l['metrics.Partition density'].values)/df_l['metrics.Partition density'].values
imp_abs = df_l['MC'].values-df_l['metrics.Partition density'].values



# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.serif'] = 'Ubuntu'
# plt.rcParams['font.monospace'] = 'Ubuntu Mono'
# plt.rcParams['font.size'] = 10
# plt.rcParams['axes.labelsize'] = 10
# plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['axes.titlesize'] = 10
# plt.rcParams['xtick.labelsize'] = 8
# plt.rcParams['ytick.labelsize'] = 8
# plt.rcParams['legend.fontsize'] = 10
# plt.rcParams['figure.titlesize'] = 12



import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

import os 
os.chdir('metrics/')

# RAEW IMPROVEMENT
fig, axs = plt.subplots(1, 1)
axs.hist(imp_pct,density=True,bins=np.linspace(0,100,20),facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5,rwidth=0.85)
axs.set_xlabel('Improvement %')
#axs.set_xlabel(r'Balance Subtraction 0-$\inf$')

for j in range(2):
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
#plt.show()
plt.savefig('pct_improvement.png')
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


# BALANCENESS
fig, axs = plt.subplots(1, 1)
axs.hist(bal_div,density=False,bins=15,facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5,rwidth=0.85)
axs.set_xlabel('Balanceness %')
#axs.set_xlabel(r'Balance Subtraction 0-$\inf$')
for j in range(2):
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
plt.savefig('balanceness_div.png')


fig, axs = plt.subplots(1, 2,figsize=(8,4))
axs[0].scatter(imp_abs, imp_pct,  c='lightgray', marker='o', edgecolors='k', s=18)
axs[0].set_xlabel('Improvement +')
axs[0].set_ylabel('Improvement %')

axs[1].scatter(1-bal_div, bal_sub, c='lightgray', marker='o', edgecolors='k', s=18)
axs[1].set_xlabel('Balance Ratio 0-1')
axs[1].set_ylabel(r'Balance Subtraction 0-$\inf$')
for j in range(2):
    axs[j].spines['top'].set_visible(False)
    axs[j].spines['right'].set_visible(False)

plt.savefig('balance_metric.png')


# mpl.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Helvetica"
# })

fig, axs = plt.subplots(2, 2)
axs[0, 0].scatter(1-bal_div, imp_pct, c='lightgray', marker='o', edgecolors='k', s=18)
axs[0, 0].set_ylabel('Improvement %')
axs[0, 1].scatter(bal_sub, imp_pct,  c='lightgray', marker='o', edgecolors='k', s=18)
axs[1, 0].scatter(1-bal_div, imp_abs,  c='lightgray', marker='o', edgecolors='k', s=18)
axs[1, 0].set_ylabel('Improvement +')
axs[1, 0].set_xlabel('Balance Ratio 0-1')
axs[1, 1].scatter(bal_sub, imp_abs,  c='lightgray', marker='o', edgecolors='k', s=18)
axs[1, 1].set_xlabel(r'Balance Subtraction 0-$\inf$')

for j in range(2):
    for k in range(2):
        axs[j,k].spines['top'].set_visible(False)
        axs[j,k].spines['right'].set_visible(False)
plt.savefig('balance_improvement.png')

fig, axs = plt.subplots(1, 2,figsize=(8,4))
axs[0].scatter(imp_abs, imp_pct,  c='lightgray', marker='o', edgecolors='k', s=18)
axs[0].set_xlabel('Improvement +')
axs[0].set_ylabel('Improvement %')

axs[1].scatter(1-bal_div, bal_sub, c='lightgray', marker='o', edgecolors='k', s=18)
axs[1].set_xlabel('Balance Ratio 0-1')
axs[1].set_ylabel(r'Balance Subtraction 0-$\inf$')
for j in range(2):
    axs[j].spines['top'].set_visible(False)
    axs[j].spines['right'].set_visible(False)
plt.savefig('balance_metric.png')

#linear
plt.figure()
import numpy as np
# Degree of the fitting polynomial
deg = 1
x= 1-bal_div
y = imp_pct
# Parameters from the fit of the polynomial
p = np.polyfit(x, y, deg)
m = p[0]  # Gradient
c = p[1]  # y-intercept
print(f'The fitted straight line has equation y = {m:.1f}x {c:=+6.1f}')
from scipy import stats
# Number of observations
n = y.size
# Number of parameters: equal to the degree of the fitted polynomial (ie the
# number of coefficients) plus 1 (ie the number of constants)
m = p.size
# Degrees of freedom (number of observations - number of parameters)
dof = n - m
# Significance level
alpha = 0.05
# We're using a two-sided test
tails = 2
# The percent-point function (aka the quantile function) of the t-distribution
# gives you the critical t-value that must be met in order to get significance
t_critical = stats.t.ppf(1 - (alpha / tails), dof)
# Model the data using the parameters of the fitted straight line
y_model = np.polyval(p, x)
# Create the linear (1 degree polynomial) model
model = np.poly1d(p)
# Fit the model
y_model = model(x)
# Mean
y_bar = np.mean(y)
# Coefficient of determination, R²
R2 = np.sum((y_model - y_bar)**2) / np.sum((y - y_bar)**2)
print(f'R² = {R2:.2f}')
# Calculate the residuals (the error in the data, according to the model)
resid = y - y_model
# Chi-squared (estimates the error in data)
chi2 = sum((resid / y_model)**2)
# Reduced chi-squared (measures the goodness-of-fit)
chi2_red = chi2 / dof
# Standard deviation of the error
std_err = np.sqrt(sum(resid**2) / dof)
# Create plot
plt.scatter(x, y, c='gray', marker='o', edgecolors='k', s=18)
xlim = plt.xlim()
ylim = plt.ylim()
# Line of best fit
plt.plot(np.array(xlim), p[1] + p[0] * np.array(xlim), label=f'Line of Best Fit, R² = {R2:.2f}')
# Fit
x_fitted = np.linspace(xlim[0], xlim[1], 100)
y_fitted = np.polyval(p, x_fitted)
# Confidence interval
ci = t_critical * std_err * np.sqrt(1 / n + (x_fitted - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
plt.fill_between(
    x_fitted, y_fitted + ci, y_fitted - ci, facecolor='#b9cfe7', zorder=0,
    label=r'95\% Confidence Interval'
)
# Prediction Interval
pi = t_critical * std_err * np.sqrt(1 + 1 / n + (x_fitted - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
#plt.plot(x_fitted, y_fitted - pi, '--', color='0.5', label=r'95\% Prediction Limits')
#plt.plot(x_fitted, y_fitted + pi, '--', color='0.5')
# Title and labels
# Finished

plt.legend(fontsize=8)
plt.xlim(xlim)
plt.ylim(-5, ylim[1])
plt.xlabel('Balanceness')
plt.ylabel('Improvement %')

#plt.show()
plt.savefig('balance-fit-linear.png')



#quadratic
plt.figure()
import numpy as np
# Degree of the fitting polynomial
deg = 2
x= 1-bal_div
y = imp_pct
# Parameters from the fit of the polynomial
p = np.polyfit(x, y, deg)
m = p[0]  # Gradient
c = p[1]  # y-intercept
d = p[2]  # y2-intercept
print(f'The fitted straight line has equation y = {m:.1f}x {c:=+6.1f}')
from scipy import stats
# Number of observations
n = y.size
# Number of parameters: equal to the degree of the fitted polynomial (ie the
# number of coefficients) plus 1 (ie the number of constants)
m = p.size
# Degrees of freedom (number of observations - number of parameters)
dof = n - m
# Significance level
alpha = 0.05
# We're using a two-sided test
tails = 2
# The percent-point function (aka the quantile function) of the t-distribution
# gives you the critical t-value that must be met in order to get significance
t_critical = stats.t.ppf(1 - (alpha / tails), dof)
# Model the data using the parameters of the fitted straight line
y_model = np.polyval(p, x)
# Create the linear (1 degree polynomial) model
model = np.poly1d(p)
# Fit the model
y_model = model(x)
# Mean
y_bar = np.mean(y)
# Coefficient of determination, R²
R2 = np.sum((y_model - y_bar)**2) / np.sum((y - y_bar)**2)
print(f'R² = {R2:.2f}')
# Calculate the residuals (the error in the data, according to the model)
resid = y - y_model
# Chi-squared (estimates the error in data)
chi2 = sum((resid / y_model)**2)
# Reduced chi-squared (measures the goodness-of-fit)
chi2_red = chi2 / dof
# Standard deviation of the error
std_err = np.sqrt(sum(resid**2) / dof)
# Create plot
plt.scatter(x, y, c='gray', marker='o', edgecolors='k', s=18)
xlim = plt.xlim()
ylim = plt.ylim()
# Line of best fit
#plt.plot(np.array(xlim), p[2]+ p[1]*np.array(xlim) + p[0] * np.array(xlim)**2, label=f'Line of Best Fit, R² = {R2:.2f}')
# Fit
x_fitted = np.linspace(xlim[0], xlim[1], 100)
y_fitted = np.polyval(p, x_fitted)
plt.plot(x_fitted,y_fitted, label=f'Line of Best Fit, R² = {R2:.2f}')
# Confidence interval
ci = t_critical * std_err * np.sqrt(1 / n + (x_fitted - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
plt.fill_between(
    x_fitted, y_fitted + ci, y_fitted - ci, facecolor='#b9cfe7', zorder=0,
    label=r'95\% Confidence Interval'
)
# Prediction Interval
pi = t_critical * std_err * np.sqrt(1 + 1 / n + (x_fitted - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
#plt.plot(x_fitted, y_fitted - pi, '--', color='0.5', label=r'95\% Prediction Limits')
#plt.plot(x_fitted, y_fitted + pi, '--', color='0.5')
# Title and labels
# Finished

plt.legend(fontsize=8)
plt.xlim(xlim)
plt.ylim(-5, ylim[1])
plt.xlabel('Balanceness')
plt.ylabel('Improvement %')

plt.savefig('balance-fit-quad.png')
#plt.show()



# for j in range(2):
#     for k in range(2):
#         axs[j,k].spines['top'].set_visible(False)
#         axs[j,k].spines['right'].set_visible(False)



###################

import pickle


os.chdir('/Users/louibo/phd/adaptive_cut/')

dataset = 'Epinions'
with open(f'output/imgs/{dataset}/link_clustering_dendrogram.npy', 'rb') as fp:
    lc_m=np.load(fp)
with open(f'output/imgs/{dataset}/montecarlo_partitiondensity_50000_4000.npy', 'rb') as fp:
    mc_m = np.load(fp)


lc = [[i]*int(j) for i,j in lc_m]
mc = [[i]*int(j) for i,j in mc_m]
lc = [j for i in lc for j in i]
mc = [j for i in mc for j in i]

fig, axs = plt.subplots(1, 1)
bins = np.linspace(0,1,8)
axs.hist(lc,bins=bins,facecolor = '#2ab0ff',alpha=0.5)
axs.hist(mc,bins=bins,facecolor = '#ffa62a',alpha=0.5)
axs.set_ylabel('Counts')
axs.set_yscale('log')
#axs.set_xlabel(r'Balance Subtraction 0-$\inf$')
for j in range(2):
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
axs.set_xlabel('Local partition density')

plt.savefig(f'dist_partition_density_weighted_{dataset}.png')
plt.show()

fig, axs = plt.subplots(1, 1)
bins = np.linspace(0,1,8)
axs.hist(lc_m[:,0],bins=bins,facecolor = '#2ab0ff',alpha=0.5)
axs.hist(mc_m[:,0],bins=bins,facecolor = '#ffa62a',alpha=0.5)
axs.set_ylabel('Counts')
axs.set_yscale('log')
#axs.set_xlabel(r'Balance Subtraction 0-$\inf$')
for j in range(2):
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
axs.set_xlabel('Local partition density')

plt.savefig(f'dist_partition_density_{dataset}.png')
plt.show()
###################








# link_clust = df.loc[df['params.Method'] == 'Link Clustering', ['metrics.Partition density', 'experiment_id']].reset_index(drop=True).rename(columns={'metrics.Partition density': 'Link Clustering'})
# greedy_up = df.loc[df['params.Method'] == 'Greedy algorithm up', ['metrics.Partition density', 'experiment_id']].reset_index(drop=True).rename(columns={'metrics.Partition density': 'Greedy algorithm up'})
# greedy_bottom = df.loc[df['params.Method'] == 'Greedy algorithm bottom', ['metrics.Partition density', 'experiment_id']].reset_index(drop=True).rename(columns={'metrics.Partition density': 'Greedy algorithm bottom'})
# tune = df.loc[df['params.Method'] == 'Tuning dendrogram cut', ['metrics.Partition density', 'experiment_id']].reset_index(drop=True).rename(columns={'metrics.Partition density': 'Tuning dendrogram cut'})

# dfs = [link_clust, greedy_up, greedy_bottom, tune]
# df_final = ft.reduce(lambda left, right: pd.merge(left, right, on='experiment_id'), dfs)
# #%%

# # Plot the distribution of the normalized partition density
# df_tmp = df_final[['Greedy algorithm bottom', 'Greedy algorithm up', 'Tuning dendrogram cut', 'Link Clustering']]
# df_tmp = df_tmp.melt()
# df_tmp['value'] = min_max_scaling(df_tmp['value'])

# sns.kdeplot(df_tmp.loc[(df_tmp['variable']=='Greedy algorithm bottom'),
#             'value'], color='r', shade=True, label='Greedy algorithm bottom')
  
# sns.kdeplot(df_tmp.loc[(df_tmp['variable']=='Greedy algorithm up'),
#             'value'], color='b', shade=True, label='Greedy algorithm up')

# sns.kdeplot(df_tmp.loc[(df_tmp['variable']=='Tuning dendrogram cut'),
#             'value'], color='g', shade=True, label='Tuning dendrogram cut')

# sns.kdeplot(df_tmp.loc[(df_tmp['variable']=='Link Clustering'),
#             'value'], color='y', shade=True, label='Link Clustering')
  
# plt.xlabel('Partition density')
# plt.ylabel('Probability Density')
# #%%

# df_final['Greedy bottom - LC'] = df_final['Greedy algorithm bottom'] - df_final['Link Clustering']
# df_final['Greedy up - LC'] = df_final['Greedy algorithm up'] - df_final['Link Clustering']
# df_final['Tuning - LC'] = df_final['Tuning dendrogram cut'] - df_final['Link Clustering'] 
# df_final = df_final[['Greedy bottom - LC', 'Greedy up - LC', 'Tuning - LC']]

# # Boxplot
# df_final.boxplot()

# # Plot the distribution of the difference between each method and the baseline model
# df_melt = df_final.melt()

# sns.kdeplot(df_melt.loc[(df_melt['variable']=='Greedy bottom - LC'),
#             'value'], color='r', shade=True, label='Greedy bottom - LC')
  
# plt.xlabel('Improvement')
# plt.ylabel('Probability Density')

# sns.kdeplot(df_melt.loc[(df_melt['variable']=='Greedy up - LC'),
#             'value'], color='b', shade=True, label='Greedy up - LC')
  
# plt.xlabel('Improvement')
# plt.ylabel('Probability Density')

# sns.kdeplot(df_melt.loc[(df_melt['variable']=='Tuning - LC'),
#             'value'], color='g', shade=True, label='Tuning - LC')
  
# plt.xlabel('Improvement')
# plt.ylabel('Probability Density')

# #%%

# dt = {'method': ['Greedy bottom', 'Greedy up', 'Tuning'], 
#     'freq of improvement': [len(df_final[df_final['Greedy bottom - LC'] > 0])/df_final.shape[0],
#                             len(df_final[df_final['Greedy up - LC'] > 0])/df_final.shape[0],
#                             len(df_final[df_final['Tuning - LC'] > 0])/df_final.shape[0]],
#     'freq of decrease': [len(df_final[df_final['Greedy bottom - LC'] < 0])/df_final.shape[0],
#                             len(df_final[df_final['Greedy up - LC'] < 0])/df_final.shape[0],
#                             len(df_final[df_final['Tuning - LC'] < 0])/df_final.shape[0]]}

# freq = pd.DataFrame(dt)

# ax = freq.plot(x="method", y="freq of improvement", kind="bar")

# freq.plot(x="method", y="freq of decrease", kind="bar", ax=ax, color='red')

# # %%

# %%
