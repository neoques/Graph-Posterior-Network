import yaml
import glob
import os 
import itertools
import re
from collections import OrderedDict

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

def generate_all_plotlys(df, folder, color_type='dist_embedding_beta', x = 'dist_reg', ys = ['test_accuracy', 'test_ood_detection_epistemic_auroc']):
    plotly_dir = "../plotly"
    curr_dir = f"../plotly/{folder}"
    os.makedirs(f"../plotly/{folder}", exist_ok=True)
    print(f'Making - {curr_dir}')
    uniques = OrderedDict(sorted(df.nunique().to_dict().items()))
    unique_types = OrderedDict(sorted(df.apply(lambda x: list(set(x))).to_dict().items()))
    
    assert uniques[x] > 1
    assert uniques[color_type] > 1
    
    for poppable in [x, color_type, 'val_accuracy', *ys]:
        uniques.pop(poppable)
        unique_types.pop(poppable)
    
    for key, val in list(uniques.items()):
        if val <= 1:
            uniques.pop(key)
            unique_types.pop(key)    
            
    keys = list(unique_types.keys())
    for a_set_of_uniques in itertools.product(*unique_types.values()):
        simp_df = df[np.logical_and.reduce(
                [df[keys[i]]==a_unique for i, a_unique in enumerate(a_set_of_uniques)]
            )]
        title = " ".join([f"{keys[i]}=={a_unique}" for i, a_unique in enumerate(a_set_of_uniques)])
        plotly_plot_df(simp_df, title, color_type=color_type, x=x, ys=ys, folder=folder)

def plotly_plot_df(df, title, color_type, x, ys, folder):
    data = []
    colors = ['blue', 'red', 'green', 'purple']
    df = df.sort_values([x, color_type])
    for color, a_color_type in zip(colors, df[color_type].unique()):
        df_tmp = df[df[color_type] == a_color_type]
        data.append(go.Scatter(x=df_tmp[x], y=df_tmp[ys[1]],
                            line=dict(color=color, width=4),
                            mode='lines+markers',
                            name=a_color_type + ' - ROC'))

        data.append(go.Scatter(x=df_tmp[x], y=df_tmp[ys[0]],
                            line=dict(color=color, width=4, dash='dash'),
                            mode='lines+markers',
                            name=a_color_type + ' - Test Acc'))

    layout = go.Layout(
        xaxis=dict(title=x),
        yaxis=dict(title=r''),
        title=title
    )
    fig = go.Figure(data=data, layout=layout)
    fig.update_xaxes(type="log")
    fig.write_html(os.path.join("../plotly", re.sub(" |==","-",f'{folder}/{title}.html')))

def get_df(trial_name):
    recent_dirs = []
    result_list = sorted(glob.glob("../ray_results/*"))
    for a_dir in result_list:
        if trial_name in a_dir:
            recent_dirs.append(a_dir)

    params = []
    results = []
    for a_recent_dir in recent_dirs:
        params.extend(sorted(glob.glob(a_recent_dir+"/*/params.json")))
        results.extend(sorted(glob.glob(a_recent_dir+"/*/result.json")))
    
    dicts = []
    for (a_param, a_result) in zip(params, results):
        with open(a_param) as f:
            a_param_dict = yaml.safe_load(f)
        param_dict = {a_key: a_param_dict['model'].get(a_key) for a_key in param_keys} | {a_key: a_param_dict['data'].get(a_key) for a_key in dataset_keys}
        with open(a_result) as f:
            a_result_dict = yaml.safe_load(f)

        if a_result_dict is not None:
            result_dict = {a_key: a_result_dict.get(a_key)  for a_key in result_keys}
            dicts.append(param_dict | result_dict)

    return pd.DataFrame.from_dict(dicts)

plotly_dir = '../plotly'

datasets = ['CoraML', ]#, 'CiteSeer', 'PubMed'] #, 'AmazonPhotos', 'AmazonComputers', 'CoauthorCS', 'CoauthorPhysics']
# datasets = ['CiteSeer']
param_keys = ['dist_reg', 'orig_dist_reg', 'KNN_K', 'dist_sigma', 'dist_embedding_beta', 'activation_type', 'entropy_reg', 'lipschitz_reg', 'lipschitz_init']
dataset_keys = ['dataset', ]
result_keys = ['val_accuracy', 'test_ood_detection_epistemic_auroc', 'test_accuracy']

## ActivationFunctionDirichletEntropy
df = get_df('ActivationFunctionDirichletEntropy')
df_mean = df.groupby(['dataset', 'activation_type']).mean().reset_index()
df_std = df.groupby(['dataset', 'activation_type']).std().reset_index()
df_cnt = df.groupby(['dataset', 'activation_type']).count().reset_index()
df_mean_table = df_mean[['dataset', 'activation_type', 'val_accuracy', 'test_ood_detection_epistemic_auroc']].pivot(index = 'activation_type', columns = 'dataset')
df_mean_table = df_mean_table.rename(columns={"test_ood_detection_epistemic_auroc": "ROC", "val_accuracy": "Val Acc"})
df_mean_table.columns = df_mean_table.columns.swaplevel(0, 1)
df_mean_table.sort_index(axis=1, level=0, inplace=True)
cols = df_mean_table.columns.to_list()
cols = [*cols[2:4], *cols[0:2], *cols[4:]]
cols = [cols[1], cols[0], cols[3], cols[2], cols[5], cols[4]]
df_mean_table[cols].sort_values(('CoraML', 'ROC'))
fig = px.box(df.sort_values('test_ood_detection_epistemic_auroc'), x='activation_type', y='test_ood_detection_epistemic_auroc', color='dataset') 
fig.write_html(os.path.join(plotly_dir, 'Activations-Box-CitationNetworks.html'.replace(" ", '-')))

os.makedirs(os.path.join(plotly_dir, "GDByActFun"), exist_ok=True)
## DenseActivationFunctionWithOrigDistance
df = get_df('ActivationFunctionDirichletEntropy')
df_mean = df.groupby(['dataset', 'activation_type']).mean().reset_index()
df_std = df.groupby(['dataset', 'activation_type']).std().reset_index()
df_cnt = df.groupby(['dataset', 'activation_type']).count().reset_index()
df_mean_table = df_mean[['dataset', 'activation_type', 'val_accuracy', 'test_ood_detection_epistemic_auroc']].pivot(index = 'activation_type', columns = 'dataset')
df_mean_table = df_mean_table.rename(columns={"test_ood_detection_epistemic_auroc": "ROC", "val_accuracy": "Val Acc"})
df_mean_table.columns = df_mean_table.columns.swaplevel(0, 1)
df_mean_table.sort_index(axis=1, level=0, inplace=True)
cols = df_mean_table.columns.to_list()
cols = [*cols[2:4], *cols[0:2], *cols[4:]]
cols = [cols[1], cols[0], cols[3], cols[2], cols[5], cols[4]]
df_mean_table[cols].sort_values(('CoraML', 'ROC'))
fig = px.box(df.sort_values('test_ood_detection_epistemic_auroc'), x='activation_type', y='test_ood_detection_epistemic_auroc', color='dataset') 
fig.write_html(os.path.join(plotly_dir, 'Activations-Box-CitationNetworks.html'.replace(" ", '-')))

