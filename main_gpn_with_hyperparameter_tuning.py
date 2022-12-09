# %%
from ray import tune
import ray
import yaml
import train_and_eval
import copy

import numpy as np

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="3,4,5,6"

ray.init(log_to_driver=False)

# Get original config
def config_helper(config):
    return train_and_eval.run_experiment(**config)


# config_orig['model']['dim_hidden'] = tune.grid_search([256])
# config_orig['model']['num_layers'] = tune.grid_search([1])
# config_orig['model']['dropout_prob'] = tune.grid_search([0.25])
# config_orig['training']['weight_decay'] = 0.0
# config_orig['data']['split'] = 'public'

# Parameters
# dataset_names =  [] #,
# datasets_16 = ['CoraML', 'CiteSeer', 'PubMed', 'CoauthorCS', 'CoauthorPhysics']
# datasets_16 = []
datasets_16 = ['CoraML', 'CiteSeer', 'PubMed',  ]
# datasets_10 = ['AmazonPhotos', 'AmazonComputers', ]
datasets_10 = []


save_tsne = False
# dataset_names = ['ogbn-arxiv', ]
# parameterless_activations = ['ELU', 'Hardshrink', 'Hardsigmoid', 'Hardtanh', 'Hardswish', 'LeakyReLU', 'LogSigmoid', 
#     'PReLU', 'ReLU', 'ReLU6', 'RReLU', 'SELU', 'CELU', 'GELU', 'Sigmoid', 'SiLU', 'Mish', 
#     'Softplus', 'Softshrink', 'Softsign', 'Tanh', 'Tanhshrink']
# parameterless_activations = ['ReLU']
# parametered_activations = ['Threshold', 'MultiheadAttention']

# good_activations = ['ReLU', 'LogSigmoid', 'GELU']
samples = 1
max_concurrent_trials = 16
max_fail = 0
std_search_range = [np.sqrt(10) ** i for i in range(-20, 0)]
coarse_search = [0, *[10 ** i for i in range(0, 10)]]
high_search_range = [10 ** i for i in range(-15, 15)]

# betas = ['dirichlet-evidence', 'preflow-l2'] # , 'postflow-l2', 
budgets = np.linspace(0.1, 1, 10)

# Missclassification
dataset_class = {
    'ood_flag': False
}

# Standard Leave Out Classes
dataset_2 = {
    'ood_flag': True,
    'ood_setting': 'poisoning',
    'ood_type': 'leave_out_classes',
    'ood_num_left_out_classes': -1,
    'ood_leave_out_last_classes': True,
}

# Normal Distribution Pertubation on features
dataset_3_norm = {
    'ood_flag': True,
    'ood_setting': "evasion",
    'ood_type': "perturb_features",
    'ood_dataset_type': "isolated",
    'ood_perturbation_type': "normal",
}

# Bernouli Distribution Pertubation on features
dataset_3_ber = {
    'ood_flag': True,
    'ood_setting': "evasion",
    'ood_type': "perturb_features",
    'ood_dataset_type': "isolated",
    'ood_perturbation_type': "bernoulli_0.5",
}

# Normal Distribution Feature Pertubation
dataset_4_norm = {
    'ood_flag': True,
    'ood_setting':  "evasion",
    'ood_type':  "perturb_features",
    'ood_dataset_type':  "budget",
    'ood_budget_per_graph':  tune.grid_search(budgets),
    'ood_perturbation_type':  "normal",
    'ood_noise_scale': 1.0,
}

dataset_4_ber = {
    'ood_flag': True,
    'ood_setting': "evasion",
    'ood_type': "perturb_features",
    'ood_dataset_type': "budget",
    'ood_budget_per_graph': tune.grid_search(budgets),
    'ood_perturbation_type': "bernoulli_0.5",
}

# Edge Pertubation
dataset_dice = {
    'ood_flag': True,
    'ood_setting': "evasion",
    'ood_type': "random_attack_dice",
    'ood_dataset_type': "budget",
    'ood_budget_per_graph': tune.grid_search(budgets),
}
# dataset_class
# Classification
datasets = [dataset_2] #, dataset_3_norm, dataset_3_ber, dataset_4_ber, dataset_4_norm , dataset_dice]
# dataset_titles = ['Preserving_Self_KNN-Weighting-Leave_Out_Classes_Poisoning', ] #,'Leave_Out_Classes_Poisoning' 'Isolated_Normal_Evasion', 'Isolated_Ber_Evasion', 'Budget_Ber_Evasion', 'Budget_Normal_Evasion',  'Dice_Evasion']
dataset_titles = ['Decoder_L2_']
for dataset_names, N in zip([datasets_16, datasets_10], [16, 10]):
    with open(f'configs/gpn/ood_loc_gpn_{N}.yaml') as f:
        config_orig = yaml.safe_load(f)

    for a_dataset, a_title in zip(datasets, dataset_titles):
        def run(config, name):
            try:
                tune.run(
                    config_helper,
                    metric='val_accuracy',
                    mode="max", 
                    num_samples=samples,
                    max_concurrent_trials=max_concurrent_trials,
                    resources_per_trial={"gpu": 0.25, 'cpu': 8},
                    config=config,
                    max_failures=max_fail,
                    name = name)
            except:
                pass

        # # #################################################################################################
        # # #################################        Activations   ##########################################
        # # #################################################################################################
        # # Activation Function Trials With Dirichlet Entropy
        # config = copy.deepcopy(config_orig)
        # config['data'] = config['data'] | a_dataset
        # config['data']['dataset'] = tune.grid_search(dataset_names)
        # config['model']['activation_type'] = tune.grid_search(parameterless_activations)
        # config['model']['entropy_reg'] = 1e-4
        # run(config, a_title + '_Entropy')

        # # Activation Function Trials - Graph latent Distance
        # config = copy.deepcopy(config_orig)
        # config['data'] = config['data'] | a_dataset
        # config['data']['dataset'] = tune.grid_search(dataset_names)
        # config['model']['activation_type'] = tune.grid_search(parameterless_activations)
        # config['model']['orig_dist_reg'] = 1e-4
        # config['model']['dist_embedding_beta'] = tune.grid_search(betas)
        # run(config, a_title + '_Graph')

        # # Activation Function Trials - KNN Distance 
        # config = copy.deepcopy(config_orig)
        # config['data'] = config['data'] | a_dataset
        # config['data']['dataset'] = tune.grid_search(dataset_names)
        # config['model']['activation_type'] = tune.grid_search(parameterless_activations)
        # config['model']['dist_embedding_beta'] = tune.grid_search(betas)
        # config['model']['dist_reg'] = 1e-4
        # run(config, a_title + '_KNN')

        # #################################################################################################
        # #################################        Parameter Searching   ##################################
        # #################################################################################################
        # #################################   Entropy   ##################################
        # config = copy.deepcopy(config_orig)
        # config['data']['dataset'] = tune.grid_search(dataset_names)
        # # config['model']['entropy_reg'] = tune.grid_search(std_search_range)
        # config['model']['activation_type'] = tune.grid_search(['Hardtanh'])
        # config['model']['save_tsne'] = save_tsne
        # run(config, f'{a_title}-TSNE-Activation')
        
        # #################################################################################################
        # #################################   Decoder   ##################################
        config = copy.deepcopy(config_orig)
        config['data']['dataset'] = tune.grid_search(dataset_names)
        config['model']['decoder_reg'] = tune.grid_search(high_search_range)
        config['model']['activation_type'] = tune.grid_search(['ReLU', 'LogSigmoid', 'GELU'])
        config['model']['save_tsne'] = save_tsne
        run(config, f'{a_title}-Decoder')

        # # # #################################   Graph Dist   ##################################
        # config = copy.deepcopy(config_orig)
        # # config['model']['dist_embedding_beta'] = tune.grid_search(betas)
        # config['data']['dataset'] = tune.grid_search(dataset_names)
        # # config['model']['entropy_reg'] = tune.grid_search([1e-5])
        # config['model']['orig_dist_reg'] = tune.grid_search([1e-6, 1e-4, 1e-2]) 
        # # config['model']['activation_type'] = 'ReLU'
        # # config['model']['dist_perserving'] = tune.grid_search([True, False])
        # config['model']['save_tsne'] = save_tsne
        # run(config, f'{a_title}-TSNE-Graph')

        # #################################   KNN Dist   ##################################
        # config = copy.deepcopy(config_orig)
        # config['data']['dataset'] = tune.grid_search(dataset_names)
        # config['model']['dist_reg'] = tune.grid_search(std_search_range)
        # config['model']['KNN_K'] = tune.grid_search([5, ])
        # config['model']['activation_type'] = tune.grid_search(good_activations)
        # config['model']['dist_embedding_beta'] = tune.grid_search(betas)
        # config['model']['dist_perserving'] = tune.grid_search([False])
        # config['model']['knn_mode'] = tune.grid_search(['knn'])
        # config['model']['save_tsne'] = save_tsne
        # run(config, f'{a_title}Tune-KNN')

        # #################################   Stress    ##################################
        # config = copy.deepcopy(config_orig)
        # config['data']['dataset'] = tune.grid_search(dataset_names)
        # config['model']['activation_type'] = tune.grid_search(good_activations)
        # config['model']['stress_scaling'] = tune.grid_search(['constant', 'linear'])
        # config['model']['stress_drop_last_N'] = 0
        # config['model']['stress_use_graph'] = tune.grid_search([True, False])
        # config['model']['stress_drop_orthog'] = tune.grid_search([False])
        # config['model']['stress_row_normalize'] = tune.grid_search([False])
        # config['model']['stress_sym_single_dists'] = tune.grid_search([True, False])
        # config['model']['stress_force_connected'] = tune.grid_search(['true-random', 'true-fully', 'false'])
        # config['model']['entropy_reg'] = tune.grid_search([1e-5, ])
        # config['model']['stress_reg'] = 1e-2
        # config['model']['stress_reg'] = tune.grid_search(std_search_range)
        # run(config, f'{a_title}-Tune-Stress-Scaling-2')

        # #################################   Lipschitz Reg    ##################################
        # config = copy.deepcopy(config_orig)
        # config['model']['entropy_reg'] = 1e-4
        # config['training']['stopping_patience'] = 500
        # config['model']['activation_type'] = tune.grid_search(good_activations)
        # config['model']['lipschitz_reg'] = tune.grid_search([0, *[math.sqrt(10) ** i for i in range(-30, 30)]])
        # config['data']['dataset'] = tune.grid_search(a_dataset)
        # run(config, 'Tune-Lipschitz-Patient')

        # #################################   Lipschitz Init   ##################################
        # config = copy.deepcopy(config_orig)
        # config['model']['entropy_reg'] = 1e-4
        # config['training']['stopping_patience'] = 500
        # config['model']['activation_type'] = tune.grid_search(good_activations)
        # config['model']['lipschitz_reg'] = 1e-4
        # config['model']['lipschitz_init'] = tune.grid_search([0, *[math.sqrt(10) ** i for i in range(-15, 45)]])
        # config['data']['dataset'] = tune.grid_search(a_dataset)
        # run(config, 'Tune-LipschitzInit-Patient')

        ##################################   Overlapping   ##################################
        # config = copy.deepcopy(config_orig)
        # config['model']['entropy_reg'] = tune.grid(coarse_search)
        # config['model']['lipschitz_reg'] = tune.grid(coarse_search)
        # config['model']['orig_dist_reg'] = tune.grid(coarse_search)
        # config['data']['dataset'] = tune.grid_search(a_dataset)
        # run(config, 'FullGridSearch')

