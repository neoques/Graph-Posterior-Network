import os
from collections import OrderedDict
from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import numpy as np
import random
import string
import matplotlib as mpl
import plotly.express as px
import plotly
import pdb
import sklearn.manifold
import sys
import plotly.graph_objects as go

# from tsne_torch import TorchTSNE as TSNE

from gpn.nn import uce_loss, entropy_reg
from gpn.nn.loss import mse_p, mae_b
from gpn.layers import APPNPPropagation, LinearSequentialLayer
from gpn.utils.config import ModelConfiguration
from gpn.utils import apply_mask
from gpn.utils import Prediction, ModelConfiguration
from gpn.layers import Density, Evidence, ConnectedComponents, utils
from gpn.nn import LipschitzLayers
from .model import Model
import gpn.utils.loss as custom_loss

class SmoothMapping(nn.Module):
    def __init__(self, base):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha: trainable parameter
        '''
        super(SmoothMapping, self).__init__()
        self.base = base
        
    def forward(self, input):
        return torch.pow(self.base, input) - 1/torch.pow(self.base, input)


class GPN(Model):
    """Graph Posterior Network model"""

    def __init__(self, params: ModelConfiguration):
        super().__init__(params)
        self.i = 0
        if self.params.lipschitz_reg == 0:
            linear_layer = nn.Linear
        else:
            linear_layer = LipschitzLayers.LipschitzLinear
        

        if 'SmoothMapping' in params.activation_type:
            base = float(params.activation_type.split('-')[-1])
            self.activation_function = SmoothMapping(base)
        else:
            self.activation_function = getattr(nn, params.activation_type)()

        self.linear_input_encoder = nn.Linear(self.params.dim_features, self.params.dim_hidden)
        
        layers = [
            self.activation_function, 
            nn.Dropout(p=self.params.dropout_prob)
        ]

        for _ in range(self.params.num_layers):
            layers.extend([ 
                linear_layer(self.params.dim_hidden, self.params.dim_hidden),
                self.activation_function, 
                nn.Dropout(p=self.params.dropout_prob)
                ]
            )

            
        self.latent_encoder = nn.Sequential(*layers, linear_layer(self.params.dim_hidden, self.params.dim_latent))

        use_batched = True if self.params.use_batched_flow else False 
        self.flow = Density(
            dim_latent=self.params.dim_latent,
            num_mixture_elements=self.params.num_classes,
            radial_layers=self.params.radial_layers,
            maf_layers=self.params.maf_layers,
            gaussian_layers=self.params.gaussian_layers,
            use_batched_flow=use_batched)

        self.evidence = Evidence(scale=self.params.alpha_evidence_scale)

        self.propagation = APPNPPropagation(
            K=self.params.K,
            alpha=self.params.alpha_teleport,
            add_self_loops=self.params.add_self_loops,
            cached=False,
            normalization='sym')

        if 'dirichlet' in self.params.dist_embedding_beta:
            distance_fun = 'dirichlet'
        elif 'l2' in self.params.dist_embedding_beta:
            distance_fun = 'l2'
        elif 'l1' in self.params.dist_embedding_beta:
            distance_fun = 'l1'
        else:
            raise NotImplementedError

        if self.params.decoder_reg is not None and self.params.decoder_reg != 0:
            layers = [
                nn.Linear(self.params.dim_latent, self.params.dim_hidden),
                self.activation_function, 
                nn.Linear(self.params.dim_hidden, self.params.dim_features),
            ]
            self.decoder = nn.Sequential(*layers)
                
        if self.params.dist_reg is not None and self.params.dist_reg != 0:
            self.custom_dist = utils.KNN_Distance(save_to_orig=self.params.dist_perserving, 
                                                mode=self.params.knn_mode, 
                                                KNN_K = self.params.KNN_K, 
                                                sigma = self.params.dist_sigma, 
                                                distance_fun=distance_fun)
        else:
            self.custom_dist = None
         
        if self.params.orig_dist_reg is not None and self.params.orig_dist_reg != 0:
            self.graph_dist = utils.GraphDistance(save_to_orig=self.params.dist_perserving, 
                                                  distance_fun=distance_fun)
        else:
            self.graph_dist = None
            
        if self.params.stress_reg is not None and self.params.stress_reg != 0:
            self.stress = utils.Stress(use_graph = self.params.stress_use_graph, 
                                       metric=self.params.stress_metric, 
                                       knn_k = self.params.stress_knn_k,
                                       scaling = self.params.stress_scaling, 
                                       row_normalize = self.params.stress_row_normalize,
                                       drop_last = self.params.stress_drop_last_N,
                                       drop_orthog = self.params.stress_drop_orthog,
                                       force_connected = self.params.stress_force_connected,
                                       sym_single_dists = self.params.stress_sym_single_dists)
        else:
            self.stress = None
        
        random.seed()
        self.id = ''.join(random.choices(string.ascii_lowercase, k=5))
        assert self.params.pre_train_mode in ('encoder', 'flow', None)
        assert self.params.likelihood_type in ('UCE', 'nll_train', 'nll_train_and_val', 'nll_consistency', None)
        self.random_points = None
         
    def forward(self, data: Data) -> Prediction:
        return self.forward_impl(data)

    def forward_impl(self, data: Data) -> Prediction:
        data = data.to(device='cuda')
        edge_index = data.edge_index if data.edge_index is not None else data.adj_t
        if self.i == 0:
            if self.graph_dist is not None:
                self.graph_dist.first_forward(data.x, edge_index)
            if self.custom_dist is not None:
                self.custom_dist.first_forward(data.x)
            if self.stress is not None:
                eigenvectors = self.stress.init_helper(data.x, edge_index) 

            
        self.i = self.i + 1
            
        if self.params.lipschitz_reg == 0:
            h = self.linear_input_encoder(data.x)
            z = self.latent_encoder(h)
            lipschitz_constant = torch.tensor([0])
        else:
            raise NotImplementedError
            h, c0 = self.linear_input_encoder(data.x)
            h1 = self.act(h)
            z, c1 = self.latent_encoder(h1)
            lipschitz_constant = c0 * c1
        # Goal here is to ensure that z is embedded such that the embedding preserves distances, 
        # to that end we construct a KNN matrix generate L, and then generate the regularization term.

        # compute feature evidence (with Normalizing Flows)
        # log p(z, c) = log p(z | c) p(c)
        p_c = self.get_class_probalities(data)

        flow, predensity = self.flow(z)
        log_q_ft_per_class = flow + p_c.view(1, -1).log()

        if '-plus-classes' in self.params.alpha_evidence_scale:
            further_scale = self.params.num_classes
        else:
            further_scale = 1.0

        beta_ft = self.evidence(log_q_ft_per_class, dim=self.params.dim_latent, further_scale=further_scale).exp()

        alpha_features = 1.0 + beta_ft
        
        # if self.orig_dist_reg.use_dist_preserving:
        #     orig_lap_eig_dist = self.orig_dist_reg.init_helper(data.x, edge_index)
        #     self.orig_dist_reg.use_dist_preserving = False
        if self.params.decoder_reg is not None and self.params.decoder_reg != 0:
            decoded_features = self.decoder(z)
            # decoder_reg = nn.CosineSimilarity(dim=-1)(data.x, decoded_features)
            decoder_reg = torch.linalg.vector_norm(data.x - decoded_features, ord=2, dim=-1)
        else:
            decoder_reg = None
                
        if not self.stress is None:
            stress = self.stress(z)
        else:
            stress = None

        if self.graph_dist is not None:
            if 'evidence' in self.params.dist_embedding_beta.casefold():
                orig_lap_eig_dist = self.graph_dist(alpha_features, edge_index)
            elif 'preflow' in self.params.dist_embedding_beta.casefold():
                orig_lap_eig_dist = self.graph_dist(z, edge_index)
            elif 'postflow' in self.params.dist_embedding_beta.casefold():
                orig_lap_eig_dist = self.graph_dist(predensity, edge_index)
            else:
                raise NotImplementedError
        else:
            orig_lap_eig_dist = None
            
        if self.custom_dist is not None:
            if 'evidence' in self.params.dist_embedding_beta.casefold():
                lap_eig_dist = self.custom_dist(alpha_features)
            elif 'preflow' in self.params.dist_embedding_beta.casefold():
                lap_eig_dist = self.custom_dist(z)
            elif 'postflow' in self.params.dist_embedding_beta.casefold():
                lap_eig_dist = self.custom_dist(predensity)
            else:
                raise NotImplementedError
        else:
            lap_eig_dist = None
            
        beta = self.propagation(beta_ft, edge_index)
        alpha = 1.0 + beta

        soft = alpha / alpha.sum(-1, keepdim=True)
        logits = None
        log_soft = soft.log()

        max_soft, hard = soft.max(dim=-1)
        
        # ---------------------------------------------------------------------------------
        # print(self.activation_function.alpha.detach().cpu().numpy())
        
        pred = Prediction(
            decoder_reg=decoder_reg,
            # predictions and intermediary scores
            alpha=alpha,
            soft=soft,
            log_soft=log_soft,
            hard=hard,
            
            orig_dist_reg = orig_lap_eig_dist,
            lap_eig_dist = lap_eig_dist,
            stress = stress,
            lipschitz_constant = lipschitz_constant,

            logits=logits,
            latent=z,
            latent_features=z,

            hidden=h,
            hidden_features=h,

            evidence=beta.sum(-1),
            evidence_ft=beta_ft.sum(-1),
            log_ft = flow,
            log_ft_per_class=log_q_ft_per_class,

            # prediction confidence scores
            prediction_confidence_aleatoric=max_soft,
            prediction_confidence_epistemic=alpha[torch.arange(hard.size(0)), hard],
            prediction_confidence_structure=None,

            # sample confidence scores
            sample_confidence_aleatoric=max_soft,
            sample_confidence_epistemic=alpha.sum(-1),
            sample_confidence_features=alpha_features.sum(-1),
            sample_confidence_structure=None
        )
        # ---------------------------------------------------------------------------------

        return pred

    def make_tsne(self, data: Data):
        # if self.i == 1 and self.params.stress_reg != 0:
        #     plt.figure()
        #     plt.scatter(*iso_embedding.T, c=data.y.detach().cpu().numpy(), s=1)
        #     plt.savefig(f"plots/{self.id}_embedding_{self.params.orig_dist_reg}_{self.params.dist_reg}_{self.params.stress_reg}_{self.i}.png")
        # pdb.set_trace()
        if self.params.save_tsne:
            edge_index = data.edge_index if data.edge_index is not None else data.adj_t
            h = self.linear_input_encoder(data.x)
            z = self.latent_encoder(h)
            p_c = self.get_class_probalities(data)

            flow, predensity = self.flow(z)
            log_q_ft_per_class = flow + p_c.view(1, -1).log()
            

            if '-plus-classes' in self.params.alpha_evidence_scale:
                further_scale = self.params.num_classes
            else:
                further_scale = 1.0
            
            beta_ft = self.evidence(log_q_ft_per_class, dim=self.params.dim_latent, further_scale=further_scale).exp()
            beta = self.propagation(beta_ft, edge_index)
            alpha = 1.0 + beta
            os.chdir(self.params.curr_dir)
                      
            labels = data.y.detach().cpu().numpy()
            idx = np.argsort(labels)
            labels = labels.astype(str)
            labels = labels[idx]
            
            train_mask = data.train_mask.detach().cpu().numpy()
            train_mask = train_mask[idx]
            
            z = z.detach().cpu().numpy()
            
            evidence = alpha.sum(axis=-1)  
            evidence = evidence.detach().cpu().numpy()
            evidence = evidence[idx]
            evidence = np.sqrt(evidence)

            n_colors = len(np.unique(labels))
            colors = px.colors.sample_colorscale(plotly.colors.cyclical.Phase, [n/(n_colors -1) for n in range(n_colors)])
            colors[-1] = 'rgb(0, 0, 0)'
            
            if self.random_points is not None:
                self.random_points = self.random_points.detach().cpu().numpy()
            # ###########################################################################################################
            # TSNE = sklearn.manifold.TSNE(n_jobs=-1)
            # X_emb = TSNE.fit_transform(z)
            # # X_emb = TSNE(n_components=2, perplexity=30, learning_rate=100).fit_transform(z.detach().cpu().numpy())
            # X_emb = X_emb[idx, :]
            ood_labels = data.y[data.ood_mask].unique().detach().cpu().numpy()
            for a_val in ood_labels: 
                labels = np.char.replace(labels, str(a_val), 'OOD')

            # fig = px.scatter(x=X_emb[:, 0], y=X_emb[:, 1], symbol=train_mask, size=evidence, color=labels, color_discrete_sequence=colors, labels={"color": "Class ID"}, opacity=0.5)
            # fig.update_layout(height=2000, width=2000, legend=dict(bgcolor='rgba(0, 0, 0, 0)', orientation="v", yanchor="top", xanchor="right", y=1.0, x=1.0), margin=dict(l=0, r=0, t=0, b=0)) #,
            # fig.update_layout(plot_bgcolor='rgba(255, 255, 255, 255)', legend=dict(title_font_family="Times New Roman", font=dict(size=60)))
            
            # fig.update_traces(marker_sizemin=3, marker=dict(line=dict(width=0.0)))
            
            # fig.update_xaxes(visible=False)
            # fig.update_yaxes(visible=False)
            # fig.write_image(f"tsne.png")
            # fig.write_html(f"tsne.html")
            # ###########################################################################################################
            
            # TSNE = sklearn.manifold.TSNE(n_jobs=-1)
            # log_q_ft_per_class = log_q_ft_per_class.detach().cpu().numpy()
            # X_emb = TSNE.fit_transform(log_q_ft_per_class)
            # X_emb = X_emb[idx, :]
            
            # fig = px.scatter(x=X_emb[:, 0], y=X_emb[:, 1], symbol=train_mask, size=evidence, color=labels, color_discrete_sequence=colors, labels={"color": "Class ID"}, opacity=0.5)
            # fig.update_layout(height=2000, width=2000, legend=dict(bgcolor='rgba(0, 0, 0, 0)', orientation="v", yanchor="top", xanchor="right", y=1.0, x=1.0)) #,
            # fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            # fig.update_layout(plot_bgcolor='rgba(255, 255, 255, 255)', legend=dict(title_font_family="Times New Roman", font=dict(size=60)))
                                          
            # fig.update_traces(marker_sizemin=3, marker=dict(line=dict(width=0.0)))
            
            # fig.update_xaxes(visible=False)
            # fig.update_yaxes(visible=False)
            # fig.write_image(f"tsne_alpha.png")
            # fig.write_html(f"tsne_alpha.html")
            # ###########################################################################################################
            z = z / np.sqrt( np.linalg.norm(z, ord=2, axis=-1, keepdims=True) )
            fig = px.scatter_3d(x=z[idx, 0], y=z[idx, 1], z=z[idx, 2], size=evidence, symbol=train_mask, color=labels, color_discrete_sequence=colors, labels={"color": "Class ID"}, opacity=0.5)
            if self.random_points is not None:
                fig2 = px.scatter_3d(x = self.random_points[:, 0], y = self.random_points[:, 1], z = self.random_points[:, 2])
                fig = go.Figure(data = fig.data + fig2.data) 
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig.write_image(f"latent.png")
            fig.write_html(f"latent.html")   
            # fig.write_image(f"tsne.eps")
            
            
            # N = np.unique(labels).size
            # cmap = plt.cm.gist_rainbow
            # cmaplist = [cmap(i) for i in range(cmap.N)]
            # cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
            # bounds = np.linspace(0,N,N+1)
            
            # _, ax = plt.subplots(1,1, figsize=(6,6))
            # scat = ax.scatter(*X_emb.T, c=labels, s=0.5)
            # plt.colorbar(scat, spacing='proportional',ticks=bounds)
            # plt.tight_layout()
            # plt.savefig(f"TSNE.png")

    def get_optimizer(self, lr: float, weight_decay: float) -> Tuple[optim.Adam, optim.Adam]:
        flow_lr = lr if self.params.factor_flow_lr is None else self.params.factor_flow_lr * lr
        flow_weight_decay = weight_decay if self.params.flow_weight_decay is None else self.params.flow_weight_decay

        flow_params = list(self.flow.named_parameters())
        flow_param_names = [f'flow.{p[0]}' for p in flow_params]
        flow_param_weights = [p[1] for p in flow_params]

        all_params = list(self.named_parameters())
        params = [p[1] for p in all_params if p[0] not in flow_param_names]

        # all params except for flow
        flow_optimizer = optim.Adam(flow_param_weights, lr=flow_lr, weight_decay=flow_weight_decay)
        model_optimizer = optim.Adam(
            [{'params': flow_param_weights, 'lr': flow_lr, 'weight_decay': flow_weight_decay},
             {'params': params}],
            lr=lr, weight_decay=weight_decay)

        return model_optimizer, flow_optimizer

    def get_warmup_optimizer(self, lr: float, weight_decay: float) -> optim.Adam:
        model_optimizer, flow_optimizer = self.get_optimizer(lr, weight_decay)

        if self.params.pre_train_mode == 'encoder':
            warmup_optimizer = model_optimizer
        else:
            warmup_optimizer = flow_optimizer

        return warmup_optimizer

    def get_finetune_optimizer(self, lr: float, weight_decay: float) -> optim.Adam:
        # similar to warmup
        return self.get_warmup_optimizer(lr, weight_decay)

    def apply_loss(self, prediction: Prediction, data: Data, approximate=True) -> Tuple[torch.Tensor, torch.Tensor]:
        def is_active(x):
            return x is not None and x != 0
        alpha_train, y = apply_mask(data, prediction.alpha, split='train')
        latent, y = apply_mask(data, prediction.latent, split='train')
        evidences, _ = apply_mask(data, prediction.evidence_ft, split='not_train')

        # Losses
        result_dict = dict()
        if is_active(self.params.uce_loss):   
            result_dict['UCE'] = uce_loss(alpha_train, y, reduction='sum') * self.params.uce_loss
        if is_active(self.params.mse_loss):   
            result_dict['MPE'] = mse_p(alpha_train, y, reduction='sum') * self.params.mse_loss
        if is_active(self.params.mae_loss):   
            result_dict['MAE'] = mae_b(alpha_train, y, reduction='sum') * self.params.mae_loss
        if is_active(self.params.nll_loss):   
            result_dict['LogLike'] = self.params.nll_loss * torch.nn.functional.nll_loss(
                (alpha_train/alpha_train.sum(axis=-1, keepdim=True)).log(), y)
        if is_active(self.params.entropy_reg):   
            result_dict['REG'] = entropy_reg(alpha_train, self.params.entropy_reg, approximate=approximate, reduction='sum')
        if is_active(self.params.LDA_loss):
            result_dict['LDA'] = custom_loss.linear_discriminative_loss(y, latent, 1e-1).sum() * self.params.LDA_loss
        if is_active(self.stress):   
            result_dict['STRESS'] =             prediction.stress.sum().sqrt()        * self.params.stress_reg
        if is_active(self.custom_dist):
            result_dict['DIST'] =               prediction.lap_eig_dist.sum()         * self.params.dist_reg
        if is_active(self.graph_dist):   
            result_dict['ORIG_DIST'] =          prediction.orig_dist_reg.sum()        * self.params.orig_dist_reg
        if is_active(self.params.lipschitz_reg):   
            result_dict['LIPSCHITZ'] =          prediction.lipschitz_constant.sum()   * self.params.lipschitz_reg
        if is_active(self.params.decoder_reg):   
            result_dict['DECODER'] =            prediction.decoder_reg.sum()          * self.params.decoder_reg 
        if is_active(self.params.fixed_point_loss):
            if self.random_points is None:
                # def sample_spherical(npoints, ndim=3):
                #     vec = np.random.randn(npoints, ndim)
                #     vec /= np.linalg.norm(vec, axis=1, keepdims=True)
                #     return vec
                # self.random_points = torch.Tensor(sample_spherical(len(y.unique()), self.params.dim_latent))
                self.random_points = torch.Tensor(
                    [
                        [np.sqrt(8/9), 0, -1/3],
                        [-np.sqrt(2/9), np.sqrt(2/3), -1/3],
                        [-np.sqrt(2/9), -np.sqrt(2/3), -1/3],
                        [0, 0, 1]
                    ]
                )
                
                self.random_points = self.random_points[y, :]
                tmp = torch.zeros([self.random_points.shape[0], latent.shape[-1]])
                tmp[:, :self.random_points.shape[1]] = self.random_points
                self.random_points = tmp.to(device=latent.device)
            if True:
                result_dict['Activation'] = torch.float_power(evidences, 0.1).sum() * 1.0
    
                
            # result_dict['FixedPoint'] = torch.maximum(
            #     torch.linalg.vector_norm(self.random_points - latent, ord=2, dim=-1), 
            #     torch.Tensor([0.1]).to(device=latent.device)
            # ).sum() #+ (1/torch.linalg.vector_norm(prediction.latent, ord=1, dim=-1)).sum()
            result_dict['FixedPoint'] = torch.linalg.vector_norm(self.random_points - latent, ord=2, dim=-1).sum()
        # Regularization Terms    
        # print(result_dict)
        return result_dict

    def loss(self, prediction: Prediction, data: Data) -> Dict[str, torch.Tensor]:
        result_dict = self.apply_loss(prediction, data)
        n_train = data.train_mask.sum() if self.params.loss_reduction == 'mean' else 1
        result_dict = {a_key: a_val/n_train for a_key, a_val in result_dict.items()}
        return result_dict

    def warmup_loss(self, prediction: Prediction, data: Data) -> Dict[str, torch.Tensor]:
        if self.params.pre_train_mode == 'encoder':
            return self.CE_loss(prediction, data)

        return self.loss(prediction, data)

    def finetune_loss(self, prediction: Prediction, data: Data) -> Dict[str, torch.Tensor]:
        return self.warmup_loss(prediction, data)

    def likelihood(self, prediction: Prediction, data: Data) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_class_probalities(self, data: Data) -> torch.Tensor:
        l_c = torch.zeros(self.params.num_classes, device=data.x.device)
        y_train = data.y[data.train_mask]

        # calculate class_counts L(c)
        for c in range(self.params.num_classes):
            class_count = (y_train == c).int().sum()
            l_c[c] = class_count

        L = l_c.sum()
        p_c = l_c / L

        return p_c
