import random
import pdb
import itertools
from functools import partial
from re import T
from typing import Optional, List, Any
import torch
import torch.nn as nn
from torch import Tensor
import torch_geometric.utils as tu
from torch_geometric.typing import Adj
from torch_geometric.nn import MessagePassing, knn_graph
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, mul, sum
from torch.nn import Sequential as Seq, Linear, ReLU
from torchmetrics.functional import pairwise_cosine_similarity
import tqdm
from sklearn.neighbors import kneighbors_graph
import sklearn.manifold
import scipy.sparse
import numpy as np
from sklearn.utils.graph_shortest_path import graph_shortest_path
import scipy.sparse as sp
import networkx as nx

def propagation_wrapper(
        propagation: callable, x: Tensor, edge_index: Adj,
        unc_node_weight: Optional[Tensor] = None,
        unc_edge_weight: Optional[Tensor] = None,
        node_normalization: str = 'none',
        return_normalizer: bool = False,
        **kwargs) -> Tensor:
    """wraps default propagation layer with the option of weighting edges or nodes additionally

    Args:
        propagation (callable): original propagation method
        x (Tensor): node features
        edge_index (Adj): edges
        unc_node_weight (Optional[Tensor], optional): additional weight of nodes. Defaults to None.
        unc_edge_weight (Optional[Tensor], optional): additional weight of edges. Defaults to None.
        node_normalization (str, optional): mode of node normalization ('none', 'reweight', 'reweight_and_scale'). Defaults to 'none'.
        return_normalizer (bool, optional): whether or whether not to return normalization factor. Defaults to False.

    Raises:
        AssertionError: raised if unsupported mode of normalization is passed

    Returns:
        Tensor: node features after propagation
    """

    kwargs.setdefault('edge_weight', None)
    edge_weight = kwargs['edge_weight']

    if unc_node_weight is None:
        unc_node_weight = torch.ones_like(x[:, 0]).view(-1, 1)

    # original scale of weighting
    ones = torch.ones_like(unc_node_weight)

    if unc_edge_weight is None:
        x = torch.cat([unc_node_weight * x, unc_node_weight, ones], dim=-1)
        x = propagation(x, edge_index=edge_index, **kwargs)
        dif_ones = x[:, -1].view(-1, 1)
        dif_w = x[:, -2].view(-1, 1)
        dif_x = x[:, 0:-2]

    # unc_edge_weight is not None
    else:
        unc_edge_weight = unc_edge_weight if edge_weight is None else edge_weight * unc_edge_weight

        # diffuse 1 with previous weighting
        dif_ones = propagation(ones, edge_index=edge_index, **kwargs)

        # diffuse x, w with new weighting
        kwargs['edge_weight'] = unc_edge_weight
        x = torch.cat([unc_node_weight * x, unc_node_weight], dim=-1)
        x = propagation(x, edge_index=edge_index, **kwargs)
        dif_w = x[:, -1].view(-1, 1)
        dif_x = x[:, 0:-1]

    if node_normalization in ('reweight_and_scale', None):
        # sum_u c_vu * (sum_u c_vu * w_u * x_u) / (sum_u c_vu * w_u)
        x = dif_ones * dif_x / dif_w

    elif node_normalization == 'reweight':
        x = dif_x / dif_w

    elif node_normalization == 'none':
        dif_w = None
        x = dif_x

    else:
        raise AssertionError

    if return_normalizer:
        return x, dif_w
    return x


def mat_norm(edge_index: Adj, edge_weight: Optional[Tensor] = None, num_nodes: Optional[int] = None,
             add_self_loops: bool = True, dtype: Optional[Any] = None,
             normalization: str = 'sym', **kwargs) -> Adj:
    """computes normalization of adjanceny matrix

    Args:
        edge_index (Adj): representation of edges in graph
        edge_weight (Optional[Tensor], optional): optional tensor of edge weights. Defaults to None.
        num_nodes (Optional[int], optional): number of nodes. Defaults to None.
        add_self_loops (bool, optional): flag to add self-loops to edges. Defaults to True.
        dtype (Optional[Any], optional): dtype . Defaults to None.
        normalization (str, optional): ['sym', 'gcn', 'in-degree', 'out-degree', 'rw', 'in-degree-sym', 'sym-var']. Defaults to 'sym'.

    Raises:
        AssertionError: raised if unsupported normalization is passed to the function

    Returns:
        Adj: normalized adjacency matrix
    """

    if normalization in ('sym', 'gcn'):
        return gcn_norm(
            edge_index, edge_weight=edge_weight, num_nodes=num_nodes,
            add_self_loops=add_self_loops, dtype=dtype, **kwargs)

    if normalization in ('in-degree', 'out-degree', 'rw'):
        return deg_norm(
            edge_index, edge_weight=edge_weight, num_nodes=num_nodes,
            add_self_loops=add_self_loops, dtype=dtype)

    if normalization in ('in-degree-sym', 'sym-var'):
        return inv_norm(
            edge_index, edge_weight=edge_weight, num_nodes=num_nodes,
            add_self_loops=add_self_loops, dtype=dtype)

    raise AssertionError


def deg_norm(edge_index: Adj, edge_weight: Optional[Tensor] = None, num_nodes: Optional[int] = None,
             add_self_loops: bool = True, dtype: Optional[Any] = None,
             normalization: str = 'in-degree') -> Adj:
    """degree normalization

    Args:
        edge_index (Adj): representation of edges in graph
        edge_weight (Optional[Tensor], optional): optional tensor of edge weights. Defaults to None.
        num_nodes (Optional[int], optional): number of nodes. Defaults to None.
        add_self_loops (bool, optional): flag to add self-loops to edges. Defaults to True.
        dtype (Optional[Any], optional): dtype . Defaults to None.
        normalization (str, optional): ['in-degree', 'out-degree', 'rw']. Defaults to 'sym'.

    Raises:
        AssertionError: raised if unsupported normalization is passed to the function

    Returns:
        Adj: normalized adjacency matrix
    """

    fill_value = 1.0

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)

        if normalization == 'in-degree':

            in_deg = sum(adj_t, dim=0)
            in_deg_inv_sqrt = in_deg.pow_(-0.5)
            in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float('inf'), 0.)
            # A = D_in^-1 * A
            adj_t = mul(adj_t, in_deg_inv_sqrt.view(1, -1))

        elif normalization in ('out-degree', 'rw'):
            out_deg = sum(adj_t, dim=1)
            out_deg_inv_sqrt = out_deg.pow_(-0.5)
            out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float('inf'), 0.)
            # A = A * D_out^-1
            adj_t = mul(adj_t, out_deg_inv_sqrt.view(-1, 1))

        else:
            raise AssertionError

        return adj_t

    num_nodes = tu.num_nodes.maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones(
            (edge_index.size(1), ),
            dtype=dtype,
            device=edge_index.device)

    if add_self_loops:
        edge_index, edge_weight = tu.add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index[0], edge_index[1]

    if normalization == 'in-degree':
        in_deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        in_deg_inv = 1.0 / in_deg
        in_deg_inv.masked_fill_(in_deg_inv == float('inf'), 0)
        edge_weight = in_deg_inv[col] * edge_weight

    elif normalization in ('out-degree', 'rw'):
        out_deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        out_deg_inv = 1.0 / out_deg
        out_deg_inv.masked_fill_(out_deg_inv == float('inf'), 0)
        edge_weight = out_deg_inv[row] * edge_weight

    else:
        raise AssertionError

    return edge_index, edge_weight


def gcn_norm(edge_index: Adj, edge_weight: Optional[Tensor] = None, num_nodes: Optional[int] = None,
             improved: bool = False, add_self_loops: bool = True, dtype: Optional[Any] = None) -> Adj:
    """gcn-like normalization of adjacency matrix

    Args:
        edge_index (Adj): representation of edges in graph
        edge_weight (Optional[Tensor], optional): optional tensor of edge weights. Defaults to None.
        num_nodes (Optional[int], optional): number of nodes. Defaults to None.
        improved (bool, optional): whether or whether not to use improved normalization (weighting self-loops twice). Defaults to False.
        add_self_loops (bool, optional): flag to add self-loops to edges. Defaults to True.
        dtype (Optional[Any], optional): dtype . Defaults to None.

    Returns:
        Adj: normalized adjacency matrix
    """


    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)

        in_deg = sum(adj_t, dim=1)
        #out_deg = sum(adj_t, dim=0)

        in_deg_inv_sqrt = in_deg.pow_(-0.5)
        # out_deg_inv_sqrt = out_deg.pow_(-0.5)

        in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float('inf'), 0.)
        # out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float('inf'), 0.)
        # A = D_in^-0.5 * A * D_out^-0.5
        adj_t = mul(adj_t, in_deg_inv_sqrt.view(1, -1))
        adj_t = mul(adj_t, in_deg_inv_sqrt.view(-1, 1))

        return adj_t

    num_nodes = tu.num_nodes.maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = tu.add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]

    out_deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    in_deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)

    in_deg_inv_sqrt = in_deg.pow_(-0.5)
    # out_deg_inv_sqrt = out_deg.pow_(-0.5)

    # out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float('inf'), 0)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float('inf'), 0)
    # A = D_in^-0.5 * A * D_out^-0.5
    edge_weight = in_deg_inv_sqrt[col] * edge_weight * in_deg_inv_sqrt[row]

    return edge_index, edge_weight


def inv_norm(edge_index: Adj, edge_weight: Optional[Tensor] = None, num_nodes: Optional[int] = None,
             add_self_loops: bool = True, dtype: Optional[Any] = None) -> Adj:
    """normalization layer with symmetric inverse-degree normalization

    Args:
        edge_index (Adj): representation of edges in graph
        edge_weight (Optional[Tensor], optional): optional tensor of edge weights. Defaults to None.
        num_nodes (Optional[int], optional): number of nodes. Defaults to None.
        add_self_loops (bool, optional): flag to add self-loops to edges. Defaults to True.
        dtype (Optional[Any], optional): dtype . Defaults to None.

    Returns:
        Adj: normalized adjacency matrix
    """

    fill_value = 1.0

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)

        in_deg = sum(adj_t, dim=1)
        in_deg_inv = in_deg.pow_(-1.0)
        in_deg_inv.masked_fill_(in_deg_inv == float('inf'), 0.)
        adj_t = mul(adj_t, in_deg_inv.view(1, -1))
        adj_t = mul(adj_t, in_deg_inv.view(-1, 1))

        return adj_t

    num_nodes = tu.num_nodes.maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = tu.add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]

    out_deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    in_deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)

    in_deg_inv = in_deg.pow_(-1.0)
    in_deg_inv.masked_fill_(in_deg_inv == float('inf'), 0)
    # A = D_in^-1 * A * D_out^-1
    edge_weight = in_deg_inv[col] * edge_weight * in_deg_inv[row]

    return edge_index, edge_weight


class PropagationChain(nn.Module):
    """convenience layer which allows creation of a list chain of propagations (similar to torch.nn.Sequential)"""

    def __init__(self, propagations: List[callable], activations: Optional[List[callable]] = None):
        super().__init__()
        self.propagations = propagations
        self.activations = activations

    def forward(self, x, edge_index, **kwargs):
        h = x
        for i, p in enumerate(self.propagations):
            h = p(h, edge_index=edge_index, **kwargs)
            if self.activations is not None:
                act = self.activations[i]
                h = act(h)

        return h


class GraphIdentity(nn.Module):
    """simple no-op layer compatible with API of typical graph-convolutional layers"""
    def __init__(self, *_, **__):
        super().__init__()

    def forward(self, x: Tensor, *_, **__) -> Tensor:
        return x


class ConnectedComponents(MessagePassing):
    """layer finding connected components of a graph"""
    def __init__(self):
        super().__init__(aggr="max")

    def forward(self, data):
        x = torch.arange(data.num_nodes).view(-1, 1)
        last_x = torch.zeros_like(x)

        while not x.equal(last_x):
            last_x = x.clone()
            x = self.propagate(data.edge_index, x=x)
            x = torch.max(x, last_x)

        unique, perm = torch.unique(x, return_inverse=True)
        perm = perm.view(-1)

        if "batch" not in data:
            return unique.size(0), perm

        cc_batch = unique.scatter(dim=-1, index=perm, src=data.batch)
        return cc_batch.bincount(minlength=data.num_graphs), perm

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


def l1_dist(x_i, x_j):
    return (x_i-x_j).norm(p=1, dim=-1).view(-1, 1)
    

def l2_dist(x_i, x_j):
    return (x_i-x_j).norm(p=2, dim=-1).square().view(-1, 1)
    
    
def kl_divergence(x_i, x_j, dist):     
    x_i = dist(x_i)
    x_j = dist(x_j)
    return torch.distributions.kl.kl_divergence(x_i, x_j).view(-1, 1) + torch.distributions.kl.kl_divergence(x_j, x_i).view(-1, 1)
        
def to_scipy(x):
    return scipy.sparse.csr_array(x.detach().cpu().numpy())
        
class GraphDistance(MessagePassing):
    def __init__(self, distance_fun = 'l2', save_to_orig=True):
        super().__init__(aggr='add')
        self.orig_dists = None
        self.save_to_orig=save_to_orig
        self.scale_factor = nn.Parameter(torch.Tensor(1, 1))
        nn.init.constant_(self.scale_factor, 0.4)
        # pdb.set_trace()
        if distance_fun == 'l2':
            self.distance_fun = l2_dist
        elif distance_fun == 'l1':
            self.distance_fun = l1_dist
        elif distance_fun == 'dirichlet':
            self.distance_fun = partial(kl_divergence, dist = torch.distributions.dirichlet.Dirichlet)
        else: 
            raise NotImplementedError

    def first_forward(self, x, edge_indexs):
        if self.save_to_orig:
            self.propagate(x=x, edge_index = edge_indexs, edge_weight=torch.ones([edge_indexs.shape[0], 1]))

    def forward(self, proj_x, edge_indexs):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        dists = []
        if len(proj_x.shape) == 3:
            for i in range(proj_x.shape[0]): 
                dists.append(self.propagate(x=proj_x[i, ...], edge_index = edge_indexs, edge_weight=torch.ones([edge_indexs.shape[0], 1])))
            return torch.sum(torch.stack(dists), dim=0)
        else:
            return self.propagate(x=proj_x, edge_index = edge_indexs, edge_weight=torch.ones([edge_indexs.shape[0], 1]))

    def message(self, x_i, x_j, edge_weight):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]   
        
        if self.save_to_orig:
            self.orig_dists = (to_scipy(x_i) * to_scipy(x_j)).sum(axis=1)[:, np.newaxis]
            self.orig_dists = torch.Tensor(self.orig_dists).to(device='cuda')
            self.save_to_orig=False
            return self.orig_dists
        else:
            if self.orig_dists is not None:
                return ((self.distance_fun(x_i, x_j) - self.scale_factor * self.orig_dists).abs() + 1e-8).sqrt()
            else:
                return self.distance_fun(x_i, x_j)

class KNN_Distance(MessagePassing):
    def __init__(self, save_to_orig=False, KNN_K=50, sigma=.0001, distance_fun='l2', mode='knn'):

        super().__init__(aggr='add')
        self.edge_indexs = None
        self.edge_weights = None
        self.KNN_K = KNN_K
        self.sigma = sigma
        self.save_to_orig = save_to_orig
        self.orig_dists = None
        
        if save_to_orig:
            self.scale_factor = nn.Parameter(torch.Tensor(1, 1))
            nn.init.constant_(self.scale_factor, 0.4)
        
        if mode == 'random':
            assert save_to_orig 
            self.mode = 'random'
        elif mode == 'knn' or mode is None:
            self.mode = 'knn'
        else:
            raise NotImplementedError
            
        if distance_fun == 'l2':
            self.distance_fun = l2_dist
        elif distance_fun == 'l1':
            self.distance_fun = l1_dist
        elif distance_fun == 'dirichlet':
            self.distance_fun = partial(kl_divergence, dist = torch.distributions.dirichlet.Dirichlet)
        else: 
            raise NotImplementedError
    
    def first_forward(self, x):
        if self.mode == 'random':
            a = torch.arange(x.shape[0])
            b = torch.randint(0, x.shape[0], [x.shape[0]*self.KNN_K])
            self.edge_indexs = torch.stack([a.repeat_interleave(self.KNN_K), b])
            self.edge_weights = torch.ones(self.edge_indexs.shape[1])
        elif self.mode == 'knn':
            features = x.clone()
            adj = kneighbors_graph(features.cpu().numpy(), self.KNN_K, metric='cosine')
            inner, outer, weight = scipy.sparse.find(adj)
            inner = torch.Tensor(inner).to(torch.long).to(torch.device('cuda'))
            outer = torch.Tensor(outer).to(torch.long).to(torch.device('cuda'))

            self.edge_indexs = torch.stack([inner, outer])
            cosine_distance = 1 - torch.nn.CosineSimilarity()(features[self.edge_indexs[0]], features[self.edge_indexs[1]])
            self.edge_weights = torch.exp(-cosine_distance/self.sigma).nan_to_num() 
            
        self.edge_indexs, self.edge_weights = tu.to_undirected(self.edge_indexs, self.edge_weights)
        self.edge_indexs, self.edge_weights = self.edge_indexs.to(torch.device('cuda')), self.edge_weights.to(torch.device('cuda'))
        if self.save_to_orig:
            self.forward(x.to(torch.device('cuda')))

    def forward(self, proj_x):
        dists = []
        if len(proj_x.shape) == 3:
            for i in range(proj_x.shape[0]): 
                return self.propagate(self.edge_indexs, x=proj_x[i, ...], edge_weight=self.edge_weights)
            return torch.sum(torch.stack(dists), dim=0)
        else:
            return self.propagate(self.edge_indexs, x=proj_x, edge_weight=self.edge_weights)

    def message(self, x_i, x_j, edge_weight):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        if self.save_to_orig:
            self.orig_dists = (to_scipy(x_i) * to_scipy(x_j)).sum(axis=1)[:, np.newaxis]
            self.orig_dists = torch.Tensor(self.orig_dists).to(device='cuda')
            self.save_to_orig=False
            return self.orig_dists
        else:
            if self.orig_dists is not None:
                return ((self.distance_fun(x_i, x_j) - self.scale_factor * self.orig_dists).abs() + 1e-8).sqrt()
            else:
                return self.distance_fun(x_i, x_j)

class Stress(nn.Module):
    def __init__(self, *, metric, use_graph, knn_k, scaling, row_normalize, drop_orthog, drop_last, sym_single_dists, force_connected) -> None:
        super(Stress, self).__init__()
        self.dists = None

        self.use_graph = use_graph
        self.metric_str = metric
        if metric == 'cosine':
            metric = lambda x: 1 - torch.nn.CosineSimilarity()(*x)
        else:
            raise NotImplementedError
        self.metric = metric
        self.knn_k = knn_k
        self.scaling = scaling
        self.row_normalize = row_normalize
        self.drop_orthog = drop_orthog
        self.drop_last = drop_last
        self.sym_single_dists = sym_single_dists
        self.force_connected = force_connected
        self.removed = None
        
    def generate_projection(self, geo_dist, k=3):
        e, u = sp.linalg.eigs(geo_dist, k=5)
        print(u.shape)
        u = u[:, 1:]
        print(e)
        return u.real
    
    @staticmethod
    def remove_disconnected(sp_mat):
        G = nx.DiGraph()
        row, col, val = sp.find(sp_mat)
        G.add_weighted_edges_from([(u, v, w) for u, v, w in zip(row, col, val)])   
        G_largest_component = max(nx.strongly_connected_components(G), key=len)
        removed = np.sort(list(set(G.nodes) - set(G_largest_component)))
        G = G.subgraph(G_largest_component)
        
        return nx.to_scipy_sparse_matrix(G), removed
        
    @staticmethod
    def force_strongly_connected(sp_mat, do_random):
        G = nx.from_scipy_sparse_matrix(sp_mat, create_using=nx.DiGraph)
        G_components = list(nx.strongly_connected_components(G))
        
        central_nodes = []
        for a_comp in G_components:
            sub_G = G.subgraph(a_comp)
            central_nodes.append(max(sub_G.degree, key=lambda x: x[1])[0])
        comp_cnt = len(central_nodes)
        
        if comp_cnt == 1:
            return sp_mat
        
        if do_random:
            for i, u in enumerate(central_nodes):
                active = set(range(comp_cnt)) - set([i])
                # pdb.set_trace()
                for j_C in np.random.choice(list(active), size=5):
                    G.add_edge(u, random.choice(list(G_components[j_C])), weight=1)
                    G.add_edge(random.choice(list(G_components[j_C])), u, weight=1)
        else:
            for u, v in itertools.product(central_nodes, central_nodes):
                if (u,v) not in G.edges:
                    G.add_edge(u, v, weight=1)
                if (v, u) not in G.edges:
                    G.add_edge(v, u, weight=1)
        return nx.to_scipy_sparse_matrix(G) 
    
    @staticmethod
    def remove_orthogonal_edges(sp_mat):
        rows, cols, edge_vals = sp.find(sp_mat)
        G = nx.from_scipy_sparse_matrix(sp_mat, create_using=nx.DiGraph)

        while np.max(edge_vals) > .99:
            ind = np.argmax(edge_vals)
            G.remove_edge(rows[ind], cols[ind])
            if not nx.is_strongly_connected(G):
                G.add_edge(rows[ind], cols[ind], weight=edge_vals[ind])
            rows = np.delete(rows, ind)
            cols = np.delete(cols, ind)
            edge_vals = np.delete(edge_vals, ind) 
        return nx.to_scipy_sparse_matrix(G)            
    
    def do_scaling(self, dists):
        if self.scaling == 'log':
            dists.data = np.log(1 + dists.data) + .1
        elif self.scaling == 'sqrt':
            dists.data = np.sqrt(dists.data)
        elif self.scaling == 'square':
            dists.data = np.square(dists.data)
        elif self.scaling == 'linear':
            dists.data = np.abs(dists.data)
        elif self.scaling == 'constant':
            dists.data = np.ones_like(dists.data)
        elif self.scaling == 'kernel':
            dists.data = np.exp(-dists.data)
        elif self.scaling == 'forced-linear':
            dists.data[np.argsort(dists.data)] = np.linspace(0.01, 1, dists.data.size)
        else:
            raise ValueError
        return dists
    
    def drop_farthest_edges(self, single_dists):
        rows, cols, edge_vals = sp.find(single_dists)
        G = nx.Graph()
        G.add_weighted_edges_from([(u, v, w) for u, v, w in zip(rows, cols, edge_vals)])    
        for _ in range(self.drop_last):
            bridges = list(nx.bridges(G))
            ind = -1
            while (ind == -1) or (tuple(sorted([rows[ind], cols[ind]])) in bridges):
                if ind != -1:
                    rows = np.delete(rows, ind)
                    cols = np.delete(cols, ind)
                    edge_vals = np.delete(edge_vals, ind)
                ind = np.argmax(edge_vals)
            G.remove_edge(*sorted([int(rows[ind]), int(cols[ind])]))
            
            rows = np.delete(rows, ind)
            cols = np.delete(cols, ind)
            edge_vals = np.delete(edge_vals, ind) 
                            
        return nx.to_scipy_sparse_matrix(G)

    @staticmethod
    def do_row_normalize(sp_mat):
        sp_mat.data = sp_mat.data - 2
        rows, cols, edge_vals = sp.find(sp_mat)
        hot_dists = sp.coo_matrix((np.ones_like(edge_vals),(rows, cols)), shape=sp_mat.shape)
        min_row = sp_mat.min(axis=1)
        sp_mat.data = sp_mat.data + 2
        min_row.data = min_row.data + 2
        min_dists = hot_dists.multiply(min_row)
        sp_mat = sp_mat - min_dists
        sp_mat.data = sp_mat.data + .01
        return sp_mat

    def init_helper(self, features: Tensor, edge_list) -> Tensor:
        N = features.shape[0]
        if self.use_graph:
            edge_list = edge_list.cpu().numpy()
            rows, cols = edge_list
        else:
            adj = kneighbors_graph(features.cpu().numpy(), self.knn_k, metric=self.metric_str)
            rows, cols = adj.nonzero()
        
        edge_vals = self.metric([features[rows, :], features[cols, :]]).cpu().numpy()
        single_dists = sp.coo_matrix((edge_vals, (rows, cols)), shape=(N, N))
        
        if 'true' in self.force_connected.casefold():
            if 'random' in self.force_connected.casefold():
                gen_random_connections = True
            elif 'fully' in self.force_connected.casefold():
                gen_random_connections = False
            else:
                raise NotImplementedError
            single_dists = self.force_strongly_connected(single_dists, gen_random_connections)
        elif 'false' in self.force_connected.casefold():
            pass
        elif 'remove' in self.force_connected.casefold():
            single_dists, self.removed = self.remove_disconnected(single_dists)
        else:
            raise ValueError
            
        if self.drop_orthog:
            single_dists = self.remove_orthogonal_edges(single_dists)

        if self.drop_last:
            single_dists = self.drop_farthest_edges(single_dists)

        if self.row_normalize:
            single_dists = self.do_row_normalize(single_dists)
        
        if self.scaling is not None:
            single_dists = self.do_scaling(single_dists)

        if self.sym_single_dists:
            single_dists = single_dists.T + single_dists
            
        geo_dist = graph_shortest_path(single_dists, directed=False)
        v = self.generate_projection(geo_dist, k=3)
        self.dists = torch.Tensor(geo_dist).to(device='cuda')
        return v
        
    def forward(self, features: Tensor) -> Tensor:
        # We partially sum it here, we finish the summation later (to appease the formating requirements for nodes being fed in)
        dists = (self.dists - torch.cdist(features, features)).abs()
        return dists.sum(-1, keepdim=True)