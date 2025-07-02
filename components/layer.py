from typing import List, Optional
import math

import numpy as np
import torch
import torch_geometric.transforms
import torch_sparse
from class_resolver.contrib.torch import activation_resolver
from torch import nn, Tensor
from torch.distributions import Normal
from torch.nn import Parameter, ModuleList, LayerNorm
from torch_geometric.nn.conv.gcn_conv import gcn_norm, GCNConv
from torch_geometric.typing import Adj, OptTensor, OptPairTensor, SparseTensor, torch_sparse
from torch_scatter import scatter
from torch_geometric.nn import global_mean_pool, global_max_pool, MessagePassing, SAGEConv, GATConv
from torch_geometric.nn.inits import zeros, glorot
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index, softmax, to_torch_sparse_tensor,
    add_self_loops,
    remove_self_loops   
)
from utils.utils import get_laplacian
import torch.nn.functional as F
from torch.nn import LSTM, Linear
from dgl.nn.pytorch.gt import BiasedMHA
from torch_sparse import SparseTensor
from scipy.special import comb

from utils.utils import adj_norm, adj_to_directed_ADPA_norm

class BaseMLP(torch.nn.Module):

    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            num_layers: int,
            dropout: float = 0.,
            act: str = 'relu',
            norm: str = None,
            keep_last_act: bool = False,
            incep_alpha: float = 0.,
            residual: bool = False,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.act = activation_resolver.make(act)
        self.keep_last_act = keep_last_act
        self.alpha = incep_alpha
        self.residual = residual

        self.norms, eps = None, 1e-9
        if norm == 'layer':
            self.norms = ModuleList()
            for _ in range(num_layers - 1):
                self.norms.append(nn.LayerNorm(hidden_channels, eps=eps))
            self.norms.append(nn.LayerNorm(out_channels, eps=eps))
        elif norm == 'batch':
            self.norms = ModuleList()
            for _ in range(num_layers - 1):
                self.norms.append(nn.BatchNorm1d(hidden_channels, eps=eps))
            self.norms.append(nn.BatchNorm1d(out_channels, eps=eps))

        self.lins = torch.nn.ModuleList()
        if num_layers == 1:
            self.lins.append(Linear(in_channels, out_channels))
        else:
            self.lins.append(Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(Linear(hidden_channels, hidden_channels))
            self.lins.append(Linear(hidden_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        if self.norms:
            for norm in self.norms:
                norm.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        x_0 = x_pre = None
        for i in range(self.num_layers):
            x = self.lins[i].forward(x)

            if i < self.num_layers - 1 or self.keep_last_act:
                if self.norms:
                    x = self.norms[i](x)

                if self.residual:
                    if x_pre is not None:
                        x = x + x_pre
                    x_pre = x

                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

                if x_0 is not None and self.alpha > 0:
                    x = (1 - self.alpha) * x + self.alpha * x_0
                else:
                    x_0 = x

        return x


class EgoGraphPooling(nn.Module):

    def __init__(self, mode, num_groups=1):
        super().__init__()
        self.mode = mode.lower()
        assert self.mode in ['mean', 'max']

        self.num_groups = num_groups

    def forward(self, x_root, xs, p, batch, group_ptr=None):
        if group_ptr is not None and p.size(0) > xs[0].size(0):
            p = scatter(p, group_ptr, reduce='sum')

        for i, x in enumerate(xs):
            x = x * p.view(-1, 1)
            if self.mode == 'mean':
                xs[i] = global_mean_pool(x, batch)
            elif self.mode == 'max':
                xs[i] = global_max_pool(x, batch)

        x = torch.cat([x_root] + xs, dim=-1)

        if self.num_groups > 1:
            x = x.view(self.num_groups, -1, x.size(-1))  # G * batch_size * F
            x = torch.cat([x[i] for i in range(x.size(0))], dim=-1)  # batch_size * (F * G)

        return x


class MultiScaleUpdate(nn.Module):

    def __init__(self, mode, channels, num_layers):
        super().__init__()
        self.mode = mode.lower()
        assert self.mode in ['cat', 'max', 'lstm']

        if mode == 'lstm':
            assert channels is not None, 'channels cannot be None for lstm'
            assert num_layers is not None, 'num_layers cannot be None for lstm'
            self.lstm = LSTM(channels, (num_layers * channels) // 2,
                             bidirectional=True, batch_first=True)
            self.att = Linear(2 * ((num_layers * channels) // 2), 1)
            self.channels = channels
            self.num_layers = num_layers
        else:
            self.lstm = None
            self.att = None
            self.channels = None
            self.num_layers = None

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.lstm is not None:
            self.lstm.reset_parameters()
        if self.att is not None:
            self.att.reset_parameters()

    def forward(self, xs: List[Tensor]) -> Tensor:
        if self.mode == 'cat':
            return torch.cat(xs, dim=-1)
        elif self.mode == 'max':
            return torch.stack(xs, dim=-1).max(dim=-1)[0]
        elif self.mode == 'sum':
            return torch.stack(xs, dim=-1).sum(dim=-1)
        elif self.mode == 'lstm':
            assert self.lstm is not None and self.att is not None
            x = torch.stack(xs, dim=1)  # [num_nodes, num_layers, num_channels]
            alpha, _ = self.lstm(x)
            alpha = self.att(alpha).squeeze(-1)  # [num_nodes, num_layers]
            alpha = torch.softmax(alpha, dim=-1)
            return (x * alpha.unsqueeze(-1)).sum(dim=1)
        else:
            raise NotImplementedError


class GenLinear(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, identity_mapping=False):
        super().__init__()
        self.identity = identity_mapping and in_channels == out_channels
        if self.identity:
            self.weight = Parameter(torch.empty(in_channels, in_channels))
        else:
            self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, *args, **kwargs):
        if self.identity:
            return torch.addmm(x, x, self.weight, beta=1, alpha=1)
        else:
            return self.lin(x)


class MLPGCNConv(MessagePassing):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
        bias: bool = False,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.mlp = BaseMLP(in_channels,
                           hidden_channels,
                           out_channels,
                           num_layers,
                           dropout=dropout,
                           keep_last_act=False)

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.mlp.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple "
                             f"of node features as input while this layer "
                             f"does not support bipartite message passing. "
                             f"Please try other layers such as 'SAGEConv' or "
                             f"'GraphConv' instead")

        if self.normalize:
            cache = self._cached_adj_t
            if cache is None:
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim),
                    self.improved, self.add_self_loops, self.flow, x.dtype)
                if self.cached:
                    self._cached_adj_t = edge_index
            else:
                edge_index = cache

        x = self.mlp(x)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message_and_aggregate(self, adj_t, x) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)


class GATConvWithNorm(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        negative_slope: float = 0.2,
        edge_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        norm: bool = True,
        keep_out_channels: bool = False,
        self_concat: bool = True,
        symmetric_norm=False,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.edge_dropout = edge_dropout
        self.attn_dropout = attn_dropout
        self.norm = norm
        self.keep_out_channels = keep_out_channels
        self.self_concat = self_concat
        self.symmetric_norm = symmetric_norm

        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        self.lin_src = torch.nn.Linear(in_channels, heads * out_channels, bias=True)
        self.lin_dst = torch.nn.Linear(in_channels, heads * out_channels, bias=True)

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if self.norm:
            self.norm_self = torch.nn.LayerNorm([out_channels], eps=1e-9)
            self.norm_neigh = torch.nn.LayerNorm([out_channels], eps=1e-9)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        torch.nn.init.xavier_uniform_(self.att_src)
        torch.nn.init.xavier_uniform_(self.att_dst)
        if self.norm:
            self.norm_self.reset_parameters()
            self.norm_neigh.reset_parameters()

    def forward(self, x, edge_index: Adj, edge_attr=None, size=None,
                return_attention_weights=None):
        H, C = self.heads, self.out_channels

        # Transform the input node features
        x_neigh = F.relu(self.lin_src(x)).view(-1, H, C)
        x_self = F.relu(self.lin_dst(x)).view(-1, H, C)

        if self.symmetric_norm:
            deg = edge_index.sum(dim=1).clamp(min=1)
            deg_norm = deg.pow_(-0.5)
            shp = deg_norm.shape + (1,) * (x_neigh.dim() - 1)
            x_neigh = x_neigh * torch.reshape(deg_norm, shp)

        # Compute node-level attention coefficients
        alpha_src = (x_neigh * self.att_src).sum(dim=-1)
        alpha_dst = (x_self * self.att_dst).sum(dim=-1)
        alpha = (alpha_src, alpha_dst)

        # Add self loop
        edge_index = torch_sparse.set_diag(edge_index)

        if edge_attr is None:
            edge_attr = torch.ones(edge_index.nnz(), device=edge_index.device())
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)
        out = self.propagate(edge_index, x=(x_neigh, x_self), alpha=alpha, size=size)

        # head-wide normalization
        if self.norm:
            out = self.norm_neigh(out)
            x_self = self.norm_self(x_self)

        # keep channels
        if self.keep_out_channels:
            out = out.mean(dim=1)
            x_self = x_self.mean(dim=1)
        else:
            out = out.view(-1, self.heads * self.out_channels)
            x_self = x_self.view(-1, self.heads * self.out_channels)

        if self.symmetric_norm:
            deg = edge_index.sum(dim=1).clamp(min=1)
            deg_norm = deg.pow_(0.5)
            shp = deg_norm.shape + (1,) * (out.dim() - 1)
            out = out * torch.reshape(deg_norm, shp)

        # self concat
        if self.self_concat:
            out = (out + x_self) / 2

        return out

    def edge_update(self, alpha_j, alpha_i, edge_attr, index, ptr, size_i) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)

        if self.training and self.edge_dropout > 0:
            perm = torch.randperm(alpha.shape[0], device=alpha.device)
            bound = int(alpha.shape[0] * self.edge_dropout)
            idx = perm[bound:]
            _alpha = torch.zeros_like(alpha)
            _alpha[idx] = softmax(alpha[idx], index[idx], num_nodes=size_i)
        else:
            _alpha = softmax(alpha, index, ptr, size_i)

        return F.dropout(_alpha, p=self.attn_dropout, training=self.training)

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class TransformerLayer(nn.Module):

    def __init__(
        self,
        feat_size,
        hidden_size,
        num_heads,
        attn_bias_type="add",
        norm_first=False,
        dropout=0.1,
        attn_dropout=0.1,
        activation=nn.ReLU(),
    ):
        super().__init__()

        self.norm_first = norm_first

        self.attn = BiasedMHA(
            feat_size=feat_size,
            num_heads=num_heads,
            attn_bias_type=attn_bias_type,
            attn_drop=attn_dropout,
        )
        self.ffn = nn.Sequential(
            nn.Linear(feat_size, hidden_size),
            activation,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, feat_size),
            nn.Dropout(p=dropout),
        )

        self.dropout = nn.Dropout(p=dropout)
        self.attn_layer_norm = nn.LayerNorm(feat_size)
        self.ffn_layer_norm = nn.LayerNorm(feat_size)

    def forward(self, nfeat, attn_bias=None, attn_mask=None):
        residual = nfeat
        if self.norm_first:
            nfeat = self.attn_layer_norm(nfeat)
        nfeat = self.attn(nfeat, attn_bias, attn_mask)
        nfeat = self.dropout(nfeat)
        nfeat = residual + nfeat
        if not self.norm_first:
            nfeat = self.attn_layer_norm(nfeat)
        residual = nfeat
        if self.norm_first:
            nfeat = self.ffn_layer_norm(nfeat)
        nfeat = self.ffn(nfeat)
        nfeat = residual + nfeat
        if not self.norm_first:
            nfeat = self.ffn_layer_norm(nfeat)
        nfeat = nfeat.squeeze(0)
        return nfeat


class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha
        self.Gamma = Gamma

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like, note that in this case, alpha has to be a integer.
            # It means where the peak at when initializing GPR weights.
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha * (1-alpha) ** np.arange(K+1)
            TEMP[-1] = (1-alpha) ** K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = alpha ** np.arange(K + 1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma
        else:
            raise NotImplementedError

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        if self.Init == 'SGC':
            self.temp.data[self.alpha] = 1.0
        elif self.Init == 'PPR':
            for k in range(self.K+1):
                self.temp.data[k] = self.alpha*(1-self.alpha)**k
            self.temp.data[-1] = (1-self.alpha)**self.K
        elif self.Init == 'NPPR':
            for k in range(self.K+1):
                self.temp.data[k] = self.alpha**k
            self.temp.data = self.temp.data/torch.sum(torch.abs(self.temp.data))
        elif self.Init == 'Random':
            bound = np.sqrt(3/(self.K+1))
            torch.nn.init.uniform_(self.temp,-bound,bound)
            self.temp.data = self.temp.data/torch.sum(torch.abs(self.temp.data))
        elif self.Init == 'WS':
            self.temp.data = self.Gamma

    def forward(self, x, edge_index, edge_weight=None):
        edge_index = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x * self.temp[0]
        for k in range(self.K):
            x = edge_index @ x
            gamma = self.temp[k+1]
            hidden = hidden + gamma * x
        return hidden

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class FeedForward(nn.Module):
    """
    2-layer MLP with GeLU (fancy version of ReLU) as activation
    """

    def __init__(self, dims, expansion_factor=1., dropout=0., use_single_layer=False,
                 out_dims=0, use_act=True):
        super().__init__()

        self.use_single_layer = use_single_layer
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        self.use_act = use_act

        out_dims = dims if out_dims == 0 else out_dims

        if use_single_layer:
            self.linear_0 = nn.Linear(dims, out_dims)
            self.detached_linear_0 = nn.Linear(dims, out_dims)
        else:
            self.linear_0 = nn.Linear(dims, int(expansion_factor * dims))
            self.detached_linear_0 = nn.Linear(dims, int(expansion_factor * dims))
            self.linear_1 = nn.Linear(int(expansion_factor * dims), out_dims)

        self.reset_parameters()

    def reset_parameters(self):
        self.linear_0.reset_parameters()
        if not self.use_single_layer:
            self.linear_1.reset_parameters()

    def forward(self, x):
        x = self.linear_0(x)

        if self.use_act:
            x = F.gelu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        if not self.use_single_layer:
            x = self.linear_1(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class MixerBlock(nn.Module):
    """
    out = X.T + MLP_Layernorm(X.T)     # apply token mixing
    out = out.T + MLP_Layernorm(out.T) # apply channel mixing
    """

    def __init__(self, num_neighbor, dim_feat,
                 token_expansion_factor=0.5,
                 channel_expansion_factor=4.,
                 dropout=0.,
                 ):
        super().__init__()

        self.token_layernorm = nn.LayerNorm(dim_feat)
        self.token_forward = FeedForward(num_neighbor, token_expansion_factor, dropout)

        self.channel_layernorm = nn.LayerNorm(dim_feat)
        self.channel_forward = FeedForward(dim_feat, channel_expansion_factor, dropout)

    def reset_parameters(self):
        self.token_layernorm.reset_parameters()
        self.token_forward.reset_parameters()
        self.channel_layernorm.reset_parameters()
        self.channel_forward.reset_parameters()

    def token_mixer(self, x):
        x = self.token_layernorm(x).permute(0, 2, 1)
        x = self.token_forward(x).permute(0, 2, 1)
        return x

    def channel_mixer(self, x):
        x = self.channel_layernorm(x)
        x = self.channel_forward(x)
        return x

    def forward(self, x):
        x = x + self.token_mixer(x)
        x = x + self.channel_mixer(x)
        return x


class JK_GAMLP(nn.Module):
    def __init__(self, hidden, num_layers, dropout=0.5, jk_attn_dropout=0.5):
        super(JK_GAMLP, self).__init__()
        self.num_layers = num_layers
        self.prelu = nn.PReLU()

        self.lr_jk_ref = nn.Linear(num_layers * hidden, hidden)
        self.lr_att = nn.Linear(hidden + hidden, 1)

        self.dropout = nn.Dropout(dropout)
        self.att_drop = nn.Dropout(jk_attn_dropout)
        self.act = torch.nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.lr_att.weight, gain=gain)
        nn.init.zeros_(self.lr_att.bias)
        self.lr_jk_ref.reset_parameters()

    def forward(self, input_list):
        num_node = input_list[0].shape[0]

        concat_features = torch.cat(input_list, dim=1)
        jk_ref = self.dropout(self.prelu(self.lr_jk_ref(concat_features)))
        attention_scores = [self.act(self.lr_att(torch.cat((jk_ref, x), dim=1))).view(num_node, 1) for x in
                            input_list]
        W = torch.cat(attention_scores, dim=1)
        W = F.softmax(W, 1)

        right_1 = torch.mul(input_list[0], self.att_drop(
            W[:, 0].view(num_node, 1)))
        for i in range(1, self.num_layers):
            right_1 = right_1 + \
                torch.mul(input_list[i], self.att_drop(
                    W[:, i].view(num_node, 1)))
        return right_1


# class DirGCNConv(torch.nn.Module):
#     def __init__(self, input_dim, output_dim,
#                  dir_alpha=0.5,  # 0 (only in-edges), 1 (only out-edges), 0.5 (both)
#                  **kwargs
#                  ):
#         super(DirGCNConv, self).__init__()
#
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#
#         self.conv_src_to_dst = GCNConv(input_dim, output_dim)
#         self.conv_dst_to_src = GCNConv(input_dim, output_dim)
#         self.alpha = dir_alpha
#
#     def forward(self, x, edge_index):
#         if isinstance(edge_index, Tensor):
#             edge_index = to_torch_sparse_tensor(edge_index)
#
#         return (
#             (1 - self.alpha) * self.conv_src_to_dst(x, edge_index)
#             + self.alpha * self.conv_dst_to_src(x, edge_index.t())
#         )


class GCN2ATPConv(torch.nn.Module):
    def __init__(self, channels: int, r: Tensor, alpha: float, theta: float = None,
                 layer: int = None, shared_weights: bool = True, **kwargs):
        super().__init__()

        self.channels = channels
        self.alpha = alpha
        self.beta = 1.
        if theta is not None or layer is not None:
            assert theta is not None and layer is not None
            self.beta = math.log(theta / layer + 1)
        self.adj_norm: Optional[Adj] = None
        self.r = r

        self.weight1 = Parameter(torch.empty(channels, channels))

        if shared_weights:
            self.register_parameter('weight2', None)
        else:
            self.weight2 = Parameter(torch.empty(channels, channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)

    def forward(self, x: Tensor, x_0: Tensor, edge_index: Adj) -> Tensor:
        if self.adj_norm is None:
            self.adj_norm = adj_norm(edge_index, norm='attn', exponent=self.r)

        x = self.adj_norm @ x

        x.mul_(1 - self.alpha)
        x_0 = self.alpha * x_0[:x.size(0)]

        if self.weight2 is None:
            out = x.add_(x_0)
            out = torch.addmm(out, out, self.weight1, beta=1. - self.beta,
                              alpha=self.beta)
        else:
            out = torch.addmm(x, x, self.weight1, beta=1. - self.beta,
                              alpha=self.beta)
            out = out + torch.addmm(x_0, x_0, self.weight2,
                                    beta=1. - self.beta, alpha=self.beta)
        return out


class GCNATPConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, r, **kwargs):
        super(GCNATPConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.r = r

        self.lin = Linear(input_dim, output_dim)
        self.adj_norm: Optional[Adj] = None

    def forward(self, x, edge_index):
        if isinstance(edge_index, Tensor):
            edge_index = to_torch_sparse_tensor(edge_index)

        if self.adj_norm is None:
            self.adj_norm = adj_norm(edge_index, norm='attn', exponent=self.r)

        return self.lin(self.adj_norm @ x)


class GCNNormConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, deg_norm='rw', **kwargs):
        super(GCNNormConv, self).__init__()

        assert deg_norm in ['rw', 'src']
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.deg_norm = deg_norm

        self.lin = Linear(input_dim, output_dim)
        self.adj_norm: Optional[Adj] = None

    def forward(self, x, edge_index):
        if isinstance(edge_index, Tensor):
            edge_index = to_torch_sparse_tensor(edge_index)

        if self.adj_norm is None:
            self.adj_norm = adj_norm(edge_index, norm=self.deg_norm)

        return self.lin(self.adj_norm @ x)


class DPAConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, inv_exponent=0.5):
        super(DPAConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.exponent = inv_exponent

        self.lins = torch.nn.ModuleList([
            Linear(input_dim, output_dim) for _ in range(6)
        ])

        self.deg_list = None

    def forward(self, x, edge_index,):
        adj = edge_index
        adj_t = adj.t()
        adj_ = torch_sparse.fill_diag(adj, 1.0)
        adj_t_ = torch_sparse.fill_diag(adj_t, 1.0)

        if self.deg_list is None:

            def get_deg_inv(_adj):
                _in_deg = _adj.sum(dim=0)
                _in_deg_inv = _in_deg.pow(self.exponent - 1)
                _in_deg_inv.masked_fill_(_in_deg_inv == float("inf"), 0.0)

                _out_deg = _adj.sum(dim=1)
                _out_deg_inv = _out_deg.pow(-self.exponent)
                _out_deg_inv.masked_fill_(_out_deg_inv == float("inf"), 0.0)

                return _in_deg_inv.to(x.device), _out_deg_inv.to(x.device)

            self.deg_list = [
                get_deg_inv(adj_),
                get_deg_inv(adj_t_),
                get_deg_inv(adj @ adj),
                get_deg_inv(adj @ adj_t),
                get_deg_inv(adj_t @ adj),
                get_deg_inv(adj_t @ adj_t),
                # get_deg_inv(adj.cpu() @ adj.cpu()),
                # get_deg_inv(adj.cpu() @ adj_t.cpu()),
                # get_deg_inv(adj_t.cpu() @ adj.cpu()),
                # get_deg_inv(adj_t.cpu() @ adj_t.cpu()),
            ]

        ## calculate Directed Patterns
        total = 0

        in_deg_inv, out_deg_inv = self.deg_list[0]
        _x = (out_deg_inv.view(-1, 1) * adj_ * in_deg_inv.view(1, -1)) @ x
        total += self.lins[0](_x)

        in_deg_inv, out_deg_inv = self.deg_list[1]
        _x = (out_deg_inv.view(-1, 1) * adj_t_ * in_deg_inv.view(1, -1)) @ x
        total += self.lins[1](_x)

        in_deg_inv, out_deg_inv = self.deg_list[2]
        _x = (out_deg_inv.view(-1, 1) * adj) @ (adj * in_deg_inv.view(1, -1) @ x)
        total += self.lins[2](_x)

        in_deg_inv, out_deg_inv = self.deg_list[3]
        _x = (out_deg_inv.view(-1, 1) * adj) @ (adj_t * in_deg_inv.view(1, -1) @ x)
        total += self.lins[3](_x)

        in_deg_inv, out_deg_inv = self.deg_list[4]
        _x = (out_deg_inv.view(-1, 1) * adj_t) @ (adj * in_deg_inv.view(1, -1) @ x)
        total += self.lins[4](_x)

        in_deg_inv, out_deg_inv = self.deg_list[5]
        _x = (out_deg_inv.view(-1, 1) * adj_t) @ (adj_t * in_deg_inv.view(1, -1) @ x)
        total += self.lins[5](_x)

        return total
        # return total / len(self.lins)


class FaberConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dir_alpha=0.5,
                 k_plus=1, exponent=-0.25, weight_penalty='exp',
                 zero_order=False):
        super(FaberConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.K_plus = k_plus
        self.exponent = exponent
        self.weight_penalty = weight_penalty
        self.zero_order = zero_order

        if self.zero_order:
            # Zero Order Lins
            # Source to destination
            self.lin_src_to_dst_zero = Linear(input_dim, output_dim)
            # Source to destination
            self.lin_dst_to_src_zero = Linear(input_dim, output_dim)

        # Lins for positive powers:
        self.lins_src_to_dst = torch.nn.ModuleList([
            Linear(input_dim, output_dim) for _ in range(k_plus)
        ])

        self.lins_dst_to_src = torch.nn.ModuleList([
            Linear(input_dim, output_dim) for _ in range(k_plus)
        ])

        self.alpha = dir_alpha
        self.adj_norm, self.adj_t_norm = None, None

    def forward(self, x, edge_index,):
        if isinstance(edge_index, Tensor):
            edge_index = to_torch_sparse_tensor(edge_index)

        if self.adj_norm is None:
            self.adj_norm = adj_norm(edge_index, norm="dir", exponent=self.exponent)
            self.adj_t_norm = adj_norm(edge_index.t(), norm="dir", exponent=self.exponent)

        y = self.adj_norm @ x
        y_t = self.adj_t_norm @ x
        sum_src_to_dst = self.lins_src_to_dst[0](y)
        sum_dst_to_src = self.lins_dst_to_src[0](y_t)
        if self.zero_order:
            sum_src_to_dst = sum_src_to_dst + self.lin_src_to_dst_zero(x)
            sum_dst_to_src = sum_dst_to_src + self.lin_dst_to_src_zero(x)

        if self.K_plus > 1:
            if self.weight_penalty == 'exp':
                for i in range(1, self.K_plus):
                    y = self.adj_norm @ y
                    y_t = self.adj_t_norm @ y

                    sum_src_to_dst = sum_src_to_dst + self.lins_src_to_dst[i](y) / (2 ** i)
                    sum_dst_to_src = sum_dst_to_src + self.lins_dst_to_src[i](y_t) / (2 ** i)

            elif self.weight_penalty == 'lin':
                for i in range(1, self.K_plus):
                    y = self.adj_norm @ y
                    y_t = self.adj_t_norm @ y

                    sum_src_to_dst = sum_src_to_dst + self.lins_src_to_dst[i](y) / i
                    sum_dst_to_src = sum_dst_to_src + self.lins_dst_to_src[i](y_t) / i
            else:
                for i in range(1, self.K_plus):
                    y = self.adj_norm @ y
                    y_t = self.adj_t_norm @ y

                    sum_src_to_dst = sum_src_to_dst + self.lins_src_to_dst[i](y)
                    sum_dst_to_src = sum_dst_to_src + self.lins_dst_to_src[i](y_t)

        total = self.alpha * sum_src_to_dst + (1 - self.alpha) * sum_dst_to_src

        return total


class DirGCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dir_alpha, **kwargs):
        super(DirGCNConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lin_src_to_dst = Linear(input_dim, output_dim)
        self.lin_dst_to_src = Linear(input_dim, output_dim)
        self.alpha = dir_alpha
        self.adj_norm = None
        self.adj_t_norm = None

    def forward(self, x, edge_index):
        if isinstance(edge_index, Tensor):
            edge_index = to_torch_sparse_tensor(edge_index)

        if self.adj_norm is None:
            self.adj_norm = adj_norm(edge_index, norm="dir")
            self.adj_t_norm = adj_norm(edge_index.t(), norm="dir")

        if self.alpha == 0:
            return self.lin_src_to_dst(self.adj_norm @ x)
        elif self.alpha == 1:
            return self.lin_dst_to_src(self.adj_t_norm @ x)
        else:
            return (
                    (1 - self.alpha) * self.lin_src_to_dst(self.adj_norm @ x)
                    + self.alpha * self.lin_dst_to_src(self.adj_t_norm @ x)
            )


class GCNConvWithNorm(MessagePassing):
    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
        adj_norm: str = "sym",
        bias: bool = True,
        **kwargs,
    ):
        aggr = kwargs.pop('aggr', 'add')
        super().__init__(aggr=aggr)

        # if add_self_loops is None:
        #     add_self_loops = True

        if add_self_loops and not adj_norm:
            raise ValueError(f"'{self.__class__.__name__}' does not support "
                             f"adding self-loops to the graph when no "
                             f"on-the-fly normalization is applied")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.adj_norm = adj_norm

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = Linear(in_channels, out_channels, bias=False)

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple "
                             f"of node features as input while this layer "
                             f"does not support bipartite message passing. "
                             f"Please try other layers such as 'SAGEConv' or "
                             f"'GraphConv' instead")

        if self.adj_norm:
            # if isinstance(edge_index, Tensor):
            #     cache = self._cached_edge_index
            #     if cache is None:
            #         edge_index, edge_weight = gcn_norm(  # yapf: disable
            #             edge_index, edge_weight, x.size(self.node_dim),
            #             self.improved, self.add_self_loops, self.flow, x.dtype)
            #         if self.cached:
            #             self._cached_edge_index = (edge_index, edge_weight)
            #     else:
            #         edge_index, edge_weight = cache[0], cache[1]

            # el
            if isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = adj_norm(edge_index, norm=self.adj_norm, add_self_loop=self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
            else:
                raise NotImplementedError

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

class BernConv(MessagePassing):

    def __init__(self, in_channels, out_channels, K,bias=True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(BernConv, self).__init__(**kwargs)
        assert K > 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.K=K

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(-2) + self.weight.size(-1)))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.fill_(0)

    def forward(self,x,edge_index,coe,edge_weight=None):

        TEMP=F.relu(coe)

        #L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index,norm='sym')

        #2I-L
        edge_index2, norm2 = add_self_loops(edge_index1,-norm1,fill_value=2.,num_nodes=x.size(self.node_dim))

        tmp=[]
        tmp.append(x)
        for i in range(self.K):
            x=self.propagate(edge_index2,x=x,norm=norm2,size=None)
            tmp.append(x)

        out=(comb(self.K,0)/(2**self.K))*TEMP[0]*tmp[self.K]

        for i in range(self.K):
                x=tmp[self.K-i-1]
                x=self.propagate(edge_index1,x=x,norm=norm1,size=None)
                for j in range(i):
                        x=self.propagate(edge_index1,x=x,norm=norm1,size=None)

                out=out+(comb(self.K,i+1)/(2**self.K))*TEMP[i+1]*x

        out=out@self.weight
        if self.bias is not None:
                out+=self.bias
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.weight.size(0), self.normalization)

class DirSAGEConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim,
                 dir_alpha=0.5,  # 0 (only in-edges), 1 (only out-edges), 0.5 (both)
                 **kwargs
                 ):
        super(DirSAGEConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_src_to_dst = SAGEConv(input_dim, output_dim, flow="source_to_target", root_weight=False, **kwargs)
        self.conv_dst_to_src = SAGEConv(input_dim, output_dim, flow="target_to_source", root_weight=False, **kwargs)
        self.lin_self = Linear(input_dim, output_dim)
        self.alpha = dir_alpha

    def forward(self, x, edge_index):
        return (
            self.lin_self(x)
            + (1 - self.alpha) * self.conv_src_to_dst(x, edge_index)
            + self.alpha * self.conv_dst_to_src(x, edge_index)
        )


class DirGATConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, heads,
                 dir_alpha=0.5,  # 0 (only in-edges), 1 (only out-edges), 0.5 (both)
                 **kwargs
                 ):
        super(DirGATConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_src_to_dst = GATConv(input_dim, output_dim, heads=heads, **kwargs)
        self.conv_dst_to_src = GATConv(input_dim, output_dim, heads=heads, **kwargs)
        self.alpha = dir_alpha

    def forward(self, x, edge_index):
        if isinstance(edge_index, Tensor):
            edge_index = to_torch_sparse_tensor(edge_index)

        if self.alpha == 0:
            return self.conv_src_to_dst(x, edge_index)
        elif self.alpha == 1:
            return self.alpha * self.conv_dst_to_src(x, edge_index.t())
        else:
            return (
                (1 - self.alpha) * self.conv_src_to_dst(x, edge_index)
                + self.alpha * self.conv_dst_to_src(x, edge_index.t())
            )


class AttentionChannelMixing(nn.Module):
    def __init__(
        self,
        in_dims,
        out_dim,
        layer_norm=False,
    ):
        super(AttentionChannelMixing, self).__init__()
        self.num_channels = len(in_dims)
        self.layer_norm = layer_norm

        self.channel_weights = nn.ParameterList()
        self.att_vectors = nn.ParameterList()
        self.layer_norms = nn.ModuleList()
        for i in range(self.num_channels):
            self.channel_weights.append(
                Parameter(torch.FloatTensor(in_dims[i], out_dim)))
            self.att_vectors.append(
                Parameter(torch.FloatTensor(out_dim, 1)))
            self.layer_norms.append(nn.LayerNorm(out_dim))

        self.att_trans = Parameter(torch.FloatTensor(self.num_channels, self.num_channels))

        self.reset_parameters()

    def reset_parameters(self):
        std_channel = 1.0 / math.sqrt(self.channel_weights[0].size(1))
        std_att_vec = 1.0 / math.sqrt(self.att_vectors[0].size(1))
        std_att_trans = 1.0 / math.sqrt(self.att_trans.size(1))

        for i in range(self.num_channels):
            self.channel_weights[i].data.uniform_(-std_channel, std_channel)
            self.att_vectors[i].data.uniform_(-std_att_vec, std_att_trans)
            self.layer_norms[i].reset_parameters()

        self.att_trans.data.uniform_(-std_att_trans, std_att_trans)

    def attention(self, embeddings):
        assert len(embeddings) == self.num_channels
        if self.layer_norm:
            embeddings = [self.layer_norms[i](emb) for i, emb in enumerate(embeddings)]
        logits = (
            torch.mm(
                torch.sigmoid(torch.cat(
                    [torch.mm(emb, self.att_vectors[i]) for i, emb in enumerate(embeddings)], dim=1,
                )),
                self.att_trans,
            )
            / self.num_channels
        )
        att = torch.softmax(logits, 1)
        return [att[:, i][:, None] for i in range(self.num_channels)]

    def forward(self, xs):
        xs = [F.relu(torch.mm(x, self.channel_weights[i])) for i, x in enumerate(xs)]
        atts = self.attention(xs)
        gating_weights = torch.stack(atts, dim=0).squeeze()
        return len(xs) * sum([att * x for att, x in zip(atts, xs)]), gating_weights


class ACMGCNConv(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        num_nodes,
        model_type,
        variant=False,
        structure_info=False,
        layer_norm=False,
        active_filters=[True, True, True], # the activeness of [low pass, high pass, I]
    ):
        super(ACMGCNConv, self).__init__()
        (
            self.in_features,
            self.out_features,
            self.model_type,
            self.structure_info,
            self.variant,
            self.has_layer_norm,
            self.active_filters,
        ) = (
            in_features,
            out_features,
            model_type,
            structure_info,
            variant,
            layer_norm,
            active_filters,
        )
        self.att_low, self.att_high, self.att_mlp = 0, 0, 0
        self.weight_low, self.weight_high, self.weight_mlp = (
            Parameter(torch.FloatTensor(in_features, out_features)),
            Parameter(torch.FloatTensor(in_features, out_features)),
            Parameter(torch.FloatTensor(in_features, out_features)),
        )
        self.att_vec_low, self.att_vec_high, self.att_vec_mlp = (
            Parameter(torch.FloatTensor(1 * out_features, 1)),
            Parameter(torch.FloatTensor(1 * out_features, 1)),
            Parameter(torch.FloatTensor(1 * out_features, 1)),
        )
        self.layer_norm_low, self.layer_norm_high, self.layer_norm_mlp = (
            nn.LayerNorm(out_features),
            nn.LayerNorm(out_features),
            nn.LayerNorm(out_features),
        )
        self.layer_norm_struc_low, self.layer_norm_struc_high = nn.LayerNorm(
            out_features
        ), nn.LayerNorm(out_features)
        self.att_struc_low = Parameter(
            torch.FloatTensor(1 * out_features, 1)
        )
        self.struc_low = Parameter(torch.FloatTensor(num_nodes, out_features))
        if self.model_type == "acmgcn":
            self.att_vec = Parameter(torch.FloatTensor(3, 3))
        else:
            self.att_vec = Parameter(torch.FloatTensor(4, 4))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight_mlp.size(1))
        std_att = 1.0 / math.sqrt(self.att_vec_mlp.size(1))
        std_att_vec = 1.0 / math.sqrt(self.att_vec.size(1))

        self.weight_low.data.uniform_(-stdv, stdv)
        self.weight_high.data.uniform_(-stdv, stdv)
        self.weight_mlp.data.uniform_(-stdv, stdv)
        self.struc_low.data.uniform_(-stdv, stdv)

        self.att_vec_high.data.uniform_(-std_att, std_att)
        self.att_vec_low.data.uniform_(-std_att, std_att)
        self.att_vec_mlp.data.uniform_(-std_att, std_att)
        self.att_struc_low.data.uniform_(-std_att, std_att)

        self.att_vec.data.uniform_(-std_att_vec, std_att_vec)

        self.layer_norm_low.reset_parameters()
        self.layer_norm_high.reset_parameters()
        self.layer_norm_mlp.reset_parameters()
        self.layer_norm_struc_low.reset_parameters()
        self.layer_norm_struc_high.reset_parameters()

    def attention3(self, output_low, output_high, output_mlp):
        T = 3
        if self.has_layer_norm:
            output_low, output_high, output_mlp = (
                self.layer_norm_low(output_low),
                self.layer_norm_high(output_high),
                self.layer_norm_mlp(output_mlp),
            )
        logits = (
            torch.mm(
                torch.sigmoid(
                    torch.cat(
                        [
                            torch.mm((output_low), self.att_vec_low),
                            torch.mm((output_high), self.att_vec_high),
                            torch.mm((output_mlp), self.att_vec_mlp),
                        ],
                        1,
                    )
                ),
                self.att_vec,
            )
            / T
        )
        att = torch.softmax(logits, 1)
        return att[:, 0][:, None], att[:, 1][:, None], att[:, 2][:, None]

    def attention4(self, output_low, output_high, output_mlp, struc_low):
        T = 4
        if self.has_layer_norm:
            feature_concat = torch.cat(
                [
                    torch.mm(self.layer_norm_low(output_low), self.att_vec_low),
                    torch.mm(self.layer_norm_high(output_high), self.att_vec_high),
                    torch.mm(self.layer_norm_mlp(output_mlp), self.att_vec_mlp),
                    torch.mm(self.layer_norm_struc_low(struc_low), self.att_struc_low),
                ],
                1,
            )
        else:
            feature_concat = torch.cat(
                [
                    torch.mm(output_low, self.att_vec_low),
                    torch.mm(output_high, self.att_vec_high),
                    torch.mm(output_mlp, self.att_vec_mlp),
                    torch.mm(struc_low, self.att_struc_low),
                ],
                1,
            )

        logits = torch.mm(torch.sigmoid(feature_concat), self.att_vec) / T
        att = torch.softmax(logits, 1)
        return (
            att[:, 0][:, None],
            att[:, 1][:, None],
            att[:, 2][:, None],
            att[:, 3][:, None],
        )

    def forward(self, input, adj_low, adj_high, adj_low_unnormalized):
        if self.variant:
            output_low = adj_low @ F.relu(torch.mm(input, self.weight_low))
            output_high = adj_high @ F.relu(torch.mm(input, self.weight_high))
            output_mlp = F.relu(torch.mm(input, self.weight_mlp))
        else:
            output_low = F.relu(adj_low @ torch.mm(input, self.weight_low))
            output_high = F.relu(adj_high @ torch.mm(input, self.weight_high))
            output_mlp = F.relu(torch.mm(input, self.weight_mlp))

        if self.model_type == "acmgcn":
            self.att_low, self.att_high, self.att_mlp = self.attention3(
                output_low, output_high, output_mlp
            )
            return 3 * (
                self.att_low * output_low * self.active_filters[0]
                + self.att_high * output_high * self.active_filters[1]
                + self.att_mlp * output_mlp * self.active_filters[2]
            )
        else:
            output_struc_low = F.relu(
                torch.mm(adj_low_unnormalized.to_torch_sparse_coo_tensor(), self.struc_low)
            )
            (
                self.att_low,
                self.att_high,
                self.att_mlp,
                self.att_struc_vec_low,
            ) = self.attention4(
                output_low, output_high, output_mlp, output_struc_low
            )
            return 1 * (
                self.att_low * output_low * self.active_filters[0]
                + self.att_high * output_high * self.active_filters[1]
                + self.att_mlp * output_mlp * self.active_filters[2]
                + self.att_struc_vec_low * output_struc_low
            )

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class PolyGlobalAttn(torch.nn.Module):
    """ Global attention layer for Polynormer
    """
    def __init__(self, hidden_channels, heads, num_layers, beta, dropout, qk_shared=True):
        super(PolyGlobalAttn, self).__init__()

        self.hidden_channels = hidden_channels
        self.heads = heads
        self.num_layers = num_layers
        self.beta = beta
        self.dropout = dropout
        self.qk_shared = qk_shared

        if self.beta < 0:
            self.betas = torch.nn.Parameter(torch.zeros(num_layers, heads*hidden_channels))
        else:
            self.betas = torch.nn.Parameter(torch.ones(num_layers, heads*hidden_channels)*self.beta)

        self.h_lins = torch.nn.ModuleList()
        if not self.qk_shared:
            self.q_lins = torch.nn.ModuleList()
        self.k_lins = torch.nn.ModuleList()
        self.v_lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        for i in range(num_layers):
            self.h_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            if not self.qk_shared:
                self.q_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.k_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.v_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.lns.append(torch.nn.LayerNorm(heads*hidden_channels))
        self.lin_out = torch.nn.Linear(heads*hidden_channels, heads*hidden_channels)

    def reset_parameters(self):
        for h_lin in self.h_lins:
            h_lin.reset_parameters()
        if not self.qk_shared:
            for q_lin in self.q_lins:
                q_lin.reset_parameters()
        for k_lin in self.k_lins:
            k_lin.reset_parameters()
        for v_lin in self.v_lins:
            v_lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        if self.beta < 0:
            torch.nn.init.xavier_normal_(self.betas)
        else:
            torch.nn.init.constant_(self.betas, self.beta)
        self.lin_out.reset_parameters()

    def forward(self, x):
        seq_len, _ = x.size()
        for i in range(self.num_layers):
            h = self.h_lins[i](x)
            k = F.sigmoid(self.k_lins[i](x)).view(seq_len, self.hidden_channels, self.heads)
            if self.qk_shared:
                q = k
            else:
                q = F.sigmoid(self.q_lins[i](x)).view(seq_len, self.hidden_channels, self.heads)
            v = self.v_lins[i](x).view(seq_len, self.hidden_channels, self.heads)

            # numerator
            kv = torch.einsum('ndh, nmh -> dmh', k, v)
            num = torch.einsum('ndh, dmh -> nmh', q, kv)

            # denominator
            k_sum = torch.einsum('ndh -> dh', k)
            den = torch.einsum('ndh, dh -> nh', q, k_sum).unsqueeze(1)

            # linear global attention based on kernel trick
            if self.beta < 0:
                beta = F.sigmoid(self.betas[i]).unsqueeze(0)
            else:
                beta = self.betas[i].unsqueeze(0)
            x = (num/den).reshape(seq_len, -1)
            x = self.lns[i](x) * (h+beta)
            x = F.relu(self.lin_out(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class G2(nn.Module):
    def __init__(self, conv, p=2., conv_type='SAGE', activation=nn.ReLU()):
        super(G2, self).__init__()
        self.conv = conv
        self.p = p
        self.activation = activation
        self.conv_type = conv_type

    def forward(self, X, edge_index):
        n_nodes = X.size(0)
        if self.conv_type == 'GAT':
            X = F.elu(self.conv(X, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
        else:
            X = self.activation(self.conv(X, edge_index))
        if isinstance(edge_index, SparseTensor):
            row, col, _ = edge_index.coo()
            edge_index = torch.stack([row, col], dim=0)
        gg = torch.tanh(scatter((torch.abs(X[edge_index[0]] - X[edge_index[1]]) ** self.p).squeeze(-1),
                                 edge_index[0], 0, dim_size=X.size(0), reduce='mean'))
        return gg


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp, edge_index, edge_attr):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # Note by Haotao:
        # self._batch_index: shape=(N_batch). The re-order indices from 0 to N_batch-1.
        # inp_exp: shape=inp.shape. The input Tensor re-ordered by self._batch_index along the batch dimension.
        # self._part_sizes: shape=(N_experts), sum=N_batch. self._part_sizes[i] is the number of samples routed towards expert[i].
        # return value: list [Tensor with shape[0]=self._part_sizes[i] for i in range(N_experts)]

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        edge_index_exp = edge_index[:,self._batch_index]
        edge_attr_exp = edge_attr[self._batch_index]
        return torch.split(inp_exp, self._part_sizes, dim=0), torch.split(edge_index_exp, self._part_sizes, dim=1), torch.split(edge_attr_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class GMoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, output_size, num_experts, noisy_gating=True, k=4, coef=1e-2, sage=False):
        super(GMoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.k = k
        self.loss_coef = coef
        # instantiate experts
        conv = SAGEConv if sage else GCNConv
        self.experts = nn.ModuleList([conv(input_size, output_size, normalize=False) for i in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, edge_index, edge_attr=None):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= self.loss_coef

        expert_outputs = []
        for i in range(self.num_experts):
            expert_i_output = self.experts[i](x, edge_index, edge_attr)
            expert_outputs.append(expert_i_output)
        expert_outputs = torch.stack(expert_outputs, dim=1) # shape=[num_nodes, num_experts, d_feature]

        # gates: shape=[num_nodes, num_experts]
        y = gates.unsqueeze(dim=-1) * expert_outputs
        y = y.mean(dim=1)

        return y, loss
