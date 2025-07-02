from typing import Callable, List, Optional, Union

import math
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn import Linear, ModuleList, LayerNorm, Parameter, Softmax, Dropout
from class_resolver.contrib.torch import activation_resolver
from torch.nn.init import calculate_gain, xavier_uniform_
from torch_geometric.nn import GCN2Conv, GPSConv, GINEConv, GatedGraphConv, APPNP, MixHopConv

from torch_geometric.nn.conv import (
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    PNAConv,
    SAGEConv,
)
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_geometric.typing import Adj

import torch
from torch_sparse import matmul, SparseTensor, fill_diag, remove_diag, mul
from torch_sparse import sum as sparsesum

from components.layer import EgoGraphPooling, GenLinear, MLPGCNConv, BaseMLP, GATConvWithNorm, GCNConvWithNorm, BernConv, GPR_prop, \
    JK_GAMLP, DirGCNConv, DirSAGEConv, DirGATConv, ACMGCNConv, PolyGlobalAttn, G2, GMoE, FaberConv, DPAConv, \
    GCNNormConv, GCNATPConv, GCN2ATPConv
from utils.utils import dropout_edge, adj_norm, Dict


class GenGNN(torch.nn.Module):

    def __init__(
            self,
            name: str,
            in_dim: int,
            hidden_dim: int,
            conv_layers: int,
            out_dim: Optional[int] = None,
            subg_pool: List[int] = None,
            pool_type: str = 'mean',
            init_layers: int = 0,
            init_dropout: float = 0.0,
            dropout: float = 0.0,
            dropedge: int = 0,
            act: Union[str, Callable, None] = "relu",
            norm: str = None,  # layer, batch
            out_norm: bool = True,
            residual: Optional[str] = None,  # sum, incep, cat
            jk: Optional[str] = None,  # cat, last, max, lstm
            act_first: bool = False,
            incep_alpha: float = 0.5,
            **kwargs,
    ):
        super().__init__()
        # jk = None if conv_layers <= 1 else jk
        self.name = name.upper()

        self.in_channels = in_dim
        self.hidden_dim = hidden_dim
        self.out_channels = out_dim
        self.conv_layers = conv_layers

        self.init_dropout = init_dropout
        self.dropout = dropout
        self.dropedge = dropedge
        self.act = activation_resolver.make(act)
        self.norm = norm
        self.out_norm = out_norm
        assert residual in ['sum', 'incep', 'cat', None]
        self.residual = residual
        self.jk_mode = jk
        self.act_first = act_first
        self.subg_pool = subg_pool
        self.alpha = incep_alpha

        enc_gnn = ['GCNII', 'GCNII-ATP', 'PNA', 'GPS']
        self.has_classifier = (jk is not None
                               or subg_pool is not None
                               or self.name in enc_gnn
                               or self.name == 'MIXHOP')
        if self.name in enc_gnn: init_layers = max(1, init_layers)

        # init conv_layers & SAGE-like linear residual layer
        self.convs = ModuleList()
        if self.residual == 'cat':
            self.res_lins = nn.ModuleList()

        if 'bern' in self.name.lower():
            self.coe = Parameter(torch.Tensor(kwargs['K']+1))
            self.coe.data.fill_(1)

        # first
        if 'GAT' in self.name and conv_layers == 1:
            kwargs['last_layer'] = True
        # if 'GCNII' in self.name:
        #     kwargs['layer'] = 1
        if init_layers > 0:
            has_conv = self.has_classifier or conv_layers >= 1
            self.mlp = BaseMLP(in_dim,
                               hidden_dim,
                               hidden_dim if has_conv else out_dim,
                               init_layers,
                               act=act,
                               dropout=dropout,
                               norm=norm,
                               keep_last_act=has_conv)

            if conv_layers == 1 and not self.has_classifier:
                self.convs.append(
                    self.init_conv(hidden_dim, out_dim, **kwargs))
                if hasattr(self, 'res_lins'):
                    self.res_lins.append(nn.Linear(hidden_dim, out_dim))
            else:
                self.convs.append(
                    self.init_conv(hidden_dim, hidden_dim, **kwargs))
                if hasattr(self, 'res_lins'):
                    self.res_lins.append(nn.Linear(hidden_dim, hidden_dim))
        else:
            if conv_layers == 1 and not self.has_classifier:
                self.convs.append(
                    self.init_conv(in_dim, out_dim, **kwargs))
                if hasattr(self, 'res_lins'):
                    self.res_lins.append(nn.Linear(in_dim, out_dim))
            else:
                self.convs.append(
                    self.init_conv(in_dim, hidden_dim, **kwargs))
                if hasattr(self, 'res_lins'):
                    self.res_lins.append(nn.Linear(in_dim, hidden_dim))

        # hidden
        for l in range(2, conv_layers):
            # if 'GCNII' in self.name:
            #     kwargs['layer'] = l
            if 'GAT' in self.name and hidden_dim % kwargs.get('heads', 1) != 0:
                adjusted_hidden_dim = hidden_dim // kwargs.get('heads', 1)
                adjusted_hidden_dim *= kwargs.get('heads', 1)
                print(f'WARNING: {self.name} hidden_dim {hidden_dim} is not divisible by heads {kwargs.get("heads", 1)}, choosing to use {adjusted_hidden_dim} instead.')
                hidden_dim = adjusted_hidden_dim
            self.convs.append(
                self.init_conv(hidden_dim, hidden_dim, **kwargs))
            if hasattr(self, 'res_lins'):
                self.res_lins.append(nn.Linear(hidden_dim, hidden_dim))

        # last
        if conv_layers > 1:
            if 'GAT' in self.name:
                kwargs['last_layer'] = True
            # elif 'GCNII' in self.name:
            #     kwargs['layer'] = conv_layers
            if self.has_classifier:
                if hidden_dim % kwargs.get('heads', 1) != 0:
                    adjusted_hidden_dim = hidden_dim // kwargs.get('heads', 1)
                    adjusted_hidden_dim *= kwargs.get('heads', 1)
                    print(f'WARNING: {self.name} hidden_dim {hidden_dim} is not divisible by heads {kwargs.get("heads", 1)}, choosing to use {adjusted_hidden_dim} instead.')
                    hidden_dim = adjusted_hidden_dim

                self.convs.append(
                    self.init_conv(hidden_dim, hidden_dim, **kwargs))
                if hasattr(self, 'res_lins'):
                    self.res_lins.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                self.convs.append(
                    self.init_conv(hidden_dim, out_dim, **kwargs))
                if hasattr(self, 'res_lins'):
                    self.res_lins.append(nn.Linear(hidden_dim, out_dim))

        # normalizations
        self.norms = None
        if conv_layers > 0:
            self.norms, eps = ModuleList(), 1e-9
            num_norm = conv_layers if self.has_classifier else conv_layers - 1
            for _ in range(num_norm):
                if norm == 'layer' and hidden_dim > 1:
                    self.norms.append(nn.LayerNorm(hidden_dim, eps=eps))
                elif norm == 'batch':
                    self.norms.append(nn.BatchNorm1d(hidden_dim, eps=eps))

        # multi-scale update
        num_layers = conv_layers + (1 if hasattr(self, 'mlp') else 0)
        if jk in ['max', 'cat', 'lstm']:
            self.jk = JumpingKnowledge(jk, hidden_dim, num_layers)
        elif jk == 'attn':
            self.jk = JK_GAMLP(hidden_dim, num_layers, dropout)
        if self.subg_pool:
            self.pool = EgoGraphPooling(pool_type, num_groups=1)

        # output classifier
        if self.has_classifier:
            if jk == 'cat':
                in_dim = num_layers * hidden_dim
            else:
                in_dim = hidden_dim
            if subg_pool:
                in_dim += len(subg_pool) * hidden_dim
            self.lin = Linear(in_dim, self.out_channels)

        # self.reset_parameters()

    def init_conv(self, in_dim: int, out_dim: int,
                  **kwargs):
        raise NotImplementedError

    def reset_parameters(self):
        if hasattr(self, 'mlp'):
            self.mlp.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()
        if hasattr(self, 'jk'):
            self.jk.reset_parameters()
        if hasattr(self, 'lin'):
            self.lin.reset_parameters()
        if hasattr(self, 'res_lins'):
            for lin in self.res_lins:
                lin.reset_parameters()
        if hasattr(self, 'coe'):
            self.coe.data.fill_(1)

    def forward(self, x: Tensor, edge_index: Adj = None, ego_ptr=None, *args, **kwargs) -> Tensor:
        x = F.dropout(x, p=self.init_dropout, training=self.training)

        xs = []
        x_0 = x_pre = None
        if hasattr(self, 'mlp'):
            x = self.mlp(x)
            x_0 = x_pre = x
            xs.append(x)

        edge_index = dropout_edge(edge_index, p=self.dropedge, training=self.training)

        for i, conv in enumerate(self.convs):
            _x = self.res_lins[i](x) if self.residual == 'cat' else 0

            if 'GCNII' in self.name:
                x = conv(x, x_0, edge_index, *args, **kwargs)
            elif 'bern' in self.name.lower():
                x = conv(x, edge_index, self.coe, *args, **kwargs)
            else:
                x = conv(x, edge_index, *args, **kwargs)

            x += _x

            if i < self.conv_layers - 1 or self.has_classifier:
                if self.norms:
                    x = self.norms[i](x)
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.out_norm:
                    x = F.normalize(x, p=2, dim=1)

                if x_0 is not None and x_pre is not None:
                    if self.residual == 'sum':
                        x = x + x_pre
                        x_pre = x
                    elif self.residual == 'incep':
                        x = (1 - self.alpha) * x + self.alpha * x_0
                else:
                    x_0 = x_pre = x
                if self.has_classifier:
                    xs.append(x)

        # multi-scale update
        if hasattr(self, 'jk'):
            x = self.jk(xs)
        if ego_ptr is not None:
            x = x[ego_ptr]
            if self.subg_pool:  # cora without pool is better
                xs = [xs[i] for i in self.subg_pool]
                x = self.pool(x, xs, **kwargs)

        # output layer
        if self.has_classifier:
            # if is_final:
            #     torch.save(x.cpu(), '/home/gangda/workspace/adapt-hop/processed/interpret/' +
            #                f'penn94_{self.name}_conv{self.conv_layers}_emb.pt')
            x = self.lin(x)

        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.conv_layers})')


class MLP(GenGNN):

    def init_conv(self, in_dim: int, out_dim: int,
                  **kwargs):
        for k in ['adj_norm', 'jk', 'residual']:
            kwargs.pop(k, None)
        return GenLinear(in_dim, out_dim, **kwargs)


class GCNNorm(GenGNN):

    def init_conv(self, in_dim: int, out_dim: int,
                  **kwargs):
        return GCNNormConv(in_dim, out_dim, **kwargs)


class GCNATP(GenGNN):

    def init_conv(self, in_dim: int, out_dim: int,
                  **kwargs):
        r = kwargs.pop('r', None)
        assert isinstance(r, Tensor)
        return GCNATPConv(in_dim, out_dim, r=r, **kwargs)


class GCNIIATP(GenGNN):

    def init_conv(self, in_dim: int, out_dim: int,
                  **kwargs):
        r = kwargs.pop('r', None)
        assert isinstance(r, Tensor)
        assert in_dim == out_dim
        return GCN2ATPConv(out_dim, r=r, **kwargs)


class GCN(GenGNN):

    def init_conv(self, in_dim: int, out_dim: int,
                  **kwargs):
        return GCNConv(in_dim, out_dim, **kwargs)

class GCNWithNorm(GenGNN):

    def init_conv(self, in_dim: int, out_dim: int,
                  **kwargs):
        return GCNConvWithNorm(in_dim, out_dim, **kwargs)

class BernNet(GenGNN):

    def init_conv(self, in_dim: int, out_dim: int,
                  **kwargs):
        adj_norm = kwargs.pop('adj_norm', True)
        return BernConv(in_dim, out_dim, **kwargs)

class MixHop(GenGNN):

    def init_conv(self, in_dim: int, out_dim: int,
                  **kwargs):
        powers = kwargs.pop('powers', [0, 1, 2])
        out_dim = out_dim // len(powers)
        return MixHopConv(in_dim, out_dim, powers=powers, **kwargs)


class MLPGCN(GenGNN):

    def init_conv(self, in_dim: int, out_dim: int,
                  **kwargs):
        return MLPGCNConv(in_dim, 256, out_dim, 3,
                          dropout=self.dropout, **kwargs)


class GraphSAGE(GenGNN):

    def init_conv(self, in_dim: int, out_dim: int,
                  **kwargs):
        return SAGEConv(in_dim, out_dim, **kwargs)


class GIN(GenGNN):

    def init_conv(self, in_dim: int, out_dim: int,
                  **kwargs):
        from torch_geometric.nn.models.mlp import MLP as PyGMLP

        mlp = PyGMLP([in_dim, out_dim, out_dim], batch_norm=True)
        conv = GINConv(mlp, **kwargs)
        conv.in_channels = in_dim

        return conv


class GAT(GenGNN):

    def init_conv(self, in_dim: int, out_dim: int,
                  **kwargs):
        v2 = kwargs.pop('v2', False)
        heads = kwargs.pop('heads', 1)
        # NOTICE: we will not be using dropout. Instead we use attn_dropout
        attn_dropout = kwargs.pop('attn_dropout', 0)
        edge_dropout = kwargs.pop('edge_dropout', 0) # only used in CATGAT
        symmetric_norm = kwargs.pop('symmetric_norm', False)  # only used in CATGAT
        adj_norm = kwargs.pop('adj_norm', True)  # not used
        last_layer = kwargs.pop('last_layer', False)

        if not last_layer:
            out_dim = out_dim // heads

        Conv = GATConv if not v2 else GATv2Conv
        conv = Conv(in_dim, out_dim, heads=heads, concat=not last_layer,
                    dropout=attn_dropout, **kwargs)
        return conv


class CATGAT(GenGNN):

    def init_conv(self, in_dim: int, out_dim: int,
                  **kwargs):
        v2 = kwargs.pop('v2', False) # useless argument
        heads = kwargs.pop('heads', 1)
        last_layer = kwargs.pop('last_layer', False)
        if not last_layer:
            out_dim = out_dim // heads
        return GATConvWithNorm(in_dim, out_dim, heads=heads,
                               keep_out_channels=last_layer,
                               self_concat=True,
                               norm=True,
                               **kwargs)


class FaberNet(GenGNN):

    def init_conv(self, in_dim: int, out_dim: int,
                  **kwargs):
        return FaberConv(in_dim, out_dim, **kwargs)


class DPA(GenGNN):

    def init_conv(self, in_dim: int, out_dim: int,
                  **kwargs):
        return DPAConv(in_dim, out_dim, **kwargs)


class DirGCN(GenGNN):

    def init_conv(self, in_dim: int, out_dim: int,
                  **kwargs):
        return DirGCNConv(in_dim, out_dim, **kwargs)


class DirSAGE(GenGNN):

    def init_conv(self, in_dim: int, out_dim: int,
                  **kwargs):
        return DirSAGEConv(in_dim, out_dim, **kwargs)


class DirGAT(GenGNN):

    def init_conv(self, in_dim: int, out_dim: int,
                  **kwargs):
        heads = kwargs.pop('heads', 1)
        attn_dropout = kwargs.pop('attn_dropout', 0)
        last_layer = kwargs.pop('last_layer', False)

        if not last_layer:
            out_dim = out_dim // heads

        conv = DirGATConv(in_dim, out_dim, heads=heads, concat=not last_layer,
                          dropout=attn_dropout, **kwargs)
        return conv


class PNA(GenGNN):

    def init_conv(self, in_dim: int, out_dim: int,
                  **kwargs):
        return PNAConv(in_dim, out_dim, **kwargs)


class GPS(GenGNN):

    def init_conv(self, in_dim: int, out_dim: int,
                  **kwargs):
        assert in_dim == out_dim
        conv_type = kwargs.pop('conv_type', 'GatedGCN')
        attn_kwargs = {'dropout': kwargs.pop('attn_dropout', 0)}

        if conv_type == 'GatedGCN':
            conv = GatedGraphConv(out_dim, num_layers=1)
        # elif conv_type == 'GINE':
        #     net = nn.Sequential(
        #         Linear(out_dim, out_dim),
        #         nn.ReLU(),
        #         Linear(out_dim, out_dim),
        #     )
        #     conv = GINEConv(net)
        else:
            raise NotImplementedError

        return GPSConv(out_dim, conv, dropout=self.dropout,
                       attn_kwargs=attn_kwargs, **kwargs)


class GCNII(GenGNN):

    def init_conv(self, in_dim: int, out_dim: int,
                  **kwargs):
        assert in_dim == out_dim
        return GCN2Conv(out_dim, **kwargs)


class G2GNN(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 conv_layers,
                 init_dropout=0,
                 dropout=0,
                 conv_type='SAGE',
                 p=2.,
                 use_gg_conv=True,
                 **kwargs
                 ):
        super(G2GNN, self).__init__()
        self.conv_type = conv_type
        self.enc = nn.Linear(in_dim, hidden_dim)
        self.dec = nn.Linear(hidden_dim, out_dim)
        self.drop_in = init_dropout
        self.drop = dropout
        self.nlayers = conv_layers
        if conv_type == 'GCN':
            self.conv = GCNConv(hidden_dim, hidden_dim)
            if use_gg_conv:
                self.conv_gg = GCNConv(hidden_dim, hidden_dim)
        elif conv_type == 'SAGE':
            self.conv = SAGEConv(hidden_dim, hidden_dim)
            if use_gg_conv:
                self.conv_gg = SAGEConv(hidden_dim, hidden_dim)
        elif conv_type == 'GAT':
            self.conv = GATConv(hidden_dim, hidden_dim, heads=4, concat=True)
            if use_gg_conv:
                self.conv_gg = GATConv(hidden_dim, hidden_dim, heads=4, concat=True)
        else:
            print('specified graph conv not implemented')

        if use_gg_conv:
            self.G2 = G2(self.conv_gg, p, conv_type, activation=nn.ReLU())
        else:
            self.G2 = G2(self.conv, p, conv_type, activation=nn.ReLU())

    def forward(self, X, edge_index):
        n_nodes = X.size(0)
        X = F.dropout(X, self.drop_in, training=self.training)
        X = torch.relu(self.enc(X))

        for i in range(self.nlayers):
            if self.conv_type == 'GAT':
                X_ = F.elu(self.conv(X, edge_index)).view(n_nodes, -1, 4).mean(dim=-1)
            else:
                X_ = torch.relu(self.conv(X, edge_index))
            tau = self.G2(X, edge_index)
            X = (1 - tau) * X + tau * X_
        X = F.dropout(X, self.drop, training=self.training)

        return self.dec(X)


class SGC(torch.nn.Module):

    def __init__(self, config, in_dim: int, out_dim: int, num_hops: int):
        super(SGC, self).__init__()
        self.K = num_hops
        self._cached_x = None
        self.dropedge = config.dropedge
        self.mlp = get_model(config, in_dim, out_dim)

    def forward(self, x, adj):
        adj = dropout_edge(adj, p=self.dropedge, training=self.training)

        if self._cached_x is None:
            adj = adj_norm(adj, 'sym')
            for k in range(self.K):
                x = adj @ x
            self._cached_x = x
        x = self._cached_x.detach()
        x = self.mlp(x)
        return x


class GPRGNN(torch.nn.Module):

    def __init__(self,
                 in_dim: int,
                 hidden_dim,
                 out_dim: int,
                 K: int,  # num propagation
                 name: str,  # model name ['APPNP', 'GPRGNN']
                 init_dropout: float = 0.5,
                 dropout: float = 0.5,
                 prop_dropout: float = 0.5,
                 gpr_init: str = 'PPR',  # ['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null']
                 alpha: float = 0.1,
                 gamma: float = None,
                 **kwargs,
                 ):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(in_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, out_dim)

        if name.upper() == 'APPNP':
            self.prop1 = APPNP(K, alpha)
        elif name.upper() == 'GPRGNN':
            self.prop1 = GPR_prop(K, alpha, gpr_init, gamma)

        self.init_dropout = init_dropout
        self.dropout = dropout
        self.prop_dropout = prop_dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.init_dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        x = F.dropout(x, p=self.prop_dropout, training=self.training)
        x = self.prop1(x, edge_index)

        return x


class GAMLP(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 conv_layers,
                 dropout=0.5,
                 init_dropout=0.,
                 attn_dropout=0.5,
                 norm='batch',
                 n_layers_1=4,
                 n_layers_2=4,
                 act='leaky_relu',
                 alpha=0.5,
                 pre_process=True,
                 init_res=False,
                 **kwargs
                 ):
        super(GAMLP, self).__init__()
        self.num_hops = conv_layers
        self.prelu = nn.PReLU()
        if pre_process:
            self.process = nn.ModuleList(
                [BaseMLP(in_dim, hidden_dim, hidden_dim, 2, dropout, act='prelu', norm=norm)
                 for _ in range(self.num_hops+1)])
            self.lr_jk_ref = BaseMLP(
                (self.num_hops+1) * hidden_dim, hidden_dim, hidden_dim, n_layers_1, dropout,
                act='prelu', incep_alpha=alpha, residual=True, norm=norm)
            self.lr_att = nn.Linear(hidden_dim + hidden_dim, 1)
            self.res_fc = nn.Linear(in_dim, hidden_dim)
            self.lr_output = BaseMLP(
                hidden_dim, hidden_dim, out_dim, n_layers_2, dropout,
                act='prelu', incep_alpha=alpha, residual=True, norm=norm)
        else:
            self.lr_jk_ref = BaseMLP(
                (self.num_hops+1) * in_dim, hidden_dim, hidden_dim, n_layers_1, dropout,
                act='prelu', incep_alpha=alpha, residual=True, norm=norm)
            self.lr_att = nn.Linear(in_dim + hidden_dim, 1)
            self.res_fc = nn.Linear(in_dim, in_dim)
            self.lr_output = BaseMLP(
                in_dim, hidden_dim, out_dim, n_layers_2, dropout,
                act='prelu', incep_alpha=alpha, residual=True, norm=norm)
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(init_dropout)
        self.att_drop = nn.Dropout(attn_dropout)
        self.pre_process = pre_process
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        self.residual = init_res
        self._cached_xs = None
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.lr_att.weight, gain=gain)
        nn.init.zeros_(self.lr_att.bias)
        nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)
        nn.init.zeros_(self.res_fc.bias)
        self.lr_output.reset_parameters()
        self.lr_jk_ref.reset_parameters()
        if self.pre_process:
            for layer in self.process:
                layer.reset_parameters()

    def forward(self, x, edge_index):
        if self._cached_xs is None:
            self._cached_xs = [x]
            adj = adj_norm(edge_index, 'sym')
            for k in range(self.num_hops):
                self._cached_xs.append(adj @ self._cached_xs[-1])

        num_node = self._cached_xs[0].shape[0]
        feature_list = [self.input_drop(feature) for feature in self._cached_xs]
        input_list = []
        if self.pre_process:
            for i in range(len(feature_list)):
                input_list.append(self.process[i](feature_list[i]))
        else:
            input_list = feature_list
        concat_features = torch.cat(input_list, dim=1)
        jk_ref = self.dropout(self.prelu(self.lr_jk_ref(concat_features)))
        attention_scores = [self.act(self.lr_att(torch.cat((jk_ref, x), dim=1))
                                     ).view(num_node, 1) for x in input_list]
        W = torch.cat(attention_scores, dim=1)
        W = F.softmax(W, 1)
        right_1 = torch.mul(input_list[0], self.att_drop(
            W[:, 0].view(num_node, 1)))
        for i in range(1, self.num_hops+1):
            right_1 = right_1 + \
                      torch.mul(input_list[i], self.att_drop(
                          W[:, i].view(num_node, 1)))
        if self.residual:
            right_1 += self.res_fc(feature_list[0])
            right_1 = self.dropout(self.prelu(right_1))
        right_1 = self.lr_output(right_1)
        return right_1


class ACMGCN(nn.Module):
    def __init__(
            self,
            in_dim,
            hidden_dim,
            out_dim,
            conv_layers,
            num_nodes,
            dropout,
            model_type,  # acmgcn, acmgcnpp
            variant=False,
            layer_norm=False,
            init_layers_X=1,
            init_dropout=0.,
            active_filters=[1,1,1],
            norm='batch',
            **kwargs,
    ):
        super(ACMGCN, self).__init__()

        assert model_type in ['acmgcn', 'acmgcnp', 'acmgcnpp']
        if model_type == "acmgcnpp":
            self.mlpX = BaseMLP(in_dim, hidden_dim, hidden_dim, num_layers=init_layers_X, dropout=0,
                                norm=norm)
        self.gcns = nn.ModuleList()
        self.model_type, self.dropout, self.init_dropout, self.norm = model_type, dropout, init_dropout, norm
        self.cached_adj, self.cached_adj_low, self.cached_adj_high = None, None, None
        if conv_layers == 1:
            self.gcns.append(
                ACMGCNConv(
                    in_dim,
                    out_dim,
                    num_nodes,
                    model_type=model_type,
                    variant=variant,
                    layer_norm=layer_norm,
                    active_filters=active_filters
                )
            )
        else:
            self.gcns.append(
                ACMGCNConv(
                    in_dim,
                    hidden_dim,
                    num_nodes,
                    model_type=model_type,
                    variant=variant,
                    layer_norm=layer_norm,
                    active_filters=active_filters
                )
            )
            for l in range(2, conv_layers):
                self.gcns.append(
                    ACMGCNConv(
                        hidden_dim,
                        hidden_dim,
                        num_nodes,
                        model_type=model_type,
                        variant=variant,
                        layer_norm=layer_norm,
                        active_filters=active_filters
                    )
                )
            self.gcns.append(
                ACMGCNConv(
                    hidden_dim,
                    out_dim,
                    num_nodes,
                    model_type=model_type,
                    variant=variant,
                    layer_norm=layer_norm,
                    active_filters=active_filters
                )
            )
        # normalizations
        self.norms = None
        if conv_layers > 0:
            self.norms, eps = ModuleList(), 1e-9
            num_norm = conv_layers - 1
            for _ in range(num_norm):
                if norm == 'layer' and hidden_dim > 1:
                    self.norms.append(nn.LayerNorm(hidden_dim, eps=eps))
                elif norm == 'batch':
                    self.norms.append(nn.BatchNorm1d(hidden_dim, eps=eps))
        self.reset_parameters()

    def reset_parameters(self):
        if self.model_type == "acmgcnpp":
            self.mlpX.reset_parameters()
        else:
            pass
        for norm in self.norms or []:
            norm.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, self.init_dropout, training=self.training)
        if self.cached_adj is None:
            self.cached_adj = edge_index
            self.cached_adj_low = adj_norm(edge_index, norm='rw')
            self.cached_adj_high = (
                    edge_index.eye(x.size(0), device=x.device)
                    + torch.full((x.size(0), 1), -1, device=x.device) * self.cached_adj_low
            )
        adj_unnorm, adj_low, adj_high = self.cached_adj, self.cached_adj_low, self.cached_adj_high

        x = F.dropout(x, self.dropout, training=self.training)

        xX = None
        if self.model_type == "acmgcnpp":
            xX = F.dropout(
                F.relu(self.mlpX(x)),
                self.dropout,
                training=self.training,
            )

        for l in range(len(self.gcns) - 1):
            x = self.gcns[l](x, adj_low, adj_high, adj_unnorm)
            if self.norms:
                x = self.norms[l](x)
            x = F.dropout((F.relu(x)), self.dropout, training=self.training)

        if self.model_type == "acmgcnpp":
            fea2 = self.gcns[-1](x + xX, adj_low, adj_high, adj_unnorm)
        else:
            fea2 = self.gcns[-1](x, adj_low, adj_high, adj_unnorm)

        return fea2


class NHGCN(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 hidden_dim,
                 dropout,
                 final_dropout=0.6,
                 add_self=1,
                 final_agg='',
                 ):
        super().__init__()
        self.W1L = Parameter(torch.empty(in_dim,  hidden_dim))
        self.W1H = Parameter(torch.empty(in_dim, hidden_dim))
        self.W2L = Parameter(torch.empty(hidden_dim, hidden_dim))
        self.W2H = Parameter(torch.empty(hidden_dim, hidden_dim))

        self.lam = Parameter(torch.zeros(3))
        self.lam1 = Parameter(torch.zeros(2))
        self.lam2 = Parameter(torch.zeros(2))
        self.dropout = Dropout(p=dropout)
        self.dropout2 = Dropout(p=dropout)
        self.finaldp = Dropout(p=final_dropout)
        self.act = F.relu

        self.WX = Parameter(torch.empty(in_dim, hidden_dim))
        # self.lin2 = Linear(3 * params.hidden, num_classes,bias=False)
        self.lin1 = Linear(hidden_dim, out_dim)
        self.add_self = add_self
        self._cached_adj_l = None
        self._cached_adj_h = None
        self.reset_parameter()

    def reset_parameter(self):
        xavier_uniform_(self.W1L, gain=calculate_gain('relu'))
        xavier_uniform_(self.W1H, gain=calculate_gain('relu'))
        xavier_uniform_(self.W2L, gain=calculate_gain('relu'))
        xavier_uniform_(self.W2H, gain=calculate_gain('relu'))
        xavier_uniform_(self.WX, gain=calculate_gain('relu'))

    def agg_norm(self, adj_t, mask, mtype='target'):
        # TODO: A^2
        if mtype == 'target':
            A_tilde = mul(adj_t, mask.view(-1,1))
        elif mtype == 'source':
            A_tilde = mul(adj_t, mask.view(1,-1))
        else:
            A_tilde = SparseTensor.from_torch_sparse_coo_tensor(
                torch.sparse.mm(
                    mask, torch.sparse.mm(
                        mask, adj_t.to_torch_sparse_coo_tensor())))
        if self.addself:
            A_tilde = fill_diag(A_tilde, 1.)
        else:
            A_tilde = remove_diag(A_tilde)
        D_tilde = sparsesum(A_tilde, dim=1)
        D_tilde_sq = D_tilde.pow_(-0.5)
        D_tilde_sq.masked_fill_(D_tilde_sq == float('inf'), 0.)
        A_hat = mul(A_tilde, D_tilde_sq.view(-1, 1))
        A_hat = mul(A_hat, D_tilde_sq.view(1, -1))

        return A_hat

    def forward(self, data):
        x = SparseTensor.from_dense(data.x)
        cc_mask = data.cc_mask
        # cc_mask_t = torch.unsqueeze(data.cc_mask, dim=-1)
        rev_cc_mask = torch.ones_like(cc_mask) - cc_mask
        # rev_cc_mask = 1 / (cc_mask + 1)
        # rev_cc_mask_t = torch.unsqueeze(rev_cc_mask, dim=-1)
        edge_index = data.edge_index
        adj_t = SparseTensor(row=edge_index[1], col=edge_index[0])

        # low_cc mask
        if data.update_cc:
            A_hat_l = self.agg_norm(adj_t, cc_mask, 'source')
            self._cached_adj_l = A_hat_l
        else:
            A_hat_l = self._cached_adj_l

        # high_cc mask
        if data.update_cc:
            A_hat_h = self.agg_norm(adj_t, rev_cc_mask, 'source')
            self._cached_adj_h = A_hat_h
        else:
            A_hat_h = self._cached_adj_h

        xl = matmul(A_hat_l, x)
        xl = matmul(xl, self.W1L)
        xl = self.act(xl)
        xl = self.dropout(xl)
        xl = torch.mm(matmul(A_hat_l, xl), self.W2L)
        # high_cc partion
        xh = matmul(A_hat_h, x)
        xh = matmul(xh, self.W1H)
        xh = self.act(xh)
        xh = self.dropout(xh)
        xh = torch.mm(matmul(A_hat_h, xh), self.W2H)

        x = matmul(x, self.WX)
        x = self.act(xh)
        x = self.dropout(xh)

        lamx, laml, lamh = Softmax()(self.lam)
        if self.args.finalagg == 'add':
            xf = lamx * x + laml * xl + lamh * xh
            xf = self.act(xf)
            xf = self.finaldp(xf)
            xf = self.lin1(xf)
        elif self.args.finalagg == 'max':
            xf = torch.stack((x, xl, xh), dim=0)
            xf = torch.max(xf, dim=0)[0]
            xf = self.act(xf)
            xf = self.finaldp(xf)
            xf = self.lin1(xf)
        return xf


class Polynormer(torch.nn.Module):
    def __init__(
            self,
            in_dim,
            hidden_dim,
            out_dim,
            local_layers=3,
            global_layers=2,
            init_dropout=0.15,
            dropout=0.5,
            global_dropout=0.5,
            heads=1,
            beta=-1,
            pre_ln=False,
            post_bn=False,
            **kwargs,
    ):
        super(Polynormer, self).__init__()

        self._global = False
        self.in_drop = init_dropout
        self.dropout = dropout
        self.pre_ln = pre_ln
        self.post_bn = post_bn

        ## Two initialization strategies on beta
        self.beta = beta
        if self.beta < 0:
            self.betas = torch.nn.Parameter(torch.zeros(local_layers, heads * hidden_dim))
        else:
            self.betas = torch.nn.Parameter(torch.ones(local_layers, heads * hidden_dim) * self.beta)

        self.h_lins = torch.nn.ModuleList()
        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = torch.nn.ModuleList()
        if self.post_bn:
            self.post_bns = torch.nn.ModuleList()

        for _ in range(local_layers):
            self.h_lins.append(torch.nn.Linear(heads * hidden_dim, heads * hidden_dim))
            self.local_convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads,
                                            concat=True, add_self_loops=False, bias=False))
            self.lins.append(torch.nn.Linear(heads * hidden_dim, heads * hidden_dim))
            self.lns.append(torch.nn.LayerNorm(heads * hidden_dim))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(heads * hidden_dim))
            if self.post_bn:
                self.post_bns.append(torch.nn.BatchNorm1d(heads * hidden_dim))

        self.lin_in = torch.nn.Linear(in_dim, heads * hidden_dim)
        self.ln = torch.nn.LayerNorm(heads * hidden_dim)
        self.global_attn = PolyGlobalAttn(hidden_dim, heads, global_layers, beta, global_dropout)
        self.pred_local = torch.nn.Linear(heads * hidden_dim, out_dim)
        self.pred_global = torch.nn.Linear(heads * hidden_dim, out_dim)

    def reset_parameters(self):
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for h_lin in self.h_lins:
            h_lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        if self.post_bn:
            for p_bn in self.post_bns:
                p_bn.reset_parameters()
        self.lin_in.reset_parameters()
        self.ln.reset_parameters()
        self.global_attn.reset_parameters()
        self.pred_local.reset_parameters()
        self.pred_global.reset_parameters()
        if self.beta < 0:
            torch.nn.init.xavier_normal_(self.betas)
        else:
            torch.nn.init.constant_(self.betas, self.beta)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.in_drop, training=self.training)
        x = self.lin_in(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        ## equivalent local attention
        x_local = 0
        for i, local_conv in enumerate(self.local_convs):
            if self.pre_ln:
                x = self.pre_lns[i](x)
            h = self.h_lins[i](x)
            h = F.relu(h)
            x = local_conv(x, edge_index) + self.lins[i](x)
            if self.post_bn:
                x = self.post_bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.beta < 0:
                beta = F.sigmoid(self.betas[i]).unsqueeze(0)
            else:
                beta = self.betas[i].unsqueeze(0)
            x = (1 - beta) * self.lns[i](h * x) + beta * x
            x_local = x_local + x

        ## equivalent global attention
        if self._global:
            x_global = self.global_attn(self.ln(x_local))
            x = self.pred_global(x_global)
        else:
            x = self.pred_local(x_local)

        return x


class GCN_SpMoE(torch.nn.Module):
    def __init__(
            self,
            in_dim,
            hidden_dim,
            out_dim,
            conv_layers,
            init_dropout,
            dropout,
            num_experts=4,
            k=1,
            coef=1e-2,
            **kwargs,
    ):
        super(GCN_SpMoE, self).__init__()

        self.cached_adj = None
        self.load_balance_loss = None
        self.num_layers = conv_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_dim, hidden_dim, normalize=False))
        for layer_idx in range(conv_layers - 2):
            if layer_idx % 2 == 0:
                ffn = GMoE(input_size=hidden_dim, output_size=hidden_dim, num_experts=num_experts, k=k, coef=coef)
                self.convs.append(ffn)
            else:
                self.convs.append(
                    GCNConv(hidden_dim, hidden_dim, normalize=False))
        self.convs.append(
            GCNConv(hidden_dim, out_dim, normalize=False))

        self.init_dropout = init_dropout
        self.dropout = dropout

    def forward(self, x, adj_t):
        x = F.dropout(x, p=self.init_dropout, training=self.training)
        if self.cached_adj is None:
            self.cached_adj = adj_norm(adj_t, norm='sym')
        adj_t = self.cached_adj

        self.load_balance_loss = 0  # initialize load_balance_loss to 0 at the beginning of each forward pass.
        for conv in self.convs[:-1]:
            if isinstance(conv, GMoE):
                x, _layer_load_balance_loss = conv(x, adj_t)
                self.load_balance_loss += _layer_load_balance_loss
            else:
                x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        self.load_balance_loss /= math.ceil((self.num_layers-2)/2)
        return x


class SAGE_SpMoE(torch.nn.Module):
    def __init__(
            self,
            in_dim,
            hidden_dim,
            out_dim,
            conv_layers,
            init_dropout=0.,
            dropout=0.,
            num_experts=4,
            k=1,
            coef=1e-2,
            **kwargs,
    ):
        super(SAGE_SpMoE, self).__init__()

        self.load_balance_loss = None
        self.num_layers = conv_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for layer_idx in range(conv_layers - 2):
            if layer_idx % 2 == 0:
                ffn = GMoE(input_size=hidden_dim, output_size=hidden_dim, num_experts=num_experts, k=k, coef=coef, sage=True)
                self.convs.append(ffn)
            else:
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, out_dim))

        self.init_dropout = init_dropout
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        x = F.dropout(x, p=self.init_dropout, training=self.training)

        self.load_balance_loss = 0  # initialize load_balance_loss to 0 at the beginning of each forward pass.
        for conv in self.convs[:-1]:
            if isinstance(conv, GMoE):
                x, _layer_load_balance_loss = conv(x, adj_t)
                self.load_balance_loss += _layer_load_balance_loss
            else:
                x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        self.load_balance_loss /= math.ceil((self.num_layers-2)/2)
        return x


class DA_MoE(nn.Module):
    def __init__(
            self,
            in_dim,
            hidden_dim,
            out_dim,
            conv_layers,
            init_dropout=0.,
            dropout=0.5,
            gate_type='GCN',  # 'GCN','liner','SAGE'
            gnn_type='GCN',
            min_layers=1,
            noisy_gating=True,
            k=4,
            coef=1e-3,
            gate_dropout=0.2,
            **kwargs,
    ):
        super(DA_MoE, self).__init__()
        self.load_balance_loss = None
        self.noisy_gating = noisy_gating
        self.num_experts = conv_layers
        self.k = k
        self.loss_coef = coef
        self.init_dropout = init_dropout
        self.gate_dropout = gate_dropout
        self.gate_type = gate_type

        # instantiate experts
        assert (self.k <= self.num_experts)
        self.experts = torch.nn.ModuleList()

        # model
        if gnn_type == 'GCN':
            for i in range(min_layers, self.num_experts + min_layers):
                self.experts.append(
                    GCN(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, conv_layers=i, dropout=dropout, name='GCN'))

        # gate
        if gate_type == 'liner':
            self.w_gate = nn.Parameter(torch.zeros(in_dim, self.num_experts), requires_grad=True)
        elif gate_type == 'GCN':
            self.gate_model = GCN(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=self.num_experts,
                                  conv_layers=2, dropout=dropout, name='GCN')

        elif gate_type == 'SAGE':
            self.gate_model = GraphSAGE(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=self.num_experts,
                                        conv_layers=2, dropout=dropout, name='SAGE')

        self.w_noise = nn.Parameter(torch.zeros(in_dim, self.num_experts), requires_grad=True)
        
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

    @staticmethod
    def cv_squared(x):
        """The squared coefficient of variation of a sample.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    @staticmethod
    def _gates_to_load(gates):
        """Compute the true load per expert, given the gates.
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
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
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, adj_t, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
        """
        # clean_logits = x @ self.w_gate
        if self.gate_type == 'liner':
            clean_logits = x @ self.w_gate
        elif self.gate_type == 'GCN' or 'SAGE':
            clean_logits = self.gate_model(x, adj_t)

        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = (self.softplus(raw_noise_stddev) + noise_epsilon)
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

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

    def forward(self, x, adj_t):
        x = F.dropout(x, p=self.init_dropout, training=self.training)

        gates, load = self.noisy_top_k_gating(x, adj_t, self.training)
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= self.loss_coef

        expert_outputs = []
        for i in range(self.num_experts):
            input_x = x
            output = self.experts[i](input_x, adj_t)
            expert_outputs.append(output)
        expert_outputs = torch.stack(expert_outputs, dim=1)

        y = gates.unsqueeze(dim=-1) * expert_outputs
        y = y.sum(dim=1)

        self.load_balance_loss = loss
        return y


def get_model(config, num_features, num_classes):
    model_name = config.name.upper()

    if model_name == 'SGC':
        config = Dict(config.copy())
        config.name = 'MLP'
        num_hops = config.conv_layers
        config.conv_layers = config.init_layers
        config.init_layers = 0
        # return SGConv(num_features, num_classes, K=config.conv_layers, cached=True)
        return SGC(config, num_features, num_classes, num_hops)
    if model_name == 'MIXHOP':
        # check hidden dimension compatibility with powers
        powers = config.powers if 'powers' in config else [0, 1, 2]
        if config.hidden_dim % len(powers) != 0:
            config.hidden_dim = int(round(config.hidden_dim / len(powers)) * len(powers))
            print(f'Warning: hidden_dim is not divisible by the number of powers. Adjusting hidden_dim to the nearest multiple {config.hidden_dim}')
    model_dict = {
        'GCN': GCNWithNorm,
        'BERNNET': BernNet,
        'GCN-ATP': GCNATP,
        'GCNII-ATP': GCNIIATP,
        'GCNNORM': GCNNorm,
        'SAGE': GraphSAGE,
        'GAT': GAT,
        'FABER': FaberNet,
        'DPA': DPA,
        'DIRGCN': DirGCN,
        'DIRSAGE': DirSAGE,
        'DIRGAT': DirGAT,
        'MLP': MLP,
        'MLPGCN': MLPGCN,
        'CATGAT': CATGAT,
        'GIN': GIN,
        'PNA': PNA,
        'GCNII': GCNII,
        'MIXHOP': MixHop,
        'GPS': GPS,
        'APPNP': GPRGNN,
        'GPRGNN': GPRGNN,
        'GAMLP': GAMLP,
        'ACMGCN': ACMGCN,
        'POLYNORMER': Polynormer,
        'G2GNN': G2GNN,
        'GMOE-GCN': GCN_SpMoE,
        'GMOE-SAGE': SAGE_SpMoE,
        'DAMOE': DA_MoE,
    }

    return model_dict[model_name](in_dim=num_features, out_dim=num_classes, **config)
