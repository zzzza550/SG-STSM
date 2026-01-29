from collections import OrderedDict
from datetime import time

import math
import torch
from matplotlib import pyplot as plt
from torch import nn
from timm.models.layers import DropPath

# from model.modules.graph import SpatialGCN
from model.mamba_ssm.modules.mamba2 import Mamba2

from model.modules.attention import Attention
from model.modules.mlp import MLP
from einops import rearrange


class TransformerBlock(nn.Module):
    """
    Implementation of AGFormer block.
    """

    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, qkv_bias=False, qk_scale=None, use_layer_scale=True, layer_scale_init_value=1e-5,
                 mode='spatial', mixer_type="attention", use_temporal_similarity=True,
                 n_frames=243):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if mixer_type == 'attention':
            self.mixer = Attention(dim, dim, num_heads, qkv_bias, qk_scale, attn_drop,
                                   proj_drop=drop, mode=mode)
        else:
            raise NotImplementedError("-----------------------")
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep GraphFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        """
        x: tensor with shape [B, T, J, C]
        """
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(0).unsqueeze(0)
                * self.mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MSTPFormerBlock(nn.Module):
    """
    Implementation of MSTPFormer block. It has two ST and TS branches followed by adaptive fusion.
    """

    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, use_layer_scale=True, qkv_bias=False, qk_scale=None, layer_scale_init_value=1e-5,
                 use_adaptive_fusion=True, hierarchical=False, use_temporal_similarity=True,
                 n_frames=243):
        super().__init__()
        self.hierarchical = hierarchical
        dim = dim // 2 if hierarchical else dim

        # Attention
        self.att_spatial = TransformerBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                            qk_scale, use_layer_scale, layer_scale_init_value,
                                            mode='spatial', mixer_type="attention",
                                            use_temporal_similarity=use_temporal_similarity,
                                            n_frames=n_frames)
        self.att_temporal = TransformerBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                             qk_scale, use_layer_scale, layer_scale_init_value,
                                             mode='temporal', mixer_type="attention",
                                             use_temporal_similarity=use_temporal_similarity,
                                             n_frames=n_frames)

        # mamba
        self.mamba_spatial = SpatialMambaBlock(dim, mlp_ratio, drop, drop_path=drop_path)
        # self.mamba_temporal = TemporalMambaBlock(dim, mlp_ratio, drop, drop_path=drop_path)
        self.mamba_temporal = StructuredSSMTemporalBlock(
            dim=dim, num_joints=17,
            ssm_hidden=128, heads=4, headdim=16,
            mlp_ratio=4.0, drop=0.0, drop_path=0.1,
            use_adaptive_adj=True,
            couple_hidden=True, beta_init=0.2)
        self.a = 0
        # GCN-Spatial
        self.gcn_spatial = GCNSpatialBlock(dim, dim, num_nodes=17, bias=True)
        self.use_adaptive_fusion = use_adaptive_fusion
        if self.use_adaptive_fusion:
            self.fusion = nn.Linear(dim * 2, 2)
            self._init_fusion()

        self.s_fusion = nn.Linear(dim * 3, 3)
        self._init_s_fusion()
        self.s_alpha = nn.Parameter(torch.tensor(0.3))
        self.s_drop = nn.Dropout(0.1)

    def _init_fusion(self):
        self.fusion.weight.data.fill_(0)
        self.fusion.bias.data.fill_(0.5)

    def _init_s_fusion(self):
        self.fusion.weight.data.fill_(0)
        self.fusion.bias.data.fill_(0.5)

    def spatial_fusion(self, x_gcn, x_att, x_mamba):
        x_cat = torch.cat([x_gcn, x_att, x_mamba], dim=-1)
        alpha = self.s_fusion(x_cat)
        alpha = torch.softmax(alpha, dim=-1)
        out = (x_gcn * alpha[..., 0:1] +
               x_att * alpha[..., 1:2] +
               x_mamba * alpha[..., 2:3])
        return self.s_drop(out)

    def forward(self, x):
        """
        x: tensor with shape [B, T, J, C]
        """
        if self.hierarchical:
            B, T, J, C = x.shape
            # x_att, x_graph = x[..., :C // 2], x[..., C // 2:]
            x_att, x_mamba = x[..., :C // 2], x[..., C // 2:]

            x_att = self.att_temporal(self.att_spatial(x_att))

            # x_graph = self.graph_temporal(self.graph_spatial(x_graph + x_att))
            x_mamba = self.mamba_temporal(self.mamba_spatial(x_mamba + x_att))
        else:


            x_att = self.att_temporal(self.att_spatial(x))
            x_mamba = self.mamba_temporal(self.mamba_spatial(self.gcn_spatial(x)))


        if self.hierarchical:
            x = torch.cat((x_att, x_mamba), dim=-1)
        elif self.use_adaptive_fusion:
            alpha = torch.cat((x_att, x_mamba), dim=-1)
            alpha = self.fusion(alpha)
            alpha = alpha.softmax(dim=-1)
            x = x_att * alpha[..., 0:1] + x_mamba * alpha[..., 1:2]

        #     x = (x_att + x_mamba) * 0.5

        return x


def create_layers(dim, n_layers, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop_rate=0., drop_path_rate=0.,
                  num_heads=8, use_layer_scale=True, qkv_bias=False, qkv_scale=None, layer_scale_init_value=1e-5,
                  use_adaptive_fusion=True, hierarchical=False,
                  n_frames=243):
    """
    generates MSTPFormer layers
    """
    layers = []
    for _ in range(n_layers):
        layers.append(MSTPFormerBlock(dim=dim,
                                      mlp_ratio=mlp_ratio,
                                      act_layer=act_layer,
                                      attn_drop=attn_drop,
                                      drop=drop_rate,
                                      drop_path=drop_path_rate,
                                      num_heads=num_heads,
                                      use_layer_scale=use_layer_scale,
                                      layer_scale_init_value=layer_scale_init_value,
                                      qkv_bias=qkv_bias,
                                      qk_scale=qkv_scale,
                                      use_adaptive_fusion=use_adaptive_fusion,
                                      hierarchical=hierarchical,
                                      n_frames=n_frames))
    layers = nn.Sequential(*layers)

    return layers


class MSTPFormer(nn.Module):
    """
    MSTPFormer, the main class of our model.
    """

    def __init__(self, n_layers, dim_in, dim_feat, dim_rep=512, dim_out=3, mlp_ratio=4, act_layer=nn.GELU, attn_drop=0.,
                 drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5, use_adaptive_fusion=True,
                 num_heads=4, qkv_bias=False, qkv_scale=None, hierarchical=False, num_joints=17,
                 n_frames=243):
        """
        :param n_layers: Number of layers.
        :param dim_in: Input dimension.
        :param dim_feat: Feature dimension.
        :param dim_rep: Motion representation dimension
        :param dim_out: output dimension. For 3D pose lifting it is set to 3
        :param mlp_ratio: MLP ratio.
        :param act_layer: Activation layer.
        :param drop: Dropout rate.
        :param drop_path: Stochastic drop probability.
        :param use_layer_scale: Whether to use layer scaling or not.
        :param layer_scale_init_value: Layer scale init value in case of using layer scaling.
        :param use_adaptive_fusion: Whether to use adaptive fusion or not.
        :param num_heads: Number of attention heads in attention branch
        :param qkv_bias: Whether to include bias in the linear layers that create query, key, and value or not.
        :param qkv_scale: scale factor to multiply after outer product of query and key. If None, it's set to
                          1 / sqrt(dim_feature // num_heads)
        :param hierarchical: Whether to use hierarchical structure or not.
        :param num_joints: Number of joints.
        :param n_frames: Number of frames. Default is 243
        """
        super().__init__()

        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.norm = nn.LayerNorm(dim_feat)

        # self.temp_embed = nn.Parameter(torch.zeros(1, n_frames, 1, dim_feat))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        # nn.init.trunc_normal_(self.temp_embed, std=0.02)
        # nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.layers = create_layers(dim=dim_feat,
                                    n_layers=n_layers,
                                    mlp_ratio=mlp_ratio,
                                    act_layer=act_layer,
                                    attn_drop=attn_drop,
                                    drop_rate=drop,
                                    drop_path_rate=drop_path,
                                    num_heads=num_heads,
                                    use_layer_scale=use_layer_scale,
                                    qkv_bias=qkv_bias,
                                    qkv_scale=qkv_scale,
                                    layer_scale_init_value=layer_scale_init_value,
                                    use_adaptive_fusion=use_adaptive_fusion,
                                    hierarchical=hierarchical,
                                    n_frames=n_frames)

        self.rep_logit = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(dim_feat, dim_rep)),
            ('act', nn.Tanh())
        ]))

        self.head = nn.Linear(dim_rep, dim_out)

    def forward(self, x, return_rep=False):
        """
        :param x: tensor with shape [B, T, J, C] (T=243, J=17, C=3)
        :param return_rep: Returns motion representation feature volume (In case of using this as backbone)
        """
        # b, f, j, c = x.shape

        x = self.joints_embed(x)

        x = x + self.pos_embed

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.rep_logit(x)
        if return_rep:
            return x

        x = self.head(x)

        return x




class SpatialMambaBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4., drop=0., layer_idx=0,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, comb=False, changedim=False, currentdim=0,
                 depth=0, vis=False,
                 ):
        super().__init__()

        self.changedim = changedim
        self.currentdim = currentdim
        self.depth = depth
        if self.changedim:
            assert self.depth > 0
        self.norm1 = norm_layer(dim)
        self.attn = Mamba2(dim, 128, 4, headdim=32)
        self.attn_inv = Mamba2(dim, 128, 4, headdim=32)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if self.changedim and self.currentdim < self.depth // 2:
            self.reduction = nn.Conv1d(dim, dim // 2, kernel_size=1)
            # self.reduction = nn.Linear(dim, dim//2)
            # print("spat 1")
        elif self.changedim and depth > self.currentdim > self.depth // 2:
            self.improve = nn.Conv1d(dim, dim * 2, kernel_size=1)
            # self.improve = nn.Linear(dim, dim*2)
            # print("spat 2")
        self.vis = vis

    def forward(self, x, vis=False):
        assert len(x.shape) == 4, 'input shape must 4'
        b, f, j, c = x.shape
        x = rearrange(x, 'b f j c -> (b f) j c')
        res = x
        x = self.norm1(x)  # LN
        x_inv = torch.flip(x, dims=[-1])
        x = self.attn(x)  # Mamba
        x_inv = self.attn_inv(x_inv)
        x_fuse = x + x_inv
        x = self.drop_path(x_fuse)
        x = x + res
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.changedim and self.currentdim < self.depth // 2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.reduction(x)
            x = rearrange(x, 'b c t -> b t c')
        elif self.changedim and self.depth > self.currentdim > self.depth // 2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.improve(x)
            x = rearrange(x, 'b c t -> b t c')
        x = rearrange(x, '(b f) j c -> b f j c', b=b, f=f)
        return x


class GCNSpatialBlock(nn.Module):
    def __init__(self, dim_in, dim_out, num_nodes, use_dynamic_adj=True, use_residual=True, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_nodes = num_nodes
        self.use_dynamic_adj = use_dynamic_adj
        self.use_residual = use_residual
        self.alpha = nn.Parameter(torch.zeros(num_nodes))


        adj_1st_order_raw = torch.zeros((num_nodes, num_nodes))
        CONNECTIONS = {
            0: [1, 7], 1: [0, 2], 2: [1, 3], 3: [2],
            4: [0, 5], 5: [4, 6], 6: [5],
            7: [0, 8], 8: [7, 9, 11, 14],
            9: [8, 10], 10: [9],
            11: [8, 12], 12: [11, 13], 13: [12],
            14: [8, 15], 15: [14, 16], 16: [15]
        }
        for i, neighbors in CONNECTIONS.items():
            for j in neighbors:
                adj_1st_order_raw[i, j] = 1


        self.register_buffer('adj_1st_order_base', adj_1st_order_raw + torch.eye(num_nodes))


        adj_2nd_order_raw_paths = torch.matmul(adj_1st_order_raw.float(), adj_1st_order_raw.float())


        adj_2nd_order_pure = (adj_2nd_order_raw_paths > 0).float()

        adj_2nd_order_pure = adj_2nd_order_pure - adj_1st_order_raw - torch.eye(num_nodes)
        adj_2nd_order_pure = (adj_2nd_order_pure > 0).float()

        self.register_buffer('adj_2nd_order_base', adj_2nd_order_pure)


        self.adj_learnable_1st = nn.Parameter(torch.zeros_like(self.adj_1st_order_base))
        self.adj_learnable_2nd = nn.Parameter(torch.zeros_like(self.adj_2nd_order_base))


        self.weight_static_1st = nn.Parameter(torch.ones(1))
        self.weight_static_2nd = nn.Parameter(torch.ones(1))

        # ==== GCN 权重 ====
        self.W = nn.Parameter(torch.Tensor(dim_in, dim_out))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(dim_out))
            fan_in = self.dim_in
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)


        if self.use_dynamic_adj:
            self.gate_linear = nn.Linear(dim_in, 1)


        self.bn = nn.BatchNorm1d(dim_out)
        self.relu = nn.ReLU()


        self.res_proj = nn.Identity()
        if use_residual and dim_in != dim_out:
            self.res_proj = nn.Linear(dim_in, dim_out)

    def forward(self, x):  # x: (B, T, J, C_in)
        B, T, J, C = x.shape
        x = x.view(B * T, J, C)  # (B*T, J, C_in)
        x_input = x


        A_1st_order_weighted = self.adj_1st_order_base + torch.sigmoid(self.adj_learnable_1st)
        A_2nd_order_weighted = self.adj_2nd_order_base + torch.sigmoid(self.adj_learnable_2nd)


        combined_static_adj = (
                nn.functional.softplus(self.weight_static_1st) * A_1st_order_weighted +
                nn.functional.softplus(self.weight_static_2nd) * A_2nd_order_weighted
        )

        combined_static_adj = (combined_static_adj + combined_static_adj.transpose(0, 1)) / 2


        A = combined_static_adj.unsqueeze(0)  # (1, J, J) 稍后广播

        if self.use_dynamic_adj:

            x_norm = nn.functional.normalize(x, p=2, dim=-1)
            dynamic_adj = torch.matmul(x_norm, x_norm.transpose(1, 2))
            dynamic_adj = self.relu(dynamic_adj) + torch.eye(J, device=x.device).unsqueeze(0)


            gate = torch.sigmoid(self.gate_linear(x))

            A = gate * combined_static_adj.unsqueeze(0) + (1 - gate) * dynamic_adj
        else:
            A = A.expand(B * T, -1, -1)


        deg = A.sum(-1)
        D_inv_sqrt_values = torch.pow(deg + 1e-6, -0.5)
        D_inv_sqrt = torch.diag_embed(D_inv_sqrt_values)


        A_hat = torch.matmul(torch.matmul(D_inv_sqrt, A), D_inv_sqrt)


        h = torch.matmul(A_hat, torch.matmul(x, self.W))
        if self.bias is not None:
            h += self.bias


        h = self.bn(h.permute(0, 2, 1)).permute(0, 2, 1)


        h = self.relu(h)


        if self.use_residual:
            h = h + self.res_proj(x_input)

        output = h.view(B, T, J, self.dim_out)
        return output


DEFAULT_CONNECTIONS = {
    0: [1, 7], 1: [0, 2], 2: [1, 3], 3: [2],
    4: [0, 5], 5: [4, 6], 6: [5],
    7: [0, 8], 8: [7, 9, 11, 14],
    9: [8, 10], 10: [9],
    11: [8, 12], 12: [11, 13], 13: [12],
    14: [8, 15], 15: [14, 16], 16: [15]
}


def build_adj(num_joints, connections=None, add_self=True):
    A = torch.zeros(num_joints, num_joints)
    conn = connections if connections is not None else DEFAULT_CONNECTIONS
    for i, neigh in conn.items():
        for j in neigh:
            A[i, j] = 1.0
            A[j, i] = 1.0
    if add_self:
        A += torch.eye(num_joints)
    return A


def normalize_adj_symmetric(A: torch.Tensor, eps=1e-6):
    # A: (J, J)
    deg = A.sum(-1)  # (J,)
    inv_sqrt = (deg + eps).pow(-0.5)  # (J,)
    D_inv = torch.diag(inv_sqrt)
    return D_inv @ A @ D_inv  # D^{-1/2} A D^{-1/2}


class StructuredSSMTemporalBlock(nn.Module):


    def __init__(
            self,
            dim,
            num_joints,
            ssm_hidden=128,
            heads=4,
            headdim=16,
            mlp_ratio=4.0,
            drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            connections=None,
            use_adaptive_adj=True,
            gate_type='per_joint',
            couple_hidden=True,
            beta_init=0.2
    ):
        super().__init__()
        self.dim = dim
        self.J = num_joints
        self.use_adaptive_adj = use_adaptive_adj
        self.couple_hidden = couple_hidden


        A_base = build_adj(num_joints, connections=connections, add_self=True)  # (J, J)
        self.register_buffer('A_base', A_base)
        if use_adaptive_adj:
            self.A_delta = nn.Parameter(torch.zeros(num_joints, num_joints))
            nn.init.zeros_(self.A_delta)
            self.w_static = nn.Parameter(torch.ones(1))
            self.w_adapt = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('A_delta', None)
            self.register_parameter('w_static', None)
            self.register_parameter('w_adapt', None)


        if gate_type == 'per_joint':
            self.gate_mlp = nn.Sequential(
                nn.Linear(dim, dim // 2),
                act_layer(),
                nn.Linear(dim // 2, 1)
            )
        else:
            self.gate_mlp = nn.Sequential(
                nn.Linear(dim, dim // 2),
                act_layer(),
                nn.Linear(dim // 2, num_joints)
            )
        self.gate_type = gate_type

        # SSM
        self.norm1 = norm_layer(dim)
        self.mamba_fwd = Mamba2(dim, ssm_hidden, heads, headdim=headdim)
        self.mamba_bwd = Mamba2(dim, ssm_hidden, heads, headdim=headdim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)


        if couple_hidden:
            self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))

    def _make_Ahat(self, device):
        A = self.A_base.to(device)
        if self.use_adaptive_adj:

            A_adapt = 0.5 * (self.A_delta + self.A_delta.t())
            A_adapt = torch.nn.functional.softplus(A_adapt)
            A = torch.nn.functional.softplus(self.w_static) * A + \
                torch.nn.functional.softplus(self.w_adapt) * A_adapt

        A_hat = normalize_adj_symmetric(A)
        return A_hat  # (J, J)

    def _precondition_input(self, x, A_hat):
        """
        x: (B, T, J, C)
        A_hat: (J, J)
        return: (B, T, J, C)
        """
        B, T, J, C = x.shape

        x_stat = x.mean(dim=1)
        if self.gate_type == 'per_joint':
            g = torch.sigmoid(self.gate_mlp(x_stat)).view(B, 1, J, 1)  # (B,1,J,1)
        else:
            g = torch.sigmoid(self.gate_mlp(x_stat.mean(dim=1))).view(B, 1, J, 1)  # (B,1,J,1)

        # xg[b,t,j,:] = sum_k A_hat[j,k] * x[b,t,k,:]
        x_spread = torch.einsum('btkc,jk->btjc', x, A_hat)

        x_pre = g * x + (1.0 - g) * x_spread
        return x_pre

    def _couple_hidden(self, y, A_hat):
        """
        y: (B, T, J, C)
        """
        if not self.couple_hidden:
            return y
        y_spread = torch.einsum('btkc,jk->btjc', y, A_hat)
        return y + self.beta * y_spread

    def forward(self, x):
        """
        x: (B, T, J, C)
        """
        B, T, J, C = x.shape
        assert J == self.J and C == self.dim, "Input shape mismatch."

        A_hat = self._make_Ahat(x.device)

        x_in = self._precondition_input(x, A_hat)


        z = rearrange(x_in, 'b t j c -> (b j) t c')
        res = z
        z = self.norm1(z)

        z_f = self.mamba_fwd(z)  # (B*J, T, C)
        z_b = self.mamba_bwd(torch.flip(z, dims=[1]))
        z_b = torch.flip(z_b, dims=[1])  # flip

        z = self.drop_path(z_f + z_b) + res
        z = z + self.drop_path(self.mlp(self.norm2(z)))  # FFN

        y = rearrange(z, '(b j) t c -> b t j c', b=B, j=J)


        y = self._couple_hidden(y, A_hat)

        return y



