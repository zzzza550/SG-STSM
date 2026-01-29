from collections import OrderedDict
import math
import torch
from torch import nn
from timm.models.layers import DropPath


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
        self.mamba_temporal = TemporalMambaBlock(dim, mlp_ratio, drop, drop_path=drop_path)

        # GCN-Spatial
        # self.gcn_spatial = GCNSpatialBlock(dim, dim, num_nodes = 17, bias=True)

        self.use_adaptive_fusion = use_adaptive_fusion
        if self.use_adaptive_fusion:
            self.fusion = nn.Linear(dim * 2, 2)
            self._init_fusion()

    def _init_fusion(self):
        self.fusion.weight.data.fill_(0)
        self.fusion.bias.data.fill_(0.5)

    def forward(self, x):
        """
        x: tensor with shape [B, T, J, C]
        """
        # # if self.hierarchical:
        # B, T, J, C = x.shape
        # x_mamba =  x.shape

        # x_graph = self.graph_temporal(self.graph_spatial(x_graph + x_att))
        # x_mamba = self.mamba_spatial(self.mamba_temporal(x))
        # x_att = self.att_temporal(self.att_spatial(x))
        # ///////////////////////////////////////////////////////
        # x_att = self.att_temporal(self.att_spatial(x))
        # y = x_att
        # x_mamba = self.mamba_temporal(self.mamba_spatial(y))
        #
        # # alpha = self.fusion(x_mamba)
        # # alpha = alpha.softmax(dim=-1)
        # z = x_mamba
        #
        # return z

        if self.hierarchical:
            B, T, J, C = x.shape
            # x_att, x_graph = x[..., :C // 2], x[..., C // 2:]
            x_att, x_mamba = x[..., :C // 2], x[..., C // 2:]

            x_att = self.att_temporal(self.att_spatial(x_att))
            # x_graph = self.graph_temporal(self.graph_spatial(x_graph + x_att))
            x_mamba = self.mamba_temporal(self.mamba_spatial(x_mamba + x_att))
        else:
            # x_gcn = self.gcn_spatial(x)
            x_att = self.att_temporal(self.att_spatial(x))
            # x_graph = self.graph_temporal(self.mamba_spatial(x))
            x_mamba = self.mamba_temporal(self.mamba_spatial(x))

        if self.hierarchical:
            x = torch.cat((x_att, x_mamba), dim=-1)
        elif self.use_adaptive_fusion:
            alpha = torch.cat((x_att, x_mamba), dim=-1)
            alpha = self.fusion(alpha)
            alpha = alpha.softmax(dim=-1)
            # # 计算权重的平均值
            # alpha_att_mean = alpha[..., 0].mean().item()  # x_att 权重均值
            # alpha_mamba_mean = alpha[..., 1].mean().item()  # x_mamba 权重均值
            #
            # # 输出结果
            # if alpha_att_mean > alpha_mamba_mean:
            #     print(f"x_att 的整体占比更大：{alpha_att_mean:.4f} vs {alpha_mamba_mean:.4f}")
            # else:
            #     print(f"x_mamba 的整体占比更大：{alpha_mamba_mean:.4f} vs {alpha_att_mean:.4f}")
            x = x_att * alpha[..., 0:1] + x_mamba * alpha[..., 1:2]
        # else:
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


class TemporalMambaBlock(nn.Module):

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
        self.attn = Mamba2(dim, 128, 4, headdim=16)
        self.attn_inv = Mamba2(dim, 128, 4, headdim=16)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if self.changedim and self.currentdim < self.depth // 2:
            self.reduction = nn.Conv1d(dim, dim // 2, kernel_size=1)
            # self.reduction = nn.Linear(dim, dim//2)
        elif self.changedim and depth > self.currentdim > self.depth // 2:
            self.improve = nn.Conv1d(dim, dim * 2, kernel_size=1)
            # self.improve = nn.Linear(dim, dim*2)
        self.vis = vis

    def forward(self, x, vis=False):
        assert len(x.shape) == 4, 'input shape must 4'
        b, f, j, c = x.shape
        x = rearrange(x, 'b f j c -> (b j) f c')
        res = x
        x = self.norm1(x)
        x_inv = torch.flip(x, dims=[-1])
        x = self.attn(x)
        x_inv = self.attn_inv(x_inv)
        x_fuse = x + x_inv
        x = self.drop_path(x_fuse)
        if isinstance(x, tuple):
            x = x[0]
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
        x = rearrange(x, '(b j) f c -> b f j c', b=b, j=j)
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

        #x_inv = torch.flip(x, dims=[-1])
        x = self.attn(x)  # Mamba
        #x_inv = self.attn_inv(x_inv)
        #x_fuse = x + x_inv
        x = self.drop_path(x)
        if isinstance(x, tuple):
            x = x[0]
        #x = x_fuse + res
        x = x + res
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.changedim and self.currentdim < self.depth // 2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.reduction(x)
            x = rearrange(x, 'b c t -> b t c')
            # print("tn 1")
        elif self.changedim and self.depth > self.currentdim > self.depth // 2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.improve(x)
            x = rearrange(x, 'b c t -> b t c')
            # print("tn 1")
        x = rearrange(x, '(b f) j c -> b f j c', b=b, f=f)
        return x


class GCNSpatialBlock(nn.Module):  # Structural Reinforcement Learning Module
    def __init__(self, dim_in, dim_out, num_nodes, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_nodes = num_nodes
        self.adj = torch.zeros((self.num_nodes, self.num_nodes))
        self.M = nn.Parameter(torch.ones(size=(self.adj.size(0), dim_out), dtype=torch.float))
        self.W = nn.Parameter(torch.zeros(size=(2, dim_in, dim_out), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(num_nodes)

        CONNECTIONS = {10: [9], 9: [8, 10], 8: [7, 9], 14: [15, 8], 15: [16, 14], 11: [12, 8], 12: [13, 11],
                       7: [0, 8], 0: [1, 7], 1: [2, 0], 2: [3, 1], 4: [5, 0], 5: [6, 4], 16: [15], 13: [12], 3: [2],
                       6: [5]}

        for i in range(self.num_nodes):
            connected_nodes = CONNECTIONS[i]
            for j in connected_nodes:
                self.adj[i, j] = 1

        self.adj2 = nn.Parameter(torch.ones_like(self.adj))
        nn.init.constant_(self.adj2, 1e-6)

        if bias:
            self.bias = nn.Parameter(torch.zeros(dim_out, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

        # self.w3 = nn.Parameter(torch.ones(size=(dim_out, dim_out//4 ), dtype=torch.float))
        #
        # self.FC = nn.Linear(dim_out // 4*243, 17)

    def init_adj3(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()  # (b, f, j ,c)->(b, j, f, c)

        x = torch.matmul(x, self.w3)  #
        x = x.view(x.size(0), x.size(1), -1)
        x = self.FC(x)

        return x

    def forward(self, x):

        # adj3=self.init_adj3(x).unsqueeze(1).repeat(1, x.size(1), 1,1)

        # x = x.permute(0, 2, 1, 3)  # (B, J, T, C)，将 J 移到第 2 维
        # x = x.contiguous().view(x.size(0), x.size(1), 4, -1)  # (B, J, 4, T * C / 4)
        # self.adj = x.mean(dim=2)  # 对第 2 维（4 分块）求平均，得到 (B, J, T * C / 4)
        # self.adj2 = x.mean(dim=2)
        #
        # x_adj2 = x.view(x.size(0), x.size(1), 4, -1)  # (B, J, C*F)
        # x_adj2 = x_adj2.mean(dim=2) adj( 3 , 1 , 17 ,17 )

        h0 = torch.matmul(x, self.W[0])
        h1 = torch.matmul(x, self.W[1])

        # adj = self.adj.to(x.device) + self.adj2.to(x.device) + adj3.to(x.device)
        adj = self.adj.to(x.device) + self.adj2.to(x.device)
        # adj = (adj.transpose(-1,-2) + adj) / 3

        adj = (adj.T + adj) / 2
        # adj = self.adj.to(x.device) + adj2
        # adj = (adj + adj.transpose(-1, -2)) / 2

        E = torch.eye(adj.size(0), dtype=torch.float).to(x.device)

        output = torch.matmul(adj * E, self.M * h0) + torch.matmul(adj * (1 - E), self.M * h1)

        if self.bias is not None:
            output = output + self.bias.view(1, 1, -1)

        # output = output.permute(0, 2, 1, 3).contiguous()
        #
        # output = self.relu(self.bn(output))
        #
        # output = output.permute(0, 2, 1, 3)

        return (output * 1e-9)


def _test():
    from torchprofile import profile_macs
    import warnings
    print("Torch version:", torch.__version__)
    print("Torch location:", torch.__file__)



    warnings.filterwarnings('ignore')
    b, c, t, j = 1, 3, 27, 17
    random_x = torch.randn((b, t, j, c)).to('cuda')

    model = MSTPFormer(n_layers=13, dim_in=3, dim_feat=128, mlp_ratio=4, hierarchical=False,
                       n_frames=t).to('cuda')
    model.eval()

    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print(f"Model parameter #: {model_params:,}")
    print(f"Model FLOPS #: {profile_macs(model, random_x):,}")

    # Warm-up to avoid timing fluctuations
    for _ in range(10):
        _ = model(random_x)

    import time
    num_iterations = 100
    # Measure the inference time for 'num_iterations' iterations
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(random_x)
    end_time = time.time()

    # Calculate the average inference time per iteration
    average_inference_time = (end_time - start_time) / num_iterations

    # Calculate FPS
    fps = 1.0 / average_inference_time

    print(f"FPS: {fps}")

    out = model(random_x)

    assert out.shape == (b, t, j, 3), f"Output shape should be {b}x{t}x{j}x3 but it is {out.shape}"
def te():
    import warnings
    from thop import profile
    print("Torch version:", torch.__version__)
    print("Torch location:", torch.__file__)
    warnings.filterwarnings('ignore')

    b, c, t, j = 1, 3, 27, 17
    random_x = torch.randn((b, t, j, c)).cuda()

    model = MSTPFormer(n_layers=13, dim_in=3, dim_feat=128, mlp_ratio=4, hierarchical=False,
                       n_frames=t).to('cuda')

    # 参数量
    model_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameter #: {model_params:,}")

    # FLOPs/MACs 统计
    macs, params = profile(model, inputs=(random_x,), verbose=False)
    print(f"MACs #: {macs:,}")
    print(f"Params #: {params:,}")

    # 估算 FPS
    for _ in range(10):  # warm-up
        _ = model(random_x)

    import time
    num_iterations = 100
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(random_x)
    end_time = time.time()

    avg_time = (end_time - start_time) / num_iterations
    fps = 1.0 / avg_time
    print(f"FPS: {fps:.2f}")

    out = model(random_x)
    assert out.shape == (b, t, j, 3), f"Output shape mismatch: {out.shape}"



if __name__ == '__main__':
    #_test()
    te()

