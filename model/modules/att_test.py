from torch import nn
import torch

class Attention(nn.Module):
    """
    A simplified version of attention from DSTFormer that also considers x tensor to be (B, T, J, C) instead of
    (B * T, J, C)
    """

    def __init__(self, dim_in, dim_out, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 mode='spatial',n_frames = 243, dim_feat=128, num_joints=17):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_in // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_in, dim_out)
        self.mode = mode
        self.qkv = nn.Linear(dim_in, dim_in * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_map = None

        self.attn_spatial = None
        self.attn_temporal = None
        self.step = 0
        self.vis_every = 20



    def forward(self, x):
        B, T, J, C = x.shape
        #x = x + self.temp_embed  #

        # x = x.reshape(-1, J, C)
        # BT = x.shape[0]
        # x = self.proj(x)
        # x = x + self.pos_embed
        #
        # _, J, C = x.shape
        # x = x.reshape(-1, T, J, C) + self.temp_embed[:,:T,:,:]
        # x = x.reshape(BT, J, C)
        # x = self.proj_drop(x)

        qkv = self.qkv(x).reshape(B, T, J, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2,
                                                                                           5)  # (3, B, H, T, J, C)
        if self.mode == 'temporal':
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = self.forward_temporal(q, k, v)

        elif self.mode == 'spatial':
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = self.forward_spatial(q, k, v)
        else:
            raise NotImplementedError(self.mode)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def forward_spatial(self, q, k, v):
        B, H, T, J, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, J, J)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        self.attn_map = attn
        self.attn_spatial = attn.detach()



        x = attn @ v  # (B, H, T, J, C)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, T, J, C * self.num_heads)
        return x  # (B, T, J, C)


    def forward_temporal(self, q, k, v):
        B, H, T, J, C = q.shape
        qt = q.transpose(2, 3)  # (B, H, J, T, C)
        kt = k.transpose(2, 3)  # (B, H, J, T, C)
        vt = v.transpose(2, 3)  # (B, H, J, T, C)

        attn = (qt @ kt.transpose(-2, -1)) * self.scale  # (B, H, J, T, T)
        attn = attn.softmax(dim=-1)

        self.step += 1
        if self.step % 20 == 0:  # 既然已经训练好，可以频繁看图
            with torch.no_grad():
                # 1. 提取第一个 Batch
                # 2. 对 Head 维度取平均 -> 得到 (J, T_query, T_key)
                attn_vis = attn[0].mean(dim=0)

                # 3. 按照你的要求：
                # 纵轴 (Query): 第 8 个关节 (J=8)，第 0 帧 (T=0)
                # 横轴 (Key): 所有帧 (0:243)
                # 结果形状应该是 (243,)
                target_slice = attn_vis[8, 0, :].cpu().numpy()

                import matplotlib.pyplot as plt
                import numpy as np
                import seaborn as sns

                # 将其转换为 (1, 243) 形状以便 Heatmap 渲染
                plot_data = np.expand_dims(target_slice, axis=0)

                plt.figure(figsize=(15, 2))  # 长条形状

                # 移除所有可能导致混淆的变换，直接画出这一行
                sns.heatmap(
                    plot_data,
                    cmap="viridis",
                    xticklabels=20,
                    yticklabels=["J8-F0"],
                    cbar=True,
                    robust=True  # 保证颜色对比度
                )

                plt.xlabel("Key Frame (0-243)")
                plt.ylabel("Query Position")
                plt.title(f"Attention View: Joint 8 at Frame 0")

                plt.tight_layout()
                plt.savefig(f"hotmap/slice_j8_f0_step_{self.step}.png")
                plt.close()



        x = attn @ vt  # (B, H, J, T, C)
        x = x.permute(0, 3, 2, 1, 4).reshape(B, T, J, C * self.num_heads)
        return x  # (B, T, J, C)
