import math
import torch
from torch import nn

# 定义骨架连接关系
CONNECTIONS = {10: [9], 9: [8, 10], 8: [7, 9], 14: [15, 8], 15: [16, 14],
               11: [12, 8], 12: [13, 11], 7: [0, 8], 0: [1, 7],
               1: [2, 0], 2: [3, 1], 4: [5, 0], 5: [6, 4],
               16: [15], 13: [12], 3: [2], 6: [5]}





class SpatialGCN(nn.Module):
    def __init__(self, dim_in, dim_out, adj , bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.M = nn.Parameter(torch.ones(size=(adj.size(0), dim_out), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.W = nn.Parameter(torch.zeros(size=(2, dim_in, dim_out), dtype=torch.float))
        self.adj = adj
        self.adj2 = nn.Parameter(torch.ones_like(adj))

        if bias:
            self.bias = nn.Parameter(torch.zeros(dim_out, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def _init_spatial_adj(self):
        adj = torch.zeros((self.num_nodes, self.num_nodes))
        connections = self.connections if self.connections is not None else CONNECTIONS

        for i in range(self.num_nodes):
            connected_nodes = connections[i]
            for j in connected_nodes:
                adj[i, j] = 1
        return adj


    def forward(self, x):

        x = x.permute(0, 2, 1, 3)  # (B, J, T, C)，将 J 移到第 2 维
        x = x.contiguous().view(x.size(0), x.size(1), 4, -1)  # (B, J, 4, T * C / 4)
        x_adj = x.mean(dim=2)  # 对第 2 维（4 分块）求平均，得到 (B, J, T * C / 4)

        x_adj = x.view()

        x_adj2 = x.view(x.size(0), x.size(1), 4, -1)  # (B, J, C*F)
        x_adj2 = x_adj2.mean(dim=2)


        h0 = torch.matmul(x, self.W[0])
        h1 = torch.matmul(x, self.W[1])

        adj = self.adj.to(x.device) + adj2
        adj = (adj + adj.transpose(-1, -2)) / 2


        E = torch.eye(adj.size(0), dtype=torch.float).to(x.device)

        output = torch.matmul(adj * E, self.M * h0) + torch.matmul(adj * (1 - E), self.M * h1)

        if self.bias is not None:
            output = output + self.bias.view(1, 1, -1)


        output = output.unsqueeze(1)  # (B, 1, J, J)

        return output

        # adj = self.adj.to(x.device) + self.adj2.to(x.device)
        # adj = (adj.T + adj) / 2  # 对称化邻接矩阵
        #
        # E = torch.eye(adj.size(0), dtype=torch.float).to(x.device)
        #
        # output = torch.matmul(adj * E, self.M * h0) + torch.matmul(adj * (1 - E), self.M * h1)
        # if self.bias is not None:
        #     return output + self.bias.view(1, 1, -1)
        # else:
        #     return output


