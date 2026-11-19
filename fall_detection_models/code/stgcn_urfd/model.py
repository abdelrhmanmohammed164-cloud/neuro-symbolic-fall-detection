"""
model.py
ST-GCN neural stream for fall detection, following the paper's Section 3.3.1:

  Eq. 2  coordinate stream  X  = {(x, y)}  per joint and frame
  Eq. 3  motion stream      dX = P_t - P_{t-1},  0 at t = 1   (backward difference, causal)
  Eq. 4  graph: spatial body edges + temporal edges between the same joint in consecutive frames
  Eq. 5  h = phi(h_sp || h_tm),  h_sp = g_sp(X; G),  h_tm = g_tm(dX; G)
  Eq. 6  p_N = sigma(w^T h + b)

Not given in the paper (engineering choices, documented):
  - depth / width of g_sp and g_tm: 6 ST-GCN blocks each, 32 -> 64 -> 128 channels (small data)
  - phi: Linear(256 -> 128) + ReLU + Dropout(0.5)
  - dropout 0.25 inside blocks, learnable edge importance
  - missing joints: the motion of a joint is 0 unless the joint is valid in both frames

Input: (N, C, T, V, M) with channels x/W, y/H, conf (conf = 0 marks a missing joint).
By default only (x, y) enter the coordinate stream, as in Eq. 2; --use_conf adds conf as a 3rd channel.
Output: (N,) logits.
"""
import torch
import torch.nn as nn

from graph import Graph


class GraphConv(nn.Module):
    """Spatial graph convolution over K adjacency partitions (Yan et al., 2018)."""

    def __init__(self, in_c, out_c, K):
        super().__init__()
        self.K = K
        self.conv = nn.Conv2d(in_c, out_c * K, kernel_size=1)

    def forward(self, x, A):                        # x (N, C, T, V), A (K, V, V)
        n, _, t, v = x.shape
        x = self.conv(x).view(n, self.K, -1, t, v)
        return torch.einsum("nkctv,kvw->nctw", x, A).contiguous()


class STGCNBlock(nn.Module):
    def __init__(self, in_c, out_c, K, stride=1, dropout=0.25, residual=True, t_kernel=9):
        super().__init__()
        pad = (t_kernel - 1) // 2
        self.gcn = GraphConv(in_c, out_c, K)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, (t_kernel, 1), (stride, 1), (pad, 0)),   # temporal edges
            nn.BatchNorm2d(out_c), nn.Dropout(dropout, inplace=True))
        if not residual:
            self.res = None
        elif in_c == out_c and stride == 1:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(nn.Conv2d(in_c, out_c, 1, (stride, 1)), nn.BatchNorm2d(out_c))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        out = self.tcn(self.gcn(x, A))
        if self.res is not None:
            out = out + self.res(x)
        return self.relu(out)


class STGCNBackbone(nn.Module):
    """g(.; G): stack of ST-GCN blocks + global average pooling -> (N, C_out)."""

    def __init__(self, in_channels, A, channels=(32, 32, 64, 64, 128, 128), strides=(1, 1, 2, 1, 2, 1),
                 block_dropout=0.25, edge_importance=True):
        super().__init__()
        K, V = A.shape[0], A.shape[1]
        self.data_bn = nn.BatchNorm1d(in_channels * V)
        blocks, c_in = [], in_channels
        for i, (c, s) in enumerate(zip(channels, strides)):
            blocks.append(STGCNBlock(c_in, c, K, s, block_dropout, residual=(i > 0)))
            c_in = c
        self.blocks = nn.ModuleList(blocks)
        self.edge_imp = nn.ParameterList([nn.Parameter(torch.ones_like(A)) for _ in blocks]
                                         if edge_importance else [])
        self.out_channels = c_in

    def forward(self, x, A):                        # (N, C, T, V, M)
        n, c, t, v, m = x.shape
        x = x.permute(0, 4, 3, 1, 2).reshape(n * m, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).reshape(n * m, c, t, v)
        for i, blk in enumerate(self.blocks):
            x = blk(x, A * self.edge_imp[i] if len(self.edge_imp) else A)
        return x.mean(dim=(2, 3)).view(n, m, -1).mean(dim=1)


def motion_stream(x):
    """Eq. 3: dP_t = P_t - P_{t-1} for t >= 2, 0 at t = 1. x: (N, C>=3, T, V, M).
    A joint's motion is 0 unless it is valid (conf > 0) in both frames."""
    xy, valid = x[:, :2], (x[:, 2:3] > 0)
    d = torch.zeros_like(xy)
    both = valid[:, :, 1:] & valid[:, :, :-1]
    d[:, :, 1:] = torch.where(both, xy[:, :, 1:] - xy[:, :, :-1], torch.zeros_like(xy[:, :, 1:]))
    return d


class TwoStreamSTGCN(nn.Module):
    def __init__(self, use_conf=False, channels=(32, 32, 64, 64, 128, 128), strides=(1, 1, 2, 1, 2, 1),
                 block_dropout=0.25, head_dropout=0.5, fusion_dim=128):
        super().__init__()
        A = torch.tensor(Graph().A)
        self.register_buffer("A", A)
        self.use_conf = use_conf
        self.g_sp = STGCNBackbone(3 if use_conf else 2, A, channels, strides, block_dropout)
        self.g_tm = STGCNBackbone(2, A, channels, strides, block_dropout)
        h = self.g_sp.out_channels + self.g_tm.out_channels
        self.phi = nn.Sequential(nn.Linear(h, fusion_dim), nn.ReLU(inplace=True), nn.Dropout(head_dropout))
        self.cls = nn.Linear(fusion_dim, 1)                        # w^T h + b  (Eq. 6, sigmoid in the loss)

    def streams(self, x):
        coord = x[:, :3] if self.use_conf else x[:, :2]
        return coord, motion_stream(x)

    def forward(self, x):
        coord, motion = self.streams(x)
        h = torch.cat([self.g_sp(coord, self.A), self.g_tm(motion, self.A)], dim=1)   # Eq. 5
        return self.cls(self.phi(h)).squeeze(-1)


if __name__ == "__main__":
    m = TwoStreamSTGCN()
    x = torch.rand(4, 3, 80, 14, 1)
    print("logits", m(x).shape, "params", sum(p.numel() for p in m.parameters()))
