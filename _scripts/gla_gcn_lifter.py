import torch
import torch.nn as nn

# This is the core GLA-GCN model architecture, adapted for our use case.
# It defines the structure that the pre-trained weights will be loaded into.


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class GCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(GCN_Block, self).__init__()
        self.gcn1 = GCN(in_channels, out_channels, A)
        self.tcn1 = TCN(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TCN(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(GCN, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.A_size = A.size()
        self.A = nn.Parameter(A.clone())
        self.conv_d = nn.ModuleList()
        for i in range(self.A_size[0]):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x):
        N, C, T, V = x.shape
        y = None
        for i in range(self.A_size[0]):
            y_t = x.clone().permute(0, 3, 1, 2).contiguous().view(N, V, C * T)
            y_t = (
                self.conv_d[i](y_t.permute(0, 2, 1).contiguous().view(N, C * T, V, 1))
                .permute(0, 2, 1)
                .contiguous()
                .view(N, V, C, T)
            )
            y_t = torch.einsum(
                "n c t v, v w -> n c t w", (y_t.permute(0, 2, 3, 1), self.A[i])
            )
            if y is None:
                y = y_t.unsqueeze(-1)
            else:
                y = torch.cat([y, y_t.unsqueeze(-1)], -1)
        y = y.sum(-1).permute(0, 1, 3, 2).contiguous()
        return y


class TCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(TCN, self).__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, A, num_joints=17):
        super(Model, self).__init__()
        self.num_joints = num_joints
        self.data_bn = nn.BatchNorm1d(in_channels * self.num_joints)
        bn_init(self.data_bn, 1)
        self.gcn_blocks = nn.ModuleList(
            [
                GCN_Block(in_channels, 64, A),
                GCN_Block(64, 64, A),
                GCN_Block(64, 64, A),
                GCN_Block(64, 64, A),
            ]
        )
        self.fc = nn.Linear(64, out_channels)

    def forward(self, x):
        N, C, T, V = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        for gcn in self.gcn_blocks:
            x = gcn(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.fc(x)
        x = x.view(x.size(0), T, self.num_joints, -1)
        return x
