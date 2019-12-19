import torch

import torch.nn as nn

import torch.nn.functional as F

class BasicConv1d(nn.Module):

    def __init__(self, C_in, C_out, **kwargs):

        super().__init__()
        self.conv = nn.Conv1d(C_in, C_out, **kwargs)
        self.bn = nn.BatchNorm1d(C_out)

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

    
class Inception(nn.Module):

    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):

        super().__init__()

        self.b1 = BasicConv1d(in_planes, n1x1, kernel_size=1)
        self.b2_1x1_a = BasicConv1d(in_planes, n3x3red, kernel_size=1)
        self.b2_3x3_b = BasicConv1d(n3x3red, n3x3, kernel_size=3, padding=1)
        self.b3_1x1_a = BasicConv1d(in_planes, n5x5red, kernel_size=1)
        self.b3_3x3_b = BasicConv1d(n5x5red, n5x5, kernel_size=3, padding=1)
        self.b3_3x3_c = BasicConv1d(n5x5, n5x5, kernel_size=3, padding=1)
        self.b4_pool = nn.MaxPool1d(3, stride=1, padding=1)
        self.b4_1x1 = BasicConv1d(in_planes, pool_planes, kernel_size=1)

    def forward(self, x):

        y1 = self.b1(x)
        y2 = self.b2_3x3_b(self.b2_1x1_a(x))
        y3 = self.b3_3x3_c(self.b3_3x3_b(self.b3_1x1_a(x)))
        y4 = self.b4_1x1(self.b4_pool(x))
        return torch.cat([y1, y2, y3, y4], 1)


