import torch
from torch import nn
from torch.nn import functional as F


class NLTimeBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded',
                 dimension=3, parametric=False):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLTimeBlock, self).__init__()

        assert dimension in [1, 2, 3]

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.parametric = parametric
        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        batch_size = x.size(0)
        _,C,T,H,W=x.size()

        # (N, CHW,T)

        g_x = x.permute(0, 1, 3, 4, 2).reshape(batch_size, self.inter_channels * H * W, T)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            if self.parametric:
                theta_x = self.theta(x).permute(0,1,3,4,2).reshape(batch_size, self.inter_channels*H*W, T)
                phi_x = self.phi(x).permute(0,1,3,4,2).reshape(batch_size, self.inter_channels*H*W, T)
            else:
                theta_x = x.permute(0,1,3,4,2).reshape(batch_size, self.inter_channels*H*W, T)
                phi_x = x.permute(0,1,3,4,2).reshape(batch_size, self.inter_channels*H*W, T)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            if self.parametric:
                theta_x = self.theta(x).permute(0,1,3,4,2).reshape(batch_size, self.inter_channels*H*W, T)
                phi_x = self.phi(x).permute(0,1,3,4,2).reshape(batch_size, self.inter_channels*H*W, T)
            else:
                theta_x = x.permute(0,1,3,4,2).reshape(batch_size, self.inter_channels*H*W, T)
                phi_x = x.permute(0,1,3,4,2).reshape(batch_size, self.inter_channels*H*W, T)
            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)

        # contiguous here just allocates contiguous chunk of memory
        # T,CHW -> CHW,T
        y = y.permute(0, 2, 1).contiguous()
        y = y.reshape(batch_size, self.inter_channels, H,W,T).permute(0,1,4,2,3)

        # residual connection
        z = y + x

        return z