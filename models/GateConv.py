import torch
import torch.nn as nn
import torch.nn.functional as F


class GateConv(torch.nn.Module):
    """
    Gate Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2)):
        super(GateConv, self).__init__()
        self.gate01 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.activation = activation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        #print('input', input.shape)
        x = self.gate01(input)
        if self.activation is None:
            return x
        x, y = torch.chunk(x, 2, 1)
        y = torch.sigmoid(y)
        x = self.activation(x)
        x = x * y
        #print('output', x.shape)
        return x


class GateDeConv(torch.nn.Module):
    """
    Gate Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2)):
        super(GateDeConv, self).__init__()
        self.gate01 = GateConv(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, activation=activation)

    def forward(self, input):
        x = F.interpolate(input, scale_factor=2, mode='bilinear')
        x = self.gate01(x)
        return x