import torch
import torch.nn as nn
import numpy as np


def get_pad(in_, ksize, stride, atrous=1):
    out_ = np.ceil(float(in_) / stride)
    return int(((out_ - 1) * stride + atrous * (ksize - 1) + 1 - in_) / 2)


class SNConv(torch.nn.Module):
    """
    SN convolution for spetral normalization conv
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(SNConv, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.activation = activation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        # print('input', input.shape)
        x = self.conv2d(input)
        # print('output', x.shape)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x
