import torch
import torch.nn as nn
from models.SNConv import SNConv
import numpy as np


def get_pad(in_, ksize, stride, atrous=1):
    out_ = np.ceil(float(in_) / stride)
    return int(((out_ - 1) * stride + atrous * (ksize - 1) + 1 - in_) / 2)


class Discriminator(nn.Module):
    def __init__(self, inputChannels):
        super(Discriminator, self).__init__()
        cnum = 32
        self.discriminator = nn.Sequential(
            SNConv(inputChannels, 2 * cnum, 5, 1, padding=2),
            SNConv(2 * cnum, 2 * cnum, 3, 2, padding=get_pad(256, 4, 2)),
            SNConv(2 * cnum, 4 * cnum, 3, 2, padding=get_pad(128, 4, 2)),
            SNConv(4 * cnum, 8 * cnum, 3, 2, padding=get_pad(64, 4, 2)),
            SNConv(8 * cnum, 16 * cnum, 3, 2, padding=get_pad(32, 4, 2)),
            SNConv(16 * cnum, 16 * cnum, 3, 2, padding=get_pad(16, 4, 2)),
            SNConv(16 * cnum, 16 * cnum, 3, 2, padding=get_pad(8, 4, 2)),
        )

        self.shrink = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4),
            nn.Sigmoid()
        )

    def forward(self, input, gray):
        all_feat = torch.cat((input, gray), 1)
        all_feat = self.discriminator(all_feat)
        return self.shrink(all_feat).view(input.shape[0], -1)

# test = Discriminator(4)
# ori = torch.ones((1, 3, 256, 256))
# x = torch.ones((1, 1, 256, 256))
# x = test(x, ori)
# print(x)
