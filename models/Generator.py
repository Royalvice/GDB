import torch
import torch.nn as nn
from models.GateConv import GateConv, GateDeConv
from models.SNConv import get_pad
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, same_shape=True, **kwargs):
        super(Residual, self).__init__()
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = GateConv(in_channels, 2 * in_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = GateConv(in_channels, out_channels, kernel_size=3, padding=1)
        if not same_shape:
            self.conv3 = GateConv(in_channels, out_channels, kernel_size=1,
                                  stride=strides)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        if not self.same_shape:
            x = self.conv3(x)
        out = out + x
        return F.relu(out)


class Generator(nn.Module):
    """
    Generator Using Gate Convolution
    """

    def __init__(self, input_c):
        super(Generator, self).__init__()
        # corase downsample
        self.c = 64
        # corase d 1st
        self.corase_a1_ = GateConv(input_c, 2 * self.c, kernel_size=5, stride=1, padding=2)
        self.corase_a1 = GateConv(self.c, 2 * self.c, kernel_size=3, stride=2, padding=1)
        self.corase_a2 = GateConv(self.c, 2 * self.c, kernel_size=3, stride=1, padding=1)
        self.corase_a3 = GateConv(self.c, 2 * self.c, kernel_size=3, stride=1, padding=1)
        # corase res block
        self.res1 = Residual(self.c, 2 * self.c, same_shape=False)
        self.res2 = Residual(self.c, 2 * self.c)
        self.res3 = Residual(self.c, 4 * self.c, same_shape=False)
        self.res4 = Residual(2 * self.c, 4 * self.c)
        self.res5 = Residual(2 * self.c, 8 * self.c, same_shape=False)
        # self.nn = ConvWithActivation(256, 512, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2))
        self.res6 = Residual(4 * self.c, 8 * self.c)
        self.res7 = Residual(4 * self.c, 16 * self.c, same_shape=False)
        self.res8 = Residual(8 * self.c, 16 * self.c)
        # corase upsample
        # corase u 1st
        self.corase_c1 = GateDeConv(8 * self.c, 8 * self.c, kernel_size=3, stride=1, padding=1)
        self.corase_c2 = GateDeConv(8 * self.c, 4 * self.c, kernel_size=3, stride=1, padding=1)
        self.corase_c3 = GateDeConv(4 * self.c, 2 * self.c, kernel_size=3, stride=1, padding=1)
        self.corase_c4 = GateDeConv(2 * self.c, 2 * self.c, kernel_size=3, stride=1, padding=1)
        self.corase_c5 = GateDeConv(2 * self.c, 2, kernel_size=3, stride=1, padding=1, activation=torch.sigmoid)

        self.corase_c1s = GateDeConv(8 * self.c, 8 * self.c, kernel_size=3, stride=1, padding=1)
        self.corase_c2s = GateDeConv(8 * self.c, 4 * self.c, kernel_size=3, stride=1, padding=1)
        self.corase_c3s = GateDeConv(4 * self.c, 2 * self.c, kernel_size=3, stride=1, padding=1)
        self.corase_c4s = GateDeConv(2 * self.c, 2 * self.c, kernel_size=3, stride=1, padding=1)
        self.corase_c5s = GateDeConv(2 * self.c, 2, kernel_size=3, stride=1, padding=1, activation=torch.sigmoid)

        # refine network
        self.re_a1 = GateConv(4, 2 * self.c, kernel_size=5, stride=1, padding=2)
        self.re_a2 = GateConv(self.c, 4 * self.c, kernel_size=3, stride=2, padding=1)
        self.re_a3 = GateConv(2 * self.c, 4 * self.c, kernel_size=3, stride=1, padding=1)
        self.re_a4 = GateConv(2 * self.c, 8 * self.c, kernel_size=3, stride=2, padding=1)
        self.re_a5 = GateConv(4 * self.c, 8 * self.c, kernel_size=3, stride=1, padding=1)
        self.re_a6 = GateConv(4 * self.c, 8 * self.c, kernel_size=3, stride=1, padding=1)
        self.re_a7 = GateConv(4 * self.c, 8 * self.c, kernel_size=3, stride=1, padding=1)
        self.dialation_conv1 = nn.Sequential(
            GateConv(4 * self.c, 8 * self.c, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            GateConv(4 * self.c, 8 * self.c, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
            GateConv(4 * self.c, 8 * self.c, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
            GateConv(4 * self.c, 8 * self.c, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16)),
        )
        self.re_b1 = GateConv(4 * self.c, 8 * self.c, kernel_size=3, stride=1, padding=1)
        self.re_b2 = GateConv(8 * self.c, 8 * self.c, kernel_size=3, stride=1, padding=1)
        self.re_b3 = GateConv(4 * self.c, 8 * self.c, kernel_size=3, stride=1, padding=1)
        self.re_b4 = GateDeConv(8 * self.c, 4 * self.c, kernel_size=3, stride=1, padding=1)
        self.re_b5 = GateConv(4 * self.c, 4 * self.c, kernel_size=3, stride=1, padding=1)
        self.re_b6 = GateDeConv(4 * self.c, 2 * self.c, kernel_size=3, stride=1, padding=1)
        self.re_b7 = GateConv(1 * self.c, 2 * self.c, kernel_size=3, stride=1, padding=1)
        self.re_b8 = GateConv(1 * self.c, 2, kernel_size=3, stride=1, padding=1, activation=torch.sigmoid)
        self.skip01 = nn.Sequential(
            nn.Conv2d(self.c, 2 * self.c, kernel_size=1, stride=1, padding=0),
            GateConv(2 * self.c, 8 * self.c, kernel_size=3, stride=1, padding=1),
            GateConv(4 * self.c, 16 * self.c, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(8 * self.c, 4 * self.c, kernel_size=1, stride=1, padding=0)
        )
        self.skip02 = nn.Sequential(
            nn.Conv2d(self.c, 2 * self.c, kernel_size=1, stride=1, padding=0),
            GateConv(2 * self.c, 8 * self.c, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(4 * self.c, 2 * self.c, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, ori, ostu, sobel, gray, ori_full, ostu_full, sobel_full, img_x, img_y, img_size):
        img_input = torch.cat((ori, ostu, sobel), 1)
        skip_connections = []
        x = self.corase_a1_(img_input)
        x = self.corase_a1(x)
        x = self.corase_a2(x)
        x = self.corase_a3(x)
        skip_connections.append(x)
        x = self.res1(x)
        x = self.res2(x)
        skip_connections.append(x)
        x = self.res3(x)
        x = self.res4(x)
        skip_connections.append(x)
        x = self.res5(x)
        x = self.res6(x)
        skip_connections.append(x)
        x = self.res7(x)
        x = self.res8(x)

        feature_temp = x

        x = self.corase_c1(x)
        x = torch.cat((x, skip_connections[3]), 1)
        x = self.corase_c2(x)
        x = torch.cat((x, skip_connections[2]), 1)
        x = self.corase_c3(x)
        x = torch.cat((x, skip_connections[1]), 1)
        x = self.corase_c4(x)
        x = torch.cat((x, skip_connections[0]), 1)
        x = self.corase_c5(x)
        corase_out = x
        y = self.corase_c1s(feature_temp)
        y = torch.cat((y, skip_connections[3]), 1)
        y = self.corase_c2s(y)
        y = torch.cat((y, skip_connections[2]), 1)
        y = self.corase_c3s(y)
        y = torch.cat((y, skip_connections[1]), 1)
        y = self.corase_c4s(y)
        y = torch.cat((y, skip_connections[0]), 1)
        y = self.corase_c5s(y)
        edge_out = y

        # img ori
        img_ori_input = torch.cat((ori_full, ostu_full, sobel_full), 1)
        skip_connections_ori = []
        x_full = self.corase_a1_(img_ori_input)
        x_full = self.corase_a1(x_full)
        x_full = self.corase_a2(x_full)
        x_full = self.corase_a3(x_full)
        skip_connections_ori.append(x_full)
        x_full = self.res1(x_full)
        x_full = self.res2(x_full)
        skip_connections_ori.append(x_full)
        x_full = self.res3(x_full)
        x_full = self.res4(x_full)
        skip_connections_ori.append(x_full)
        x_full = self.res5(x_full)
        x_full = self.res6(x_full)
        skip_connections_ori.append(x_full)
        x_full = self.res7(x_full)
        x_full = self.res8(x_full)
        x_full = self.corase_c1(x_full)
        x_full = torch.cat((x_full, skip_connections_ori[3]), 1)
        x_full = self.corase_c2(x_full)
        x_full = torch.cat((x_full, skip_connections_ori[2]), 1)
        x_full = self.corase_c3(x_full)
        x_full = torch.cat((x_full, skip_connections_ori[1]), 1)
        x_full = self.corase_c4(x_full)
        x_full = torch.cat((x_full, skip_connections_ori[0]), 1)
        x_full = self.corase_c5(x_full)
        corase_out_ori = x_full
        x_full = F.interpolate(x_full, (img_size[0], img_size[1]), mode='bilinear')
        corase_out_ori_full = x_full
        img_x, img_y = int(img_x[0]), int(img_y[0])
        x = torch.cat((gray, corase_out, x_full[:, :, img_x*256:img_x*256 + 256, img_y*256:img_y*256 + 256], y), 1)
        x = self.re_a1(x)
        x = self.re_a2(x)
        x = self.re_a3(x)
        skip_connections.append(x)
        x = self.re_a4(x)
        x = self.re_a5(x)
        x = self.re_a6(x)
        x = self.re_a7(x)
        skip_connections.append(x)
        x = self.dialation_conv1(x)
        x = self.re_b1(x)
        x = torch.cat((x, self.skip01(skip_connections[1])), 1)
        x = self.re_b2(x)
        x = self.re_b3(x)
        x = torch.cat((x, skip_connections[5]), 1)
        x = self.re_b4(x)
        x = torch.cat((x, self.skip02(skip_connections[0])), 1)
        x = self.re_b5(x)
        x = torch.cat((x, skip_connections[4]), 1)
        x = self.re_b6(x)
        x = self.re_b7(x)
        x = self.re_b8(x)
        return corase_out, corase_out_ori, corase_out_ori_full, edge_out, x


# test = Generator(5)
# x = torch.ones((1, 3, 256, 256))
# ostu = torch.ones((1, 1, 256, 256))
# sobel = torch.ones((1, 1, 256, 256))
# x = test(x, ostu, sobel, sobel)
