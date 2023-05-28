import torch
import torch.nn as nn
from models.Discriminator import Discriminator
import numpy as np
import torch.nn.functional as F


def dice_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1)

    input = input
    target = target

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    dice_loss = torch.mean(d)
    return 1 - dice_loss


def is_white(img):
    return not np.any(1 - np.array(img.cpu()))


class Loss_Doc(nn.Module):
    def __init__(self, lr=0.00001, ganLoss=True):
        super(Loss_Doc, self).__init__()
        self.ganLoss = ganLoss
        self.l1 = nn.L1Loss()
        self.cross_entropy = nn.BCELoss()
        if self.ganLoss:
            self.discriminator_c = Discriminator(4)
            self.discriminator_r = Discriminator(4)
            self.D_optimizer_c = torch.optim.Adam(self.discriminator_c.parameters(), lr=lr)
            self.D_optimizer_r = torch.optim.Adam(self.discriminator_r.parameters(), lr=lr)

    def forward(self, img, corase_out, edge_out, re_out, gt, gt_Sobel, corase_out_ori, corase_out_ori_full, gt_ori, gt_ori_full, img_ori):
        l1_loss = self.l1(corase_out, gt) + 2 * self.l1(re_out, gt) + 0.5*(self.l1(
            corase_out_ori, gt_ori) + self.l1(corase_out_ori_full, gt_ori_full)) + self.l1(edge_out, gt_Sobel)
        cross_entropy_loss = self.cross_entropy(corase_out, gt) + 2 * self.cross_entropy(re_out,
                                                                                         gt) + 0.5*(self.cross_entropy(corase_out_ori, gt_ori) + self.cross_entropy(corase_out_ori_full, gt_ori_full)) + self.cross_entropy(edge_out, gt_Sobel)
        if is_white(gt):
            flag_white = True
            mask_loss = dice_loss(corase_out, gt) + 2 * dice_loss(re_out, gt) + dice_loss(1 - edge_out, 1 - gt_Sobel) + 0.5*(dice_loss(
            corase_out_ori, gt_ori) + dice_loss(corase_out_ori_full, gt_ori_full))
        else:
            flag_white = False
            mask_loss = dice_loss(1 - corase_out, 1 - gt) + 2 * dice_loss(1 - re_out, 1 - gt) + dice_loss(edge_out,
                                                                                          gt_Sobel) + 0.5 * (
                                    dice_loss(
                                        1 - corase_out_ori, 1 - gt_ori) + dice_loss(1 - corase_out_ori_full, 1 - gt_ori_full))
        if self.ganLoss:
            self.discriminator_c.zero_grad()
            D_real_c_full = self.discriminator_c(gt_ori, img_ori)
            D_real_c_full = D_real_c_full.mean().sum() * -1
            D_fake_c_full = self.discriminator_c(corase_out_ori, img_ori)
            D_fake_c_full = D_fake_c_full.mean().sum() * 1
            D_loss_c_full = torch.mean(F.relu(1. + D_real_c_full)) + torch.mean(F.relu(1. + D_fake_c_full))
            if not flag_white:
                D_real_c = self.discriminator_c(gt, img)
                D_real_c = D_real_c.mean().sum() * -1
                D_fake_c = self.discriminator_c(corase_out, img)
                D_fake_c = D_fake_c.mean().sum() * 1
                D_loss_c = torch.mean(F.relu(1. + D_real_c)) + torch.mean(F.relu(1. + D_fake_c))

                self.discriminator_r.zero_grad()
                D_real_r = self.discriminator_r(gt, img)
                D_real_r = D_real_r.mean().sum() * -1
                D_fake_r = self.discriminator_r(re_out, img)
                D_fake_r = D_fake_r.mean().sum() * 1
                D_loss_r = torch.mean(F.relu(1. + D_real_r)) + torch.mean(F.relu(1. + D_fake_r))
                self.D_optimizer_r.zero_grad()
                D_loss_r.backward(retain_graph=True)
                self.D_optimizer_r.step()
                D_fake_r_ = self.discriminator_r(re_out, img)
                D_fake_r_ = -torch.mean(D_fake_r_)  # Generator loss
            else:
                D_fake_r_ = torch.Tensor([0]).cuda()
                D_loss_c = torch.Tensor([0]).cuda()
                D_real_c = torch.Tensor([0]).cuda()
                D_real_r = torch.Tensor([0]).cuda()

            self.D_optimizer_c.zero_grad()
            D_loss_c_all = D_loss_c_full + D_loss_c
            D_loss_c_all.backward(retain_graph=True)
            self.D_optimizer_c.step()
            D_fake_c_full_ = self.discriminator_c(corase_out_ori, img_ori)
            D_fake_c_full_ = -torch.mean(D_fake_c_full_)
            if not flag_white:
                D_fake_c_ = self.discriminator_c(corase_out, img)
                D_fake_c_ = -torch.mean(D_fake_c_)
            else:
                D_fake_c_ = torch.Tensor([0]).cuda()
        else:
            D_loss_c_all, D_loss_c_full, D_loss_c, D_real_c_full, D_fake_c_full_, D_real_c, D_fake_c_, D_real_r, D_fake_r_ = torch.Tensor([0]).cuda(),torch.Tensor([0]).cuda(),torch.Tensor([0]).cuda(),torch.Tensor([0]).cuda(),torch.Tensor([0]).cuda(),torch.Tensor([0]).cuda(),torch.Tensor([0]).cuda(),torch.Tensor([0]).cuda(),torch.Tensor([0]).cuda()

        return l1_loss, cross_entropy_loss, mask_loss, D_loss_c_all, D_loss_c_full, D_loss_c, D_real_c_full, D_fake_c_full_, D_real_c, D_fake_c_, D_real_r, D_fake_r_
