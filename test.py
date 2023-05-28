import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from data.dataloder import DocData
from models.Generator import Generator
from utils.utils import cv_imwrite

parser = argparse.ArgumentParser()
parser.add_argument('--numOfWorkers', type=int, default=0,
                    help='workers for dataloader')
parser.add_argument('--batchSize', type=int, default=1, help='batch size')
parser.add_argument('--loadSize', type=int, default=256,
                    help='image loading size')
parser.add_argument('--dataRoot', type=str,
                    default='', help='path for test data')
parser.add_argument('--pretrained', type=str, default='',
                    help='pretrained models')
parser.add_argument('--savePath', type=str, default='./results', help='path for saving results')
parser.add_argument('--only_one', type=bool, default=True, help='only one')
args = parser.parse_args()
cuda = torch.cuda.is_available()
if not os.path.exists('./results'):
    os.makedirs('./results')

if cuda:
    print('Cuda is available!')
    cudnn.benchmark = True
batchSize = args.batchSize
loadSize = (args.loadSize, args.loadSize)
dataRoot = args.dataRoot
savePath = args.savePath
corase_out_path = os.path.join(savePath, r'corase_out')
refine_out_path = os.path.join(savePath, r'refine_out')
interm = os.path.join(savePath, r'interm')
if not os.path.exists(savePath):
    os.makedirs(savePath)
    os.makedirs(corase_out_path)
    os.makedirs(refine_out_path)
    os.makedirs(interm)
Doc_data = DocData(dataRoot, loadSize, training=False, only_one=args.only_one)
Doc_data = DataLoader(Doc_data, batch_size=batchSize, shuffle=True, num_workers=args.numOfWorkers, drop_last=False,
                      pin_memory=True)
netG = Generator(5)
netG.load_state_dict(torch.load(args.pretrained))
if cuda:
    netG = netG.cuda()

for param in netG.parameters():
    param.requires_grad = False
print('ok')
netG.eval()
for num, (
inputImg, ostu, sobel, gt, gt_Sobel, gray, imgOri, ostuOri, sobelOri, gtOri, gtOri_fullsize, img_x, img_y, imgOriSize,
name) in enumerate(Doc_data):
    if cuda:
        inputImg = inputImg.cuda()
        ostu = ostu.cuda()
        sobel = sobel.cuda()
        gt = gt.cuda()
        gray = gray.cuda()
        imgOri = imgOri.cuda()
        ostuOri = ostuOri.cuda()
        sobelOri = sobelOri.cuda()
        gtOri = gtOri.cuda()
        gtOri_fullsize = gtOri_fullsize.cuda()
        gt_Sobel = gt_Sobel.cuda()
    corase_out, corase_out_ori, corase_out_ori_full, edge_out, refine_out = netG(inputImg, ostu, sobel, gray, imgOri,
                                                                                 ostuOri, sobelOri, img_x, img_y,
                                                                                 imgOriSize)
    corase_out = corase_out.data.cpu()
    refine_out = refine_out.data.cpu()
    print(num)
    print(name)
    save_image(corase_out, os.path.join(corase_out_path, name[0]))
    save_image(refine_out, os.path.join(refine_out_path, name[0]))
    outt = torch.cat((gray.data.cpu(), ostu.cpu(), sobel.cpu(), gt.data.cpu(), corase_out, refine_out,
                      corase_out_ori_full[:, :, int(img_x[0]) * 256:int(img_x[0]) * 256 + 256,
                      int(img_y[0]) * 256:int(img_y[0]) * 256 + 256].data.cpu(), gtOri.data.cpu(),
                      corase_out_ori.data.cpu(), edge_out.data.cpu(), gt_Sobel.data.cpu()), 3)
    outt = outt.transpose(1, 2).transpose(2, 3).detach().cpu().numpy()
    outt = outt[0].astype(np.float32)
    cv_imwrite(os.path.join(interm, name[0]), outt)
