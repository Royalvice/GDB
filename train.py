import os
import argparse
import torch
import cv2
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader
from data.dataloder import DocData
from loss.Loss import Loss_Doc
from models.Generator import Generator
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.empty_cache()
parser = argparse.ArgumentParser()
parser.add_argument('--numOfWorkers', type=int, default=0,
                    help='workers for dataloader')
parser.add_argument('--modelsSavePath', type=str, default='./checkSave/D2019',
                    help='path for saving models')
parser.add_argument('--batchSize', type=int, default=1)
parser.add_argument('--loadSize', type=int, default=256,
                    help='image loading size')
parser.add_argument('--dataRoot', type=str,
                    default=r'')
parser.add_argument('--pretrained', type=str, default=r'', help='pretrained models for finetuning')
parser.add_argument('--num_epochs', type=int, default=60, help='epochs')
parser.add_argument('--only_one', type=bool, default=False, help='only one')
parser.add_argument('--testRoot', type=str, default=r'')
parser.add_argument('--testSave', type=str, default=r'./results/D2019')
parser.add_argument('--ganLoss', type=bool, default=True)
args = parser.parse_args()

#torch.cuda.set_per_process_memory_fraction(0.7, 0)
def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    return cv_img


def cv_imwrite(filename, src):
    cv2.imencode('.tiff', src)[1].tofile(filename)


def init__result_Dir():
    work_dir = os.path.join(os.getcwd(), 'Training')
    max_model = 0
    for root, j, file in os.walk(work_dir):
        for dirs in j:
            try:
                temp = int(dirs)
                if temp > max_model:
                    max_model = temp
            except:
                continue
        break
    max_model += 1
    path = os.path.join(work_dir, str(max_model))
    os.mkdir(path)
    return path

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1028)
cuda = torch.cuda.is_available()
if cuda:
    print('Cuda is available!')
    cudnn.enable = True
    cudnn.benchmark = True
batchSize = args.batchSize
loadSize = (args.loadSize, args.loadSize)
if not os.path.exists(args.modelsSavePath):
    os.makedirs(args.modelsSavePath)
dataRoot = args.dataRoot

Doc_data = DocData(dataRoot, loadSize, training=False, only_one=args.only_one)
Doc_data = DataLoader(Doc_data, batch_size=batchSize, shuffle=True, num_workers=args.numOfWorkers, drop_last=False,
                      pin_memory=True)

generator = Generator(5)

if args.pretrained != '':
    print('loaded ')
    generator.load_state_dict(torch.load(args.pretrained))

if cuda:
    generator = generator.to('cuda')

G_optimizer = optim.Adam(generator.parameters(), lr=0.00003)
loss_doc = Loss_Doc(lr=0.00001, ganLoss=args.ganLoss)

if cuda:
    loss_doc = loss_doc.cuda()
print('Datasets:', len(Doc_data.dataset))
num_epochs = args.num_epochs
path1 = init__result_Dir()
for epoch in range(1, num_epochs + 1):
    generator.train()

    for num, (inputImg, ostu, sobel, gt, gt_Sobel, gray, imgOri, ostuOri, sobelOri, gtOri, gtOri_fullsize, img_x, img_y, imgOriSize, name) in enumerate(Doc_data):
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
        generator.zero_grad()
        corase_out, corase_out_ori, corase_out_ori_full, edge_out, refine_out = generator(inputImg, ostu, sobel, gray, imgOri, ostuOri, sobelOri, img_x, img_y, imgOriSize)
        l1_loss, cross_entropy_loss, mask_loss, D_loss_c_all, D_loss_c_full, D_loss_c, D_real_c_full, D_fake_c_full_, D_real_c, D_fake_c_, D_real_r, D_fake_r_ = loss_doc(inputImg, corase_out, edge_out, refine_out, gt, gt_Sobel, corase_out_ori, corase_out_ori_full, gtOri, gtOri_fullsize, imgOri)
        G_loss = 0.1 * (D_fake_c_ + D_fake_c_full_ + D_fake_r_) + cross_entropy_loss + 10 * l1_loss + mask_loss
        G_loss = G_loss.sum()
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()
        print('[{}/{}] Generator Loss of epoch{} is {:.4f}  D_real_c_full:{:.4f}   D_fake_c_full:{:.4f}   D_real_c:{:.4f}   D_fake_c:{:.4f}    D_real_r:{:.4f}   D_fake_r:{:.4f}    mask_loss:{:.4f}'
              'cross_loss:{:.4f} '
              'l1_loss:{:.4f}'.format(num, len(Doc_data.dataset) // batchSize, epoch, G_loss.item(), D_real_c_full.sum(

        ).item(), D_fake_c_full_.sum().item(), D_real_c.sum().item(), D_fake_c_.sum().item(), D_real_r.sum().item(), D_fake_r_.sum().item(), mask_loss.sum().item(), cross_entropy_loss.sum().item(), l1_loss.sum().item()))
        if num % 1000 == 0:
            outt = torch.cat((gray, ostu, sobel, gt, corase_out,  refine_out, corase_out_ori_full[:, :, int(img_x[0])*256:int(img_x[0])*256 + 256, int(img_y[0])*256:int(img_y[0])*256 + 256], gtOri, corase_out_ori, edge_out, gt_Sobel), 3)
            outt = outt.transpose(1, 2).transpose(2, 3).detach().cpu().numpy()
            outt = outt[0].astype(np.float32)
            cv_imwrite(os.path.join(path1, str(epoch)+'_'+str(num)+name[0]+'.tiff'), outt)
    if epoch % 10 == 0:
        torch.save(generator.state_dict(), args.modelsSavePath + '/Net_{}.pth'.format(epoch))