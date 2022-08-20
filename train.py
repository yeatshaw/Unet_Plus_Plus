import indicator
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import cv2 as cv
import numpy as np
from math import floor
"""
/*………………图片展示分类结果………………*/
参数
    Unet：Unet网络
"""
def show_result(Unet):
    img_gray = cv.imread('test_gray.jpg')
    img_gt = cv.imread('test_gt.jpg')
    H = img_gray.shape[0]
    W = img_gray.shape[1]
    img_gray = img_gray[:(int(floor(H/16)*16)),:(int(floor(W/16)*16)),:]
    img_gt = img_gt[:(int(floor(H/16)*16)),:(int(floor(W/16)*16)),:]
    H = img_gray.shape[0]
    W = img_gray.shape[1]
    img_out = np.zeros((H,W*3,3), dtype=np.uint8)
    img_out[:,:W,:] = img_gray
    img_out[:,2*W:3*W,:] = img_gt

    in_put = (torch.tensor(img_gray, dtype=torch.float)[:,:,0].unsqueeze(0).unsqueeze(0)) / 255
    out_put = Unet.forward(in_put)
    img_predict = out_put.round().int().squeeze(0).squeeze(0).numpy() * 255
    for i in range(3):
        img_out[:,W:2*W,i] = img_predict
    cv.imshow('comprison',img_out)
    cv.waitKey(0)

def Unet_train(Unet, Unet_optim, model_type, classify_type, ratio_optim, layer, epoch, train_loader, params_path, show_num=0):
    writer = SummaryWriter('log')
    for e in range(epoch):
        print("Epoch:",e+1)
        for b, (src, trg) in enumerate(train_loader):
            Unet_optim.zero_grad()
            if model_type == 'boost':
                ratio_optim.zero_grad()
            out_put = Unet.forward(src)
            loss = F.binary_cross_entropy(out_put, trg) + torch.mean(1 - 2 * trg.mul(out_put) / (trg.mul(trg) + out_put.mul(out_put)))
            loss.backward()
            Unet_optim.step()
            if model_type == 'boost':
                ratio_optim.step()
                for i in range(3):
                    if Unet.ratio[i] < 0:
                        Unet.ratio[i] = 0

            predict_image = out_put.round().int()
            label = trg.int()
            acc = indicator.UnetAccuracy(predict_image, label)
            mIoU = indicator.mIoU(predict_image, label)
            writer.add_scalar('Accurary/train', acc, e * 279 + b)
            writer.add_scalar('Loss/train', loss, e *279 + b)
            writer.add_scalar('mIoU/train', mIoU, e *279 + b)
            print("Batch:",b+1,"Loss:",loss.detach().numpy(),"Accuracy:",acc.numpy(),"mIoU:",mIoU.numpy())
        torch.save(Unet.state_dict(), params_path)
        if model_type == 'boost':
            torch.save(Unet.ratio, './pretrain/ratio.pt')
        if show_num!=0:
            if (e+1) % show_num == 0 :
                Unet.eval()
                show_result(Unet)
                Unet.train()
    writer.close()

def Unet_test(Unet, test_loader):
    Unet.eval()
    MIOU = []
    BIOU = []
    for b, (src, trg) in enumerate (test_loader):
        out_put = Unet.forward(src)
        predict_image = out_put.round().int()
        label = trg.int()
        mIoU = indicator.mIoU(predict_image, label)
        bIoU = indicator.get_biou(predict_image, label)
        print("Batch:",b+1,"mIoU:",mIoU.numpy(),"bIoU:",bIoU.numpy())
        MIOU.append(mIoU.numpy().tolist())
        BIOU.append(bIoU.numpy().tolist())
    print("mean mIoU:", torch.mean(torch.tensor(MIOU)), "mean bIoU:", torch.mean(torch.tensor(BIOU)))