import cv2 as cv
import torch
import torch.nn.functional as F
import numpy as np
"""
/*………………获得边界………………*/
参数
    pic：需要获得边界的图像
    is_mask：判断pic是否分好类，若分好则为True，未分则为False
"""
def get_boundary(pic,is_mask):
    if not is_mask:
        pic = torch.argmax(pic,1).cpu().numpy().astype('float64')
    else:
        pic = pic.squeeze(1).cpu().numpy()
    batch, width, height = pic.shape
    new_pic = np.zeros([batch, width + 2, height + 2])
    mask_erode = np.zeros([batch, width, height])
    dil = int(round(0.02*np.sqrt(width ** 2 + height ** 2)))
    if dil < 1:
        dil = 1
    for i in range(batch):
        new_pic[i] = cv.copyMakeBorder(pic[i], 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    for j in range(batch):
        pic_erode = cv.erode(new_pic[j],kernel,iterations=dil)
        mask_erode[j] = pic_erode[1: width + 1, 1: height + 1]
    return torch.from_numpy(pic-mask_erode)
"""
/*………………计算Boundary IoU………………*/
参数
    pre_pic：预测图像
    real_pic：ground true
"""
def get_biou(pre_pic ,real_pic):
    inter = 0
    union = 0
    pre_pic = get_boundary(pre_pic, is_mask=True)
    real_pic = get_boundary(real_pic, is_mask=True)
    batch, width, height = pre_pic.shape
    for i in range(batch):
        predict = pre_pic[i]
        mask = real_pic[i]
        inter += ((predict * mask) > 0).sum()
        union += ((predict + mask) > 0).sum()
    if union < 1:
        return 0
    biou = (inter/union)
    return biou
"""
/*………………计算分类正确率………………*/
参数
    predict：预测图像
    label：ground true
"""
def UnetAccuracy(predict, label):
    temp = torch.zeros_like(label,dtype=torch.float)
    temp[predict == label] = 1
    return torch.mean(temp)
"""
/*………………计算mIoU………………*/
参数
    predict：预测图像
    label：ground true
"""
def mIoU(predict, label):
    Jiao = torch.sum(predict & label)
    Bing = torch.sum(predict | label)
    return Jiao/Bing