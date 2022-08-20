import os
import torch
import torch.optim as optim 
import cv2 as cv
import UnetPlusPlus

from torch.utils.data import DataLoader
from dataset import LoadData
from train import Unet_train, Unet_test

"""
/*………………模型选择………………*/
参数
    model_type_list：包含两种模型——集成模型（boost）、剪枝模型（single)
    model_type：0为集成模型，1为剪枝模型
    classify_type_list：分类策略——softmax、sigmoid
    classify_type：0为softmax分类，1为sigmoid分类
    pretrain：是否使用预训练模型
    layer_num：仅在选择剪枝模型才有用
    epoch：迭代次数
    show_num：每n个epoch展示一次图片结果，若n为0则不展示
"""
if __name__ == '__main__':
    model_type_list = ['boost','single']
    model_type = 0
    classify_type_list = ['softmax','sigmoid']
    classify_type = 1
    pretrain = True
    layer_num = 4
    epoch = 100
    show_num = 0

    if model_type == 0:
        layer_num = 4
    ratio=torch.tensor([0.1, 0.15 ,0.25])
    
    params_path = './pretrain/' + model_type_list[model_type] + '_' + classify_type_list[classify_type] + str(layer_num) + '.pth'
    if pretrain:
        Unet_params = torch.load(params_path, map_location='cpu')
        if model_type == 0:
            ratio = torch.load('./pretrain/ratio.pt', map_location='cpu')
    Unet = UnetPlusPlus.UnetPlusPlus(1, ratio,  model_type_list[model_type], classify_type_list[classify_type], layer_num)
    if pretrain:
        Unet.load_state_dict(Unet_params)

#………………加载数据………………
    train_loader, test_loader = LoadData("./data/train_data", "./data/train_label", "./data/test_data", "./data/test_label")
#………………初始化优化器………………
    Unet_optim = optim.Adam(Unet.parameters(), lr=0.001, betas=(0.9,0.999))
    ratio_optim = optim.Adam([Unet.ratio], lr=0.001, betas=(0.9,0.999))
#………………训练………………
    #Unet_train(Unet, Unet_optim, model_type_list[model_type], classify_type_list[classify_type], ratio_optim, layer_num, epoch, train_loader, params_path, show_num)
#………………测试………………
    Unet_test(Unet, test_loader)