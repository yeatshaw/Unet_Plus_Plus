import os
import cv2 as cv
import torch
from torch.utils.data import Dataset, DataLoader
from math import floor

class Mydata(Dataset):
    def __init__(self, data_path, label_path):
        self.data = []
        self.label = []
        dirs = os.listdir(data_path)
        for d in dirs:
            img = cv.imread(data_path + '/' + d)
            img_tensor = (torch.tensor(img, dtype=torch.float)[:,:,0].unsqueeze(0)) / 255
            img_tensor = img_tensor[:,:(int(floor(img_tensor.shape[1]/16)*16)),:(int(floor(img_tensor.shape[2]/16)*16))]
            self.data.append(img_tensor)
        for d in dirs:
            img = cv.imread(label_path + '/' + d)
            img_tensor = (torch.tensor(img, dtype=torch.float)[:,:,0].unsqueeze(0)) / 255
            img_tensor = img_tensor[:,:(int(floor(img_tensor.shape[1]/16)*16)),:(int(floor(img_tensor.shape[2]/16)*16))]
            self.label.append(img_tensor)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
"""
/*………………获得数据集加载器………………*/
参数
    train_data_path：训练数据路径
    train_label_path：训练数据标签路径
    test_data_path：测试数据路径
    test_label_path：测试数据集标签路径
    batchsize=1：batch的大小
    shuf=True：每一次迭代是否打乱数据集
"""
def LoadData(train_data_path, train_label_path, test_data_path, test_label_path, batchsize=1, shuf=True):
    train_dataset = Mydata(train_data_path, train_label_path)
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=shuf)
    test_dataset = Mydata(test_data_path, test_label_path)
    test_loader = DataLoader(test_dataset, batch_size=batchsize)
    return train_loader, test_loader