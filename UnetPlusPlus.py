import torch
import torch.nn as nn

#………………基础的VGG块，用于构成Unet++网络………………
class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out = self.relu(out)

        return out

"""
/*………………Unet++网络………………*/
参数
    input_channels：输入图片的通道大小
    model_type：Unet++使用的模型
    classify_type：结果采用的分类策略
    layer：剪枝模型的Unet层数
"""
class UnetPlusPlus(nn.Module):
    def __init__(self, input_channels, ratio, model_type='boost', classify_type = 'sigmoid', layer=4):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]
        self.layer = layer
        self.ratio = ratio
        self.model_type = model_type
        self.classify_type = classify_type
        self.ratio.requires_grad = True
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down = nn.MaxPool2d(2,2)
        
        #第一斜列（左上到右下）
        self.block0_0 = VGGBlock(input_channels, nb_filter[0])
        self.block1_0 = VGGBlock(nb_filter[0], nb_filter[1])
        self.block2_0 = VGGBlock(nb_filter[1], nb_filter[2])
        self.block3_0 = VGGBlock(nb_filter[2], nb_filter[3])
        self.block4_0 = VGGBlock(nb_filter[3], nb_filter[4])
        #第二斜列
        self.block0_1 = VGGBlock(nb_filter[0] * 1 + nb_filter[1], nb_filter[0])
        self.block1_1 = VGGBlock(nb_filter[1] * 1 + nb_filter[2], nb_filter[1])
        self.block2_1 = VGGBlock(nb_filter[2] * 1 + nb_filter[3], nb_filter[2])
        self.block3_1 = VGGBlock(nb_filter[3] * 1 + nb_filter[4], nb_filter[3])
        #第三斜列
        self.block0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.block1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.block2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])
        #第四斜列
        self.block0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.block1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])
        #第五斜列
        self.block0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])
        if self.classify_type == 'sigmoid':
            self.final1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
            self.classify = nn.Sigmoid() 
        elif self.classify_type == 'softmax':
            self.final1 = nn.Conv2d(nb_filter[0], 2, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], 2, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], 2, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], 2, kernel_size=1)
            self.classify = nn.Softmax(dim=1) 

    def forward(self, x):
        out_put = []
        result = 0 
        layer = self.layer
        if layer > 0:
            x0_0 = self.block0_0(x)
            x1_0 = self.block1_0(self.down(x0_0))
            x0_1 = self.block0_1(torch.cat((x0_0, self.up(x1_0)), 1))
            out_put.append(self.classify(self.final1(x0_1)))
            layer = layer - 1
        if layer > 0:
            x2_0 = self.block2_0(self.down(x1_0))
            x1_1 = self.block1_1(torch.cat((x1_0, self.up(x2_0)), 1))
            x0_2 = self.block0_2(torch.cat((x0_0, x0_1, self.up(x1_1)), 1))
            out_put.append(self.classify(self.final2(x0_2)))
            layer = layer - 1
        if layer > 0:
            x3_0 = self.block3_0(self.down(x2_0))
            x2_1 = self.block2_1(torch.cat((x2_0, self.up(x3_0)), 1))
            x1_2 = self.block1_2(torch.cat((x1_0, x1_1, self.up(x2_1)), 1))
            x0_3 = self.block0_3(torch.cat((x0_0, x0_1, x0_2, self.up(x1_2)), 1))
            out_put.append(self.classify(self.final3(x0_3)))
            layer = layer - 1
        if layer > 0:
            x4_0 = self.block4_0(self.down(x3_0))
            x3_1 = self.block3_1(torch.cat((x3_0, self.up(x4_0)), 1))
            x2_2 = self.block2_2(torch.cat((x2_0, x2_1, self.up(x3_1)), 1))
            x1_3 = self.block1_3(torch.cat((x1_0, x1_1, x1_2, self.up(x2_2)), 1))
            x0_4 = self.block0_4(torch.cat((x0_0, x0_1, x0_2, x0_3, self.up(x1_3)), 1))
            out_put.append(self.classify(self.final4(x0_4)))
            layer = layer - 1

        if self.model_type == 'boost':
            if self.classify_type == 'sigmoid':
                for i in range(3):
                    result = result + self.ratio[i] * out_put[i]
                result = result + (1 - torch.sum(self.ratio)) * out_put[3]
            elif self.classify_type == 'softmax':
                for i in range(3):
                    result = result + self.ratio[i] * out_put[i][:,1,:,:]
                result = result + (1 - torch.sum(self.ratio)) * out_put[3][:,1,:,:]
        elif self.model_type == 'single':
            if self.classify_type == 'sigmoid':
                result = out_put[self.layer - 1]
            elif self.classify_type == 'softmax':
                result = out_put[self.layer - 1][:,1,:,:].unsqueeze(0)
                print(result)
        return result