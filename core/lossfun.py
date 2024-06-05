from torch import nn
import torch
import torch.nn.functional as F
from utils import print_log

def MAE(pred_y, batch_y):
    return nn.L1Loss()(pred_y, batch_y)


def MSE(pred_y, batch_y):
    return nn.MSELoss()(pred_y, batch_y)


def SmoothL1Loss(pred_y, batch_y):
    return nn.SmoothL1Loss()(pred_y, batch_y)


def CEL(pred_y, batch_y):
    return nn.CrossEntropyLoss()(pred_y, batch_y)


def Log(pred_y, batch_y):
    return nn.LogSoftmax()(pred_y, batch_y)


class Regularization(torch.nn.Module):
    def __init__(self,model,weight_decay,p=2,dist=False):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=2为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            raise ValueError("param weight_decay can not <=0")
        self.dist = dist
        self.check_dist(model)
        self.weight_decay=weight_decay
        self.p=p
        self.weight_list=self.get_weight(self.model)
 
    def to(self,device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device=device
        super().to(device)
        return self
 
    def forward(self, model):
        self.check_dist(model)
        self.weight_list=self.get_weight(self.model)#获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, self.p)
        return reg_loss
 
    def check_dist(self, model):
        if self.dist:
            self.model=model.module
        else:
            self.model=model

    def get_weight(self,model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list
 
    def regularization_loss(self,weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        reg_loss=0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
 
        reg_loss=weight_decay*reg_loss
        return reg_loss

