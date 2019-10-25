#-*-coding:utf-8-*-
from MNSIM.Interface import quantize
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections

class LeNet(nn.Module):
    def __init__(self, hardware_config, num_classes):
        super(LeNet, self).__init__()
        # hardware_config = {'fix_method': 'SINGLE_FIX_TEST', 'xbar_size': 512, 'input_bit': 2, 'weight_bit': 1, 'quantize_bit': 10}
        # hardware_config = {'fix_method': 'FIX_TRAIN', 'xbar_size': 512, 'input_bit': 2, 'quantize_bit': 10}
        quantize_config = {'weight_bit': 9, 'activation_bit': 9, 'point_shift': -2}
        # conv layer 1
        c1_layer_config = {'type': 'conv', 'in_channels': 3, 'out_channels': 6, 'kernel_size': 5}
        self.c1 = quantize.QuantizeLayer(hardware_config, c1_layer_config, quantize_config)
        self.relu1 = nn.ReLU()
        self.s1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # conv layer 2
        c2_layer_config = {'type': 'conv', 'in_channels': 6, 'out_channels': 16, 'kernel_size': 5}
        self.c2 = quantize.QuantizeLayer(hardware_config, c2_layer_config, quantize_config)
        self.relu2 = nn.ReLU()
        self.s2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # conv layer 3
        c3_layer_config = {'type': 'conv', 'in_channels': 16, 'out_channels': 120, 'kernel_size': 5}
        self.c3 = quantize.QuantizeLayer(hardware_config, c3_layer_config, quantize_config)
        self.relu3 = nn.ReLU()
        # fc layer 4
        fc4_layer_config = {'type': 'fc', 'in_features': 120, 'out_features': 84}
        self.fc4 = quantize.QuantizeLayer(hardware_config, fc4_layer_config, quantize_config)
        self.relu4 = nn.ReLU()
        # fc layer 4
        fc5_layer_config = {'type': 'fc', 'in_features': 84, 'out_features': num_classes}
        self.fc5 = quantize.QuantizeLayer(hardware_config, fc5_layer_config, quantize_config)

    def forward(self, x):
        # 对输入进行尺寸和定点位置的说明
        quantize.last_activation_scale = 1. / 255.
        quantize.last_activation_bit = 9
        # 前向计算
        x = self.s1(self.relu1(self.c1(x)))
        x = self.s2(self.relu2(self.c2(x)))
        x = self.relu3(self.c3(x))
        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc4(x))
        x = self.fc5(x)
        return x
    def get_weight(self):
        net_bit_weights = collections.OrderedDict()
        net_bit_weights['c1'] = self.c1.get_weight()
        net_bit_weights['c2'] = self.c2.get_weight()
        net_bit_weights['c3'] = self.c3.get_weight()
        net_bit_weights['fc4'] = self.fc4.get_weight()
        net_bit_weights['fc5'] = self.fc5.get_weight()
        return net_bit_weights
    def set_weight_forward(self, x, net_bit_weights):
        # 对输入进行尺寸和定点位置的说明
        quantize.last_activation_scale = 1. / 255.
        quantize.last_activation_bit = 9
        # 前向计算
        x = self.s1(self.relu1(self.c1.set_weight_forward(x, net_bit_weights['c1'])))
        x = self.s2(self.relu2(self.c2.set_weight_forward(x, net_bit_weights['c2'])))
        x = self.relu3(self.c3.set_weight_forward(x, net_bit_weights['c3']))
        x = x.view(x.size(0), -1)
        x = self.relu4(self.fc4.set_weight_forward(x, net_bit_weights['fc4']))
        x = self.fc5.set_weight_forward(x, net_bit_weights['fc5'])
        return x

def get_net(hardware_config = None):
    # initial config
    if hardware_config == None:
        hardware_config = {'fix_method': 'SINGLE_FIX_TEST', 'xbar_size': 512, 'input_bit': 2, 'weight_bit': 1, 'quantize_bit': 10}
    net = LeNet(hardware_config, 10)
    return net

if __name__ == '__main__':
    net = get_net()
    print(net)
    for param in net.parameters():
        print(type(param.data), param.size())
    print('这是LeNet网络，要求输入尺寸必为3x32x32，输出为10维分类结果')
