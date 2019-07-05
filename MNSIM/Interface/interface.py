#!/home/sunhanbo/software/anaconda3/bin/python
#-*-coding:utf-8-*-
import math
import numpy as np
import quantize
import torch
import torch.nn as nn
import collections
from importlib import import_module
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--net', help = 'select net')
parser.add_argument('-w', '--weight', help = 'select weight file')
args = parser.parse_args()
assert args.net
assert args.weight

# crossbar的尺寸，PE的尺寸指成对出现的crossbar数量，BANK尺寸指BANK二维大小
crossbar_size = 128
PE_size = 8
BANK_size = (4, 4)

# net
net_module = import_module(args.net)
activation_bit = net_module.activation_bit
weight_bit = net_module.weight_bit
net = net_module.get_net()
net.load_state_dict(torch.load(args.weight))

# transfer
ASIC_ARRAY = []
module_define = collections.OrderedDict()
input_size = 32
with torch.no_grad():
    for name, module in net.named_modules():
        if isinstance(module, nn.MaxPool2d):
            output_size = input_size / 2
        else:
            output_size = input_size
        if isinstance(module, quantize.QuantizeConv2d) or isinstance(module, quantize.QuantizePowerConv2d):
            weight = module.conv2d.weight
            scale = torch.max(torch.abs(weight))
            thres = 2 ** (weight_bit - 1) - 1
            weight.div_(scale).mul_(thres).round_()
            weight = weight.reshape(weight.size(0), -1).t()
            # layer_define
            tmp_define = collections.OrderedDict()
            tmp_define['Name'] = 'Conv'
            tmp_define['Inputsize'] = input_size
            tmp_define['Outputsize'] = output_size
            tmp_define['Kernelsize'] = module.conv2d.kernel_size[0]
            tmp_define['Stride'] = module.conv2d.stride[0]
            tmp_define['Inputchannel'] = module.conv2d.in_channels
            tmp_define['Outputchannel'] = module.conv2d.out_channels
            tmp_define['Inputbit'] = activation_bit
            tmp_define['Weightbit'] = weight_bit
            tmp_define['Outputbit'] = activation_bit
            module_define[name] = tmp_define
            print('generate %s' % name)
            # test
            # weight = torch.arange(49.).reshape((7,7)) - 30
            # weight_bit = 7
            # split bit
            weight_list = []
            sign_weight = torch.sign(weight)
            value_weight = torch.abs(weight)
            for i in range(weight_bit - 1):
                weight_list.append(torch.fmod(value_weight, 2))
                value_weight.div_(2).floor_()
            # add sign
            sign_weight_list = []
            for tmp in weight_list:
                positive = torch.mul(tmp, (sign_weight == 1).float())
                negative = torch.mul(tmp, (sign_weight == -1).float())
                sign_weight_list.append([positive, negative])
            # split channel
            H = weight.size(0)
            W = weight.size(1)
            H_range = math.ceil(H / (BANK_size[0] * crossbar_size))
            W_range = math.ceil(W / (BANK_size[1] * crossbar_size))
            CrossArray = []
            for h in range(H_range):
                for w in range(W_range):
                    base_h = h * BANK_size[0] * crossbar_size
                    base_w = w * BANK_size[1] * crossbar_size
                    total_h = min(H, (h+1)*BANK_size[0]*crossbar_size) - base_h
                    total_w = min(W, (w+1)*BANK_size[1]*crossbar_size) - base_w
                    # BANK内部
                    split_h_range = math.ceil(total_h/crossbar_size)
                    split_w_range = math.ceil(total_w/crossbar_size)
                    for split_h in range(split_h_range):
                        for split_w in range(split_w_range):
                            PE_array = []
                            split_base_h = base_h + split_h*crossbar_size
                            split_base_w = base_w + split_w*crossbar_size
                            total_split_h = min(H, split_base_h+crossbar_size) - split_base_h
                            total_split_w = min(W, split_base_w+crossbar_size) - split_base_w
                            # 写入新的矩阵
                            for i in range(weight_bit - 1):
                                tmp_positive = sign_weight_list[i][0][split_base_h:(split_base_h+total_split_h),split_base_w:(split_base_w+total_split_w)].clone()
                                tmp_negative = sign_weight_list[i][1][split_base_h:(split_base_h+total_split_h),split_base_w:(split_base_w+total_split_w)].clone()
                                PE_array.append([tmp_positive.numpy().astype(np.uint8), tmp_negative.numpy().astype(np.uint8)])
                            CrossArray.append(PE_array)
            A = math.ceil(len(CrossArray) / (BANK_size[0]*BANK_size[1]))
            print(A)
            for i in range(A):
                BANK_array = []
                for j in range(BANK_size[0]):
                    ROW_array = []
                    for k in range(BANK_size[1]):
                        index = i * (BANK_size[0]*BANK_size[1]) + j*(BANK_size[1])+k
                        if index > len(CrossArray) - 1:
                            continue
                        ROW_array.append(CrossArray[index])
                    BANK_array.append(ROW_array)
                ASIC_ARRAY.append(BANK_array)
        input_size = output_size

    torch.save(ASIC_ARRAY, './zoo/mnsim_weight.pt')
    torch.save(module_define, './zoo/mnsim_net.pt')
