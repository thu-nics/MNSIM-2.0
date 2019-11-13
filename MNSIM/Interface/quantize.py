#-*-coding:utf-8-*-
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np
import math
import copy
import collections

# last_activation_scale, last_weight_scale for activation and weight scale in last calculation
# last_activation_bit, last_weight_bit for activation and weight bit last time
global last_activation_scale
global last_weight_scale
global last_activation_bit
global last_weight_bit
# quantize Function
class QuantizeFunction(Function):
    @staticmethod
    def forward(ctx, input, qbit, mode, last_value = None, training = None):
        global last_weight_scale
        global last_activation_scale
        global last_weight_bit
        global last_activation_bit
        # last_value change only when training
        if mode == 'weight':
            last_weight_bit = qbit
            scale = torch.max(torch.abs(input)).item()
        elif mode == 'activation':
            last_activation_bit = qbit
            if training:
                ratio = 0.707
                tmp = last_value.item()
                tmp = ratio * tmp + (1 - ratio) * torch.max(torch.abs(input)).item()
                last_value.data[0] = tmp
            scale = last_value.data[0]
        else:
            assert 0, 'not support %s' % mode
        # transfer
        thres = 2 ** (qbit - 1) - 1
        output = input / scale
        output = torch.clamp(torch.round(output * thres), 0 - thres, thres - 0)
        output = output * scale / thres
        if mode == 'weight':
            last_weight_scale = scale / thres
        elif mode == 'activation':
            last_activation_scale = scale / thres
        else:
            assert 0, 'not support %s' % mode
        return output
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None
Quantize = QuantizeFunction.apply

# AB = 1
# N = 512
# Q = 10
# # METHOD = 'TRADITION'
# # METHOD = 'FIX_TRAIN'
# # METHOD = 'SINGLE_FIX_TEST'
# METHOD = ''
# quantize layer, include conv and fc without bias
class QuantizeLayer(nn.Module):
    def __init__(self, hardware_config, layer_config, quantize_config):
        super(QuantizeLayer, self).__init__()
        # load hardware layer and quantize config in setting
        self.hardware_config = copy.deepcopy(hardware_config)
        self.layer_config = copy.deepcopy(layer_config)
        self.quantize_config = copy.deepcopy(quantize_config)
        # split weights
        if self.layer_config['type'] == 'conv':
            channel_N = (self.hardware_config['xbar_size'] // (self.layer_config['kernel_size'] ** 2))
            complete_bar_num = self.layer_config['in_channels'] // channel_N
            residual_col_num = self.layer_config['in_channels'] % channel_N
            in_channels_list = []
            if residual_col_num > 0:
                in_channels_list = [channel_N] * complete_bar_num + [residual_col_num]
            else:
                in_channels_list = [channel_N] * complete_bar_num
            # generate Module List
            if 'stride' not in self.layer_config.keys():
                self.layer_config['stride'] = 1
            if 'padding' not in self.layer_config.keys():
                self.layer_config['padding'] = 0
            self.layer_list = nn.ModuleList([nn.Conv2d(i, self.layer_config['out_channels'], self.layer_config['kernel_size'],
                                                        stride = self.layer_config['stride'], padding = self.layer_config['padding'], dilation = 1, groups = 1, bias = False)
                                                        for i in in_channels_list])
            self.split_input = channel_N
        elif self.layer_config['type'] == 'fc':
            complete_bar_num = self.layer_config['in_features'] // self.hardware_config['xbar_size']
            residual_col_num = self.layer_config['in_features'] % self.hardware_config['xbar_size']
            if residual_col_num > 0:
                in_features_list = [self.hardware_config['xbar_size']] * complete_bar_num + [residual_col_num]
            else:
                in_features_list = [self.hardware_config['xbar_size']] * complete_bar_num
            # generate Module List
            self.layer_list = nn.ModuleList([nn.Linear(i, self.layer_config['out_features'], False) for i in in_features_list])
            self.split_input = self.hardware_config['xbar_size']
        else:
            raise NotImplementedError
        self.last_value = nn.Parameter(torch.zeros(1))
        self.bit_scale_list = nn.Parameter(torch.zeros(3, 2))
        # input shape, output shape, and layer information
        self.input_shape = torch.Size()
        self.output_shape = torch.Size()
        self.layer_info = None

    def structure_forward(self, input):
        # test in TRADITION
        self.input_shape = input.shape
        input_list = torch.split(input, self.split_input, dim = 1)
        output = None
        for i in range(len(self.layer_list)):
            if i == 0:
                output = self.layer_list[i](input_list[i])
            else:
                output.add_(self.layer_list[i](input_list[i]))
        self.output_shape = output.shape
        # layer_info
        self.layer_info = collections.OrderedDict()
        if self.layer_config['type'] == 'conv':
            self.layer_info['type'] = 'conv'
            self.layer_info['Inputsize'] = list(self.input_shape)[2:]
            self.layer_info['Outputsize'] = list(self.output_shape)[2:]
            self.layer_info['Kernelsize'] = self.layer_config['kernel_size']
            self.layer_info['Stride'] = 1
            self.layer_info['Inputchannel'] = int(self.input_shape[1])
            self.layer_info['Outputchannel'] = int(self.output_shape[1])
        elif self.layer_config['type'] == 'fc':
            self.layer_info['type'] = 'fc'
            self.layer_info['Infeature'] = int(self.input_shape[1])
            self.layer_info['Outfeature'] = int(self.input_shape[1])
        self.layer_info['Inputbit'] = int(self.bit_scale_list[0,0].item())
        self.layer_info['Weightbit'] = int(self.bit_scale_list[1,0].item())
        self.layer_info['outputbit'] = int(self.bit_scale_list[2,0].item())
        self.layer_info['row_split_num'] = len(self.layer_list)
        self.layer_info['weight_cycle'] = (int(self.bit_scale_list[1, 0].item()) - 1) // (self.hardware_config['weight_bit'])
        return output

    def forward(self, input, method = 'SINGLE_FIX_TEST', adc_action = 'SCALE'):
        METHOD = method
        # float method
        if METHOD == 'TRADITION':
            input_list = torch.split(input, self.split_input, dim = 1)
            output = None
            for i in range(len(self.layer_list)):
                if i == 0:
                    output = self.layer_list[i](input_list[i])
                else:
                    output.add_(self.layer_list[i](input_list[i]))
            return output
        # fix training
        if METHOD == 'FIX_TRAIN':
            weight = torch.cat([l.weight for l in self.layer_list], dim = 1)
            # quantize weight
            global last_weight_scale
            global last_activation_scale
            global last_weight_bit
            global last_activation_bit
            # last activation bit and scale
            self.bit_scale_list.data[0, 0] = last_activation_bit
            self.bit_scale_list.data[0, 1] = last_activation_scale
            weight = Quantize(weight, self.quantize_config['weight_bit'], 'weight', None, None)
            # weight bit and scale
            self.bit_scale_list.data[1, 0] = last_weight_bit
            self.bit_scale_list.data[1, 1] = last_weight_scale
            if self.layer_config['type'] == 'conv':
                output = F.conv2d(input, weight, None, self.layer_config['stride'], self.layer_config['padding'], 1, 1)
            elif self.layer_config['type'] == 'fc':
                output = F.linear(input, weight, None)
            else:
                raise NotImplementedError
            output = Quantize(output, self.quantize_config['activation_bit'], 'activation', self.last_value, self.training)
            # output activation bit and scale
            self.bit_scale_list.data[2, 0] = last_activation_bit
            self.bit_scale_list.data[2, 1] = last_activation_scale
            return output
        if METHOD == 'SINGLE_FIX_TEST':
            assert self.training == False
            bit_weights = self.get_bit_weights()
            output = self.set_weights_forward(input, bit_weights, adc_action)
            return output
        assert 0
    def get_bit_weights(self):
        weight_bit = int(self.bit_scale_list[1, 0].item())
        weight_scale = self.bit_scale_list[1, 1].item()
        assert weight_bit != 0 and weight_scale != 0, f'weight bit and scale should be given by the params'
        bit_weights = collections.OrderedDict()
        for layer_num, l in enumerate(self.layer_list):
            # assert (weight_bit - 1) % self.hardware_config['weight_bit'] == 0, generate weight cycle
            weight_cycle = math.ceil((weight_bit - 1) / self.hardware_config['weight_bit'])
            # transfer part weight
            thres = 2 ** (weight_bit - 1) - 1
            weight_digit = torch.clamp(torch.round(l.weight / weight_scale), 0 - thres, thres - 0)
            # split weight into bit
            sign_weight = torch.sign(weight_digit)
            weight_digit = torch.abs(weight_digit)
            base = 1
            step = 2 ** self.hardware_config['weight_bit']
            for j in range(weight_cycle):
                tmp = torch.fmod(weight_digit, base * step) - torch.fmod(weight_digit, base)
                tmp = torch.mul(sign_weight, tmp)
                tmp = copy.deepcopy((tmp / base).detach().cpu().numpy())
                bit_weights[f'split{layer_num}_weight{j}_positive'] = np.where(tmp > 0, tmp, 0)
                bit_weights[f'split{layer_num}_weight{j}_negative'] = np.where(tmp < 0, -tmp, 0)
                base = base * step
        return bit_weights
    def set_weights_forward(self, input, bit_weights, adc_action):
        assert self.training == False
        output = None
        input_list = torch.split(input, self.split_input, dim = 1)
        scale = self.last_value.item()
        weight_bit = int(self.bit_scale_list[1, 0].item())
        weight_scale = self.bit_scale_list[1, 1].item()
        for layer_num, l in enumerate(self.layer_list):
            # assert (weight_bit - 1) % self.hardware_config['weight_bit'] == 0, generate weight cycle
            weight_cycle = math.ceil((weight_bit - 1) / self.hardware_config['weight_bit'])
            weight_container = []
            base = 1
            step = 2 ** self.hardware_config['weight_bit']
            for j in range(weight_cycle):
                tmp = bit_weights[f'split{layer_num}_weight{j}_positive'] - bit_weights[f'split{layer_num}_weight{j}_negative']
                tmp = torch.from_numpy(tmp)
                weight_container.append(tmp.to(device = input.device, dtype = input.dtype))
                base = base * step
            activation_in_bit = int(self.bit_scale_list[0, 0].item())
            activation_in_scale = self.bit_scale_list[0, 1].item()
            thres = 2 ** (activation_in_bit - 1) - 1
            activation_in_digit = torch.clamp(torch.round(input_list[layer_num] / activation_in_scale), 0 - thres, thres - 0)
            # assert (activation_in_bit - 1) % self.hardware_config['input_bit'] == 0, generate activation_in cycle
            activation_in_cycle = math.ceil((activation_in_bit - 1) / self.hardware_config['input_bit'])
            # split activation into bit
            sign_activation_in = torch.sign(activation_in_digit)
            activation_in_digit = torch.abs(activation_in_digit)
            base = 1
            step = 2 ** self.hardware_config['input_bit']
            activation_in_container = []
            for i in range(activation_in_cycle):
                tmp = torch.fmod(activation_in_digit, base * step) -  torch.fmod(activation_in_digit, base)
                activation_in_container.append(torch.mul(sign_activation_in, tmp) / base)
                base = base * step
            # calculation and add
            point_shift = self.quantize_config['point_shift']
            Q = self.hardware_config['quantize_bit']
            for i in range(activation_in_cycle):
                for j in range(weight_cycle):
                    tmp = None
                    if self.layer_config['type'] == 'conv':
                        tmp = F.conv2d(activation_in_container[i], weight_container[j], None, 1, 0, 1, 1)
                    elif self.layer_config['type'] == 'fc':
                        tmp = F.linear(activation_in_container[i], weight_container[j], None)
                    else:
                        raise NotImplementedError
                    if adc_action == 'SCALE':
                        tmp = tmp * weight_scale * activation_in_scale
                        tmp = tmp / scale * (2 ** ((activation_in_cycle - 1) * self.hardware_config['input_bit'] + \
                                                   (weight_cycle - 1) * self.hardware_config['weight_bit']))
                        transfer_point = point_shift + (Q - 1)
                        tmp = tmp * (2 ** transfer_point)
                        tmp = torch.clamp(torch.round(tmp), 1 - 2 ** (Q - 1), 2 ** (Q - 1) - 1)
                        tmp = tmp / (2 ** transfer_point)
                    elif adc_action == 'FIX':
                        # fix scale range
                        fix_scale_range = (2 ** self.hardware_config['input_bit'] - 1) * \
                                          (2 ** self.hardware_config['weight_bit'] - 1) * \
                                          self.hardware_config['xbar_size']
                        tmp = tmp / fix_scale_range * (2 ** (Q - 1))
                        tmp = torch.clamp(torch.round(tmp), 1 - 2 ** (Q - 1), 2 ** (Q - 1) - 1)
                        tmp = tmp * fix_scale_range / (2 ** (Q - 1))
                        tmp = tmp * weight_scale * activation_in_scale
                        tmp = tmp / scale * (2 ** ((activation_in_cycle - 1) * self.hardware_config['input_bit'] + \
                                                   (weight_cycle - 1) * self.hardware_config['weight_bit']))
                    else:
                        assert 0, f'can not support {adc_action}'
                    # scale
                    scale_point = (activation_in_cycle - 1 - i) * self.hardware_config['input_bit'] + \
                                  (weight_cycle - 1 - j) * self.hardware_config['weight_bit']
                    tmp = tmp / (2 ** scale_point)
                    # add
                    if torch.is_tensor(output):
                        output = output + tmp
                    else:
                        output = tmp
        # quantize output
        activation_out_bit = int(self.bit_scale_list[0, 0].item())
        activation_out_scale = self.bit_scale_list[0, 1].item()
        thres = 2 ** (activation_out_bit - 1) - 1
        output = torch.clamp(torch.round(output * thres), 0 - thres, thres - 0)
        output = output * scale / thres
        return output
    def extra_repr(self):
        return str(self.hardware_config) + str(self.layer_config) + str(self.quantize_config)
