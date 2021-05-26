#-*-coding:utf-8-*-
import collections
import copy
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from MNSIM.Interface import quantize

class NetworkGraph(nn.Module):
    def __init__(self, hardware_config, layer_config_list, quantize_config_list, input_index_list, input_params):
        super(NetworkGraph, self).__init__()
        # same length for layer_config_list , quantize_config_list and input_index_list
        assert len(layer_config_list) == len(quantize_config_list)
        assert len(layer_config_list) == len(input_index_list)
        # layer list
        self.layer_list = nn.ModuleList()
        # add layer to layer list by layer_config, quantize_config, and input_index
        for layer_config, quantize_config in zip(layer_config_list, quantize_config_list):
            assert 'type' in layer_config.keys()
            if layer_config['type'] in quantize.QuantizeLayerStr:
                layer = quantize.QuantizeLayer(hardware_config, layer_config, quantize_config)
            elif layer_config['type'] in quantize.StraightLayerStr:
                layer = quantize.StraightLayer(hardware_config, layer_config, quantize_config)
            else:
                assert 0, f'not support {layer_config["type"]}'
            self.layer_list.append(layer)
        # save input_index_list, input_index is a list
        self.input_index_list = copy.deepcopy(input_index_list)
        self.input_params = copy.deepcopy(input_params)
    def forward(self, x, method = 'SINGLE_FIX_TEST', adc_action = 'SCALE'):
        # input fix information
        quantize.last_activation_scale = self.input_params['activation_scale']
        quantize.last_activation_bit = self.input_params['activation_bit']
        # forward
        tensor_list = [x]
        for i, layer in enumerate(self.layer_list):
            # find the input tensor
            input_index = self.input_index_list[i]
            assert len(input_index) in [1, 2]
            if len(input_index) == 1:
                tensor_list.append(layer.forward(tensor_list[input_index[0] + i + 1], method, adc_action))
            else:
                tensor_list.append(
                    layer.forward([
                        tensor_list[input_index[0] + i + 1],
                        tensor_list[input_index[1] + i + 1],
                    ],
                    method,
                    adc_action,
                    )
                )
        return tensor_list[-1]
    def get_weights(self):
        net_bit_weights = []
        for layer in self.layer_list:
            net_bit_weights.append(layer.get_bit_weights())
        return net_bit_weights
    def set_weights_forward(self, x, net_bit_weights, adc_action = 'SCALE'):
        # input fix information
        quantize.last_activation_scale = self.input_params['activation_scale']
        quantize.last_activation_bit = self.input_params['activation_bit']
        # filter None
        net_bit_weights = list(filter(lambda x:x!=None, net_bit_weights))
        # forward
        tensor_list = [x]
        count = 0
        for i, layer in enumerate(self.layer_list):
            # find the input tensor
            input_index = self.input_index_list[i]
            assert len(input_index) in [1, 2]
            if isinstance(layer, quantize.QuantizeLayer):
                tensor_list.append(layer.set_weights_forward(tensor_list[input_index[0] + i + 1], net_bit_weights[count], adc_action))
                # tensor_list.append(layer.forward(tensor_list[input_index[0] + i + 1], 'SINGLE_FIX_TEST', adc_action))
                count = count + 1
            else:
                if len(input_index) == 1:
                    tensor_list.append(layer.forward(tensor_list[input_index[0] + i + 1], 'FIX_TRAIN', None))
                else:
                    tensor_list.append(
                    layer.forward([
                        tensor_list[input_index[0] + i + 1],
                        tensor_list[input_index[1] + i + 1],
                    ],
                    'FIX_TRAIN',
                    None,
                    )
                )
        return tensor_list[-1]
    def get_structure(self):
        # forward structure
        x = torch.zeros(self.input_params['input_shape'])
        self.to(x.device)
        self.eval()
        tensor_list = [x]
        for i, layer in enumerate(self.layer_list):
            # find the input tensor
            input_index = self.input_index_list[i]
            assert len(input_index) in [1, 2]
            # print(tensor_list[input_index[0]+i+1].shape)
            if len(input_index) == 1:
                tensor_list.append(layer.structure_forward(tensor_list[input_index[0] + i + 1]))
            else:
                tensor_list.append(
                    layer.structure_forward([
                        tensor_list[input_index[0] + i + 1],
                        tensor_list[input_index[1] + i + 1],
                    ],
                    )
                )
        # structure information, stored as list
        net_info = []
        for layer in self.layer_list:
            net_info.append(layer.layer_info)
        return net_info
    def load_change_weights(self, state_dict):
        # input is a state dict, weights
        # concat all layer_list weights
        keys_map = collections.OrderedDict()
        for key in state_dict.keys():
            tmp_key = re.sub('\.layer_list\.\d+\.weight$', '', key)
            if tmp_key not in keys_map.keys():
                keys_map[tmp_key] = [key]
            else:
                keys_map[tmp_key].append(key)
        # concat and split
        tmp_state_dict = collections.OrderedDict()
        for tmp_key, key_list in keys_map.items():
            if len(key_list) == 1 and tmp_key == key_list[0]:
                # print('origin weights')
                tmp_state_dict[tmp_key] = state_dict[key_list[0]]
            else:
                # print(f'transfer weights {tmp_key}')
                # get layer info
                layer_config = None
                hardware_config = None
                for i in range(len(self.layer_list)):
                    name = f'layer_list.{i}'
                    if name == tmp_key:
                        layer_config = self.layer_list[i].layer_config
                        hardware_config = self.layer_list[i].hardware_config
                assert layer_config, 'layer must have layer config'
                assert hardware_config, 'layer must have hardware config'
                # concat weights
                total_weights = torch.cat([state_dict[key] for key in key_list], dim = 1)
                # split weights
                if layer_config['type'] == 'conv':
                    split_len = (hardware_config['xbar_size'] // (layer_config['kernel_size'] ** 2))
                elif layer_config['type'] == 'fc':
                    split_len = hardware_config['xbar_size']
                else:
                    assert 0, f'not support {layer_config["type"]}'
                weights_list = torch.split(total_weights, split_len, dim = 1)
                # load weights
                for i, weights in enumerate(weights_list):
                    tmp_state_dict[tmp_key + f'.layer_list.{i}.weight'] = weights
        # load weights
        self.load_state_dict(tmp_state_dict)

def get_net(hardware_config = None, cate = 'lenet', num_classes = 10):
    # initial config
    if hardware_config == None:
        hardware_config = {'xbar_size': 512, 'input_bit': 2, 'weight_bit': 1, 'quantize_bit': 10}
        # hardware_config = {'xbar_size': 256, 'input_bit': 1, 'weight_bit': 1, 'quantize_bit': 8}
    # layer_config_list, quantize_config_list, and input_index_list
    layer_config_list = []
    quantize_config_list = []
    input_index_list = []
    # layer by layer
    assert cate in ['lenet', 'vgg16', 'vgg8', 'alexnet', 'resnet18']
    if cate.startswith('lenet'):
        layer_config_list.append({'type': 'conv', 'in_channels': 3, 'out_channels': 6, 'kernel_size': 5})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 6, 'out_channels': 16, 'kernel_size': 5})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 16, 'out_channels': 120, 'kernel_size': 5})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 120, 'out_features': 84})
        layer_config_list.append({'type': 'dropout'})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'fc', 'in_features': 84, 'out_features': num_classes})
    elif cate.startswith('vgg16'):
        layer_config_list.append({'type': 'conv', 'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 2048, 'out_features': num_classes})
        # layer_config_list.append({'type': 'dropout'})
        # layer_config_list.append({'type': 'relu'})
        # layer_config_list.append({'type': 'fc', 'in_features': 512, 'out_features': num_classes})
        # layer_config_list.append({'type': 'dropout'})
        # layer_config_list.append({'type': 'relu'})
        # layer_config_list.append({'type': 'fc', 'in_features': 4096, 'out_features': num_classes})
    elif cate.startswith('vgg8'):
        layer_config_list.append({'type': 'conv', 'in_channels': 3, 'out_channels': 128, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'padding': 0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 1024, 'out_features': num_classes})
    elif cate.startswith('alexnet'):
        layer_config_list.append({'type': 'conv', 'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'padding': 1, 'stride': 2})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 192, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 192, 'out_channels': 384, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 384, 'out_channels': 256, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 1024, 'out_features': 512})
        layer_config_list.append({'type': 'dropout'})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'fc', 'in_features': 512, 'out_features': num_classes})
    elif cate.startswith('resnet18'):
        layer_config_list.append({'type': 'conv', 'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'padding': 1, 'stride': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
        # block 1
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'padding': 1, 'stride': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'padding': 1, 'stride': 1})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -4]})
        layer_config_list.append({'type': 'relu'})
        # block 2
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'padding': 1, 'stride': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'padding': 1, 'stride': 1})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -4]})
        layer_config_list.append({'type': 'relu'})
        # block 3
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'padding': 1, 'stride': 2})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'padding': 1, 'stride': 1})
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'padding': 1, 'stride': 2, 'input_index': [-4]})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -2]})
        layer_config_list.append({'type': 'relu'})
        # block 4
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'padding': 1, 'stride': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'padding': 1, 'stride': 1})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -4]})
        layer_config_list.append({'type': 'relu'})
        # block 5
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'padding': 1, 'stride': 2})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'padding': 1, 'stride': 1})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'padding': 1, 'stride': 2, 'input_index': [-4]})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -2]})
        layer_config_list.append({'type': 'relu'})
        # block 6
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'padding': 1, 'stride': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'padding': 1, 'stride': 1})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -4]})
        layer_config_list.append({'type': 'relu'})
        # block 7
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'padding': 1, 'stride': 2})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'padding': 1, 'stride': 1})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'padding': 1, 'stride': 2, 'input_index': [-4]})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -2]})
        layer_config_list.append({'type': 'relu'})
        # block 8
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'padding': 1, 'stride': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'padding': 1, 'stride': 1})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -4]})
        layer_config_list.append({'type': 'relu'})
        # output
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 2048, 'out_features': 512})
        layer_config_list.append({'type': 'dropout'})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'fc', 'in_features': 512, 'out_features': num_classes})
    else:
        assert 0, f'not support {cate}'
    for i in range(len(layer_config_list)):
        quantize_config_list.append({'weight_bit': 9, 'activation_bit': 9, 'point_shift': -2})
        if 'input_index' in layer_config_list[i]:
            input_index_list.append(layer_config_list[i]['input_index'])
        else:
            input_index_list.append([-1])
    input_params = {'activation_scale': 1. / 255., 'activation_bit': 9, 'input_shape': (1, 3, 32, 32)}
    # add bn for every conv
    L = len(layer_config_list)
    for i in range(L-1, -1, -1):
        if layer_config_list[i]['type'] == 'conv':
            # continue
            layer_config_list.insert(i+1, {'type': 'bn', 'features': layer_config_list[i]['out_channels']})
            quantize_config_list.insert(i+1, {'weight_bit': 9, 'activation_bit': 9, 'point_shift': -2})
            input_index_list.insert(i+1, [-1])
            for j in range(i + 2, len(layer_config_list), 1):
                for relative_input_index in range(len(input_index_list[j])):
                    if j + input_index_list[j][relative_input_index] < i + 1:
                        input_index_list[j][relative_input_index] -= 1
    # print(layer_config_list)
    # print(quantize_config_list)
    # print(input_index_list)
    # generate net
    net = NetworkGraph(hardware_config, layer_config_list, quantize_config_list, input_index_list, input_params)
    return net

if __name__ == '__main__':
    assert len(sys.argv) == 3
    net = get_net(cate = sys.argv[1], num_classes = int(sys.argv[2]))
    print(net)
    for name, param in net.named_parameters():
        print(name, type(param.data), param.size())
    print(f'this is network input shape {net.input_params["input_shape"]}，output shape {net.layer_list[-1].layer_config["out_features"]}')
