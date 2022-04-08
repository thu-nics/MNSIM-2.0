#-*-coding:utf-8-*-
"""
@FileName:
    layer.py
@Description:
    base class for quantize layers
@Authors:
    Hanbo Sun(sun-hb17@mails.tsinghua.edu.cn)
@CreateTime:
    2022/04/07 14:37
"""
import abc
import copy
import functools
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from MNSIM.Interface.utils.component import Component

def _get_thres(bit_width):
    """
    get threshold for quantize
    """
    return 2 ** (bit_width - 1) - 1

def _get_cycle(bit_width, bit_split):
    """
    get bit cycle
    """
    return math.ceil((bit_width - 1) / bit_split)

class QuantizeFunction(Function):
    """
    quantize function user-defined
    """
    RATIO = 0.707
    @staticmethod
    def forward(ctx, inputs, quantize_cfg, last_bit_scale):
        """
        forward function
        quantize_cfg: mode, phase, bit
        last_bit_scale: first bit, last scale, not for range scale
        """
        # get scale
        if quantize_cfg["mode"] == "weight":
            scale = torch.max(torch.abs(inputs)).item()
        elif quantize_cfg["mode"] == "activation":
            r_scale = last_bit_scale[1].item() * _get_thres(last_bit_scale[0].item())
            if quantize_cfg["phase"] == "train":
                t_scale = (3*torch.std(inputs) + torch.abs(torch.mean(inputs))).item()
                if r_scale <= 0:
                    scale = t_scale
                else:
                    scale = QuantizeFunction.RATIO * r_scale + \
                        (1 - QuantizeFunction.RATIO) * t_scale
            elif quantize_cfg["phase"] == "test":
                scale = r_scale
            else:
                assert False, "phase should be train or test"
        else:
            assert False, "mode should be weight or activation"
        # quantize
        bit = quantize_cfg["bit"]
        thres = _get_thres(bit)
        output = inputs / (scale / thres)
        output = torch.clamp(torch.round(output), min=-thres, max=thres)
        output = output / (thres / scale)
        # save bit and scale, output
        last_bit_scale[0].fill_(bit)
        last_bit_scale[1].fill_(scale / thres)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None
Quantize = QuantizeFunction.apply

def split_by_bit(tensor, scale, bit_width, bit_split):
    """
    split tensor with scale and bit_width to bit_split
    """
    weight_cycle = _get_cycle(bit_width, bit_split) # weight cycle
    thres = _get_thres(bit_width) # threshold
    quantize_tensor = torch.clamp(torch.round(tensor / scale), min=-thres, max=thres)
    # split
    sign_tensor = torch.sign(quantize_tensor)
    abs_tensor = torch.abs(quantize_tensor)
    # traverse every bit_split
    t_weight = [0] + \
        [torch.fmod(abs_tensor, 2**(bit_split*(i+1))) for i in range(weight_cycle-1)] + \
        [abs_tensor]
    o_weight = [
        torch.mul(sign_tensor, (t_weight[i+1] - t_weight[i]) / (2**(bit_split*i)))
        for i in range(weight_cycle)
    ]
    return o_weight

class BaseLayer(nn.Module, Component):
    """
    base layer for all other kind of layer
    two types of layer: weight_layer and transfer layer
        with split parameter (conv and fc)
        without split parameter (bn, pooling, relu, element_sum, flatten, dropout)
    """
    REGISTRY = "layer"
    def __init__(self, layer_ini):
        # init super class
        nn.Module.__init__(self)
        Component.__init__(self)
        # copy layer_ini and set buffer_list, init layer info
        self.layer_ini = copy.deepcopy(layer_ini)
        self.layer_info = {}
        self.set_buffer_list()
        # log
        self.logger.info(f"init {self.__class__} layer by \n {layer_ini}")

    def set_buffer_list(self):
        """
        set buffer list for this layer
        input, weight, output; bit_width, bit_scale
        """
        # set buffer list
        self.register_buffer("bit_scale_list", torch.FloatTensor([
            [self.layer_ini["quantize"]["input"], -1],
            [self.layer_ini["quantize"]["weight"], -1],
            [self.layer_ini["quantize"]["output"], -1],
        ]))

    def structure_forward(self, inputs):
        """
        forward for only one simple pass to get input shape and output shape
        """
        # forward by tradition method and input_config is None
        output = self.forward(inputs, method="TRADITION", input_config=None)
        # get layer info about input, output and type
        self.get_part_structure(inputs, output)
        # other info
        self.get_general_structure()
        # quantize info
        self.layer_info["Inputbit"] = int(self.bit_scale_list[0,0].item())
        self.layer_info["Weightbit"] = int(self.bit_scale_list[1,0].item())
        self.layer_info["outputbit"] = int(self.bit_scale_list[2,0].item())
        self.layer_info["type"] = self.layer_ini["layer"]["type"]
        self.layer_info["Inputindex"] = self.get_input_index()
        self.layer_info["Outputindex"] = [1]
        return output

    def get_input_index(self):
        return self.layer_ini["layer"].get("input_index", [-1])

    @abc.abstractmethod
    def forward(self, inputs, method="SINGLE_FIX_TEST", input_config=None):
        """
        forward function under method and input_config
        input_config is for the bit_scale list for the inputs
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_quantize_weight_bit_list(self):
        raise NotImplementedError

    @abc.abstractmethod
    def set_weights_forward(self, inputs, quantize_weight_bit_list, input_config=None):
        raise NotImplementedError

    @abc.abstractmethod
    def get_general_structure(self):
        """
        get general structure of this layer, row_split_num and others
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_part_structure(self, inputs, output):
        """
        get layer info about input, output and type
        """
        raise NotImplementedError


class BaseWeightLayer(BaseLayer):
    """
    base class for quantize weight layers
    """
    def __init__(self, layer_ini):
        super(BaseWeightLayer, self).__init__(layer_ini)
        # init extra attr for weight layers
        self.input_split_num = None
        self.layer_list = None
        self.partial_func = None
        # set extra module list
        self.set_module_list()

    def set_module_list(self):
        """
        set module list for this layer
        """
        input_split_num, layer_config_list, layer_cls, partial_func = self.get_module_config()
        self.input_split_num = input_split_num
        self.partial_func = partial_func
        self.layer_list = nn.ModuleList([
            layer_cls(**layer_config) for layer_config in layer_config_list
        ])

    @abc.abstractmethod
    def get_module_config(self):
        """
        get module config for this layer
        """
        raise NotImplementedError

    def get_general_structure(self):
        self.layer_info["row_split_num"] = len(self.layer_list)
        self.layer_info["weight_cycle"] = \
            _get_cycle(self.bit_scale_list[1,0].item(), self.layer_ini["hardware"]["cell_bit"])

    def forward(self, inputs, method="SINGLE_FIX_TEST", input_config=None):
        """
        forward for this layer
        """
        # different pass for different method
        # for method is TRADITION, we use traditional forward
        if method == "TRADITION":
            input_list = torch.split(inputs, self.input_split_num, dim=1)
            output_list = [l(v) for l, v in zip(self.layer_list, input_list)]
            return torch.sum(torch.stack(output_list, dim=0), dim=0)
        # for method is FIX_TRAIN, we use fix train forward
        if method == "FIX_TRAIN":
            weight_total = torch.cat([l.weight for l in self.layer_list], dim=1)
            quantize_weight =  Quantize(weight_total, {
                "mode": "weight",
                "phase": None,
                "bit": self.bit_scale_list[1,0].item()
            }, self.bit_scale_list[1])
            output = self.partial_func(inputs, quantize_weight)
            quantize_output = Quantize(output, {
                "mode": "activation",
                "phase": "train" if self.training else "test",
                "bit": self.bit_scale_list[2,0].item()
            }, self.bit_scale_list[2])
            return quantize_output
        # for method is SINGLE_FIX_TEST, we get weight and set weight forward
        if method == "SINGLE_FIX_TEST":
            assert not self.training, "method is SINGLE_FIX_TEST, but now is training"
            assert input_config is not None, "method is SINGLE_FIX_TEST, but input_config is None"
            quantize_weight_bit_list = self.get_quantize_weight_bit_list()
            quantize_output = self.set_weights_forward(inputs, quantize_weight_bit_list, input_config)
            return quantize_output
        assert False, "method should be TRADITION, FIX_TRAIN or SINGLE_FIX_TEST"

    def get_quantize_weight_bit_list(self):
        """
        get quantize weight bit list
        """
        weight_scale = self.bit_scale_list[1,1].item()
        weight_bit_width = self.bit_scale_list[1,0].item()
        weight_bit_split = self.layer_ini["hardware"]["cell_bit"]
        quantize_weight_bit_list = [
            split_by_bit(l.weight, weight_scale, weight_bit_width, weight_bit_split)
            for l in self.layer_list
        ]
        return quantize_weight_bit_list

    def set_weights_forward(self, inputs, quantize_weight_bit_list, input_config=None):
        """
        set weights forward
        """
        assert not self.training, "function is set_weights_forward, but training"
        self.bit_scale_list[0].copy_(input_config[0]) # for weight_layer, only be one input
        input_list = torch.split(inputs, self.input_split_num, dim=1)
        # for weight info
        weight_cycle = _get_cycle(
            self.bit_scale_list[1,0].item(),
            self.layer_ini["hardware"]["cell_bit"]
        )
        # for input activation info
        input_cycle = _get_cycle(
            self.bit_scale_list[0,0].item(),
            self.layer_ini["hardware"]["dac_bit"]
        )
        # for output activation info
        Q = self.layer_ini["hardware"]["adc_bit"]
        pf = self.layer_ini["hardware"]["point_shift"]
        output_scale = self.bit_scale_list[2,1].item() * \
            _get_thres(self.bit_scale_list[2,0].item())
        mul_scale = self.bit_scale_list[1,1].item() * self.bit_scale_list[0,1].item() * \
            (2 ** ((weight_cycle - 1)*self.layer_ini["hardware"]["cell_bit"])) * \
            (2 ** ((input_cycle - 1)*self.layer_ini["hardware"]["dac_bit"]))
        transfer_scale = 2 ** (pf + Q - 1)
        output_thres = _get_thres(Q)
        # accumulate output for layer, weight_cycle and input_cycle
        output_list = []
        for layer_num, module_weight_bit_list in enumerate(quantize_weight_bit_list):
            # for every module
            input_activation_bit_list = split_by_bit(input_list[layer_num], \
                self.bit_scale_list[0,1].item(), self.bit_scale_list[0,0].item(),
                self.layer_ini["hardware"]["dac_bit"],
            )
            # for every weight cycle and input cycle
            for i in range(input_cycle):
                for j in range(weight_cycle):
                    tmp = self.partial_func(input_activation_bit_list[i], module_weight_bit_list[j])
                    # scale for tmp, and quantize
                    tmp = tmp * (mul_scale / output_scale * transfer_scale)
                    tmp = torch.clamp(torch.round(tmp), min=-output_thres, max=output_thres)
                    # scale point for bit shift
                    scale_point = (input_cycle-1-i)*self.layer_ini["hardware"]["dac_bit"] + \
                        (weight_cycle-1-j)*self.layer_ini["hardware"]["cell_bit"]
                    tmp = tmp / (transfer_scale * (2**scale_point))
                    output_list.append(tmp)
        # sum output
        output = torch.sum(torch.stack(output_list, dim=0), dim=0)
        # quantize output
        output_thres = _get_thres(self.bit_scale_list[2,0].item())
        output = torch.clamp(torch.round(output*output_thres), min=-output_thres, max=output_thres)
        return output / (output_thres / output_scale)


def split_by_num(num, base):
    """
    split base by num, e.g., 10 = [3, 3, 3, 1]
    """
    assert num > 0, "num should be greater than 0"
    assert base > 0, "base should be greater than 0"
    return [base] * (num // base) + [num % base] * (num % base > 0)

class QuantizeConv(BaseWeightLayer):
    """
    quantize conv layer
    """
    NAME = "conv"

    def get_module_config(self):
        """
        get module config for this conv layer
        return: input_split_num, layer_config_list, layer_cls, partial_func
        """
        # basic config for conv layer
        input_split_num = math.floor(self.layer_ini["hardware"]["xbar_row"] / \
            (self.layer_ini["layer"]["kernel_size"] ** 2))
        in_channels_list = split_by_num(
            self.layer_ini["layer"]["in_channels"],
            input_split_num
        )
        layer_config_list = [{
            "in_channels": in_channels,
            "out_channels": self.layer_ini["layer"]["out_channels"],
            "bias": False,
            "kernel_size": self.layer_ini["layer"]["kernel_size"],
            "stride": self.layer_ini["layer"].get("stride", 1),
            "padding": self.layer_ini["layer"].get("padding", 0),
        } for in_channels in in_channels_list]
        partial_func = functools.partial(F.conv2d, bias=None,\
            stride=layer_config_list[0]["stride"], padding=layer_config_list[0]["padding"]
        )
        return input_split_num, layer_config_list, nn.Conv2d, partial_func

    def get_part_structure(self, inputs, output):
        input_shape = inputs.shape
        output_shape = output.shape
        self.layer_info["Inputchannel"] = int(input_shape[1])
        self.layer_info["Inputsize"] = list(input_shape[2:])
        self.layer_info["Kernelsize"] = self.layer_ini["layer"]["kernel_size"]
        self.layer_info["Stride"] = self.layer_ini["layer"].get("stride", 1)
        self.layer_info["Padding"] = self.layer_ini["layer"].get("padding", 0)
        self.layer_info["Outputchannel"] = int(output_shape[1])
        self.layer_info["Outputsize"] = list(output_shape[2:])

class QuantizeFC(BaseWeightLayer):
    """
    quantize fc layer
    """
    NAME = "fc"

    def get_module_config(self):
        """
        get module config for this fc layer
        return: input_split_num, layer_config_list, layer_cls, partial_func
        """
        # basic config for fc layer
        input_split_num = self.layer_ini["hardware"]["xbar_row"]
        in_features_list = split_by_num(self.layer_ini["layer"]["in_features"], input_split_num)
        layer_config_list = [{
            "in_features": in_features,
            "out_features": self.layer_ini["layer"]["out_features"],
            "bias": False,
        } for in_features in in_features_list]
        partial_func = functools.partial(F.linear, bias=None)
        return input_split_num, layer_config_list, nn.Linear, partial_func

    def get_part_structure(self, inputs, output):
        input_shape = inputs.shape
        output_shape = output.shape
        self.layer_info["Infeature"] = int(input_shape[1])
        self.layer_info["Outfeature"] = int(output_shape[1])

class BaseTransferLayer(BaseLayer):
    """
    base class for quantize transfer layers
    """
    def __init__(self, layer_ini):
        super(BaseTransferLayer, self).__init__(layer_ini)
        # init extra attr for weight layers
        self.layer = None
        # set extra module list
        self.set_module()

    @abc.abstractmethod
    def set_module(self):
        """
        get module config for this layer
        """
        raise NotImplementedError

    def get_general_structure(self):
        pass

    def get_part_structure(self, inputs, output):
        pass

    def forward(self, inputs, method="SINGLE_FIX_TEST", input_config=None):
        """
        forward for this layer
        """
        # different pass for different method
        # for method is TRADITION, we use traditional forward
        if method == "TRADITION":
            return self.layer(inputs)
        # for method is FIX_TRAIN, we use fix train forward
        if method == "FIX_TRAIN":
            output = self.layer(inputs)
            return self.condition_quantize_output(output, None)
        # for method is SINGLE_FIX_TEST
        if method == "SINGLE_FIX_TEST":
            assert not self.training, "method is SINGLE_FIX_TEST, but now is training"
            assert input_config is not None, "method is SINGLE_FIX_TEST, but input_config is None"
            output = self.layer(inputs)
            return self.condition_quantize_output(output, input_config)
        assert False, "method should be TRADITION, FIX_TRAIN or SINGLE_FIX_TEST"

    def condition_quantize_output(self, output, input_config=None):
        """
        condition quantize output, general, bypass
        """
        if input_config is not None:
            assert len(input_config) == 1, "input_config should be len 1 for bypass"
            self.bit_scale_list[0].copy_(input_config[0])
        assert torch.isclose(
            self.bit_scale_list[0,0],
            self.bit_scale_list[2,0]
        ).item(), "bit width should be same for input and output"
        self.bit_scale_list[2,1] = self.bit_scale_list[0,1]
        return output

    def get_quantize_weight_bit_list(self):
        """
        get quantize weight bit list
        """
        return None

    def set_weights_forward(self, inputs, quantize_weight_bit_list, input_config=None):
        """
        set weights forward
        """
        assert not self.training, "function is set_weights_forward, but training"
        self.forward(self, inputs, "SINGLE_FIX_TEST", input_config)

class QuantizeBN(BaseTransferLayer):
    """
    quantize bn layer
    """
    NAME = "bn"
    def set_module(self):
        """
        set module for bn
        """
        self.layer = nn.BatchNorm2d(self.layer_ini["layer"]["num_features"])

    def condition_quantize_output(self, output, input_config=None):
        if input_config is not None:
            assert len(input_config) == 1, "input_config should be len 1 for bn"
            self.bit_scale_list[0].copy_(input_config[0])
        quantize_output = Quantize(output, {
            "mode": "activation",
            "phase": "train" if self.training else "test",
            "bit": self.bit_scale_list[2,0].item()
        }, self.bit_scale_list[2])
        return quantize_output

    def get_part_structure(self, inputs, output):
        self.layer_info["features"] = self.layer_ini["layer"]["num_features"]

class EleSumLayer(nn.Module):
    """
    element sum layer
    """
    def __init__(self):
        super(EleSumLayer, self).__init__()
    def forward(self, x):
        """
        forward for this layer
        """
        return x[0] + x[1]

class QuantizeEleSum(BaseTransferLayer):
    """
    quantize element_sum layer
    """
    NAME = "element_sum"
    def set_module(self):
        """
        set module for element_sum
        """
        self.layer = EleSumLayer()

    def condition_quantize_output(self, output, input_config=None):
        if input_config is not None:
            assert len(input_config) == 2, "input_config should be len 2 for bn"
        quantize_output = Quantize(output, {
            "mode": "activation",
            "phase": "train" if self.training else "test",
            "bit": self.bit_scale_list[2,0].item()
        }, self.bit_scale_list[2])
        return quantize_output

class QuantizePooling(BaseTransferLayer):
    """
    quantize pooling layer
    """
    NAME = "pooling"
    def set_module(self):
        self.layer = nn.MaxPool2d(
            kernel_size=self.layer_ini["layer"]["kernel_size"],
            stride=self.layer_ini["layer"].get(
                "stride",
                self.layer_ini["layer"]["kernel_size"]
            ),
            padding=self.layer_ini["layer"].get("padding", 0)
        )

    def get_part_structure(self, inputs, output):
        input_shape = inputs.shape
        output_shape = output.shape
        self.layer_info["Inputchannel"] = int(input_shape[1])
        self.layer_info["Inputsize"] = list(input_shape)[2:]
        self.layer_info["Kernelsize"] = self.layer.kernel_size[0]
        self.layer_info["Stride"] = self.layer.stride[0]
        self.layer_info["Padding"] = self.layer.padding[0]
        self.layer_info["Outputchannel"] = int(output_shape[1])
        self.layer_info["Outputsize"] = list(output_shape)[2:]

class QuantizeRelu(BaseTransferLayer):
    """
    quantize relu layer
    """
    NAME = "relu"
    def set_module(self):
        self.layer = nn.ReLU()

class QuantizeView(BaseTransferLayer):
    """
    quantize flatten layer
    """
    NAME = "view"
    def set_module(self):
        self.layer = nn.Flatten()

class QuantizeDropout(BaseTransferLayer):
    """
    quantize dropout layer
    """
    NAME = "dropout"
    def set_module(self):
        self.layer = nn.Dropout(self.layer_ini["layer"]["p"])
