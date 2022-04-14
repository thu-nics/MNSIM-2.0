#-*-coding:utf-8-*-
"""
@FileName:
    model.py
@Description:
    This file is used to define the model of the network.
@Authors:
    Hanbo Sun(sun-hb17@mails.tsinghua.edu.cn)
@CreateTime:
    2022/04/08 15:21
"""
import abc
import copy
import functools

import torch
from torch import nn
from MNSIM.Interface.layer import BaseLayer
from MNSIM.Interface.utils import yaml_io
from MNSIM.Interface.utils.component import Component


def traverse(self, inputs, func):
    """
    Traverse the network and apply the function.
    """
    self.tensor_list[0] = inputs
    for i, layer in enumerate(self.layer_list):
        t_input = [self.tensor_list[i+1+j] for j in layer.get_input_index()]
        t_input_config = [self.input_config_list[i+1+j] for j in layer.get_input_index()]
        self.tensor_list[i+1] = func(layer, t_input, t_input_config, i)
        self.input_config_list[i+1] = layer.bit_scale_list[2]

class BaseModel(nn.Module, Component):
    """
    This class is used to define the base model
    """
    REGISTRY = "model"
    def __init__(self, model_config_path, dataset_info):
        # Initialize the super class
        nn.Module.__init__(self)
        Component.__init__(self)
        # Initialize the layer list
        layer_ini_list = self.get_layer_ini_list(model_config_path)
        self.layer_list = nn.ModuleList([
            BaseLayer.get_class_(layer_ini["layer"]["type"])(layer_ini)
            for layer_ini in layer_ini_list
        ])
        # multi for input, and each for output
        self.tensor_list = [None] * (len(self.layer_list) + 1)
        self.input_config_list = [None] * (len(self.layer_list) + 1)
        self.input_config_list[0] = dataset_info["bit_scale"]
        self.dataset_shape = dataset_info["shape"]
        # logger
        self.logger.info(f"initialize the model with {len(self.layer_list)} layers")

    @abc.abstractmethod
    def get_layer_ini_list(self, config_path):
        """
        This function is used to get the layer ini list
        """
        raise NotImplementedError

    def forward(self, inputs, method="SINGLE_FIX_TEST"):
        """
        This function is used to forward the network, for different method
        """
        traverse(self, inputs,
            lambda layer, t_input, t_input_config, i: layer.forward(t_input, method, t_input_config)
        )
        for i, tensor in enumerate(self.tensor_list):
            torch.save(tensor, f'model_inter_{i}.pth')
        return self.tensor_list[-1]

    def get_weights(self):
        """
        This function is used to get the weights of the network
        """
        return [layer.get_quantize_weight_bit_list() for layer in self.layer_list]

    def set_weights_forward(self, inputs, net_bit_weights):
        """
        This function is used to set the weights of the network
        """
        traverse(self, inputs,
            lambda layer, t_input, t_input_config, i: \
                layer.set_weights_forward(t_input, net_bit_weights[i], t_input_config)
        )
        return self.tensor_list[-1]

    def get_structure(self):
        """
        This function is used to get the structure of the network
        """
        inputs = torch.zeros(self.dataset_shape)
        with torch.no_grad():
            traverse(self, inputs,
                lambda layer, t_input, t_input_config, i: \
                    layer.structure_forward(t_input)
            )
        return [layer.layer_info for layer in self.layer_list]

    def load_change_weights(self, state_dict):
        """
        This function is used to load the weights of the network
        """
        for i, layer in enumerate(self.layer_list):
            prefix = f"layer_list.{i}."
            origin_weight = dict([(k[len(prefix):], v)
                for k, v in state_dict.items()
                if k.startswith(prefix)
            ])
            layer.load_change_weights(origin_weight)

def update_config(base, config):
    """
    This function is used to update the config
    """
    base = copy.deepcopy(base)
    base.update(config)
    return base

def format_yaml(yaml_file):
    """"
    This function is used to format the yaml file
    """
    # get basic config
    config = yaml_io.read_yaml(yaml_file)
    # generate_config
    general = [
        config.get("hardware_config", {}),
        config.get("quantize_config", {}),
        config.get("layer_config", {})
    ]
    # traverse the model config
    layer_ini_list = []
    for i, layer_ini in enumerate(config["model_config"]):
        # specific config
        specific = [
            layer_ini.get("hardware_config", {}),
            layer_ini.get("quantize_config", {}),
            dict([(k, v) for k, v in layer_ini.items() \
                if k not in ["hardware_config", "quantize_config"]])
        ]
        # modify
        layer_ini.clear()
        for j, cate in enumerate(["hardware", "quantize", "layer"]):
            layer_ini[cate] = update_config(general[j], specific[j])
        layer_ini["EXTRA"] = {"name": f"L{i:02d}", "output": [f"L{i:02d}_out"]}
        layer_ini_list.append(layer_ini)
    # set input_blob and output_blob
    tensor_name_list = ["input"] + functools.reduce(lambda x,y: x+y,
        [layer_ini["EXTRA"]["output"] for layer_ini in layer_ini_list]
    )
    for i, layer_ini in enumerate(layer_ini_list):
        layer_ini["EXTRA"]["input"] = [
            tensor_name_list[i+1+j] for j in layer_ini["layer"]["input_index"]
        ]
    # add for conv
    for i in range(len(layer_ini_list)-1, -1, -1):
        layer_ini = layer_ini_list[i]
        if layer_ini["layer"]["type"] == "conv" and \
            layer_ini["layer"]["conv_add_bn"] is True:
            sp_name = f"{layer_ini['EXTRA']['name']}_sp"
            bn_ini = {
                "hardware": copy.deepcopy(layer_ini["hardware"]),
                "quantize": {
                    "input": layer_ini["quantize"]["output"],
                    "weight": layer_ini["quantize"]["weight"],
                    "output": layer_ini["quantize"]["output"],
                },
                "layer": {"type": "bn", "input_index": [-1],
                    "num_features": layer_ini["layer"]["out_channels"]},
                "EXTRA": {
                    "name": sp_name,
                    "input": [sp_name],
                    "output": layer_ini["EXTRA"]["output"]}
            }
            layer_ini["EXTRA"]["output"] = [sp_name]
            layer_ini_list.insert(i+1, bn_ini)
    # change input index by name
    tensor_name_list = ["input"] + functools.reduce(lambda x,y: x+y,
        [layer_ini["EXTRA"]["output"] for layer_ini in layer_ini_list]
    )
    for i, layer_ini in enumerate(layer_ini_list):
        input_point = [
            tensor_name_list.index(name) for name in layer_ini["EXTRA"]["input"]
        ]
        layer_ini["layer"]["input_index"] = [
            point-1-i for point in input_point
        ]
    # return
    return layer_ini_list

class Model(BaseModel):
    """
    This class is used to define the model based on yaml config
    """
    NAME = "yaml"
    def get_layer_ini_list(self, config_path):
        return format_yaml(config_path)
