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

import torch
from torch import nn
from MNSIM.Interface.layer import BaseLayer
from MNSIM.Interface.utils import yaml_io
from MNSIM.Interface.utils.component import Component
from MNSIM.Interface.utils.utils import SingleMap, DoubleMap


def traverse(self, inputs, func):
    """
    Traverse the network and apply the function.
    """
    self.tensor_list[0] = inputs
    for i, layer in enumerate(self.layer_list):
        t_inputs = [self.tensor_list[i+1+j] for j in layer.get_input_index()]
        t_inputs_bit_scale_list = [self.bit_scale_list[i+1+j] for j in layer.get_input_index()]
        self.tensor_list[i+1] = func(layer, t_inputs, t_inputs_bit_scale_list, i)
        self.bit_scale_list[i+1] = layer.bit_scale_list[2]

def _transfer_hardware_config(hardware_config):
    """
    This function is used to transfer the hardware config
    """
    key_transfer = [
        ("weight_bit", "cell_bit"),
        ("input_bit", "dac_bit"),
        ("quantize_bit", "adc_bit"),
        ("xbar_row", "xbar_row")
    ]
    o_config = {}
    for k1, k2 in key_transfer:
        if k1 in hardware_config:
            o_config[k2] = hardware_config[k1]
    return o_config

class BaseModel(nn.Module, Component):
    """
    This class is used to define the base model
    """
    REGISTRY = "model"
    def __init__(self, model_config_path, hardware_config, dataset):
        # Initialize the super class
        nn.Module.__init__(self)
        Component.__init__(self)
        # Initialize the layer list
        self.hardware_config = _transfer_hardware_config(hardware_config)
        self.layer_ini_list = self.get_layer_ini_list(model_config_path)
        self.dataset = dataset
        # modify conv, the last fc based on the dataset
        self.modify_last_fc(self.dataset)
        self.modify_conv_with_bn()
        self.layer_list = nn.ModuleList([
            BaseLayer.get_class_(layer_ini["layer"]["type"])(layer_ini)
            for layer_ini in self.layer_ini_list
        ])
        # tensor and bit scale for all tensors
        self.tensor_list = [None] * (len(self.layer_list) + 1)
        self.bit_scale_list = [None] * (len(self.layer_list) + 1)
        # get dataset info
        dataset_info = dataset.get_dataset_info()
        self.bit_scale_list[0] = dataset_info["bit_scale"]
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
            lambda layer, t_inputs, t_inputs_bit_scale_list, i: \
                layer.forward(t_inputs, method, t_inputs_bit_scale_list)
        )
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
            lambda layer, t_inputs, t_inputs_bit_scale_list, i: \
                layer.set_weights_forward(t_inputs, net_bit_weights[i], t_inputs_bit_scale_list)
        )
        return self.tensor_list[-1]

    def get_structure(self):
        """
        This function is used to get the structure of the network
        """
        inputs = torch.zeros(self.dataset_shape, device=torch.device("cpu"))
        self.to(torch.device("cpu"))
        with torch.no_grad():
            traverse(self, inputs,
                lambda layer, t_inputs, t_inputs_bit_scale_list, i: \
                    layer.structure_forward(t_inputs, t_inputs_bit_scale_list)
            )
        return [copy.deepcopy(layer.layer_info) for layer in self.layer_list]

    def get_key_structure(self):
        """
        This function to get the key structure
        """
        # get layer info and key index
        layer_info_list = self.get_structure()
        self.get_key_layers()
        # update
        key_layer_info_list = list()
        for layer_ini, layer_info in zip(self.layer_ini_list, layer_info_list):
            if layer_ini["EXTRA"]["key_index"] is not None:
                layer_info["Layerindex"] = layer_ini["EXTRA"]["key_index"]
                layer_info["Inputindex"] = layer_ini["EXTRA"]["key_input_index_list"]
                layer_info["Outputindex"] = layer_ini["EXTRA"]["key_output_index_list"]
                key_layer_info_list.append([(layer_info, None)])
        return key_layer_info_list

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

    def modify_conv_with_bn(self):
        """
        This function is used to modify the conv with bn
        """
        # add extra for name, index, and connections
        index_name_connection = DoubleMap((-1, "input"))
        for i, layer_ini in enumerate(self.layer_ini_list):
            # extra, from input index to input name, set output name and output index
            layer_ini["EXTRA"]["name"] = f"layer_{i}_" + layer_ini["layer"]["type"]
            layer_ini["EXTRA"]["input_name_list"] = [
                index_name_connection.find(0, i+index)
                for index in layer_ini["layer"]["input_index"]
            ]
            layer_ini["EXTRA"]["output_name_list"] = [layer_ini["EXTRA"]["name"] + "_output"]
            index_name_connection.add_more((i, layer_ini["EXTRA"]["output_name_list"][0]))
        # add for conv
        for i in range(len(self.layer_ini_list)-1, -1, -1):
            layer_ini = self.layer_ini_list[i]
            if layer_ini["layer"]["type"] == "conv" and \
                layer_ini["layer"]["conv_add_bn"] is True:
                sp_name = f"{layer_ini['EXTRA']['name']}_bn"
                sp_ini = {
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
                        "input_name_list": [sp_name + "_output"],
                        "output_name_list": layer_ini["EXTRA"]["output_name_list"]}
                }
                layer_ini["EXTRA"]["output_name_list"] = sp_ini["EXTRA"]["input_name_list"]
                self.layer_ini_list.insert(i+1, sp_ini)
        # from name to index
        index_name_connection.clear()
        index_name_connection.add_more((-1, "input"))
        for i, layer_ini in enumerate(self.layer_ini_list):
            layer_ini["layer"]["input_index"] = [
                index_name_connection.find(1, input_name) - i
                for input_name in layer_ini["EXTRA"]["input_name_list"]
            ]
            index_name_connection.add_more((i, layer_ini["EXTRA"]["output_name_list"][0]))

    def modify_last_fc(self, dataset):
        """
        modify the last fc based on the dataset
        """
        last_layer = self.layer_ini_list[-1]
        assert last_layer["layer"]["type"] == "fc", \
            f"The last layer should be fc, but {last_layer['layer']['type']} is found"
        last_layer["layer"]["out_features"] = dataset.get_num_classes()

    def get_key_layers(self):
        """
        get key layers
        """
        index_name_connection = SingleMap((-1, "input"))
        key_count = 0
        for i, (layer_ini, layer) in enumerate(zip(self.layer_ini_list, self.layer_list)):
            layer_ini["EXTRA"]["key_input_name_list"] = [
                index_name_connection.find(0, i+index)
                for index in layer_ini["layer"]["input_index"]
            ]
            layer_ini["EXTRA"]["key_output_name_list"] = layer_ini["EXTRA"]["output_name_list"]
            if layer.key_layer_flag():
                layer_ini["EXTRA"]["key_index"] = key_count
                key_count += 1
                index_name_connection.add_more((i, layer_ini["EXTRA"]["key_output_name_list"][0]))
            else:
                layer_ini["EXTRA"]["key_index"] = None
                assert len(layer_ini["EXTRA"]["key_input_name_list"]) == 1, \
                    f"The layer {layer_ini['layer']['type']} should have only one input"
                index_name_connection.add_more((i, layer_ini["EXTRA"]["key_input_name_list"][0]))
        # from name to list
        index_name_connection = DoubleMap((-1, "input"))
        for layer_ini, layer in zip(self.layer_ini_list, self.layer_list):
            if layer.key_layer_flag():
                # add for input
                key_index = layer_ini["EXTRA"]["key_index"]
                layer_ini["EXTRA"]["key_input_index_list"] = [
                    index_name_connection.find(1, input_name) - key_index
                    for input_name in layer_ini["EXTRA"]["key_input_name_list"]
                ]
                index_name_connection.add_more((
                    key_index, layer_ini["EXTRA"]["key_output_name_list"][0]
                ))
                # add for output
                output_name = layer_ini["EXTRA"]["key_output_name_list"][0]
                layer_ini["EXTRA"]["key_output_index_list"] = list()
                for t_layer_ini, t_layer in zip(self.layer_ini_list, self.layer_list):
                    if t_layer.key_layer_flag() and \
                        output_name in t_layer_ini["EXTRA"]["key_input_name_list"]:
                        layer_ini["EXTRA"]["key_output_index_list"].append(
                            t_layer_ini["EXTRA"]["key_index"] - key_index
                        )
    @abc.abstractmethod
    def get_name(self):
        """
        get name
        """
        raise NotImplementedError

def update_config(base, config):
    """
    This function is used to update the config
    """
    base = copy.deepcopy(base)
    base.update(config)
    return base

def format_yaml(yaml_file, hardware_config):
    """"
    This function is used to format the yaml file
    """
    # get basic config
    config = yaml_io.read_yaml(yaml_file)
    model_name = config["name"]
    # generate_config
    general = [
        config.get("hardware_config", {}),
        config.get("quantize_config", {}),
        config.get("layer_config", {})
    ]
    general[0].update(hardware_config)
    # traverse the model config
    layer_ini_list = []
    for _, layer_ini in enumerate(config["model_config"]):
        # specific config
        specific = [
            layer_ini.get("hardware_config", {}),
            layer_ini.get("quantize_config", {}),
            dict([(k, v) for k, v in layer_ini.items() \
                if k not in ["hardware_config", "quantize_config"]])
        ]
        # modify
        layer_ini = dict()
        for j, cate in enumerate(["hardware", "quantize", "layer"]):
            layer_ini[cate] = update_config(general[j], specific[j])
        # add empty EXTRA
        layer_ini["EXTRA"] = dict()
        layer_ini_list.append(layer_ini)
    return model_name, layer_ini_list

class Model(BaseModel):
    """
    This class is used to define the model based on yaml config
    """
    NAME = "yaml"
    def get_layer_ini_list(self, config_path):
        model_name, layer_ini_list = format_yaml(config_path, self.hardware_config)
        self.model_name = model_name
        return layer_ini_list

    def get_name(self):
        return self.model_name
