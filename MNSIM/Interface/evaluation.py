#-*-coding:utf-8-*-
"""
@FileName:
    evaluation.py
@Description:
    interface for evaluation
@Authors:
    Hanbo Sun(sun-hb17@mails.tsinghua.edu.cn)
@CreateTime:
    2022/04/15 17:33
"""
import copy
import math

import numpy as np
from MNSIM.Hardware_Model.Buffer import buffer
from MNSIM.Interface.dataset import ClassificationBaseDataset
from MNSIM.Interface.layer import split_by_num
from MNSIM.Interface.model import BaseModel
from MNSIM.Interface.trainer import BaseTrainer
from MNSIM.Interface.utils.component import Component
from MNSIM.Interface.utils.utils import _init_component, load_sim_config
from MNSIM.Interface.utils.yaml_io import read_yaml
from MNSIM.Latency_Model.PE_latency import PE_latency_analysis
from MNSIM.Latency_Model.Pooling_latency import pooling_latency_analysis


class EvaluationInterface(Component):
    """
    evaluation interface for MNSIM
    two method: evaluate and get_structure
    """
    REGISTRY = "interface"
    NAME = "train_test_interface"
    def __init__(self, simconfig_path, evaluation_config_path):
        super(EvaluationInterface, self).__init__()
        # load config
        self.simconfig_path = simconfig_path
        self.hardware_config = load_sim_config(simconfig_path)
        self.evaluation_config = read_yaml(evaluation_config_path)
        # init dataset
        self.dataset = _init_component(
            ClassificationBaseDataset, self.evaluation_config, "dataset"
        )
        # update config by hardware config, init model
        self.model = _init_component(
            BaseModel, self.evaluation_config, "model",
            {
                "hardware_config": self.hardware_config,
                "dataset": self.dataset
        })
        # trainer
        self.trainer = _init_component(
            BaseTrainer, self.evaluation_config, "trainer",
            {
                "dataset": self.dataset,
                "model": self.model,
        })

    def get_structure(self):
        """
        get the key structure of the model
        """
        return self.model.get_key_structure()

    def origin_evaluate(self, method):
        """
        evaluate the model under different test mode
        """
        self.trainer.set_test_mode(method)
        return self.trainer.test(0)

    def get_net_bits(self):
        """
        return the bits of the model
        """
        return self.model.get_weights()

    def set_net_bits_evaluate(self, net_bits):
        """
        set the bits of the model and evaluate
        """
        return self.trainer.evaluate(net_bits)

    def noc_data(self):
        """
        get the noc data
        """
        tile_max_time = 0.
        # get key structure and key layers
        key_layer_info_list = list(
            map(lambda x: x[0][0], self.model.get_key_structure()))
        key_layers = list(
            filter(lambda x: x.key_layer_flag(), self.model.layer_list)
        )
        assert len(key_layer_info_list) == len(key_layers), \
            f"key structure and key layers are not match"
        in_buf_size = 4
        out_buf_size = 4
        intra_tile_bandwidth = 20
        # generate tile behavior
        tile_behavior_list = list()
        for layer_id, (key_layer_info, key_layer) in enumerate(zip(key_layer_info_list, key_layers)):
            if key_layer_info["type"] == "conv":
                # for type conv
                # get PE array shape
                in_channel_list = [layer.in_channels for layer in key_layer.layer_list]
                output_channel = key_layer_info["Outputchannel"]
                output_channel_list = split_by_num(
                    output_channel, self.hardware_config["xbar_column"])
                PE_latency_array = np.zeros(
                    (len(in_channel_list), len(output_channel_list)))
                # get PE latency
                digital_period = None
                ADC_precision = None
                for i in range(len(in_channel_list)):
                    for j in range(len(output_channel_list)):
                        in_channels = in_channel_list[i]
                        out_channels = output_channel_list[j]
                        # init PE instance
                        PE_instance = PE_latency_analysis(
                            SimConfig_path=self.simconfig_path,
                            read_row=in_channels*(key_layer_info["Kernelsize"]**2),
                            read_column=out_channels,
                            indata=in_channels*key_layer_info["Inputbit"]/8,
                            rdata=in_channels*(key_layer_info["Kernelsize"]**2)*key_layer_info["Inputbit"]/8,
                            inprecision=key_layer_info["Inputbit"],
                            default_buf_size=in_buf_size
                        )
                        PE_latency_array[i, j] = PE_instance.PE_latency
                        digital_period = PE_instance.digital_period
                        ADC_precision = PE_instance.PE.ADC_precision
                # output buffer
                output_buffer = buffer(self.simconfig_path, 2, out_buf_size)
                # tile info list
                tile_info_list = list()
                tile_flag_list = []
                # get tile info
                for i in range(math.ceil(len(in_channel_list)/self.hardware_config["tile_row"])):
                    for j in range(math.ceil(len(output_channel_list)/self.hardware_config["tile_column"])):
                        PE_latency_slice = PE_latency_array[
                            i*self.hardware_config["tile_row"]:(i+1)*self.hardware_config["tile_row"],
                            j*self.hardware_config["tile_column"]:(j+1)*self.hardware_config["tile_column"]
                        ]
                        PE_latency = np.max(PE_latency_slice)
                        # get input and output index
                        in_channel_index = (
                            int(np.sum(in_channel_list[:i*self.hardware_config["tile_row"]])),
                            int(np.sum(in_channel_list[:(i+1)*self.hardware_config["tile_row"]]))
                        )
                        out_channel_index = (
                            int(np.sum(output_channel_list[:j*self.hardware_config["tile_column"]])),
                            int(np.sum(output_channel_list[:(j+1)*self.hardware_config["tile_column"]]))
                        )
                        tile_flag_list.append((
                            out_channel_index[0], out_channel_index[1],
                            len(tile_flag_list), len(tile_behavior_list) + len(tile_flag_list)
                        ))
                        column = out_channel_index[1] - out_channel_index[0]
                        # add transfer time
                        H_level = math.ceil(math.log2(PE_latency_slice.shape[0]))
                        joint_time = (2**H_level - 1)*digital_period
                        transfer_latency = ((2**H_level - 1)*ADC_precision + 2**H_level - H_level - 1) * \
                            column / intra_tile_bandwidth
                        output_buffer.calculate_buf_write_latency(wdata=(ADC_precision + H_level)*column/8)
                        # get tile latency
                        tile_latency = PE_latency + joint_time + transfer_latency + output_buffer.buf_wlatency
                        # get inchannels and outchannels
                        tile_info_list.append([in_channel_index, tile_latency, out_channel_index])
                        tile_max_time = max(tile_max_time, tile_latency)
                # set tile behavior
                merge_node_id = len(tile_behavior_list) + len(tile_info_list)
                for tile_id, tile_info in enumerate(tile_info_list):
                    tile_behavior = dict()
                    tile_behavior["task_id"] = None
                    tile_behavior["layer_id"] = layer_id
                    tile_behavior["tile_id"] = len(tile_behavior_list)
                    tile_behavior["target_tile_id"] = [merge_node_id]
                    tile_behavior["source_tile_id"] = []
                    tile_behavior["merge_flag"] = False
                    dependence = [dict()
                        for _ in range(key_layer_info["Outputsize"][0]*key_layer_info["Outputsize"][1])
                    ]
                    # for wait, output and latency
                    for i in range(key_layer_info["Outputsize"][0]):
                        for j in range(key_layer_info["Outputsize"][1]):
                            # wait
                            wait_row = i * key_layer_info["Stride"] + key_layer_info["Kernelsize"] - 1 - \
                                key_layer_info["Padding"]
                            wait_row = min(max(0, wait_row), key_layer_info["Inputsize"][0] - 1)
                            wait_column = j * key_layer_info["Stride"] + key_layer_info["Kernelsize"] - 1 - \
                                key_layer_info["Padding"]
                            wait_column = min(max(0, wait_column), key_layer_info["Inputsize"][1] - 1)
                            dependence[i*key_layer_info["Outputsize"][1] + j]["wait"] = [[
                                wait_row, wait_column, 0, key_layer_info["Inputchannel"], \
                                key_layer_info["Inputbit"], key_layer_info["Inputchannel"],
                                None, None, -1, None
                            ]]
                            # output
                            dependence[i*key_layer_info["Outputsize"][1] + j]["output"] = [[
                                i, j, tile_info[2][0], tile_info[2][1],
                                key_layer_info["outputbit"], key_layer_info["Outputchannel"],
                                None, layer_id, tile_id, tile_behavior["tile_id"]
                            ]]
                            # latency
                            dependence[i*key_layer_info["Outputsize"][1] + j]["latency"] = float(tile_info[1])
                            # drop
                            dependence[i*key_layer_info["Outputsize"][1] + j]["drop"] = []
                    # for drop
                    for i in range(key_layer_info["Inputsize"][0]):
                        for j in range(key_layer_info["Inputsize"][1]):
                            drop_row = math.floor((i + key_layer_info["Padding"])*1.0/key_layer_info["Stride"])
                            drop_row = min(max(0, drop_row), key_layer_info["Outputsize"][0] - 1)
                            drop_column = math.floor((j + key_layer_info["Padding"])*1.0/key_layer_info["Stride"])
                            drop_column = min(max(0, drop_column), key_layer_info["Outputsize"][1] - 1)
                            origin_drop = dependence[drop_row*key_layer_info["Outputsize"][1] + drop_column]["drop"]
                            dependence[drop_row*key_layer_info["Outputsize"][1] + drop_column]["drop"] = [[
                                i, j, 0, key_layer_info["Inputchannel"], \
                                key_layer_info["Inputbit"], key_layer_info["Inputchannel"],
                                None, None, -1, None
                            ]] + origin_drop
                    tile_behavior["dependence"] = dependence
                    tile_behavior_list.append(tile_behavior)
                # set merge node
                tile_behavior = dict()
                tile_behavior["task_id"] = None
                tile_behavior["layer_id"] = layer_id
                tile_behavior["tile_id"] = merge_node_id
                tile_behavior["target_tile_id"] = []
                tile_behavior["source_tile_id"] = [true_id for _, _, _, true_id in tile_flag_list]
                tile_behavior["merge_flag"] = True
                dependence = [dict()
                    for _ in range(key_layer_info["Outputsize"][0]*key_layer_info["Outputsize"][1])
                ]
                # for wait, output and latency
                for i in range(key_layer_info["Outputsize"][0]):
                    for j in range(key_layer_info["Outputsize"][1]):
                        # wait
                        dependence[i*key_layer_info["Outputsize"][1] + j]["wait"] = [[
                            i, j, start, end,
                            key_layer_info["outputbit"], key_layer_info["Outputchannel"],
                            None, layer_id, tile_id, true_id
                        ] for start, end, tile_id, true_id in tile_flag_list]
                        # output
                        dependence[i*key_layer_info["Outputsize"][1] + j]["output"] = [[
                            i, j, 0, key_layer_info["Outputchannel"],
                            key_layer_info["outputbit"], key_layer_info["Outputchannel"],
                            None, layer_id, -1, tile_behavior["tile_id"]
                        ]]
                        # latency
                        dependence[i*key_layer_info["Outputsize"][1] + j]["latency"] = 1
                        # drop
                        dependence[i*key_layer_info["Outputsize"][1] + j]["drop"] = \
                            copy.deepcopy(dependence[i*key_layer_info["Outputsize"][1] + j]["wait"])
                tile_behavior["dependence"] = dependence
                tile_behavior_list.append(tile_behavior)
            elif key_layer_info["type"] == "fc":
                # get PE array shape
                in_feature_list = [layer.in_features for layer in key_layer.layer_list]
                output_feature = key_layer_info["Outfeature"]
                output_feature_list = split_by_num(
                    output_feature, self.hardware_config["xbar_column"])
                PE_latency_array = np.zeros(
                    (len(in_feature_list), len(output_feature_list)))
                # get PE latency
                digital_period = None
                ADC_precision = None
                for i in range(len(in_feature_list)):
                    for j in range(len(output_feature_list)):
                        in_features = in_feature_list[i]
                        out_features = output_feature_list[j]
                        # init PE instance
                        PE_instance = PE_latency_analysis(
                            SimConfig_path=self.simconfig_path,
                            read_row=in_features,
                            read_column=out_features,
                            indata=in_features*key_layer_info["Inputbit"]/8,
                            rdata=in_features*key_layer_info["Inputbit"]/8,
                            inprecision=key_layer_info["Inputbit"],
                            default_buf_size=in_buf_size
                        )
                        PE_latency_array[i, j] = PE_instance.PE_latency
                        digital_period = PE_instance.digital_period
                        ADC_precision = PE_instance.PE.ADC_precision
                # output buffer
                output_buffer = buffer(self.simconfig_path, 2, out_buf_size)
                # tile info list
                tile_info_list = list()
                tile_flag_list = []
                # get tile info
                for i in range(math.ceil(len(in_feature_list)/self.hardware_config["tile_row"])):
                    for j in range(math.ceil(len(output_feature_list)/self.hardware_config["tile_column"])):
                        PE_latency_slice = PE_latency_array[
                            i*self.hardware_config["tile_row"]:(i+1)*self.hardware_config["tile_row"],
                            j*self.hardware_config["tile_column"]:(j+1)*self.hardware_config["tile_column"]
                        ]
                        PE_latency = np.max(PE_latency_slice)
                        # get input and output index
                        in_feature_index = (
                            int(np.sum(in_feature_list[:i*self.hardware_config["tile_row"]])),
                            int(np.sum(in_feature_list[:(i+1)*self.hardware_config["tile_row"]]))
                        )
                        out_feature_index = (
                            int(np.sum(output_feature_list[:j*self.hardware_config["tile_column"]])),
                            int(np.sum(output_feature_list[:(j+1)*self.hardware_config["tile_column"]]))
                        )
                        tile_flag_list.append((
                            out_feature_index[0], out_feature_index[1],
                            len(tile_flag_list), len(tile_behavior_list) + len(tile_flag_list)
                        ))
                        column = out_feature_index[1] - out_feature_index[0]
                        # add transfer time
                        H_level = math.ceil(math.log2(PE_latency_slice.shape[0]))
                        joint_time = (2**H_level - 1)*digital_period
                        transfer_latency = ((2**H_level - 1)*ADC_precision + 2**H_level - H_level - 1) * \
                            column / intra_tile_bandwidth
                        output_buffer.calculate_buf_write_latency(wdata=(ADC_precision + H_level)*column/8)
                        # get tile latency
                        tile_latency = PE_latency + joint_time + transfer_latency + output_buffer.buf_wlatency
                        # get infeatures and outfeatures
                        tile_info_list.append([in_feature_index, tile_latency, out_feature_index])
                        tile_max_time = max(tile_max_time, tile_latency)
                # set tile behavior
                merge_node_id = len(tile_behavior_list) + len(tile_info_list)
                for tile_id, tile_info in enumerate(tile_info_list):
                    tile_behavior = dict()
                    tile_behavior["task_id"] = None
                    tile_behavior["layer_id"] = layer_id
                    tile_behavior["tile_id"] = len(tile_behavior_list)
                    tile_behavior["target_tile_id"] = [merge_node_id]
                    tile_behavior["source_tile_id"] = []
                    tile_behavior["merge_flag"] = False
                    dependence = [dict()]
                    # for wait, output and latency
                    dependence[0]["wait"] = [[
                        0, 0, 0, key_layer_info["Infeature"],
                        key_layer_info["Inputbit"], key_layer_info["Infeature"],
                        None, None, -1, None
                    ]]
                    dependence[0]["output"] = [[
                        0, 0, tile_info[2][0], tile_info[2][1],
                        key_layer_info["outputbit"], key_layer_info["Outfeature"],
                        None, layer_id, tile_id, tile_behavior["tile_id"]
                    ]]
                    # latency
                    dependence[0]["latency"] = pooling_latency
                    # drop
                    dependence[0]["drop"] = copy.deepcopy(dependence[0]["wait"])
                    tile_behavior["dependence"] = dependence
                    tile_behavior_list.append(tile_behavior)
                # merge node
                tile_behavior = dict()
                tile_behavior["task_id"] = None
                tile_behavior["layer_id"] = layer_id
                tile_behavior["tile_id"] = merge_node_id
                tile_behavior["target_tile_id"] = []
                tile_behavior["source_tile_id"] = [true_id for _, _, _, true_id in tile_flag_list]
                tile_behavior["merge_flag"] = True
                dependence = [dict()]
                # for wait, output and latency
                dependence[0]["wait"] = [[
                    0, 0, start, end,
                    key_layer_info["outputbit"], key_layer_info["Outfeature"],
                    None, layer_id, tile_id, true_id
                ] for start, end, tile_id, true_id in tile_flag_list]
                dependence[0]["output"] = [[
                    0, 0, 0, key_layer_info["Outfeature"],
                    key_layer_info["outputbit"], key_layer_info["Outfeature"],
                    None, layer_id, -1, tile_behavior["tile_id"]
                ]]
                # latency
                dependence[0]["latency"] = 1
                # drop
                dependence[0]["drop"] = copy.deepcopy(dependence[0]["wait"])
                tile_behavior["dependence"] = dependence
                tile_behavior_list.append(tile_behavior)
            elif key_layer_info["type"] == "pooling":
                # for pooling, there is only one merge node
                tile_behavior = dict()
                tile_behavior["task_id"] = None
                tile_behavior["layer_id"] = layer_id
                tile_behavior["tile_id"] = len(tile_behavior_list)
                tile_behavior["target_tile_id"] = []
                tile_behavior["source_tile_id"] = []
                tile_behavior["merge_flag"] = True
                # get pooling latency, only one PE in one Tile
                pooling_instance = pooling_latency_analysis(
                    SimConfig_path=self.simconfig_path,
                    indata=in_channels*key_layer_info["Inputbit"]/8,
                    rdata=in_channels*key_layer_info["Inputbit"]/8,
                    outprecision=key_layer_info["outputbit"],
                    default_inbuf_size=in_buf_size,
                )
                output_buffer = buffer(self.simconfig_path, 2, out_buf_size)
                output_buffer.calculate_buf_write_latency(
                    wdata=key_layer_info["Outputchannel"]*key_layer_info["outputbit"]/8
                )
                pooling_latency = pooling_instance.pooling_latency + \
                    output_buffer.buf_wlatency
                tile_max_time = max(tile_max_time, pooling_latency)
                # traverse all tiles
                dependence = [dict()
                        for _ in range(key_layer_info["Outputsize"][0]*key_layer_info["Outputsize"][1])
                    ]
                # for wait, output and latency
                for i in range(key_layer_info["Outputsize"][0]):
                    for j in range(key_layer_info["Outputsize"][1]):
                        # wait
                        wait_row = i * key_layer_info["Stride"] + key_layer_info["Kernelsize"] - 1 - \
                            key_layer_info["Padding"]
                        wait_row = min(max(0, wait_row), key_layer_info["Inputsize"][0] - 1)
                        wait_column = j * key_layer_info["Stride"] + key_layer_info["Kernelsize"] - 1 - \
                            key_layer_info["Padding"]
                        wait_column = min(max(0, wait_column), key_layer_info["Inputsize"][1] - 1)
                        dependence[i*key_layer_info["Outputsize"][1] + j]["wait"] = [[
                            wait_row, wait_column, 0, key_layer_info["Inputchannel"], \
                            key_layer_info["Inputbit"], key_layer_info["Inputchannel"],
                            None, None, -1, None
                        ]]
                        # output
                        dependence[i*key_layer_info["Outputsize"][1] + j]["output"] = [[
                            i, j, 0, key_layer_info["Outputchannel"],
                            key_layer_info["outputbit"], key_layer_info["Outputchannel"],
                            None, layer_id, -1, tile_behavior["tile_id"]
                        ]]
                        # latency
                        dependence[i*key_layer_info["Outputsize"][1] + j]["latency"] = float(pooling_latency)
                        # drop
                        dependence[i*key_layer_info["Outputsize"][1] + j]["drop"] = []
                # for drop
                for i in range(key_layer_info["Inputsize"][0]):
                    for j in range(key_layer_info["Inputsize"][1]):
                        drop_row = math.floor((i + key_layer_info["Padding"])*1.0/key_layer_info["Stride"])
                        drop_row = min(max(0, drop_row), key_layer_info["Outputsize"][0] - 1)
                        drop_column = math.floor((j + key_layer_info["Padding"])*1.0/key_layer_info["Stride"])
                        drop_column = min(max(0, drop_column), key_layer_info["Outputsize"][1] - 1)
                        origin_drop = dependence[drop_row*key_layer_info["Outputsize"][1] + drop_column]["drop"]
                        dependence[drop_row*key_layer_info["Outputsize"][1] + drop_column]["drop"] = [[
                            i, j, 0, key_layer_info["Inputchannel"], \
                            key_layer_info["Inputbit"], key_layer_info["Inputchannel"],
                            None, None, -1, None
                        ]] + origin_drop
                tile_behavior["dependence"] = dependence
                tile_behavior_list.append(tile_behavior)
            elif key_layer_info["type"] == "element_sum":
                input_buffer = buffer(self.simconfig_path, 1, in_buf_size)
                output_buffer = buffer(self.simconfig_path, 2, out_buf_size)
                out_channels = key_layer_info["Outputchannel"]
                input_buffer.calculate_buf_read_latency(rdata=2*out_channels*key_layer_info["Inputbit"]/8)
                output_buffer.calculate_buf_write_latency(wdata=out_channels*key_layer_info["outputbit"]/8)
                PE_latency = input_buffer.buf_rlatency + output_buffer.buf_wlatency
                tile_max_time = max(tile_max_time, PE_latency)
                # for element sum, there is only one tile
                tile_behavior = dict()
                tile_behavior["task_id"] = None
                tile_behavior["layer_id"] = layer_id
                tile_behavior["tile_id"] = len(tile_behavior_list)
                tile_behavior["target_tile_id"] = []
                tile_behavior["source_tile_id"] = []
                tile_behavior["merge_flag"] = True
                # traverse all tiles
                dependence = [
                    dict() for _ in range(key_layer_info["Outputsize"][0]*key_layer_info["Outputsize"][1])
                ]
                # for wait, output and latency
                for i in range(key_layer_info["Outputsize"][0]):
                    for j in range(key_layer_info["Outputsize"][1]):
                        # wait
                        dependence[i*key_layer_info["Outputsize"][1] + j]["wait"] = [[
                            i, j, 0, key_layer_info["Outputchannel"], \
                            key_layer_info["Inputbit"], key_layer_info["Outputchannel"],
                            None, None, -1, None
                        ] for _ in range(2)]
                        # output
                        dependence[i*key_layer_info["Outputsize"][1] + j]["output"] = [[
                            i, j, 0, key_layer_info["Outputchannel"],
                            key_layer_info["outputbit"], key_layer_info["Outputchannel"],
                            None, layer_id, -1, tile_behavior["tile_id"]
                        ]]
                        # latency
                        dependence[i*key_layer_info["Outputsize"][1] + j]["latency"] = float(PE_latency)
                        # drop
                        dependence[i*key_layer_info["Outputsize"][1] + j]["drop"] = copy.deepcopy(
                            dependence[i*key_layer_info["Outputsize"][1] + j]["wait"]
                        )
                tile_behavior["dependence"] = dependence
                tile_behavior_list.append(tile_behavior)
            else:
                raise NotImplementedError(f"{key_layer_info['type']} is not implemented")
        # max tile time
        print(f"max tile time: {tile_max_time} ns")
        print(f"need {len(tile_behavior_list)} tiles")
        # link tile, true tile id
        def _get_true_tile_id(layer_index):
            for tile_behavior in tile_behavior_list:
                if tile_behavior["layer_id"] == layer_index and tile_behavior["merge_flag"]:
                    return tile_behavior["tile_id"]
            return None
        for layer_id, (key_layer_info, key_layer) in enumerate(zip(key_layer_info_list, key_layers)):
            # for element sum
            input_index_list = [layer_id + i for i in key_layer_info["Inputindex"]]
            input_tile_list = [_get_true_tile_id(i) for i in input_index_list]
            target_tile_list = []
            if key_layer_info["type"] == "element_sum":
                assert len(input_index_list) == 2
                assert min(input_index_list) >= 0, "input index should be >= 0"
                # change for input
                for tile_behavior in tile_behavior_list:
                    if tile_behavior["layer_id"] == layer_id and tile_behavior["merge_flag"]:
                        target_tile_list.append(tile_behavior["tile_id"])
                        tile_behavior["source_tile_id"].append(input_tile_list[0])
                        tile_behavior["source_tile_id"].append(input_tile_list[1])
                        for i in range(len(tile_behavior["dependence"])):
                            tile_behavior["dependence"][i]["wait"][0][7] = copy.deepcopy(input_index_list[0])
                            tile_behavior["dependence"][i]["wait"][0][9] = copy.deepcopy(input_tile_list[0])
                            tile_behavior["dependence"][i]["wait"][1][7] = copy.deepcopy(input_index_list[1])
                            tile_behavior["dependence"][i]["wait"][1][9] = copy.deepcopy(input_tile_list[1])
                            # drop
                            tile_behavior["dependence"][i]["drop"] = copy.deepcopy(tile_behavior["dependence"][i]["wait"])
            else:
                # for conv, fc and pooling
                assert len(input_index_list) == 1
                if input_index_list[0] < 0:
                    # for the input
                    assert layer_id == 0
                    assert key_layer_info["type"] == "conv"
                    for tile_behavior in tile_behavior_list:
                        if tile_behavior["layer_id"] == 0 and not tile_behavior["merge_flag"]:
                            tile_behavior["source_tile_id"] = [-1]
                    continue
                # change for input
                for tile_behavior in tile_behavior_list:
                    if (key_layer_info["type"] == "pooling" and tile_behavior["layer_id"] == layer_id and \
                        tile_behavior["merge_flag"]) or \
                        (key_layer_info["type"] != "pooling" and tile_behavior["layer_id"] == layer_id and \
                        not tile_behavior["merge_flag"]
                    ):
                        target_tile_list.append(tile_behavior["tile_id"])
                        tile_behavior["source_tile_id"].append(input_tile_list[0])
                        for i in range(len(tile_behavior["dependence"])):
                            for wait in tile_behavior["dependence"][i]["wait"]:
                                wait[7] = input_index_list[0]
                                wait[9] = input_tile_list[0]
                            for drop in tile_behavior["dependence"][i]["drop"]:
                                drop[7] = input_index_list[0]
                                drop[9] = input_tile_list[0]
            # change for output
            for tile_behavior in tile_behavior_list:
                if tile_behavior["layer_id"] in input_index_list and tile_behavior["merge_flag"]:
                    tile_behavior["target_tile_id"] = tile_behavior["target_tile_id"] + target_tile_list
        # last layer
        for tile_behavior in tile_behavior_list:
            if tile_behavior["layer_id"] == len(key_layer_info_list) - 1 and tile_behavior["merge_flag"]:
                tile_behavior["target_tile_id"] = [-1]
        # modify for th first fc
        for layer_id, (key_layer_info, key_layer) in enumerate(zip(key_layer_info_list, key_layers)):
            if key_layer_info["type"] == "fc":
                input_index_list = [layer_id + i for i in key_layer_info["Inputindex"]]
                assert len(input_index_list) == 1
                # check for the input
                real_wait = []
                for tile_behavior in tile_behavior_list:
                    if tile_behavior["layer_id"] == input_index_list[0] and tile_behavior["merge_flag"]:
                        for i in range(len(tile_behavior["dependence"])):
                            real_wait = real_wait + tile_behavior["dependence"][i]["output"]
                        break
                # change for fc
                for tile_behavior in tile_behavior_list:
                    if tile_behavior["layer_id"] == layer_id and not tile_behavior["merge_flag"]:
                        assert len(tile_behavior["dependence"]) == 1
                        tile_behavior["dependence"][0]["wait"] = copy.deepcopy(real_wait)
                        tile_behavior["dependence"][0]["drop"] = copy.deepcopy(real_wait)
                break
        # return
        return tile_behavior_list
