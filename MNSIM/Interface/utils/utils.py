#-*-coding:utf-8-*-
"""
@FileName:
    utils.py
@Description:
    Interface utils
@Authors:
    Hanbo Sun(sun-hb17@mails.tsinghua.edu.cn)
@CreateTime:
    2021/11/19 14:37
"""
import collections
import configparser
import copy
import math
import os


def load_sim_config(SimConfig_path):
    """
    load SimConfig
    return hardware_config, xbar_column, tile_row, tile_column
    hardware_config:
        xbar_size: crossbar row
        input_bit: input activation bit
        weight_bit: weights bit
        quantize_bit: output activation bit
    xbar_column,
    tile_row and tile_column
    """
    xbar_config = configparser.ConfigParser()
    xbar_config.read(SimConfig_path, encoding="UTF-8")
    hardware_config = collections.OrderedDict()
    # xbar_size
    xbar_size = list(
        map(int, xbar_config.get("Crossbar level", "Xbar_Size").split(","))
    )
    xbar_row = xbar_size[0]
    xbar_column = xbar_size[1]
    hardware_config["xbar_row"] = xbar_row
    hardware_config["xbar_column"] = xbar_column
    # xbar bit
    xbar_bit = int(xbar_config.get("Device level", "Device_Level"))
    hardware_config["weight_bit"] = math.floor(math.log2(xbar_bit))
    # input bit and ADC bit
    ADC_choice = int(xbar_config.get("Interface level", "ADC_Choice"))
    DAC_choice = int(xbar_config.get("Interface level", "DAC_Choice"))
    temp_DAC_bit = int(xbar_config.get("Interface level", "DAC_Precision"))
    temp_ADC_bit = int(xbar_config.get("Interface level", "ADC_Precision"))
    ADC_precision_dict = {
        -1: temp_ADC_bit,
        1: 10,
        # reference: A 10b 1.5GS/s Pipelined-SAR ADC with Background Second-Stage
        # Common-Mode Regulation and Offset Calibration in 14nm CMOS FinFET
        2: 8,
        # reference: ISAAC: A Convolutional Neural Network Accelerator with
        # In-Situ Analog Arithmetic in Crossbars
        3: 8,
        # reference: A >3GHz ERBW 1.1GS/s 8b Two-Step SAR ADC
        # with Recursive-Weight DAC
        4: 6,
        # reference: Area-Efficient 1GS/s 6b SAR ADC
        # with Charge-Injection-Cell-Based DAC
        5: 8,  # ASPDAC1
        6: 6,  # ASPDAC2
        7: 4,  # ASPDAC3
    }
    DAC_precision_dict = {
        -1: temp_DAC_bit,
        1: 1,  # 1-bit
        2: 2,  # 2-bit
        3: 3,  # 3-bit
        4: 4,  # 4-bit
        5: 6,  # 6-bit
        6: 8,  # 8-bit
    }
    input_bit = DAC_precision_dict[DAC_choice]
    quantize_bit = ADC_precision_dict[ADC_choice]
    hardware_config["input_bit"] = input_bit
    hardware_config["quantize_bit"] = quantize_bit
    # group num
    # pe_group_num = int(xbar_config.get('Process element level', 'Group_Num'))
    tile_size = list(map(int, xbar_config.get("Tile level", "PE_Num").split(",")))
    tile_row = tile_size[0]
    tile_column = tile_size[1]
    hardware_config["tile_row"] = tile_row
    hardware_config["tile_column"] = tile_column
    return hardware_config

def _init_component(cls, config, name, base_cfg=None):
    """
    init a subinstance of cls, based on config and name
    """
    t_type = config.get(f"{name}_type", None)
    t_cfg = config.get(f"{name}_cfg", {})
    base_cfg = {} if base_cfg is None else base_cfg
    base_cfg.update(t_cfg)
    return cls.get_class_(t_type)(**base_cfg)

class SingleMap(object):
    """
    single direction map, one-multi pair
    """
    def __init__(self, pair=None):
        super(SingleMap, self).__init__()
        self._map = {}
        if pair is not None:
            self.add_more(pair)

    def clear(self):
        """
        clear _map
        """
        self._map.clear()

    def add_more(self, pair):
        """
        add one pair
        """
        assert isinstance(pair, tuple) and len(pair) == 2, \
            "pair should be a tuple with two elements"
        assert pair[0] not in self._map.keys(), "pair[0] should not be in _map keys"
        self._map[pair[0]] = copy.deepcopy(pair[1])

    def find(self, position, kv):
        """
        find the pair for kv based on the position
        """
        assert position in (0, 1), "position should be 0 or 1"
        if position == 0:
            if kv in self._map.keys():
                return self._map[kv]
            raise NotImplementedError(f"{kv} not in _map keys")
        if kv in self._map.values():
            for k, v in self._map.items():
                if v == kv:
                    return k
        raise NotImplementedError(f"{kv} not in _map values")


class DoubleMap(SingleMap):
    """
    double direction, one-one pair
    """
    def add_more(self, pair):
        """
        add pair
        """
        assert pair[1] not in self._map.values(), "pair[1] should not be in _map values"
        super(DoubleMap, self).add_more(pair)

def recursion_compare(a, b):
    a_type = type(a) if not isinstance(a, collections.OrderedDict) else type({})
    b_type = type(b) if not isinstance(b, collections.OrderedDict) else type({})
    if a_type != b_type:
        print("not same type")
        return False
    if a_type == dict:
        if not a.keys() == b.keys():
            print("not same keys")
            return False
        for k in a.keys():
            if not recursion_compare(a[k], b[k]):
                print(a[k], b[k])
                return False
        return True
    if a_type in [list, tuple]:
        if not len(a) == len(b):
            print("not same length")
            return False
        for i in range(len(a)):
            if not recursion_compare(a[i], b[i]):
                print(a[i], b[i])
                return False
        return True
    return a == b

def get_home_path():
    """
    get home path
    now is MNSIM/Interface/utils/utils.py
    return the home path of MNSIM_Python
    """
    return os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ))
