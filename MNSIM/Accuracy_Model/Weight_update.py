import sys
import os
import math
import random
import configparser as cp
import numpy as np
from MNSIM.Hardware_Model import *
from MNSIM.Hardware_Model.Crossbar import crossbar
from MNSIM.Interface.interface import *

def weight_update(SimConfig_path, weight, is_SAF=1, is_Variation=1):
    # print("Hardware config file is loaded:", SimConfig_path)
    wu_config = cp.ConfigParser()
    wu_config.read(SimConfig_path, encoding='UTF-8')
    SAF_dist = list(map(int, wu_config.get('Device level', 'Device_SAF').split(',')))
    variation = float(wu_config.get('Device level', 'Device_Variation'))
    device_level = int(wu_config.get('Device level', 'Device_Level'))
    assert device_level >= 0, "NVM resistance level < 0"
    device_resistance = np.array(list(map(float, wu_config.get('Device level', 'Device_Resistance').split(','))))
    assert device_level == len(device_resistance), "NVM resistance setting error"
    # assume the resistance distribution of MLC is linear
    max_value = 2 ** math.floor(math.log2(device_level)) - 1
    interval = 0
    for i in range(len(device_resistance) - 1):
        interval += 1 / device_resistance[i + 1] - 1 / device_resistance[i]
    interval /= len(device_resistance) - 1
    for i in range(len(weight)):
        for label, value in weight[i].items():
            # print(value.shape)
            if (is_Variation):  # Consider the effect of variation
                for j in range(len(device_resistance)):
                    temp_var = np.random.normal(loc=device_resistance[j],
                                                scale=device_resistance[j] * variation / 100,
                                                size=value.shape)
                    temp_var = (1 / temp_var - 1 / device_resistance[j]) / interval
                    value = np.where(value == j, value + temp_var, value)
                    # print(temp_var)
            if (is_SAF):
                SAF = np.random.random_sample(value.shape)
                value = np.where(SAF < float(SAF_dist[0] / 100), 0, value)
                value = np.where(SAF > 1 - float(SAF_dist[-1] / 100), max_value, value)
                # print(value)
            weight[i].update({label: value.astype(float)})
    return weight

# class weight_update():
#     def __init__(self, SimConfig_path, weight, is_SAF=1, is_Variation=1):
#         self.SimConfig_path = SimConfig_path
#         print("Hardware config file is loaded:", SimConfig_path)
#         wu_config = cp.ConfigParser()
#         wu_config.read(SimConfig_path, encoding='UTF-8')
#         self.SAF = list(map(int, wu_config.get('Device level', 'Device_SAF').split(',')))
#         self.variation = float(wu_config.get('Device level', 'Device_Variation'))
#         self.device_level = int(wu_config.get('Device level', 'Device_Level'))
#         assert self.device_level >= 0, "NVM resistance level < 0"
#         self.device_resistance = np.array(list(map(float, wu_config.get('Device level', 'Device_Resistance').split(','))))
#         assert self.device_level == len(self.device_resistance), "NVM resistance setting error"
#         # assume the resistance distribution of MLC is linear
#         self.max_value = 2 ** math.floor(math.log2(self.device_level))-1
#         self.interval = 0
#         for i in range(len(self.device_resistance)-1):
#             self.interval += 1/self.device_resistance[i+1]-1/self.device_resistance[i]
#         self.interval /= len(self.device_resistance)-1
#         self.is_SAF = is_SAF
#         self.is_Variation = is_Variation
#         self.weight = weight
#
#
#     # def non_ideal_analyzer(self):
#     #     i = 0
#     #     for net in weight:
#     #         for label,value in net.items():
#     #             # print(value.shape)
#     #             if(self.is_Variation): # Consider the effect of variation
#     #                 for i in range(len(self.device_resistance)):
#     #                     temp_var = np.random.normal(loc=self.device_resistance[i],
#     #                                                 scale=self.device_resistance[i]*self.variation/100,
#     #                                                 size=value.shape)
#     #                     temp_var = (1/temp_var-1/self.device_resistance[i])/self.interval
#     #                     value = np.where(value==i,value+temp_var,value)
#     #                     # print(temp_var)
#     #             if(self.is_SAF):
#     #                 SAF = np.random.random_sample(value.shape)
#     #                 value = np.where(SAF<float(self.SAF[0]),0,value)
#     #                 value = np.where(SAF>1-float(self.SAF[-1]/100),self.max_value,value)
#     #                 # print(value)
#     #             weight[i].update({label:value})
#     #         i = i+1
#     #     return weight
#
#     def non_ideal_analyzer(self):
#         for i in range(len(weight)):
#             for label,value in weight[i].items():
#                 # print(value.shape)
#                 if(self.is_Variation): # Consider the effect of variation
#                     for j in range(len(self.device_resistance)):
#                         temp_var = np.random.normal(loc=self.device_resistance[j],
#                                                     scale=self.device_resistance[j]*self.variation/100,
#                                                     size=value.shape)
#                         temp_var = (1/temp_var-1/self.device_resistance[j])/self.interval
#                         value = np.where(value==j,value+temp_var,value)
#                         # print(temp_var)
#                 if(self.is_SAF):
#                     SAF = np.random.random_sample(value.shape)
#                     value = np.where(SAF<float(self.SAF[0]/100),0,value)
#                     value = np.where(SAF>1-float(self.SAF[-1]/100),self.max_value,value)
#                     # print(value)
#                 weight[i].update({label:value.astype(float)})
#         return weight

if __name__ == '__main__':
    SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")
    weights_file_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),"cifar10_lenet_train_params.pth")
    __TestInterface = TrainTestInterface('MNSIM.Interface.lenet', 'MNSIM.Interface.cifar10', SimConfig_path,
                                         weights_file_path, 0)
    structure_file = __TestInterface.get_structure()
    weight = __TestInterface.get_net_bits()
    weight_2 = weight_update(SimConfig_path, weight, is_Variation=1,is_SAF=0)
    # weight_2 = _test.non_ideal_analyzer()
    # print(weight_2[0]['split0_weight0_positive'].dtype)
    weight = __TestInterface.get_net_bits()
    # print(weight_2-weight)
    print(__TestInterface.set_net_bits_evaluate(weight_2))









