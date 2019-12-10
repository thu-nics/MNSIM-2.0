#!/usr/bin/python
# -*-coding:utf-8-*-
import torch
import sys
import os
import math
import argparse
import numpy as np
import torch
import collections
import configparser
from importlib import import_module
from MNSIM.Interface.interface import *
from MNSIM.Accuracy_Model.Weight_update import weight_update
from MNSIM.Mapping_Model.Behavior_mapping import behavior_mapping
from MNSIM.Mapping_Model.Tile_connection_graph import TCG
from MNSIM.Latency_Model.Model_latency import Model_latency

def main():
    # home_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    # print("home path", home_path)
    # if __name__=='__main__':
    #     home_path = os.path.dirname(os.path.dirname(os.getcwd()))
    #     print(1)
    # else:
    home_path = os.getcwd()
    print(home_path)
    SimConfig_path = os.path.join(home_path, "SimConfig.ini")
    weights_file_path = os.path.join(home_path, "vgg8_channels_bit/vgg8_128_9_params.pth")
    # print(SimConfig_path)
    parser = argparse.ArgumentParser(description='MNSIM example')
    parser.add_argument("-HWdes", "--hardware_description", default=SimConfig_path,
                        help="Hardware description file location & name, default:/MNSIM_Python/SimConfig.ini")
    parser.add_argument("-Weights", "--weight", default=weights_file_path,
                        help="NN model weights file location & name, default:/MNSIM_Python/vgg8_channels_bit/vgg8_128_9_params.pth")
    parser.add_argument("-NN", "--NN", default='vgg8_128_9',
                        help="NN model description (name), default: vgg8_128_9")
    parser.add_argument("-DisHW", "--disable_hardware_modeling", action='store_true', default=False,
                        help="Disable hardware modeling, default: false")
    parser.add_argument("-DisAccu", "--disable_accuracy_simulation", action='store_true', default=False,
                        help="Disable accuracy simulation, default: false")
    parser.add_argument("-SAF", "--enable_SAF", action='store_true', default=False,
                        help="Enable simulate SAF, default: false")
    parser.add_argument("-Var", "--enable_variation", action='store_true', default=False,
                        help="Enable simulate variation, default: false")
    parser.add_argument("-FixRange", "--enable_fixed_Qrange", action='store_true', default=False,
                        help="Enable fixed quantization range (max value), default: false")
    parser.add_argument("-D", "--device", default=0,
                        help="Determine hardware device for simulation, default: CPU")
    parser.add_argument("-DisModOut", "--disable_module_output", action='store_true', default=False,
                        help="Disable module simulation results output, default: false")
    parser.add_argument("-DisLayOut", "--disable_layer_output", action='store_true', default=False,
                        help="Disable layer-wise simulation results output, default: false")
    args = parser.parse_args()
    print("Hardware description file location:", args.hardware_description)
    print("Software model file location:", args.weight)
    print("Whether perform hardware simulation:", not(args.disable_hardware_modeling))
    print("Whether perform accuracy simulation:", not(args.disable_accuracy_simulation))
    print("Whether consider SAFs:", args.enable_SAF)
    print("Whether consider variations:", args.enable_variation)
    if args.enable_fixed_Qrange:
        print("Quantization range: fixed range (depends on the maximum value)")
    else:
        print("Quantization range: dynamic range (depends on the data distribution)")
    __TestInterface = TrainTestInterface(args.NN, 'MNSIM.Interface.cifar10', args.hardware_description,
                                         args.weight, args.device)
    structure_file = __TestInterface.get_structure()
    weight = __TestInterface.get_net_bits()
    # print(structure_file)
    # print(__TestInterface.origin_evaluate(method = 'FIX_TRAIN', adc_action = 'SCALE'))
    # print(__TestInterface.set_net_bits_evaluate(weight, adc_action = 'SCALE'))

    if not(args.disable_hardware_modeling):
        __bm = behavior_mapping(structure_file,args.hardware_description)
        __bm.config_behavior_mapping()
        __bm.behavior_mapping_area()
        __bm.behavior_mapping_utilization()
        __bm.behavior_mapping_latency()
        __bm.behavior_mapping_power()
        __bm.behavior_mapping_energy()
        __bm.behavior_mapping_output(not(args.disable_module_output), not(args.disable_layer_output))
        __latency = Model_latency(structure_file, args.hardware_description)
        __latency.calculate_model_latency_2()
        __latency.model_latency_output()

    if not(args.disable_accuracy_simulation):
        print("===================================================")
        print("Accuracy simulation will take a few minutes on GPU")
        weight = __TestInterface.get_net_bits()
        weight_2 = weight_update(args.hardware_description, weight,
                                 is_Variation=args.enable_variation, is_SAF=args.enable_SAF)
        if not(args.enable_fixed_Qrange):
            print("Original accuracy:", __TestInterface.origin_evaluate(method = 'FIX_TRAIN', adc_action = 'SCALE'))
            print("PIM-based computing accuracy:", __TestInterface.set_net_bits_evaluate(weight_2,adc_action='SCALE'))
        else:
            print("Original accuracy:", __TestInterface.origin_evaluate(method='FIX_TRAIN', adc_action='FIX'))
            print("PIM-based computing accuracy:", __TestInterface.set_net_bits_evaluate(weight_2, adc_action='FIX'))



    # print(structure_file)
if __name__ == '__main__':
    main()
