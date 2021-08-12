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
from MNSIM.Area_Model.Model_Area import Model_area
from MNSIM.Power_Model.Model_inference_power import Model_inference_power
from MNSIM.Energy_Model.Model_energy import Model_energy


def Data_clean():
    path = os.getcwd()
    NoC_file = path + '/MNSIM/NoC/'
    inj_file = 'inj_dir'
    log_file = 'log'
    res_file = 'Final_Results'
    files = os.listdir(NoC_file)
    for file in files:
        if file == inj_file:
            for target in os.listdir(NoC_file + inj_file):
                os.remove(NoC_file + inj_file + '/' + target)
        elif file == log_file:
            for target in os.listdir(NoC_file + log_file):
                os.remove(NoC_file + log_file + '/' + target)
        elif file == res_file:
            for target in os.listdir(NoC_file + res_file):
                os.remove(NoC_file + res_file + '/' + target)
        else:
            continue
    print("Removed unnecessary file.")


def main():
    # home_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    # print("home path", home_path)
    # if __name__=='__main__':
    #     home_path = os.path.dirname(os.path.dirname(os.getcwd()))
    #     print(1)
    # else:
    home_path = os.getcwd()
    # print(home_path)
    SimConfig_path = os.path.join(home_path, "SimConfig.ini")
    weights_file_path = os.path.join(home_path, "cifar10_vgg8_params.pth")
    # print(SimConfig_path)
    parser = argparse.ArgumentParser(description='MNSIM example')
    parser.add_argument("-AutoDelete", "--file_auto_delete", default=True,
                        help="Whether delete the unnecessary files automatically")
    # parser.add_argument("-NoC", "--NoC_computation", default=False,
    #                     help="Whether call booksim to compute the NoC part")
    parser.add_argument("-HWdes", "--hardware_description", default=SimConfig_path,
                        help="Hardware description file location & name, default:/MNSIM_Python/SimConfig.ini")
    parser.add_argument("-Weights", "--weights", default=weights_file_path,
                        help="NN model weights file location & name, default:/MNSIM_Python/cifar10_vgg8_params.pth")
    parser.add_argument("-NN", "--NN", default='vgg8',
                        help="NN model description (name), default: vgg8")
    parser.add_argument("-DisHW", "--disable_hardware_modeling", action='store_true', default=False,
                        help="Disable hardware modeling, default: false")
    parser.add_argument("-DisAccu", "--disable_accuracy_simulation", action='store_true', default=False,
                        help="Disable accuracy simulation, default: false")
    parser.add_argument("-SAF", "--enable_SAF", action='store_true', default=False,
                        help="Enable simulate SAF, default: false")
    parser.add_argument("-Var", "--enable_variation", action='store_true', default=False,
                        help="Enable simulate variation, default: false")
    parser.add_argument("-Rratio", "--enable_R_ratio", action='store_true', default=False,
                        help="Enable simulate the effect of R ratio, default: false")
    parser.add_argument("-FixRange", "--enable_fixed_Qrange", action='store_true', default=False,
                        help="Enable fixed quantization range (max value), default: false")
    parser.add_argument("-DisPipe", "--disable_inner_pipeline", action='store_true', default=False,
                        help="Disable inner layer pipeline in latency modeling, default: false")
    parser.add_argument("-D", "--device", default=0,
                        help="Determine hardware device for simulation, default: CPU")
    parser.add_argument("-DisModOut", "--disable_module_output", action='store_true', default=False,
                        help="Disable module simulation results output, default: false")
    parser.add_argument("-DisLayOut", "--disable_layer_output", action='store_true', default=False,
                        help="Disable layer-wise simulation results output, default: false")
    args = parser.parse_args()
    if args.file_auto_delete:
        print("use the root mode by 'sudo -s'")
        Data_clean()
    else:
        print("You should make sure that the files are removed which may cause confusions")
    print("Hardware description file location:", args.hardware_description)
    print("Software model file location:", args.weights)
    print("Whether perform hardware simulation:", not (args.disable_hardware_modeling))
    print("Whether perform accuracy simulation:", not (args.disable_accuracy_simulation))
    print("Whether consider SAFs:", args.enable_SAF)
    print("Whether consider variations:", args.enable_variation)
    if args.enable_fixed_Qrange:
        print("Quantization range: fixed range (depends on the maximum value)")
    else:
        print("Quantization range: dynamic range (depends on the data distribution)")
    # __TestInterface = TrainTestInterface(args.NN, 'MNSIM.Interface.cifar10', args.hardware_description,
    #                                      args.weights, args.device)
    __TestInterface = TrainTestInterface(network_module=args.NN, dataset_module='MNSIM.Interface.cifar10', SimConfig_path=args.hardware_description,
                                         weights_file=args.weights, device=args.device)
    structure_file = __TestInterface.get_structure()
    # weight = __TestInterface.get_net_bits()
    # print(structure_file)
    # print(__TestInterface.origin_evaluate(method = 'FIX_TRAIN', adc_action = 'SCALE'))
    # print(__TestInterface.set_net_bits_evaluate(weight, adc_action = 'SCALE'))
    TCG_mapping = TCG(structure_file, args.hardware_description, args.disable_inner_pipeline)
    # print(TCG_mapping.max_inbuf_size)
    # print(TCG_mapping.max_outbuf_size)
    if not (args.disable_hardware_modeling):
        __latency = Model_latency(NetStruct=structure_file, SimConfig_path=args.hardware_description,
                                  TCG_mapping=TCG_mapping)
        if not (args.disable_inner_pipeline):
            __latency.calculate_model_latency(mode=1)
            # __latency.calculate_model_latency_nopipe()
        else:
            __latency.calculate_model_latency_nopipe()
        print("========================Latency Results=================================")
        __latency.model_latency_output(not (args.disable_module_output), not (args.disable_layer_output))

        __area = Model_area(NetStruct=structure_file, SimConfig_path=args.hardware_description, TCG_mapping=TCG_mapping)
        print("========================Area Results=================================")
        __area.model_area_output(not (args.disable_module_output), not (args.disable_layer_output))
        __power = Model_inference_power(NetStruct=structure_file, SimConfig_path=args.hardware_description,
                                        TCG_mapping=TCG_mapping)
        print("========================Power Results=================================")
        __power.model_power_output(not (args.disable_module_output), not (args.disable_layer_output))
        __energy = Model_energy(NetStruct=structure_file, SimConfig_path=args.hardware_description,
                                TCG_mapping=TCG_mapping,
                                model_latency=__latency, model_power=__power)
        print("========================Energy Results=================================")
        __energy.model_energy_output(not (args.disable_module_output), not (args.disable_layer_output))

    if not (args.disable_accuracy_simulation):
        print("======================================")
        print("Accuracy simulation will take a few minutes on GPU")
        weight = __TestInterface.get_net_bits()
        weight_2 = weight_update(args.hardware_description, weight,
                                 is_Variation=args.enable_variation, is_SAF=args.enable_SAF, is_Rratio=args.enable_R_ratio)
        if not (args.enable_fixed_Qrange):
            print("Original accuracy:", __TestInterface.origin_evaluate(method='FIX_TRAIN', adc_action='SCALE'))
            print("PIM-based computing accuracy:", __TestInterface.set_net_bits_evaluate(weight_2, adc_action='SCALE'))
        else:
            print("Original accuracy:", __TestInterface.origin_evaluate(method='FIX_TRAIN', adc_action='FIX'))
            print("PIM-based computing accuracy:", __TestInterface.set_net_bits_evaluate(weight_2, adc_action='FIX'))

    # print(structure_file)


if __name__ == '__main__':
    # Data_clean()
    main()
