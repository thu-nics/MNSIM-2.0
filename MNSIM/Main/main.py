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


def main():
    work_path = os.path.dirname(os.getcwd())
    print(work_path)
    sys.path.append(work_path)
    SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")
    weights_file_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                                          "cifar10_lenet_train_params.pth")
    # print(SimConfig_path)
    parser = argparse.ArgumentParser(description='MNSIM example')
    parser.add_argument("-H", "--hardware_description", default=SimConfig_path,
                        help="Hardware description file location & name, default:/MNSIM_Python_v1.5/SimConfig.ini")
    parser.add_argument("-S", "--software_model_description", default=weights_file_path,
                        help="Hardware description file location & name, default:/MNSIM_Python_v1.5/cifar10_lenet_train_params.pth")
    parser.add_argument("-DH", "--disable_hardware_modeling", action='store_true', default=False,
                        help="Disable hardware modeling, default: false")
    parser.add_argument("-DA", "--disable_accuracy_simulation", action='store_true', default=False,
                        help="Disable accuracy simulation, default: false")
    parser.add_argument("-DSAF", "--disable_SAF", action='store_true', default=False,
                        help="Disable simulate SAF, default: false")
    parser.add_argument("-DV", "--disable_variation", action='store_true', default=False,
                        help="Disable simulate variation, default: false")
    args = parser.parse_args()
    print(args.hardware_description)
    print(args.software_model_description)
    print(args.disable_hardware_modeling)

    print(os.path.join(os.path.dirname(os.getcwd()), "Interface/cifar10"))
    __TestInterface = TrainTestInterface('MNSIM.Interface.lenet', 'MNSIM.Interface.cifar10', SimConfig_path, weights_file_path, 'cpu')
    structure_file = __TestInterface.get_structure()
    weight = __TestInterface.get_net_bits()
    print(structure_file)
if __name__ == '__main__':
    main()