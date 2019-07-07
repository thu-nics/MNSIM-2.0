import sys
import os
import math
import random
import configparser as cp
import numpy as np
from MNSIM.Hardware_Model import *
from MNSIM.Hardware_Model.Crossbar import crossbar

test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),"SimConfig.ini")


class crossbar_accuracy():
    def __init__(self, SimConfig_path):
        self.SimConfig_path = SimConfig_path
        xbar = crossbar(SimConfig_path)
        print("Hardware config file is loaded:", SimConfig_path)
        ca_config = cp.ConfigParser()
        ca_config.read(SimConfig_path, encoding='UTF-8')
        self.SAF = list(map(int, ca_config.get('Device level', 'Device_SAF').split(',')))
        self.read_voltage = xbar.device_read_voltage
        print(self.read_voltage)
        # print(self.SAF)
        self.Load_Resistance = int(ca_config.get('Crossbar level', 'Load_Resistance'))  # TODO : change in crossbar.py
        if self.Load_Resistance == -1:
            self.Load_Resistance = 2e8
        self.wire_resistance = xbar.wire_resistance
        # if self.standard_wire_resistance == 0:
        #     self.wire_conduction = 0
        # else:
        #     self.wire_conduction = 1/self.standard_wire_resistance
        self.cell_type = xbar.cell_type
        self.standard_cell_resistance = xbar.device_resistance
        self.device_bit_level = xbar.device_bit_level
        self.decice_variation = xbar.decice_variation
        self.real_matrix = []
        self.row = 0
        self.column = 0
        print(self.row)
        print(self.column)
        self.enable_matrix = []

    def SAF_effect(self):
        bound1 = self.SAF[0]*0.01
        bound2 = self.SAF[1]*0.01 + bound1
        for i in range(self.row):
            temp = []
            for j in range(self.column):
                num = random.uniform(0,1)
                if num <= bound1:
                    temp.append(-1) # LRS
                elif num <= bound2:
                    temp.append(-2) # HRS
                else:
                    temp.append(1)
            self.enable_matrix.append(temp)



    def matrix_accuracy(self, read_matrix):
        ''' matrix is full of 0,1,2... '''
        # real_matrix = []
        self.column = len(read_matrix[0])
        self.row = len(read_matrix)
        self.SAF_effect()
        print(read_matrix)
        for i in range(self.row):
            temp = []
            for j in range(self.column):
                if self.enable_matrix[i][j] == 1:
                    resistance = self.standard_cell_resistance[read_matrix[i][j]]
                    temp_resistance = random.uniform(resistance*(1-0.01*self.decice_variation), resistance*(1+0.01*self.decice_variation))
                elif self.enable_matrix[i][j] == -1:
                    temp_resistance = self.standard_cell_resistance[-1]
                else:
                    temp_resistance = self.standard_cell_resistance[0]
                temp_resistance += self.wire_resistance*(self.row + j - i + 1)
                temp.append(1/temp_resistance)
            self.real_matrix.append(temp)


    def vector_accuracy(self, read_vector):
        ''' consider the effect of the ADC '''
        self.real_vector = []
        for j in range(self.column):
            voltage = 0
            for i in range(self.row):
                temp_voltage = self.read_voltage[read_vector[i]]
                print(temp_voltage)
                voltage += temp_voltage * self.Load_Resistance / (self.Load_Resistance + 1/self.real_matrix[i][j])
            self.real_vector.append(voltage)


    def Xbar_accuracy_output(self):
        print("--------------Accuracy model--------------")
        print("the real_matrix_weight of the crossbar is:\n")
        for i in range(self.row):
            print(self.real_matrix[i])
        print("the real vector of the output is:", self.real_vector)

        print(self.enable_matrix)


def Xbar_accuracy_test():
    print("load file:", test_SimConfig_path)
    _xbar_accuracy = crossbar_accuracy(test_SimConfig_path)
    print('------------')
    _xbar_accuracy.matrix_accuracy(read_matrix=[
        [1,0,1,1],
        [0,0,1,0],
        [1,0,1,0]
    ])
    _xbar_accuracy.vector_accuracy(read_vector=[0,0,1])
    _xbar_accuracy.Xbar_accuracy_output()


if __name__ == '__main__':
    Xbar_accuracy_test()









