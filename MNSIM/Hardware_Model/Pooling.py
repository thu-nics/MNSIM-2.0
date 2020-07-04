#!/usr/bin/python
#-*-coding:utf-8-*-
import configparser as cp
import os
import math
test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),"SimConfig.ini")

class Pooling(object):
    def __init__(self, SimConfig_path):
        Pooling_config = cp.ConfigParser()
        Pooling_config.read(SimConfig_path, encoding='UTF-8')
        # self.Pooling_choice = Pooling_config.get()

        self.Pooling_unit_num = int(Pooling_config.get('Tile level', 'Pooling_unit_num'))
        if self.Pooling_unit_num == 0:
            self.Pooling_unit_num = 64
        self.Pooling_Tech = int(Pooling_config.get('Tile level', 'Pooling_Tech'))
        if self.Pooling_Tech == 0:
            self.Pooling_Tech = 65
        self.Pooling_area = float(Pooling_config.get('Tile level', 'Pooling_area'))

        self.Pooling_power = 0
        self.Pooling_energy = 0
        self.Pooling_area = 0
        self.Pooling_latency = 0

        self.Pooling_shape = list(map(int, Pooling_config.get('Tile level', 'Pooling_shape').split(',')))
        assert len(self.Pooling_shape) == 2, "Input of Pooling shape is illegal"
        self.Pooling_size = self.Pooling_shape[0] * self.Pooling_shape[1]
        if self.Pooling_size == 0:
            self.Pooling_size = 9

    def calculate_Pooling_area(self):
        # unit um^2 Technology & pooling size & unit num
        Pooling_area_dict = {
            65: {
                9: {
                   64:  91917.1562
                }
            }
        }
        if self.Pooling_Tech in [65]:
            if self.Pooling_size in [9]:
                ''' 线性插值 '''
                self.Pooling_area = Pooling_area_dict[self.Pooling_Tech][self.Pooling_size][self.Pooling_unit_num]*self.Pooling_unit_num/64
        else:
            self.Pooling_area = Pooling_area_dict[65][9][64] * pow((self.Pooling_Tech/65),2)*self.Pooling_unit_num/64


    def calculate_Pooling_power(self):
        # Unit : W
        Pooling_power_dict = {
            65: {
                9: {
                   64:  3.082*1e-3
                }
            }
        }
        if self.Pooling_Tech in [65]:
            if self.Pooling_size in [9]:
                ''' 线性插值 '''
                self.Pooling_power = Pooling_power_dict[self.Pooling_Tech][self.Pooling_size][self.Pooling_unit_num]
        else:
            self.Pooling_power = Pooling_power_dict[65][9][64] * pow((self.Pooling_Tech/65),2)

    def calculate_Pooling_latency(self, inchannel = 64, insize = 9):
        # Unit: ns
        Pooling_latency_dict = {
            65: {
                9: {
                    64: 100
                }
            }
        }
        self.Pooling_latency = 100*math.ceil(inchannel/self.Pooling_unit_num)*math.ceil(insize/self.Pooling_size)
        # if self.Pooling_Tech in [65]:
        #     if self.Pooling_size in [9]:
        #         ''' 线性插值 '''
        #         self.Pooling_latency = Pooling_latency_dict[self.Pooling_Tech][self.Pooling_size][self.Pooling_unit_num]*math.ceil(inchannel/self.Pooling_unit_num)

    def calculate_Pooling_energy(self):
        #unit mW
        self.Pooling_energy = self.Pooling_power * self.Pooling_latency

    def Pooling_output(self):
        # if self.Pooling_choice == -1:
        #     print("ADC_choice: User defined")
        # else:
        #     print("default configuration")
        print("Pooling_shape: %d x %d" %(self.Pooling_shape[0], self.Pooling_shape[1]))
        print("Pooling_area:", self.Pooling_area, "um^2")
        print("Pooling_power:", self.Pooling_power, "W")
        print("Pooling_latency:", self.Pooling_latency, "ns")
        print("Pooling_energy:", self.Pooling_energy, "nJ")


def Pooling_test():
    print("load file:", test_SimConfig_path)
    _Pooling = Pooling(test_SimConfig_path)
    _Pooling.calculate_Pooling_area()
    _Pooling.calculate_Pooling_power()
    _Pooling.calculate_Pooling_latency()
    _Pooling.calculate_Pooling_energy()
    _Pooling.Pooling_output()


if __name__ == '__main__':
    Pooling_test()
