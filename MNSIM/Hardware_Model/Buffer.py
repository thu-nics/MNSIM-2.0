#!/usr/bin/python
# -*-coding:utf-8-*-
import configparser as cp
import os
import math

test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")


# Default SimConfig file path: MNSIM_Python/SimConfig.ini


class buffer(object):
    def __init__(self, SimConfig_path, buf_level = 1, default_buf_size = 16):
        # buf_level: 1: PE input buffer, 2: tile output buffer, 3: DFU buffer
        buf_config = cp.ConfigParser()
        buf_config.read(SimConfig_path, encoding='UTF-8')
        self.buf_choice = int(buf_config.get('Architecture level', 'Buffer_Choice'))

        # unit: nm
        self.buf_Tech = int(buf_config.get('Architecture level', 'Buffer_Technology'))
        if self.buf_Tech == 0:
            self.buf_Tech = 65
        # buffer size unit: KB
        if buf_level == 1:
            self.buf_Size = float(buf_config.get('Process element level', 'PE_inBuf_Size'))
        elif buf_level == 2:
            self.buf_Size = float(buf_config.get('Process element level', 'Tile_outBuf_Size'))
        else:
            self.buf_Size = float(buf_config.get('Process element level', 'DFU_Buf_Size'))
        if self.buf_Size == 0:
            self.buf_Size = default_buf_size
        self.buf_bitwidth = int(buf_config.get('Architecture level', 'Buffer_Bitwidth'))
        if self.buf_bitwidth == 0:
            self.buf_bitwidth = 256 # bit
        self.index = 0
        if self.buf_Tech >= 90:
            if self.buf_Size <= 2:
                if self.buf_bitwidth <= 64:
                    self.index = 0
                elif self.buf_bitwidth <= 128:
                    self.index = 1
                elif self.buf_bitwidth <= 256:
                    self.index = 2
                else:
                    self.index = 3
            elif self.buf_Size <= 4:
                if self.buf_bitwidth <= 64:
                    self.index = 4
                elif self.buf_bitwidth <= 128:
                    self.index = 5
                elif self.buf_bitwidth <= 256:
                    self.index = 6
                else:
                    self.index = 7
            elif self.buf_Size <= 8:
                if self.buf_bitwidth <= 64:
                    self.index = 8
                elif self.buf_bitwidth <= 128:
                    self.index = 9
                elif self.buf_bitwidth <= 256:
                    self.index = 10
                else:
                    self.index = 11
            elif self.buf_Size <= 16:
                if self.buf_bitwidth <= 64:
                    self.index = 12
                elif self.buf_bitwidth <= 128:
                    self.index = 13
                elif self.buf_bitwidth <= 256:
                    self.index = 14
                else:
                    self.index = 15
            elif self.buf_Size <= 32:
                if self.buf_bitwidth <= 64:
                    self.index = 16
                elif self.buf_bitwidth <= 128:
                    self.index = 17
                elif self.buf_bitwidth <= 256:
                    self.index = 18
                else:
                    self.index = 19
            elif self.buf_Size <= 64:
                if self.buf_bitwidth <= 64:
                    self.index = 20
                elif self.buf_bitwidth <= 128:
                    self.index = 21
                elif self.buf_bitwidth <= 256:
                    self.index = 22
                else:
                    self.index = 23
            elif self.buf_Size <= 128:
                if self.buf_bitwidth <= 64:
                    self.index = 24
                elif self.buf_bitwidth <= 128:
                    self.index = 25
                elif self.buf_bitwidth <= 256:
                    self.index = 26
                else:
                    self.index = 27
            elif self.buf_Size <= 256:
                if self.buf_bitwidth <= 64:
                    self.index = 28
                elif self.buf_bitwidth <= 128:
                    self.index = 29
                elif self.buf_bitwidth <= 256:
                    self.index = 30
                else:
                    self.index = 31
            else:
                if self.buf_bitwidth <= 64:
                    self.index = 32
                elif self.buf_bitwidth <= 128:
                    self.index = 33
                elif self.buf_bitwidth <= 256:
                    self.index = 34
                else:
                    self.index = 35
        elif self.buf_Tech >= 65:
            if self.buf_Size <= 2:
                if self.buf_bitwidth <= 64:
                    self.index = 0+36
                elif self.buf_bitwidth <= 128:
                    self.index = 1+36
                elif self.buf_bitwidth <= 256:
                    self.index = 2+36
                else:
                    self.index = 3+36
            elif self.buf_Size <= 4:
                if self.buf_bitwidth <= 64:
                    self.index = 4+36
                elif self.buf_bitwidth <= 128:
                    self.index = 5+36
                elif self.buf_bitwidth <= 256:
                    self.index = 6+36
                else:
                    self.index = 7+36
            elif self.buf_Size <= 8:
                if self.buf_bitwidth <= 64:
                    self.index = 8+36
                elif self.buf_bitwidth <= 128:
                    self.index = 9+36
                elif self.buf_bitwidth <= 256:
                    self.index = 10+36
                else:
                    self.index = 11+36
            elif self.buf_Size <= 16:
                if self.buf_bitwidth <= 64:
                    self.index = 12+36
                elif self.buf_bitwidth <= 128:
                    self.index = 13+36
                elif self.buf_bitwidth <= 256:
                    self.index = 14+36
                else:
                    self.index = 15+36
            elif self.buf_Size <= 32:
                if self.buf_bitwidth <= 64:
                    self.index = 16+36
                elif self.buf_bitwidth <= 128:
                    self.index = 17+36
                elif self.buf_bitwidth <= 256:
                    self.index = 18+36
                else:
                    self.index = 19+36
            elif self.buf_Size <= 64:
                if self.buf_bitwidth <= 64:
                    self.index = 20+36
                elif self.buf_bitwidth <= 128:
                    self.index = 21+36
                elif self.buf_bitwidth <= 256:
                    self.index = 22+36
                else:
                    self.index = 23+36
            elif self.buf_Size <= 128:
                if self.buf_bitwidth <= 64:
                    self.index = 24+36
                elif self.buf_bitwidth <= 128:
                    self.index = 25+36
                elif self.buf_bitwidth <= 256:
                    self.index = 26+36
                else:
                    self.index = 27+36
            elif self.buf_Size <= 256:
                if self.buf_bitwidth <= 64:
                    self.index = 28+36
                elif self.buf_bitwidth <= 128:
                    self.index = 29+36
                elif self.buf_bitwidth <= 256:
                    self.index = 30+36
                else:
                    self.index = 31+36
            else:
                if self.buf_bitwidth <= 64:
                    self.index = 32+36
                elif self.buf_bitwidth <= 128:
                    self.index = 33+36
                elif self.buf_bitwidth <= 256:
                    self.index = 34+36
                else:
                    self.index = 35+36
        else:
            if self.buf_Size <= 2:
                if self.buf_bitwidth <= 64:
                    self.index = 0+72
                elif self.buf_bitwidth <= 128:
                    self.index = 1+72
                elif self.buf_bitwidth <= 256:
                    self.index = 2+72
                else:
                    self.index = 3+72
            elif self.buf_Size <= 4:
                if self.buf_bitwidth <= 64:
                    self.index = 4+72
                elif self.buf_bitwidth <= 128:
                    self.index = 5+72
                elif self.buf_bitwidth <= 256:
                    self.index = 6+72
                else:
                    self.index = 7+72
            elif self.buf_Size <= 8:
                if self.buf_bitwidth <= 64:
                    self.index = 8+72
                elif self.buf_bitwidth <= 128:
                    self.index = 9+72
                elif self.buf_bitwidth <= 256:
                    self.index = 10+72
                else:
                    self.index = 11+72
            elif self.buf_Size <= 16:
                if self.buf_bitwidth <= 64:
                    self.index = 12+72
                elif self.buf_bitwidth <= 128:
                    self.index = 13+72
                elif self.buf_bitwidth <= 256:
                    self.index = 14+72
                else:
                    self.index = 15+72
            elif self.buf_Size <= 32:
                if self.buf_bitwidth <= 64:
                    self.index = 16+72
                elif self.buf_bitwidth <= 128:
                    self.index = 17+72
                elif self.buf_bitwidth <= 256:
                    self.index = 18+72
                else:
                    self.index = 19+72
            elif self.buf_Size <= 64:
                if self.buf_bitwidth <= 64:
                    self.index = 20+72
                elif self.buf_bitwidth <= 128:
                    self.index = 21+72
                elif self.buf_bitwidth <= 256:
                    self.index = 22+72
                else:
                    self.index = 23+72
            elif self.buf_Size <= 128:
                if self.buf_bitwidth <= 64:
                    self.index = 24+72
                elif self.buf_bitwidth <= 128:
                    self.index = 25+72
                elif self.buf_bitwidth <= 256:
                    self.index = 26+72
                else:
                    self.index = 27+72
            elif self.buf_Size <= 256:
                if self.buf_bitwidth <= 64:
                    self.index = 28+72
                elif self.buf_bitwidth <= 128:
                    self.index = 29+72
                elif self.buf_bitwidth <= 256:
                    self.index = 30+72
                else:
                    self.index = 31+72
            else:
                if self.buf_bitwidth <= 64:
                    self.index = 32+72
                elif self.buf_bitwidth <= 128:
                    self.index = 33+72
                elif self.buf_bitwidth <= 256:
                    self.index = 34+72
                else:
                    self.index = 35+72

        if buf_level == 1:
            self.buf_area = float(buf_config.get('Process element level', 'PE_inBuf_Area'))
        elif buf_level == 2:
            self.buf_area = float(buf_config.get('Process element level', 'Tile_outBuf_Area'))
        else:
            self.buf_area = float(buf_config.get('Process element level', 'DFU_Buf_Area'))
        if self.buf_area == 0:
            self.calculate_buf_area()
        # TODO: 读取文件里的rpower和wpower是否合理
        self.buf_rpower = float(buf_config.get('Architecture level', 'Buffer_ReadPower'))
        self.buf_wpower = float(buf_config.get('Architecture level', 'Buffer_WritePower'))
        #self.buf_frequency = float(buf_config.get('Architecture level', 'Buffer_Frequency'))

        self.buf_renergy = 0
        self.buf_rlatency = 0
        # self.calculate_buf_read_latency()
        self.buf_wenergy = 0
        self.buf_wlatency = 0
        # self.calculate_buf_write_energy()
        self.dynamic_buf_rpower = 0
        self.dynamic_buf_wpower = 0
        self.leakage_power = 0
        sram_cycle = [0.429117, 0.516288, 0.516288, -1, 0.493667, 0.513399, 0.513399, 0.731628, 0.545851, 0.545851,
                      0.545851, 0.73042,
                      0.888161, 0.888161, 0.888161, 0.880783, 0.970756, 0.970756, 0.970756, 1.77142, 1.8639, 1.8639,
                      1.8639, 1.8639,
                      2.03915, 2.03915, 2.03915, 2.03915, 5.06442, 5.06442, 5.06442, 5.06442, 5.06442, 5.06442, 5.43619,
                      5.43619,
                      0.294314, 0.354413, 0.348287, -1, 0.371007, 0.371007, 0.36528, 0.555128, 0.40418, 0.40418,
                      0.398423, 0.576657,
                      0.61696, 0.61696, 0.61696, 0.611204, 0.685607, 0.685607, 0.685607, 1.28541, 1.36208, 1.36208,
                      1.36208, 1.36208,
                      1.50531, 1.50531, 1.50531, 1.50531, 3.66005, 3.85607, 3.85607, 3.85607, 3.85607, 3.85607, 4.16804,
                      4.16804,
                      0.161935, 0.218222, 0.214715, -1, 0.220376, 0.220376, 0.217178, 0.388917, 0.240058, 0.240058,
                      0.398597, 0.394621,
                      0.419332, 0.419332, 0.419332, 0.416098, 0.459782, 0.459782, 0.459782, 1.0329, 1.08402, 1.08402,
                      1.08402, 1.08402,
                      1.17245, 1.17245, 1.17245, 1.17245, 0.546324, 0.729275, 3.45785, 3.45784, 1.17245, 3.45784,
                      3.45784, 3.6791
                      ]  # uniit: ns
        self.buf_cycle = 20#sram_cycle[self.index]
        assert self.buf_cycle != -1, "Error: No available for 2KB SRAM buffer with 512-bit bus bitwidth"
        sram_leakage_power = [1.006428, 1.136656, 1.253224, -1, 1.95684, 2.15962, 2.28782, 2.62008, 4.1222, 4.18056,
                              4.3319, 4.55214,
                              8.86342, 9.07178, 9.65236, 11.29236, 17.46566, 17.73578, 18.44038, 19.29436, 32.1096,
                              32.4554, 33.348, 35.5564,
                              63.688, 64.1578, 65.2978, 68.0018, 121.8756, 122.4862, 126.032, 127.5844, 243.484,
                              244.982, 245.722, 250.266,
                              4.33154, 4.5899, 5.99122, -1, 7.75758, 8.25868, 9.76298, 14.09702, 14.98076, 15.5835,
                              17.29178, 20.9042,
                              26.6168, 27.3042, 29.192, 34.5, 52.3788, 53.2686, 55.5624, 58.7496, 96.3408, 97.4886,
                              100.408, 107.6106,
                              191.03, 192.5814, 196.3112, 205.136, 367.718, 367.692, 372.76, 384.374, 730.276, 735.414,
                              737.946, 752.802,
                              3.06844, 3.22544, 4.1977, -1, 5.51418, 5.85956, 6.90244, 9.92618, 10.65116, 11.06626,
                              11.4876, 14.86772,
                              19.25676, 19.72618, 21.0248, 24.6852, 37.9354, 38.5436, 40.1206, 42.1506, 69.7894,
                              70.5706, 72.5732, 77.5184,
                              138.4138, 139.4718, 142.0298, 148.0884, 299.266, 299.666, 269.828, 277.794, 551.922,
                              532.732, 539.688, 544.616] # unit:mW
        self.leakage_power = sram_leakage_power[self.index]
        assert self.leakage_power != -1, "Error: No available for 2KB SRAM buffer with 512-bit bus bitwidth"


    def calculate_buf_area(self):
        '''
        the buf_choice is about sram or dram, the area increases as linear.
        :return:
        '''
        # unit: um^2
        # todo: Add DRAM parameters
        if self.buf_Size == 0:
            self.buf_area = 0
        else:
            sram_area = [0.0405803,0.0944387,0.170796,-1,0.0686947,0.129533,0.21469158,0.544121685,0.156974886,0.199132286,0.301779123,0.629568,
                         0.616313686,0.781358547,1.190747266,2.277664127,1.060004216,1.271763571,1.774928364,3.368257932,1.974404875,2.239130702,2.897282324,4.503881809,
                         3.583918894,3.942740844,4.787940384,6.782865957,6.974574682,7.387286214,8.510494573,11.13794477,14.22112082,15.77955259,15.1903479,18.56649221,
                         0.069914533,0.13213,0.266413,-1,0.117554,0.172761,0.319445,0.885538,0.185984,0.253431,0.424684,0.98738,
                         0.321651,0.407856,0.621693,1.18947,0.552979,0.663547,0.92629, 1.75898,1.02955,1.16829,1.51204,2.35122,
                         1.86953,2.05688,2.49820595,3.532879835,4.110732386,3.853902632,4.440562411,5.812866337,7.345004718,8.232069955,7.924741251,9.68755797,
                         0.026389593,0.049106366,0.099338763,-1,0.043660718,0.064295773,0.119224229,0.333979075,0.06910098,0.094371317,0.177026169,0.372578216,
                         0.12167763,0.154207863,0.234897385,0.426436442,0.209317302,0.251074,0.350283787,0.664606538,0.389828369,0.442063443,0.571793236,0.88844989,
                         0.707812758,0.779322414,0.945248378,1.335949489,1.692348078,1.758800264,1.680181012,2.198122156,2.972110865,3.116036246,3.763339813,3.665484947]
            self.buf_area = sram_area[self.index]*1e6
            assert self.buf_area != -1, "Error: No available for 2KB SRAM buffer with 512-bit bus bitwidth"



    def calculate_buf_read_power(self):
        '''
        buf_choice
        buf_Size
        buf_Tech
        :return:
        '''
        # unit: mW
        # todo: Add DRAM parameters
        if self.buf_Size == 0:
            self.buf_rpower = 0
        else:
            sram_dynamic_read_energy = [0.0075695,0.0204901,0.0374838,-1,0.00854257,0.0227852,0.041063,0.12054,0.018382,0.0275777,0.0484019,0.127604,
                                       0.044837,0.0706276,0.131809,0.295237,0.0618356,0.0943086,0.169224,0.411876,0.103042,0.14809,0.247932,0.485822,
                                       0.156421,0.21497,0.342013,0.634498,0.27502,0.358447,0.535426,0.927837,0.356492,0.508902,0.795817,1.29724,0.00823002,
                                       0.0183766,0.0407038,-1,0.0120839,0.0211627,0.0454619,0.137905,0.0158937,0.026906,0.0551001,0.147335,0.0248548,0.0394126,
                                       0.0739843,0.166219,0.0349618,0.053362,0.0956746,0.232081,0.0583935,0.0838554,0.140319,0.274961,0.0905117,0.123634,0.19553,
                                       0.361147,0.131652,0.206955,0.307099,0.52924,0.205608,0.291566,0.462242,0.746234,0.00366611,0.00805114,0.0174977,-1,0.00545155,
                                       0.00929243,0.0195678,0.0587335,0.00717156,0.0118251,0.0272345,0.0628241,0.0113831,0.0175339,0.0321178,0.0709932,0.0160867,
                                       0.0238524,0.0416899,0.0996046,0.0272693,0.0380164,0.0618288,0.118533,0.0426012,0.0565679,0.0868665,0.156594,0.0624646,
                                       0.097861,0.138328,0.231921,0.0933836,0.131128,0.210481,0.330414] # unit: nJ
            dynamic_read_energy = sram_dynamic_read_energy[self.index]
            assert dynamic_read_energy != -1, "Error: No available for 2KB SRAM buffer with 512-bit bus bitwidth"
            self.dynamic_buf_rpower = dynamic_read_energy/self.buf_cycle*1e3
            #self.dynamic_buf_rpower = 0
            self.buf_rpower = self.dynamic_buf_rpower + self.leakage_power


    def calculate_buf_write_power(self):
        # unit: mW
        # todo: Add DRAM parameters
        if self.buf_Size == 0:
            self.buf_wpower = 0
        else:
            sram_dynamic_write_energy = [0.0131361,0.0223358,0.0422325,-1,0.0199484,0.0271014,0.0516509,0.130038,0.0211344,0.0368348,
                                         0.0706683,0.14878,0.0437094,0.0779207,0.155944,0.345919,0.0601423,0.109611,0.218248,0.460145,0.08827501,
                                         0.144703,0.278356,0.583869,0.116896,0.209321,0.404114,0.8321,0.162096,0.279398,0.524128,1.05204,0.243037,
                                         0.429853,0.775468,1.54815,0.00983737,0.0198855,0.0427535,-1,0.0129198,0.0243774,0.0516229,0.142004,
                                         0.0177355,0.0335322,0.0694838,0.159657,0.0237539,0.0430962,0.0872368,0.194986,0.0330736,0.0610694,0.122573,
                                         0.258586,0.0450215,0.0800791,0.155734,0.328758,0.0643685,0.116708,0.22704,0.469528,0.0950392,0.154668,0.293248,
                                         0.592261,0.133284,0.23928,0.435794,0.873636,0.00431575,0.00864673,0.0182943,-1,0.00571403,0.0105917,0.0221228,
                                         0.0603267,0.00778673,0.0145317,0.0298331,0.0679341,0.010522,0.0187642,0.0375312,0.0831368,0.0145269,0.0264935,
                                         0.0527328,0.110431,0.0199489,0.0348969,0.067111,0.140623,0.0282674,0.0506538,0.0977918,0.201198,0.0595076,0.108967,
                                         0.1265,0.253771,0.0789223,0.102461,0.198653,0.374836] # unit: nJ
            dynamic_write_energy = sram_dynamic_write_energy[self.index]
            assert dynamic_write_energy != -1, "Error: No available for 2KB SRAM buffer with 512-bit bus bitwidth"
            self.dynamic_buf_wpower = dynamic_write_energy / self.buf_cycle * 1e3
            #self.dynamic_buf_wpower = 0
            self.buf_wpower = self.dynamic_buf_wpower + self.leakage_power


    def calculate_buf_read_latency(self, rdata=0):
        # unit: ns, Byte(data)
        self.buf_rlatency = math.ceil(rdata*8/self.buf_bitwidth)*self.buf_cycle

    def calculate_buf_write_latency(self, wdata=0):
        # unit: ns, Byte(data)
        self.buf_wlatency = math.ceil(wdata*8/self.buf_bitwidth)*self.buf_cycle

    def calculate_buf_read_energy(self, rdata=0):
        # unit: nJ
        sram_dynamic_read_energy = [0.0075695, 0.0204901, 0.0374838, -1, 0.00854257, 0.0227852, 0.041063, 0.12054,
                                    0.018382, 0.0275777, 0.0484019, 0.127604,
                                    0.044837, 0.0706276, 0.131809, 0.295237, 0.0618356, 0.0943086, 0.169224, 0.411876,
                                    0.103042, 0.14809, 0.247932, 0.485822,
                                    0.156421, 0.21497, 0.342013, 0.634498, 0.27502, 0.358447, 0.535426, 0.927837,
                                    0.356492, 0.508902, 0.795817, 1.29724, 0.00823002,
                                    0.0183766, 0.0407038, -1, 0.0120839, 0.0211627, 0.0454619, 0.137905, 0.0158937,
                                    0.026906, 0.0551001, 0.147335, 0.0248548, 0.0394126,
                                    0.0739843, 0.166219, 0.0349618, 0.053362, 0.0956746, 0.232081, 0.0583935, 0.0838554,
                                    0.140319, 0.274961, 0.0905117, 0.123634, 0.19553,
                                    0.361147, 0.131652, 0.206955, 0.307099, 0.52924, 0.205608, 0.291566, 0.462242,
                                    0.746234, 0.00366611, 0.00805114, 0.0174977, -1, 0.00545155,
                                    0.00929243, 0.0195678, 0.0587335, 0.00717156, 0.0118251, 0.0272345, 0.0628241,
                                    0.0113831, 0.0175339, 0.0321178, 0.0709932, 0.0160867,
                                    0.0238524, 0.0416899, 0.0996046, 0.0272693, 0.0380164, 0.0618288, 0.118533,
                                    0.0426012, 0.0565679, 0.0868665, 0.156594, 0.0624646,
                                    0.097861, 0.138328, 0.231921, 0.0933836, 0.131128, 0.210481, 0.330414]  # unit: nJ
        dynamic_read_energy = sram_dynamic_read_energy[self.index]
        assert dynamic_read_energy != -1, "Error: No available for 2KB SRAM buffer with 512-bit bus bitwidth"
        self.buf_renergy = (dynamic_read_energy+self.buf_cycle*self.leakage_power/1e3)*math.ceil(rdata*8/self.buf_bitwidth)

    def calculate_buf_write_energy(self, wdata=0):
        # unit: nJ
        sram_dynamic_write_energy = [0.0131361, 0.0223358, 0.0422325, -1, 0.0199484, 0.0271014, 0.0516509, 0.130038,
                                     0.0211344, 0.0368348,
                                     0.0706683, 0.14878, 0.0437094, 0.0779207, 0.155944, 0.345919, 0.0601423, 0.109611,
                                     0.218248, 0.460145, 0.08827501,
                                     0.144703, 0.278356, 0.583869, 0.116896, 0.209321, 0.404114, 0.8321, 0.162096,
                                     0.279398, 0.524128, 1.05204, 0.243037,
                                     0.429853, 0.775468, 1.54815, 0.00983737, 0.0198855, 0.0427535, -1, 0.0129198,
                                     0.0243774, 0.0516229, 0.142004,
                                     0.0177355, 0.0335322, 0.0694838, 0.159657, 0.0237539, 0.0430962, 0.0872368,
                                     0.194986, 0.0330736, 0.0610694, 0.122573,
                                     0.258586, 0.0450215, 0.0800791, 0.155734, 0.328758, 0.0643685, 0.116708, 0.22704,
                                     0.469528, 0.0950392, 0.154668, 0.293248,
                                     0.592261, 0.133284, 0.23928, 0.435794, 0.873636, 0.00431575, 0.00864673, 0.0182943,
                                     -1, 0.00571403, 0.0105917, 0.0221228,
                                     0.0603267, 0.00778673, 0.0145317, 0.0298331, 0.0679341, 0.010522, 0.0187642,
                                     0.0375312, 0.0831368, 0.0145269, 0.0264935,
                                     0.0527328, 0.110431, 0.0199489, 0.0348969, 0.067111, 0.140623, 0.0282674,
                                     0.0506538, 0.0977918, 0.201198, 0.0595076, 0.108967,
                                     0.1265, 0.253771, 0.0789223, 0.102461, 0.198653, 0.374836]  # unit: nJ
        dynamic_write_energy = sram_dynamic_write_energy[self.index]
        assert dynamic_write_energy != -1, "Error: No available for 2KB SRAM buffer with 512-bit bus bitwidth"
        self.buf_wenergy = (dynamic_write_energy + self.buf_cycle * self.leakage_power / 1e3) * math.ceil(
            wdata * 8 / self.buf_bitwidth)

    def buf_output(self):
        if self.buf_choice == -1:
            print("buf_choice: User defined")
        else:
            print("buf_choice:", self.buf_choice)
        print("buf_Size:", self.buf_Size, "bytes")
        print("buf_Tech:", self.buf_Tech, "nm")
        print("buf_area:", self.buf_area, "um^2")
        print("buf_read_power:", self.buf_rpower, "W")
        print("buf_dynamic_rpower:", self.dynamic_buf_rpower, "mW")
        print("buf_read_energy:", self.buf_renergy, "nJ")
        print("buf_read_latency:", self.buf_rlatency, "ns")
        print("buf_write_power:", self.buf_wpower, "W")
        print("buf_dynamic_wpower:", self.dynamic_buf_wpower, "mW")
        print("buf_write_energy:", self.buf_wenergy, "nJ")
        print("buf_leakage_power:", self.leakage_power, "mW")
        print("buf_write_latency:", self.buf_wlatency, "ns")


def buf_test():
    print("load file:", test_SimConfig_path)
    _buf = buffer(test_SimConfig_path)
    _buf.calculate_buf_area()
    _buf.calculate_buf_read_power()
    _buf.calculate_buf_read_latency()
    _buf.calculate_buf_read_energy()
    _buf.calculate_buf_write_power()
    _buf.calculate_buf_write_latency()
    _buf.calculate_buf_write_energy()
    _buf.buf_output()


if __name__ == '__main__':
    buf_test()
