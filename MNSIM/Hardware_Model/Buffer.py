#!/usr/bin/python
# -*-coding:utf-8-*-
import configparser as cp
import os
import math

test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")


# Default SimConfig file path: MNSIM_Python/SimConfig.ini

def LinearCaculate(size, data):
    lower_bound = math.floor(size / 2048) - 1
    if lower_bound < 0:
        power = data[0] / 2048 * size
    else:
        power = (size - 2048 * 2**lower_bound) / (2048 * 2**lower_bound) * (data[lower_bound + 1] - data[lower_bound]) \
                + data[lower_bound]
    return power


class buffer(object):
    def __init__(self, SimConfig_path):
        buf_config = cp.ConfigParser()
        buf_config.read(SimConfig_path, encoding='UTF-8')
        self.buf_choice = int(buf_config.get('Architecture level', 'Buffer_Choice'))
        self.buf_area = float(buf_config.get('Architecture level', 'Buffer_Area'))
        # unit: nm
        self.buf_Tech = int(buf_config.get('Architecture level', 'Buffer_Technology'))
        if self.buf_Tech == 0:
            self.buf_Tech = 22
        # KB
        self.buf_Size = float(buf_config.get('Architecture level', 'Buffer_Capacity'))
        if self.buf_Size == 0:
            self.buf_Size = 16

        if self.buf_area == 0:
            self.calculate_buf_area()
        # TODO: 读取文件里的rpower和wpower是否合理
        self.buf_rpower = float(buf_config.get('Architecture level', 'Buffer_ReadPower'))
        self.buf_wpower = float(buf_config.get('Architecture level', 'Buffer_WritePower'))
        self.buf_bitwidth = float(buf_config.get('Architecture level', 'Buffer_Bitwidth'))


        # unit: Byte
        buf_width_dict = {0: 8, 1: 64 * 8, 2: 32 * 8, 3: 16 * 8}
        if self.buf_bitwidth in buf_width_dict:
            self.buf_bitwidth = buf_width_dict[0]
        buf_wfrequency_dict = {0: 1372, 1: 998.9, 2: 1589, 3: 1660}
        self.buf_rfrequency = float(buf_config.get(('Architecture level'), 'Buffer_ReadFrequency'))
        # unit: MHz
        if self.buf_rfrequency in buf_wfrequency_dict:
            self.buf_rfrequency = 1372
        buf_rfrequency_dict = {0: 1306, 1: 998.9, 2: 1589, 3: 1660}
        self.buf_wfrequency = float(buf_config.get(('Architecture level'), 'Buffer_WriteFrequency'))
        # unit: MHz
        if self.buf_wfrequency in buf_rfrequency_dict:
            self.buf_wfrequency = 1306
        self.buf_renergy = 0
        self.buf_rlatency = 0
        # self.calculate_buf_read_latency()
        self.buf_wenergy = 0
        self.buf_wlatency = 0
        # self.calculate_buf_write_energy()
        self.dynamic_buf_rpower = 0
        self.dynamic_buf_wpower = 0
        self.leakage_power = 0


    def calculate_buf_area(self):
        '''
        the buf_choice is about sram or dram, the area increases as linear.
        :return:
        '''
        # todo: change the unit into mm^2
        # unit: um^2
        ''' buffer technology '''
        sram_param_dict = {22: [0.0062, 1.0194], 32: [0.004, 0.6744], 40: [0.0019, 0.3222], 90: [0.0315, 5.2282]}
        dram_param_dict = {22: [0.0008, 0.0392], 32: [0.0012, 0.0461], 40: [0.0026, 0.0683], 90: [0.0128, 0.592]}
        if self.buf_choice != -1:
            assert self.buf_choice in [1, 2]
            assert self.buf_Tech in [22, 32, 40, 90], "the Technology of buffer is illegal"
            if self.buf_choice is 1:
                self.buf_area = (self.buf_Size * sram_param_dict[self.buf_Tech][0] + sram_param_dict[self.buf_Tech][1])*1e6

            elif self.buf_choice is 2:
                self.buf_area = (self.buf_Size * dram_param_dict[self.buf_Tech][0] + dram_param_dict[self.buf_Tech][1])*1e6



    def calculate_buf_read_power(self):
        '''
        buf_choice
        buf_Size
        buf_Tech
        :return:
        '''
        # unit: mW
        # todo: change the unit into nJ ??
        range = math.ceil(self.buf_Size/2048)
        sram_dynamic_rpower_dict = {
            22: [24.58644692636836, 21.256680531480068, 16.630583603036584, 14.955019387590541, 14.552721904914858,
                 14.257453645247653, 12.527513793888135, 15.518218262806235, 14.294173411058916, 12.51284044801866,
                 16.45505738866356, 15.398484327588525, 13.717732207478887, 11.419544538453302, 8.665835778978261],
            32: [29.63490174539241, 25.144562713942936, 19.62063471965257, 17.873127969312, 18.905345968508605,
                 18.018235899146582, 15.723097777716921, 20.71955405788429, 18.510839616813907, 16.334889657949883,
                 22.02049502929944, 20.681490961446165, 18.73011577040343, 16.190697604655696, 13.074672495123739],
            40: [29.63490174539241, 25.144562713942936, 19.62063471965257, 17.873127969312, 18.905345968508605,
                 18.018235899146582, 15.723097777716921, 20.71955405788429, 18.510839616813907, 16.334889657949883,
                 22.02049502929944, 20.681490961446165, 18.73011577040343, 16.190697604655696, 13.074672495123739],
            90: [52.383805276449415, 44.37317131426524, 34.84710077256134, 31.39500415877679, 34.77577468633599,
                 34.490551860570235, 30.263038598896543, 42.48749115726415, 38.142483189606374, 34.37125197169005,
                 49.56160095336047, 47.21163075497303, 44.02150725842623, 40.12512548739461, 35.312430579797535]
        }
        sram_leakage_power = {
            22: [2.42854, 3.54801, 5.61291, 11.4183, 23.084, 46.1475, 78.511, 171.338, 292.098, 530.104, 1173.47,
                 2131.32, 4046.65, 7874.45, 15530.2],
            32: [4.06907, 6.36693, 10.626, 21.734, 44.0492, 87.6308, 155.592, 324.812, 577.957, 1070.5, 2322.98,
                 4306.92, 8273.83, 16200.9, 32055.2],
            40: [5.90615, 9.34077, 15.7349, 32.1237, 65.0152, 129.268, 231.709, 478.67, 861.842, 1603.65, 3465.46,
                 6453.81, 12424.1, 24354.4, 48215.4],
            90: [2.69023, 4.27071, 7.21407, 14.7199, 29.8378, 59.7241, 106.782, 220.305, 396.275, 738.045, 1592.69,
                 2969.55, 5718.32, 11211.1, 22196.8]
        }
        dram_dynamic_rpower_dict = {
            22: [5.681848876595699, 6.9811730309205275, 11.519467006258639, 8.10154645283615, 6.581302792658147,
                 5.900831057759254, 5.725089599784555, 7.78228702489136, 7.480759423788605, 7.400832514933912,
                 7.200510637177808, 7.753335479486354, 8.786647382482153, 8.959563134053493, 9.216515473798376],
            32: [7.236587110619756, 8.707150015780643, 14.596740785106435, 17.136735119555528, 16.231649371927908,
                 12.893354037630123, 12.045127437155477, 10.688682659298424, 10.29624937270162, 10.196571034059529,
                 9.69737572047803, 11.936414739345064, 12.135244112283946, 12.452589307509026, 12.892932290091725],
            40: [6.00413669478087, 10.443264794158491, 17.518400785966282, 20.306613107796057, 33.742406512473444,
                 12.115899982438531, 13.870181710614307, 12.190883930115897, 10.291594768299696, 11.782946929021605,
                 11.253307875894988, 14.095464361477127, 14.502955682673132, 15.029589041095893, 15.727820241175348],
            90: [8.278404192149114, 14.261612256610029, 24.26827029012119, 28.076699764079873, 48.58402035645765,
                 33.94574450749927, 29.87777535577492, 25.69488976589619, 26.32735142879692, 24.10225640040032,
                 24.665088619910982, 25.958915613109475, 27.673132633122997, 33.60599015066913, 33.9591579033123]
        }
        dram_leakage_power = {
            22: [0.80335, 1.55197, 2.90346, 5.80693, 11.6139, 23.2277, 46.1862, 81.8447, 163.272, 326.543, 728.684,
                 1306.17, 2582.94, 5165.88, 10331.8],
            32: [1.61758, 3.13207, 5.88851, 11.5717, 21.4912, 42.9824, 85.9648, 170.404, 339.819, 679.638, 1476.06,
                 2687.83, 5375.66, 10751.3, 21502.6],
            40: [2.02492, 4.66081, 8.78423, 17.2699, 31.8008, 69.0798, 129.371, 256.446, 552.638, 1022.8, 2202.63,
                 4046.17, 8092.34, 16184.7, 32369.4],
            90: [0.93135, 2.14261, 4.03893, 7.94048, 14.6479, 29.2958, 58.5917, 117.183, 234.367, 466.596, 931.805,
                 1863.61, 3727.22, 6915.89, 14845]
        }
        if self.buf_choice != -1:
            assert self.buf_choice in [1, 2]
            if self.buf_choice is 1:
                ''' 线性插值 '''
                self.dynamic_buf_rpower = LinearCaculate(self.buf_Size, sram_dynamic_rpower_dict[self.buf_Tech])
                self.leakage_power = LinearCaculate(self.buf_Size, sram_leakage_power[self.buf_Tech])

            elif self.buf_choice is 2:
                self.dynamic_buf_rpower = LinearCaculate(self.buf_Size, dram_dynamic_rpower_dict[self.buf_Tech])
                self.leakage_power = LinearCaculate(self.buf_Size, dram_leakage_power[self.buf_Tech])
        if self.buf_rpower == 0:
            self.buf_rpower = self.dynamic_buf_rpower + self.leakage_power
        #
        #
        # buf_rpower_dict = {1: 0.06 * 1e-3
        #                    }
        # if self.buf_choice != -1:
        #     assert self.buf_choice in [1]
        #     self.buf_rpower = buf_rpower_dict[self.buf_choice]

    def calculate_buf_write_power(self):
        # unit: W
        sram_dynamic_wpower_dict = {
            22: [29.3559897079659, 29.713923128350277, 29.530313651210307, 25.06269706042524, 21.94130601547564,
                 20.231697781658134, 21.454993708946155, 19.37077951002227, 20.48340313543494, 21.06338832333147,
                 20.007023868974393, 20.658973707418422, 20.318119336830588, 18.301609401318288, 14.622542600538111],
            32: [35.39851092395947, 35.331527619066925, 35.27583072378753, 30.257008996510823, 28.580154490478073,
                 25.69427070302727, 27.232004710199497, 25.937265770163453, 26.75242923117652, 27.892681861309633,
                 26.92143086207399, 28.002222347127162, 28.115886394768236, 26.403908745931517, 22.502559172211537],
            40: [35.39851092395947, 35.331527619066925, 35.27583072378753, 30.257008996510823, 28.580154490478073,
                 25.69427070302727, 27.232004710199497, 25.937265770163453, 26.75242923117652, 27.892681861309633,
                 26.92143086207399, 28.002222347127162, 28.115886394768236, 26.403908745931517, 22.502559172211537],
            90: [62.47828345243852, 62.466561175135844, 63.247021535501034, 53.59512980692658, 52.799359202307436,
                 49.21837852434092, 52.83040146832812, 53.27702651853158, 55.388535979058624, 59.30383289275703,
                 60.73959918114395, 64.24097242908681, 66.62819573136711, 66.19188635483901, 61.62546412363937]
        }
        dram_dynamic_wpower_dict = {
            22: [3.706883046022031, 3.921325418623195, 5.054852386203041, 3.9908753357560434, 3.7851853095769874,
                 3.8453333860048495, 4.167187756509934, 4.911138366537072, 5.326949438806781, 5.74173116254764,
                 6.119545690840774, 6.81240067654919, 7.396402439733693, 7.9164496786638, 8.462919213404739],
            32: [5.27699161084737, 5.593601364333531, 7.639589728232387, 6.818362819272997, 7.883506280720917,
                 6.937121414334673, 7.591702891372499, 7.296593325244988, 7.71822168107739, 8.19333640175923,
                 8.405686722518405, 9.600735659072807, 10.402987161597224, 11.141248112537074, 11.936863723414977],
            40: [5.181210213039781, 7.064149083904494, 9.78507719601658, 8.742072902805608, 11.789490799851796,
                 6.817519206964049, 9.029084978228566, 8.517021276595743, 7.9714657263610675, 9.564827207786548,
                 9.820295942720763, 11.438135493494391, 12.50739816329699, 13.503543378995435, 14.602981442044417],
            90: [7.831898041136469, 10.770445086486584, 15.336791528950911, 14.213391345675653, 20.13615632388908,
                 15.593317605050055, 16.433687291751003, 15.603024766149424, 18.464324091331296, 17.9898653003729,
                 19.871953123298216, 22.13599430123531, 24.729953107531664, 28.518664298921703, 30.44195819387776]
        }

        if self.buf_choice != -1:
            assert self.buf_choice in [1, 2]
            if self.buf_choice is 1:
                ''' 线性插值 '''
                self.dynamic_buf_wpower = LinearCaculate(self.buf_Size, sram_dynamic_wpower_dict[self.buf_Tech])

            elif self.buf_choice is 2:
                self.dynamic_buf_wpower = LinearCaculate(self.buf_Size, dram_dynamic_wpower_dict[self.buf_Tech])

        if self.buf_wpower == 0:
            self.buf_wpower = self.dynamic_buf_wpower + self.leakage_power
        # buf_wpower_dict = {1: 0.02 * 1e-3
        #                    }
        # if self.buf_choice != -1:
        #     assert self.buf_choice in [1]
        #     self.buf_wpower = buf_wpower_dict[self.buf_choice]

    def calculate_buf_read_latency(self, rdata=0):
        # unit: ns, Byte
        self.buf_rlatency = rdata / self.buf_bitwidth / self.buf_rfrequency * 1e3

    def calculate_buf_write_latency(self, wdata=0):
        # unit: ns
        self.buf_wlatency = wdata / self.buf_bitwidth / self.buf_wfrequency * 1e3

    def calculate_buf_read_energy(self, rdata=0):
        # unit: nJ
        self.calculate_buf_read_latency(rdata)
        self.buf_renergy = self.buf_rlatency * self.buf_rpower

    def calculate_buf_write_energy(self, wdata=0):
        # unit: nJ
        self.calculate_buf_write_latency(wdata)
        self.buf_wenergy = self.buf_wlatency * self.buf_rpower

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
