#!/usr/bin/python
#-*-coding:utf-8-*-
import configparser as cp
import os
import math
test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),"SimConfig.ini")
	#Default SimConfig file path: MNSIM_Python/SimConfig.ini


class reg(object):
	def __init__(self, SimConfig_path, bitwidth = None):
		# frequency unit: MHz
		reg_config = cp.ConfigParser()
		reg_config.read(SimConfig_path, encoding='UTF-8')
		self.reg_tech = int(reg_config.get('Digital module', 'Reg_Tech'))
		if self.reg_tech <= 0:
			self.reg_tech = 65
		self.reg_area = float(reg_config.get('Digital module', 'Reg_Area'))
		self.reg_power = float(reg_config.get('Digital module', 'Reg_Power'))
		if bitwidth is None:
			self.bitwidth = 8
		else:
			self.bitwidth = bitwidth
		assert self.bitwidth > 0
		self.reg_frequency =float(reg_config.get('Digital module', 'Digital_Frequency'))
		if self.reg_frequency is None:
			self.reg_frequency = 100
		assert self.reg_frequency > 0
		self.reg_latency = 1.0/self.reg_frequency
		self.calculate_reg_power()
		self.reg_energy = 0
		# print("reg configuration is loaded")

	def calculate_reg_area(self):
		# unit: um^2
		if self.reg_area == 0:
			reg_area_dict = {
				4: 1.4256,
				8: 1.4256,
				16:1.4256
			}
			if self.bitwidth <= 4:
				self.reg_area = reg_area_dict[4]*pow((self.reg_tech/65),2)
			elif self.bitwidth <= 8:
				self.reg_area = reg_area_dict[8]*pow((self.reg_tech/65),2)
			else:
				self.reg_area = reg_area_dict[16]*pow((self.reg_tech/65),2)


	def calculate_reg_power(self):
		# unit: W
		if self.reg_power == 0:
			reg_power_dict = {
				4: 18.8e-9,
				8: 18.8e-9,
				16: 18.8e-9
			}
			if self.bitwidth <= 4:
				self.reg_power = reg_power_dict[4]*pow((self.reg_tech/65),2)
			elif self.bitwidth <= 8:
				self.reg_power = reg_power_dict[8]*pow((self.reg_tech/65),2)
			else:
				self.reg_power = reg_power_dict[16]*pow((self.reg_tech/65),2)


	def calculate_reg_energy(self):
		assert self.reg_power >= 0
		assert self.reg_latency >= 0
		self.reg_energy = self.reg_latency * self.reg_power

	def reg_output(self):
		print("reg_area:", self.reg_area, "um^2")
		print("reg_bitwidth:", self.bitwidth, "bit")
		print("reg_power:", self.reg_power, "W")
		print("reg_latency:", self.reg_latency, "ns")
		print("reg_energy:", self.reg_energy, "nJ")
	
def reg_test():
	print("load file:",test_SimConfig_path)
	_reg = reg(test_SimConfig_path)
	_reg.calculate_reg_area()
	_reg.calculate_reg_power()
	_reg.calculate_reg_energy()
	_reg.reg_output()


if __name__ == '__main__':
	reg_test()