#!/usr/bin/python
#-*-coding:utf-8-*-
import configparser as cp
import os
import math
test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),"SimConfig.ini")
	#Default SimConfig file path: MNSIM_Python/SimConfig.ini


class shiftreg(object):
	def __init__(self, SimConfig_path, max_shiftbase = None):
		# frequency unit: MHz
		shiftreg_config = cp.ConfigParser()
		shiftreg_config.read(SimConfig_path, encoding='UTF-8')
		self.shiftreg_tech = int(shiftreg_config.get('Digital module', 'ShiftReg_Tech'))
		if self.shiftreg_tech <= 0:
			self.shiftreg_tech = 65
		self.shiftreg_area = float(shiftreg_config.get('Digital module', 'ShiftReg_Area'))
		self.shiftreg_power = float(shiftreg_config.get('Digital module', 'ShiftReg_Power'))
		if max_shiftbase is None:
			self.max_shiftbase = 16
		else:
			self.max_shiftbase = max_shiftbase
		assert self.max_shiftbase > 0
		self.shiftreg_frequency =float(shiftreg_config.get('Digital module', 'Digital_Frequency'))
		if self.shiftreg_frequency is None:
			self.shiftreg_frequency = 100
		assert self.shiftreg_frequency > 0
		self.shiftreg_latency = 1.0/self.shiftreg_frequency
		self.calculate_shiftreg_power()
		self.shiftreg_energy = 0
		# print("shiftreg configuration is loaded")

	def calculate_shiftreg_area(self):
		# unit: um^2
		if self.shiftreg_area == 0:
			shiftreg_area_dict = {
				4: 1.42,#228.96,
				8: 1.42,#217.44,
				16:1.42#230.40
			}
			if self.max_shiftbase <= 4:
				self.shiftreg_area = shiftreg_area_dict[4]*pow((self.shiftreg_tech/65),2)
			elif self.max_shiftbase <= 8:
				self.shiftreg_area = shiftreg_area_dict[8]*pow((self.shiftreg_tech/65),2)
			else:
				self.shiftreg_area = shiftreg_area_dict[16]*pow((self.shiftreg_tech/65),2)


	def calculate_shiftreg_power(self):
		# unit: W
		if self.shiftreg_power == 0:
			shiftreg_power_dict = {
				4: 8.1e-7,#2.13e-4,
				8: 8.1e-7,#1.97e-4,
				16: 8.1e-7#1.24e-4
			}
			if self.max_shiftbase <= 4:
				self.shiftreg_power = shiftreg_power_dict[4]*pow((self.shiftreg_tech/65),2)
			elif self.max_shiftbase <= 8:
				self.shiftreg_power = shiftreg_power_dict[8]*pow((self.shiftreg_tech/65),2)
			else:
				self.shiftreg_power = shiftreg_power_dict[16]*pow((self.shiftreg_tech/65),2)


	def calculate_shiftreg_energy(self):
		assert self.shiftreg_power >= 0
		assert self.shiftreg_latency >= 0
		self.shiftreg_energy = self.shiftreg_latency * self.shiftreg_power

	def shiftreg_output(self):
		print("shiftreg_area:", self.shiftreg_area, "um^2")
		print("shiftreg_bitwidth:", self.max_shiftbase, "bit")
		print("shiftreg_power:", self.shiftreg_power, "W")
		print("shiftreg_latency:", self.shiftreg_latency, "ns")
		print("shiftreg_energy:", self.shiftreg_energy, "nJ")
	
def shiftreg_test():
	print("load file:",test_SimConfig_path)
	_shiftreg = shiftreg(test_SimConfig_path)
	_shiftreg.calculate_shiftreg_area()
	_shiftreg.calculate_shiftreg_power()
	_shiftreg.calculate_shiftreg_energy()
	_shiftreg.shiftreg_output()


if __name__ == '__main__':
	shiftreg_test()