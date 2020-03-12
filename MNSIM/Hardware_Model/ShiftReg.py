#!/usr/bin/python
#-*-coding:utf-8-*-
import configparser as cp
import os
import math
test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),"SimConfig.ini")
	#Default SimConfig file path: MNSIM_Python/SimConfig.ini


class shiftreg(object):
	def __init__(self, SimConfig_path, max_shiftbase = None, frequency = None):
		# frequency unit: MHz
		shiftreg_config = cp.ConfigParser()
		shiftreg_config.read(SimConfig_path, encoding='UTF-8')
		self.shiftreg_tech = int(shiftreg_config.get('Digital module', 'ShiftReg_Tech'))
		self.shiftreg_area = float(shiftreg_config.get('Digital module', 'ShiftReg_Area'))
		self.shiftreg_power = float(shiftreg_config.get('Digital module', 'ShiftReg_Power'))
		if max_shiftbase is None:
			self.max_shiftbase = 8
		else:
			self.max_shiftbase = max_shiftbase
		assert self.max_shiftbase > 0
		if frequency is None:
			self.shiftreg_frequency = 100
		else:
			self.shiftreg_frequency = frequency
		assert self.shiftreg_frequency > 0
		self.shiftreg_latency = 1.0/self.shiftreg_frequency
		self.calculate_shiftreg_power()
		self.shiftreg_energy = 0
		# print("shiftreg configuration is loaded")

	def calculate_shiftreg_area(self):
		# unit: um^2
		if self.shiftreg_area == 0:
			shiftreg_area_dict = {130: {4: 1503,
										8: 1582.08,
										16: 1783.68},
								  65: {4: 375.84,
									   8: 395.52,
									   16: 445.92},
								  55: {4: 269.09,
									   8: 283.18,
									   16: 319.27},
								  45: {4: 180.13,
									   8: 189.57,
									   16: 213.72},
								  28: {4: 69.74,
									   8: 73.39,
									   16: 82.75}
			}
			# TODO: add circuits simulation results
			if self.shiftreg_tech <= 28:
				if self.max_shiftbase <= 4:
					self.shiftreg_area = shiftreg_area_dict[28][4]
				elif self.max_shiftbase <= 8:
					self.shiftreg_area = shiftreg_area_dict[28][8]
				else:
					self.shiftreg_area = shiftreg_area_dict[28][16]
			elif self.shiftreg_tech <= 45:
				if self.max_shiftbase <= 4:
					self.shiftreg_area = shiftreg_area_dict[45][4]
				elif self.max_shiftbase <= 8:
					self.shiftreg_area = shiftreg_area_dict[45][8]
				else:
					self.shiftreg_area = shiftreg_area_dict[45][16]
			elif self.shiftreg_tech <= 55:
				if self.max_shiftbase <= 4:
					self.shiftreg_area = shiftreg_area_dict[55][4]
				elif self.max_shiftbase <= 8:
					self.shiftreg_area = shiftreg_area_dict[55][8]
				else:
					self.shiftreg_area = shiftreg_area_dict[55][16]
			elif self.shiftreg_tech <= 65:
				if self.max_shiftbase <= 4:
					self.shiftreg_area = shiftreg_area_dict[65][4]
				elif self.max_shiftbase <= 8:
					self.shiftreg_area = shiftreg_area_dict[65][8]
				else:
					self.shiftreg_area = shiftreg_area_dict[65][16]
			else:
				if self.max_shiftbase <= 4:
					self.shiftreg_area = shiftreg_area_dict[130][4]
				elif self.max_shiftbase <= 8:
					self.shiftreg_area = shiftreg_area_dict[130][8]
				else:
					self.shiftreg_area = shiftreg_area_dict[130][16]

	def calculate_shiftreg_power(self):
		# unit: W
		if self.shiftreg_power == 0:
			shiftreg_power_dict = {130: {4: 19.88e-4,
										8: 20.12e-4,
										16: 16.28e-4},
								  65: {4: 4.97e-4,
									   8: 5.03e-4,
									   16: 4.07e-4},
								  55: {4: 3.56e-4,
									   8: 3.60e-4,
									   16: 2.91e-4},
								  45: {4: 2.38e-4,
									   8: 2.41e-4,
									   16: 1.95e-4},
								  28: {4: 9.2e-5,
									   8: 0.93e-4,
									   16: 0.76e-4}
			}
			# TODO: add circuits simulation results
			if self.shiftreg_tech <= 28:
				if self.max_shiftbase <= 4:
					self.shiftreg_power = shiftreg_power_dict[28][4]
				elif self.max_shiftbase <= 8:
					self.shiftreg_power = shiftreg_power_dict[28][8]
				else:
					self.shiftreg_power = shiftreg_power_dict[28][16]
			elif self.shiftreg_tech <= 45:
				if self.max_shiftbase <= 4:
					self.shiftreg_power = shiftreg_power_dict[45][4]
				elif self.max_shiftbase <= 8:
					self.shiftreg_power = shiftreg_power_dict[45][8]
				else:
					self.shiftreg_power = shiftreg_power_dict[45][16]
			elif self.shiftreg_tech <= 55:
				if self.max_shiftbase <= 4:
					self.shiftreg_power = shiftreg_power_dict[55][4]
				elif self.max_shiftbase <= 8:
					self.shiftreg_power = shiftreg_power_dict[55][8]
				else:
					self.shiftreg_power = shiftreg_power_dict[55][16]
			elif self.shiftreg_tech <= 65:
				if self.max_shiftbase <= 4:
					self.shiftreg_power = shiftreg_power_dict[65][4]
				elif self.max_shiftbase <= 8:
					self.shiftreg_power = shiftreg_power_dict[65][8]
				else:
					self.shiftreg_power = shiftreg_power_dict[65][16]
			else:
				if self.max_shiftbase <= 4:
					self.shiftreg_power = shiftreg_power_dict[130][4]
				elif self.max_shiftbase <= 8:
					self.shiftreg_power = shiftreg_power_dict[130][8]
				else:
					self.shiftreg_power = shiftreg_power_dict[130][16]

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