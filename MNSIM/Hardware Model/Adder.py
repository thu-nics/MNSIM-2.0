#!/usr/bin/python
#-*-coding:utf-8-*-
import configparser as cp
import os
import math
test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),"SimConfig.ini")
	#Default SimConfig file path: MNSIM_Python/SimConfig.ini


class adder(object):
	def __init__(self, SimConfig_path, bitwidth = None, frequency = None):
		# frequency unit: MHz
		adder_config = cp.ConfigParser()
		adder_config.read(SimConfig_path)
		self.adder_tech = int(adder_config.get('Bank level', 'Adder_Tech'))
		self.adder_area = float(adder_config.get('Bank level', 'Adder_Area'))
		self.adder_power = float(adder_config.get('Bank level', 'Adder_Power'))
		if bitwidth is None:
			self.adder_bitwidth = 8
		else:
			self.adder_bitwidth = bitwidth
		assert self.adder_bitwidth > 0
		if frequency is None:
			self.adder_frequency = 100
		else:
			self.adder_frequency = frequency
		assert self.adder_frequency > 0
		self.adder_latency = 1.0/self.adder_frequency
		self.adder_energy = 0
		# print("Adder configuration is loaded")
		# self.calculate_adder_area()
		# self.calculate_adder_power()
		# self.calculate_adder_energy()

	def calculate_adder_area(self):
		# unit: um^2
		if self.adder_area == 0:
			adder_area_dict = {130: 0,
							   65: 0,
							   55: 0,
							   45: 0,
							   28: 0
			}
			# TODO: add circuits simulation results
			if self.adder_tech <= 28:
				self.adder_area = adder_area_dict[28] * self.adder_bitwidth
			elif self.adder_tech <= 45:
				self.adder_area = adder_area_dict[45] * self.adder_bitwidth
			elif self.adder_tech <= 55:
				self.adder_area = adder_area_dict[55] * self.adder_bitwidth
			elif self.adder_tech <= 65:
				self.adder_area = adder_area_dict[65] * self.adder_bitwidth
			else:
				self.adder_area = adder_area_dict[130] * self.adder_bitwidth

	def calculate_adder_power(self):
		# unit: W
		if self.adder_power == 0:
			adder_power_dict = {130: 0,
							   65: 0,
							   55: 0,
							   45: 0,
							   28: 0
			}
			# TODO: add circuits simulation results
			if self.adder_tech <= 28:
				self.adder_power = adder_power_dict[28] * self.adder_bitwidth
			elif self.adder_tech <= 45:
				self.adder_power = adder_power_dict[45] * self.adder_bitwidth
			elif self.adder_tech <= 55:
				self.adder_power = adder_power_dict[55] * self.adder_bitwidth
			elif self.adder_tech <= 65:
				self.adder_power = adder_power_dict[65] * self.adder_bitwidth
			else:
				self.adder_power = adder_power_dict[130] * self.adder_bitwidth

	def calculate_adder_energy(self):
		assert self.adder_power >= 0
		assert self.adder_latency >= 0
		self.adder_energy = self.adder_latency * self.adder_power

	def adder_output(self):
		print("adder_area:", self.adder_area, "um^2")
		print("adder_bitwidth:", self.adder_bitwidth, "bit")
		print("adder_power:", self.adder_power, "W")
		print("adder_latency:", self.adder_latency, "ns")
		print("adder_energy:", self.adder_energy, "nJ")
	
def adder_test():
	print("load file:",test_SimConfig_path)
	_adder = adder(test_SimConfig_path)
	_adder.calculate_adder_area()
	_adder.calculate_adder_power()
	_adder.calculate_adder_energy()
	_adder.adder_output()


if __name__ == '__main__':
	adder_test()