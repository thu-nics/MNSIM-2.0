#linqiushi modified
#!/usr/bin/python
#-*-coding:utf-8-*-
import configparser as cp
import os
import math
test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),"SimConfig.ini")
	#Default SimConfig file path: MNSIM_Python/SimConfig.ini


class multiplier(object):
	def __init__(self, SimConfig_path, bitwidth = None):
		# frequency unit: MHz
		multiplier_config = cp.ConfigParser()
		multiplier_config.read(SimConfig_path, encoding='UTF-8')
		self.multiplier_tech = int(multiplier_config.get('Digital module', 'Multiplier_Tech'))
		self.multiplier_area = float(multiplier_config.get('Digital module', 'Multiplier_Area'))
		self.multiplier_power = float(multiplier_config.get('Digital module', 'Multiplier_Power'))
		if bitwidth is None:
			self.multiplier_bitwidth = 8
		else:
			self.multiplier_bitwidth = bitwidth
		assert self.multiplier_bitwidth > 0
		self.multiplier_frequency = float(multiplier_config.get('Digital module', 'Digital_Frequency'))
		if self.multiplier_frequency is None:
			self.multiplier_frequency = 100
		assert self.multiplier_frequency > 0
		self.multiplier_latency = 1.0/self.multiplier_frequency
		self.multiplier_energy = 0
		self.calculate_multiplier_power()

	def calculate_multiplier_area(self):
		# unit: um^2
		if self.multiplier_area == 0:
			multiplier_area_dict = {130: 10*14*130*130/1e6, #ref: Implementation of an Efficient 14-Transistor Full multiplier (.18μm technology) Using DTMOS 2.5e-9
							   65: 1.42,#10*14*65*65/1e6,
							   55: 10*14*55*55/1e6,
							   45: 10*14*45*45/1e6,
							   28: 10*14*28*28/1e6
			}
			# TODO: add circuits simulation results
			if self.multiplier_tech <= 28:
				self.multiplier_area = multiplier_area_dict[28] * self.multiplier_bitwidth
			elif self.multiplier_tech <= 45:
				self.multiplier_area = multiplier_area_dict[45] * self.multiplier_bitwidth
			elif self.multiplier_tech <= 55:
				self.multiplier_area = multiplier_area_dict[55] * self.multiplier_bitwidth
			elif self.multiplier_tech <= 65:
				self.multiplier_area = multiplier_area_dict[65] * self.multiplier_bitwidth
			else:
				self.multiplier_area = multiplier_area_dict[130] * self.multiplier_bitwidth

	def calculate_multiplier_power(self):
		# unit: W
		if self.multiplier_power == 0:
			multiplier_power_dict = {130: 2.5e-9,
							   65: 3e-7,
							   55: 2.5e-9,
							   45: 2.5e-9,
							   28: 2.5e-9
			}
			# TODO: add circuits simulation results
			if self.multiplier_tech <= 28:
				self.multiplier_power = multiplier_power_dict[28] * self.multiplier_bitwidth
			elif self.multiplier_tech <= 45:
				self.multiplier_power = multiplier_power_dict[45] * self.multiplier_bitwidth
			elif self.multiplier_tech <= 55:
				self.multiplier_power = multiplier_power_dict[55] * self.multiplier_bitwidth
			elif self.multiplier_tech <= 65:
				self.multiplier_power = multiplier_power_dict[65] * self.multiplier_bitwidth
			else:
				self.multiplier_power = multiplier_power_dict[130] * self.multiplier_bitwidth

	def calculate_multiplier_energy(self):
		assert self.multiplier_power >= 0
		assert self.multiplier_latency >= 0
		self.multiplier_energy = self.multiplier_latency * self.multiplier_power

	def multiplier_output(self):
		print("multiplier_area:", self.multiplier_area, "um^2")
		print("multiplier_bitwidth:", self.multiplier_bitwidth, "bit")
		print("multiplier_power:", self.multiplier_power, "W")
		print("multiplier_latency:", self.multiplier_latency, "ns")
		print("multiplier_energy:", self.multiplier_energy, "nJ")
	
def multiplier_test():
	print("load file:",test_SimConfig_path)
	_multiplier = multiplier(test_SimConfig_path)
	_multiplier.calculate_multiplier_area()
	_multiplier.calculate_multiplier_power()
	_multiplier.calculate_multiplier_energy()
	_multiplier.multiplier_output()


if __name__ == '__main__':
	multiplier_test()
#linqiushi above