#!/usr/bin/python
#-*-coding:utf-8-*-
import configparser as cp
import os
import math
test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),"SimConfig.ini")
	#Default SimConfig file path: MNSIM_Python/SimConfig.ini


class DAC(object):
	def __init__(self, SimConfig_path):
		DAC_config = cp.ConfigParser()
		DAC_config.read(SimConfig_path, encoding='UTF-8')
		self.DAC_choice = int(DAC_config.get('Interface level', 'DAC_Choice'))
		self.DAC_area = float(DAC_config.get('Interface level', 'DAC_Area'))
		self.DAC_precision = int(DAC_config.get('Interface level', 'DAC_Precision'))
		self.DAC_power = float(DAC_config.get('Interface level', 'DAC_Power'))
		self.DAC_sample_rate = float(DAC_config.get('Interface level', 'DAC_Sample_Rate'))
		self.DAC_energy = 0
		# print("DAC configuration is loaded")
		# self.calculate_DAC_area()
		self.calculate_DAC_precision()
		# self.calculate_DAC_power()
		# self.calculate_DAC_sample_rate()
		# self.calculate_DAC_energy()

	def calculate_DAC_area(self):
		#unit: um^2
		#Data reference: ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars
		DAC_area_dict = {1: 0.166, #1-bit
						 2: 0.332, #2-bit
						 3: 0.664, #3-bit
						 4: 1.328, #4-bit
						 5: 5.312, #6-bit
						 6: 21.248 #8-bit
		}
		if self.DAC_choice != -1:
			assert self.DAC_choice in [1,2,3,4,5,6]
			self.DAC_area = DAC_area_dict[self.DAC_choice]

	def calculate_DAC_precision(self):
		#Data reference: ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars
		DAC_precision_dict = {1: 1, #1-bit
						 2: 2, #2-bit
						 3: 3, #3-bit
						 4: 4, #4-bit
						 5: 6, #6-bit
						 6: 8 #8-bit
		}
		if self.DAC_choice != -1:
			assert self.DAC_choice in [1,2,3,4,5,6]
			self.DAC_precision = DAC_precision_dict[self.DAC_choice]

	def calculate_DAC_power(self):
		#unit: W
		#Data reference: ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars
		DAC_power_dict = {1: 0.0039*1e-3, #1-bit
						 2: 0.0078*1e-3, #2-bit
						 3: 0.0156*1e-3, #3-bit
						 4: 0.0312*1e-3, #4-bit
						 5: 0.1248*1e-3, #6-bit
						 6: 0.4992 #8-bit
		}
		if self.DAC_choice != -1:
			assert self.DAC_choice in [1,2,3,4,5,6]
			self.DAC_power = DAC_power_dict[self.DAC_choice]

	def calculate_DAC_sample_rate(self):
		#unit: GSamples/s
		if self.DAC_choice != -1:
			self.DAC_sample_rate = 1

	def calculate_DAC_energy(self):
		#unit: nJ
		self.DAC_energy = 1 / self.DAC_sample_rate * self.DAC_power

	def DAC_output(self):
		if self.DAC_choice == -1:
			print("DAC_choice: User defined")
		else:
			print("DAC_choice:", self.DAC_choice)
		print("DAC_area:", self.DAC_area, "um^2")
		print("DAC_precision:", self.DAC_precision, "bit")
		print("DAC_power:", self.DAC_power, "W")
		print("DAC_sample_rate:", self.DAC_sample_rate, "Gbit/s")
		print("DAC_energy:", self.DAC_energy, "nJ")
	
def DAC_test():
	print("load file:",test_SimConfig_path)
	_DAC = DAC(test_SimConfig_path)
	_DAC.calculate_DAC_area()
	_DAC.calculate_DAC_power()
	_DAC.calculate_DAC_sample_rate()
	_DAC.calculate_DAC_energy()
	_DAC.DAC_output()


if __name__ == '__main__':
	DAC_test()