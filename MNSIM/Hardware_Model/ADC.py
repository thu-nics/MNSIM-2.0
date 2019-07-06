#!/usr/bin/python
#-*-coding:utf-8-*-
import configparser as cp
import os
import math
test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),"SimConfig.ini")
	#Default SimConfig file path: MNSIM_Python/SimConfig.ini


class ADC(object):
	def __init__(self, SimConfig_path):
		ADC_config = cp.ConfigParser()
		ADC_config.read(SimConfig_path, encoding='UTF-8')
		self.ADC_choice = int(ADC_config.get('Interface level', 'ADC_Choice'))
		self.ADC_area = float(ADC_config.get('Interface level', 'ADC_Area'))
		self.ADC_precision = int(ADC_config.get('Interface level', 'ADC_Precision'))
		self.ADC_power = float(ADC_config.get('Interface level', 'ADC_Power'))
		self.ADC_sample_rate = float(ADC_config.get('Interface level', 'ADC_Sample_Rate'))
		self.ADC_energy = 0
		# print("ADC configuration is loaded")
		# self.calculate_ADC_area()
		self.calculate_ADC_precision()
		# self.calculate_ADC_power()
		# self.calculate_ADC_sample_rate()
		# self.calculate_ADC_energy()

	def calculate_ADC_area(self):
		#unit: um^2
		ADC_area_dict = {1: 1600, #reference: A 10b 1.5GS/s Pipelined-SAR ADC with Background Second-Stage Common-Mode Regulation and Offset Calibration in 14nm CMOS FinFET
						 2: 1600, #reference: ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars
						 3: 1650, #reference: A >3GHz ERBW 1.1GS/s 8b Two-Step SAR ADC with Recursive-Weight DAC
						 4: 580 #reference: Area-Efficient 1GS/s 6b SAR ADC with Charge-Injection-Cell-Based DAC
		}
		if self.ADC_choice != -1:
			assert self.ADC_choice in [1,2,3,4]
			self.ADC_area = ADC_area_dict[self.ADC_choice]

	def calculate_ADC_precision(self):
		ADC_precision_dict = {1: 10, #reference: A 10b 1.5GS/s Pipelined-SAR ADC with Background Second-Stage Common-Mode Regulation and Offset Calibration in 14nm CMOS FinFET
						 2: 8, #reference: ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars
						 3: 8, #reference: A >3GHz ERBW 1.1GS/s 8b Two-Step SAR ADC with Recursive-Weight DAC
						 4: 6 #reference: Area-Efficient 1GS/s 6b SAR ADC with Charge-Injection-Cell-Based DAC
		}
		if self.ADC_choice != -1:
			assert self.ADC_choice in [1,2,3,4]
			self.ADC_precision = ADC_precision_dict[self.ADC_choice]

	def calculate_ADC_power(self):
		#unit: W
		ADC_power_dict = {1: 6.92*1e-3, #reference: A 10b 1.5GS/s Pipelined-SAR ADC with Background Second-Stage Common-Mode Regulation and Offset Calibration in 14nm CMOS FinFET
						 2: 2*1e-3, #reference: ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars
						 3: 4*1e-3, #reference: A >3GHz ERBW 1.1GS/s 8b Two-Step SAR ADC with Recursive-Weight DAC
						 4: 1.26*1e-3 #reference: Area-Efficient 1GS/s 6b SAR ADC with Charge-Injection-Cell-Based DAC
		}
		if self.ADC_choice != -1:
			assert self.ADC_choice in [1,2,3,4]
			self.ADC_power = ADC_power_dict[self.ADC_choice]

	def calculate_ADC_sample_rate(self):
		#unit: GSamples/s
		ADC_sample_rate_dict = {1: 1.5, #reference: A 10b 1.5GS/s Pipelined-SAR ADC with Background Second-Stage Common-Mode Regulation and Offset Calibration in 14nm CMOS FinFET
								2: 1, #reference: ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars
								3: 1.1, #reference: A >3GHz ERBW 1.1GS/s 8b Two-Step SAR ADC with Recursive-Weight DAC
								4: 1 #reference: Area-Efficient 1GS/s 6b SAR ADC with Charge-Injection-Cell-Based DAC
		}
		if self.ADC_choice != -1:
			assert self.ADC_choice in [1,2,3,4]
			self.ADC_sample_rate = ADC_sample_rate_dict[self.ADC_choice]

	def calculate_ADC_energy(self):
		#unit: nJ
		self.ADC_energy = 1 / self.ADC_sample_rate * self.ADC_power

	def ADC_output(self):
		if self.ADC_choice == -1:
			print("ADC_choice: User defined")
		else:
			print("ADC_choice:", self.ADC_choice)
		print("ADC_area:", self.ADC_area, "um^2")
		print("ADC_precision:", self.ADC_precision, "bit")
		print("ADC_power:", self.ADC_power, "W")
		print("ADC_sample_rate:", self.ADC_sample_rate, "Gbit/s")
		print("ADC_energy:", self.ADC_energy, "nJ")
	
def ADC_test():
	print("load file:",test_SimConfig_path)
	_ADC = ADC(test_SimConfig_path)
	_ADC.calculate_ADC_area()
	_ADC.calculate_ADC_power()
	_ADC.calculate_ADC_sample_rate()
	_ADC.calculate_ADC_energy()
	_ADC.ADC_output()


if __name__ == '__main__':
	ADC_test()