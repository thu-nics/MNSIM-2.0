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
		self.PIM_type_adc = int(ADC_config.get('Process element level', 'PIM_Type'))
		self.ADC_choice = int(ADC_config.get('Interface level', 'ADC_Choice'))
		if self.PIM_type_adc == 1 and self.ADC_choice != -1:
			# digital PIM architecture
			self.ADC_choice = 8
		self.ADC_area = float(ADC_config.get('Interface level', 'ADC_Area'))
		self.ADC_precision = int(ADC_config.get('Interface level', 'ADC_Precision'))
		self.ADC_power = float(ADC_config.get('Interface level', 'ADC_Power'))
		self.ADC_sample_rate = float(ADC_config.get('Interface level', 'ADC_Sample_Rate'))
		self.ADC_latency = 0
		self.ADC_energy = 0
		self.ADC_interval = list(map(int, ADC_config.get('Interface level', 'ADC_Interval_Thres').split(',')))
		# print("ADC configuration is loaded")
		self.logic_op = int(ADC_config.get('Interface level', 'Logic_Op'))
		self.calculate_ADC_precision()
		self.calculate_ADC_sample_rate()

	def calculate_ADC_area(self):
		#unit: um^2
		ADC_area_dict = {1: 1600, #reference: A 10b 1.5GS/s Pipelined-SAR ADC with Background Second-Stage Common-Mode Regulation and Offset Calibration in 14nm CMOS FinFET
						 2: 1200, #reference: ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars
						 3: 1650, #reference: A >3GHz ERBW 1.1GS/s 8b Two-Step SAR ADC with Recursive-Weight DAC
						 4: 580, #reference: Area-Efficient 1GS/s 6b SAR ADC with Charge-Injection-Cell-Based DAC
						 5: 1650, #ASPDAC1
						 6: 1650, #ASPDAC2
						 7: 500, #ASPDAC3
						 8: 1, #SA @ 28nm
						 9: 15899 #Qi Liu
		}
		if self.ADC_choice != -1:
			assert self.ADC_choice in [1,2,3,4,5,6,7,8,9]
			self.ADC_area = ADC_area_dict[self.ADC_choice]
		if self.logic_op == 0: # Notice: scale from 65nm to 28nm
			self.ADC_area += 17.28*0.18 # AND gate
		elif self.logic_op == 1:
			self.ADC_area += 17.28*0.18 # OR gate
		elif self.logic_op == 2:
			self.ADC_area += 19.2*0.18 # XOR gate

	def calculate_ADC_precision(self):
		ADC_precision_dict = {1: 10, #reference: A 10b 1.5GS/s Pipelined-SAR ADC with Background Second-Stage Common-Mode Regulation and Offset Calibration in 14nm CMOS FinFET
						 2: 8, #reference: ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars
						 3: 8, #reference: A >3GHz ERBW 1.1GS/s 8b Two-Step SAR ADC with Recursive-Weight DAC
						 4: 6, #reference: Area-Efficient 1GS/s 6b SAR ADC with Charge-Injection-Cell-Based DAC
						 5: 8, #ASPDAC1
						 6: 6, #ASPDAC2
						 7: 4, #ASPDAC3
						 8: 1, #SA
						 9: 8
		}
		if self.ADC_choice != -1:
			assert self.ADC_choice in [1,2,3,4,5,6,7,8,9]
			self.ADC_precision = ADC_precision_dict[self.ADC_choice]

	def calculate_ADC_power(self):
		#unit: W
		ADC_power_dict = {1: 6.92*1e-3, #reference: A 10b 1.5GS/s Pipelined-SAR ADC with Background Second-Stage Common-Mode Regulation and Offset Calibration in 14nm CMOS FinFET
						 2: 2*1e-3, #reference: ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars
						 3: 4*1e-3, #reference: A >3GHz ERBW 1.1GS/s 8b Two-Step SAR ADC with Recursive-Weight DAC
						 4: 1.26*1e-3, #reference: Area-Efficient 1GS/s 6b SAR ADC with Charge-Injection-Cell-Based DAC
						 5: 4e-3, #ASPDAC1
						 6: 1.26e-3, #ASPDAC2
						 7: 0.7e-3, #ASPDAC3
						 8: 0.1086*15e-6, #SA reference: Comparative Study of Sense Amplifiers for SRAM (scale from 1.2V@65nm to 0.8V@28nm)
						 9: 8*0.0073*1e-3
		}
		if self.ADC_choice != -1:
			assert self.ADC_choice in [1,2,3,4,5,6,7,8,9]
			self.ADC_power = ADC_power_dict[self.ADC_choice]
		# Todo: upodate logic gate power
		# if self.logic_op == 0:
		# 	self.ADC_power += 0.127*0.4*1e-3
		# elif self.logic_op == 1:
		# 	self.ADC_power += 0.126*0.4*1e-3
		# elif self.logic_op == 2:
		# 	self.ADC_power += 0.243*0.4*1e-3

	def calculate_ADC_sample_rate(self):
		#unit: GSamples/s
		ADC_sample_rate_dict = {1: 1.5, #reference: A 10b 1.5GS/s Pipelined-SAR ADC with Background Second-Stage Common-Mode Regulation and Offset Calibration in 14nm CMOS FinFET
								2: 1.28, #reference: ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars
								3: 1.1, #reference: A >3GHz ERBW 1.1GS/s 8b Two-Step SAR ADC with Recursive-Weight DAC
								4: 1, #reference: Area-Efficient 1GS/s 6b SAR ADC with Charge-Injection-Cell-Based DAC
								5: 1.1, #ASPDAC1
								6: 1, #ASPDAC2
								7: 1,
								8: 1, #SA reference: Comparative Study of Sense Amplifiers for SRAM
								9: 6
		}
		if self.ADC_choice != -1:
			assert self.ADC_choice in [1,2,3,4,5,6,7,8,9]
			self.ADC_sample_rate = ADC_sample_rate_dict[self.ADC_choice]

	def calculate_ADC_latency(self):
		# unit: ns
		if self.ADC_precision == 1:
			self.ADC_latency = 1 / self.ADC_sample_rate
		elif self.ADC_choice == 9:
			self.ADC_latency = 1/self.ADC_sample_rate * (2**self.ADC_precision)
		else:	
			self.ADC_latency = 1 / self.ADC_sample_rate * (self.ADC_precision + 2)
		# Todo: upodate logic gate latency
		# if self.logic_op != -1:
		# 	self.ADC_latency += 1


	def calculate_ADC_energy(self):
		#unit: nJ
		self.ADC_energy = self.ADC_latency * self.ADC_power
		# Todo: upodate logic gate energy
		# if self.logic_op == -1:
		# 	self.ADC_energy = self.ADC_latency * self.ADC_power
		# elif self.logic_op == 0:
		# 	self.ADC_energy = (self.ADC_latency-1)*(self.ADC_power-0.127*0.4*1e-3) + 0.127*0.4*1e-3
		# elif self.logic_op == 1:
		# 	self.ADC_energy = (self.ADC_latency-1)*(self.ADC_power-0.126*0.4*1e-3) + 0.126*0.4*1e-3
		# elif self.logic_op == 2:
		# 	self.ADC_energy = (self.ADC_latency-1)*(self.ADC_power-0.243*0.4*1e-3) + 0.243*0.4*1e-3

	def config_ADC_interval(self, SimConfig_path, WL_num = 0):
		if self.ADC_interval[0] == -1: #User defined
			ADC_config = cp.ConfigParser()
			ADC_config.read(SimConfig_path, encoding='UTF-8')
			self.ADC_interval = (2**self.ADC_precision-1) * [0.0]
			V_in = list(map(float, ADC_config.get('Device level', 'Read_Voltage').split(',')))
			R = list(map(float, ADC_config.get('Device level', 'Device_Resistance').split(',')))
			# Rs = math.sqrt(R[0]*R[-1])
			Rs = float(ADC_config.get('Crossbar level', 'Load_Resistance'))
			if Rs == -1:
				Rs = math.sqrt(R[0] * R[-1])
			assert Rs > 0, "Load resistance must be > 0"

			step = math.ceil(WL_num/(2**self.ADC_precision))
			temp = step
			V_max = WL_num*V_in[-1]/R[-1]*Rs
			for i in range(len(self.ADC_interval)):
				if temp < WL_num+1:
					self.ADC_interval[i] = 0.5 * ((temp-1)*V_in[-1]/R[-1]*Rs+(WL_num-temp+1)*V_in[0]/R[-1]*Rs+
						temp*V_in[-1]/R[-1]*Rs+(WL_num-temp)*V_in[0]/R[0]*Rs)
					temp += step
				else:
					self.ADC_interval[i] = V_max
			# print(self.ADC_interval)

	def calculate_sensing_results(self, V_in):
		# Notice: before calculating sensing results, config_ADC_interval must be calculated
		start = 0
		end = 2**self.ADC_precision-2
		V_out = 0
		temp = 0
		if V_in < self.ADC_interval[0]:
			V_out = 0
		elif V_in > self.ADC_interval[-1]:
			V_out = 2**self.ADC_precision -1
		else:
			while start < end:
				temp = int(1/2*(start+end))
				if temp == start:
					V_out = temp + 1
					break
				if V_in > self.ADC_interval[temp]:
					start = temp
					V_out = temp
				else:
					end = temp
					V_out = temp
		return V_out

	def ADC_output(self):
		if self.ADC_choice == -1:
			print("ADC_choice: User defined")
		else:
			print("ADC_choice:", self.ADC_choice)
		print("ADC_area:", self.ADC_area, "um^2")
		print("ADC_precision:", self.ADC_precision, "bit")
		print("ADC_power:", self.ADC_power, "W")
		print("ADC_sample_rate:", self.ADC_sample_rate, "Gbit/s")
		print("ADC_latency:", self.ADC_latency, "ns")
		print("ADC_energy:", self.ADC_energy, "nJ")
	
def ADC_test():
	print("load file:",test_SimConfig_path)
	_ADC = ADC(test_SimConfig_path)
	_ADC.calculate_ADC_area()
	_ADC.calculate_ADC_power()
	_ADC.calculate_ADC_sample_rate()
	_ADC.calculate_ADC_latency()
	_ADC.calculate_ADC_energy()
	_ADC.config_ADC_interval(test_SimConfig_path,256)
	result = _ADC.calculate_sensing_results(100)
	_ADC.ADC_output()


if __name__ == '__main__':
	ADC_test()