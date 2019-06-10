#!/usr/bin/python
# -*-coding:utf-8-*-
import configparser as cp
import os
import math
from numpy import *
import numpy as np
from PE import ProcessElement
from Adder import adder
from ShiftReg import shiftreg
test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),"SimConfig.ini")
# Default SimConfig file path: MNSIM_Python/SimConfig.ini


class bank():
	def __init__(self, SimConfig_path, layer_num = None):
		# layer_num is a list with the size of 1xPE_num
		bank_config = cp.ConfigParser()
		bank_config.read(SimConfig_path)
		self.bank_PE_num = list(map(int, bank_config.get('Bank level', 'PE_Num').split(',')))
		if self.bank_PE_num[0] == 0:
			self.bank_PE_num[0] = 8
			self.bank_PE_num[1] = 8
		assert self.bank_PE_num[0] > 0
		assert self.bank_PE_num[1] > 0
		self.bank_PE_total_num = self.bank_PE_num[0] * self.bank_PE_num[1]
		self.bank_adder_num = int(bank_config.get('Bank level', 'Bank_Adder_Num'))
		self.bank_adder_level = int(bank_config.get('Bank level', 'Bank_Adder_Level'))
		self.bank_shiftreg_num = int(bank_config.get('Bank level', 'Bank_ShiftReg_Num'))
		self.bank_shiftreg_level = int(bank_config.get('Bank level', 'Bank_ShiftReg_Level'))
		self.bank_PE = []
		if layer_num is None:
			layer_num = self.bank_PE_total_num * [0]
		for index in range(self.bank_PE_total_num):
			temp_PE = ProcessElement(SimConfig_path, layer_num[index])
			self.bank_PE.append(temp_PE)
		self.bank_adder = adder(SimConfig_path)
		self.bank_shiftreg = shiftreg(SimConfig_path)

		self.bank_usage_utilization = 0

		self.bank_area = 0
		self.bank_xbar_area = 0
		self.bank_ADC_area = 0
		self.bank_DAC_area = 0
		self.bank_digital_area = 0
		self.bank_adder_area = 0
		self.bank_shiftreg_area = 0

		self.bank_read_power = 0
		self.bank_xbar_read_power = 0
		self.bank_ADC_read_power = 0
		self.bank_DAC_read_power = 0
		self.bank_digital_read_power = 0
		self.bank_adder_read_power = 0
		self.bank_shiftreg_read_power = 0

		self.bank_write_power = 0
		self.bank_xbar_write_power = 0
		self.bank_ADC_write_power = 0
		self.bank_DAC_write_power = 0
		self.bank_digital_write_power = 0
		self.bank_adder_write_power = 0
		self.bank_shiftreg_write_power = 0

		self.bank_read_latency = 0
		self.bank_xbar_read_latency = 0
		self.bank_ADC_read_latency = 0
		self.bank_DAC_read_latency = 0
		self.bank_digital_read_latency = 0
		self.bank_adder_read_latency = 0
		self.bank_shiftreg_read_latency = 0
		self.bank_layer_read_latency = {0:0}

		self.bank_write_latency = 0
		self.bank_xbar_write_latency = 0
		self.bank_ADC_write_latency = 0
		self.bank_DAC_write_latency = 0
		self.bank_digital_write_latency = 0
		self.bank_adder_write_latency = 0
		self.bank_shiftreg_write_latency = 0
		self.bank_layer_write_latency = {0:0}

		self.bank_read_energy = 0
		self.bank_xbar_read_energy = 0
		self.bank_ADC_read_energy = 0
		self.bank_DAC_read_energy = 0
		self.bank_digital_read_energy = 0
		self.bank_adder_read_energy = 0
		self.bank_shiftreg_read_energy = 0

		self.bank_write_energy = 0
		self.bank_xbar_write_energy = 0
		self.bank_ADC_write_energy = 0
		self.bank_DAC_write_energy = 0
		self.bank_digital_write_energy = 0
		self.bank_adder_write_energy = 0
		self.bank_shiftreg_write_energy = 0
		print("bank configuration is loaded")
		self.calculate_intra_PE_connection()


	def calculate_intra_PE_connection(self):
		# default configuration: H-tree structure
		level = math.ceil(math.log2(self.bank_PE_total_num))
		if self.bank_adder_level == 0:
			self.bank_adder_level = level
		if self.bank_shiftreg_level == 0:
			self.bank_shiftreg_level = level
		index = self.bank_PE_total_num
		temp_num = 0
		while index/2 >= 1:
			temp_num += int(index/2) + index%2
			index = int(index/2)
		temp_num = self.bank_PE[0].PE_ADC_num * temp_num
		if self.bank_adder_num == 0:
			self.bank_adder_num = temp_num
		if self.bank_shiftreg_num == 0:
			self.bank_shiftreg_num = temp_num

	def calculate_bank_area(self):
		# unit: um^2
		self.bank_area = 0
		self.bank_xbar_area = 0
		self.bank_ADC_area = 0
		self.bank_DAC_area = 0
		self.bank_digital_area = 0
		self.bank_adder.calculate_adder_area()
		self.bank_shiftreg.calculate_shiftreg_area()
		for index in range(self.bank_PE_total_num):
			self.bank_PE[index].calculate_PE_area()
			self.bank_xbar_area += self.bank_PE[index].PE_xbar_area
			self.bank_ADC_area += self.bank_PE[index].PE_ADC_area
			self.bank_DAC_area += self.bank_PE[index].PE_DAC_area
			self.bank_digital_area += self.bank_PE[index].PE_digital_area
		self.bank_adder_area = self.bank_adder_num * self.bank_adder.adder_area
		self.bank_shiftreg_area = self.bank_shiftreg_num * self.bank_shiftreg.shiftreg_area
		self.bank_digital_area += self.bank_adder_area + self.bank_shiftreg_area
		self.bank_area = self.bank_xbar_area + self.bank_ADC_area + self.bank_DAC_area + self.bank_digital_area


	def calculate_bank_read_latency(self, read_row = None, read_column = None):
		# unit: ns
		# read_row / read_column are lists with the length of 1xbank_PE_total_num
		PE_read_latency = []
		PE_xbar_read_latency = []
		PE_ADC_read_latency = []
		PE_DAC_read_latency = []
		PE_digital_read_latency = []
		self.bank_layer_read_latency = {0:0}
		layer_PE_num = {0:0} #the number of PEs which process each layer
		if read_row is None:
			read_row = self.bank_PE_total_num * [self.bank_PE[0].xbar_row * self.bank_PE[0].PE_multiplex_xbar_num[0]]
		assert min(read_row) >= 0
		if read_column is None:
			read_column = self.bank_PE_total_num * [self.bank_PE[0].xbar_column * self.bank_PE[0].PE_multiplex_xbar_num[1]]
		assert min(read_column) >= 0
		for index in range(self.bank_PE_total_num):
			self.bank_PE[index].calculate_PE_read_latency(read_row[index], read_column[index])
			PE_read_latency.append(self.bank_PE[index].PE_read_latency)
			PE_xbar_read_latency.append(self.bank_PE[index].PE_xbar_read_latency)
			PE_ADC_read_latency.append(self.bank_PE[index].PE_ADC_read_latency)
			PE_DAC_read_latency.append(self.bank_PE[index].PE_DAC_read_latency)
			PE_digital_read_latency.append(self.bank_PE[index].PE_digital_read_latency)
			layer_num = self.bank_PE[index].layer_num
			if layer_num in self.bank_layer_read_latency.keys():
				if self.bank_PE[index].PE_read_latency >= self.bank_layer_read_latency[layer_num]:
					self.bank_layer_read_latency[layer_num] = self.bank_PE[index].PE_read_latency
					layer_PE_num[layer_num] += 1
			else:
				self.bank_layer_read_latency[layer_num] = self.bank_PE[index].PE_read_latency
				layer_PE_num[layer_num] = 1
		index = PE_read_latency.index(max(PE_read_latency))
		self.bank_xbar_read_latency = PE_xbar_read_latency[index]
		self.bank_ADC_read_latency = PE_ADC_read_latency[index]
		self.bank_DAC_read_latency = PE_DAC_read_latency[index]
		self.bank_adder_read_latency = self.bank_adder_level * self.bank_adder.adder_latency
		self.bank_shiftreg_read_latency = self.bank_shiftreg_level * self.bank_shiftreg.shiftreg_latency
		self.bank_digital_read_latency = PE_digital_read_latency[index] + \
										 self.bank_adder_read_latency + \
										 self.bank_shiftreg_read_latency
		self.bank_read_latency = self.bank_xbar_read_latency + self.bank_ADC_read_latency + \
								 self.bank_DAC_read_latency + self.bank_digital_read_latency
		for layer in layer_PE_num.keys():
			if layer_PE_num[layer] != 0:
				layer_merge_level = math.ceil(math.log2(layer_PE_num[layer]))
				self.bank_layer_read_latency[layer] += layer_merge_level * (self.bank_adder.adder_latency
																		+ self.bank_shiftreg.shiftreg_latency)

	def calculate_bank_write_latency(self, write_row = None, write_column = None):
		# unit: ns
		# read_row / read_column are lists with the length of 1xbank_PE_total_num
		PE_write_latency = []
		PE_xbar_write_latency = []
		PE_ADC_write_latency = []
		PE_DAC_write_latency = []
		PE_digital_write_latency = []
		self.bank_layer_write_latency = {0:0}
		if write_row is None:
			write_row = self.bank_PE_total_num * [self.bank_PE[0].xbar_row * self.bank_PE[0].PE_multiplex_xbar_num[0]]
		assert min(write_row) >= 0
		if write_column is None:
			write_column = self.bank_PE_total_num * [self.bank_PE[0].xbar_column * self.bank_PE[0].PE_multiplex_xbar_num[1]]
		assert min(write_column) >= 0
		for index in range(self.bank_PE_total_num):
			self.bank_PE[index].calculate_PE_write_latency(write_row[index], write_column[index])
			PE_write_latency.append(self.bank_PE[index].PE_write_energy)
			PE_xbar_write_latency.append(self.bank_PE[index].PE_xbar_write_latency)
			PE_ADC_write_latency.append(self.bank_PE[index].PE_ADC_write_latency)
			PE_DAC_write_latency.append(self.bank_PE[index].PE_DAC_write_latency)
			PE_digital_write_latency.append(self.bank_PE[index].PE_digital_write_latency)
			layer_num = self.bank_PE[index].layer_num
			if layer_num in self.bank_layer_write_latency.keys():
				if self.bank_PE[index].PE_write_latency >= self.bank_layer_write_latency[layer_num]:
					self.bank_layer_write_latency[layer_num] = self.bank_PE[index].PE_write_latency
			else:
				self.bank_layer_write_latency[layer_num] = self.bank_PE[index].PE_write_latency
		index = PE_write_latency.index(max(PE_write_latency))
		self.bank_xbar_write_latency = PE_xbar_write_latency[index]
		self.bank_ADC_write_latency = PE_ADC_write_latency[index]
		self.bank_DAC_write_latency = PE_DAC_write_latency[index]
		self.bank_digital_write_latency = PE_digital_write_latency[index]
		self.bank_write_latency = self.bank_xbar_write_latency + self.bank_ADC_write_latency + \
								 self.bank_DAC_write_latency + self.bank_digital_write_latency

	def calculate_bank_read_power(self, cal_mode = 0, read_column = None, read_row = None, num_cell = None):
		# unit: W
		# read_column / read_row are 2D lists with the size of (bank_PE_total_num, num_group)
		# num_cell is a 3D list with the size of (bank_PE_total_num, num_group, num_resistance_level)
		self.bank_read_power = 0
		self.bank_xbar_read_power = 0
		self.bank_ADC_read_power = 0
		self.bank_DAC_read_power = 0
		self.bank_digital_read_power = 0

		if read_row is None:
			read_row = self.bank_PE_total_num * [self.bank_PE[0].num_group *
												 [self.bank_PE[0].xbar_row * self.bank_PE[0].PE_multiplex_xbar_num[0]]]
		assert min(min(read_row)) >= 0
		if read_column is None:
			read_column = self.bank_PE_total_num * [self.bank_PE[0].num_group *
													[self.bank_PE[0].xbar_column * self.bank_PE[0].PE_multiplex_xbar_num[1]]]
		assert min(min(read_column)) >= 0
		for index in range(self.bank_PE_total_num):
			if num_cell is None:
				self.bank_PE[index].calculate_PE_read_power(cal_mode, read_column[index], read_row[index], None)
			else:
				self.bank_PE[index].calculate_PE_read_power(cal_mode, read_column[index], read_row[index], num_cell[index])
			self.bank_xbar_read_power += self.bank_PE[index].PE_xbar_read_power
			self.bank_ADC_read_power += self.bank_PE[index].PE_ADC_read_power
			self.bank_DAC_read_power += self.bank_PE[index].PE_DAC_read_power
			self.bank_digital_read_power += self.bank_PE[index].PE_digital_read_power
		self.bank_adder_read_power = self.bank_adder_num * self.bank_adder.adder_power
		self.bank_shiftreg_read_power = self.bank_shiftreg_num * self.bank_shiftreg.shiftreg_power
		self.bank_digital_read_power += self.bank_adder_read_power + self.bank_shiftreg_read_power
		# TODO: more accurate estimation of adder/shiftreg number
		self.bank_read_power = self.bank_xbar_read_power + self.bank_ADC_read_power \
							   + self.bank_DAC_read_power + self.bank_digital_read_power

	def calculate_bank_write_power(self, cal_mode = 0, write_column = None, write_row = None, num_cell = None):
		# unit: W
		# write_column / write_row are 2D lists with the size of (bank_PE_total_num, num_group)
		# num_cell is a 3D list with the size of (bank_PE_total_num, num_group, num_resistance_level)
		self.bank_write_power = 0
		self.bank_xbar_write_power = 0
		self.bank_ADC_write_power = 0
		self.bank_DAC_write_power = 0
		self.bank_digital_write_power = 0
		if write_row is None:
			write_row = self.bank_PE_total_num * [self.bank_PE[0].num_group *
												 [self.bank_PE[0].xbar_row * self.bank_PE[0].PE_multiplex_xbar_num[0]]]
		assert min(min(write_row)) >= 0
		if write_column is None:
			write_column = self.bank_PE_total_num * [self.bank_PE[0].num_group *
													[self.bank_PE[0].xbar_column *
													 self.bank_PE[0].PE_multiplex_xbar_num[1]]]
		assert min(min(write_column)) >= 0
		for index in range(self.bank_PE_total_num):
			if num_cell is None:
				self.bank_PE[index].calculate_PE_write_power(cal_mode, write_column[index], write_row[index], None)
			else:
				self.bank_PE[index].calculate_PE_write_power(cal_mode, write_column[index], write_row[index], num_cell[index])
			self.bank_xbar_write_power += self.bank_PE[index].PE_xbar_write_power
			self.bank_ADC_write_power += self.bank_PE[index].PE_ADC_write_power
			self.bank_DAC_write_power += self.bank_PE[index].PE_DAC_write_power
			self.bank_digital_write_power += self.bank_PE[index].PE_digital_write_power
		self.bank_write_power = self.bank_xbar_write_power + self.bank_ADC_write_power \
							   + self.bank_DAC_write_power + self.bank_digital_write_power

	def calculate_bank_read_energy(self):
		# unit: nJ
		self.bank_read_energy = 0
		self.bank_xbar_read_energy = 0
		self.bank_ADC_read_energy = 0
		self.bank_DAC_read_energy = 0
		self.bank_digital_read_energy = 0
		self.bank_adder_read_energy = 0
		self.bank_shiftreg_read_energy = 0
		for index in range(self.bank_PE_total_num):
			self.bank_PE[index].calculate_PE_read_energy()
			self.bank_xbar_read_energy += self.bank_PE[index].PE_xbar_read_energy
			self.bank_ADC_read_energy += self.bank_PE[index].PE_ADC_read_energy
			self.bank_DAC_read_energy += self.bank_PE[index].PE_DAC_read_energy
			self.bank_digital_read_energy += self.bank_PE[index].PE_digital_read_energy
		self.bank_adder_read_energy = self.bank_adder_read_power * self.bank_adder_read_latency
		self.bank_shiftreg_read_energy = self.bank_shiftreg_read_power * self.bank_shiftreg_read_latency
		self.bank_digital_read_energy += self.bank_adder_read_energy + self.bank_shiftreg_read_energy
		self.bank_read_energy = self.bank_xbar_read_energy + self.bank_ADC_read_energy \
								+ self.bank_DAC_read_energy + self.bank_digital_read_energy

	def calculate_bank_write_energy(self):
		# unit: nJ
		self.bank_write_energy = 0
		self.bank_xbar_write_energy = 0
		self.bank_ADC_write_energy = 0
		self.bank_DAC_write_energy = 0
		self.bank_digital_write_energy = 0
		self.bank_adder_write_energy = 0
		self.bank_shiftreg_write_energy = 0
		for index in range(self.bank_PE_total_num):
			self.bank_PE[index].calculate_PE_write_energy()
			self.bank_xbar_write_energy += self.bank_PE[index].PE_xbar_write_energy
			self.bank_ADC_write_energy += self.bank_PE[index].PE_ADC_write_energy
			self.bank_DAC_write_energy += self.bank_PE[index].PE_DAC_write_energy
			self.bank_digital_write_energy += self.bank_PE[index].PE_digital_write_energy
		self.bank_adder_write_energy = self.bank_adder_write_power * self.bank_adder_write_latency
		self.bank_shiftreg_write_energy = self.bank_shiftreg_write_power * self.bank_shiftreg_write_latency
		self.bank_digital_write_energy += self.bank_adder_write_energy + self.bank_shiftreg_write_energy
		self.bank_write_energy = self.bank_xbar_write_energy + self.bank_ADC_write_energy \
								+ self.bank_DAC_write_energy + self.bank_digital_write_energy

	def bank_output(self):
		self.bank_PE[0].PE_output()
		print("-------------------------Bank Configurations-------------------------")
		print("total PE number in one bank:", self.bank_PE_total_num, "(", self.bank_PE_num, ")")
		print("total adder number in one bank:", self.bank_adder_num)
		print("			the level of adders is:", self.bank_adder_level)
		print("total shift-reg number in one bank:", self.bank_shiftreg_num)
		print("			the level of shift-reg is:", self.bank_shiftreg_level)
		print("----------------------Bank Area Simulation Results-------------------")
		print("bank area:", self.bank_area, "um^2")
		print("			crossbar area:", self.bank_xbar_area, "um^2")
		print("			DAC area:", self.bank_DAC_area, "um^2")
		print("			ADC area:", self.bank_ADC_area, "um^2")
		print("			digital part area:", self.bank_digital_area, "um^2")
		print("				|---adder area:", self.bank_adder_area, "um^2")
		print("				|---shift-reg area:", self.bank_shiftreg_area, "um^2")
		print("--------------------Bank Latency Simulation Results------------------")
		print("bank read latency:", self.bank_read_latency, "ns")
		print("			crossbar read latency:", self.bank_xbar_read_latency, "ns")
		print("			DAC read latency:", self.bank_DAC_read_latency, "ns")
		print("			ADC read latency:", self.bank_ADC_read_latency, "ns")
		print("			digital part read latency:", self.bank_digital_read_latency, "ns")
		print("				|---adder read latency:", self.bank_adder_read_latency, "ns")
		print("				|---shift-reg read latency:", self.bank_shiftreg_read_latency, "ns")
		print("bank write latency:", self.bank_write_latency, "ns")
		print("			crossbar write latency:", self.bank_xbar_write_latency, "ns")
		print("			DAC write latency:", self.bank_DAC_write_latency, "ns")
		print("			ADC write latency:", self.bank_ADC_write_latency, "ns")
		print("			digital part write latency:", self.bank_digital_write_latency, "ns")
		print("				|---adder write latency:", self.bank_adder_write_latency, "ns")
		print("				|---shift-reg write latency:", self.bank_shiftreg_write_latency, "ns")
		print("--------------------Bank Power Simulation Results-------------------")
		print("bank read power:", self.bank_read_power, "W")
		print("			crossbar read power:", self.bank_xbar_read_power, "W")
		print("			DAC read power:", self.bank_DAC_read_power, "W")
		print("			ADC read power:", self.bank_ADC_read_power, "W")
		print("			digital part read power:", self.bank_digital_read_power, "W")
		print("				|---adder read power:", self.bank_adder_read_power, "W")
		print("				|---shift-reg read power:", self.bank_shiftreg_read_power, "W")
		print("bank write power:", self.bank_write_power, "W")
		print("			crossbar write power:", self.bank_xbar_write_power, "W")
		print("			DAC write power:", self.bank_DAC_write_power, "W")
		print("			ADC write power:", self.bank_ADC_write_power, "W")
		print("			digital part write power:", self.bank_digital_write_power, "W")
		print("				|---adder write power:", self.bank_adder_write_power, "W")
		print("				|---shift-reg write power:", self.bank_shiftreg_write_power, "W")
		print("------------------Energy Simulation Results----------------------")
		print("bank read energy:", self.bank_read_energy, "nJ")
		print("			crossbar read energy:", self.bank_xbar_read_energy, "nJ")
		print("			DAC read energy:", self.bank_DAC_read_energy, "nJ")
		print("			ADC read energy:", self.bank_ADC_read_energy, "nJ")
		print("			digital part read energy:", self.bank_digital_read_energy, "nJ")
		print("				|---adder read energy:", self.bank_adder_read_energy, "nJ")
		print("				|---shift-reg read energy:", self.bank_shiftreg_read_energy, "nJ")
		print("bank write energy:", self.bank_write_energy, "nJ")
		print("			crossbar write energy:", self.bank_xbar_write_energy, "nJ")
		print("			DAC write energy:", self.bank_DAC_write_energy, "nJ")
		print("			ADC write energy:", self.bank_ADC_write_energy, "nJ")
		print("			digital part write energy:", self.bank_digital_write_energy, "nJ")
		print("				|---adder write energy:", self.bank_adder_write_energy, "nJ")
		print("				|---shift-reg write energy:", self.bank_shiftreg_write_energy, "nJ")
		print("-----------------------------------------------------------------")
	
def bank_test():
	print("load file:",test_SimConfig_path)
	_bank = bank(test_SimConfig_path,[1,1,1,2])
	_bank.calculate_bank_area()
	_bank.calculate_bank_read_latency()
	_bank.calculate_bank_write_latency()
	_bank.calculate_bank_read_power()
	_bank.calculate_bank_write_power()
	_bank.calculate_bank_read_energy()
	_bank.calculate_bank_write_energy()
	_bank.bank_output()


if __name__ == '__main__':
	bank_test()