#!/usr/bin/python
# -*-coding:utf-8-*-
import configparser as cp
import os
import math
from numpy import *
import numpy as np
from MNSIM.Hardware_Model.Crossbar import crossbar
from MNSIM.Hardware_Model.DAC import DAC
from MNSIM.Hardware_Model.ADC import ADC
from MNSIM.Hardware_Model.Adder import adder
from MNSIM.Hardware_Model.ShiftReg import shiftreg
test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),"SimConfig.ini")
# Default SimConfig file path: MNSIM_Python/SimConfig.ini


class ProcessElement(crossbar, DAC, ADC):
	def __init__(self, SimConfig_path):
		crossbar.__init__(self, SimConfig_path)
		DAC.__init__(self, SimConfig_path)
		ADC.__init__(self, SimConfig_path)
		PE_config = cp.ConfigParser()
		PE_config.read(SimConfig_path, encoding='UTF-8')
		__xbar_polarity = int(PE_config.get('Process element level', 'Xbar_Polarity'))
		# self.PE_multiplex_xbar_num = list(
		# 	map(int, PE_config.get('Process element level', 'Multiplex_Xbar_Num').split(',')))
		if __xbar_polarity == 1:
			self.PE_multiplex_xbar_num = [1,1]
		else:
			assert __xbar_polarity == 2, "Crossbar polarity must be 1 or 2"
			self.PE_multiplex_xbar_num = [1,2]
		self.group_num = int(PE_config.get('Process element level', 'Group_Num'))
		if self.group_num == 0:
			self.group_num = 1
		self.num_occupied_group = 0
		self.PE_xbar_num = self.group_num * self.PE_multiplex_xbar_num[0] * self.PE_multiplex_xbar_num[1]
		# self.polarity = PE_config.get('Algorithm Configuration', 'Weight_Polarity')
		# if self.polarity == 2:
		# 	assert self.PE_xbar_num[1]%2 == 0
		self.PE_simulation_level = int(PE_config.get('Algorithm Configuration', 'Simulation_Level'))
		self.PE_xbar_list = []
		self.PE_xbar_enable = []
		for i in range(self.group_num):
			self.PE_xbar_list.append([])
			self.PE_xbar_enable.append([])
			for j in range(self.PE_multiplex_xbar_num[0] * self.PE_multiplex_xbar_num[1]):
				__xbar = crossbar(SimConfig_path)
				self.PE_xbar_list[i].append(__xbar)
				self.PE_xbar_enable[i].append(0)

		self.PE_multiplex_ADC_num = int(PE_config.get('Process element level', 'ADC_Num'))
		self.PE_multiplex_DAC_num = int(PE_config.get('Process element level', 'DAC_Num'))
		self.PE_ADC_num = 0
		self.PE_DAC_num = 0

		self.input_demux = 0
		self.input_demux_power = 0
		self.input_demux_area = 0
		self.output_mux = 0
		self.output_mux_power = 0
		self.output_mux_area = 0

		self.calculate_ADC_num()
		self.calculate_DAC_num()

		self.PE_adder = adder(SimConfig_path)
		self.PE_adder_num = 0
		self.PE_shiftreg = shiftreg(SimConfig_path)

		self.PE_utilization = 0
		self.PE_max_occupied_column = 0

		self.PE_area = 0
		self.PE_xbar_area = 0
		self.PE_ADC_area = 0
		self.PE_DAC_area = 0
		self.PE_adder_area = 0
		self.PE_shiftreg_area = 0
		self.PE_input_demux_area = 0
		self.PE_output_mux_area = 0
		self.PE_digital_area = 0

		self.PE_read_power = 0
		self.PE_xbar_read_power = 0
		self.PE_ADC_read_power = 0
		self.PE_DAC_read_power = 0
		self.PE_adder_read_power = 0
		self.PE_shiftreg_read_power = 0
		self.input_demux_read_power = 0
		self.output_mux_read_power = 0
		self.PE_digital_read_power = 0

		self.PE_write_power = 0
		self.PE_xbar_write_power = 0
		self.PE_ADC_write_power = 0
		self.PE_DAC_write_power = 0
		self.PE_adder_write_power = 0
		self.PE_shiftreg_write_power = 0
		self.input_demux_write_power = 0
		self.output_mux_write_power = 0
		self.PE_digital_write_power = 0

		self.PE_read_latency = 0
		self.PE_xbar_read_latency = 0
		self.PE_ADC_read_latency = 0
		self.PE_DAC_read_latency = 0
		self.PE_adder_read_latency = 0
		self.PE_shiftreg_read_latency = 0
		self.input_demux_read_latency = 0
		self.output_mux_read_latency = 0
		self.PE_digital_read_latency = 0

		self.PE_write_latency = 0
		self.PE_xbar_write_latency = 0
		self.PE_ADC_write_latency = 0
		self.PE_DAC_write_latency = 0
		self.PE_adder_write_latency = 0
		self.PE_shiftreg_write_latency = 0
		self.input_demux_write_latency = 0
		self.output_mux_write_latency = 0
		self.PE_digital_write_latency = 0

		self.PE_read_energy = 0
		self.PE_xbar_read_energy = 0
		self.PE_ADC_read_energy = 0
		self.PE_DAC_read_energy = 0
		self.PE_adder_read_energy = 0
		self.PE_shiftreg_read_energy = 0
		self.input_demux_read_energy = 0
		self.output_mux_read_energy = 0
		self.PE_digital_read_energy = 0

		self.PE_write_energy = 0
		self.PE_xbar_write_energy = 0
		self.PE_ADC_write_energy = 0
		self.PE_DAC_write_energy = 0
		self.PE_adder_write_energy = 0
		self.PE_shiftreg_write_energy = 0
		self.input_demux_write_energy = 0
		self.output_mux_write_energy = 0
		self.PE_digital_write_energy = 0

		self.calculate_inter_PE_connection()

		# print("Process element configuration is loaded")
		# self.calculate_demux_area()
		# self.calculate_mux_area()
		# self.calculate_PE_area()
		# self.calculate_PE_read_latency()
		# self.calculate_PE_write_latency()
		# self.calculate_demux_power()
		# self.calculate_mux_power()
		# self.calculate_PE_read_power()
		# self.calculate_PE_write_power()
		# self.calculate_PE_read_energy()
		# self.calculate_PE_write_energy()

	def calculate_ADC_num(self):
		self.calculate_xbar_area()
		self.calculate_ADC_area()
		# print("xbar area", self.xbar_area)
		# print("ADC area", self.ADC_area)
		# print("mul", self.PE_multiplex_xbar_num[1])
		if self.PE_multiplex_ADC_num == 0:
			self.PE_multiplex_ADC_num = min(math.ceil(math.sqrt(self.xbar_area)*self.PE_multiplex_xbar_num[1]/math.sqrt(self.ADC_area)), self.xbar_column)
		else:
			assert self.PE_multiplex_ADC_num > 0, "ADC number in one group < 0"
		# print("ADC_num", self.PE_multiplex_ADC_num)
		self.PE_ADC_num = self.group_num * self.PE_multiplex_ADC_num
		# self.output_mux = math.ceil(self.xbar_column*self.PE_multiplex_xbar_num[1]/self.PE_multiplex_ADC_num)
		self.output_mux = math.ceil(self.xbar_column/self.PE_multiplex_ADC_num)
		# print("output_mux",self.output_mux)
		assert self.output_mux > 0

	def calculate_DAC_num(self):
		self.calculate_xbar_area()
		self.calculate_DAC_area()
		if self.PE_multiplex_DAC_num == 0:
			self.PE_multiplex_DAC_num = min(math.ceil(math.sqrt(self.xbar_area) * self.PE_multiplex_xbar_num[0] / math.sqrt(self.DAC_area)), self.xbar_row)
		else:
			assert self.PE_multiplex_DAC_num > 0, "DAC number in one group < 0"
		self.PE_DAC_num = self.group_num * self.PE_multiplex_DAC_num
		# print(self.PE_multiplex_DAC_num)
		self.input_demux = math.ceil(self.xbar_row*self.PE_multiplex_xbar_num[0]/self.PE_multiplex_DAC_num)
		assert self.input_demux > 0

	def calculate_demux_area(self):
		transistor_area = 10* self.transistor_tech * self.transistor_tech / 1000000
		demux_area_dict = {2: 8*transistor_area, # 2-1: 8 transistors
						   4: 24*transistor_area, # 4-1: 3 * 2-1
						   8: 72*transistor_area,
						   16: 216*transistor_area,
						   32: 648*transistor_area,
						   64: 1944*transistor_area
		}
		# unit: um^2
		# TODO: add circuits simulation results
		if self.input_demux <= 2:
			self.input_demux_area = demux_area_dict[2]
		elif self.input_demux<=4:
			self.input_demux_area = demux_area_dict[4]
		elif self.input_demux<=8:
			self.input_demux_area = demux_area_dict[8]
		elif self.input_demux<=16:
			self.input_demux_area = demux_area_dict[16]
		elif self.input_demux<=32:
			self.input_demux_area = demux_area_dict[32]
		else:
			self.input_demux_area = demux_area_dict[64]

	def calculate_mux_area(self):
		transistor_area = 10* self.transistor_tech * self.transistor_tech / 1000000
		mux_area_dict = {2: 8*transistor_area,
						 4: 24*transistor_area,
						 8: 72*transistor_area,
						 16: 216*transistor_area,
						 32: 648*transistor_area,
						 64: 1944*transistor_area
		}
		# unit: um^2
		# TODO: add circuits simulation results
		if self.output_mux <= 2:
			self.output_mux_area = mux_area_dict[2]
		elif self.output_mux <= 4:
			self.output_mux_area = mux_area_dict[4]
		elif self.output_mux <= 8:
			self.output_mux_area = mux_area_dict[8]
		elif self.output_mux <= 16:
			self.output_mux_area = mux_area_dict[16]
		elif self.output_mux <= 32:
			self.output_mux_area = mux_area_dict[32]
		else:
			self.output_mux_area = mux_area_dict[64]

	def calculate_inter_PE_connection(self):
		temp = self.group_num
		self.PE_adder_num = 0
		while temp/2 >= 1:
			self.PE_adder_num += int(temp/2)
			temp = int(temp/2) + temp%2

	def PE_read_config(self, read_row = None, read_column = None, read_matrix = None, read_vector = None):
		# read_row and read_column are lists with the length of #occupied groups
		# read_matrix is a 2D list of matrices. The size of the list is (#occupied groups x Xbar_Polarity)
		# read_vector is a list of vectors with the length of #occupied groups
		self.PE_utilization = 0
		if self.PE_simulation_level == 0:
			if (read_row is None) or (read_column is None):
				self.num_occupied_group = self.group_num
				for i in range(self.group_num):
					if self.PE_multiplex_xbar_num[1] == 1:
						self.PE_xbar_list[i][0].xbar_read_config()
						self.PE_xbar_enable[i][0] = 1
						self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
					else:
						self.PE_xbar_list[i][0].xbar_read_config()
						self.PE_xbar_enable[i][0] = 1
						self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
						self.PE_xbar_list[i][1].xbar_read_config()
						self.PE_xbar_enable[i][1] = 1
						self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
			else:
				assert len(read_row) == len(read_column), "read_row and read_column must be equal in length"
				self.num_occupied_group = len(read_row)
				assert self.num_occupied_group <= self.group_num, "The length of read_row exceeds the group number in one PE"
				for i in range(self.group_num):
					if i < self.num_occupied_group:
						if self.PE_multiplex_xbar_num[1] == 1:
							self.PE_xbar_list[i][0].xbar_read_config(read_row = read_row[i], read_column = read_column[i])
							self.PE_xbar_enable[i][0] = 1
							self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
						else:
							self.PE_xbar_list[i][0].xbar_read_config(read_row=read_row[i], read_column=read_column[i])
							self.PE_xbar_enable[i][0] = 1
							self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
							self.PE_xbar_list[i][1].xbar_read_config(read_row=read_row[i], read_column=read_column[i])
							self.PE_xbar_enable[i][1] = 1
							self.PE_utilization += self.PE_xbar_list[i][1].xbar_utilization
					else:
						if self.PE_multiplex_xbar_num[1] == 1:
							self.PE_xbar_enable[i][0] = 0
						else:
							self.PE_xbar_enable[i][0] = 0
							self.PE_xbar_enable[i][1] = 0
		else:
			if read_matrix is None:
				self.num_occupied_group = self.group_num
				for i in range(self.group_num):
					if self.PE_multiplex_xbar_num[1] == 1:
						self.PE_xbar_list[i][0].xbar_read_config()
						self.PE_xbar_enable[i][0] = 1
						self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
					else:
						self.PE_xbar_list[i][0].xbar_read_config()
						self.PE_xbar_enable[i][0] = 1
						self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
						self.PE_xbar_list[i][1].xbar_read_config()
						self.PE_xbar_enable[i][1] = 1
						self.PE_utilization += self.PE_xbar_list[i][1].xbar_utilization
			else:
				if read_vector is None:
					self.num_occupied_group = len(read_matrix)
					assert self.num_occupied_group <= self.group_num, "The number of read_matrix exceeds the group number in one PE"
					for i in range(self.group_num):
						if i < self.num_occupied_group:
							if self.PE_multiplex_xbar_num[1] == 1:
								self.PE_xbar_list[i][0].xbar_read_config(read_matrix=read_matrix[i][0])
								self.PE_xbar_enable[i][0] = 1
								self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
							else:
								self.PE_xbar_list[i][0].xbar_read_config(read_matrix=read_matrix[i][0])
								self.PE_xbar_enable[i][0] = 1
								self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
								self.PE_xbar_list[i][1].xbar_read_config(read_matrix=read_matrix[i][1])
								self.PE_xbar_enable[i][1] = 1
								self.PE_utilization += self.PE_xbar_list[i][1].xbar_utilization
						else:
							if self.PE_multiplex_xbar_num[1] == 1:
								self.PE_xbar_enable[i][0] = 0
							else:
								self.PE_xbar_enable[i][0] = 0
								self.PE_xbar_enable[i][1] = 0
				else:
					assert len(read_matrix) == len(read_vector), "The number of read_matrix and read_vector must be equal"
					self.num_occupied_group = len(read_matrix)
					assert self.num_occupied_group <= self.group_num, "The number of read_matrix/read_vector exceeds the group number in one PE"
					for i in range(self.group_num):
						if i < self.num_occupied_group:
							if self.PE_multiplex_xbar_num[1] == 1:
								self.PE_xbar_list[i][0].xbar_read_config(read_matrix = read_matrix[i][0], read_vector = read_vector[i])
								self.PE_xbar_enable[i][0] = 1
								self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
							else:
								self.PE_xbar_list[i][0].xbar_read_config(read_matrix = read_matrix[i][0], read_vector = read_vector[i])
								self.PE_xbar_enable[i][0] = 1
								self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
								self.PE_xbar_list[i][1].xbar_read_config(read_matrix = read_matrix[i][1], read_vector = read_vector[i])
								self.PE_xbar_enable[i][1] = 1
								self.PE_utilization += self.PE_xbar_list[i][1].xbar_utilization
						else:
							if self.PE_multiplex_xbar_num[1] == 1:
								self.PE_xbar_enable[i][0] = 0
							else:
								self.PE_xbar_enable[i][0] = 0
								self.PE_xbar_enable[i][1] = 0
		self.PE_utilization /= (self.group_num * self.PE_multiplex_xbar_num[1])

	def PE_write_config(self, write_row=None, write_column=None, write_matrix=None, write_vector=None):
		# write_row and write_column are array with the length of #occupied groups
		# write_matrix is a 2D array of matrices. The size of the list is (#occupied groups x Xbar_Polarity)
		# write_vector is a array of vector with the length of #occupied groups
		if self.PE_simulation_level == 0:
			if (write_row is None) or (write_column is None):
				self.num_occupied_group = self.group_num
				for i in range(self.group_num):
					if self.PE_multiplex_xbar_num[1] == 1:
						self.PE_xbar_list[i][0].xbar_write_config()
						self.PE_xbar_enable[i][0] = 1
						self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
					else:
						self.PE_xbar_list[i][0].xbar_write_config()
						self.PE_xbar_enable[i][0] = 1
						self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
						self.PE_xbar_list[i][1].xbar_write_config()
						self.PE_xbar_enable[i][1] = 1
						self.PE_utilization += self.PE_xbar_list[i][1].xbar_utilization
			else:
				assert len(write_row) == len(write_column), "write_row and write_column must be equal in length"
				self.num_occupied_group = len(write_row)
				assert self.num_occupied_group <= self.group_num, "The length of write_row exceeds the group number in one PE"
				for i in range(self.group_num):
					if i < self.num_occupied_group:
						if self.PE_multiplex_xbar_num[1] == 1:
							self.PE_xbar_list[i][0].xbar_write_config(write_row=write_row[i], write_column=write_column[i])
							self.PE_xbar_enable[i][0] = 1
							self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
						else:
							self.PE_xbar_list[i][0].xbar_write_config(write_row=write_row[i], write_column=write_column[i])
							self.PE_xbar_enable[i][0] = 1
							self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
							self.PE_xbar_list[i][1].xbar_write_config(write_row=write_row[i], write_column=write_column[i])
							self.PE_xbar_enable[i][1] = 1
							self.PE_utilization += self.PE_xbar_list[i][1].xbar_utilization
					else:
						if self.PE_multiplex_xbar_num[1] == 1:
							self.PE_xbar_enable[i][0] = 0
						else:
							self.PE_xbar_enable[i][0] = 0
							self.PE_xbar_enable[i][1] = 0
		else:
			if write_matrix is None:
				self.num_occupied_group = self.group_num
				for i in range(self.group_num):
					if self.PE_multiplex_xbar_num[1] == 1:
						self.PE_xbar_list[i][0].xbar_write_config()
						self.PE_xbar_enable[i][0] = 1
						self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
					else:
						self.PE_xbar_list[i][0].xbar_write_config()
						self.PE_xbar_enable[i][0] = 1
						self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
						self.PE_xbar_list[i][1].xbar_write_config()
						self.PE_xbar_enable[i][1] = 1
						self.PE_utilization += self.PE_xbar_list[i][1].xbar_utilization
			else:
				if write_vector is None:
					self.num_occupied_group = len(write_matrix)
					assert self.num_occupied_group <= self.group_num, "The number of write_matrix exceeds the group number in one PE"
					for i in range(self.group_num):
						if i < self.num_occupied_group:
							if self.PE_multiplex_xbar_num[1] == 1:
								self.PE_xbar_list[i][0].xbar_write_config(write_matrix=write_matrix[i][0])
								self.PE_xbar_enable[i][0] = 1
								self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
							else:
								self.PE_xbar_list[i][0].xbar_write_config(write_matrix=write_matrix[i][0])
								self.PE_xbar_enable[i][0] = 1
								self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
								self.PE_xbar_list[i][1].xbar_write_config(write_matrix=write_matrix[i][1])
								self.PE_xbar_enable[i][1] = 1
								self.PE_utilization += self.PE_xbar_list[i][1].xbar_utilization
						else:
							if self.PE_multiplex_xbar_num[1] == 1:
								self.PE_xbar_enable[i][0] = 0
							else:
								self.PE_xbar_enable[i][0] = 0
								self.PE_xbar_enable[i][1] = 0
				else:
					assert len(write_matrix) == len(write_vector), "The number of write_matrix and write_vector must be equal"
					self.num_occupied_group = len(write_matrix)
					assert self.num_occupied_group <= self.group_num, "The number of write_matrix/write_vector exceeds the group number in one PE"
					for i in range(self.group_num):
						if i < self.num_occupied_group:
							if self.PE_multiplex_xbar_num[1] == 1:
								self.PE_xbar_list[i][0].xbar_write_config(write_matrix=write_matrix[i][0],
																	  write_vector=write_vector[i])
								self.PE_xbar_enable[i][0] = 1
								self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
							else:
								self.PE_xbar_list[i][0].xbar_write_config(write_matrix=write_matrix[i][0],
																	  write_vector=write_vector[i])
								self.PE_xbar_enable[i][0] = 1
								self.PE_utilization += self.PE_xbar_list[i][0].xbar_utilization
								self.PE_xbar_list[i][1].xbar_write_config(write_matrix=write_matrix[i][1],
																	  write_vector=write_vector[i])
								self.PE_xbar_enable[i][1] = 1
								self.PE_utilization += self.PE_xbar_list[i][1].xbar_utilization
						else:
							if self.PE_multiplex_xbar_num[1] == 1:
								self.PE_xbar_enable[i][0] = 0
							else:
								self.PE_xbar_enable[i][0] = 0
								self.PE_xbar_enable[i][1] = 0
		self.PE_utilization /= (self.group_num * self.PE_multiplex_xbar_num[1])

	def calculate_PE_area(self):
		# unit: um^2
		self.calculate_xbar_area()
		self.calculate_demux_area()
		self.calculate_mux_area()
		self.calculate_DAC_area()
		self.calculate_ADC_area()
		self.PE_adder.calculate_adder_area()
		self.PE_shiftreg.calculate_shiftreg_area()
		self.PE_xbar_area = self.PE_xbar_num*self.xbar_area
		self.PE_ADC_area = self.ADC_area*self.PE_ADC_num
		self.PE_DAC_area = self.DAC_area*self.PE_DAC_num
		self.PE_adder_area = self.PE_multiplex_ADC_num*self.PE_adder_num*self.PE_adder.adder_area
		self.PE_shiftreg_area = self.PE_ADC_num*self.PE_shiftreg.shiftreg_area
		self.PE_input_demux_area = self.input_demux_area*self.PE_DAC_num
		self.PE_output_mux_area = self.output_mux_area*self.PE_ADC_num
		self.PE_digital_area = self.PE_adder_area + self.PE_shiftreg_area + self.PE_input_demux_area + self.PE_output_mux_area
		self.PE_area = self.PE_xbar_area + self.PE_ADC_area + self.PE_DAC_area + self.PE_digital_area

	def calculate_PE_read_latency(self):
		# Notice: before calculating latency, PE_read_config must be executed
		# unit: ns
		self.PE_xbar_read_latency = 0
		self.calculate_xbar_read_latency()
		self.calculate_DAC_sample_rate()
		self.calculate_ADC_sample_rate()
		max_multiple_time = 0
		for i in range(self.group_num):
			if self.PE_xbar_enable[i][0] == 1:
				multiple_time = math.ceil(self.PE_xbar_list[i][0].xbar_num_read_row/self.PE_multiplex_DAC_num)\
						   * math.ceil(self.PE_xbar_list[i][0].xbar_num_read_column/self.PE_multiplex_ADC_num)
				if multiple_time > max_multiple_time:
					max_multiple_time = multiple_time
		self.PE_xbar_read_latency = max_multiple_time * self.xbar_read_latency
		self.PE_ADC_read_latency = max_multiple_time * 1 / self.ADC_sample_rate * (self.ADC_precision + 2)
		self.PE_DAC_read_latency = max_multiple_time * 1 / self.DAC_sample_rate * (self.DAC_precision + 2)
		# TODO: check the ADDA latency formalution
		# TODO: Add the mux/demux latency
		self.input_demux_read_latency = 0
		self.output_mux_read_latency = 0
		self.PE_adder_read_latency = max_multiple_time * self.PE_adder.adder_latency * math.ceil(math.log2(self.group_num))
		self.PE_shiftreg_read_latency = max_multiple_time * self.PE_shiftreg.shiftreg_latency
		self.PE_digital_read_latency = self.input_demux_read_latency + self.output_mux_read_latency\
									   + self.PE_adder_read_latency + self.PE_shiftreg_read_latency
		# TODO: PipeLine optimization
		# ignore the latency of MUX and DEMUX
		self.PE_read_latency = self.PE_xbar_read_latency + self.PE_ADC_read_latency\
							   + self.PE_DAC_read_latency + self.PE_digital_read_latency

	def calculate_PE_write_latency(self):
		# Notice: before calculating latency, PE_write_config must be executed
		# unit: ns
		self.PE_xbar_write_latency = 0
		self.calculate_DAC_sample_rate()
		max_write_row = 0
		for i in range(self.group_num):
			if self.PE_xbar_enable[i][0] == 1:
				write_row = self.PE_xbar_list[i][0].xbar_num_write_row
				if write_row > max_write_row:
					max_write_row = write_row
				self.PE_xbar_list[i][0].calculate_xbar_write_latency()
				if self.PE_xbar_list[i][0].xbar_write_latency > self.PE_xbar_write_latency:
					self.PE_xbar_write_latency = self.PE_xbar_list[i][0].xbar_write_latency
		self.PE_DAC_write_latency = max_write_row * 1 / self.DAC_sample_rate * self.DAC_precision
		self.PE_ADC_write_latency = 0
		# TODO: Add the demux latency
		self.input_demux_write_latency = 0
		self.output_mux_write_latency = 0
		self.PE_adder_write_latency = 0
		self.PE_shiftreg_write_latency = 0
		self.PE_digital_write_latency = self.input_demux_write_latency + self.output_mux_write_latency \
										+ self.PE_adder_write_latency + self.PE_shiftreg_write_latency
		# TODO: PipeLine optimization
		self.PE_write_latency = self.PE_xbar_write_latency + self.PE_ADC_write_latency\
								+ self.PE_DAC_write_latency + self.PE_digital_write_latency

	def calculate_demux_power(self):
		transistor_power = 10*1.2/1e9
		demux_power_dict = {2: 8*transistor_power,
						 4: 24*transistor_power,
						 8: 72*transistor_power,
						 16: 216*transistor_power,
						 32: 648*transistor_power,
						 64: 1944*transistor_power
		}
		# unit: W
		# TODO: add circuits simulation results
		if self.input_demux <= 2:
			self.input_demux_power = demux_power_dict[2]
		elif self.input_demux<=4:
			self.input_demux_power = demux_power_dict[4]
		elif self.input_demux<=8:
			self.input_demux_power = demux_power_dict[8]
		elif self.input_demux<=16:
			self.input_demux_power = demux_power_dict[16]
		elif self.input_demux<=32:
			self.input_demux_power = demux_power_dict[32]
		else:
			self.input_demux_power = demux_power_dict[64]

	def calculate_mux_power(self):
		transistor_power = 10*1.2/1e9
		mux_power_dict = {2: 8*transistor_power,
						 4: 24*transistor_power,
						 8: 72*transistor_power,
						 16: 216*transistor_power,
						 32: 648*transistor_power,
						 64: 1944*transistor_power
		}
		# unit: W
		# TODO: add circuits simulation results
		if self.output_mux <= 2:
			self.output_mux_power = mux_power_dict[2]
		elif self.output_mux <= 4:
			self.output_mux_power = mux_power_dict[4]
		elif self.output_mux <= 8:
			self.output_mux_power = mux_power_dict[8]
		elif self.output_mux <= 16:
			self.output_mux_power = mux_power_dict[16]
		elif self.output_mux <= 32:
			self.output_mux_power = mux_power_dict[32]
		else:
			self.output_mux_power = mux_power_dict[64]

	def calculate_PE_read_power(self):
		# unit: W
		# Notice: before calculating latency, PE_read_config must be executed
		self.calculate_DAC_power()
		self.calculate_ADC_power()
		self.calculate_demux_power()
		self.calculate_mux_power()
		self.PE_read_power = 0
		self.PE_xbar_read_power = 0
		self.PE_ADC_read_power = 0
		self.PE_DAC_read_power = 0
		self.PE_adder_read_power = 0
		self.PE_shiftreg_read_power = 0
		self.input_demux_read_power = 0
		self.output_mux_read_power = 0
		self.PE_digital_read_power = 0
		self.PE_max_occupied_column = 0
		if self.num_occupied_group != 0:
			for i in range(self.group_num):
				if self.PE_xbar_enable[i][0] == 1:
					if self.PE_multiplex_xbar_num[1] == 1:
						self.PE_xbar_list[i][0].calculate_xbar_read_power()
						self.PE_xbar_read_power += self.PE_xbar_list[i][0].xbar_read_power/self.input_demux/self.output_mux
					else:
						self.PE_xbar_list[i][0].calculate_xbar_read_power()
						self.PE_xbar_read_power += self.PE_xbar_list[i][0].xbar_read_power/self.input_demux/self.output_mux
						self.PE_xbar_list[i][1].calculate_xbar_read_power()
						self.PE_xbar_read_power += self.PE_xbar_list[i][1].xbar_read_power/self.input_demux/self.output_mux
					self.PE_DAC_read_power += math.ceil(self.PE_xbar_list[i][0].xbar_num_read_row/self.input_demux)*self.DAC_power
					self.PE_ADC_read_power += math.ceil(self.PE_xbar_list[i][0].xbar_num_read_column/self.output_mux)*self.ADC_power
					self.input_demux_read_power += math.ceil(self.PE_xbar_list[i][0].xbar_num_read_row/self.input_demux)*self.input_demux_power
					self.output_mux_read_power += math.ceil(self.PE_xbar_list[i][0].xbar_num_read_column/self.output_mux)*self.output_mux_power
					if self.PE_xbar_list[i][0].xbar_num_read_column > self.PE_max_occupied_column:
						self.PE_max_occupied_column = self.PE_xbar_list[i][0].xbar_num_read_column
			self.PE_adder_read_power = (self.num_occupied_group-1)*self.PE_max_occupied_column*self.PE_adder.adder_power
			self.PE_shiftreg_read_power = (self.num_occupied_group-1)*self.PE_max_occupied_column*self.PE_shiftreg.shiftreg_power
			self.PE_digital_read_power = self.input_demux_read_power + self.output_mux_read_power + self.PE_adder_read_power + self.PE_shiftreg_read_power
			self.PE_read_power = self.PE_xbar_read_power + self.PE_DAC_read_power + self.PE_ADC_read_power + self.PE_digital_read_power

	def calculate_PE_write_power(self):
		# unit: W
		# Notice: before calculating latency, PE_write_config must be executed
		self.calculate_DAC_power()
		self.calculate_ADC_power()
		self.calculate_demux_power()
		self.calculate_mux_power()
		self.PE_write_power = 0
		self.PE_xbar_write_power = 0
		self.PE_ADC_write_power = 0
		self.PE_DAC_write_power = 0
		self.PE_adder_write_power = 0
		self.PE_shiftreg_write_power = 0
		self.input_demux_write_power = 0
		self.output_mux_write_power = 0
		self.PE_digital_write_power = 0
		if self.num_occupied_group != 0:
			for i in range(self.group_num):
				if self.PE_xbar_enable[i][0] == 1:
					if self.PE_multiplex_xbar_num[1] == 1:
						self.PE_xbar_list[i][0].calculate_xbar_write_power()
						self.PE_xbar_write_power += self.PE_xbar_list[i][0].xbar_write_power
					else:
						self.PE_xbar_list[i][0].calculate_xbar_write_power()
						self.PE_xbar_write_power += self.PE_xbar_list[i][0].xbar_write_power
						self.PE_xbar_list[i][1].calculate_xbar_write_power()
						self.PE_xbar_write_power += self.PE_xbar_list[i][1].xbar_write_power
					self.PE_DAC_write_power += math.ceil(self.PE_xbar_list[i][0].xbar_num_write_row/self.input_demux)*self.DAC_power
					self.PE_ADC_write_power += 0
					# Assume ADCs are idle in write process
					self.input_demux_write_power += math.ceil(self.PE_xbar_list[i][0].xbar_num_write_row/self.input_demux)*self.input_demux_power
			self.PE_digital_write_power = self.input_demux_write_power + self.output_mux_read_power + self.PE_adder_read_power + self.PE_shiftreg_read_power
			self.PE_write_power = self.PE_xbar_write_power + self.PE_DAC_write_power + self.PE_ADC_write_power + self.PE_digital_write_power

	def calculate_PE_read_energy(self):
		# unit: nJ
		# Notice: before calculating energy, PE_read_config and calculate_PE_read_power must be executed
		self.PE_xbar_read_energy = self.PE_xbar_read_latency * self.PE_xbar_read_power
		self.PE_DAC_read_energy = self.PE_DAC_read_latency * self.PE_DAC_read_power
		self.PE_ADC_read_energy = self.PE_ADC_read_latency * self.PE_ADC_read_power
		self.PE_adder_read_energy = self.PE_adder_read_power * self.PE_adder_read_latency
		self.PE_shiftreg_read_energy = self.PE_shiftreg_read_power * self.PE_shiftreg_read_latency
		self.input_demux_read_energy = self.input_demux_read_power * self.input_demux_read_latency
		self.output_mux_read_energy = self.output_mux_read_power * self.output_mux_read_latency
		self.PE_digital_read_energy = self.PE_adder_read_energy + self.PE_shiftreg_read_energy + self.input_demux_read_energy + self.output_mux_read_energy
		self.PE_read_energy = self.PE_xbar_read_energy + self.PE_DAC_read_energy + \
							  self.PE_ADC_read_energy + self.PE_digital_read_energy

	def calculate_PE_write_energy(self):
		# unit: nJ
		# Notice: before calculating energy, PE_write_config and calculate_PE_write_power must be executed
		self.PE_xbar_write_energy = self.PE_xbar_write_latency * self.PE_xbar_write_power
		self.PE_DAC_write_energy = self.PE_DAC_write_latency * self.PE_DAC_write_power
		self.PE_ADC_write_energy = self.PE_ADC_write_latency * self.PE_ADC_write_power
		self.PE_adder_write_energy = self.PE_adder_write_power * self.PE_adder_write_latency
		self.PE_shiftreg_write_energy = self.PE_shiftreg_write_power * self.PE_shiftreg_write_latency
		self.input_demux_write_energy = self.input_demux_write_power * self.input_demux_write_latency
		self.output_mux_write_energy = self.output_mux_write_latency * self.output_mux_write_power
		self.PE_digital_write_energy = self.PE_adder_write_energy + self.PE_shiftreg_write_energy + self.input_demux_write_energy + self.output_mux_write_energy
		self.PE_write_energy = self.PE_xbar_write_energy + self.PE_DAC_write_energy + \
							  self.PE_ADC_write_energy + self.PE_digital_write_energy

	def PE_output(self):
		print("---------------------Crossbar Configurations-----------------------")
		crossbar.xbar_output(self)
		print("------------------------DAC Configurations-------------------------")
		DAC.DAC_output(self)
		print("------------------------ADC Configurations-------------------------")
		ADC.ADC_output(self)
		print("-------------------------PE Configurations-------------------------")
		print("total crossbar number in one PE:", self.PE_xbar_num)
		print("			the number of crossbars sharing a set of interfaces:",self.PE_multiplex_xbar_num)
		print("total utilization rate:", self.PE_utilization)
		print("total DAC number in one PE:", self.PE_DAC_num)
		print("			the number of DAC in one set of interfaces:", self.PE_multiplex_DAC_num)
		print("total ADC number in one PE:", self.PE_ADC_num)
		print("			the number of ADC in one set of interfaces:", self.PE_multiplex_ADC_num)
		print("---------------------PE Area Simulation Results--------------------")
		print("PE area:", self.PE_area, "um^2")
		print("			crossbar area:", self.PE_xbar_area, "um^2")
		print("			DAC area:", self.PE_DAC_area, "um^2")
		print("			ADC area:", self.PE_ADC_area, "um^2")
		print("			digital part area:", self.PE_digital_area, "um^2")
		print("			|---adder area:", self.PE_adder_area, "um^2")
		print("			|---shift-reg area:", self.PE_shiftreg_area, "um^2")
		print("			|---input_demux area:", self.PE_input_demux_area, "um^2")
		print("			|---output_mux area:", self.PE_output_mux_area, "um^2")
		print("--------------------PE Latency Simulation Results-----------------")
		print("PE read latency:", self.PE_read_latency, "ns")
		print("			crossbar read latency:", self.PE_xbar_read_latency, "ns")
		print("			DAC read latency:", self.PE_DAC_read_latency, "ns")
		print("			ADC read latency:", self.PE_ADC_read_latency, "ns")
		print("			digital part read latency:", self.PE_digital_read_latency, "ns")
		print("PE write latency:", self.PE_write_latency, "ns")
		print("			crossbar write latency:", self.PE_xbar_write_latency, "ns")
		print("			DAC write latency:", self.PE_DAC_write_latency, "ns")
		print("			ADC write latency:", self.PE_ADC_write_latency, "ns")
		print("			digital part write latency:", self.PE_digital_write_latency, "ns")
		print("--------------------PE Power Simulation Results-------------------")
		print("PE read power:", self.PE_read_power, "W")
		print("			crossbar read power:", self.PE_xbar_read_power, "W")
		print("			DAC read power:", self.PE_DAC_read_power, "W")
		print("			ADC read power:", self.PE_ADC_read_power, "W")
		print("			digital part read power:", self.PE_digital_read_power, "W")
		print("PE write power:", self.PE_write_power, "W")
		print("			crossbar write power:", self.PE_xbar_write_power, "W")
		print("			DAC write power:", self.PE_DAC_write_power, "W")
		print("			ADC write power:", self.PE_ADC_write_power, "W")
		print("			digital part write power:", self.PE_digital_write_power, "W")
		print("------------------PE Energy Simulation Results--------------------")
		print("PE read energy:", self.PE_read_energy, "nJ")
		print("			crossbar read energy:", self.PE_xbar_read_energy, "nJ")
		print("			DAC read energy:", self.PE_DAC_read_energy, "nJ")
		print("			ADC read energy:", self.PE_ADC_read_energy, "nJ")
		print("			digital part read energy:", self.PE_digital_read_energy, "nJ")
		print("PE write energy:", self.PE_write_energy, "nJ")
		print("			crossbar write energy:", self.PE_xbar_write_energy, "nJ")
		print("			DAC write energy:", self.PE_DAC_write_energy, "nJ")
		print("			ADC write energy:", self.PE_ADC_write_energy, "nJ")
		print("			digital part write energy:", self.PE_digital_write_energy, "nJ")
		print("-----------------------------------------------------------------")
	
def PE_test():
	print("load file:",test_SimConfig_path)
	_PE = ProcessElement(test_SimConfig_path)
	# print(_PE.xbar_column)
	_PE.calculate_PE_area()
	_PE.calculate_PE_area()
	# _PE.PE_read_config(read_matrix=[ [ [[0,1],[1,1]], [[1,1],[0,0]] ], [ [[0,1,1],[1,1,0],[0,0,0]], [[1,1,1],[0,0,1],[1,0,1]] ] ],
	# 				read_vector=[[[0],[1]] , [[1],[0],[1]] ])
	_PE.PE_read_config()
	_PE.calculate_PE_read_power()
	_PE.calculate_PE_read_latency()
	_PE.calculate_PE_read_energy()
	# _PE.PE_write_config()
	# _PE.calculate_PE_write_power()
	# _PE.calculate_PE_write_latency()
	# _PE.calculate_PE_write_energy()
	_PE.PE_output()


if __name__ == '__main__':
	PE_test()