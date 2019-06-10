#!/usr/bin/python
# -*-coding:utf-8-*-
import configparser as cp
import os
import math
from numpy import *
import numpy as np
from Crossbar import crossbar
from DAC import DAC
from ADC import ADC
from Adder import adder
from ShiftReg import shiftreg
test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),"SimConfig.ini")
# Default SimConfig file path: MNSIM_Python/SimConfig.ini


class ProcessElement(crossbar, DAC, ADC):
	def __init__(self, SimConfig_path, layer_num = 0):
		crossbar.__init__(self, SimConfig_path)
		DAC.__init__(self, SimConfig_path)
		ADC.__init__(self, SimConfig_path)
		PE_config = cp.ConfigParser()
		PE_config.read(SimConfig_path)
		self.layer_num = layer_num
		self.PE_xbar_num = list(map(int, PE_config.get('Process element level', 'Xbar_Num').split(',')))
		if self.PE_xbar_num[0] == 0:
			self.PE_xbar_num[0] = 1
			self.PE_xbar_num[1] = 2
		assert self.PE_xbar_num[0] > 0
		assert self.PE_xbar_num[1] > 0
		self.PE_multiplex_xbar_num = list(map(int, PE_config.get('Process element level', 'Multiplex_Xbar_Num').split(',')))
		if self.PE_multiplex_xbar_num[0] == 0:
			self.PE_multiplex_xbar_num[0] = 1
			self.PE_multiplex_xbar_num[1] = 2
		assert self.PE_multiplex_xbar_num[0] > 0 & self.PE_multiplex_xbar_num[0] <= self.PE_xbar_num[0]
		assert self.PE_multiplex_xbar_num[1] > 0 & self.PE_multiplex_xbar_num[1] <= self.PE_xbar_num[1]
		self.num_group = int(
			self.PE_xbar_num[0] / self.PE_multiplex_xbar_num[0] * self.PE_xbar_num[1] / self.PE_multiplex_xbar_num[1])






		self.PE_ADC_num = int(PE_config.get('Process element level', 'ADC_Num'))
		self.PE_DAC_num = int(PE_config.get('Process element level', 'DAC_Num'))
		self.PE_multiplex_ADC_num = 0
		self.PE_multiplex_DAC_num = 0
		self.input_demux = 0
		self.input_demux_area = 0
		self.input_demux_power = 0
		self.output_mux = 0
		self.output_mux_area = 0
		self.output_mux_power = 0

		self.PE_usage_utilization = 0

		self.PE_area = 0
		self.PE_xbar_area = 0
		self.PE_ADC_area = 0
		self.PE_DAC_area = 0
		self.PE_digital_area = 0

		self.PE_read_power = 0
		self.PE_xbar_read_power = 0
		self.PE_ADC_read_power = 0
		self.PE_DAC_read_power = 0
		self.PE_digital_read_power = 0

		self.PE_write_power = 0
		self.PE_xbar_write_power = 0
		self.PE_ADC_write_power = 0
		self.PE_DAC_write_power = 0
		self.PE_digital_write_power = 0

		self.PE_read_latency = 0
		self.PE_xbar_read_latency = 0
		self.PE_ADC_read_latency = 0
		self.PE_DAC_read_latency = 0
		self.PE_digital_read_latency = 0

		self.PE_write_latency = 0
		self.PE_xbar_write_latency = 0
		self.PE_ADC_write_latency = 0
		self.PE_DAC_write_latency = 0
		self.PE_digital_write_latency = 0

		self.PE_read_energy = 0
		self.PE_xbar_read_energy = 0
		self.PE_ADC_read_energy = 0
		self.PE_DAC_read_energy = 0
		self.PE_digital_read_energy = 0

		self.PE_write_energy = 0
		self.PE_xbar_write_energy = 0
		self.PE_ADC_write_energy = 0
		self.PE_DAC_write_energy = 0
		self.PE_digital_write_energy = 0

		self.PE_
		# print("Process element configuration is loaded")
		self.calculate_ADC_num()
		self.calculate_DAC_num()
		# self.calculate_demux_area()
		# self.calculate_mux_area()
		# self.calculate_PE_area()
		# self.calculate_PE_read_latency()
		# self.calculate_PE_write_latency()
		self.calculate_demux_power()
		self.calculate_mux_power()
		# self.calculate_PE_read_power()
		# self.calculate_PE_write_power()
		# self.calculate_PE_read_energy()
		# self.calculate_PE_write_energy()

	def calculate_ADC_num(self):
		self.calculate_xbar_area()
		self.calculate_ADC_area()
		if self.PE_ADC_num == 0:
			self.PE_multiplex_ADC_num = int(math.sqrt(self.xbar_area)*self.PE_multiplex_xbar_num[1]/math.sqrt(self.ADC_area))
			self.PE_ADC_num = self.PE_xbar_num[0]/self.PE_multiplex_xbar_num[0] * self.PE_xbar_num[1]/self.PE_multiplex_xbar_num[1]\
									  * self.PE_multiplex_ADC_num
			self.output_mux = math.ceil(self.xbar_column*self.PE_multiplex_xbar_num[1]/self.PE_multiplex_ADC_num)
		else:
			assert self.PE_ADC_num > 0
			self.PE_multiplex_ADC_num = int(self.PE_ADC_num/(self.PE_xbar_num[0]
										/self.PE_multiplex_xbar_num[0]*self.PE_xbar_num[1]/self.PE_multiplex_xbar_num[1]))
			self.output_mux = math.ceil(self.xbar_column*self.PE_multiplex_xbar_num[1]/self.PE_multiplex_ADC_num)

	def calculate_DAC_num(self):
		self.calculate_xbar_area()
		self.calculate_DAC_area()
		if self.PE_DAC_num == 0:
			self.PE_multiplex_DAC_num = int(math.sqrt(self.xbar_area) * self.PE_multiplex_xbar_num[0] / math.sqrt(self.DAC_area))
			self.PE_DAC_num = self.PE_xbar_num[0] / self.PE_multiplex_xbar_num[0] * self.PE_xbar_num[1] / self.PE_multiplex_xbar_num[1] \
									  * self.PE_multiplex_DAC_num
			self.input_demux = math.ceil(self.xbar_row*self.PE_multiplex_xbar_num[0]/self.PE_multiplex_DAC_num)
		else:
			assert self.PE_DAC_num > 0
			self.PE_multiplex_DAC_num = int(self.PE_DAC_num/(self.PE_xbar_num[0]
							            /self.PE_multiplex_xbar_num[0]*self.PE_xbar_num[1]/self.PE_multiplex_xbar_num[1]))
			self.input_demux = math.ceil(self.xbar_row*self.PE_multiplex_xbar_num[0]/self.PE_multiplex_DAC_num)

	def calculate_demux_area(self):
		demux_area_dict = {2: 0,
						   4: 0,
						   8: 0,
						   16: 0,
						   32: 0,
						   64: 0
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
		mux_area_dict = {2: 0,
						 4: 0,
						 8: 0,
						 16: 0,
						 32: 0,
						 64: 0
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

	def calculate_PE_area(self):
		# unit: um^2
		self.calculate_xbar_area()
		self.calculate_demux_area()
		self.calculate_mux_area()
		self.calculate_DAC_area()
		self.calculate_ADC_area()
		self.PE_xbar_area = self.PE_xbar_num[0]*self.PE_xbar_num[1]*self.xbar_area
		self.PE_ADC_area = self.ADC_area*self.PE_ADC_num
		self.PE_DAC_area = self.DAC_area*self.PE_DAC_num
		self.PE_digital_area = self.input_demux_area*self.PE_DAC_num+self.output_mux_area*self.PE_ADC_num
		self.PE_area = self.PE_xbar_area + self.PE_ADC_area + self.PE_DAC_area + self.PE_digital_area

	def calculate_PE_read_latency(self, read_row = None, read_column = None):
		# unit: ns
		if read_row is None:
			read_row = self.xbar_row * self.PE_multiplex_xbar_num[0]
		assert read_row >= 0
		if read_column is None:
			read_column = self.xbar_column * self.PE_multiplex_xbar_num[1]
		assert read_column >= 0
		multiple_time = math.ceil(read_row/self.PE_multiplex_DAC_num)*math.ceil(read_column/self.PE_multiplex_ADC_num)
		self.calculate_xbar_read_latency()
		self.calculate_DAC_sample_rate()
		self.calculate_ADC_sample_rate()
		self.PE_xbar_read_latency = multiple_time * self.xbar_read_latency
		self.PE_ADC_read_latency = multiple_time * 1/self.ADC_sample_rate
		self.PE_DAC_read_latency = multiple_time * 1/self.DAC_sample_rate
		self.PE_digital_read_energy = 0
		self.PE_read_latency = self.PE_xbar_read_latency+self.PE_ADC_read_latency+self.PE_DAC_read_latency+self.PE_digital_read_latency
		# ignore the latency of MUX and DEMUX

	def calculate_PE_write_latency(self, write_row = None, write_column = None):
		# unit: ns
		if write_row is None:
			write_row = self.xbar_row * self.PE_multiplex_xbar_num[0]
		assert write_row >= 0
		if write_column is None:
			write_column = self.xbar_column * self.PE_multiplex_xbar_num[1]
		assert write_column >= 0
		multiple_time = min(math.ceil(write_row/self.input_demux), write_column) * min(self.input_demux, write_row)
		self.calculate_xbar_write_latency()
		self.calculate_DAC_sample_rate()
		self.calculate_ADC_sample_rate()
		self.PE_xbar_write_latency = multiple_time * self.device_write_latency
		self.PE_DAC_write_latency = multiple_time * 1/self.DAC_sample_rate
		self.PE_ADC_write_latency = 0
		self.PE_digital_write_latency = 0
		self.PE_write_latency = self.PE_xbar_write_latency+self.PE_ADC_write_latency+self.PE_DAC_write_latency+self.PE_digital_write_latency
		# ignore the latency of DEMUX

	def calculate_demux_power(self):
		demux_power_dict = {2: 0,
						   4: 0,
						   8: 0,
						   16: 0,
						   32: 0,
						   64: 0
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
		mux_power_dict = {2: 0,
						 4: 0,
						 8: 0,
						 16: 0,
						 32: 0,
						 64: 0
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

	def calculate_PE_usage_utilization(self, app_mode = 0, read_column = None, read_row = None,
									   write_column = None, write_row = None):
		# app_mode: 0: read mode; 1: write mode
		# read_column/read_row/write_column/write_row are lists with the length of num_group
		num_xbar = self.PE_xbar_num[0] * self.PE_xbar_num[1]
		# num_group = int(self.PE_xbar_num[0] / self.PE_multiplex_xbar_num[0] * self.PE_xbar_num[1] / self.PE_multiplex_xbar_num[1])
		if app_mode == 0:
			if read_column is None:
				read_column_group = self.xbar_column * self.PE_multiplex_xbar_num[1]
				read_column = self.num_group * [read_column_group]
			if read_row is None:
				read_row_group = self.xbar_row * self.PE_multiplex_xbar_num[0]
				read_row = self.num_group * [read_row_group]
			num_total_read = np.multiply(np.array(read_column), np.array(read_row)).tolist()
			self.PE_usage_utilization = sum(num_total_read) / (self.xbar_column * self.xbar_row * num_xbar)
		else:
			if write_column is None:
				write_column_group = self.xbar_column * self.PE_multiplex_xbar_num[1]
				write_column = self.num_group * [write_column_group]
			if write_row is None:
				write_row_group = self.xbar_row * self.PE_multiplex_xbar_num[0]
				write_row = self.num_group * [write_row_group]
			num_total_write = np.multiply(np.array(write_column), np.array(write_row)).tolist()
			self.PE_usage_utilization = sum(num_total_write) / (self.xbar_column * self.xbar_row * num_xbar)

	def calculate_PE_read_power(self, cal_mode = 0, read_column = None, read_row = None, num_cell = None):
		# unit: W
		# read_column / read_row are lists with the length of num_group
		# num_cell is a 2D list with the size of (num_group, num_resistance_level)
		num_xbar = self.PE_xbar_num[0]*self.PE_xbar_num[1]
		# num_group = int(self.PE_xbar_num[0]/self.PE_multiplex_xbar_num[0] * self.PE_xbar_num[1]/self.PE_multiplex_xbar_num[1])
		self.calculate_DAC_power()
		self.calculate_ADC_power()
		self.calculate_demux_power()
		self.calculate_mux_power()
		if cal_mode == 0:
			if read_column is None:
				read_column_group = self.xbar_column*self.PE_multiplex_xbar_num[1]
				read_column = self.num_group*[read_column_group]
			if read_row is None:
				read_row_group = self.xbar_row*self.PE_multiplex_xbar_num[0]
				read_row = self.num_group*[read_row_group]
			assert min(read_column) >= 0
			assert min(read_row) >= 0
			utilization = (np.multiply(np.array(read_column), np.array(read_row))/
						  (self.xbar_column*self.xbar_row*self.PE_multiplex_xbar_num[0]*self.PE_multiplex_xbar_num[1])).tolist()
			self.PE_read_power = 0
			self.PE_xbar_read_power = 0
			self.PE_ADC_read_power = 0
			self.PE_DAC_read_power = 0
			self.PE_digital_read_power = 0
			for index in range(self.num_group):
				self.calculate_xbar_read_power(0, utilization[index], None, read_row[index], read_column[index])
				self.PE_xbar_read_power += self.xbar_read_power
				self.PE_DAC_read_power += math.ceil(read_row[index]/self.input_demux)*self.DAC_power
				self.PE_ADC_read_power += math.ceil(read_column[index]/self.output_mux)*self.ADC_power
				self.PE_digital_read_power += math.ceil(read_row[index]/self.input_demux)*self.input_demux_power\
									  + math.ceil(read_column[index]/self.output_mux)*self.output_mux_power
			self.PE_xbar_read_power = self.PE_xbar_read_power * int(max(read_column)/self.PE_multiplex_ADC_num) / max(read_column) \
									  * int(max(read_row)/self.PE_multiplex_DAC_num) / max(read_row)
			self.PE_read_power = self.PE_xbar_read_power+self.PE_DAC_read_power+self.PE_ADC_read_power+self.PE_digital_read_power
		else:
			self.PE_read_power = 0
			self.PE_xbar_read_power = 0
			self.PE_ADC_read_power = 0
			self.PE_DAC_read_power = 0
			self.PE_digital_read_power = 0
			if num_cell is None:
				self.calculate_xbar_read_power(1, 1, None, self.xbar_row, self.xbar_column)
				self.PE_xbar_read_power = self.xbar_read_power*self.PE_xbar_num[0]*self.PE_xbar_num[1]
				self.PE_DAC_read_power = self.DAC_power*self.PE_DAC_num
				self.PE_ADC_read_power = self.ADC_power*self.PE_ADC_num
				self.PE_digital_read_power = self.PE_DAC_num*self.input_demux_power+self.PE_ADC_num*self.output_mux_power
				self.PE_xbar_read_power = self.PE_xbar_read_power / self.input_demux / self.output_mux
				self.PE_read_power = self.PE_xbar_read_power+self.PE_DAC_read_power+self.PE_ADC_read_power\
									 +self.PE_digital_read_power
			else:
				if read_column is None:
					read_column_group = self.xbar_column * self.PE_multiplex_xbar_num[1]
					read_column = self.num_group * [read_column_group]
				if read_row is None:
					read_row_group = self.xbar_row * self.PE_multiplex_xbar_num[0]
					read_row = self.num_group * [read_row_group]
				assert min(read_column) >= 0
				assert min(read_row) >= 0
				# for index in range(num_xbar):
				# 	self.calculate_xbar_read_power(1, 1, num_cell[index])
				# 	self.PE_xbar_read_power += self.xbar_read_power
				for index in range(self.num_group):
					self.calculate_xbar_read_power(1,1,num_cell[index],read_row[index],read_column[index])
					self.PE_xbar_read_power += self.xbar_read_power
					self.PE_DAC_read_power += math.ceil(read_row[index] / self.input_demux) * self.DAC_power
					self.PE_ADC_read_power += math.ceil(read_column[index] / self.output_mux) * self.ADC_power
					self.PE_digital_read_power += math.ceil(read_row[index] / self.input_demux) * self.input_demux_power \
									    + math.ceil(read_column[index] / self.output_mux) * self.output_mux_power
				self.PE_xbar_read_power = self.PE_xbar_read_power * int(max(read_column) / self.PE_multiplex_ADC_num)\
										  / max(read_column) * int(max(read_row) / self.PE_multiplex_DAC_num) / max(read_row)
				self.PE_read_power = self.PE_xbar_read_power + self.PE_DAC_read_power + self.PE_ADC_read_power \
									 + self.PE_digital_read_power

	def calculate_PE_write_power(self, cal_mode = 0, write_column = None, write_row = None, num_cell = None):
		# unit: W
		# write_column / write_row are lists with the length of num_group
		# num_cell is a 2D list with the size of (num_group, num_resistance_level)
		num_xbar = self.PE_xbar_num[0] * self.PE_xbar_num[1]
		# num_group = int(self.PE_xbar_num[0] / self.PE_multiplex_xbar_num[0] * self.PE_xbar_num[1] / self.PE_multiplex_xbar_num[1])
		self.calculate_DAC_power()
		self.calculate_ADC_power()
		self.calculate_mux_power()
		self.calculate_demux_power()
		if cal_mode == 0:
			if write_column is None:
				write_column_group = self.xbar_column * self.PE_multiplex_xbar_num[1]
				write_column = self.num_group * [write_column_group]
			if write_row is None:
				write_row_group = self.xbar_row * self.PE_multiplex_xbar_num[0]
				write_row = self.num_group * [write_row_group]
			assert min(write_column) >= 0
			assert min(write_row) >= 0
			num_total_write = np.multiply(np.array(write_column), np.array(write_row)).tolist()
			# self.PE_usage_utilization = sum(num_total_write)/(self.xbar_column*self.xbar_row*num_xbar)
			self.PE_write_power = 0
			self.PE_xbar_write_power = 0
			self.PE_ADC_write_power = 0
			self.PE_DAC_write_power = 0
			self.PE_digital_write_power = 0
			for index in range(self.num_group):
				write_num = max(math.ceil(write_row[index]/self.input_demux), write_column[index])
				self.calculate_xbar_write_power(0, write_num, None, None, None)
				self.PE_xbar_write_power += self.xbar_write_power
				self.PE_DAC_write_power += math.ceil(write_row[index] / self.input_demux) * self.DAC_power
				self.PE_ADC_write_power += 0 # Assume ADCs are idle in write process
				self.PE_digital_write_power += math.ceil(write_row[index] / self.input_demux) * self.input_demux_power
			self.PE_write_power = self.PE_xbar_write_power + self.PE_DAC_write_power + self.PE_ADC_write_power + self.PE_digital_write_power
		else:
			self.PE_write_power = 0
			self.PE_xbar_write_power = 0
			self.PE_ADC_write_power = 0
			self.PE_DAC_write_power = 0
			self.PE_digital_write_power = 0
			if num_cell is None:
				self.calculate_xbar_write_power(1, None, None, None, None)
				self.PE_xbar_write_power = self.xbar_write_power * self.PE_xbar_num[0] * self.PE_xbar_num[1]
				self.PE_DAC_write_power = self.DAC_power * self.PE_DAC_num
				self.PE_ADC_write_power = 0 # Assume ADCs are idle in write process
				self.PE_digital_write_power = self.PE_DAC_num * self.input_demux_power
				self.PE_write_power = self.PE_xbar_write_power + self.PE_DAC_write_power + self.PE_ADC_write_power \
									 + self.PE_digital_write_power
			else:
				if write_column is None:
					write_column_group = self.xbar_column * self.PE_multiplex_xbar_num[1]
					write_column = self.num_group * [write_column_group]
				if write_row is None:
					write_row_group = self.xbar_row * self.PE_multiplex_xbar_num[0]
					write_row = self.num_group * [write_row_group]
				assert min(write_column) >= 0
				assert min(write_row) >= 0
				for index in range(self.num_group):
					multiple_time = min(math.ceil(write_row[index] / self.input_demux), write_column[index]) * \
									min(self.input_demux, write_row[index])
					self.calculate_xbar_write_power(1,1,num_cell[index],write_row[index],write_column[index])
					self.PE_xbar_write_power += self.xbar_write_power/multiple_time
					self.PE_DAC_write_power += math.ceil(write_row[index] / self.input_demux) * self.DAC_power
					self.PE_ADC_write_power += 0 # Assume ADCs are idle in write process
					self.PE_digital_write_power += math.ceil(write_row[index] / self.input_demux) * self.input_demux_power
				self.PE_write_power = self.PE_xbar_write_power + self.PE_DAC_write_power + self.PE_ADC_write_power \
									 + self.PE_digital_write_power

	def calculate_PE_read_energy(self):
		# unit: nJ
		self.PE_xbar_read_energy = self.PE_xbar_read_latency * self.PE_xbar_read_power
		self.PE_DAC_read_energy = self.PE_DAC_read_latency * self.PE_DAC_read_power
		self.PE_ADC_read_energy = self.PE_ADC_read_latency * self.PE_ADC_read_power
		self.PE_digital_read_energy = self.PE_digital_read_latency * self.PE_digital_read_power
		self.PE_read_energy = self.PE_xbar_read_energy + self.PE_DAC_read_energy + \
							  self.PE_ADC_read_energy + self.PE_digital_read_energy

	def calculate_PE_write_energy(self):
		# unit: nJ
		self.PE_xbar_write_energy = self.PE_xbar_write_latency * self.PE_xbar_write_power
		self.PE_DAC_write_energy = self.PE_DAC_write_latency * self.PE_DAC_write_power
		self.PE_ADC_write_energy = self.PE_ADC_write_latency * self.PE_ADC_write_power
		self.PE_digital_write_energy = self.PE_digital_write_latency * self.PE_digital_write_power
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
	_PE = ProcessElement(test_SimConfig_path,1)
	_PE.calculate_PE_area()
	_PE.calculate_PE_read_power()
	_PE.calculate_PE_write_power()
	_PE.calculate_PE_read_latency()
	_PE.calculate_PE_write_latency()
	_PE.calculate_PE_read_energy()
	_PE.calculate_PE_write_energy()
	_PE.PE_output()


if __name__ == '__main__':
	PE_test()