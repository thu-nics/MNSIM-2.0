#!/usr/bin/python
# -*-coding:utf-8-*-
import configparser as cp
import os
import math
from numpy import *
import numpy as np
from MNSIM.Hardware_Model.PE import ProcessElement
from MNSIM.Hardware_Model.Adder import adder
from MNSIM.Hardware_Model.Buffer import buffer
from MNSIM.Hardware_Model.ShiftReg import shiftreg
from MNSIM.Hardware_Model.Reg import reg
from MNSIM.Hardware_Model.JointModule import JointModule
from MNSIM.Hardware_Model.Pooling import Pooling
test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),"SimConfig.ini")
# Default SimConfig file path: MNSIM_Python/SimConfig.ini


class tile(ProcessElement):
	def __init__(self, SimConfig_path):
		# layer_num is a list with the size of 1xPE_num
		ProcessElement.__init__(self, SimConfig_path)
		tile_config = cp.ConfigParser()
		tile_config.read(SimConfig_path, encoding='UTF-8')
		self.tile_PE_num = list(map(int, tile_config.get('Tile level', 'PE_Num').split(',')))
		if self.tile_PE_num[0] == 0:
			self.tile_PE_num[0] = 4
			self.tile_PE_num[1] = 4
		assert self.tile_PE_num[0] > 0, "PE number in one PE < 0"
		assert self.tile_PE_num[1] > 0, "PE number in one PE < 0"
		self.tile_PE_total_num = self.tile_PE_num[0] * self.tile_PE_num[1]
		self.tile_simulation_level = int(tile_config.get('Algorithm Configuration', 'Simulation_Level'))
		self.tile_PE_list = []
		self.tile_PE_enable = []
		for i in range(self.tile_PE_num[0]):
			self.tile_PE_list.append([])
			self.tile_PE_enable.append([])
			for j in range(self.tile_PE_num[1]):
				__PE = ProcessElement(SimConfig_path)
				self.tile_PE_list[i].append(__PE)
				self.tile_PE_enable[i].append(0)
		self.layer_type = 'conv'
		self.tile_layer_num = 0
		self.tile_activation_precision = 0
		self.tile_sliding_times = 0
		self.tile_adder_num = 0
		self.tile_shiftreg_num = 0
		self.tile_jointmodule_num = 0
		self.tile_adder = adder(SimConfig_path)
		self.tile_shiftreg = shiftreg(SimConfig_path)
		self.tile_iReg = reg(SimConfig_path)
		self.tile_oReg = reg(SimConfig_path)
		self.tile_jointmodule = JointModule(SimConfig_path)
		self.tile_buffer = buffer(SimConfig_path)
		self.tile_pooling = Pooling(SimConfig_path)

		self.tile_utilization = 0
		self.num_occupied_PE = 0

		self.tile_area = 0
		self.tile_xbar_area = 0
		self.tile_ADC_area = 0
		self.tile_DAC_area = 0
		self.tile_digital_area = 0
		self.tile_adder_area = 0
		self.tile_shiftreg_area = 0
		self.tile_iReg_area = 0
		self.tile_oReg_area = 0
		self.tile_input_demux_area = 0
		self.tile_output_mux_area = 0
		self.tile_jointmodule_area = 0
		self.tile_pooling_area = 0
		self.tile_buffer_area = 0

		self.tile_read_power = 0
		self.tile_xbar_read_power = 0
		self.tile_ADC_read_power = 0
		self.tile_DAC_read_power = 0
		self.tile_digital_read_power = 0
		self.tile_adder_read_power = 0
		self.tile_shiftreg_read_power = 0
		self.tile_iReg_read_power = 0
		self.tile_oReg_read_power = 0
		self.tile_input_demux_read_power = 0
		self.tile_output_mux_read_power = 0
		self.tile_jointmodule_read_power = 0
		self.tile_pooling_read_power = 0
		self.tile_buffer_read_power = 0
		self.tile_buffer_r_read_power = 0
		self.tile_buffer_w_read_power = 0

		self.tile_write_power = 0
		self.tile_xbar_write_power = 0
		self.tile_ADC_write_power = 0
		self.tile_DAC_write_power = 0
		self.tile_digital_write_power = 0
		self.tile_adder_write_power = 0
		self.tile_shiftreg_write_power = 0
		self.tile_iReg_write_power = 0
		self.tile_input_demux_write_power = 0
		self.tile_output_mux_write_power = 0
		self.tile_jointmodule_write_power = 0

		self.tile_read_latency = 0
		self.tile_xbar_read_latency = 0
		self.tile_ADC_read_latency = 0
		self.tile_DAC_read_latency = 0
		self.tile_digital_read_latency = 0
		self.tile_adder_read_latency = 0
		self.tile_shiftreg_read_latency = 0
		self.tile_iReg_read_latency = 0
		self.tile_input_demux_read_latency = 0
		self.tile_output_mux_read_latency = 0
		self.tile_jointmodule_read_latency = 0
		# self.tile_layer_read_latency = {0:0}

		self.tile_write_latency = 0
		self.tile_xbar_write_latency = 0
		self.tile_ADC_write_latency = 0
		self.tile_DAC_write_latency = 0
		self.tile_digital_write_latency = 0
		self.tile_adder_write_latency = 0
		self.tile_shiftreg_write_latency = 0
		self.tile_iReg_write_latency = 0
		self.tile_input_demux_write_latency = 0
		self.tile_output_mux_write_latency = 0
		self.tile_jointmodule_write_latency = 0
		# self.tile_layer_write_latency = {0:0}

		self.tile_read_energy = 0
		self.tile_xbar_read_energy = 0
		self.tile_ADC_read_energy = 0
		self.tile_DAC_read_energy = 0
		self.tile_digital_read_energy = 0
		self.tile_adder_read_energy = 0
		self.tile_shiftreg_read_energy = 0
		self.tile_iReg_read_energy = 0
		self.tile_input_demux_read_energy = 0
		self.tile_output_mux_read_energy = 0
		self.tile_jointmodule_read_energy = 0

		self.tile_write_energy = 0
		self.tile_xbar_write_energy = 0
		self.tile_ADC_write_energy = 0
		self.tile_DAC_write_energy = 0
		self.tile_digital_write_energy = 0
		self.tile_adder_write_energy = 0
		self.tile_shiftreg_write_energy = 0
		self.tile_iReg_write_energy = 0
		self.tile_input_demux_write_energy = 0
		self.tile_output_mux_write_energy = 0
		self.tile_jointmodule_write_energy = 0
		# print("tile configuration is loaded")
		self.calculate_intra_PE_connection()


	def calculate_intra_PE_connection(self):
		# default configuration: H-tree structure
		index = self.tile_PE_total_num
		temp_num = 0
		while index/2 >= 1:
			temp_num += int(index/2) + index%2
			index = int(index/2)
		temp_num *= self.tile_PE_list[0][0].PE_ADC_num
		self.tile_adder_num = temp_num
		self.tile_shiftreg_num = temp_num
		self.tile_jointmodule_num = temp_num

	def update_tile_buf_size(self, SimConfig_path, default_buf_size = 16):
		self.tile_buffer = buffer(SimConfig_path=SimConfig_path, default_buf_size=default_buf_size)

	def calculate_tile_area(self, SimConfig_path=None, default_inbuf_size = 16, default_outbuf_size = 4):
		# unit: um^2
		self.tile_area = 0
		self.tile_xbar_area = 0
		self.tile_ADC_area = 0
		self.tile_DAC_area = 0
		self.tile_input_demux_area = 0
		self.tile_output_mux_area = 0
		self.tile_shiftreg_area = 0
		self.tile_iReg_area = 0
		self.tile_oReg_area = 0
		self.tile_adder_area = 0
		self.tile_buffer_area = 0
		self.tile_digital_area = 0
		self.tile_adder.calculate_adder_area()
		self.tile_shiftreg.calculate_shiftreg_area()
		self.tile_iReg.calculate_reg_area()
		self.tile_oReg.calculate_reg_area()
		self.tile_jointmodule.calculate_jointmodule_area()
		self.tile_buffer = buffer(SimConfig_path=SimConfig_path,buf_level=2,default_buf_size=default_outbuf_size)
		self.tile_buffer.calculate_buf_area()
		self.tile_pooling.calculate_Pooling_area()

		for i in range(self.tile_PE_num[0]):
			for j in range(self.tile_PE_num[1]):
				self.tile_PE_list[i][j].calculate_PE_area(SimConfig_path=SimConfig_path, default_inbuf_size = default_inbuf_size)
				self.tile_xbar_area += self.tile_PE_list[i][j].PE_xbar_area
				self.tile_ADC_area += self.tile_PE_list[i][j].PE_ADC_area
				self.tile_DAC_area += self.tile_PE_list[i][j].PE_DAC_area
				# self.tile_digital_area += self.tile_PE_list[i][j].PE_digital_area
				self.tile_input_demux_area += self.tile_PE_list[i][j].PE_input_demux_area
				self.tile_output_mux_area += self.tile_PE_list[i][j].PE_output_mux_area
				self.tile_shiftreg_area += self.tile_PE_list[i][j].PE_shiftreg_area
				self.tile_iReg_area += self.tile_PE_list[i][j].PE_iReg_area
				self.tile_oReg_area += self.tile_PE_list[i][j].PE_oReg_area
				self.tile_adder_area += self.tile_PE_list[i][j].PE_adder_area
				self.tile_buffer_area += self.tile_PE_list[i][j].PE_inbuf_area
		# self.tile_adder_area += self.tile_adder_num * self.tile_adder.adder_area
		# self.tile_shiftreg_area += self.tile_shiftreg_num * self.tile_shiftreg.shiftreg_area
		self.tile_jointmodule_area = self.tile_jointmodule_num * self.tile_jointmodule.jointmodule_area
		self.tile_digital_area = self.tile_input_demux_area + self.tile_output_mux_area + self.tile_adder_area \
								 + self.tile_shiftreg_area + self.tile_jointmodule_area + self.tile_iReg_area + self.tile_oReg_area
		self.tile_pooling_area = self.tile_pooling.Pooling_area
		self.tile_buffer_area += self.tile_buffer.buf_area
		self.tile_area = self.tile_xbar_area + self.tile_ADC_area + self.tile_DAC_area + self.tile_digital_area + self.tile_buffer_area+self.tile_pooling_area

	def calculate_tile_read_power_fast(self, max_column=0, max_row=0, max_PE=0, max_group=0, layer_type=None,
									   SimConfig_path=None, default_inbuf_size = 16, default_outbuf_size = 4):
		# max_column: maximum used column in one crossbar in this tile
		# max_row: maximum used row in one crossbar in this tile
		# max_PE: maximum used PE in this tile
		# max_group: maximum used groups in one PE
		# unit: W
		# coarse but fast estimation
		self.tile_read_power = 0
		self.tile_xbar_read_power = 0
		self.tile_ADC_read_power = 0
		self.tile_DAC_read_power = 0
		self.tile_digital_read_power = 0
		self.tile_adder_read_power = 0
		self.tile_shiftreg_read_power = 0
		self.tile_iReg_read_power = 0
		self.tile_oReg_read_power = 0
		self.tile_input_demux_read_power = 0
		self.tile_output_mux_read_power = 0
		self.tile_jointmodule_read_power = 0
		self.tile_pooling_read_power = 0
		self.tile_buffer_read_power = 0
		self.tile_buffer_r_read_power = 0
		self.tile_buffer_w_read_power = 0
		self.tile_buffer = buffer(SimConfig_path=SimConfig_path, buf_level=2, default_buf_size=default_outbuf_size)
		if layer_type == 'pooling':
			self.tile_pooling.calculate_Pooling_power()
			self.tile_pooling_read_power = self.tile_pooling.Pooling_power
		elif layer_type == 'conv' or layer_type  == 'fc':
			self.calculate_PE_read_power_fast(max_column=max_column, max_row=max_row, max_group=max_group,
											  SimConfig_path=SimConfig_path, default_inbuf_size = default_inbuf_size)
			self.tile_xbar_read_power = max_PE * self.PE_xbar_read_power
			self.tile_ADC_read_power = max_PE * self.PE_ADC_read_power
			self.tile_DAC_read_power = max_PE * self.PE_DAC_read_power
			self.tile_adder_read_power = max_PE * self.PE_adder_read_power
			self.tile_shiftreg_read_power = max_PE * self.PE_shiftreg_read_power
			self.tile_iReg_read_power = max_PE * self.PE_iReg_read_power
			self.tile_oReg_read_power = max_PE * self.PE_oReg_read_power
			self.tile_input_demux_read_power = max_PE * self.input_demux_read_power
			self.tile_output_mux_read_power = max_PE * self.output_mux_read_power
			self.tile_jointmodule_read_power = (max_PE-1)*math.ceil(max_column/self.output_mux)*self.tile_jointmodule.jointmodule_power
			self.tile_digital_read_power = self.tile_adder_read_power+self.tile_shiftreg_read_power+\
											self.tile_input_demux_read_power+self.tile_output_mux_read_power+self.tile_jointmodule_read_power
			self.tile_buffer_r_read_power = max_PE * self.PE_inbuf_read_rpower
			self.tile_buffer_w_read_power = max_PE * self.PE_inbuf_read_wpower
		self.tile_buffer.calculate_buf_read_power()
		self.tile_buffer.calculate_buf_write_power()
		self.tile_buffer_r_read_power += self.tile_buffer.buf_rpower * 1e-3
		self.tile_buffer_w_read_power += self.tile_buffer.buf_wpower * 1e-3
		self.tile_buffer_read_power = self.tile_buffer_r_read_power + self.tile_buffer_w_read_power
		self.tile_digital_read_power = self.tile_adder_read_power+self.tile_shiftreg_read_power+self.tile_iReg_read_power+self.tile_oReg_read_power+\
									   self.tile_input_demux_read_power+self.tile_output_mux_read_power+self.tile_jointmodule_read_power
		self.tile_read_power = self.tile_xbar_read_power+self.tile_ADC_read_power+self.tile_DAC_read_power+\
							   self.tile_digital_read_power+self.tile_pooling_read_power+self.tile_buffer_read_power

	def tile_read_config(self, layer_num = 0, activation_precision = 0, sliding_times = 0,
						 read_row = None, read_column = None, read_matrix = None, read_vector = None):
		# read_row and read_column are 2D lists with the size of (#occupied_PE x #occupied groups)
		# read_matrix is a 3D list of matrices, with the size of (#occupied_PE x #occupied groups x Xbar_Polarity)
		# read_vector is a 2D list of vectors, with the size of (#occupied_PE x #occupied groups)
		self.tile_layer_num = layer_num
		self.tile_activation_precision = activation_precision
		self.tile_sliding_times = sliding_times
		self.tile_utilization = 0
		self.num_occupied_PE = 0
		if self.tile_simulation_level == 0:
			if (read_row is None) or (read_column is None):
				self.num_occupied_group = self.tile_PE_total_num
				for i in range(self.tile_PE_num[0]):
					for j in range(self.tile_PE_num[1]):
						# temp_index = i*self.tile_PE_num[0] + self.tile_PE_num[1]
						self.tile_PE_list[i][j].PE_read_config()
						self.tile_PE_enable[i][j] = 1
						self.tile_utilization += self.tile_PE_list[i][j].PE_utilization
			else:
				assert len(read_row) == len(read_column), "read_row and read_column must be equal in length"
				self.num_occupied_PE = len(read_row)
				assert self.num_occupied_PE <= self.tile_PE_total_num, "The length of read_row exceeds the PE number in one tile"
				for i in range(self.tile_PE_num[0]):
					for j in range(self.tile_PE_num[1]):
						temp_index = i * self.tile_PE_num[0] + j
						if temp_index < self.num_occupied_PE:
							self.tile_PE_list[i][j].PE_read_config(read_row = read_row[temp_index],
																   read_column = read_column[temp_index])
							self.tile_PE_enable[i][j] = 1
							self.tile_utilization += self.tile_PE_list[i][j].PE_utilization
						else:
							self.tile_PE_enable[i][j] = 0
		else:
			if read_matrix is None:
				self.num_occupied_group = self.tile_PE_total_num
				for i in range(self.tile_PE_num[0]):
					for j in range(self.tile_PE_num[1]):
						self.tile_PE_list[i][j].PE_read_config()
						self.tile_PE_enable[i][j] = 1
						self.tile_utilization += self.tile_PE_list[i][j].PE_utilization
			else:
				if read_vector is None:
					self.num_occupied_PE = len(read_matrix)
					assert self.num_occupied_PE <= self.tile_PE_total_num, "The number of read_matrix exceeds the PE number in one tile"
					for i in range(self.tile_PE_num[0]):
						for j in range(self.tile_PE_num[1]):
							temp_index = i * self.tile_PE_num[0] + j
							if temp_index < self.num_occupied_PE:
								self.tile_PE_list[i][j].PE_read_config(read_matrix = read_matrix[temp_index])
								self.tile_PE_enable[i][j] = 1
								self.tile_utilization += self.tile_PE_list[i][j].PE_utilization
							else:
								self.tile_PE_enable[i][j] = 0
				else:
					assert len(read_matrix) == len(read_vector), "The number of read_matrix and read_vector must be equal"
					self.num_occupied_PE = len(read_matrix)
					for i in range(self.tile_PE_num[0]):
						for j in range(self.tile_PE_num[1]):
							temp_index = i * self.tile_PE_num[0] + j
							if temp_index < self.num_occupied_PE:
								self.tile_PE_list[i][j].PE_read_config(read_matrix = read_matrix[temp_index],
																	   read_vector = read_vector[temp_index])
								self.tile_PE_enable[i][j] = 1
								self.tile_utilization += self.tile_PE_list[i][j].PE_utilization
							else:
								self.tile_PE_enable[i][j] = 0
		self.tile_utilization /= self.tile_PE_total_num

	'''def tile_write_config(self, write_row = None, write_column = None, write_matrix = None, write_vector = None):
		# write_row and write_column are 2D lists with the size of (#occupied_PE x #occupied groups)
		# write_matrix is a 3D list of matrices, with the size of (#occupied_PE x #occupied groups x Xbar_Polarity)
		# write_vector is a 2D list of vectors, with the size of (#occupied_PE x #occupied groups)
		self.tile_utilization = 0
		if self.tile_simulation_level == 0:
			if (write_row is None) or (write_column is None):
				self.num_occupied_PE = self.tile_PE_total_num
				for i in range(self.tile_PE_num[0]):
					for j in range(self.tile_PE_num[1]):
						# temp_index = i*self.tile_PE_num[0] + self.tile_PE_num[1]
						self.tile_PE_list[i][j].PE_write_config()
						self.tile_PE_enable[i][j] = 1
						self.tile_utilization += self.tile_PE_list[i][j].PE_utilization
			else:
				assert len(write_row) == len(write_column), "write_row and write_column must be equal in length"
				self.num_occupied_PE = len(write_row)
				assert self.num_occupied_PE <= self.tile_PE_total_num, "The length of write_row exceeds the PE number in one tile"
				for i in range(self.tile_PE_num[0]):
					for j in range(self.tile_PE_num[1]):
						temp_index = i * self.tile_PE_num[0] + self.tile_PE_num[1]
						if temp_index < self.num_occupied_PE:
							self.tile_PE_list[i][j].PE_write_config(write_row = write_row[temp_index],
																	write_column = write_column[temp_index])
							self.tile_PE_enable[i][j] = 1
							self.tile_utilization += self.tile_PE_list[i][j].PE_utilization
						else:
							self.tile_PE_enable[i][j] = 0
		else:
			if write_matrix is None:
				self.num_occupied_PE = self.tile_PE_total_num
				for i in range(self.tile_PE_num[0]):
					for j in range(self.tile_PE_num[1]):
						self.tile_PE_list[i][j].PE_write_config()
						self.tile_PE_enable[i][j] = 1
						self.tile_utilization += self.tile_PE_list[i][j].PE_utilization
			else:
				if write_vector is None:
					self.num_occupied_PE = len(write_matrix)
					assert self.num_occupied_PE <= self.tile_PE_total_num, "The number of write_matrix exceeds the PE number in one tile"
					for i in range(self.tile_PE_num[0]):
						for j in range(self.tile_PE_num[1]):
							temp_index = i * self.tile_PE_num[0] + self.tile_PE_num[1]
							if temp_index < self.num_occupied_PE:
								self.tile_PE_list[i][j].PE_write_config(write_matrix = write_matrix[temp_index])
								self.tile_PE_enable[i][j] = 1
								self.tile_utilization += self.tile_PE_list[i][j].PE_utilization
							else:
								self.tile_PE_enable[i][j] = 0
				else:
					assert len(write_matrix) == len(write_vector), "The number of write_matrix and write_vector must be equal"
					self.num_occupied_PE = len(write_matrix)
					for i in range(self.tile_PE_num[0]):
						for j in range(self.tile_PE_num[1]):
							temp_index = i * self.tile_PE_num[0] + self.tile_PE_num[1]
							if temp_index < self.num_occupied_PE:
								self.tile_PE_list[i][j].PE_write_config(write_matrix = write_matrix[temp_index],
																		write_vector = write_vector[temp_index])
								self.tile_PE_enable[i][j] = 1
								self.tile_utilization += self.tile_PE_list[i][j].PE_utilization
							else:
								self.tile_PE_enable[i][j] = 0
		self.tile_utilization /= self.tile_PE_total_num'''

	'''def calculate_tile_read_latency(self):
		# Notice: before calculating latency, tile_read_config must be executed
		# unit: ns
		self.tile_read_latency = 0
		self.tile_xbar_read_latency = 0
		self.tile_ADC_read_latency = 0
		self.tile_DAC_read_latency = 0
		self.tile_digital_read_latency = 0
		self.tile_adder_read_latency = 0
		self.tile_shiftreg_read_latency = 0
		self.tile_input_demux_read_latency = 0
		self.tile_output_mux_read_latency = 0
		if self.num_occupied_PE != 0:
			temp_latency = 0
			for i in range(self.tile_PE_num[0]):
				for j in range(self.tile_PE_num[1]):
					if self.tile_PE_enable[i][j] == 1:
						self.tile_PE_list[i][j].calculate_PE_read_latency()
						if self.tile_PE_list[i][j].PE_read_latency > temp_latency:
							temp_latency = self.tile_PE_list[i][j].PE_read_latency
							self.tile_xbar_read_latency = self.tile_PE_list[i][j].PE_xbar_read_latency
							self.tile_ADC_read_latency = self.tile_PE_list[i][j].PE_ADC_read_latency
							self.tile_DAC_read_latency = self.tile_PE_list[i][j].PE_DAC_read_latency
							self.tile_digital_read_latency = self.tile_PE_list[i][j].PE_digital_read_latency
							self.tile_shiftreg_read_latency = self.tile_PE_list[i][j].PE_shiftreg_read_latency
							self.tile_adder_read_latency = self.tile_PE_list[i][j].PE_adder_read_latency
							self.tile_input_demux_read_latency = self.tile_PE_list[i][j].input_demux_read_latency
							self.tile_output_mux_read_latency = self.tile_PE_list[i][j].output_mux_read_latency
			level = math.ceil(math.log2(self.num_occupied_PE))
			multiple_time = math.ceil(self.tile_activation_precision / self.tile_PE_list[0][0].DAC_precision) \
							* self.tile_sliding_times
			# self.tile_shiftreg_read_latency = multiple_time * (self.tile_shiftreg_read_latency + self.tile_shiftreg.shiftreg_latency)
			# self.tile_adder_read_latency = multiple_time * (level * self.tile_adder.adder_latency + self.tile_adder_read_latency)
			self.tile_xbar_read_latency *= multiple_time
			self.tile_ADC_read_latency *= multiple_time
			self.tile_DAC_read_latency *= multiple_time
			self.tile_input_demux_read_latency *= multiple_time
			self.tile_output_mux_read_latency *= multiple_time
			self.tile_jointmodule_read_latency = self.tile_sliding_times * level * self.tile_jointmodule.jointmodule_latency
			self.tile_digital_read_latency = multiple_time * (self.tile_digital_read_latency
															  + self.tile_shiftreg.shiftreg_latency + level * self.tile_adder.adder_latency)\
											 + self.tile_jointmodule_read_latency
			self.tile_read_latency = self.tile_xbar_read_latency + self.tile_ADC_read_latency \
									 + self.tile_DAC_read_latency + self.tile_digital_read_latency
	
	def calculate_tile_write_latency(self):
		# Notice: before calculating latency, tile_write_config must be executed
		# unit: ns
		self.tile_write_latency = 0
		self.tile_xbar_write_latency = 0
		self.tile_ADC_write_latency = 0
		self.tile_DAC_write_latency = 0
		self.tile_digital_write_latency = 0
		self.tile_adder_write_latency = 0
		self.tile_shiftreg_write_latency = 0
		self.tile_input_demux_write_latency = 0
		self.tile_output_mux_write_latency = 0
		if self.num_occupied_PE != 0:
			temp_latency = 0
			for i in range(self.tile_PE_num[0]):
				for j in range(self.tile_PE_num[1]):
					if self.tile_PE_enable[i][j] == 1:
						self.tile_PE_list[i][j].calculate_PE_write_latency()
						if self.tile_PE_list[i][j].PE_write_latency > temp_latency:
							temp_latency = self.tile_PE_list[i][j].PE_write_latency
							self.tile_xbar_write_latency = self.tile_PE_list[i][j].PE_xbar_write_latency
							self.tile_ADC_write_latency = self.tile_PE_list[i][j].PE_ADC_write_latency
							self.tile_DAC_write_latency = self.tile_PE_list[i][j].PE_DAC_write_latency
							self.tile_digital_write_latency = self.tile_PE_list[i][j].PE_digital_write_latency
							self.tile_input_demux_write_latency = self.tile_PE_list[i][j].input_demux_write_latency
							self.tile_output_mux_write_latency = 0
							self.tile_shiftreg_write_latency = 0
							self.tile_adder_write_latency = 0
							self.tile_jointmodule_write_latency = 0
			self.tile_write_latency = self.tile_xbar_write_latency + self.tile_ADC_write_latency \
									  + self.tile_DAC_write_latency + self.tile_digital_write_latency'''

	def calculate_tile_read_power(self):
		# unit: W
		# Notice: before calculating power, tile_read_config must be executed
		self.tile_read_power = 0
		self.tile_xbar_read_power = 0
		self.tile_ADC_read_power = 0
		self.tile_DAC_read_power = 0
		self.tile_digital_read_power = 0
		self.tile_adder_read_power = 0
		self.tile_shiftreg_read_power = 0
		self.tile_iReg_read_power = 0
		self.tile_input_demux_read_power = 0
		self.tile_output_mux_read_power = 0
		self.tile_buffer_read_power = 0
		self.tile_buffer_r_read_power = 0
		self.tile_buffer_w_read_power = 0

		max_occupied_column = 0
		if self.num_occupied_PE != 0:
			for i in range(self.tile_PE_num[0]):
				for j in range(self.tile_PE_num[1]):
					if self.tile_PE_enable[i][j] == 1:
						self.tile_PE_list[i][j].calculate_PE_read_power()
						self.tile_xbar_read_power += self.tile_PE_list[i][j].PE_xbar_read_power
						self.tile_ADC_read_power += self.tile_PE_list[i][j].PE_ADC_read_power
						self.tile_DAC_read_power += self.tile_PE_list[i][j].PE_DAC_read_power
						self.tile_adder_read_power += self.tile_PE_list[i][j].PE_adder_read_power
						self.tile_shiftreg_read_power += self.tile_PE_list[i][j].PE_shiftreg_read_power
						self.tile_iReg_read_power += self.tile_PE_list[i][j].PE_iReg_read_power
						self.tile_input_demux_read_power += self.tile_PE_list[i][j].input_demux_read_power
						self.tile_output_mux_read_power += self.tile_PE_list[i][j].output_mux_read_power
						# self.tile_digital_read_power += self.tile_PE_list[i][j].PE_digital_read_power
						if self.tile_PE_list[i][j].PE_max_occupied_column > max_occupied_column:
							max_occupied_column = self.tile_PE_list[i][j].PE_max_occupied_column
			# TODO: more accurate estimation of adder/shiftreg number
			max_occupied_column = min(max_occupied_column, self.tile_PE_list[0][0].PE_ADC_num)
			# self.tile_adder_read_power = (self.num_occupied_PE - 1) * max_occupied_column * self.tile_adder.adder_power
			# self.tile_shiftreg_read_power = (self.num_occupied_PE - 1) * max_occupied_column * self.tile_shiftreg.shiftreg_power
			self.tile_jointmodule_read_power = (self.num_occupied_PE - 1) * math.ceil(max_occupied_column/self.output_mux) * self.tile_jointmodule.jointmodule_power

			self.tile_digital_read_power = self.tile_adder_read_power + self.tile_shiftreg_read_power + self.tile_iReg_read_power \
										   + self.tile_input_demux_read_power + self.tile_output_mux_read_power + self.tile_jointmodule_read_power
			self.tile_buffer.calculate_buf_read_power()
			self.tile_buffer.calculate_buf_write_power()
			self.tile_buffer_r_read_power = self.tile_buffer.buf_rpower * 1e-3
			self.tile_buffer_w_read_power = self.tile_buffer.buf_wpower * 1e-3
			self.tile_buffer_read_power = self.tile_buffer_r_read_power + self.tile_buffer_w_read_power
			self.tile_read_power = self.tile_xbar_read_power + self.tile_ADC_read_power + self.tile_DAC_read_power \
								   + self.tile_digital_read_power \
								   + self.tile_buffer_read_power

	'''def calculate_tile_write_power(self):
		# unit: W
		# Notice: before calculating power, tile_write_config must be executed
		self.tile_write_power = 0
		self.tile_xbar_write_power = 0
		self.tile_ADC_write_power = 0
		self.tile_DAC_write_power = 0
		self.tile_digital_write_power = 0
		self.tile_adder_write_power = 0
		self.tile_shiftreg_write_power = 0
		self.tile_input_demux_write_power = 0
		self.tile_output_mux_write_power = 0
		if self.num_occupied_PE != 0:
			for i in range(self.tile_PE_num[0]):
				for j in range(self.tile_PE_num[1]):
					if self.tile_PE_enable[i][j] == 1:
						self.tile_PE_list[i][j].calculate_PE_write_power()
						self.tile_xbar_write_power += self.tile_PE_list[i][j].PE_xbar_read_power
						self.tile_ADC_write_power += self.tile_PE_list[i][j].PE_ADC_write_power
						self.tile_DAC_write_power += self.tile_PE_list[i][j].PE_DAC_write_power
						self.tile_digital_write_power += self.tile_PE_list[i][j].PE_digital_write_power
						self.tile_adder_write_power += self.tile_PE_list[i][j].PE_adder_write_power
						self.tile_shiftreg_write_power += self.tile_PE_list[i][j].PE_shiftreg_write_power
						self.tile_input_demux_write_power += self.tile_PE_list[i][j].input_demux_write_power
						self.tile_output_mux_write_power += self.tile_PE_list[i][j].output_mux_write_power
			self.tile_write_power = self.tile_xbar_write_power + self.tile_ADC_write_power + self.tile_DAC_write_power \
									+ self.tile_digital_write_power \
									+ (self.buffer.dynamic_buf_wpower + self.buffer.leakage_power) * 1e-3

	def calculate_tile_read_energy(self):
		# unit: nJ
		# Notice: before calculating energy, tile_read_config and calculate_tile_read_power must be executed
		self.tile_read_energy = 0
		self.tile_xbar_read_energy = 0
		self.tile_ADC_read_energy = 0
		self.tile_DAC_read_energy = 0
		self.tile_digital_read_energy = 0
		self.tile_adder_read_energy = 0
		self.tile_shiftreg_read_energy = 0
		self.tile_input_demux_read_energy = 0
		self.tile_output_mux_read_energy = 0
		if self.num_occupied_PE != 0:
			self.tile_xbar_read_energy = self.tile_xbar_read_power * self.tile_xbar_read_latency
			self.tile_ADC_read_energy = self.tile_ADC_read_power * self.tile_ADC_read_latency
			self.tile_DAC_read_energy = self.tile_DAC_read_power * self.tile_DAC_read_latency
			#TODO: correct the adder and shiftreg energy calculation
			self.tile_adder_read_energy = self.tile_adder_read_power * self.tile_adder_read_latency
			self.tile_shiftreg_read_energy = self.tile_shiftreg_read_power * self.tile_shiftreg_read_latency
			self.tile_jointmodule_read_energy = self.tile_jointmodule_read_power * self.tile_jointmodule_read_latency
			self.tile_input_demux_read_energy = self.tile_input_demux_read_power * self.tile_input_demux_read_latency
			self.tile_output_mux_read_energy = self.tile_output_mux_read_power * self.tile_output_mux_read_latency
			self.tile_digital_read_energy = self.tile_adder_read_energy + self.tile_shiftreg_read_energy \
											+ self.tile_input_demux_read_energy + self.tile_output_mux_read_energy + self.tile_jointmodule_read_energy
			self.tile_read_energy = self.tile_xbar_read_energy + self.tile_ADC_read_energy \
									+ self.tile_DAC_read_energy + self.tile_digital_read_energy + self.tile_buffer.buf_renergy

	def calculate_tile_write_energy(self):
		# unit: nJ
		# Notice: before calculating energy, tile_write_config and calculate_tile_write_power must be executed
		self.tile_write_energy = 0
		self.tile_xbar_write_energy = 0
		self.tile_ADC_write_energy = 0
		self.tile_DAC_write_energy = 0
		self.tile_digital_write_energy = 0
		self.tile_adder_write_energy = 0
		self.tile_shiftreg_write_energy = 0
		self.tile_input_demux_write_energy = 0
		self.tile_output_mux_write_energy = 0
		if self.num_occupied_PE != 0:
			self.tile_xbar_write_energy = self.tile_xbar_write_power * self.tile_xbar_write_latency
			self.tile_ADC_write_energy = self.tile_ADC_write_power * self.tile_ADC_write_latency
			self.tile_DAC_write_energy = self.tile_DAC_write_power * self.tile_DAC_write_latency
			# TODO: correct the adder and shiftreg energy calculation
			self.tile_adder_write_energy = self.tile_adder_write_power * self.tile_adder_write_latency
			self.tile_shiftreg_write_energy = self.tile_shiftreg_write_power * self.tile_shiftreg_write_latency
			self.tile_input_demux_write_energy = self.tile_input_demux_write_power * self.tile_input_demux_write_latency
			self.tile_output_mux_write_energy = self.tile_output_mux_write_power * self.tile_output_mux_write_latency
			self.tile_digital_write_energy = self.tile_adder_write_energy + self.tile_shiftreg_write_energy \
											 + self.tile_input_demux_write_energy + self.tile_output_mux_write_energy
			self.tile_write_energy = self.tile_xbar_write_energy + self.tile_ADC_write_energy \
									 + self.tile_DAC_write_energy + self.tile_digital_write_energy + self.tile_buffer.buf_wenergy'''

	def tile_output(self):
		self.tile_PE_list[0][0].PE_output()
		print("-------------------------tile Configurations-------------------------")
		print("total PE number in one tile:", self.tile_PE_total_num, "(", self.tile_PE_num, ")")
		print("total adder number in one tile:", self.tile_adder_num)
		# print("			the level of adders is:", self.tile_adder_level)
		print("total shift-reg number in one tile:", self.tile_shiftreg_num)
		# print("			the level of shift-reg is:", self.tile_shiftreg_level)
		print("----------------------tile Area Simulation Results-------------------")
		print("tile area:", self.tile_area, "um^2")
		print("			crossbar area:", self.tile_xbar_area, "um^2")
		print("			DAC area:", self.tile_DAC_area, "um^2")
		print("			ADC area:", self.tile_ADC_area, "um^2")
		print("			digital part area:", self.tile_digital_area, "um^2")
		print("				|---adder area:", self.tile_adder_area, "um^2")
		print("				|---shift-reg area:", self.tile_shiftreg_area, "um^2")
		print("				|---input_demux area:", self.tile_input_demux_area, "um^2")
		print("				|---output_mux area:", self.tile_output_mux_area, "um^2")
		print("				|---JointModule area:", self.tile_jointmodule_area, "um^2")

		print("--------------------tile Latency Simulation Results------------------")
		print("tile read latency:", self.tile_read_latency, "ns")
		print("			crossbar read latency:", self.tile_xbar_read_latency, "ns")
		print("			DAC read latency:", self.tile_DAC_read_latency, "ns")
		print("			ADC read latency:", self.tile_ADC_read_latency, "ns")
		print("			digital part read latency:", self.tile_digital_read_latency, "ns")
		print("				|---adder read latency:", self.tile_adder_read_latency, "ns")
		print("				|---shift-reg read latency:", self.tile_shiftreg_read_latency, "ns")
		print("				|---input demux read latency:", self.tile_input_demux_read_latency, "ns")
		print("				|---output mux read latency:", self.tile_output_mux_read_latency, "ns")
		print("				|---JointModule read latency:", self.tile_jointmodule_read_latency, "ns")
		print("tile write latency:", self.tile_write_latency, "ns")
		print("			crossbar write latency:", self.tile_xbar_write_latency, "ns")
		print("			DAC write latency:", self.tile_DAC_write_latency, "ns")
		print("			ADC write latency:", self.tile_ADC_write_latency, "ns")
		print("			digital part write latency:", self.tile_digital_write_latency, "ns")
		print("				|---adder write latency:", self.tile_adder_write_latency, "ns")
		print("				|---shift-reg write latency:", self.tile_shiftreg_write_latency, "ns")
		print("				|---input demux write latency:", self.tile_input_demux_write_latency, "ns")
		print("				|---output mux write latency:", self.tile_output_mux_write_latency, "ns")
		print("				|---JointModule write latency:", self.tile_jointmodule_write_latency, "ns")
		print("--------------------tile Power Simulation Results-------------------")
		print("tile read power:", self.tile_read_power, "W")
		print("			crossbar read power:", self.tile_xbar_read_power, "W")
		print("			DAC read power:", self.tile_DAC_read_power, "W")
		print("			ADC read power:", self.tile_ADC_read_power, "W")
		print("			digital part read power:", self.tile_digital_read_power, "W")
		print("				|---adder read power:", self.tile_adder_read_power, "W")
		print("				|---shift-reg read power:", self.tile_shiftreg_read_power, "W")
		print("				|---input demux read power:", self.tile_input_demux_read_power, "W")
		print("				|---output mux read power:", self.tile_output_mux_read_power, "W")
		print("				|---JointModule read power:", self.tile_jointmodule_read_power, "W")
		print("			buffer read power:", (self.buffer.dynamic_buf_rpower + self.buffer.leakage_power) * 1e-3, "W")
		print("tile write power:", self.tile_write_power, "W")
		print("			crossbar write power:", self.tile_xbar_write_power, "W")
		print("			DAC write power:", self.tile_DAC_write_power, "W")
		print("			ADC write power:", self.tile_ADC_write_power, "W")
		print("			digital part write power:", self.tile_digital_write_power, "W")
		print("				|---adder write power:", self.tile_adder_write_power, "W")
		print("				|---shift-reg write power:", self.tile_shiftreg_write_power, "W")
		print("				|---input demux write power:", self.tile_input_demux_write_power, "W")
		print("				|---output mux write power:", self.tile_output_mux_write_power, "W")
		print("				|---JointModule write power:", self.tile_jointmodule_write_power, "W")
		print("			buffer write power:", (self.buffer.dynamic_buf_wpower + self.buffer.leakage_power) * 1e-3, "W")
		print("------------------Energy Simulation Results----------------------")
		print("tile read energy:", self.tile_read_energy, "nJ")
		print("			crossbar read energy:", self.tile_xbar_read_energy, "nJ")
		print("			DAC read energy:", self.tile_DAC_read_energy, "nJ")
		print("			ADC read energy:", self.tile_ADC_read_energy, "nJ")
		print("			digital part read energy:", self.tile_digital_read_energy, "nJ")
		print("				|---adder read energy:", self.tile_adder_read_energy, "nJ")
		print("				|---shift-reg read energy:", self.tile_shiftreg_read_energy, "nJ")
		print("				|---input demux read energy:", self.tile_input_demux_read_energy, "nJ")
		print("				|---output mux read energy:", self.tile_output_mux_read_energy, "nJ")
		print("				|---JointModule read energy:", self.tile_jointmodule_read_energy, "nJ")
		print("tile write energy:", self.tile_write_energy, "nJ")
		print("			crossbar write energy:", self.tile_xbar_write_energy, "nJ")
		print("			DAC write energy:", self.tile_DAC_write_energy, "nJ")
		print("			ADC write energy:", self.tile_ADC_write_energy, "nJ")
		print("			digital part write energy:", self.tile_digital_write_energy, "nJ")
		print("				|---adder write energy:", self.tile_adder_write_energy, "nJ")
		print("				|---shift-reg write energy:", self.tile_shiftreg_write_energy, "nJ")
		print("				|---input demux write energy:", self.tile_input_demux_write_energy, "nJ")
		print("				|---output mux write energy:", self.tile_output_mux_write_energy, "nJ")
		print("				|---JointModule write energy:", self.tile_jointmodule_write_energy, "nJ")
		print("-----------------------------------------------------------------")
	
def tile_test():
	print("load file:", test_SimConfig_path)
	_tile = tile(test_SimConfig_path)
	print(_tile.xbar_column)
	_tile0 = _tile
	_tile0.tile_read_config()
	_tile0.tile_write_config()
	_tile0.calculate_tile_area()
	_tile0.calculate_tile_read_latency()
	_tile0.calculate_tile_write_latency()
	_tile0.calculate_tile_read_power()
	_tile0.calculate_tile_write_power()
	_tile0.calculate_tile_read_energy()
	_tile0.calculate_tile_write_energy()
	_tile0.tile_output()


if __name__ == '__main__':
	tile_test()