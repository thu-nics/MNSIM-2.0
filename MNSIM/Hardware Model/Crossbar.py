#!/usr/bin/python
#-*-coding:utf-8-*-
import configparser as cp
import os
import math
from Device import device
test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),"SimConfig.ini")
# Default SimConfig file path: MNSIM_Python/SimConfig.ini


class crossbar(device):
	def __init__(self, SimConfig_path):
		device.__init__(self,SimConfig_path)
		xbar_config = cp.ConfigParser()
		xbar_config.read(SimConfig_path)
		self.xbar_size = list(map(int, xbar_config.get('Crossbar level', 'Xbar_Size').split(',')))
		self.xbar_row = int(self.xbar_size[0])
		self.xbar_column = int(self.xbar_size[1])
		self.cell_type = xbar_config.get('Crossbar level', 'Cell_Type')
		self.transistor_tech = int(xbar_config.get('Crossbar level', 'Transistor_Tech'))
		self.wire_resistance = float(xbar_config.get('Crossbar level', 'Wire_Resistance'))
		self.wire_capacity = float(xbar_config.get('Crossbar level', 'Wire_Capacity'))
		self.xbar_area = 0
		self.xbar_read_power = 0
		self.xbar_read_latency = 0
		self.xbar_write_power = 0
		self.xbar_write_latency = 0
		self.xbar_read_energy = 0
		self.xbar_write_energy = 0
		self.xbar_num_write_row = 0
		self.xbar_num_write_column = 0
		self.xbar_num_write_cell = len(self.device_resistance) * [0]
		self.xbar_num_read_row = 0
		self.xbar_num_read_column = 0
		self.xbar_num_read_cell = len(self.device_resistance) * [0]
		self.xbar_utilization = 0

		# print("Crossbar configuration is loaded")
		# self.calculate_xbar_area()
		# self.calculate_xbar_read_latency()
		# self.calculate_xbar_write_latency()
		# self.calculate_xbar_read_power()
		# self.calculate_xbar_write_power()
		# self.calculate_xbar_read_energy()
		# self.calculate_xbar_write_energy()

	def set_xbar_write(self, num_write_column = None, num_write_row = None, num_cell = None):
		# num_cell: a list of number of cells in each resistance state in one row
		if num_write_column is not None:
			assert num_write_column >= 0
			self.xbar_num_write_column = num_write_column
		if num_write_row is not None:
			assert num_write_row >= 0
			self.xbar_num_write_row = num_write_row
		if num_cell is not None:
			assert min(num_cell) >= 0
			self.xbar_num_write_cell = num_cell

	def set_xbar_read(self, num_read_column = None, num_read_row = None, num_cell = None):
		# num_cell: a list of number of cells in each resistance state in one crossbar
		if num_read_column is not None:
			assert num_read_column >= 0
			self.xbar_num_read_column = num_read_column
		if num_read_row is not None:
			assert num_read_row >= 0
			self.xbar_num_read_row = num_read_row
		if num_cell is not None:
			assert min(num_cell) >= 0
			self.xbar_num_read_cell = num_cell

	def calculate_xbar_area(self):
		WL_ratio = 3
			#WL_ratio is the technology parameter W/L of the transistor
		area_factor = 4
		#Area unit: um^2
		if self.cell_type[0] == '0':
			self.xbar_area = area_factor * self.xbar_row * self.xbar_column * self.device_tech * self.device_tech / 1e6
		else:
			self.xbar_area = 3 * (WL_ratio + 1) * self.xbar_row * self.xbar_column * self.transistor_tech * self.transistor_tech / 1e6

	def calculate_wire_resistance(self):
		#unit: ohm
		if self.wire_resistance < 0:
			self.wire_resistance = 1
				#TODO: Update the wire resistance calculation according to different technology sizes

	def calculate_wire_capacity(self):
		#unit: fF
		if self.wire_capacity < 0:
			self.wire_capacity = 1
				#TODO: Update the wire capacity calculation according to different technology sizes

	def calculate_xbar_read_latency(self):
		self.calculate_wire_resistance()
		self.calculate_wire_capacity()
		wire_latency = 0.5 * self.wire_resistance * self.wire_capacity * 1e3
			#unit: ns
			#TODO: Update the calculation formula considering the branches
		self.xbar_read_latency = self.device_read_latency + wire_latency

	def calculate_xbar_write_latency(self):
		self.xbar_write_latency	= self.device_write_latency * self.xbar_num_write_row
		# self.xbar_write_latency = self.device_write_latency * min(math.ceil(num_write_row/num_multi_row), num_write_column)\
		#                           * min(num_multi_row, num_write_row)
			#unit: ns
			#Assuming that the write operation of cells in one row can be performed concurrently

	def calculate_xbar_read_power(self, cal_mode = 0):
		#unit: W
		#cal_mode: 0: simple estimation, 1: detailed simulation; utilization: utilization rate of crossbar (<=1); num_cell: a list of number of cells in each resistance state
		self.xbar_read_power = 0
		self.xbar_utilization = self.xbar_num_read_row * self.xbar_num_read_column / (self.xbar_row*self.xbar_column)
		if cal_mode == 0:
			assert self.xbar_utilization <= 1
			self.calculate_device_read_power()
			self.xbar_read_power = self.xbar_num_read_row * self.xbar_num_read_column * self.device_read_power
		else:
			for index in range(len(self.xbar_num_read_cell)):
				self.calculate_device_read_power(self.device_resistance[index])
				self.xbar_read_power += self.xbar_num_read_cell[index] * self.device_read_power
		if self.cell_type[0] == '0':
			self.calculate_device_read_power(self.device_resistance[-1])
			self.xbar_read_power += 0.25 * (1 - self.xbar_utilization)*\
									self.xbar_row * self.xbar_column * self.device_read_power
			#Assuming that in 0T1R structure, the read power of unselected cell is 1/4 of the selected cells' read power

	def calculate_xbar_write_power(self, cal_mode = 0):
		#unit: W
		#cal_mode: 0: simple estimation, 1: detailed simulation; num_total_write: total number of the cells need to be written; num_cell: a list of number of cells in each resistance state
		#TODO: consider the write power of unselected cells
		self.xbar_write_power = 0
		num_total_write = self.xbar_num_write_column
		# Assuming that the write operation of cells in one row can be performed concurrently
		if cal_mode == 0:
			self.calculate_device_write_power()
			self.xbar_write_power = num_total_write * self.device_write_power
		else:
			for index in range(len(self.xbar_num_write_cell)):
				self.calculate_device_write_power(self.device_resistance[index])
				self.xbar_write_power += self.xbar_num_write_cell[index] * self.device_write_power
			self.xbar_write_power /= self.xbar_num_write_column

	def calculate_xbar_read_energy(self):
		#unit: nJ
		self.xbar_read_energy = self.xbar_read_power * self.xbar_read_latency

	def calculate_xbar_write_energy(self):
		#unit: nJ
		self.xbar_write_energy = self.xbar_write_power * self.device_write_latency
			#Do not consider the wire power in write operation		

	def xbar_output(self):
		device.device_output(self)
		print("crossbar_size:", self.xbar_size)
		print("cell_type:", self.cell_type)
		print("transistor_tech:", self.transistor_tech, "nm")
		print("wire_resistance:", self.wire_resistance, "ohm")
		print("wire_capacity:", self.wire_capacity, "fF")
		print("crossbar_area", self.xbar_area, "um^2")
		print("crossbar_read_power:", self.xbar_read_power, "W")
		print("crossbar_read_latency:", self.xbar_read_latency, "ns")
		print("crossbar_read_energy:", self.xbar_read_energy, "nJ")
		print("crossbar_write_power:", self.xbar_write_power, "W")
		print("crossbar_write_latency:", self.xbar_write_latency, "ns")
		print("crossbar_write_energy:", self.xbar_write_energy, "nJ")

	
def xbar_test():
	print("load file:",test_SimConfig_path)
	_xbar = crossbar(test_SimConfig_path)
	# _xbar.xbar_output()
	print('------------')
	_xbar.calculate_xbar_write_latency()
	_xbar.calculate_xbar_area()
	_xbar.calculate_xbar_read_latency()
	# self.calculate_xbar_write_latency()
	_xbar.calculate_xbar_read_power()
	_xbar.calculate_xbar_write_power()
	_xbar.calculate_xbar_read_energy()
	_xbar.calculate_xbar_write_energy()
	_xbar.xbar_output()


if __name__ == '__main__':
	xbar_test()
