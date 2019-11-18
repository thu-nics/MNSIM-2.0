#!/usr/bin/python
#-*-coding:utf-8-*-
import configparser as cp
import os
import math
test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),"SimConfig.ini")
	#Default SimConfig file path: MNSIM_Python/SimConfig.ini


class buffer(object):
	def __init__(self, SimConfig_path):
		buf_config = cp.ConfigParser()
		buf_config.read(SimConfig_path, encoding='UTF-8')
		self.buf_choice = int(buf_config.get('Architecture level', 'Buffer_Choice'))
		self.buf_area = float(buf_config.get('Architecture level', 'Buffer_Area'))
		self.buf_rpower = float(buf_config.get('Architecture level', 'Buffer_ReadPower'))
		self.buf_wpower = float(buf_config.get('Architecture level', 'Buffer_WritePower'))
		self.buf_bitwidth = float(buf_config.get('Architecture level', 'Buffer_Bitwidth'))
			# unit: Byte
		if self.buf_bitwidth == 0:
			self.buf_bitwidth = 8
		self.buf_rfrequency = float(buf_config.get(('Architecture level'), 'Buffer_ReadFrequency'))
			# unit: MHz
		if self.buf_rfrequency == 0:
			self.buf_rfrequency = 1372
		self.buf_wfrequency = float(buf_config.get(('Architecture level'), 'Buffer_WriteFrequency'))
		# unit: MHz
		if self.buf_wfrequency == 0:
			self.buf_wfrequency = 1306
		self.buf_renergy = 0
		self.buf_rlatency = 0
		self.buf_wenergy = 0
		self.buf_wlatency = 0
		self.calculate_buf_read_power()
		self.calculate_buf_write_power()

	def calculate_buf_area(self):
		#unit: um^2
		buf_area_dict = {1:82032
		}
		if self.buf_choice != -1:
			assert self.buf_choice in [1]
			self.buf_area = buf_area_dict[self.buf_choice]

	def calculate_buf_read_power(self):
		#unit: W
		buf_rpower_dict = {1:0.06*1e-3
		}
		if self.buf_choice != -1:
			assert self.buf_choice in [1]
			self.buf_rpower = buf_rpower_dict[self.buf_choice]

	def calculate_buf_write_power(self):
		#unit: W
		buf_wpower_dict = {1:0.02*1e-3
		}
		if self.buf_choice != -1:
			assert self.buf_choice in [1]
			self.buf_wpower = buf_wpower_dict[self.buf_choice]

	def calculate_buf_read_latency(self, rdata = 0):
		# unit: ns
		self.buf_rlatency = rdata/self.buf_bitwidth/self.buf_rfrequency*1e3

	def calculate_buf_write_latency(self, wdata = 0):
		# unit: ns
		self.buf_wlatency = wdata/self.buf_bitwidth/self.buf_wfrequency*1e3

	def calculate_buf_read_energy(self, rdata = 0):
		#unit: nJ
		self.calculate_buf_read_latency(rdata)
		self.buf_renergy = self.buf_rlatency * self.buf_rpower

	def calculate_buf_write_energy(self, wdata = 0):
		#unit: nJ
		self.calculate_buf_write_latency(wdata)
		self.buf_wenergy = self.buf_wlatency * self.buf_rpower

	def buf_output(self):
		if self.buf_choice == -1:
			print("buf_choice: User defined")
		else:
			print("buf_choice:", self.buf_choice)
		print("buf_area:", self.buf_area, "um^2")
		print("buf_read_power:", self.buf_rpower, "W")
		print("buf_read_energy:", self.buf_renergy, "nJ")
		print("buf_read_latency:", self.buf_rlatency, "ns")
		print("buf_write_power:", self.buf_wpower, "W")
		print("buf_write_energy:", self.buf_wenergy, "nJ")
		print("buf_write_latency:", self.buf_wlatency, "ns")
	
def buf_test():
	print("load file:",test_SimConfig_path)
	_buf = buffer(test_SimConfig_path)
	_buf.calculate_buf_area()
	_buf.calculate_buf_read_power()
	_buf.calculate_buf_read_latency()
	_buf.calculate_buf_read_energy()
	_buf.calculate_buf_write_power()
	_buf.calculate_buf_write_latency()
	_buf.calculate_buf_write_energy()
	_buf.buf_output()


if __name__ == '__main__':
	buf_test()