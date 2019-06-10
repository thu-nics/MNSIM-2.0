#!/usr/bin/python
#-*-coding:utf-8-*-
import configparser as cp
import os
import math
test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),"SimConfig.ini")
	#Default SimConfig file path: MNSIM_Python/SimConfig.ini


class device(object):
	def __init__(self, SimConfig_path):
		device_config = cp.ConfigParser()
		device_config.read(SimConfig_path)
		self.device_tech = float(device_config.get('Device level', 'Device_Tech'))
		self.device_read_voltage = float(device_config.get('Device level', 'Read_Voltage'))
		self.device_write_voltage = float(device_config.get('Device level', 'Write_Voltage'))
		self.device_read_latency = float(device_config.get('Device level', 'Read_Latency'))
		self.device_write_latency = float(device_config.get('Device level', 'Write_Latency'))
		self.device_bit_level = int(device_config.get('Device level', 'Device_Bit_Level'))
		self.device_resistance = list(map(float, device_config.get('Device level', 'Device_Resistance').split(',')))
		self.decice_variation = float(device_config.get('Device level', 'Device_Variation'))
			# Device variation is defined as \Delta R / R
		self.device_read_power = 0
		self.device_write_power = 0
		# print("Device configuration is loaded")
		# self.calculate_device_read_power(1)
		# self.calculate_device_write_power(1)

	def calculate_device_read_power(self, R = None): 
		#R is the resistance of memristor, None means use default resistance (Sqrt(R_on*R_off))
		if R is None:
			R = math.sqrt(float(self.device_resistance[0])*float(self.device_resistance[-1]))
		assert R > 0
		self.device_read_power = self.device_read_voltage ** 2 / R

	def calculate_device_write_power(self, R = None):
		#R is the resistance of memristor, None means use default resistance (Sqrt(R_on*R_off))
		if R is None:
			R= math.sqrt(float(self.device_resistance[0])*float(self.device_resistance[-1]))
		assert R > 0
		self.device_write_power = self.device_write_voltage ** 2 / R

	def device_output(self):
		print("device_tech:", self.device_tech, "nm")
		print("read_voltage:", self.device_read_voltage, "V")
		print("write_voltage:", self.device_write_voltage, "V")
		print("read_latency:", self.device_read_latency, "ns")
		print("write_latency:", self.device_write_latency, "ns")
		print("device_bit_level", self.device_bit_level)
		print("device_resistance:", self.device_resistance, "(ohm)")
		print("device_variation:", self.decice_variation, "%")
		print("device_read_power:", self.device_read_power, "W")
		print("device_write_power:", self.device_write_power, "W")

	
def device_test():
	print("load file:",test_SimConfig_path)
	_device = device(test_SimConfig_path)
	_device.calculate_device_read_power()
	_device.calculate_device_write_power()
	_device.device_output()


if __name__ == '__main__':
	device_test()