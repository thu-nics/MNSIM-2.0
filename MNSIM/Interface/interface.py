#-*-coding:utf-8-*-
import math
import numpy as np
import torch
import collections
import json
import configparser
from importlib import import_module

class TrainTestInterface(object):
    def __init__(self):
        netwotk_module = 'lenet'
        dataset_module = 'cifar10'
        weights_file = './zoo/cifar10_lenet_train_params.pth'
        # load net, dataset, and weights
        self.net = import_module(netwotk_module).get_net()
        self.test_loader = import_module(dataset_module).get_dataloader()[1]
        self.net.load_state_dict(torch.load(weights_file))
        self.device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    def origin_evaluate(self):
        self.net.to(self.device)
        self.net.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                images = images.to(self.device)
                test_total += labels.size(0)
                outputs = self.net(images)
                # predicted
                labels = labels.to(self.device)
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum().item()
        return test_correct / test_total
    def get_bits(self):
        self.net_bit_weights = self.net.get_weight()
        return self.net_bit_weights
    def set_bit_evaluate(self, net_bit_weights):
        self.net.to(self.device)
        self.net.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                images = images.to(self.device)
                test_total += labels.size(0)
                outputs = self.net.set_weight_forward(images, net_bit_weights)
                # predicted
                labels = labels.to(self.device)
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum().item()
        return test_correct / test_total

if __name__ == '__main__':
    inter = TrainTestInterface()
    print(inter.origin_evaluate())
    net_bit_weights = inter.get_bits()
    print(inter.set_bit_evaluate(net_bit_weights))
