#-*-coding:utf-8-*-
import argparse
from importlib import import_module

import torch

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', help = 'select gpu')
parser.add_argument('-d', '--dataset', help = 'select dataset')
parser.add_argument('-n', '--net', help = 'select net')
parser.add_argument('-t', '--train', help = 'select train')
parser.add_argument('-p', '--prefix', help = 'select prefix')
parser.add_argument('-m', '--mode', help = 'select mode', choices = ['train', 'test'])
parser.add_argument('-w', '--weight', help = 'weight file')
args = parser.parse_args()
assert args.gpu
assert args.dataset
assert args.net
assert args.train
assert args.mode
if args.mode == 'train':
    if args.prefix == None:
        args.prefix = args.dataset.split('.')[-1] + '_' + args.net
elif args.mode == 'test':
    assert args.weight
else:
    assert 0

print(f'args is {args}')
# dataloader
dataset_module = import_module(f'MNSIM.Interface.{args.dataset}')
train_loader, test_loader = dataset_module.get_dataloader()
# net
net_module = import_module('MNSIM.Interface.network')
if args.dataset.endswith('cifar10'):
    num_classes = 10
elif args.dataset.endswith('cifar100'):
    num_classes = 100
else:
    assert 0, f'unknown dataset'

net = net_module.get_net(cate = args.net, num_classes = num_classes)
# train
train_module = import_module(f'MNSIM.Interface.{args.train}')
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
print(f'run on device {device}')
# weights
if args.weight is not None:
    print(f'load weights, {args.weight}')
    net.load_change_weights(torch.load(args.weight, map_location=device))
    # net.load_state_dict(torch.load(args.weight))
if args.mode == 'train':
    train_module.train_net(net, train_loader, test_loader, device, args.prefix)
if args.mode == 'test':
    assert args.weight
    # net.load_change_weights(torch.load(args.weight))
    train_module.eval_net(net, test_loader, 0, device)
