#-*-coding:utf-8-*-
# import multiprocessing
# multiprocessing.set_start_method('spawn', True)
import os

import torch.utils.data as Data
import torchvision
import torchvision.transforms as Transforms

TRAIN_BATCH_SIZE = 128
TRAIN_NUM_WORKERS = 0
TEST_BATCH_SIZE = 100
TEST_NUM_WORKERS = 0

def get_dataloader():
    train_dataset = torchvision.datasets.CIFAR100(
        root = os.path.join(os.path.dirname(__file__), "cifar100"),
        download = True,
        train = True,
        transform = Transforms.Compose([
            Transforms.Pad(padding = 4),
            Transforms.RandomCrop(32),
            Transforms.RandomHorizontalFlip(),
            Transforms.ToTensor(),
        ])
    )
    train_loader = Data.DataLoader(
        dataset = train_dataset,
        batch_size = TRAIN_BATCH_SIZE,
        shuffle = True,
        num_workers = TRAIN_NUM_WORKERS,
        drop_last = True,
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root = os.path.join(os.path.dirname(__file__), "cifar100"),
        download = True,
        train = False,
        transform = Transforms.Compose([
            Transforms.ToTensor(),
        ])
    )
    test_loader = Data.DataLoader(
        dataset = test_dataset,
        batch_size = TEST_BATCH_SIZE,
        shuffle = False,
        num_workers = TEST_NUM_WORKERS,
        drop_last = False,
    )
    return train_loader, test_loader
if __name__  == '__main__':
    train_loader, test_loader = get_dataloader()
    print(len(train_loader))
    print(len(test_loader))
    print('this is the cifar100 dataset, output shape is 32x32x3')