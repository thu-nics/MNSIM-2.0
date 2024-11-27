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

#ImageNet_PATH needs to be changed
def get_dataloader(ImageNet_PATH='/share/linqiushi-nfs', batch_size=64, workers=3, pin_memory=True): 
    
    traindir = os.path.join(ImageNet_PATH, 'Imagenet-train/ILSVRC2012_img_train')
    valdir   = os.path.join(ImageNet_PATH, 'Imagenet-value')
    print('traindir = ',traindir)
    print('valdir = ',valdir)
    
    normalizer = Transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        Transforms.Compose([
            Transforms.RandomResizedCrop(224),
            Transforms.RandomHorizontalFlip(),
            Transforms.ToTensor(),
            normalizer
        ])
    )
    
    val_dataset = torchvision.datasets.ImageFolder(
        valdir,
        Transforms.Compose([
            Transforms.Resize(256),
            Transforms.CenterCrop(224),
            Transforms.ToTensor(),
            normalizer
        ])
    )
    print('train_dataset = ',len(train_dataset))
    print('val_dataset   = ',len(val_dataset))
    
    train_loader = Data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    val_loader = Data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader

if __name__  == '__main__':
    train_loader, test_loader = get_dataloader()
    print(len(train_loader))
    print(len(test_loader))
    print('this is the Imagenet100 dataset, output shape is 224x224x3')