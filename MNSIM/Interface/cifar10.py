#-*-coding:utf-8-*-
import torchvision
import torchvision.transforms as Transforms
import torch.utils.data as Data

TRAIN_BATCH_SIZE = 128
TRAIN_NUM_WORKERS = 4
TEST_BATCH_SIZE = 100
TEST_NUM_WORKERS = 4

def get_dataloader():
    train_dataset = torchvision.datasets.CIFAR10(root = './MNSIM/Interface/cifar10',
                                                 download = True,
                                                 train = True,
                                                 transform = Transforms.Compose([Transforms.Pad(padding = 4),
                                                                                 Transforms.RandomCrop(32),
                                                                                 Transforms.RandomHorizontalFlip(),
                                                                                 Transforms.ToTensor(),
                                                                                 ])
                                                 )
    train_loader = Data.DataLoader(dataset = train_dataset,
                                   batch_size = TRAIN_BATCH_SIZE,
                                   shuffle = True,
                                   num_workers = TRAIN_NUM_WORKERS,
                                   drop_last = True,
                                   )
    test_dataset = torchvision.datasets.CIFAR10(root = './MNSIM/Interface/cifar10',
                                                download = True,
                                                train = False,
                                                transform = Transforms.Compose([Transforms.ToTensor(),
                                                                                ])
                                                )
    test_loader = Data.DataLoader(dataset = test_dataset,
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
    print('此为cifar10的输出数据集，输出尺寸调整为3x32x32')
