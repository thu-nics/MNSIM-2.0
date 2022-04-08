#-*-coding:utf-8-*-
"""
@FileName:
    dataset.py
@Description:
    dataset class for all classification dataset
@Authors:
    Hanbo Sun(sun-hb17@mails.tsinghua.edu.cn)
@CreateTime:
    2021/11/19 16:43
"""
import abc
import copy
import os
import shutil

import torch
import torch.utils.data as Data
import torchvision
import torchvision.transforms as Transforms
from MNSIM.Interface.utils.component import Component

def check_move_dataset(interface_path, dataset_path, name):
    """
    copy origin path in interface to target path in dataset
    """
    origin_path = os.path.join(interface_path, name)
    target_path = os.path.join(dataset_path, name)
    if name is None:
        raise Exception("name should not be None")
    if os.path.exists(target_path):
        return target_path
    if os.path.exists(origin_path):
        # cp to target path
        shutil.copytree(origin_path, target_path)
    return target_path

class ClassificationBaseDataset(Component):
    """
    classification base dataset
    """
    REGISTRY = "classification_dataset"

    def __init__(self, dataset_ini):
        super(ClassificationBaseDataset, self).__init__()
        # path for interface and dataset
        interface_path = os.path.dirname(__file__)
        dataset_path = os.path.join(interface_path, "dataset")
        self.target_path = \
            check_move_dataset(
                interface_path, dataset_path, getattr(self, "NAME", None)
            )
        self.dataset_ini = copy.deepcopy(dataset_ini)

    def get_dataset(self, dataset_type, dataset_property, dataset_cls):
        """
        new attr for this dataset
        """
        assert dataset_type in ["train", "test"], \
            "dataset type must be train or test"
        dataset_cfg = self.get_dataset_cfg(dataset_type)
        self.logger.info(
            "init dataset as {}".format(dataset_property) + \
            " :\n" + str(dataset_cfg)
        )
        setattr(
            self,
            dataset_property,
            dataset_cls(**dataset_cfg)
        )

    def get_loader(self, loader_type, loader_num=0):
        """
        loader config for loader_type
        loader_type must be train or test
        loader_num, how many batch need to be load
        """
        assert loader_type in ["train", "test"], \
            "loader_type should be train or test but {}".format(loader_type)
        # loader config, batch_size and others
        loader_cfg = self.get_loader_cfg(loader_type)
        self.logger.info(
            "init loader for {}".format(loader_type) + \
            " :\n" + str(loader_cfg)
        )
        if loader_num != 0:
            assert isinstance(loader_num, int) and loader_num > 0, \
                "loader_num should in type int and bigger than 0, but {}".format(loader_num)
            indices = range(min(loader_cfg["batch_size"] * loader_num, len(loader_cfg["dataset"])))
            loader_cfg["dataset"] = Data.Subset(loader_cfg["dataset"], indices)
        return Data.DataLoader(**loader_cfg)

    @abc.abstractmethod
    def get_dataset_cfg(self, dataset_type):
        """
        return dataset config
        """
        pass

    @abc.abstractmethod
    def get_loader_cfg(self, loader_type):
        """
        get different loader config for different loader type
        should be inherited by sub classes
        loader_cfg: "batch_size", "shuffle", "num_workers", "drop_last", "dataset"
        """
        pass

    @abc.abstractmethod
    def get_num_classes(self):
        """
        return number classes of this dataset
        """
        pass

    @abc.abstractmethod
    def get_dataset_info(self):
        """
        return dataset info
        """
        raise NotImplementedError

class CIFAR10(ClassificationBaseDataset):
    """
    cifar10 dataset, name and config
    """
    NAME = "cifar10"

    def __init__(self, dataset_ini):
        super(CIFAR10, self).__init__(dataset_ini)
        self.get_dataset(
            "train",
            "train_dataset",
            getattr(torchvision.datasets, self.NAME.upper()),
        )
        self.get_dataset(
            "test",
            "test_dataset",
            getattr(torchvision.datasets, self.NAME.upper()),
        )

    def get_dataset_cfg(self, dataset_type):
        """
        return train or test dataset config
        """
        assert dataset_type in ["train", "test"], \
            "dataset type must be train or test"
        # dataset config
        dataset_cfg = {
            "root": self.target_path,
            "download": not os.path.exists(self.target_path),
            "train": dataset_type == "train",
            "transform": Transforms.Compose([
                Transforms.Pad(padding = 4),
                Transforms.RandomCrop(32),
                Transforms.RandomHorizontalFlip(),
                Transforms.ToTensor(),
            ]) if dataset_type == "train" else Transforms.ToTensor()
        }
        return dataset_cfg

    def get_loader_cfg(self, loader_type):
        """
        loader config for loader_type
        loader_type should be train or test
        loader_num, how many batch need to be load
        """
        assert loader_type in ["train", "test"], \
            "loader_type should be train or test but {}".format(loader_type)
        # loader config, batch_size and others
        loader_cfg = {
            "batch_size": self.dataset_ini.get(loader_type.upper() + "_BATCH_SIZE", None),
            "shuffle": loader_type == "train",
            "num_workers": self.dataset_ini.get(loader_type.upper() + "_NUM_WORKERS", None),
            "drop_last": loader_type == "train",
            "dataset": getattr(self, loader_type + "_dataset")
        }
        assert loader_cfg["batch_size"] is not None and loader_cfg["num_workers"] is not None
        return loader_cfg

    def get_num_classes(self):
        return 10

    def get_dataset_info(self):
        dataset_info = {
            "bit_scale": torch.FloatTensor([9, 1/255.]),
            "shape": (1, 3, 32, 32)
        }
        return dataset_info

class CIFAR100(CIFAR10):
    """
    cifar100 dataset
    """
    NAME = "cifar100"
    def get_num_classes(self):
        return 100
