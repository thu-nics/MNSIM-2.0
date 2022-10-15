#-*-coding:utf-8-*-
"""
@FileName:
    trainer.py
@Description:
    trainer for model
@Authors:
    Hanbo Sun(sun-hb17@mails.tsinghua.edu.cn)
@CreateTime:
    2022/04/15 18:46
"""
import abc
import os

import torch
from MNSIM.Interface.utils.component import Component
from torch import nn
from torch.utils.tensorboard import SummaryWriter


def traverse(self, func):
    """
    traverse the network to find the test result
    """
    device = self.device
    self.model.eval()
    self.model.to(device)
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in self.test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = func(images)
            # predicted
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
    return test_correct / test_total

class BaseTrainer(Component):
    """
    Base class for trainer
    """
    REGISTRY = "trainer"
    def __init__(self, dataset, model, trainer_config):
        super(BaseTrainer, self).__init__()
        self.dataset = dataset
        self.model = model
        self.epochs, self.device = 0, torch.device("cpu")
        self.loss, self.optimizer, self.scheduler = \
            self._init_trainer(trainer_config)
        self.train_mode = trainer_config.get("train_mode", "FIX_TRAIN")
        self.test_mode = trainer_config.get("test_mode", "SINGLE_FIX_TEST")
        self.show_name = f"{self.dataset.get_name()}_{self.model.get_name()}_{self.get_name()}" + \
            f"_{self.train_mode}_{self.test_mode}"
        # init train and test loader
        self.train_loader, self.test_loader = \
            self.dataset.get_loader("train", 0), self.dataset.get_loader("test", 0)
        # init path, log_dir, and weights
        self.save_path = trainer_config.get("save_path", None)
        self.log_dir = trainer_config["log_dir"]
        assert os.path.exists(self.log_dir), f"log_dir: {self.log_dir} is not exist"
        if "weight_path" in trainer_config:
            weight_path = trainer_config["weight_path"]
            self.logger.info(f"load weights from {weight_path}")
            self.model.load_change_weights(
                torch.load(weight_path, map_location=torch.device("cpu"))
            )
        # init writer and logger
        self.writer = SummaryWriter(
            log_dir=os.path.join(self.log_dir, self.show_name)
        )
        self.logger.info(f"init trainer for {self.show_name}")

    @abc.abstractmethod
    def _init_trainer(self, train_config):
        """
        Initialize the trainer
        """
        raise NotImplementedError

    def train(self):
        """
        train the model based one the config
        """
        epochs = self.epochs
        device = self.device
        assert self.train_mode in ("TRADITION", "FIX_TRAIN"), \
            f"train mode {self.train_mode} is not supported"
        assert os.path.exists(self.save_path), \
            f"save path {self.save_path} is not exist"
        save_name = os.path.join(self.save_path, self.show_name + ".pth")
        for epoch in range(epochs):
            # logger
            self.logger.info(f"train in epoch {epoch}")
            # train
            self.model.train()
            self.model.to(device)
            for i, (images, labels) in enumerate(self.train_loader):
                self.model.zero_grad()
                # forward
                images, labels = images.to(device), labels.to(device)
                outputs = self.model.forward(images, self.train_mode)
                loss = self.loss(outputs, labels)
                # backward
                loss.backward()
                self.optimizer.step()
                # writer
                self.writer.add_scalar("loss", loss.item(), len(self.train_loader)*epoch+i)
            self.scheduler.step()
            # test
            self.test(epoch)
        # save
        self.logger.info(f"save model to {save_name}")
        torch.save(self.model.state_dict(), save_name)

    def test(self, epoch):
        """
        test the model based one the config
        """
        accuracy = traverse(self, lambda x: self.model.forward(x, self.test_mode))
        self.writer.add_scalar("accuracy", accuracy, epoch)
        self.logger.info(f"test accuracy in epoch {epoch} under {self.test_mode} is {accuracy}")
        return accuracy

    def set_test_mode(self, test_mode):
        """
        set the test mode
        """
        self.logger.info(f"change test mode to {test_mode}")
        self.test_mode = test_mode

    def evaluate(self, net_bit_weights):
        """
        test the model based on the net_bit_weights
        """
        # transfer weights to device
        for layer_weights in net_bit_weights:
            if layer_weights is None:
                continue
            for split_weights in layer_weights:
                for i, bit_weights in enumerate(split_weights):
                    split_weights[i] = bit_weights.to(self.device)
        accuracy = traverse(self,
            lambda x: self.model.set_weights_forward(x, net_bit_weights)
        )
        self.logger.info(f"evaluation accuracy is {accuracy}")
        return accuracy

    @abc.abstractmethod
    def get_name(self):
        """
        get the name of the trainer
        """
        raise NotImplementedError


class SGDTrainer(BaseTrainer):
    """
    SGD trainer
    """
    NAME = "SGD"
    def _init_trainer(self, train_config):
        """
        init the sgd trainer
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD([{
            "params": self.model.parameters(),
            "lr": train_config["lr"],
            "weight_decay": train_config["weight_decay"]}
        ], momentum=train_config["momentum"])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=train_config["milestones"],
            gamma=train_config["gamma"],
        )
        # epoch and device
        self.epochs = train_config["epochs"]
        self.device = torch.device(f"cuda:{train_config['device']}") if \
            train_config["device"] != -1 else torch.device("cpu")
        return criterion, optimizer, scheduler

    def get_name(self):
        return self.NAME
