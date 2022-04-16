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
import functools
import torch
from torch import nn

from torch.utils.tensorboard import SummaryWriter
from MNSIM.Interface.utils.component import Component


def traverse(self, func):
    """
    traverse the network to find the test result
    """
    device = self.device
    self.model.eval()
    self.model.to(device)
    loader = self.dataset.get_loader("test", 0)
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
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
        self.loss, self.optimizer, self.scheduler = \
            self._init_trainer(trainer_config)
        # init path, log_dir, and weights
        self.save_path = trainer_config.get("save_path", None)
        self.log_dir = trainer_config["log_dir"]
        if "weight_path" in trainer_config:
            self.weight_path = trainer_config["weight_path"]
            self.model.load_change_weights(
                torch.load(self.weight_path, map_location=torch.device("cpu"))
            )
        self.train_mode = trainer_config.get("train_mode", "TRADITION")
        self.test_mode = trainer_config.get("test_mode", "TRADITION")
        # init writer and logger
        self.writer = SummaryWriter(log_dir=trainer_config["log_dir"])
        self.logger.info(f"init trainer")

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
        for epoch in range(epochs):
            # logger
            self.logger.info(f"train in epoch {epoch}")
            # train
            self.model.train()
            self.model.to(device)
            loader = self.dataset.get_loader("train", 0)
            for i, (images, labels) in enumerate(loader):
                self.model.zero_grad()
                # forward
                images, labels = images.to(device), labels.to(device)
                outputs = self.model.forward(images, self.train_mode)
                loss = self.loss(outputs, labels)
                # backward
                loss.backward()
                self.optimizer.step()
                # writer
                self.writer.add_scalar("loss", loss.item(), len(loader)*epoch+i)
            self.scheduler.step()
            # test
            self.test(epoch)

    def test(self, epoch):
        """
        test the model based one the config
        """
        accuracy = traverse(self, lambda x: self.model.forward(x, self.test_mode))
        self.writer.add_scalar("accuracy", accuracy, epoch)
        self.logger.info(f"test accuracy in epoch {epoch} is {accuracy}")

    def evaluate(self, net_bit_weights):
        """
        test the model based on the net_bit_weights
        """
        accuracy = traverse(self,
            lambda x: self.model.set_weughts_forward(x, net_bit_weights)
        )
        self.logger.info(f"evaluation accuracy is {accuracy}")


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
