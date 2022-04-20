#-*-coding:utf-8-*-
"""
@FileName:
    evaluation.py
@Description:
    interface for evaluation
@Authors:
    Hanbo Sun(sun-hb17@mails.tsinghua.edu.cn)
@CreateTime:
    2022/04/15 17:33
"""
from MNSIM.Interface.dataset import ClassificationBaseDataset
from MNSIM.Interface.model import BaseModel
from MNSIM.Interface.trainer import BaseTrainer
from MNSIM.Interface.utils.component import Component
from MNSIM.Interface.utils.utils import _init_component, load_sim_config
from MNSIM.Interface.utils.yaml_io import read_yaml


class EvaluationInterface(Component):
    """
    evaluation interface for MNSIM
    two method: evaluate and get_structure
    """
    REGISTRY = "interface"
    NAME = "train_test_interface"
    def __init__(self, simconfig_path, evaluation_config_path):
        super(EvaluationInterface, self).__init__()
        # load config
        self.hardware_config = load_sim_config(simconfig_path)
        self.evaluation_config = read_yaml(evaluation_config_path)
        # init dataset
        self.dataset = _init_component(
            ClassificationBaseDataset, self.evaluation_config, "dataset"
        )
        # update config by hardware config, init model
        self.model = _init_component(
            BaseModel, self.evaluation_config, "model",
            {
                "hardware_config": self.hardware_config,
                "dataset": self.dataset
        })
        # trainer
        self.trainer = _init_component(
            BaseTrainer, self.evaluation_config, "trainer",
            {
                "dataset": self.dataset,
                "model": self.model,
        })

    def get_structure(self):
        """
        get the key structure of the model
        """
        return self.model.get_key_structure()

    def origin_evaluate(self, method):
        """
        evaluate the model under different test mode
        """
        self.trainer.set_test_mode(method)
        return self.trainer.test(0)

    def get_net_bits(self):
        """
        return the bits of the model
        """
        return self.model.get_weights()

    def set_net_bits_evaluate(self, net_bits):
        """
        set the bits of the model and evaluate
        """
        return self.trainer.evaluate(net_bits)
