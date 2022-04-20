#-*-coding:utf-8-*-
"""
@FileName:
    init_interface.py
@Description:
    interface for init
@Authors:
    Hanbo Sun(sun-hb17@mails.tsinghua.edu.cn)
@CreateTime:
    2022/04/20 14:26
"""
import os
from MNSIM.Interface.utils.yaml_io import read_yaml, write_yaml
from MNSIM.Interface.evaluation import EvaluationInterface

def _init_evaluation_interface(network, dataset, hardware_config_path, weight_path, device):
    """
    init evaluation interface
    """
    config_path = None
    if network in ("lenet", "vgg8", "resnet18"):
        # for old interface
        model_config_path = f"MNSIM/Interface/examples/{network}.yaml"
        assert os.path.exists(model_config_path) and os.path.exists(weight_path), \
            f"{model_config_path} or {weight_path} not exist"
        # load base yaml
        base_path = "MNSIM/Interface/examples/base.yaml"
        assert os.path.exists(base_path), f"{base_path} not exist"
        config = read_yaml(base_path)
        # set base yaml
        config["dataset_type"] = dataset
        config["model_cfg"]["model_config_path"] = model_config_path
        config["trainer_cfg"]["trainer_config"]["weight_path"] = weight_path
        config["trainer_cfg"]["trainer_config"]["device"] = device
        # write
        config_path = "MNSIM/Interface/examples/.tmp.yaml"
        write_yaml(config_path, config)
    else:
        assert os.path.exists(network), f"{network} not exist"
        config_path = network
    return EvaluationInterface(hardware_config_path, config_path)
