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
from MNSIM.Interface.utils.utils import get_home_path
from MNSIM.Interface.evaluation import EvaluationInterface

def _init_evaluation_interface(network, dataset, hardware_config_path, weight_path, device):
    """
    init evaluation interface
    """
    config_path = None
    home_path = get_home_path()
    if network in ("lenet", "vgg8", "vgg11", "resnet18", "alexnet", "vgg16", "alexnet_imagenet", "vgg8_imagenet", "vgg11_imagenet", "resnet18_imagenet", "vgg16_imagenet"):
        # for old interface, copy based on base.yaml
        base_yaml_path = os.path.join(home_path, "MNSIM/Interface/examples/base.yaml")
        assert os.path.exists(base_yaml_path), f"base.yaml in {base_yaml_path} not found"
        config = read_yaml(base_yaml_path)
        # update config based on input params
        ## update for network, model config path
        model_config_path = os.path.join(home_path, f"MNSIM/Interface/examples/{network}.yaml")
        assert os.path.exists(model_config_path), f"{network} in {model_config_path} not found"
        config["model_cfg"]["model_config_path"] = model_config_path
        ## update for weight_path
        if weight_path is not None:
            assert os.path.exists(weight_path), f"{weight_path} not found"
            config["trainer_cfg"]["trainer_config"]["weight_path"] = weight_path
        else:
            config["trainer_cfg"]["trainer_config"].pop("weight_path", None)
        ## update for save path and log dir
        config["trainer_cfg"]["trainer_config"]["log_dir"] = os.path.join(
            home_path, config["trainer_cfg"]["trainer_config"]["log_dir"]
        )
        config["trainer_cfg"]["trainer_config"]["save_path"] = os.path.join(
            home_path, config["trainer_cfg"]["trainer_config"]["save_path"]
        )
        ## update for dataset and device
        config["dataset_type"] = dataset
        config["trainer_cfg"]["trainer_config"]["device"] = \
            device if device is not None else -1
        # save config
        config_path = os.path.join(home_path, "MNSIM/Interface/examples/.tmp.yaml")
        write_yaml(config_path, config)
    else:
        assert os.path.exists(network), f"{network} not exist"
        config_path = network
    return EvaluationInterface(hardware_config_path, config_path)
