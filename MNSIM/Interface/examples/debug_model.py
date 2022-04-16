#-*-coding:utf-8-*-
"""
@FileName:
    debug.py
@Description:
    add for debug
@Authors:
    Hanbo Sun(sun-hb17@mails.tsinghua.edu.cn)
@CreateTime:
    2022/04/14 15:58
"""
import torch
from MNSIM.Interface.dataset import ClassificationBaseDataset
from MNSIM.Interface.model import BaseModel

dataset_ini = {
    "TRAIN_BATCH_SIZE": 128,
    "TRAIN_NUM_WORKERS": 4,
    "TEST_BATCH_SIZE": 100,
    "TEST_NUM_WORKERS": 4,
}
model_config_path = "resnet18.yaml"
dataset = ClassificationBaseDataset.get_class_("cifar10")(dataset_ini)
model = BaseModel.get_class_("yaml")(model_config_path, {}, dataset)
model.load_change_weights(
    torch.load("../zoo/train_resnet1_params.pth", map_location=torch.device("cpu"))
)
# input
input = torch.load("model_input.pth", map_location=torch.device("cpu"))
model.eval()
output = model.forward(input, method="SINGLE_FIX_TEST")
# output = model.set_weights_forward(input, model.get_weights())
# output = model.get_structure()
print(output)
torch.save(output, "model_output.pth")
