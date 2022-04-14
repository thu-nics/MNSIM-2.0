#-*-coding:utf-8-*-
"""
@FileName:
    debug_net.py
@Description:
    debug net
@Authors:
    Hanbo Sun(sun-hb17@mails.tsinghua.edu.cn)
@CreateTime:
    2022/04/14 17:04
"""
import torch
from MNSIM.Interface.network import get_net
net = get_net(cate="resnet18", num_classes=10)
net.load_state_dict(
    torch.load("../zoo/train_resnet1_params.pth", map_location=torch.device("cpu"))
)
# input
input = torch.load("model_input.pth", map_location=torch.device("cpu"))
net.eval()
output = net.forward(input, method="SINGLE_FIX_TEST")
# output = net.set_weights_forward(input, net.get_weights())
# output = net.get_structure()
print(output)
torch.save(output, "net_output.pth")