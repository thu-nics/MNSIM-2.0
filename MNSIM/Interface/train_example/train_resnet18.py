from email import header


#-*-coding:utf-8-*-
"""
@FileName:
    train_resnet18.py
@Description:
    train resnet18 on cifar100
@Authors:
    Hanbo Sun(sun-hb17@mails.tsinghua.edu.cn)
@CreateTime:
    2022/10/15 15:45
"""
import os
import sys
from MNSIM.Interface.evaluation import EvaluationInterface
# change dir to top level
top_level = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
))
os.chdir(top_level)
# input for different solver
solver_path = sys.argv[1]
assert os.path.exists(solver_path), f"{solver_path} not found"
evaluation_interface = EvaluationInterface("SimConfig.ini", solver_path)
evaluation_interface.trainer.train()
# evaluation_interface.model.export_onnx("resnet18.onnx")