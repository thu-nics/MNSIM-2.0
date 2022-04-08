#-*-coding:utf-8-*-
"""
@FileName:
    yaml_io.py
@Description:
    read and write the yaml file
@Authors:
    Hanbo Sun(sun-hb17@mails.tsinghua.edu.cn)
@CreateTime:
    2022/04/08 17:00
"""
import yaml

# read yaml file
def read_yaml(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f.read())

# write yaml file
def write_yaml(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False)
