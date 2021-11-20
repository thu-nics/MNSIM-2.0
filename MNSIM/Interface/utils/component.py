#-*-coding:utf-8-*-
"""
@FileName:
    registry.py
@Description:
    Registry for all interface class
@Authors:
    Hanbo Sun(sun-hb17@mails.tsinghua.edu.cn)
@CreateTime:
    2021/11/19 15:59
"""
from MNSIM.Interface.utils.logger import getLogger
from MNSIM.Interface.utils.registry import RegistryMeta

class Component(object, metaclass=RegistryMeta):
    """
    component for all other parts
    init logger and other things
    """
    def __init__(self):
        self._logger = None

    @property
    def logger(self):
        """
        get logger
        """
        if self._logger is None:
            self._logger = getLogger(self.__class__.__name__)
        return self._logger

    def __getstate__(self):
        state = self.__dict__.copy()
        if "_logger" in state:
            del state["_logger"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._logger = None
