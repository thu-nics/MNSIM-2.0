#-*-coding:utf-8-*-
"""
@FileName:
    logger.py
@Description:
    logger for interface class
@Authors:
    Hanbo Sun(sun-hb17@mails.tsinghua.edu.cn)
@CreateTime:
    2021/11/19 16:02
"""
import logging
import os
import sys

# import part
__all__ = ["logger", "getLogger"]

# by default, log level is logging.INFO
LEVEL = "info"
if "MNSIM_LOG_LEVEL" in os.environ:
    LEVEL = os.environ["MNSIM_LOG_LEVEL"]
LEVEL = getattr(logging, LEVEL.upper())
LOG_FORMAT = "%(asctime)s %(name)-16s %(levelname)7s: %(message)s"
logging.basicConfig(
    stream=sys.stdout, level=LEVEL,
    format=LOG_FORMAT, datefmt="%m/%d %I:%M:%S %p"
)

logger = logging.getLogger()
def addFile(self, filename):
    """
    addFile, add File handler
    """
    handler = logging.FileHandler(filename)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    self.addHandler(handler)
# logger.__class__.addFile = addFile
logging.Logger.addFile = addFile

def getLogger(name):
    """
    child logger
    """
    return logger.getChild(name)
