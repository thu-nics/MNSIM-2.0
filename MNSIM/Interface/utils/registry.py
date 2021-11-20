#-*-coding:utf-8-*-
"""
@FileName:
    registry.py
@Description:
    registry base class
@Authors:
    Hanbo Sun(sun-hb17@mails.tsinghua.edu.cn)
@CreateTime:
    2021/11/19 16:06
"""
import abc
import collections

from MNSIM.Interface.utils.logger import getLogger

# import part
__all__ = ["RegistryMeta", "RegistryError"]

class RegistryError(Exception):
    """
    Registry error, child class of exception
    """
    def __init__(self, message):
        super(RegistryError, self).__init__(message)

LOGGER = getLogger("registry_meta")
class RegistryMeta(abc.ABCMeta):
    """
    Registry meta, in registry_dict
    """
    registry_dict = collections.defaultdict(dict)

    def __init__(cls, name, bases, namespace):
        super(RegistryMeta, cls).__init__(name, bases, namespace)
        # all child class should have REGISTRY
        if hasattr(cls, "REGISTRY"):
            # other cases, register the class
            table = cls.REGISTRY
            entry = cls.NAME if hasattr(cls, "NAME") else None
            abstract_methods = cls.__abstractmethods__
            # case 1: abstract_methods is not null and entry is None
            if (len(abstract_methods) > 0) and (entry is None):
                LOGGER.info(
                    "Register cls {} as table {}".format(
                        name, table
                    )
                )
            elif (len(abstract_methods) == 0) and (entry is not None):
                RegistryMeta.registry_dict[table][entry] = cls
                LOGGER.info(
                    "Register class {} as entry {} in table {}.".format(
                        name, entry, table
                    )
                )
            else:
                raise RegistryError("There are both or neither abstract and NAME")

    @classmethod
    def get_class(cls, table, name):
        """
        classmethod, get cls from table and name
        """
        try:
            return cls.all_classes(table)[name]
        except KeyError as error:
            raise RegistryError(
                "No registry item {} available in registry table {}.".format(
                    name, table
                )
            ) from error

    @classmethod
    def all_classes(cls, table):
        """
        classmethod, get all cls in table
        """
        try:
            return cls.registry_dict[table]
        except KeyError as error:
            raise RegistryError("No registry table {} available.".format(table)) from error

    @classmethod
    def avail_tables(cls):
        """
        classmethod, get all available table list
        """
        return cls.registry_dict.keys()

    def all_classes_(cls):
        """
        get all classes
        """
        return RegistryMeta.all_classes(cls.REGISTRY)

    def get_class_(cls, name):
        """
        get name class
        """
        return RegistryMeta.get_class(cls.REGISTRY, name)
