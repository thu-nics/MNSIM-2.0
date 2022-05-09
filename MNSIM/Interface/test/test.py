#-*-coding:utf-8-*-
#pylint:disable=import-outside-toplevel
"""
@FileName:
    test.py
@Description:
    test for mnsim interface
@Authors:
    Hanbo Sun(sun-hb17@mails.tsinghua.edu.cn)
@CreateTime:
    2021/12/16 16:24
"""
import random
import pytest


@pytest.mark.parametrize(
    argnames="dataset_name",
    argvalues=["cifar10", "cifar100"]
)
def test_dataset(dataset_name):
    """
    dataset module test
    """
    # dataset
    from MNSIM.Interface.dataset import ClassificationBaseDataset
    dataset_ini = {
        "TRAIN_BATCH_SIZE": 128,
        "TRAIN_NUM_WORKERS": 4,
        "TEST_BATCH_SIZE": 100,
        "TEST_NUM_WORKERS": 4,
    }
    dataset = ClassificationBaseDataset.get_class_(dataset_name)(dataset_ini)
    print(f"{dataset_name}, info:")
    print(f"train, test numbers are {len(dataset.train_dataset)} and {len(dataset.test_dataset)}")
    print(f"num_class is {dataset.get_num_classes()}\n")
    # dataloader
    def loader_metric(loader):
        L = len(loader)
        for sample_image, sample_label in loader:
            image_info = f"{sample_image.shape}, {sample_image.dtype}, {sample_image.device}"
            label_info = f"{sample_label.shape}, {sample_label.dtype}, {sample_label.device}"
            sample_image = sample_image.transpose(0, 1)
            sample_image = sample_image.reshape([sample_image.shape[0], -1])
            image_value = f"min: {sample_image.min(dim=1)[0]}, max: {sample_image.max(dim=1)[0]}"+\
                f", mean: {sample_image.mean(dim=1)}, std: {sample_image.std(dim=1)}"
            sample_label = sample_label.float()
            label_value = f"min: {sample_label.min()}, max: {sample_label.max()}" + \
                f", mean: {sample_label.mean()}, std: {sample_label.std()}"
            break
        metric = f"loader length is {L}\nimage and label info is\n{image_info}\n{label_info}\n"+\
            f"image and label value is\n{image_value}\n{label_value}"
        return metric
    for loader_type in ["train", "test"]:
        for loader_num in [0] + [random.randint(1, 99)]:
            loader = dataset.get_loader(loader_type, loader_num)
            metric = loader_metric(loader)
            print(f"{dataset_name}, loader type is {loader_type}, loader number is {loader_num}")
            print(metric+"\n")

def test_evaluation_interface():
    """
    evaluation interface test
    """
    import os
    import pickle
    from MNSIM.Interface.utils.init_interface import _init_evaluation_interface
    # change dir to top level
    top_level = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ))
    os.chdir(top_level)
    # init
    evaluation_interface = _init_evaluation_interface(
        "resnet18", "cifar10", "SimConfig.ini", None, -1,
        # "MNSIM/Interface/zoo/cifar10_resnet18_SGD_FIX_TRAIN_FIX_TRAIN.pth", 0
    )
    tile_behavior_list = evaluation_interface.noc_data()
    with open("tmp.pkl", "wb") as f:
        pickle.dump(tile_behavior_list, f)
