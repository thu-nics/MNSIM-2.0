# MNSIM Latency Simulation Manual on Network-on-Chip

## MNSIM Modification

To support the latency simulation based on Network-on-Chip (NoC), we update the interface between the MNSIM and the neural networks.
Now, you can use function ''\_init_evaluation_interface'' to initialize the neural network instance in the MNSIM.
![init_interface](https://raw.githubusercontent.com/ILTShade/blob_images/master/images20221101195113.png)
As for the input parameters of this function:

* the first param.: *args.NN*, the network type of the neural network, you can choose from \{lenet, resnet18, alexnet, vgg8, vgg16\} and so on.
* the second param.: *\"imagenet\:*, the dataset type, you can choose from \{cifar10, cifar100, and imagenet\}.
* the third param.: *args.hardware_description*, the hardware description file path.
* the fourth param.: *weight_path*, weight path of the per-trained model.
* the last params.: *args.device*, the device value, -1 is for CPU and others are for GPU device indexes.

The functions of the new \_\_TestInterface are all the same as the original TrainTestInterface.
You can utilize \_\_TestInterface to evaluate the accuracy and get the structure based on the member function ''\_get_structure''.

## Data preparation for NoC simulation

We support the NoC simulation based on MNSIM\_NoC, a upcoming open-source github repository.
And we utilize the MNSIM to generate the data for the NoC simulation, we call it noc_data.
In the new EvaluationInterface (the class of \_\_TestInterface), we add a member function called ''noc\_data'' to get the noc data.
The function goes as follows:
![noc\_data\_process](https://raw.githubusercontent.com/ILTShade/blob_images/master/images20221101200408.png)
It is hard to explain the function in words, and more details can be found in ''MNSIM/Interface/evaluation\.py''.
The function will generate the tile_behavior_list, which describes the behavior of each tile in the NoC.

tile_behavior_list is a list.
Each element in the list corresponds to one and only corresponding tile.
Element is a dictionary. The dictionary has the following key and corresponding values:

* task\_ID: used to mark corresponding tasks and distinguish different tasks
* layer\_ID: used to mark which layer of the corresponding task is the data processed by this tile. The number starts from 0 and increases in turn
* tile\_ID: used to mark which tile this is, and this is in the task\_ID number inside the task\_ID
* target\_tile\_ID: used to mark the tile to which the data of this tile should be *sent* after calculation
* Dependence: which is the key information of the tile, stores the *behavior* and *calculation delay* of the tile. Behavior means what the input data and output data of the tile are, and how long it takes to calculate the output data. Dependency is a list, in which each dependency corresponds to an output of the tile. Each element is a dictionary, consisting of the following keys and values
  * Wait. The wait data is a list, and each element in the list is a tuple. Each tuple contains *nine* data, which are composed of layers in the following format (x, y, start, end, bit, length, image_id, layer_id, in_id). layer_id represents which layer generated this group of data. image_id is used to mark which image of this calculation. in_ids are used to represent different input sources for the merge node. Bit and length are used to represent the bit width and total number of channels of this data. x, y, start, and end represent the horizontal and vertical axes respectively, as well as the start and end points of the channel, which are left open and right closed. The reason why the design is so complex is to ensure that all data has its uniqueness and ensure the correctness of data transmission. Currently, 9 tuples have been changed to 10 tuples, and the last one is tile_id, which can accurately indicate the tile from which the tuple data is generated, and record its number.
  * Output. Output is also a list, but the length is 1, representing the tuple of output data
  * Drop. Drop is also a list, representing tuples that can be taken from the input buffer after the data is completed. To ensure that all buffers are empty after all runs are completed
  * latency, representing the time of completing this operation

All data interactions are conducted in the form of tuples, which are unique within the task.

## Examples

You can follow the following steps to run the example:
![examples](https://raw.githubusercontent.com/ILTShade/blob_images/master/images20221101202049.png)
More details can be found in "MNSIM/Interface/test/test.py".
It should be noticed that, we recommend you to use pickle to dump the noc data to a binary file, and you should open the target file in "wb" mode.
Alternatively, you can simply use the test.py file.

```python
pytest -k test_evaluation_interface -s MNSIM/Interface/test/test.py
```

Before running the example, you should make sure that you have in the main directory of the MNSIM.

## MNSIM_NoC

The explanation of the MNSIM\_NoC is upcoming in github.
