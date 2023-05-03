import sys
import os
def add_arguments(parser):
    home_path = os.getcwd()
    # print(home_path)
    SimConfig_path = os.path.join(home_path, "SimConfig.ini")
    weights_file_path = os.path.join(home_path, "cifar10_vgg8_params.pth")
    parser.add_argument("-AutoDelete", "--file_auto_delete", default=True,
                        help="Whether delete the unnecessary files automatically")
    # parser.add_argument("-NoC", "--NoC_computation", default=False,
    #                     help="Whether call booksim to compute the NoC part")
    parser.add_argument("-HWdes", "--hardware_description", default=SimConfig_path,
                        help="Hardware description file location & name, default:/MNSIM_Python/SimConfig.ini")
    parser.add_argument("-Weights", "--weights", default=weights_file_path,
                        help="NN model weights file location & name, default:/MNSIM_Python/cifar10_vgg8_params.pth")
    parser.add_argument("-NN", "--NN", default='vgg8',
                        help="NN model description (name), default: vgg8")
    parser.add_argument("-DisHW", "--disable_hardware_modeling", action='store_true', default=False,
                        help="Disable hardware modeling, default: false")
    parser.add_argument("-DisAccu", "--disable_accuracy_simulation", action='store_true', default=False,
                        help="Disable accuracy simulation, default: false")
    parser.add_argument("-SAF", "--enable_SAF", action='store_true', default=False,
                        help="Enable simulate SAF, default: false")
    parser.add_argument("-Var", "--enable_variation", action='store_true', default=False,
                        help="Enable simulate variation, default: false")
    parser.add_argument("-Rratio", "--enable_R_ratio", action='store_true', default=False,
                        help="Enable simulate the effect of R ratio, default: false")
    parser.add_argument("-FixRange", "--enable_fixed_Qrange", action='store_true', default=False,
                        help="Enable fixed quantization range (max value), default: false")
    parser.add_argument("-DisPipe", "--disable_inner_pipeline", action='store_true', default=False,
                        help="Disable inner layer pipeline in latency modeling, default: false")
    parser.add_argument("-D", "--device", default=0,
                        help="Determine hardware device for simulation, default: CPU")
    parser.add_argument("-DisModOut", "--disable_module_output", action='store_true', default=False,
                        help="Disable module simulation results output, default: false")
    parser.add_argument("-DisLayOut", "--disable_layer_output", action='store_true', default=False,
                        help="Disable layer-wise simulation results output, default: false")
    return parser
