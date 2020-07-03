import model.comparison_DCDL_vs_ResNet_eighteen.eighteen_block_model as eighteen_block_model
import dcdl_vs_resnet_eighteen.train_network as first
import sys




def get_paths(data_set_to_use):
    path_to_use = {
        'logs': 'tmp/{}/logs/'.format(data_set_to_use),
        'store_model': 'tmp/{}/stored_models/'.format(data_set_to_use),
        'results': 'tmp/{}/results/'.format(data_set_to_use)
    }
    return path_to_use


def get_network(data_set_to_use, path_to_use):
    number_classes_to_predict = 10
    stride_of_convolution = 2
    shape_of_kernel = (3, 3)
    number_of_kernels = 64
    name_of_model = '{}_eighteen_block_model'.format(data_set_to_use)
    if data_set_to_use in 'numbers' or data_set_to_use in 'fashion':
        network = eighteen_block_model.network(path_to_use, name_of_model=name_of_model,
                                                                shape_of_kernel=shape_of_kernel,
                                                                nr_training_itaration=2000,
                                                                stride=stride_of_convolution,
                                                                number_of_kernel=number_of_kernels,
                                                                number_classes=number_classes_to_predict)
    elif data_set_to_use in 'cifar':
        input_channels = 3
        input_shape = (None, 32, 32, 3)
        network = eighteen_block_model.network(path_to_use, name_of_model=name_of_model,
                                                                shape_of_kernel=shape_of_kernel,
                                                                nr_training_itaration=50,
                                                                stride=stride_of_convolution,
                                                                number_of_kernel=number_of_kernels,
                                                                number_classes=number_classes_to_predict,
                                                                input_channels=input_channels, input_shape=input_shape,

                                                                )

    return shape_of_kernel, stride_of_convolution, number_of_kernels, network

if __name__ == '__main__':
    if len(sys.argv) > 1:

        print("used Dataset: ", sys.argv [1])
        if (sys.argv[1] in 'numbers') or (sys.argv[1] in'fashion') or (sys.argv[1] in 'cifar'):
            data_set_to_use = sys.argv [1]

        else:
            raise ValueError('You choose a dataset which is not supported. \n Datasets which are allowed are numbers(Mnist), fashion(Fashion-Mnist) and cifar')
    else:
        data_set_to_use = 'cifar'  # 'numbers' or 'fashion'
    dithering_used = 'floyd-steinberg'
    path_to_use = get_paths( data_set_to_use)

    shape_of_kernel, stride_of_convolution, number_of_kernels, network = get_network(data_set_to_use, path_to_use)
    first.train_model(network, dithering_used, data_set_to_use,path_to_use)
