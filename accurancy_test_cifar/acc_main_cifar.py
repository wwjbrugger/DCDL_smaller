
import numpy as np
import model.two_conv_block_model as model_two_convolution
import accurancy_test_cifar.acc_train_cifar as first
import accurancy_test_cifar.acc_data_generation_cifar as secound
import accurancy_test_cifar.acc_extracting_pictures_cifar as third
import accurancy_test_cifar.acc_reduce_kernel_cifar as fourths
import helper_methods as help

def get_paths(Input_from_SLS, use_label_predicted_from_nn, Training_set ):
    path_to_use = {
        'input_conv_1': 'data/data_reshape.npy',
        'label_conv_1': 'data/sign_con_1.npy',
        'kernel_conv_1': 'data/kernel_conv_1.npy',
        'logic_rules_conv_1': 'data/logic_rules_Conv_1',
        'flat_data_conv_1': 'data/logic_rules_Conv_1_data_flat.npy',
        'prediction_conv_1': 'data/prediction_for_conv_1.npy',
        'label_conv_2 ': 'data/sign_con_2.npy',
        'kernel_conv_2': 'data/kernel_conv_2.npy',
        'logic_rules_conv_2': 'data/logic_rules_Conv_2',
        'flat_data_conv_2': 'data/logic_rules_Conv_2_data_flat.npy',
        'prediction_conv_2': 'data/prediction_for_conv_2.npy',
        'logic_rules_dense': 'data/logic_rules_dense',
        'flat_data_dense': 'data/logic_rules_dense_data_flat.npy',
        'prediction_dense': 'data/prediction_dense.npy'

    }
    if Input_from_SLS:
        path_to_use['input_conv_2'] = 'data/prediction_for_conv_1.npy'
        path_to_use['input_dense'] = 'data/prediction_for_conv_2.npy'
    else:
        path_to_use['input_conv_2'] = 'data/max_pool_1.npy'
        path_to_use['input_dense'] = 'data/sign_con_2.npy'

    if use_label_predicted_from_nn:
        path_to_use['label_dense'] = 'data/arg_max.npy'
    elif use_label_predicted_from_nn and Training_set:
        path_to_use['label_dense'] = 'data/data_set_label_train_nn.npy'
    elif not Training_set:
        path_to_use['label_dense'] = 'data/data_set_label_test.npy'
    return path_to_use


if __name__ == '__main__':
    number_classes_to_predict = 2
    data_set_to_use = 'cifar'
    dithering_used= True
    SLS_Training = False        # Should SLS generate Rules
    Training_set = True       # Should Trainingset be used or test set
    use_label_predicted_from_nn = True     # for prediction in last layer should the output of the nn be used or true label
    Input_from_SLS = True
    one_against_all = 0

    Number_of_disjuntion_term_in_SLS = 40
    Maximum_Steps_in_SKS = 10000
    stride_of_convolution = 2
    shape_of_kernel = (4,4)
    number_of_kernels = 8
    path_to_use = get_paths(Input_from_SLS, use_label_predicted_from_nn, Training_set)

    network = model_two_convolution.network_two_convolution(shape_of_kernel=shape_of_kernel, nr_training_itaration=1500,
                                                            stride=stride_of_convolution, number_of_kernel=number_of_kernels,
                                                            number_classes=number_classes_to_predict, input_channels = 3, input_shape = (None,32,32,3))

    if SLS_Training:
        first.train_model(network, dithering_used, one_against_all, number_classes_to_predict = number_classes_to_predict)

    secound.acc_data_generation(network, Training_set)

#    third.visualize_kernel(one_against_all, 'data/kernel_conv_1.npy')

    third.SLS_Conv_1(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS,stride_of_convolution, SLS_Training, path_to_use)

    third.prediction_Conv_1(path_to_use)

    third.SLS_Conv_2(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS,stride_of_convolution, SLS_Training, Input_from_SLS)

    third.prediction_Conv_2(path_to_use)

    third.SLS_dense(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS,stride_of_convolution, SLS_Training, path_to_use)

    third.prediction_dense(path_to_use)

