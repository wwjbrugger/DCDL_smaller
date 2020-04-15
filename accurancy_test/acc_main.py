
import numpy as np
import model.two_conv_block_model as model_two_convolution
import accurancy_test.acc_train as first
import accurancy_test.acc_data_generation as secound
import accurancy_test.acc_extracting_pictures as third
import accurancy_test.acc_reduce_kernel as fourths
import helper_methods as help


def get_paths(Input_from_SLS, use_label_predicted_from_nn, Training_set, data_set_to_use ):
    path_to_use = {
        'logs': '/data/{}/logs/'.format(data_set_to_use),
        'store_model': '/data/{}/stored_models/'.format(data_set_to_use),

        'train_data': 'data/{}/train_data.npy'.format(data_set_to_use),
        'train_label': 'data/{}/train_label.npy'.format(data_set_to_use),
        'val_data': 'data/{}/val_data.npy'.format(data_set_to_use),
        'val_label': 'data/{}/val_label.npy'.format(data_set_to_use),
        'test_data': 'data/{}/test_data.npy'.format(data_set_to_use),
        'test_label': 'data/{}/test_label.npy'.format(data_set_to_use),

        'g_reshape': 'data/{}/data_reshape.npy'.format(data_set_to_use),
        'g_sign_con_1' : 'data/{}/sign_con_1.npy'.format(data_set_to_use),
        'g_result_conv_1': 'data/{}/result_conv_1.npy'.format(data_set_to_use),
        'g_kernel_conv_1': 'data/{}/kernel_conv_1.npy'.format(data_set_to_use),
        'g_max_pool_1' : 'data/{}/max_pool_1.npy'.format(data_set_to_use),
        'g_sign_con_2' : 'data/{}/sign_con_2.npy'.format(data_set_to_use),
        'g_result_conv_2': 'data/{}/result_conv_2.npy'.format(data_set_to_use),
        'g_kernel_conv_2': 'data/{}/kernel_conv_2.npy'.format(data_set_to_use),
        'g_arg_max': 'data/{}/arg_max.npy'.format(data_set_to_use),


        'logic_rules_conv_1': 'data/{}/logic_rules_Conv_1'.format(data_set_to_use),
        'flat_data_conv_1': 'data/{}/logic_rules_Conv_1_data_flat.npy'.format(data_set_to_use),
        'prediction_conv_1': 'data/{}/prediction_for_conv_1.npy'.format(data_set_to_use),

        'logic_rules_conv_2': 'data/{}/logic_rules_Conv_2'.format(data_set_to_use),
        'flat_data_conv_2': 'data/{}/logic_rules_Conv_2_data_flat.npy'.format(data_set_to_use),
        'prediction_conv_2': 'data/{}/prediction_for_conv_2.npy'.format(data_set_to_use),

        'logic_rules_dense': 'data/{}/logic_rules_dense'.format(data_set_to_use),
        'flat_data_dense': 'data/{}/logic_rules_dense_data_flat.npy'.format(data_set_to_use),
        'prediction_dense': 'data/{}/prediction_dense.npy'.format(data_set_to_use),

        'logic_rules_SLS':  'data/{}/logic_rules_SLS'.format(data_set_to_use),

    }
    path_to_use['input_conv_1'] = path_to_use['g_reshape']
    path_to_use['label_conv_1'] = path_to_use['g_sign_con_1']
    path_to_use['label_conv_2'] = path_to_use['g_sign_con_2']

    if Training_set:
        path_to_use['input_graph'] = path_to_use['train_data']
    else:
        path_to_use['input_graph'] = path_to_use['test_data']

    if Input_from_SLS:
        path_to_use['input_conv_2'] = 'data/{}/prediction_for_conv_1.npy'.format(data_set_to_use)
        path_to_use['input_dense'] = 'data/{}/prediction_for_conv_2.npy'.format(data_set_to_use)
    else:
        path_to_use['input_conv_2'] = path_to_use['g_max_pool_1']
        path_to_use['input_dense'] = 'data/{}/sign_con_2.npy'.format(data_set_to_use)

    if use_label_predicted_from_nn and Training_set:
        path_to_use['label_dense'] = path_to_use['g_arg_max']
    elif use_label_predicted_from_nn and not Training_set:
        path_to_use['label_dense'] = path_to_use['g_arg_max']

    elif not use_label_predicted_from_nn and Training_set:
        path_to_use['label_dense'] = path_to_use['train_label']
    elif not use_label_predicted_from_nn and not Training_set:
        path_to_use['label_dense'] = path_to_use['test_label']

    return path_to_use
def get_network(data_set_to_use, path_to_use):
    number_classes_to_predict = 2
    stride_of_convolution = 2
    shape_of_kernel = (2, 2)
    number_of_kernels = 8
    name_of_model = '{}_two_conv_3x3'.format(data_set_to_use)
    if data_set_to_use in 'numbers' or  data_set_to_use in'fashion':
        network = model_two_convolution.network_two_convolution(path_to_use, name_of_model= name_of_model,
                                                                shape_of_kernel=shape_of_kernel, nr_training_itaration=1000,
                                                            stride=stride_of_convolution, number_of_kernel=number_of_kernels,
                                                            number_classes=number_classes_to_predict)
    elif data_set_to_use in 'cifar':
        network = model_two_convolution.network_two_convolution(path_to_use, name_of_model = name_of_model,
                                                                shape_of_kernel=shape_of_kernel,
                                                                nr_training_itaration=1500,
                                                                stride=stride_of_convolution,
                                                                number_of_kernel=number_of_kernels,
                                                                number_classes=number_classes_to_predict,
                                                                input_channels=3, input_shape=(None, 32, 32, 3),
                                                                )
    return  shape_of_kernel, stride_of_convolution, number_of_kernels, network


if __name__ == '__main__':

    data_set_to_use = 'cifar'    # 'numbers' or 'fashion'
    dithering_used= True
    NN_Train =False
    SLS_Training = True       # Should SLS generate Rules
    Training_set = True         # Should Trainingset be used or test set
    use_label_predicted_from_nn = True     # for prediction in last layer should the output of the nn be used or true label
    Input_from_SLS = True                  # for extracting the rules ahould the input be the label previously calculated by SLS

    one_against_all = 0

    Number_of_disjuntion_term_in_SLS = 40
    Maximum_Steps_in_SKS = 2500

    path_to_use = get_paths(Input_from_SLS, use_label_predicted_from_nn, Training_set, data_set_to_use)
    shape_of_kernel, stride_of_convolution, number_of_kernels, network = get_network(data_set_to_use, path_to_use)

    if NN_Train:
       first.train_model(network, dithering_used, one_against_all, data_set_to_use, path_to_use)

    secound.acc_data_generation(network, path_to_use)

    #third.visualize_kernel(one_against_all, 'data/kernel_conv_1.npy')

    third.SLS_Conv_1(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS,stride_of_convolution, SLS_Training, path_to_use)

    third.prediction_Conv_1(path_to_use)

    third.SLS_Conv_2(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS,stride_of_convolution, SLS_Training, Input_from_SLS, path_to_use)


    third.prediction_Conv_2(path_to_use)

    third.SLS_dense(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, SLS_Training, path_to_use)

    third.prediction_dense(path_to_use)

