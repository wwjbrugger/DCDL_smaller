
import numpy as np
import model.two_conv_block_model as model_two_convolution
import accurancy_test_cifar.acc_train_cifar as first
import accurancy_test_cifar.acc_data_generation_cifar as secound
import accurancy_test_cifar.acc_extracting_pictures_cifar as third
import accurancy_test_cifar.acc_reduce_kernel_cifar as fourths
import helper_methods as help

if __name__ == '__main__':
    number_classes_to_predict = 2

    dithering_used= True
    SLS_Training = True        # Should SLS generate Rules
    use_train_set = True         # Should Trainingset be used or test set
    use_label_predicted_from_nn = True     # for prediction in last layer should the output of the nn be used or true label
    one_against_all = 0

    Number_of_disjuntion_term_in_SLS = 40
    Maximum_Steps_in_SKS = 10000
    stride_of_convolution = 2
    runs_of_sls = 1
    shape_of_kernel = (4,4)
    number_of_kernels = 8


    network = model_two_convolution.network_two_convolution(shape_of_kernel=shape_of_kernel, nr_training_itaration=10000,
                                                            stride=stride_of_convolution, number_of_kernel=number_of_kernels,
                                                            number_classes=number_classes_to_predict, input_channels = 3, input_shape = (None,32,32,3))

    first.train_model(network, dithering_used, one_against_all, number_classes_to_predict = number_classes_to_predict)

    secound.acc_data_generation(network, use_train_set)

#    third.visualize_kernel(one_against_all, 'data/kernel_conv_1.npy')

    third.SLS_Conv_1(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS,stride_of_convolution, SLS_Training)

    third.prediction_Conv_1()

    third.SLS_Conv_2(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS,stride_of_convolution, SLS_Training)

    third.prediction_Conv_2()

    third.SLS_dense(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, SLS_Training, use_label_predicted_from_nn)

    third.prediction_dense(use_label_predicted_from_nn, use_train_set)

