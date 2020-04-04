
import numpy as np
import model.two_conv_block_model as model_two_convolution
import accurancy_test.acc_train as first
import accurancy_test.acc_data_generation as secound
import accurancy_test.acc_extracting_pictures as third
import accurancy_test.acc_reduce_kernel as fourths
import helper_methods as help

if __name__ == '__main__':
    number_classes_to_predict = 2


    dithering_used= True
    one_against_all = 4

    Number_of_disjuntion_term_in_SLS = 40
    Maximum_Steps_in_SKS = 10000
    stride_of_convolution = 28
    runs_of_sls = 1
    shape_of_kernel = (28,28)

    network = model_two_convolution.network_two_convolution(shape_of_kernel=(4, 4), nr_training_itaration=100,
                                                            stride=2, check_every=16, number_of_kernel=8,
                                                            number_classes=number_classes_to_predict)

    #first.train_model(network, dithering_used, one_against_all, number_classes_to_predict = number_classes_to_predict)

    #secound.acc_data_generation(network)

   # third.visualize_kernel(one_against_all, 'data/kernel_conv_1.npy')
    third.visualize_kernel(one_against_all, 'data/kernel_conv_2.npy')

    third.SLS_Conv_1(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS,stride_of_convolution)

    third.prediction_Conv_1()

    third.SLS_Conv_2(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS,stride_of_convolution)

    third.prediction_Conv_2()

    third.SLS_dense(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS)
    third.prediction_dense()

