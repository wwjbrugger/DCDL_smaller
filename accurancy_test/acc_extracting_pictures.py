"""
 train a dicision tree and to perform the sls algorithm with the inout and output_one_picture of a binary layer
"""
import os
os.environ["BLD_PATH"] = "../parallel_sls/bld/Parallel_SLS_shared"

import helper_methods as help
import numpy as np


def SLS_Conv_1 (Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, stride_of_convolution):
    print('SLS Extraction for Convolution 1')
    data = np.load('data/data_reshape.npy')
    label = np.load('data/sign_con_1.npy')
    used_kernel = np.load('data/kernel_conv_1.npy')
    path_to_store= 'data/logic_rules_Conv_1'

    help.sls_convolution(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, stride_of_convolution, data, label,
                                      used_kernel, result=None, path_to_store=path_to_store)

def prediction_Conv_1():
    print('Prediction with extracted rules for Convolution 1')
    path_flat_data = 'data/logic_rules_Conv_1_data_flat.npy'
    path_label = 'data/sign_con_1.npy'
    path_logic_rule = 'data/logic_rules_Conv_1'
    path_to_store_prediction = 'data/prediction_for_conv_1.npy'
    help.prediction_SLS_fast(path_flat_data, path_label, path_logic_rule, path_to_store_prediction)


def SLS_Conv_2 (Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, stride_of_convolution):
    print('SLS Extraction for Convolution 2')
    data = np.load('data/max_pool_1.npy')
    label = np.load('data/sign_con_2.npy')
    used_kernel = np.load('data/kernel_conv_2.npy')
    path_to_store= 'data/logic_rules_Conv_2'

    help.sls_convolution(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, stride_of_convolution, data, label,
                                      used_kernel, result=None, path_to_store=path_to_store)



def prediction_Conv_2():
    print('Prediction with extracted rules for Convolution 2')
    path_flat_data = 'data/logic_rules_Conv_2_data_flat.npy'
    path_label = 'data/sign_con_2.npy'
    path_logic_rule = 'data/logic_rules_Conv_2'
    path_to_store_prediction = 'data/prediction_for_conv_2.npy'
    help.prediction_SLS_fast(path_flat_data, path_label, path_logic_rule, path_to_store_prediction)

def SLS_dense(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS ):
    print('SLS Extraction for dense layer')
    data = np.load('data/sign_con_2.npy')
    label = np.load('data/data_set_label_train_nn.npy')
    path_to_store= 'data/logic_rules_dense'

    help.sls_dense_net(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, data, label,
                                       path_to_store=path_to_store)


def prediction_dense():
    print('Prediction with extracted rules for dense layer')
    path_flat_data = 'data/logic_rules_dense_data_flat.npy'
    path_label = 'data/data_set_label_train_nn.npy'
    path_logic_rule = 'data/logic_rules_dense'
    path_to_store_prediction = 'data/prediction_dense.npy'
    help.prediction_SLS_fast(path_flat_data, path_label, path_logic_rule, path_to_store_prediction)


def visualize_kernel(one_against_all, path_to_kernel):
    print('Visualistation of the Kernel saved in {} is started '. format(path_to_kernel))
    kernel = np.load(path_to_kernel)
    if kernel.shape[2] >1:
        raise ValueError("Kernel which should be visualized has {} input channel visualization  is only for one channel implemented".format(kernel.shape[2]))
    for channel in range(kernel.shape[3]):
        help.visualize_singel_kernel(kernel[:,:,:,0],kernel.shape[0] , "Kernel {} from {} for {} againt all \n  path: {}".format(channel, kernel.shape[3], one_against_all, path_to_kernel) )

if __name__ == '__main__':
    Number_of_disjuntion_term_in_SLS = 100
    Maximum_Steps_in_SKS = 10000
    stride_of_convolution = 2
    one_against_all = 2

   # visualize_kernel(one_against_all, 'data/kernel_conv_1.npy')

   # SLS_Conv_1(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS,stride_of_convolution)

    #prediction_Conv_1()

    #SLS_Conv_2(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS,stride_of_convolution)

    #prediction_Conv_2()

    #SLS_dense(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS)
    prediction_dense()

