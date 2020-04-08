"""
 train a dicision tree and to perform the sls algorithm with the inout and output_one_picture of a binary layer
"""
import os
os.environ["BLD_PATH"] = "../parallel_sls/bld/Parallel_SLS_shared"

import helper_methods as help
import numpy as np
import pickle
from skimage.measure import block_reduce


def SLS_Conv_1 (Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, stride_of_convolution, SLS_Training = True):
    print('SLS Extraction for Convolution 1')

    data = np.load('data/data_reshape.npy')
    label = np.load('data/sign_con_1.npy')
    used_kernel = np.load('data/kernel_conv_1.npy')

    path_to_store= 'data/logic_rules_Conv_1'

    help.sls_convolution(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, stride_of_convolution, data, label,
                                      used_kernel, result=None, path_to_store=path_to_store, SLS_Training=SLS_Training)

def prediction_Conv_1():
    print('Prediction with extracted rules for Convolution 1')

    data_flat = np.load('data/logic_rules_Conv_1_data_flat.npy')
    label = np.load('data/sign_con_1.npy')
    logic_rule = pickle.load(open('data/logic_rules_Conv_1', "rb" ))

    path_to_store_prediction = 'data/prediction_for_conv_1.npy'

    help.prediction_SLS_fast(data_flat, label, logic_rule, path_to_store_prediction)

def max_pooling (data):
    data_after_max_pooling=block_reduce(data, block_size=(1, 2, 2, 1), func=np.max)
    return data_after_max_pooling



def SLS_Conv_2 (Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, stride_of_convolution, SLS_Training):
    print('SLS Extraction for Convolution 2')
    if SLS_Training:
        data = np.load('data/max_pool_1.npy')
    else:
        data = np.load('data/prediction_for_conv_1.npy')
        data = max_pooling(data)
    label = np.load('data/sign_con_2.npy')
    used_kernel = np.load('data/kernel_conv_2.npy')

    path_to_store= 'data/logic_rules_Conv_2'

    help.sls_convolution(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, stride_of_convolution, data, label,
                                      used_kernel, result=None, path_to_store=path_to_store, SLS_Training = SLS_Training)



def prediction_Conv_2():
    print('Prediction with extracted rules for Convolution 2')

    data_flat = np.load('data/logic_rules_Conv_2_data_flat.npy')
    label = np.load('data/sign_con_2.npy')
    found_formula = pickle.load(open('data/logic_rules_Conv_2', "rb"))

    path_to_store_prediction = 'data/prediction_for_conv_2.npy'

    help.prediction_SLS_fast(data_flat, label, found_formula, path_to_store_prediction)

def SLS_dense(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, SLS_Training, use_label_predicted_from_nn ):
    print('SLS Extraction for dense layer')
    if SLS_Training:
        data = np.load('data/sign_con_2.npy')
    else :
        data = np.load('data/prediction_for_conv_2.npy')

    if use_label_predicted_from_nn:
        label = np.load('data/arg_max.npy')
        label = label.reshape((-1, 1)) # to get in the same shape as one_hot_encoded but instesad of [[0,1], ...] it has values [[1], ...]
    else:
        label = np.load('data/data_set_label_train_nn.npy')
    path_to_store= 'data/logic_rules_dense'
    help.sls_dense_net(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, data, label,
                                       path_to_store=path_to_store, SLS_Training= SLS_Training)


def prediction_dense(use_label_predicted_from_nn, Training_set):
    print('Prediction with extracted rules for dense layer')

    flat_data = np.load('data/logic_rules_dense_data_flat.npy')
    if use_label_predicted_from_nn:
        label = np.load('data/arg_max.npy')
    elif not use_label_predicted_from_nn and Training_set:
        label = np.load('data/data_set_label_train_nn.npy')
    elif not Training_set:
        label = np.load('data/data_set_label_test.npy')
    else: raise ValueError('Combination of use_label_predicted_from_nn = {}, Training_set = {}'.format(use_label_predicted_from_nn, Training_set))

    logic_rule = pickle.load(open('data/logic_rules_dense', "rb"))

    path_to_store_prediction = 'data/prediction_dense.npy'

    help.prediction_SLS_fast(flat_data, label, logic_rule, path_to_store_prediction)


def visualize_kernel(one_against_all, path_to_kernel):
    print('Visualistation of the Kernel saved in {} is started '. format(path_to_kernel))
    kernel = np.load(path_to_kernel)
    if kernel.shape[2] >1:
        raise ValueError("Kernel which should be visualized has {} input channel visualization  is only for one channel implemented".format(kernel.shape[2]))
    for channel in range(kernel.shape[3]):
        help.visualize_singel_kernel(kernel[:,:,:,channel],kernel.shape[0] , "Kernel {} from {} for {} againt all \n  path: {}".format(channel, kernel.shape[3], one_against_all, path_to_kernel) )

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

