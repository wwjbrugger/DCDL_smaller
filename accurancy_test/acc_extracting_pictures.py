"""
 train a dicision tree and to perform the sls algorithm with the inout and output_one_picture of a binary layer
"""
import os
os.environ["BLD_PATH"] = "../parallel_sls/bld/Parallel_SLS_shared"
import own_scripts.ripper_by_wittgenstein as ripper
import helper_methods as help
import numpy as np
import SLS_Algorithm as SLS

def sls_convolution (Number_of_Product_term, Maximum_Steps_in_SKS, stride_of_convolution) :


    data_reshape = np.load('data/data_reshape.npy')
    sign_con_1 = np.load('data/sign_con_1.npy')
    result_conv_1 = np.load('data/result_conv_1.npy')
    kernel_conv_1 = np.load('data/kernel_conv_1.npy')

    data_reshape = help.transform_to_boolean(data_reshape)
    sign_con_1 = help.transform_to_boolean(sign_con_1)
    kernel_conv_1_width = kernel_conv_1.shape[0]
    values_under_kernel_conv_1 = help.data_in_kernel(data_reshape, stepsize=stride_of_convolution, width=kernel_conv_1_width)

    kernel_conv_1_approximation = []
    for channel in range(sign_con_1.shape[3]):
        print("Ruleextraction for kernel_conv_1 {} ".format(channel))
        data_reshape_flat, sign_con_1_flat = help.permutate_and_flaten(values_under_kernel_conv_1, sign_con_1,
                                                                      channel_training=0, channel_label=channel)

        found_formula = \
            SLS.rule_extraction_with_sls_without_validation(data_reshape_flat, sign_con_1_flat, Number_of_Product_term,
                                                            Maximum_Steps_in_SKS)
        kernel_conv_1_approximation.append(found_formula)

        label_self_calculated = help.calculate_convolution(values_under_kernel_conv_1, kernel_conv_1[:, :, :, channel], result_conv_1)

    """
    for i, formel in enumerate(kernel_conv_1_approximation):
        formel.number_of_relevant_variabels = kernel_conv_1_width * kernel_conv_1_width
        formel.built_plot(0, '{} Visualisierung von extrahierter Regel {} '.format( one_against_all, i))
    """
    formel_in_array_code = []
    for formel in kernel_conv_1_approximation:
        formel_in_array_code.append(np.reshape(formel.formel_in_arrays_code, (-1, kernel_conv_1_width, kernel_conv_1_width)))
    np.save('data/kernel_conv_1_approximation.npy', formel_in_array_code)
    return

def SLS_Conv_1 (Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, stride_of_convolution):
    data = np.load('data/data_reshape.npy')
    label = np.load('data/sign_con_1.npy')
    used_kernel = np.load('data/kernel_conv_1.npy')
    path_to_store= 'data/logic_rules_Conv_1'

    help.sls_convolution(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, stride_of_convolution, data, label,
                                      used_kernel, result=None, path_to_store=path_to_store)

def prediction_Conv_1():
    path_flat_data = 'data/logic_rules_Conv_1_data_flat.npy'
    path_label = 'data/sign_con_1.npy'
    path_logic_rule = 'data/logic_rules_Conv_1'
    path_to_store_prediction = 'data/prediction_for_conv_1.npy'
    help.prediction_SLS(path_flat_data, path_label, path_logic_rule, path_to_store_prediction)


def SLS_Conv_2 (Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, stride_of_convolution):
    data = np.load('data/max_pool_1.npy')
    label = np.load('data/sign_con_2.npy')
    used_kernel = np.load('data/kernel_conv_2.npy')
    path_to_store= 'data/logic_rules_Conv_2'

    help.sls_convolution(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, stride_of_convolution, data, label,
                                      used_kernel, result=None, path_to_store=path_to_store)



def prediction_Conv_2():
    path_flat_data = 'data/logic_rules_Conv_2_data_flat.npy'
    path_label = 'data/sign_con_2.npy'
    path_logic_rule = 'data/logic_rules_Conv_2'
    path_to_store_prediction = 'data/prediction_for_conv_2.npy'
    help.prediction_SLS(path_flat_data, path_label, path_logic_rule, path_to_store_prediction)

def SLS_dense(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS ):
    data = np.load('data/sign_con_2.npy')
    label = np.load('data/data_set_label_train_nn.npy')
    path_to_store= 'data/logic_rules_dense'

    help.sls_dense_net(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, data, label,
                                       path_to_store=path_to_store)


def prediction_dense():
    path_flat_data = 'data/logic_rules_dense_data_flat.npy'
    path_label = 'data/data_set_label_train_nn.npy'
    path_logic_rule = 'data/logic_rules_dense'
    path_to_store_prediction = 'data/prediction_dense.npy'
    help.prediction_SLS(path_flat_data, path_label, path_logic_rule, path_to_store_prediction)


def visualize_kernel(one_against_all, path_to_kernel):
    print('Visualistation of the Kernel is started ')
    kernel = np.load(path_to_kernel)

    label_for_pic = ['kernel {} '.format(i) for i in range(kernel.shape[3])]

    help.visualize_singel_kernel(kernel[:,:,:,0],kernel.shape[0] , "Kernel for {} againt all".format(one_against_all))

if __name__ == '__main__':
    Number_of_disjuntion_term_in_SLS = 100
    Maximum_Steps_in_SKS = 10000
    stride_of_convolution = 2
    one_against_all = 2

   # visualize_kernel(one_against_all, 'data/kernel_conv_1.npy')
    #sls_convolution(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, stride_of_convolution,one_against_all)

   # SLS_Conv_1(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS,stride_of_convolution)

    prediction_Conv_1()

    #SLS_Conv_2(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS,stride_of_convolution)

    #prediction_Conv_2()

    #SLS_dense(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS)
    #prediction_dense()

