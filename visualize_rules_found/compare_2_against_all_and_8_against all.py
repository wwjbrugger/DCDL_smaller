import numpy as np

import visualize_rules_found.train_network as first
import visualize_rules_found.data_generation as secound
import visualize_rules_found.extract_logic_rules as third
import visualize_rules_found.reduce_kernel as fourths
import helper_methods as help
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    number_classes_to_predict = 2

    one_against_all = 2

    Maximum_Steps_in_SKS = 1000
    stride_of_convolution = 28
    shape_of_kernel = (28,28)

    K_interval = [30]
    """
    #############################################################################
    input_8 = np.load('data/8_against_all/data_for_SLS.npy')
    output_8_nn = np.load('data/8_against_all/label_SLS.npy')
    used_kernel_8 = np.load('data/8_against_all/kernel.npy')
    help.visualize_singel_kernel(used_kernel_8, 28, 'Used Kernel for 8 against all')

    for Number_of_disjuntion_term_in_SLS in K_interval:
        found_formua = help.sls_convolution(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, stride_of_convolution,
                             data_sign= input_8, label_sign=output_8_nn, used_kernel=used_kernel_8)

        norm = help.reduce_kernel(found_formua[0].formel_in_arrays_code, mode='norm')
        help.visualize_singel_kernel(np.reshape(norm, -1), 28,
                                     '8_against label from nn \n k= {}'.format(Number_of_disjuntion_term_in_SLS), set_vmin_vmax=True)


    #############################################################################
    true_label_8 = np.load('data/8_against_all/data_set_label_train_nn.npy')

    if true_label_8.ndim == 2:
        true_label_8 = np.array([label[0] for label in true_label_8])
        true_label_8 = true_label_8.reshape((true_label_8.shape + (1, 1, 1)))


    for Number_of_disjuntion_term_in_SLS in K_interval:
        found_formua = help.sls_convolution(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS,
                                            stride_of_convolution,
                                            data_sign=input_8, label_sign=true_label_8, used_kernel=used_kernel_8)

        norm = help.reduce_kernel(found_formua[0].formel_in_arrays_code, mode='norm')
        help.visualize_singel_kernel(np.reshape(norm, -1), 28,
                                     '8_against true label k= {}'.format(Number_of_disjuntion_term_in_SLS), set_vmin_vmax=True)

    ##############################################################################

    ########################################################################
"""
    input_2 = np.load('data/2_against_all/data_for_SLS.npy')
    output_2_nn = np.load('data/2_against_all/label_SLS.npy')
    #output_2_nn = np.where(output_2_nn == -1, 1, -1)
    used_kernel_2 = np.load('data/2_against_all/kernel.npy')
    help.visualize_singel_kernel(used_kernel_2, 28, 'Used Kernel for 2 against all')
    for Number_of_disjuntion_term_in_SLS in K_interval:
        found_formua = help.sls_convolution(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS,
                                            stride_of_convolution,
                                            data_sign=input_2, label_sign=output_2_nn, used_kernel=used_kernel_2)
        flatten_data = help.transform_to_boolean(input_2).reshape((input_2.shape[0],-1))
        help.prediction_SLS_fast(flatten_data, output_2_nn, found_formua)
        for k, disjunktion in enumerate(found_formua[0].formel_in_arrays_code):
            help.visualize_singel_kernel(disjunktion,
                                         28, '2_against label from nn \n disjunktion {}'.format(k))

        norm = help.reduce_kernel(found_formua[0].formel_in_arrays_code, mode='norm')
        help.visualize_singel_kernel(np.reshape(norm, -1), 28,
                                     '2_against label from nn \n  k= {}'.format(Number_of_disjuntion_term_in_SLS), set_vmin_vmax=True)
    ###################################################################################


    true_label_2 = np.load('data/2_against_all/data_set_label_train_nn.npy')

    if true_label_2.ndim == 2:
        true_label_2= np.array([label[0] for label in true_label_2])
        true_label_2= true_label_2.reshape((true_label_2.shape + (1, 1, 1)))

    for Number_of_disjuntion_term_in_SLS in K_interval:
        found_formua = help.sls_convolution(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS,
                                            stride_of_convolution,
                                            data_sign=input_2, label_sign=true_label_2, used_kernel=used_kernel_2)

        norm = help.reduce_kernel(found_formua[0].formel_in_arrays_code, mode='norm')
        help.visualize_singel_kernel(np.reshape(norm, -1), 28,
                                     '2_against true label k= {}'.format(Number_of_disjuntion_term_in_SLS), set_vmin_vmax=True)
        flat_2 = input_2.reshape((input_2.shape[0], -1))
       # help.prediction_SLS_fast(flat_2, true_label_2, found_formua, path_to_store_prediction = 'data/prediction_2_true_label')


   # tn, fp, fn, tp = confusion_matrix(true_label_8.reshape(-1), help.transform_to_boolean(output_8_nn.reshape(-1))).ravel()
    #print('For 8 against all tn {}, fp {}, fn {}, tp {}'.format( tn, fp, fn, tp))
    tn, fp, fn, tp = confusion_matrix(true_label_2.reshape(-1), help.transform_to_boolean(output_2_nn.reshape(-1))).ravel()
    print('For 2 against all tn {}, fp {}, fn {}, tp {}'.format(tn, fp, fn, tp))

