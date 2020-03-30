"""Script to run SLS with the input data of the neural network and the true label of this data"""
import os
os.environ["BLD_PATH"] = "../parallel_sls/bld/Parallel_SLS_shared"
import numpy as np
import SLS_Algorithm as SLS
import helper_methods as help

def SLS_on_diter_data_against_true_label():
    Number_of_Product_term = 100
    Maximum_Steps_in_SKS = 10000


    training_set = np.load('data/data_set_train.npy')
    label_set_one_hot = np.load('data/data_set_label_train_nn.npy')
    label_set = [label[0] for label in label_set_one_hot]
    training_set = help.transform_to_boolean(training_set)
    label_set = help.transform_to_boolean(label_set)
    training_set_flat = np.reshape(training_set, (training_set.shape[0],-1))
    label_set_flat = label_set


    found_formula = \
        SLS.rule_extraction_with_sls_without_validation(training_set_flat, label_set_flat, Number_of_Product_term,
                                                    Maximum_Steps_in_SKS)

    accurancy = (training_set.shape[0]- found_formula.total_error) / training_set.shape[0]
    print("Accurancy of SLS: ", accurancy )
    formel_in_array_code = np.reshape(found_formula.formel_in_arrays_code, (-1, 28, 28))
    reduced_kernel = help.reduce_kernel(formel_in_array_code, mode='norm')
    help.visualize_singel_kernel(np.reshape(reduced_kernel, (-1)), 28,
                                 'norm of all SLS Formel for 4 against all  ')



if __name__ == '__main__':
    SLS_on_diter_data_against_true_label()