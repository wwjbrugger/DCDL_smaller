"""Script to run SLS with the input data of the neural network and the true label of this data"""
import os
os.environ["BLD_PATH"] = "../parallel_sls/bld/Parallel_SLS_shared"
import numpy as np
import SLS_Algorithm as SLS
import helper_methods as help
import pickle

import accurancy_test.acc_data_generation as secound

def SLS_on_diter_data_against_true_label(path_to_use):
    Number_of_Product_term = 40
    Maximum_Steps_in_SKS = 2500

    #for Number_of_Product_term in range(176,200,25):
    print('\n\n \t\t sls run ')
    print('Number_of_Product_term: ', Number_of_Product_term)
    print(path_to_use['input_graph'],' is used as input')
    training_set = np.load(path_to_use['input_graph'])
    training_set = help.transform_to_boolean(training_set)
    training_set_flat = np.reshape(training_set, (training_set.shape[0],-1))
    print(path_to_use['label_dense'], 'is used as label')
    label_set = np.load(path_to_use['label_dense'])
    if label_set.ndim == 2:
        label_set = [label[1] for label in label_set]
    label_set = help.transform_to_boolean(label_set)
    label_set_flat = label_set
    found_formula = \
        SLS.rule_extraction_with_sls_without_validation(training_set_flat, label_set_flat, Number_of_Product_term,
                                                    Maximum_Steps_in_SKS)

    accurancy = (training_set.shape[0]- found_formula.total_error) / training_set.shape[0]
    print("Accurancy of SLS: ", accurancy, '\n')
    pickle.dump(found_formula, open('data/logic_rules_SLS', "wb"))
    if 'cifar' not in path_to_use['logs']:
        formel_in_array_code = np.reshape(found_formula.formel_in_arrays_code, (-1, 28, 28))
        reduced_kernel = help.reduce_kernel(formel_in_array_code, mode='norm')
        help.visualize_singel_kernel(np.reshape(reduced_kernel, (-1)), 28,
                                     'norm of all SLS Formel for 0 against all \n  k= {}'.format(Number_of_Product_term))
    return found_formula

def predicition (found_formel, path_to_use):

        print('Prediction with extracted rules from SLS for test data')
        print('Input data :', path_to_use['test_data'])
        print('Label :', path_to_use['test_label'])

        test_data = np.load(path_to_use['test_data'])
        test_data_flat = np.reshape(test_data, (test_data.shape[0],-1))
        test_data_flat = help.transform_to_boolean(test_data_flat)

        test_label = np.load(path_to_use['test_label'])
        test_label = [label[1] for label in test_label]
        test_label = help.transform_to_boolean(test_label)

        path_to_store_prediction = path_to_use['logic_rules_SLS']

        help.prediction_SLS_fast(test_data_flat, test_label, found_formel, path_to_store_prediction)


if __name__ == '__main__':
    import accurancy_test.acc_main as main
    use_label_predicted_from_nn = True
    Input_from_SLS = None
    Training_set = True
    data_set_to_use = 'cifar'
    path_to_use = main.get_paths(Input_from_SLS, use_label_predicted_from_nn, Training_set, data_set_to_use)
    _, _, _, network = main.get_network(data_set_to_use, path_to_use)
    secound.acc_data_generation(network, path_to_use)

    found_formel = SLS_on_diter_data_against_true_label(path_to_use)
    if not use_label_predicted_from_nn:
        predicition(found_formel)