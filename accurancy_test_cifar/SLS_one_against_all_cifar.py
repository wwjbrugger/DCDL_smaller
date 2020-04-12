"""Script to run SLS with the input data of the neural network and the true label of this data"""
import os
os.environ["BLD_PATH"] = "../parallel_sls/bld/Parallel_SLS_shared"
import numpy as np
import SLS_Algorithm as SLS
import helper_methods as help
import pickle

def SLS_on_diter_data_against_true_label(use_label_predicted_from_nn):
    Number_of_Product_term = 40
    Maximum_Steps_in_SKS = 10000

#for Number_of_Product_term in range(176,200,25):
    print('Number_of_Product_term: ', Number_of_Product_term)
    print('data/data_set_train.npy is used')
    training_set = np.load('data/data_set_train.npy')
    training_set = help.transform_to_boolean(training_set)
    training_set_flat = np.reshape(training_set, (training_set.shape[0],-1))

    if use_label_predicted_from_nn:
        print('label from neural network are used : data/arg_max.npy' )
        label_set = np.load('data/arg_max.npy')
    else:
        print('tue label from are used : data_set_label_train_nn')
        label_set_one_hot = np.load('data/data_set_label_train_nn.npy')
        label_set = [label[0] for label in label_set_one_hot]
    label_set = help.transform_to_boolean(label_set)
    label_set_flat = label_set
    found_formula = \
        SLS.rule_extraction_with_sls_without_validation(training_set_flat, label_set_flat, Number_of_Product_term,
                                                    Maximum_Steps_in_SKS)

    accurancy = (training_set.shape[0]- found_formula.total_error) / training_set.shape[0]
    print("Accurancy of SLS: ", accurancy, '\n')
    pickle.dump(found_formula, open('data/logic_rules_SLS', "wb"))
    #formel_in_array_code = np.reshape(found_formula.formel_in_arrays_code, (-1, 32, 32, 3))
    #reduced_kernel = help.reduce_kernel(formel_in_array_code, mode='norm')
    #help.visualize_singel_kernel(np.reshape(reduced_kernel, (-1)), 32 ,
    #                             'norm of all SLS Formel for 0 against all \n  k= {}'.format(Number_of_Product_term))
    return found_formula

def predicition (found_formel):

        print('Prediction with extracted rules from SLS for test data')
        test_data = np.load('data/data_set_test.npy')
        test_data_flat = np.reshape(test_data, (test_data.shape[0],-1))
        test_data_flat = help.transform_to_boolean(test_data_flat)

        test_label = np.load('data/data_set_label_test.npy')
        test_label = [label[0] for label in test_label]
        test_label = help.transform_to_boolean(test_label)

        path_to_store_prediction = 'data/SLS_predicition'

        help.prediction_SLS_fast(test_data_flat, test_label, found_formel, path_to_store_prediction)


if __name__ == '__main__':
    use_label_predicted_from_nn = False

    found_formel = SLS_on_diter_data_against_true_label(use_label_predicted_from_nn)
    if not use_label_predicted_from_nn:
        predicition(found_formel)