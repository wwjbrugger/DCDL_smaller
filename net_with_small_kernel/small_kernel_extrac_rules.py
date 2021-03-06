"""
 train a dicision tree and to perform the sls algorithm with the inout and output_one_picture of a binary layer
"""
import os
os.environ["BLD_PATH"] = "../parallel_sls/bld/Parallel_SLS_shared"
import own_scripts.ripper_by_wittgenstein as ripper
import helper_methods as help
import numpy as np
import SLS_Algorithm as SLS


if __name__ == '__main__':
    print('Extration ')
    Number_of_Product_term = 196
    Maximum_Steps_in_SKS = 10000

    training_set = np.load('data/data_for_SLS.npy')
    label_set = np.load('data/label_SLS.npy')
    kernel = np.load('data/kernel.npy')
    #bias = np.load('data/bias.npy')
    result_conv = np.load('data/result_conv.npy')

    training_set = help.transform_to_boolean(training_set)

    help.visulize_input_data(training_set)
    kernel_width = kernel.shape[0]
    label_for_pic = ['kernel {} '.format(i) for i in range(kernel.shape[3])]

    help.visualize_multi_pic(np.sign(kernel), label_for_pic, "Kernel with sign values")

    values_under_kernel = help.data_in_kernel(training_set, stepsize=2, width=kernel_width)

    kernel_approximation = []
    wittgenstein = []
    for channel in range( label_set.shape[3]):
        training_set_flat, label_set_flat = help.permutate_and_flaten(values_under_kernel, label_set,
                                                                      channel_training=0, channel_label=channel)

        found_formula = \
            SLS.rule_extraction_with_sls_without_validation(training_set_flat, label_set_flat, Number_of_Product_term,
                                                            Maximum_Steps_in_SKS)
        kernel_approximation.append(found_formula)

        label_self_calculated = help.calculate_convolution(values_under_kernel, kernel[:, :, :, channel], result_conv)

        ##df = ripper.np_to_padas(training_set_flat, label_set_flat)
        # rule_set, accuracy = ripper.wittgenstein_ripper(df, 'label', max_rules=Number_of_Product_term)
        # print('rule set: \n', rule_set, '\n accuracy:', accuracy)
        # wittgenstein.append(rule_set)
        # found_formula.pretty_print_formula(' SLS Formula of first kernel')

    for i, formel in enumerate(kernel_approximation):
        formel.number_of_relevant_variabels = kernel_width * kernel_width
        formel.built_plot(0, 'Visualisierung von extrahierter Regel {} '.format(i))

