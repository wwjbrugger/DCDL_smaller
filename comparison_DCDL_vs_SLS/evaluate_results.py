from collections import defaultdict
from os import listdir
from os.path import isfile, join
import pickle
import pandas as pd
import statistics
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def return_label(path):
    return int(path[6])

def load_tables(path_to_results):
    label = []
    files = [f for f in listdir(path_to_results) if isfile(join(path_to_results, f))]
    files = sorted(files, key=return_label)
    tables = []
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 150)
    for file_name in files:
        label.append(file_name[:7])
        path = join(path_to_results, file_name)
        pd_file = pickle.load(open(path, "rb"))
        pd_file = pd_file.rename_axis(file_name[:7], axis=1)
        print(pd_file)
        tables.append(pd_file)
    return tables, label


def average_similarity_single_label(tabels, label, dataset):
    titel = 'Deep_Rule_Set similarity \n   label predicted from NN for train data \n'+ dataset
    d = defaultdict(list)
    x_values=[]
    y_values=[]
    y_stdr=[]
    for i in range (len(tabels)):
        d[label[i]].append(tabels[i].at[0, 'Concat'])
    for value in d:
        m, s = calculate_mean_std(d[value])
        y_values.append(m)
        y_stdr.append(s)
        x_values.append(value[-1])
    graph_with_error_bar(x_values, y_values, y_stdr, titel, fix_y_axis=True, x_axis_title="label", y_axis_tile='similarity [%]')

def average_accurancy_on_test_data(tabels, titel, ax):
    if titel:
        titel = 'average accurancy on test data \n {}'.format(titel)
    deep_rule_set = []
    sls_black_box_prediction = []
    sls_black_box_label = []
    neural_net = []
    for table in tabels:
        deep_rule_set.append(table.at[3,'Concat'])
        sls_black_box_prediction.append(table.at[3,'SLS prediction'])
        sls_black_box_label.append(table.at[3, 'SLS train'])
        neural_net.append(np.float64(table.at[3,'Neural network']))
    mean_deep_rule_set, stdr_deep_rule_set = calculate_mean_std(deep_rule_set)
    mean_sls_black_box_prediction, stdr_sls_black_box_prediction = calculate_mean_std(sls_black_box_prediction)
    mean_neural_net, stdr_neural_net = calculate_mean_std(neural_net)
    mean_sls_black_box_label, stdr_sls_black_box_label = calculate_mean_std(sls_black_box_label)



    x_values = ['DRS', 'BB\nPrediction', 'BB\nLabel', 'NN']
    y_values = [mean_deep_rule_set,  mean_sls_black_box_prediction, mean_sls_black_box_label, mean_neural_net]
    y_stdr=[stdr_deep_rule_set, stdr_sls_black_box_prediction, stdr_sls_black_box_label, stdr_neural_net]
    graph_with_error_bar(x_values, y_values, y_stdr, titel, fix_y_axis=True, x_axis_title="",
                         y_axis_tile='accuracy [%]', ax_out=ax)

def concat_minus_SLS_prediction(tabels):
    Concat_minus_SLS_prediction = []
    for pd_file in tabels:
        Concat_minus_SLS_prediction.append(pd_file.at[0, 'Concat'] - pd_file.at[0, 'SLS prediction'])
    return Concat_minus_SLS_prediction


def calculate_mean_std(array):
    mean = statistics.mean(array)
    standard_derivation = statistics.stdev(array)
    print('Mean: {}  '.format(mean))
    print('standard derivation:', standard_derivation)
    return mean, standard_derivation


def graph_with_error_bar(x_values, y_values, y_stdr, title,x_axis_title="", y_axis_tile='', fix_y_axis= False, ax_out = False ):
    if not ax_out:
        fig, ax = plt.subplots()
    else:
        ax = ax_out
    ax.errorbar(x_values, y_values,
                yerr=y_stdr,
                fmt='o')
    ax.set_xlabel(x_axis_title)
    ax.set_ylabel(y_axis_tile)
    ax.set_title(title)
    if fix_y_axis:
        ax.set_ylim(0.5, 1)
    if not ax_out:
        plt.show()


def three_graph_in_one():

    gs = gridspec.GridSpec(4, 4)

    ax1 = plt.subplot(gs[:2, :2])
    ax1.plot(range(0, 10), range(0, 10))

    ax2 = plt.subplot(gs[:2, 2:])
    ax2.plot(range(0, 10), range(0, 10))

    ax3 = plt.subplot(gs[2:4, 1:3])
    ax3.plot(range(0, 10), range(0, 10))

    plt.show()


if __name__ == '__main__':

    x_values = []
    y_values = []
    y_stdr = []

    for i, path_to_results in enumerate(['data/numbers/results', 'data/fashion/results', 'data/cifar/results']):
        print('\n\n' + "\033[1m" + path_to_results[5:-8] + "\033[0;0m" + '\n\n')
        table, label = load_tables(path_to_results)

        average_similarity_single_label(table, label, path_to_results[5:-8])
        #average_accurancy_on_test_data(table,path_to_results[5:-8], position[i])
        Conc_minus_SLS_pred = concat_minus_SLS_prediction(table)
        mean, stdr = calculate_mean_std(Conc_minus_SLS_pred)
        x_values.append(path_to_results[5:-8])
        y_values.append(mean)
        y_stdr.append(stdr)
    graph_with_error_bar(x_values, y_values, y_stdr, 'Deep_Rule_Set - SLS_Black_Box_Prediction \n label predicted from NN for train data', x_axis_title="Data set", y_axis_tile='similarity[%]' )

    gs = gridspec.GridSpec(4, 4)
    position = [plt.subplot(gs[:2, :2]), plt.subplot(gs[:2, 2:]), plt.subplot(gs[2:4, 1:3])]
    for i, path_to_results in enumerate(['data/numbers/results', 'data/fashion/results', 'data/cifar/results']):
        print('\n\n' + "\033[1m" + path_to_results[5:-8] + "\033[0;0m" + '\n\n')
        table, label = load_tables(path_to_results)
        average_accurancy_on_test_data(table, '', position[i])
        #average_accurancy_on_test_data(table,path_to_results[5:-8], position[i])


    plt.tight_layout()
    #plt.figure(dpi=300)
    plt.savefig('/home/jbrugger/IdeaProjects/DCDL_smaller/comparison_DCDL_vs_SLS/results/accuracy_different_approaches.png', dpi = 300)
    plt.show()