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


def average_single_label(tabels, dataset):

   # titel = 'Deep_Rule_Set similarity \n   label predicted from NN for train data \n'+ dataset

    x_values = [text.split('_', 1)[0] for text in tabels.columns]
    y_values=tabels.mean(axis = 0).tolist()
    y_stdr = tabels.std(axis = 0).tolist()

    graph_with_error_bar(x_values, y_values, y_stdr, fix_y_axis=True , y_axis_tile='accuracy [%]')



def graph_with_error_bar(x_values, y_values, y_stdr, title = "",x_axis_title="", y_axis_tile='', fix_y_axis= False, ax_out = False ):
    if not ax_out:
        fig, ax = plt.subplots()
    else:
        ax = ax_out
    ax.errorbar(x_values, y_values,
                yerr=y_stdr,
                fmt='o')
    line = 0 * np.array(y_values) + y_values[0]
    plt.plot(x_values, line, '--r')
    ax.set_xlabel(x_axis_title)
    plt.xticks(rotation=-45)
    ax.set_ylabel(y_axis_tile)
    ax.set_title(title)
    if fix_y_axis:
        min = np.min(y_values)
        max = np.max(y_values)
        ax.set_ylim((min - 0.05), (max + 0.05))
    if not ax_out:
        plt.show()
    else:
        plt.savefig('/home/jbrugger/IdeaProjects/DCDL_smaller/accurancy_test/results/accuracy_different_approaches.png',
                dpi=300)


def run():
    for i, path_to_results in enumerate(['data/dither_methods']):
        print('\n\n' + "\033[1m" + path_to_results[5:-8] + "\033[0;0m" + '\n\n')
        table, label = load_tables(path_to_results)
        table = pd.concat(table)
        cols_test = [c for c in table.columns if 'test' in c.lower()]
        df = table[cols_test]
        average_single_label(df, path_to_results[5:-8])
        """"
        cols_train = [c for c in table.columns if 'train' in c.lower()]
        df = table[cols_train]
        average_single_label(df, path_to_results[5:-8])
        """

if __name__ == '__main__':
    run()


