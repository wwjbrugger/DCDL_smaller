import pickle
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import helper_methods as help


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
        pd_file['label'] = file_name[:7]
        pd_file = pd_file.rename_axis(file_name[:7], axis=1)
        #print(pd_file)
        tables.append(pd_file)
    return tables, label



def result_single_label(dataset):
    path_cifar = 'data/dither_methods/' + dataset
    for i, path_to_results in enumerate([path_cifar]):
        table, label = load_tables(path_to_results)
        table = pd.concat(table)
        for i in range(10):
            titel = dataset + ' Label ' + str(i)
            sub_table = table[table['label'].str.contains(str(i))]
            cols_test = [c for c in sub_table.columns if 'test' in c.lower()]
            df = sub_table[cols_test]
            x_values = [text.split('_', 1)[0] for text in df.columns]
            y_values=df.mean(axis = 0).tolist()
            y_stdr = df.std(axis = 0).tolist()
            fig, ax = plt.subplots()
            save_path = 'results/dither/single_label/' + dataset + '/label_' + str(i)
            help.graph_with_error_bar(x_values, y_values, y_stdr, title=titel,fix_y_axis=True , y_axis_tile='accuracy [%]', ax_out=ax,
                                      save_path=save_path)

def result_dataset(dataset):
    path_cifar = 'data/dither_methods/' + dataset
    for i, path_to_results in enumerate([path_cifar]):
        table, label = load_tables(path_to_results)
        table = pd.concat(table)
        titel = 'average perfomance on {}'.format(dataset)
        cols_test = [c for c in table.columns if 'test' in c.lower()]
        df = table[cols_test]
        x_values = [text.split('_', 1)[0] for text in df.columns]
        y_values = df.mean(axis=0).tolist()
        y_stdr = df.std(axis=0).tolist()
        fig, ax = plt.subplots()
        save_path = 'results/dither/' + dataset + '_average_performance.png'
        help.graph_with_error_bar(x_values, y_values, y_stdr, title=titel, fix_y_axis=True,
                                  y_axis_tile='accuracy [%]', ax_out=ax,
                                  save_path=save_path)

def t_statistik(dataset):
    path_cifar = 'data/dither_methods/' + dataset
    for i, path_to_results in enumerate([path_cifar]):
        table, label = load_tables(path_to_results)
        table = pd.concat(table)
        cols_test = [c for c in table.columns if 'test' in c.lower()]
        table = table[cols_test]
        for i in range(len(table.columns)):
            for j in range(i + 1, len(table.columns), 1):
                col_1 = table.iloc[:, i]
                col_1_name = table.columns[i]
                col_2 = table.iloc[:, j]
                col_2_name = table.columns[j]
                # Spaltennamen für DCDL und Neuronales Netz raussuchen
                t_statistic, two_tailed_p_test = stats.ttest_ind(col_1, col_2)
                if two_tailed_p_test > 0.05:
                    print('{} and {} can have th same mean p_value = {}'.format(col_1_name, col_2_name,
                                                                          two_tailed_p_test))
                else:
                    print('Reject that {} and {}  have the same mean p_value = {}'.format(col_1_name, col_2_name,
                                                                                      two_tailed_p_test))
            print('\n')




def run():
   #result_single_label('cifar')
   #result_dataset('cifar')
   t_statistik('cifar')
if __name__ == '__main__':
    run()


