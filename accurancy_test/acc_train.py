import random as random

import numpy as np
import data.mnist_fashion as fashion
import data.mnist_dataset as numbers
import data.cifar_dataset as cifar
import own_scripts.dithering as dith
import helper_methods as help
import model.two_conv_block_model as model_two_convolution
import matplotlib.pyplot as plt
import os.path
from os import path


def balance_data_set(data, label, percent_of_major_label_to_keep):
    unique, counts = np.unique(label[:, 0], return_counts=True)
    print('size_of_set before balancing {} with {}'.format(data.shape[0], dict(zip(unique, counts))))
    index_to_keep = [i for i, one_hot_label in enumerate(label) if
                     one_hot_label[0] == 1 or random.random() < percent_of_major_label_to_keep]
    data = data[index_to_keep]
    label = label[index_to_keep]
    unique, counts = np.unique(label[:, 0], return_counts=True)
    print('size_of_set {} with {}'.format(data.shape[0], dict(zip(unique, counts))))
    return data, label


def prepare_dataset(size_train_nn, size_valid_nn, dithering_used, one_against_all,
                    percent_of_major_label_to_keep=None, number_class_to_predict=None, data_set_to_use=None):
    print("Dataset processing", flush=True)
    if data_set_to_use in 'fashion':
        dataset = fashion.data()
    if data_set_to_use in 'numbers':
        dataset = numbers.data()
    if data_set_to_use in 'cifar':
        dataset = cifar.data()
    dataset.get_iterator()

    train_nn, label_train_nn = dataset.get_chunk(size_train_nn)
    val, label_val = dataset.get_chunk(size_valid_nn)
    test, label_test = dataset.get_test()
    if train_nn.ndim == 3:
        train_nn = train_nn.reshape((train_nn.shape + (1,)))
        val = val.reshape((val.shape + (1,)))
        test = test.reshape((test.shape + (1,)))

    if dithering_used:
        train_nn = dith.dither_pic(train_nn)
        val = dith.dither_pic(val)
        test = dith.dither_pic(test)

    label_train_nn = help.one_class_against_all(label_train_nn, one_against_all,
                                                number_classes_output=number_class_to_predict)
    label_val = help.one_class_against_all(label_val, one_against_all, number_classes_output=number_class_to_predict)
    label_test = help.one_class_against_all(label_test, one_against_all, number_classes_output=number_class_to_predict)

    train_nn, label_train_nn = balance_data_set(train_nn, label_train_nn, percent_of_major_label_to_keep)
    val, label_val = balance_data_set(val, label_val, percent_of_major_label_to_keep)
    test, label_test = balance_data_set(test, label_test, percent_of_major_label_to_keep)

    return train_nn, label_train_nn, val, label_val, test, label_test


def train_model(network, dithering_used, one_against_all, data_set_to_use, path_to_use):
    if data_set_to_use in 'cifar':
        size_train_nn = 45000
    else:
        size_train_nn = 55000
    size_valid_nn = 5000
    percent_of_major_label_to_keep = 0.1

    print("Training", flush=True)
    """
    if path.exists(path_to_use['train_data']) and path.exists(path_to_use['train_label']) and path.exists(
            path_to_use['val_data']) and path.exists(path_to_use['val_label']) and path.exists(
            path_to_use['test_data']) and path.exists(path_to_use['test_label']):
        train_nn = np.load(path_to_use['train_data'])
        label_train_nn = np.load(path_to_use['train_label'])
        val = np.load(path_to_use['val_data'])
        label_val = np.load(path_to_use['val_label'])
        test = np.load(path_to_use['test_data'])
        label_test = np.load(path_to_use['test_label'])
        
    else:
    """
    train_nn, label_train_nn, val, label_val, test, label_test = prepare_dataset(size_train_nn, size_valid_nn,
                                                                                 dithering_used, one_against_all,
                                                                                 percent_of_major_label_to_keep=percent_of_major_label_to_keep,
                                                                                 number_class_to_predict=network.classes,
                                                                                 data_set_to_use=data_set_to_use)

    print('\n\n used data sets are saved')

    np.save(path_to_use['train_data'], train_nn)
    np.save(path_to_use['train_label'], label_train_nn)
    np.save(path_to_use['val_data'], val)
    np.save(path_to_use['val_label'], label_val)
    np.save(path_to_use['test_data'], test)
    np.save(path_to_use['test_label'], label_test)

    if data_set_to_use in 'mnist':
        class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    elif data_set_to_use in 'fashion':
        class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    elif data_set_to_use in 'cifar':
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    dith.visualize_pic(train_nn, label_train_nn, class_names,
                       "Input pic to train neuronal net with corresponding label", plt.cm.Greys)

    print("Start Training")
    network.training(train_nn, label_train_nn, val, label_val, path_to_use)

    print("\n Start evaluate with train set ")
    network.evaluate(train_nn, label_train_nn)

    print("\n Start evaluate with validation set ")
    network.evaluate(val, label_val)

    print("\n Start evaluate with test set ")
    network.evaluate(test, label_test)

    print('end')
