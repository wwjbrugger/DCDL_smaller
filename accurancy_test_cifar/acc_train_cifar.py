import random as random

import numpy as np
import data.cifar_dataset as md
import own_scripts.dithering as dith
import helper_methods as help
import model.two_conv_block_model as model_two_convolution
import matplotlib.pyplot as plt

def balance_data_set(data,label, percent_of_major_label_to_keep ):
    unique, counts = np.unique(label[:, 0], return_counts=True)
    print('size_of_set before balancing {} with {}'.format(data.shape[0], dict(zip(unique, counts))))
    index_to_keep = [i for i, one_hot_label in enumerate(label) if
                     one_hot_label[0] == 1 or random.random() < percent_of_major_label_to_keep]
    data = data[index_to_keep]
    label = label[index_to_keep]
    unique, counts = np.unique(label[:, 0], return_counts=True)
    print('size_of_set {} with {}'.format(data.shape[0], dict(zip(unique, counts))))
    return data , label


def prepare_dataset(size_train_nn, size_valid_nn, dithering_used=False, one_against_all=False,
                    percent_of_major_label_to_keep=0.1, number_class_to_predict=10):
    print("Dataset processing", flush=True)
    dataset = md.data()
    dataset.get_iterator()

    train_nn, label_train_nn = dataset.get_chunk(size_train_nn)
    val, label_val = dataset.get_chunk(size_valid_nn)
    test, label_test = dataset.get_test()

    if dithering_used:
        train_nn = dith.dither_pic(train_nn)
        val = dith.dither_pic(val)
        test = dith.dither_pic(test)

    label_train_nn = help.one_class_against_all(label_train_nn, one_against_all,
                                                number_classes_output=number_class_to_predict)
    label_val = help.one_class_against_all(label_val, one_against_all, number_classes_output=number_class_to_predict)
    label_test = help.one_class_against_all(label_test, one_against_all, number_classes_output=number_class_to_predict)

    train_nn, label_train_nn = balance_data_set(train_nn, label_train_nn, percent_of_major_label_to_keep)
    val, label_val = balance_data_set(val,label_val, percent_of_major_label_to_keep)
    test, label_test = balance_data_set(test, label_test, percent_of_major_label_to_keep)

    return train_nn, label_train_nn, val, label_val, test, label_test


def train_model(network, dithering_used, one_against_all, number_classes_to_predict):
    size_train_nn = 5000
    size_valid_nn = 5000
    percent_of_major_label_to_keep = 0.1

    print("Training", flush=True)
    train_nn, label_train_nn, val, label_val, test, label_test = prepare_dataset(size_train_nn, size_valid_nn,
                                                                                 dithering_used, one_against_all,
                                                                                 percent_of_major_label_to_keep=percent_of_major_label_to_keep,

                                                                             number_class_to_predict=number_classes_to_predict)

    print('\n\n used data sets are saved')

    np.save('data/data_set_train.npy', train_nn)
    np.save('data/data_set_label_train_nn.npy', label_train_nn)
    np.save('data/data_set_val.npy', val)
    np.save('data/data_set_label_val.npy', label_val)
    np.save('data/data_set_test.npy', test)
    np.save('data/data_set_label_test.npy', label_test)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    dith.visualize_pic(train_nn, label_train_nn, class_names,
                       "Input pic to train neuronal net with corresponding label", plt.cm.Greys)


    print("Start Training")
    network.training(train_nn, label_train_nn, val, label_val)

    print("\n Start evaluate with validation set ")
    network.evaluate(val, label_val)

    print("\n Start evaluate with test set ")
    network.evaluate(test, label_test)


    print('end')


if __name__ == '__main__':
    dithering_used = True
    one_against_all = 4
    number_classes_to_predict = 2
    network = model_two_convolution.network_two_convolution(shape_of_kernel=(4, 4), nr_training_itaration=100,
                                                            stride=2, check_every=16, number_of_kernel=8,
                                                            number_classes=number_classes_to_predict)
    train_model(network, dithering_used, one_against_all,
                                    number_classes_to_predict=number_classes_to_predict)
