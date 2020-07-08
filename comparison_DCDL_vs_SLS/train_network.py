import random as random

import numpy as np
import data.mnist_fashion as fashion
import data.mnist_dataset as numbers
import data.cifar_dataset as cifar
import helper_methods as help
import matplotlib.pyplot as plt


def balance_data_set(data, label, percent_of_major_label_to_keep):
    unique, counts = np.unique(label[:, 0], return_counts=True)
    print('size_of_set before balancing {} with {}'.format(data.shape[0], dict(zip(unique, counts))))
    index_minority = [i for i, one_hot_label in enumerate(label) if
                     one_hot_label[0] == 1]
    index_majority = [[i for i, one_hot_label in enumerate(label) if
                     one_hot_label[0] == 1]]
    index_to_keep = [i for i, one_hot_label in enumerate(label) if
                     one_hot_label[0] == 1 or random.random() < percent_of_major_label_to_keep]
    data = data[index_to_keep]
    label = label[index_to_keep]
    unique, counts = np.unique(label[:, 0], return_counts=True)
    print('size_of_set {} with {}'.format(data.shape[0], dict(zip(unique, counts))))
    return data, label


def balance_data_set_II(data_list, label_list, number_class_to_predict, seed = None):
    number_classes = label_list.shape[1]
    balanced_dataset = []
    balanced_label = []
    num_elements_minority = int(label_list.shape[0] / number_classes/ (number_classes-1))
    np.random.seed(seed)
    for i in range(number_classes):
        index = [j for j, one_hot_label in enumerate(label_list) if
                          one_hot_label[i] == 1]
        if i == number_class_to_predict:
            balanced_dataset.append(data_list[index])
            balanced_label.append(label_list[index])
        else:
            sub_sample_index = random.sample(index, num_elements_minority)
            balanced_dataset.append(data_list[sub_sample_index])
            balanced_label.append(label_list[sub_sample_index])
    balanced_dataset = np.concatenate(balanced_dataset, axis=0)
    balanced_label = np.concatenate(balanced_label, axis=0)
    idx = np.random.permutation(len(balanced_label))
    x, y = balanced_dataset[idx], balanced_label[idx]

    return x, y





def prepare_dataset(size_train_nn, size_valid_nn, dithering_used, one_against_all, data_set_to_use=None):
    print("Dataset processing", flush=True)
    if data_set_to_use in 'fashion':
        dataset = fashion.data(dither_method=dithering_used)
    if data_set_to_use in 'mnist':
        dataset = numbers.data(dither_method=dithering_used)
    if data_set_to_use in 'cifar':
        dataset = cifar.data(dither_method=dithering_used)
    dataset.get_iterator()

    train_nn, label_train_nn = dataset.get_chunk(size_train_nn)
    val, label_val = dataset.get_chunk(size_valid_nn)
    test, label_test = dataset.get_test(dither_method=dithering_used)
    # if train_nn.ndim == 3:
    #     train_nn = train_nn.reshape((train_nn.shape + (1,)))
    #     val = val.reshape((val.shape + (1,)))
    #     test = test.reshape((test.shape + (1,)))

    if data_set_to_use in 'mnist':
        class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    elif data_set_to_use in 'fashion':
        class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    elif data_set_to_use in 'cifar':
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    help.visualize_pic(train_nn, label_train_nn, class_names,
                       "Train data dither method: {}". format(dithering_used), plt.cm.Greys)
    help.visualize_pic(test, label_test, class_names,
                       "Test data dither method: {}". format(dithering_used), plt.cm.Greys)
    
    # if convert_to_grey:
    #     train_nn = help.convert_to_grey(train_nn)
    #     val = help.convert_to_grey(val)
    #     test = help.convert_to_grey(test)
    #     help.visualize_pic(train_nn, label_train_nn, class_names,
    #                        " pic in gray ", plt.cm.Greys)


    # if dithering_used:
    #     train_nn = dith.error_diffusion_dithering(train_nn, dithering_used)
    #     val = dith.error_diffusion_dithering(val, dithering_used)
    #     test = dith.error_diffusion_dithering(test, dithering_used)
    #     help.visualize_pic(train_nn, label_train_nn, class_names,
    #                        " pic after dithering", plt.cm.Greys)

    train_nn, label_train_nn = balance_data_set_II(train_nn, label_train_nn, one_against_all)
    val, label_val = balance_data_set_II(val, label_val, one_against_all)
    test, label_test = balance_data_set_II(test, label_test, one_against_all)

    if one_against_all:
        label_train_nn = help.one_class_against_all(label_train_nn, one_against_all)
        label_val = help.one_class_against_all(label_val, one_against_all)
        label_test = help.one_class_against_all(label_test, one_against_all)



    #train_nn, label_train_nn = balance_data_set(train_nn, label_train_nn, percent_of_major_label_to_keep)
    #val, label_val = balance_data_set(val, label_val, percent_of_major_label_to_keep)
    #test, label_test = balance_data_set(test, label_test, percent_of_major_label_to_keep)

    #help.visualize_pic(train_nn, label_train_nn, class_names,
    #                  " pic how they are feed into net", plt.cm.Greys)

    return train_nn, label_train_nn, val, label_val, test, label_test


def train_model(network, dithering_used, one_against_all, data_set_to_use, path_to_use, convert_to_grey, results, dither_frame):
    if data_set_to_use in 'cifar':
        size_train_nn = 45000
    else:
        size_train_nn = 55000
    size_valid_nn = 5000
    percent_of_major_label_to_keep = 0.1

    print("Training", flush=True)


    train_nn, label_train_nn, val, label_val, test, label_test = prepare_dataset(size_train_nn, size_valid_nn,
                                                                                 dithering_used, one_against_all,
                                                                                 percent_of_major_label_to_keep=percent_of_major_label_to_keep,
                                                                                 number_class_to_predict=network.classes,
                                                                                 data_set_to_use=data_set_to_use, convert_to_grey = convert_to_grey)

    print('\n\n used data sets are saved')

    np.save(path_to_use['train_data'], train_nn)
    np.save(path_to_use['train_label'], label_train_nn)
    np.save(path_to_use['val_data'], val)
    np.save(path_to_use['val_label'], label_val)
    np.save(path_to_use['test_data'], test)
    np.save(path_to_use['test_label'], label_test)



    print("Start Training")
    network.training(train_nn, label_train_nn, val, label_val, path_to_use)

    print("\n Start evaluate with train set ")
    evaluation_result = network.evaluate(train_nn, label_train_nn)
    results.at[1, 'Neural network'] = evaluation_result
    dither_frame.at[0, '{}_Train'.format(dithering_used)] = evaluation_result

    print("\n Start evaluate with validation set ")
    network.evaluate(val, label_val)

    print("\n Start evaluate with test set ")
    evaluation_result = network.evaluate(test, label_test)
    results.at[3, 'Neural network'] = evaluation_result
    dither_frame.at[0, '{}_Test'.format(dithering_used)] = evaluation_result



    print('end')

def train_further(network):
    network.training
