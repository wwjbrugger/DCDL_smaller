import matplotlib.pyplot as plt

import comparison_DCDL_vs_SLS.train_network


def train_model(network, dithering_used, data_set_to_use, path_to_use):
    if data_set_to_use in 'cifar':
        size_train_nn = 45000
    else:
        size_train_nn = 55000
    size_valid_nn = 5000
    print("Training", flush=True)
    train_nn, label_train_nn, val, label_val, test, label_test = comparison_DCDL_vs_SLS.train_network.prepare_dataset(
        size_train_nn, size_valid_nn, dithering_used, one_against_all=False, data_set_to_use=data_set_to_use)


    print("Start Training")
    _, acc_list, steps = network.training(train_nn, label_train_nn, val, label_val, path_to_use=path_to_use)
    fig, _ = plt.subplots()
    fig.axes[0].scatter(steps, acc_list)
    fig.show()
    print("\n Start evaluate with test set ")
    network.evaluate(test, label_test)
    print("\n Start evaluate with validation set ")
    network.evaluate(val, label_val)


    print('end')
