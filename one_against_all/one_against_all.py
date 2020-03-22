import numpy as np
import data.mnist_dataset as md
import own_scripts.dithering as dith
import  helper_methods as help
import model.net_with_one_convolution as model_one_convolution

def prepare_dataset(dithering_used=False, one_against_all=False):
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

    if one_against_all:
        label_train_nn = help.one_class_against_all(label_train_nn, one_against_all)
        label_val = help.one_class_against_all(label_val, one_against_all)
        label_test = help.one_class_against_all(label_test, one_against_all)

    return train_nn, label_train_nn, val, label_val, test, label_test


if __name__ == '__main__':

    size_train_nn = 45000
    size_valid_nn = 5000
    dithering_used= True
    one_against_all = 5

    print("Training", flush=True)
    train_nn, label_train_nn, val, label_val,  test, label_test = prepare_dataset(dithering_used, one_against_all )

    network = model_one_convolution.network_one_convolution(shape_of_kernel= (28,28), nr_training_itaration= 2, stride=28, check_every= 200, number_of_kernel=1)

    print("Start Training")
    network.training(train_nn, label_train_nn, test, label_test)

    print("\n Start evaluate with test set ")
    network.evaluate(test, label_test)
    print("\n Start evaluate with validation set ")
    network.evaluate(val, label_val)

    print ('\n\n used data sets are saved' )

    np.save('data/data_set_train.npy', train_nn)
    np.save('data/data_set_label_train_nn.npy', train_nn)
    np.save('data/data_set_val.npy', val)
    np.save('data/data_set_label_val.npy', label_val)
    np.save('data/data_set_test.npy', test)
    np.save('data/data_set_label_test.npy', label_test)

    print('end')