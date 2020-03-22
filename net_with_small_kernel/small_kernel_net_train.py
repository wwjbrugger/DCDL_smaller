"""

baseline-bn_before-pool_before for mnist dataset
TODo lauffähigkeit in Konsole checken Problem ModuleNotFoundError: No module named 'model'

"""
from model.one_conv_block_model import stored_network, network
from model.Gradient_helpLayers_convBlock import *
import data.mnist_dataset as md
import own_scripts.dithering as dith
import os


def load_network(name):
    model_l = stored_network(name)
    return model_l


def evaluate(arch):
    print("Evaluating", flush=True)
    restored = load_network(arch)

    size_test_nn = test.shape[0]

    counter = 0  # biased
    acc_sum = 0
    for i in range(0, size_test_nn, 512):
        start = i
        end = min(start + 512, size_test_nn)
        acc = restored.evaluate(test[start:end], label_test[start:end])
        acc_sum += acc
        counter += 1

    print("Test Accuracy", arch.name, acc_sum / counter)



if __name__ == '__main__':
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # CONFIG +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    size_train_nn = 45000
    size_valid_nn = 5000
    dithering_used= False

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # DATA +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    print("Dataset processing", flush=True)
    dataset = md.data()
    num_var = md.num_variables()

    dataset.get_iterator()
    train_nn, label_train_nn = dataset.get_chunk(size_train_nn)
    val, label_val = dataset.get_chunk(size_valid_nn)
    test, label_test = dataset.get_test()

    if dithering_used:
        train_nn = dith.dither_pic(train_nn)
        val = dith.dither_pic(val)
        test = dith.dither_pic(test)
    print("Training", flush=True)


    dcdl_network =network("small_kernel_net", avg_pool=False, real_in= dithering_used,
                         lr=1E-4, batch_size=2**8, activation=binarize_STE,
                         pool_by_stride=False, pool_before=True, pool_after=False,
                         skip=False, pool_skip=False,
                         bn_before=False, bn_after=False, ind_scaling=False, training_itaration= 50
                         )

    print("Start Training")
    dcdl_network.training(train_nn, label_train_nn, val, label_val)

    print("Start evaluate")
    evaluate(dcdl_network)
    print ('\n\n used data sets are saved' )

    np.save('data/data_set_train.npy', train_nn)
    np.save('data/data_set_label_train_nn.npy', train_nn)
    np.save('data/data_set_val.npy', val)
    np.save('data/data_set_label_val.npy', label_val)
    np.save('data/data_set_test.npy', test)
    np.save('data/data_set_label_test.npy', label_test)

    print('end')

