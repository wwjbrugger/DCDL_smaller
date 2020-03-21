"""

One Layer neural network with mnist data dithered

"""


import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from model.one_conv_block_model import stored_network, network
from model.Gradient_helpLayers_convBlock import *
import data.mnist_dataset as md
import sys


def load_network(name):
    model_l = stored_network(name)
    return model_l


def evaluate(arch, test, label_test):
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
def visulize_pic(pic_array, label_array, class_names, titel):
    """ show 10 first  pictures """
    fig = plt.figure()
    st = plt.suptitle(titel, fontsize=14)
    st.set_y(1)

    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(pic_array[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[np.argmax(label_array[i])])

    plt.tight_layout()
    plt.show()

def dither_pic (pic_array, values_max_1 = True):
    """ dither pictures """
    for i, pic in tqdm(enumerate(pic_array)):
        if values_max_1:
            picture_grey = Image.fromarray(pic*255)
        else:
            picture_grey = Image.fromarray(pic )
        picture_dither = picture_grey.convert("1")
        pic_array[i] = picture_dither
    #pic_array = np.array(pic_array)

def mnist_one_layer(dither = True ):

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # CONFIG +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    size_train_nn = 45000
    size_valid_nn = 5000
    class_names = ['T-shirt/top', 'Trousers', 'Pullover', 'Dress',
                       'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # class_names = ['zero', 'one', 'two', 'three', 'four', 'five',
    #               'six', 'seven', 'eight', 'nine']

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



    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # visulize +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    visulize_pic(train_nn, label_train_nn, class_names, "Input pictures")
    if dither:
        print("Pictures in trainset are dithered")
        dither_pic(train_nn)
        print("Pictures in test set are dithered")
        dither_pic(test)
        visulize_pic(train_nn, label_train_nn, class_names, "Dithered pictures")
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # PIPELINE +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    print("Training", flush=True)
    archs = []

    if dither:
            archs.append(network("baseline-bn_before-pool_before", avg_pool=False, real_in=True,
                             lr=1E-4, batch_size=2**8, activation=binarize_STE,
                             pool_by_stride=False, pool_before=True, pool_after=False,
                             skip=False, pool_skip=False,
                             bn_before=True, bn_after=False, ind_scaling=False
                             ))
    else:
        archs.append(network("baseline-bn_before-pool_before", avg_pool=False, real_in=False,
                             lr=1E-4, batch_size=2 ** 8, activation=binarize_STE,
                             pool_by_stride=False, pool_before=True, pool_after=False,
                             skip=False, pool_skip=False,
                             bn_before=True, bn_after=False, ind_scaling=False
                             ))
    print("Start Training")
    archs[-1].training(train_nn, label_train_nn, val, label_val)

    print("Start evaluate")
    evaluate(archs[-1],test, label_test)

mnist_one_layer(dither=False)