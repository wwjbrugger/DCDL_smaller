"""
In GReyscale Pictures high values are representet by white
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import data.mnist_dataset as md
import data.cifar_dataset as cif


def visualize_pic(pic_array, label_array, class_names, titel, colormap):
    """ show 10 first  pictures """
    for i in range(20):
        plt.subplot(5, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if pic_array.shape[3] == 1:
            plt.imshow(pic_array[i,:,:,0], cmap=colormap)
        elif pic_array.shape[3] == 3:
            plt.imshow(pic_array[i], cmap=colormap)
        else:
            raise ValueError('Picture should have 1 or 3 channels not'.format(pic_array.shape[3]))
        plt.xlabel(class_names[np.argmax(label_array[i])])

    st = plt.suptitle(titel, fontsize=14)
    st.set_y(1)
    plt.tight_layout()
    plt.show()


def dither_pic(pic_array, values_max_1=True):
    """ dither pictures """
    for channel in range(pic_array.shape[3]):
        for i, pic in tqdm(enumerate(pic_array[:, :, :,channel])):
            if values_max_1:
                picture_grey = Image.fromarray(pic * 255)
            else:
                picture_grey = Image.fromarray(pic)
            picture_dither = picture_grey.convert("1")
            picture_dither_np = np.array(picture_dither)
            pic_array[i,:,:,channel] = np.where(picture_dither_np, 1, -1)
    return pic_array



    # pic_array = np.array(pic_array)


def mnist():
    mnist_train_data = md.data().train
    mnist_train_label = md.data().label_train
    dither_pic(mnist_train_data)
    class_names = ['T-shirt/top', 'Trousers', 'Pullover', 'Dress',
                   'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    visualize_pic(mnist_train_data, mnist_train_label, class_names, "Mnist_dither", plt.cm.Greys)


def cifar_rgb():
    cifar_train_data = cif.data().train
    cifar_train_label = cif.data().label_train
    class_names = ['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks']
    color = [' red', ' green', ' blue']
    plt_maps = [plt.cm.Greys, plt.cm.Greys, plt.cm.Greys]
    # plt_maps = [plt.cm.Reds, plt.cm.Greens, plt.cm.Blues]

    visualize_pic(cifar_train_data, cifar_train_label, class_names, 'Cifar dataset full', plt.cm.Greys)

    for channel in range(cifar_train_data.shape[3]):
        visualize_pic(cifar_train_data[:, :, :, channel], cifar_train_label, class_names,
                      'Cifar_dataset' + color[channel], plt_maps[channel])

    for channel in range(cifar_train_data.shape[3]):
        dither_pic(cifar_train_data[:, :, :, channel])

    visualize_pic(cifar_train_data, cifar_train_label, class_names, 'Cifar dataset full', plt.cm.Greys)

    for channel in range(cifar_train_data.shape[3]):
        visualize_pic(cifar_train_data[:, :, :, channel], cifar_train_label, class_names,
                      'Cifar_dataset' + color[channel], plt_maps[channel])


def cifar_grey():
    cifar_train_data = cif.data().train
    cifar_train_label = cif.data().label_train
    class_names = ['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks']
    cifar_train_data_grey = []
    visualize_pic(cifar_train_data, cifar_train_label, class_names, 'Cifar dataset full', plt.cm.Greys)

    print('Converting Dataset to greyscale picture ')
    for i, pic in tqdm(enumerate(cifar_train_data)):
        pic_255 = (pic*255).copy()
        pic_255= pic_255.astype(np.uint8)
        picture = Image.fromarray(pic_255,  mode="RGB")
        picture_grey = picture.convert('L')
        cifar_train_data_grey.append(np.asarray(picture_grey)/255)
    cifar_train_data_grey = np.array(cifar_train_data_grey)

    visualize_pic(cifar_train_data_grey, cifar_train_label, class_names, 'Cifar greyscale', plt.cm.Greys)
    dither_pic(cifar_train_data_grey)
    visualize_pic(cifar_train_data_grey, cifar_train_label, class_names, 'Cifar greyscale dither', plt.cm.Greys)

if __name__ == '__main__':

    cifar_rgb()
#cifar_grey()
# mnist()
