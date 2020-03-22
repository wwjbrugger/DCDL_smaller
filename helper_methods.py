import numpy as np
import matplotlib.pyplot as plt

def calculate_padding_parameter (shape_input_pic, filter_size, stride,  ):
    in_height = shape_input_pic[1]
    in_width = shape_input_pic[2]
    out_height = np.ceil(float(in_height) / float(stride))
    out_width = np.ceil(float(in_width) / float(stride))

    pad_along_height = np.max((out_height - 1) * stride +
                           filter_size - in_height, 0)
    pad_along_width = np.max((out_width - 1) * stride +
                          filter_size - in_width, 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return ((0,0), (int(pad_top),int(pad_bottom)), (int(pad_left), int(pad_right)),  (0,0))


def data_in_kernel(arr, stepsize=2, width=4):  # kernel views
    # Need padding for convolution
    #npad = ((0, 0), (1, 1), (1, 1), (0, 0))
    #npad = ((0, 0), (0, 0), (0, 0), (0, 0))
    npad = calculate_padding_parameter(arr.shape, width, stepsize)
    training_set_padded = np.pad(arr, pad_width=npad, mode='constant', constant_values=0)

    dims = training_set_padded.shape
    temp = [training_set_padded[picture, row:row + width, col:col + width, :]  # .flatten()
            for picture in range(0, dims[0]) for row in range(0, dims[1] - width + 1, stepsize) for col in
            range(0, dims[2] - width + 1, stepsize)]
    out_arr = np.stack(temp, axis=0)

    # x = [arr[k, i:i+width, j:j+width, :].flatten()
    # [ 5x5 Subbild vom k-ten Bild] .flatten =>
    #   [[0, 1, 2, 3, 4],
    #    [5, 6, 7, 8, 9],
    #    [0, 1, 2, 3, 4],   =>  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...]
    #    [5, 6, 7, 8, 9],
    #    [0, 1, 2, 3, 4]]
    # for k in range(0, dims[0])  for i in range(0, dims[1] - width + 1, stepsize) for j in range(0, dims[2] - width + 1, stepsize) ], axis=0
    #  für jedes Bild bewege Kernel über das Bild
    # 196 Position ein 4X4 Kernel auf ein 30x30 mit Stride 2 zu plazieren. Pro Position gibt es 256 Channels gibt vor dem stack befehl ein Shape (196,(4,4,256))
    # nach dem Stack  (196,4,4,256)

    return out_arr

def permutate_and_flaten(training_set_kernel, label_set, channel_training, channel_label):
    number_kernels = training_set_kernel.shape[0]
    #random_indices = np.random.permutation(number_kernels)
    training_set_flat = training_set_kernel[:, :, :, channel_training].reshape((number_kernels, -1))
   # training_set_flat_permutated = training_set_flat[random_indices]
    label_set_flat = label_set[:, :, :, channel_label].reshape(number_kernels)
    #label_set_flat_permutated = label_set_flat[random_indices]
    return training_set_flat, label_set_flat
    # return training_set_flat_permutated, label_set_flat_permutated


def transform_to_boolean(array):
    boolean_array = np.maximum(array, 0).astype(np.bool)  # 2,4 for pooled layer
    return boolean_array


def visualize_singel_kernel(kernel, kernel_width , titel):
    f = plt.figure()
    ax = f.add_subplot(111)
    z = np.reshape(kernel, (kernel_width, kernel_width))
    mesh = ax.pcolormesh(z, cmap='gray', vmin=-1, vmax=1)

    plt.colorbar(mesh, ax=ax)
    plt.title(titel, fontsize=20)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def visualize_multi_pic(pic_array, label_array, titel):
    """ show 10 first  pictures """
    for i in range(pic_array.shape[3]):
        ax = plt.subplot(4, 3, i+1 )
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        mesh = ax.pcolormesh(pic_array[:,:,0,i], cmap='gray', vmin=-1, vmax=1)
        plt.colorbar(mesh, ax=ax)
        plt.title(label_array[i])
        plt.gca().set_aspect('equal', adjustable='box')

        #plt.imshow(pic_array[:,:,0,i], cmap=colormap)
        #plt.xlabel(label_array[i])

    st = plt.suptitle(titel, fontsize=14)
    st.set_y(1)
    plt.tight_layout()
    plt.show()

def calculate_convolution(data, kernel, true_label):
    label = []
    kernel_flaten = np.reshape(kernel, (-1))
    data_flaten = np.reshape(data, (data.shape[0], -1))
    for row in data_flaten:
        label.append(np.dot(row , kernel_flaten))
    return label

def visulize_input_data(pic):
    hight = int(np.sqrt(pic.size))
    pic = np.reshape(pic, (hight,hight))

    plt.imshow(pic, cmap='gray')

    plt.show()

def one_class_against_all(array_label, one_class=1, number_classes_output = 2):
    """
converts an array with one_hot_vector for any number of classes into a one_hot_vector,
 whether an example belongs to one class or not
    """
    shape_output = (len(array_label), number_classes_output)
    label_one_class_against_all = np.zeros(shape_output, dtype=int)
    for i, one_hot_vector in enumerate(array_label):
        if one_hot_vector.argmax() == one_class:
            label_one_class_against_all[i,0]=1
        else :
            label_one_class_against_all[i,-1]=1
    return label_one_class_against_all






