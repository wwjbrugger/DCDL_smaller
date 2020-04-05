import os
os.environ["BLD_PATH"] = "../parallel_sls/bld/Parallel_SLS_shared"
import numpy as np
import matplotlib.pyplot as plt
import SLS_Algorithm as SLS
import pickle
from tqdm import tqdm



def calculate_padding_parameter(shape_input_pic, filter_size, stride, ):
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

    return ((0, 0), (int(pad_top), int(pad_bottom)), (int(pad_left), int(pad_right)), (0, 0))


def data_in_kernel(arr, stepsize=2, width=4):  # kernel views
    # Need padding for convolution
    # npad = ((0, 0), (1, 1), (1, 1), (0, 0))
    # npad = ((0, 0), (0, 0), (0, 0), (0, 0))
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
    # random_indices = np.random.permutation(number_kernels)
    training_set_flat = training_set_kernel[:, :, :, channel_training].reshape((number_kernels, -1))
    # training_set_flat_permutated = training_set_flat[random_indices]
    label_set_flat = label_set[:, :, :, channel_label].reshape(number_kernels)
    # label_set_flat_permutated = label_set_flat[random_indices]
    return training_set_flat, label_set_flat
    # return training_set_flat_permutated, label_set_flat_permutated


def permutate_and_flaten_2(data, label, channel_label):
    num_pieces_under_kernel = data.shape[0]
    temp = []
    for pic in range(data.shape[0]):
        pic_temp = []
        for channel in range(data.shape[3]):
             pic_temp.append(np.reshape(data[pic,:,:,channel], -1))
        temp.append(np.reshape(pic_temp, -1))
    data_flatten = np.array(temp)
    label_set_flat = label[:, :, :, channel_label].reshape(-1)
    return data_flatten, label_set_flat



def permutate_and_flaten_3(data, label, channel_label):
    number_kernels = data.shape[0]
    training_set_flat = data.reshape((number_kernels, -1))
    label_set_flat = label[:, :, :, channel_label].reshape(number_kernels)
    return training_set_flat, label_set_flat




def transform_to_boolean(array):
    boolean_array = np.maximum(array, 0).astype(np.bool)  # 2,4 for pooled layer
    return boolean_array


def visualize_singel_kernel(kernel, kernel_width, titel, set_vmin_vmax = True):
    f = plt.figure()
    ax = f.add_subplot(111)
    z = np.reshape(kernel, (kernel_width, kernel_width))
    if set_vmin_vmax:
        mesh = ax.pcolormesh(z, cmap='gray', vmin=-1, vmax=1)
    else:
        mesh = ax.pcolormesh(z)
    plt.colorbar(mesh, ax=ax)
    plt.title(titel, fontsize=20)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def visualize_multi_pic(pic_array, label_array, titel):
    """ show 10 first  pictures """
    for i in range(pic_array.shape[3]):
        ax = plt.subplot(4, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        mesh = ax.pcolormesh(pic_array[:, :, 0, i], cmap='gray', vmin=-1, vmax=1)
        plt.colorbar(mesh, ax=ax)
        plt.title(label_array[i])
        plt.gca().set_aspect('equal', adjustable='box')

        # plt.imshow(pic_array[:,:,0,i], cmap=colormap)
        # plt.xlabel(label_array[i])

    st = plt.suptitle(titel, fontsize=14)
    st.set_y(1)
    plt.tight_layout()
    plt.show()


def calculate_convolution(data_flat, kernel, true_label):
    label = []
    kernel_flaten = np.reshape(kernel, (-1))
    for row in data_flat:
        label.append(np.dot(row, kernel_flaten))
    return label


def visulize_input_data(pic):
    hight = int(np.sqrt(pic.size))
    pic = np.reshape(pic, (hight, hight))

    plt.imshow(pic, cmap='gray')

    plt.show()


def one_class_against_all(array_label, one_class=1, number_classes_output=2):
    """
converts an array with one_hot_vector for any number of classes into a one_hot_vector,
 whether an example belongs to one class or not
    """
    shape_output = (len(array_label), number_classes_output)
    label_one_class_against_all = np.zeros(shape_output, dtype=int)
    for i, one_hot_vector in enumerate(array_label):
        if one_hot_vector.argmax() == one_class:
            label_one_class_against_all[i, 0] = 1
        else:
            label_one_class_against_all[i, -1] = 1
    return label_one_class_against_all


def reduce_kernel(input, mode):
    sum = np.sum(input, axis=0)
    if mode in 'sum':
        return sum
    elif mode in 'mean':
        mean = np.mean(input, axis=0)
        return mean
    elif mode in 'min_max':
        min = sum.min()
        max = sum.max()
        min_max = np.where(sum < 0, sum / min, sum / max)
        return min_max
    elif mode in 'norm':
        mean = np.mean(sum)
        max_value = sum.max()
        max_centered = max_value-mean
        norm = (sum-mean)/max_centered
        return norm
    else:
        raise ValueError("{} ist not a valid mode.".format(mode))


def sls_convolution (Number_of_Product_term, Maximum_Steps_in_SKS, stride_of_convolution, data_sign, label_sign, used_kernel, result= None, path_to_store = None, SLS_Training = True) :
    kernel_width = used_kernel.shape[0]
    data_flat, label = prepare_data_for_sls(data_sign, label_sign, kernel_width, stride_of_convolution)
    np.save(path_to_store + '_data_flat.npy', data_flat)

    logic_formulas = []
    if SLS_Training:
        for channel in range(label.shape[3]):
            print("Ruleextraction for kernel_conv_1 {} ".format(channel))
            label_flat = label[:, :, :, channel].reshape(data_flat.shape[0])

            found_formula = \
                SLS.rule_extraction_with_sls_without_validation(data_flat,label_flat, Number_of_Product_term,
                                                               Maximum_Steps_in_SKS)
            found_formula.shape_input_data = data_sign.shape
            found_formula.shape_output_data = label.shape
            logic_formulas.append(found_formula)

            accurancy = (data_flat.shape[0] - found_formula.total_error) / data_flat.shape[0]
            print("Accurancy of SLS: ", accurancy, "\n")

            if result is not None:
                label_self_calculated = calculate_convolution(data_flat, used_kernel[:, :, :, channel], result)


        if path_to_store is not None:
            pickle.dump(logic_formulas, open(path_to_store, "wb"))

        formel_in_array_code = []
        for formel in logic_formulas:
            formel_in_array_code.append(np.reshape(formel.formel_in_arrays_code, (-1, kernel_width, kernel_width)))
        np.save(path_to_store + '_in_array_code.npy', formel_in_array_code)

def prepare_data_for_sls(data_sign, label_sign, kernel_width, stride_of_convolution):
    data_bool = transform_to_boolean(data_sign)
    label_bool = transform_to_boolean(label_sign)

    data_under_kernel = data_in_kernel(data_bool, stepsize=stride_of_convolution, width=kernel_width)
    number_kernels = data_under_kernel.shape[0]

    data_bool_flat = data_under_kernel.reshape((number_kernels, -1))

    return data_bool_flat, label_bool


def sls_dense_net (Number_of_Product_term, Maximum_Steps_in_SKS, data, label, path_to_store = None, SLS_Training = True) :

    data = transform_to_boolean(data)
    label_set_one_hot = transform_to_boolean(label)
    label = np.array([label[0] for label in label_set_one_hot])
    data_flat = np.reshape(data, (data.shape[0], -1))
    np.save(path_to_store + '_data_flat.npy', data_flat)
    if SLS_Training:
        found_formula = \
            SLS.rule_extraction_with_sls_without_validation(data_flat,label, Number_of_Product_term,
                                                           Maximum_Steps_in_SKS)
        found_formula.shape_input_data = data.shape
        found_formula.shape_output_data = label.shape

        accurancy = (data_flat.shape[0] - found_formula.total_error) / data_flat.shape[0]
        print("Accurancy of SLS: ", accurancy, '\n')

        if path_to_store is not None:
            pickle.dump(found_formula, open(path_to_store, "wb"))


def prediction_SLS_fast (data_flat, label, found_formula, path_to_store_prediction):
    prediction = np.empty(label.shape, np.bool)

    if label.ndim == 2:
        label = np.array([-1 if l[0]==0 else 1 for l in label])
        print('Calculate prediction')
        prediction = prediction_for_one_kernel(data_flat, found_formula, label.shape)

    else:
        for channel in tqdm(range(label.shape[3])):
            prediction_one_channel = SLS.calc_prediction_in_C(data_flat, label[:, :, :, channel].flatten().shape, found_formula[channel])
            prediction[:, :, :, channel] = np.reshape(prediction_one_channel, label[:, :, :, channel].shape)
    error = np.sum(np.abs(label - np.where(prediction, 1 ,-1)))
    print('Error of prediction', error)
    print('Acc', (label.size-error)/label.size )
    np.save(path_to_store_prediction, prediction)




def prediction_for_one_kernel(data_under_kernel , formula, output_shape):
    output = np.array(formula.evaluate_belegung_like_c(data_under_kernel))
    output = np.reshape(output, output_shape)
    return output