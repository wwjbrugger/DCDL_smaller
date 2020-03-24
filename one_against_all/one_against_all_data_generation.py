import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import data.mnist_dataset as md
import own_scripts.dithering as dith

import net_with_maximal_kernel as net
import model.net_with_one_convolution as model_one_convolution





def one_against_all_data_generation ( network):

    train_nn = np.load('data/data_set_train.npy')
    label_train_nn = np.load('data/data_set_label_train_nn.npy')

    with tf.Session() as sess:
        network.saver.restore(sess, network.folder_to_save)
        # tensors = [n.name for n in sess.graph.as_graph_def().node]
        # op = restored.sess.graph.get_operations()

        input = sess.graph.get_operation_by_name("Placeholder").outputs[0]

        operation_data_for_SLS = sess.graph.get_operation_by_name('Reshape')
        operation_label_SLS = sess.graph.get_operation_by_name('dcdl_conv_1/conv2d/Sign')
        operation_result_conv = sess.graph.get_operation_by_name('dcdl_conv_1/conv2d/Conv2D')
        operation_kernel_conv_1_conv2d = sess.graph.get_operation_by_name('dcdl_conv_1/conv2d/kernel/read')

        input_for_SLS = sess.run(operation_data_for_SLS.outputs[0],
                                          feed_dict={input: train_nn.reshape((-1, 28, 28))})

        label_SLS = sess.run(operation_label_SLS.outputs[0],
                                          feed_dict={input: train_nn.reshape((-1, 28, 28))})

        result_conv = sess.run(operation_result_conv.outputs[0],
                                         feed_dict={input: train_nn.reshape((-1, 28, 28))})

        kernel_conv_1_conv2d = sess.run(operation_kernel_conv_1_conv2d.outputs[0],
                                         feed_dict={input: train_nn.reshape((-1, 28, 28))})



        np.save('data/data_for_SLS.npy', input_for_SLS)
        np.save('data/label_SLS.npy', label_SLS)
        np.save('data/result_conv.npy', result_conv)
        np.save('data/kernel.npy', kernel_conv_1_conv2d)

        print('data generation is finished')


if __name__ == '__main__':
    number_classes_to_predict = 2
    network = model_one_convolution.network_one_convolution(shape_of_kernel= (28,28), nr_training_itaration= 1000, stride=28, check_every= 200, number_of_kernel=1,
                                                            number_classes=number_classes_to_predict)
    one_against_all_data_generation(network)

