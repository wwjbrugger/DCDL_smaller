import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import data.mnist_dataset as md
import own_scripts.dithering as dith
import matplotlib.pyplot as plt
import net_with_maximal_kernel as net





if __name__ == '__main__':

    dithering_used = True


    number_of_input_pic = 10000
    train_nn = np.load('data/data_set_train.npy')
    label_train_nn = np.load('data/data_set_label_train_nn.npy')


    class_names = ['T-shirt/top', 'Trousers', 'Pullover', 'Dress',
                   'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    dith.visualize_pic(train_nn, label_train_nn, class_names, "Mnist", plt.cm.Greys)

    with tf.Session() as sess:
        network = net.network_one_convolution(shape_of_kernel= (28,28), nr_training_itaration= 2000, stride=28, number_of_kernel=1 )
        network.saver.restore(sess, network.folder_to_save)
        tensors = [n.name for n in sess.graph.as_graph_def().node]

        # conv1_kernel_val = restored.sess.graph.get_tensor_by_name('dcdl_conv_1/conv2d/kernel:0')
        # op = restored.sess.graph.get_operations()
        # for o in op:
        # print(str(op).replace(',', '\n'))

        input = sess.graph.get_operation_by_name("Placeholder").outputs[0]
        # operation_data_for_SLS =  sess.graph.get_operation_by_name('input0/Sign')
        #if dithering_used:
        operation_data_for_SLS = sess.graph.get_operation_by_name('Reshape')
        #else:
           # operation_data_for_SLS = sess.graph.get_operation_by_name('concat')
        # operation_label_SLS =  sess.graph.get_operation_by_name('dcdl_conv_1/Sign')
        operation_label_SLS = sess.graph.get_operation_by_name('dcdl_conv_1/conv2d/Sign')
        operation_result_conv = sess.graph.get_operation_by_name('dcdl_conv_1/conv2d/Conv2D')

        operation_kernel_conv_1_conv2d = sess.graph.get_operation_by_name('dcdl_conv_1/conv2d/kernel/read')
        # operation_bias_conv_1_conv2d = sess.graph.get_operation_by_name('dcdl_conv_1/conv2d/bias/read')

        input_for_SLS = sess.run(operation_data_for_SLS.outputs[0],
                                          feed_dict={input: train_nn.reshape((-1, 28, 28))})
        np.save('data/data_for_SLS.npy', input_for_SLS)

        label_SLS = sess.run(operation_label_SLS.outputs[0], feed_dict={input: train_nn.reshape((-1, 28, 28))})
        np.save('data/label_SLS.npy', label_SLS)

        result_conv = sess.run(operation_result_conv.outputs[0],
                                        feed_dict={input: train_nn.reshape((-1, 28, 28))})
        np.save('data/result_conv.npy', result_conv)

        kernel_conv_1_conv2d = sess.run(operation_kernel_conv_1_conv2d.outputs[0],
                                                 feed_dict={input: train_nn.reshape((-1, 28, 28))})
        np.save('data/kernel.npy', kernel_conv_1_conv2d)

        print('data generation is finished')
        # bias_conv_1_conv2d = sess.run(operation_bias_conv_1_conv2d.outputs[0], feed_dict={input: train_nn[1].reshape((1,28,28))})
        # np.save('data/bias.npy', bias_conv_1_conv2d)

