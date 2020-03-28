import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import model.two_conv_block_model as model_two_convolution





def acc_data_generation ( network):

    train_nn = np.load('data/data_set_train.npy')
    train_nn = train_nn.reshape((-1, 28, 28))
    label_train_nn = np.load('data/data_set_label_train_nn.npy')

    with tf.Session() as sess:
        network.saver.restore(sess, network.folder_to_save)
        # tensors = [n.name for n in sess.graph.as_graph_def().node]
        # op = restored.sess.graph.get_operations()

        input = sess.graph.get_operation_by_name("Placeholder").outputs[0]

        operation_reshape = sess.graph.get_operation_by_name('Reshape')
        
        operation_sign_con_1 = sess.graph.get_operation_by_name('dcdl_conv_1/conv2d/Sign')
        operation_result_conv_1 = sess.graph.get_operation_by_name('dcdl_conv_1/conv2d/Conv2D')
        operation_kernel_conv_1 = sess.graph.get_operation_by_name('dcdl_conv_1/conv2d/kernel/read')

        operation_sign_con_2 = sess.graph.get_operation_by_name('dcdl_conv_2/conv2d/Sign')
        operation_result_conv_2 = sess.graph.get_operation_by_name('dcdl_conv_2/conv2d/Conv2D')
        operation_kernel_conv_2 = sess.graph.get_operation_by_name('dcdl_conv_2/conv2d/kernel/read')

        

        reshape = sess.run(operation_reshape.outputs[0],
                                          feed_dict={input: train_nn})

        sign_con_1 = sess.run(operation_sign_con_1.outputs[0],
                                          feed_dict={input: train_nn})

        result_conv_1 = sess.run(operation_result_conv_1.outputs[0],
                                         feed_dict={input: train_nn})

        kernel_conv_1= sess.run(operation_kernel_conv_1.outputs[0],
                                         feed_dict={input: train_nn})
        

        sign_con_2 = sess.run(operation_sign_con_2.outputs[0],
                                          feed_dict={input: train_nn})

        result_conv_2 = sess.run(operation_result_conv_2.outputs[0],
                                         feed_dict={input: train_nn})

        kernel_conv_2= sess.run(operation_kernel_conv_2.outputs[0],
                                         feed_dict={input: train_nn})



        np.save('data/data_reshape.npy', reshape)
        np.save('data/sign_con_1.npy', sign_con_1)
        np.save('data/result_conv_1.npy', result_conv_1)
        np.save('data/kernel_conv_1.npy', kernel_conv_1)

        np.save('data/sign_con_2.npy', sign_con_2)
        np.save('data/result_conv_2.npy', result_conv_2)
        np.save('data/kernel_conv_2.npy', kernel_conv_2)

        print('data generation is finished')


if __name__ == '__main__':
    number_classes_to_predict = 2
    network = model_two_convolution.network_two_convolution(shape_of_kernel=(4, 4), nr_training_itaration=1000,
                                                            stride=2, check_every=16, number_of_kernel=16,
                                                            number_classes=number_classes_to_predict)
    acc_data_generation(network)

