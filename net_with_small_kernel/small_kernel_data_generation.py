from model.one_conv_block_model import stored_network, network
import model.Gradient_helpLayers_convBlock as grad_help
import own_scripts.dithering as dith

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def load_network(name):
    model_l = stored_network(name)
    return model_l

if __name__ == '__main__':

    dithering_used = False

    train_nn = np.load('data/data_set_train.npy')
    label_train_nn = np.load('data/data_set_label_train_nn.npy')

    number_of_input_pic = 25

    class_names = ['T-shirt/top', 'Trousers', 'Pullover', 'Dress',
                   'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    dith.visualize_pic(train_nn, label_train_nn, class_names, "Mnist", plt.cm.Greys)

    dcdl_network =network("small_kernel_net", avg_pool=False, real_in= dithering_used,
                         lr=1E-4, batch_size=2**8, activation=grad_help.binarize_STE,
                         pool_by_stride=False, pool_before=True, pool_after=False,
                         skip=False, pool_skip=False,
                         bn_before=False, bn_after=False, ind_scaling=False, training_itaration= 50
                         )
    print("Values of Network are loaded", flush=True)
    restored = load_network(dcdl_network)
    #restored = load_network(dcdl_network)
    #restored = stored_network(dcdl_network)

    print ('YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY')


    writer = tf.summary.FileWriter('logs/graphs_jannis_layer', restored.sess.graph)
    tensors = [n.name for n in restored.sess.graph.as_graph_def().node]

    # conv1_kernel_val = restored.sess.graph.get_tensor_by_name('dcdl_conv_1/conv2d/kernel:0')
    # op = restored.sess.graph.get_operations()
    # for o in op:
    # print(str(op).replace(',', '\n'))

    input = restored.sess.graph.get_operation_by_name("Placeholder_1").outputs[0]
    # operation_data_for_SLS =  restored.sess.graph.get_operation_by_name('input0/Sign')
    if dithering_used:
        operation_data_for_SLS = restored.sess.graph.get_operation_by_name('Reshape')
    else:
        operation_data_for_SLS = restored.sess.graph.get_operation_by_name('concat/concat')
    # operation_label_SLS =  restored.sess.graph.get_operation_by_name('dcdl_conv_1/Sign')
    operation_label_SLS = restored.sess.graph.get_operation_by_name('dcdl_conv_1/conv2d/Sign')
    operation_result_conv = restored.sess.graph.get_operation_by_name('dcdl_conv_1/conv2d/Conv2D')

    operation_kernel_conv_1_conv2d = restored.sess.graph.get_operation_by_name('dcdl_conv_1/conv2d/kernel/read')
    # operation_bias_conv_1_conv2d = restored.sess.graph.get_operation_by_name('dcdl_conv_1/conv2d/bias/read')

    input_for_SLS = restored.sess.run(operation_data_for_SLS.outputs[0],
                                      feed_dict={input: train_nn[1].reshape((1, 28, 28))})
    np.save('data/data_for_SLS.npy', input_for_SLS)

    label_SLS = restored.sess.run(operation_label_SLS.outputs[0], feed_dict={input: train_nn[1].reshape((1, 28, 28))})
    np.save('data/label_SLS.npy', label_SLS)

    result_conv = restored.sess.run(operation_result_conv.outputs[0],
                                    feed_dict={input: train_nn[1].reshape((1, 28, 28))})
    np.save('data/result_conv.npy', result_conv)

    kernel_conv_1_conv2d = restored.sess.run(operation_kernel_conv_1_conv2d.outputs[0],
                                             feed_dict={input: train_nn[1].reshape((1, 28, 28))})
    np.save('data/kernel.npy', kernel_conv_1_conv2d)

    print('data generation is finished')
    # bias_conv_1_conv2d = restored.sess.run(operation_bias_conv_1_conv2d.outputs[0], feed_dict={input: train_nn[1].reshape((1,28,28))})
    # np.save('data/bias.npy', bias_conv_1_conv2d)
