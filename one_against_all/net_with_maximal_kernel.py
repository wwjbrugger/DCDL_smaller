import numpy as np
import tensorflow as tf
import os
import data.mnist_dataset as md
import model.Gradient_helpLayers_convBlock as helper
import own_scripts.dithering as dith
import  helper_methods as help
import sys

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
#tf.compat.v1compat.v1.logging.set_verbosity(tf.compat.v1compat.v1.logging.ERROR)
class network_one_convolution():

    def __init__(self, name_of_model = "net_with_maximal_kernel", learning_rate = 1E-3,  number_classes=10,
                 input_shape = (None,28,28), nr_training_itaration = 1500,
                 batch_size=2**14, print_every = 100, check_every = 100,
                 number_of_kernel = 10, shape_of_kernel = (3,3), stride = 2, input_channels = 1,
                 input_binarized = True, activation = helper.binarize_STE,
                 use_bias_in_convolution = False):
        #tf.compat.v1.reset_default_graph()
        self.learning_rate = learning_rate
        self.classes = number_classes
        self.input_shape = input_shape
        self.nr_training_itaration = nr_training_itaration
        self.batch_size = batch_size
        self.print_every = print_every
        self.check_every = check_every
        self.folder_to_save = os.path.dirname(sys.argv[0]) + "/stored_models/" + str(name_of_model)
        self.name_of_model = name_of_model
        self.number_of_kernel = number_of_kernel
        self.shape_of_kernel = shape_of_kernel
        self.stride = stride
        self.input_channels = input_channels
        self.input_binarized = input_binarized
        self.activation = activation
        self.use_bias_in_convolution = use_bias_in_convolution


        self.built_graph()

    def built_graph(self):
        self.Input_in_Graph = tf.compat.v1.placeholder(dtype=tf.compat.v1.float32, shape=self.input_shape)
        self.True_Label = tf.compat.v1.placeholder(dtype=tf.compat.v1.float32, shape=[None, self.classes])

        X = tf.compat.v1.reshape(self.Input_in_Graph,(-1, self.input_shape[1], self.input_shape[2], self.input_channels))

        if not self.input_binarized:
            thres = []
            for i in range(1):
                thres.append(helper.input_exp(X, i, activation= self.activation, name= "binary_input" ))
            X = tf.compat.v1.concat(thres, 3)

        with tf.compat.v1.variable_scope("dcdl_conv_1", reuse=False):
            X = tf.compat.v1.layers.conv2d(inputs=X, filters=self.number_of_kernel, kernel_size=self.shape_of_kernel, strides=[
                self.stride, self.stride], padding="same", activation=self.activation, use_bias=False)

        X = tf.compat.v1.layers.flatten(X)

        self.prediction = tf.compat.v1.layers.dense(X, self.classes, tf.compat.v1.nn.softmax)

        self.loss = tf.compat.v1.reduce_mean(-tf.compat.v1.reduce_sum(self.True_Label *
                                                tf.compat.v1.log(self.prediction + 1E-10), reduction_indices=[1]))  # + reg2
        self.step = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Evaluate model
        self.one_hot_out = tf.compat.v1.argmax(self.prediction, 1)
        self.hits = tf.compat.v1.equal(self.one_hot_out, tf.compat.v1.argmax(self.True_Label, 1))
        self.accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(self.hits, tf.compat.v1.float32))

        # Initialize the variables
        self.init = tf.compat.v1.global_variables_initializer()

        # Save model
        self.saver = tf.compat.v1.train.Saver()


    def training(self, train, label_train, test, label_test, loging = True):
        loss_list, val_list = [], []
        with tf.compat.v1.Session() as sess:
            if loging:
                path_to_store_logs = os.path.dirname(sys.argv[0]) + "/logs"
                writer = tf.compat.v1.summary.FileWriter(path_to_store_logs, session=sess,
                                               graph=sess.graph)  # + self.name_of_model, sess.graph)

            sess.run(self.init)
            best_acc_so_far = 0

            for iteration in range(self.nr_training_itaration):


                indices = np.random.choice(len(train), self.batch_size)
                batch_X = train[indices]
                batch_Y = label_train[indices]
                feed_dict = {self.Input_in_Graph: batch_X, self.True_Label: batch_Y}

                _, lo, acc = sess.run([self.step, self.loss, self.accuracy], feed_dict=feed_dict)
                if iteration % self.print_every == 1:
                    print("Iteration: ", iteration, "Acc. at trainset: ", acc, flush=True)

                if iteration % self.check_every == 1:
                    indices = np.random.choice(len(test), 5000)
                    acc, lo = sess.run([self.accuracy, self.loss], feed_dict={
                        self.Input_in_Graph: test[indices], self.True_Label: label_test[indices]})
                    print("step: ", iteration, 'Accuracy at test_set: ', acc, )

                    loss_list.append(lo)
                    val_list.append(acc)

                    if acc > best_acc_so_far:
                        best_acc_so_far = acc
                        save_path = self.saver.save(sess, self.folder_to_save)
                        print('Path to store parameter: ', save_path)



    def evaluate(self, input, label):
        with tf.compat.v1.Session() as sess:
            self.saver.restore(sess, self.folder_to_save)
            size_test_nn = test.shape[0]
            counter = 0  # biased
            acc_sum = 0
            for i in range(0, size_test_nn, 512):
                start = i
                end = min(start + 512, size_test_nn)
                acc = sess.run([self.accuracy], feed_dict={self.Input_in_Graph: input, self.True_Label: label})[0]
                acc_sum += acc
                counter += 1

            print("Test Accuracy", self.name_of_model, acc_sum / counter)


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

    network = network_one_convolution(shape_of_kernel= (28,28), nr_training_itaration= 2000, stride=28, check_every= 200, number_of_kernel=1)

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
