"""

Model with 18 layer for cifar data set

"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model.Gradient_helpLayers_convBlock import *

tf.reset_default_graph()


# model +++++++++++++++++++++++++++++++++++++++++++++++++

class network():

    def __init__(self, path_to_use, name_of_model='bla', avg_pool=False, learning_rate=1E-4, number_classes=10,
                 input_shape=(None, 28, 28, 1), nr_training_itaration=1500, batch_size=2 ** 8, print_every=100,
                 check_every=10,
                 activation=binarize_STE, number_of_kernel=64, shape_of_kernel=(3, 3), stride=2, input_channels=1,
                 use_bias_in_convolution=False,
                 pool_by_stride=False, bn_before=True, bn_after=False, ind_scaling=True, pool_before=True,
                 pool_after=False, skip=False, pool_skip=False, drop_rate = 0.5):
        # config ++++++++++++++++++++++++++++++++++++++++++++++++
        tf.compat.v1.reset_default_graph()
        self.avg_pool = avg_pool
        self.learning_rate = learning_rate
        self.classes = number_classes
        self.input_shape = input_shape
        self.nr_training_itaration = nr_training_itaration
        self.batch_size = batch_size
        self.print_every = print_every
        self.check_every = check_every
        self.folder_to_save = path_to_use['store_model'] + str(name_of_model)
        self.name_of_model = name_of_model
        self.number_of_kernel = number_of_kernel
        self.shape_of_kernel = shape_of_kernel[0]
        self.stride = stride
        self.input_channels = input_channels
        self.activation = activation  # 'relu'
        self.use_bias_in_convolution = use_bias_in_convolution
        self.pool_by_stride = pool_by_stride
        self.bn_before = bn_before
        self.bn_after = bn_after
        self.ind_scaling = ind_scaling
        self.pool_before = pool_before
        self.pool_after = pool_after
        self.skip = skip
        self.pool_skip = pool_skip
        self.drop_rate = drop_rate
        self.built_graph()

    def built_graph(self):

        # targets ++++++++++++++++++++++++++++++++++++++++++++++
        self.pretrain = tf.compat.v1.placeholder(dtype=tf.compat.v1.float32)
        self.Input_in_Graph = tf.compat.v1.placeholder(dtype=tf.compat.v1.float32, shape=self.input_shape)
        self.Label = tf.compat.v1.placeholder(dtype=tf.compat.v1.float32, shape=[None, self.classes])

        pooling = self.pool_by_stride or self.pool_after or self.pool_before

        X = tf.compat.v1.reshape(self.Input_in_Graph,
                                 (-1, self.input_shape[1], self.input_shape[2], self.input_channels))


        if self.skip:
            if self.pool_skip:
                X = rule_conv_block(X, pretrain=self.pretrain, nr=1, n_filter=self.number_of_kernel,
                                    s_filter=self.shape_of_kernel,
                                    pool_by_stride=self.pool_by_stride, activation=self.activation,
                                    bn_before=self.bn_before, bn_after=self.bn_after, ind_scaling=self.ind_scaling,
                                    pool_before=self.pool_before, pool_after=self.pool_after, avg_pool=self.avg_pool,
                                    skip=False)
            else:
                X = rule_conv_block(X, pretrain=self.pretrain, nr=1, n_filter=self.number_of_kernel,
                                    s_filter=self.shape_of_kernel,
                                    activation=self.activation,
                                    bn_before=self.bn_before, bn_after=self.bn_after, ind_scaling=self.ind_scaling,
                                    skip=False)
        else:
            X = rule_conv_block(X, pretrain=self.pretrain, nr=1, n_filter=self.number_of_kernel,
                                s_filter=self.shape_of_kernel,
                                pool_by_stride=self.pool_by_stride, activation=self.activation,
                                bn_before=self.bn_before, bn_after=self.bn_after, ind_scaling=self.ind_scaling,
                                pool_before=self.pool_before, pool_after=self.pool_after, avg_pool=self.avg_pool)

        X = tf.nn.dropout(X, self.drop_rate)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Layer 3 and 4
        if self.skip:
            X = rule_conv_block(X, pretrain=self.pretrain, nr=3, n_filter=self.number_of_kernel * (2 if pooling else 1),
                                s_filter=self.shape_of_kernel, activation=self.activation,
                                bn_before=self.bn_before, bn_after=self.bn_after, ind_scaling=self.ind_scaling,
                                skip=True)
        else:
            X = rule_conv_block(X, pretrain=self.pretrain, nr=3, n_filter=self.number_of_kernel * (2 if pooling else 1),
                                s_filter=self.shape_of_kernel, activation=self.activation,
                                bn_before=self.bn_before, bn_after=self.bn_after, ind_scaling=self.ind_scaling)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Layer 5 and 6
        if self.skip:
            if self.pool_skip:
                X = rule_conv_block(X, pretrain=self.pretrain, nr=5,
                                    n_filter=self.number_of_kernel * (2 if pooling else 1),
                                    s_filter=self.shape_of_kernel, pool_by_stride=self.pool_by_stride,
                                    activation=self.activation,
                                    bn_before=self.bn_before, bn_after=self.bn_after, ind_scaling=self.ind_scaling,
                                    pool_before=self.pool_before, pool_after=self.pool_after, avg_pool=self.avg_pool,
                                    skip=True)
            else:
                X = rule_conv_block(X, pretrain=self.pretrain, nr=5,
                                    n_filter=self.number_of_kernel * (2 if pooling else 1),
                                    s_filter=self.shape_of_kernel, activation=self.activation,
                                    bn_before=self.bn_before, bn_after=self.bn_after, ind_scaling=self.ind_scaling,
                                    skip=True)
        else:
            X = rule_conv_block(X, pretrain=self.pretrain, nr=5, n_filter=self.number_of_kernel * (2 if pooling else 1),
                                s_filter=self.shape_of_kernel, pool_by_stride=self.pool_by_stride,
                                activation=self.activation,
                                bn_before=self.bn_before, bn_after=self.bn_after, ind_scaling=self.ind_scaling,
                                pool_before=self.pool_before, pool_after=self.pool_after, avg_pool=self.avg_pool)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Layer 7 and 8
        if self.skip:
            X = rule_conv_block(X, pretrain=self.pretrain, nr=7, n_filter=self.number_of_kernel * (4 if pooling else 1),
                                s_filter=self.shape_of_kernel, activation=self.activation,
                                bn_before=self.bn_before, bn_after=self.bn_after, ind_scaling=self.ind_scaling,
                                skip=True)
        else:
            X = rule_conv_block(X, pretrain=self.pretrain, nr=7, n_filter=self.number_of_kernel * (4 if pooling else 1),
                                s_filter=self.shape_of_kernel, activation=self.activation,
                                bn_before=self.bn_before, bn_after=self.bn_after, ind_scaling=self.ind_scaling)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Layer 9 and 10
        if self.skip:
            X = rule_conv_block(X, pretrain=self.pretrain, nr=9, n_filter=self.number_of_kernel * (4 if pooling else 1),
                                s_filter=self.shape_of_kernel, activation=self.activation,
                                bn_before=self.bn_before, bn_after=self.bn_after, ind_scaling=self.ind_scaling,
                                skip=True)
        else:
            X = rule_conv_block(X, pretrain=self.pretrain, nr=9, n_filter=self.number_of_kernel * (4 if pooling else 1),
                                s_filter=self.shape_of_kernel, activation=self.activation,
                                bn_before=self.bn_before, bn_after=self.bn_after, ind_scaling=self.ind_scaling)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Layer 11 and 12
        if self.skip:
            if self.pool_skip:
                X = rule_conv_block(X, pretrain=self.pretrain, nr=11,
                                    n_filter=self.number_of_kernel * (4 if pooling else 1),
                                    s_filter=self.shape_of_kernel, pool_by_stride=self.pool_by_stride,
                                    activation=self.activation,
                                    bn_before=self.bn_before, bn_after=self.bn_after, ind_scaling=self.ind_scaling,
                                    pool_before=self.pool_before, pool_after=self.pool_after, avg_pool=self.avg_pool,
                                    skip=True)
            else:
                X = rule_conv_block(X, pretrain=self.pretrain, nr=11,
                                    n_filter=self.number_of_kernel * (4 if pooling else 1),
                                    s_filter=self.shape_of_kernel, activation=self.activation,
                                    bn_before=self.bn_before, bn_after=self.bn_after, ind_scaling=self.ind_scaling,
                                    skip=True)
        else:
            X = rule_conv_block(X, pretrain=self.pretrain, nr=11,
                                n_filter=self.number_of_kernel * (4 if pooling else 1),
                                s_filter=self.shape_of_kernel, pool_by_stride=self.pool_by_stride,
                                activation=self.activation,
                                bn_before=self.bn_before, bn_after=self.bn_after, ind_scaling=self.ind_scaling,
                                pool_before=self.pool_before, pool_after=self.pool_after, avg_pool=self.avg_pool)

        X = tf.nn.dropout(X, self.drop_rate)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Layer 13 and 14
        if self.skip:
            X = rule_conv_block(X, pretrain=self.pretrain, nr=13,
                                n_filter=self.number_of_kernel * (8 if pooling else 1),
                                s_filter=self.shape_of_kernel, activation=self.activation,
                                bn_before=self.bn_before, bn_after=self.bn_after, ind_scaling=self.ind_scaling,
                                skip=True)
        else:
            X = rule_conv_block(X, pretrain=self.pretrain, nr=13,
                                n_filter=self.number_of_kernel * (8 if pooling else 1),
                                s_filter=self.shape_of_kernel, activation=self.activation,
                                bn_before=self.bn_before, bn_after=self.bn_after, ind_scaling=self.ind_scaling)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Layer 15 and 16
        if self.skip:
            X = rule_conv_block(X, pretrain=self.pretrain, nr=15,
                                n_filter=self.number_of_kernel * (8 if pooling else 1),
                                s_filter=self.shape_of_kernel, activation=self.activation,
                                bn_before=self.bn_before, bn_after=self.bn_after, ind_scaling=self.ind_scaling,
                                skip=True)
        else:
            X = rule_conv_block(X, pretrain=self.pretrain, nr=15,
                                n_filter=self.number_of_kernel * (8 if pooling else 1),
                                s_filter=self.shape_of_kernel, activation=self.activation,
                                bn_before=self.bn_before, bn_after=self.bn_after, ind_scaling=self.ind_scaling)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Layer 17 and 18
        if self.skip:
            X = rule_conv_block(X, pretrain=self.pretrain, nr=17,
                                n_filter=self.number_of_kernel * (8 if pooling else 1),
                                s_filter=self.shape_of_kernel, activation=self.activation,
                                bn_before=self.bn_before, bn_after=self.bn_after, ind_scaling=self.ind_scaling,
                                skip=True)
        else:
            X = rule_conv_block(X, pretrain=self.pretrain, nr=17,
                                n_filter=self.number_of_kernel * (8 if pooling else 1),
                                s_filter=self.shape_of_kernel, activation=self.activation,
                                bn_before=self.bn_before, bn_after=self.bn_after, ind_scaling=self.ind_scaling)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Fully connected layer and softmax output_one_picture
        X = tf.layers.flatten(X)
        X = rule_dense_eff(X, 1024, int(X.shape[1]), 1, activation=self.activation)

        self.Prediction = tf.compat.v1.layers.dense(X, self.classes, tf.compat.v1.nn.softmax)

        self.loss = tf.compat.v1.reduce_mean(
            - tf.compat.v1.reduce_sum(self.Label * tf.compat.v1.log(self.Prediction + 1E-10),
                                      reduction_indices=[1]))  # + reg2
        self.step = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Evaluate model
        self.one_hot_out = tf.compat.v1.argmax(self.Prediction, 1)
        self.hits = tf.compat.v1.equal(self.one_hot_out, tf.compat.v1.argmax(self.Label, 1))
        self.accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(self.hits, tf.compat.v1.float32))

        # Initialize the variables
        self.init = tf.compat.v1.global_variables_initializer()

        # Save model
        self.saver = tf.compat.v1.train.Saver()

    def training(self, train, label_train, val, label_val, path_to_use, loging=True):
        loss_list, acc_list, steps = [], [], []
        with  tf.compat.v1.Session() as sess:
            if loging:
                writer = tf.compat.v1.summary.FileWriter(path_to_use['logs'], session=sess,
                                                         graph=sess.graph)

            sess.run(self.init)
            best_acc_so_far = 0

            for iteration in range(self.nr_training_itaration):
                indices = np.random.choice(len(train), self.batch_size)
                batch_X = train[indices]
                batch_Y = label_train[indices]
                feed_dict = {self.Input_in_Graph: batch_X, self.Label: batch_Y, }

                _, lo, acc = sess.run([self.step, self.loss, self.accuracy], feed_dict=feed_dict)
                if iteration % self.print_every == 1:
                    print("Iteration: ", iteration, "Acc. at trainset: ", acc, flush=True)

                if iteration % self.check_every == 1:
                    indices = np.random.choice(len(val), 5000)
                    acc, lo = sess.run([self.accuracy, self.loss], feed_dict={
                        self.Input_in_Graph: val[indices], self.Label: label_val[indices]})
                    print("step: ", iteration, 'Accuracy at validation_set: ', acc, )

                    loss_list.append(lo)
                    acc_list.append(acc)
                    steps.append(iteration)

                    if acc > best_acc_so_far:
                        best_acc_so_far = acc
                        save_path = self.saver.save(sess, self.folder_to_save)
                        print('Path to store parameter: ', save_path)

        return loss_list, acc_list, steps

    def evaluate(self, input, label):
        with tf.compat.v1.Session() as sess:
            self.saver.restore(sess, self.folder_to_save)
            acc = sess.run([self.accuracy], feed_dict={self.Input_in_Graph: input, self.Label: label})[0]
            print("Test Accuracy", self.name_of_model, acc)
            return acc
