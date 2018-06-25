# Senti AI Project Structure
# Copyright (C) 2018  Abien Fred Agarap

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Implementation of the Multinomial Logistic Regression model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '1.0.0'
__author__ = 'Abien Fred Agarap'

import numpy as np
import os
import sys
import tensorflow as tf
import time


class LogisticRegression:
    """Implementation of Multinomial Logistic Regression model using TensorFlow"""

    def __init__(self, num_classes, num_features):
        """Constructs the Multinomial Logistic Regression model

        :param num_classes: The number of classes in the dataset to be used.
        :param num_features: The number of features in the dataset to be used.
        """
        self.name = 'logistic_regression'
        self.num_classes = num_classes
        self.num_features = num_features

        def __build__():
            with tf.name_scope('input'):
                # [BATCH_SIZE, NUM_FEATURES]
                input_features = tf.placeholder(dtype=tf.float32, shape=[None, self.num_features], name='features')

                # [BATCH_SIZE, NUM_CLASSES]
                input_labels = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes], name='labels')

                # define a node for learning rate input
                learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

            with tf.name_scope('model'):
                # define initial weights
                weights = tf.Variable(tf.random_normal(shape=[self.num_features, self.num_classes], stddev=0.01),
                                      name='weights')

                # define initial biases
                biases = tf.Variable(tf.random_normal(shape=[self.num_classes]), name='biases')

                # define the linear model
                linear_model = tf.add(x=tf.matmul(a=input_features, b=weights), y=biases)

                # define the prediction node
                predictions = tf.nn.softmax(linear_model, name='predictions')

            # define tensorboard report on linear_model
            tf.summary.histogram('pre-activations', linear_model)

            with tf.name_scope('training_operations'):
                # define the optimization algorithm
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

                # define the cost function
                cost = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits_v2(logits=linear_model,
                                                                                              labels=input_labels))

                # define the optimization operation
                train_op = optimizer.minimize(loss=cost)

            # define tensorboard report on cost
            tf.summary.scalar('cost', cost)

            with tf.name_scope('metric'):
                # define the prediction matching
                correct_prediction = tf.equal(x=tf.argmax(input=predictions, axis=1),
                                              y=tf.argmax(input=input_labels, axis=1))

                # define the accuracy operation
                accuracy_op = tf.reduce_mean(input_tensor=tf.cast(x=correct_prediction, dtype=tf.float32,
                                                                  name='accuracy'))

            # define tensorboard report on accuracy
            tf.summary.scalar('accuracy', accuracy_op)

            # define variable initializer op
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            # merge all tensorboard logs
            merged = tf.summary.merge_all()

            self.input_features = input_features
            self.input_labels = input_labels
            self.learning_rate = learning_rate
            self.weights = weights
            self.biases = biases
            self.predictions = predictions
            self.cost = cost
            self.train_op = train_op
            self.accuracy_op = accuracy_op
            self.init_op = init_op
            self.merged = merged

        sys.stdout.write('<log> Building graph...\n')
        __build__()
        sys.stdout.write('\n</log>')

    def train(self, training_data, training_data_size, testing_data,
              batch_size=8, checkpoint_path='./checkpoints/', epochs=1, learning_rate=1e-2, log_path='./logs/'):
        """

        :param training_data:
        :param training_data_size:
        :param testing_data:
        :param batch_size:
        :param checkpoint_path:
        :param epochs:
        :param learning_rate:
        :param log_path:
        :return:
        """

        # define assert statements for quick bug checking
        assert type(training_data) is list, \
            'Expected data type : list, but {} is {}'.format(training_data, type(training_data))
        assert type(training_data_size) is int, \
            'Expected data type : int, but {} is {}'.format(training_data_size, type(training_data_size))
        assert type(testing_data) is list, \
            'Expected data type : list, but {} is {}'.format(testing_data, type(testing_data))
        assert type(batch_size) is int, \
            'Expected data type : int, but {} is {}'.format(batch_size, type(batch_size))
        assert batch_size > 0, 'Expected value greater than 0, but {} is not.'.format(batch_size)
        assert type(checkpoint_path) is str, \
            'Expected data type : str, but {} is {}'.format(checkpoint_path, type(checkpoint_path))
        assert type(epochs) is int, \
            'Expected data type : int, but {} is {}'.format(epochs, type(epochs))
        assert epochs > 0, 'Expected value greater than 0, but {} is not.'.format(epochs)
        assert type(learning_rate) is float, \
            'Expected data type : float, but {} is {}'.format(learning_rate, type(learning_rate))
        assert type(log_path) is str, \
            'Expected data type : str, but {} is {}'.format(log_path, type(log_path))

        # create checkpoint_path if it does not exist
        if not os.path.exists(path=checkpoint_path):
            os.mkdir(path=checkpoint_path)

        # create log_path if it does not exist
        if not os.path.exists(path=log_path):
            os.mkdir(path=log_path)

        # create trained model saver
        saver = tf.train.Saver()

        # define timestamp
        timestamp = str(time.asctime())

        # define tensorboard writer
        train_writer = tf.summary.FileWriter(logdir=os.path.join(log_path, timestamp + '-training'))

        with tf.Session() as sess:

            # run variable initializer
            sess.run(self.init_op)

            # get checkpoint state
            checkpoint = tf.train.get_checkpoint_state(checkpoint_dir=checkpoint_path)

            # restore trained model if one exists
            if checkpoint and checkpoint.model_checkpoint_path:
                saver = tf.train.import_meta_graph(meta_graph_or_file=checkpoint.model_checkpoint_path + '.meta')
                saver.restore(sess=sess, save_path=tf.train.latest_checkpoint(checkpoint_dir=checkpoint_path))

            try:
                for step in range(epochs * training_data_size):

                    # get batch of features and labels
                    training_features, training_labels = self.next_batch(batch_size=batch_size,
                                                                         features=training_data[0],
                                                                         labels=training_data[1])

                    # define a dictionary feed for the computational graph
                    feed_dict = {self.input_features: training_features,
                                 self.input_labels: training_labels,
                                 self.learning_rate: learning_rate}

                    # get tensor values by running tensor operations
                    _, training_summary, cost_value, accuracy_value = sess.run([self.train_op,
                                                                                self.merged,
                                                                                self.cost,
                                                                                self.accuracy_op],
                                                                               feed_dict=feed_dict)

                    if step % 100 == 0 and step > 0:

                        # display step loss and step accuracy
                        print('[step {}] loss : {}, accuracy : {}'.format(step, cost_value, accuracy_value))

                        # add tensorboard summary
                        train_writer.add_summary(summary=training_summary, global_step=step)

                        # save trained model at current step
                        saver.save(sess=sess, save_path=os.path.join(checkpoint_path, self.name), global_step=step)

            except KeyboardInterrupt:
                print('Training interrupted at {}'.format(step))
                os._exit(1)
            finally:
                print('Training done at step {}'.format(step))

                # define a dictionary feed for computational graph
                feed_dict = {self.input_features: testing_data[0], self.input_labels: testing_data[1]}

                # get tensor values by running tensor operations
                cost_value, accuracy_value = sess.run([self.cost, self.accuracy_op], feed_dict=feed_dict)

                # display test loss and test accuracy
                print('Test Loss : {}, Test Accuracy : {}'.format(cost_value, accuracy_value))

    @staticmethod
    def next_batch(batch_size, features, labels):
        """Returns a batch of features and labels

        :param batch_size: The number of data in a batch.
        :param features: The features to be batched.
        :param labels: The labels to be batched.
        :return:
        """

        # define indices from 0 to n - 1
        indices = np.arange(start=0, stop=features.shape[0])

        # shuffle the indices
        np.random.shuffle(indices)

        # get a batch of shuffled indices
        indices = indices[:batch_size]

        # return batches of shuffled features and labels
        return features[indices], labels[indices]