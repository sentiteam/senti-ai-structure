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

import sys
import tensorflow as tf


class LogisticRegression:
    """Implementation of Multinomial Logistic Regression model using TensorFlow"""

    def __init__(self, num_classes, num_features):
        """Constructs the Multinomial Logistic Regression model

        :param num_classes: The number of classes in the dataset to be used.
        :param num_features: The number of features in the dataset to be used.
        """
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
                linear_model = tf.add(x=tf.matmul(a=weights, b=input_features), y=biases)

                # define the prediction node
                predictions = tf.nn.softmax(linear_model, name='predictions')

            with tf.name_scope('training_operations'):
                # define the optimization algorithm
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

                # define the cost function
                cost = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits_v2(logits=linear_model,
                                                                                              labels=input_labels))

                # define the optimization operation
                train_op = optimizer.minimize(loss=cost)

            with tf.name_scope('metric'):
                correct_prediction = tf.equal(x=tf.argmax(input=predictions, axis=1),
                                              y=tf.argmax(input=input_labels, axis=1))
                accuracy_op = tf.reduce_mean(input_tensor=tf.cast(x=correct_prediction, dtype=tf.float32,
                                                                  name='accuracy'))

            self.input_features = input_features
            self.input_labels = input_labels
            self.learning_rate = learning_rate
            self.weights = weights
            self.biases = biases
            self.predictions = predictions
            self.train_op = train_op
            self.accuracy_op = accuracy_op

        sys.stdout.write('<log> Building graph...\n')
        __build__()
        sys.stdout.write('\n</log>')

    def train(self):
        pass
