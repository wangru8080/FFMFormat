# !/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, log_loss
import numpy as np
from DataParse import DataParse
from sklearn.base import BaseEstimator, TransformerMixin

class FFM(BaseEstimator, TransformerMixin):
    def __init__(self, feature_size, field_size, embedding_size=8, epochs=10, batch_size=128,
                 learning_rate=0.001, optimizer_type='adam', random_seed=2018, loss_type='logloss', metric_type='auc'):
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.metric_type = metric_type

        self.__init_graph()

    def __init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.feature_index = tf.placeholder(tf.int32, [None, self.field_size], name='feature_index')
            self.feature_value = tf.placeholder(tf.float32, [None, self.field_size], name='feature_value')
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')

            weights = {}
            biases = {}

            with tf.name_scope('init'):
                weights['feature_embeddings'] = tf.Variable(tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01), name='feature_embeddings')
                self.embeddings = tf.nn.embedding_lookup(weights['feature_embeddings'], self.feature_index) # [None, field_size, embedding_size]
                feat_value = tf.reshape(self.feature_value, shape=[-1, self.field_size, 1]) # [None, field_size, 1]
                self.embeddings = tf.multiply(self.embeddings, feat_value) # [None, field_size, embedding_size]

            with tf.name_scope('first_order'):
                biases['feature_bias'] = tf.Variable(tf.random_uniform([self.feature_size, 1], 0.0, 1.0), name='feature_bias')
                self.y_first_order = tf.nn.embedding_lookup(biases["feature_bias"], self.feature_index)  # [None, field_size, 1]
                self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feat_value), 2) # [None, field_size]

            with tf.name_scope('second_order'):
                pair_wise_product_list = []
                for i in range(self.field_size):
                    for j in range(i + 1, self.field_size):
                        pair_wise_product_list.append(
                            tf.multiply(self.embeddings[:, i, :], self.embeddings[:, j, :]))  # [None, embedding_size]
                self.pair_wise_product = tf.stack(pair_wise_product_list)  # [embedding_size*(embedding_size + 1)/2, None, embedding_size]
                self.pair_wise_product = tf.transpose(self.pair_wise_product, perm=[1, 0, 2], name='pair_wise_product')  # [None, field_size*(field_size - 1)/2, embedding_size]

                self.y_second_order = tf.reshape(self.pair_wise_product, shape=[-1, int(self.field_size * (self.field_size - 1) / 2) * self.embedding_size])

            with tf.name_scope('out'):
                input_size = self.y_first_order.shape.as_list()[1] + self.y_second_order.shape.as_list()[1]
                glorot = np.sqrt(2.0 / (input_size + 1))
                weights['out'] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                    dtype=np.float32)
                biases['out_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)
                self.out = tf.concat([self.y_first_order, self.y_second_order], axis=1)
                self.out = tf.add(tf.matmul(self.out, weights['out']), biases['out_bias'])

            # loss
            if self.loss_type == 'logloss':
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)

            elif self.loss_type == 'mse':
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))

            # optimizer
            if self.optimizer_type == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            elif self.optimizer_type == 'adagrad':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)

            elif self.optimizer_type == 'gd':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                    self.loss)

            elif self.optimizer_type == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                            momentum=0.95).minimize(
                    self.loss)

            elif self.optimizer_type == 'rmsprop':
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def fit(self, feat_index, feat_val, label):
        '''

        :param feat_index: [[idx1_1, idx1_2,...], [idx2_1, idx2_2,...],...]
                            idxi_j is the feature index of feature field j of sample i in the training set
        :param feat_val: [[value1_1, value1_2,...], [value2_1, value2_2,...]...]
                        valuei_j is the feature value of feature field j of sample i in the training set
        :param label: [[label1], [label2], [label3], [label4],...]
        :return: None
        '''
        for epoch in range(self.epochs):
            for i in range(0, len(feat_index), self.batch_size):
                feat_index_batch = feat_index[i: i + self.batch_size]
                feat_val_batch = feat_val[i: i + self.batch_size]
                batch_y = label[i: i + self.batch_size]

                feed_dict = {
                    self.feature_index: feat_index_batch,
                    self.feature_value: feat_val_batch,
                    self.label: batch_y
                }
                cost, opt = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
            train_metric = self.evaluate(feat_index, feat_val, label)
            print('[%s] train-%s=%.4f' % (epoch + 1, self.metric_type, train_metric))

    def predict(self, feat_index, feat_val):
        feed_dict = {
            self.feature_index: feat_index,
            self.feature_value: feat_val
        }
        y_pred = self.sess.run(self.out, feed_dict=feed_dict)
        return y_pred

    def evaluate(self, feat_index, feat_val, label):
        y_pred = self.predict(feat_index, feat_val)
        if self.metric_type == 'auc':
            return roc_auc_score(label, y_pred)
        elif self.metric_type == 'logloss':
            return log_loss(label, y_pred)

if __name__ == '__main__':
    print('read dataset...')
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    y_train = pd.read_csv('data/y_train.csv')
    y_val = pd.read_csv('data/y_val.csv')

    continuous_feature = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    category_feature = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex',
                        'native_country']

    dataParse = DataParse(continuous_feature=continuous_feature, category_feature=category_feature)
    dataParse.FeatureDictionary(train, test)
    train_feature_index, train_feature_val = dataParse.parse(train)
    test_feature_index, test_feature_val = dataParse.parse(test)

    y_train = y_train.values.reshape(-1, 1)
    y_val = y_val.values.reshape(-1, 1)

    model = FFM(feature_size=dataParse.feature_size,
               field_size=dataParse.field_size,
               metric_type='auc',
               optimizer_type='adam')

    model.fit(train_feature_index, train_feature_val, y_train)

    test_metric = model.evaluate(test_feature_index, test_feature_val, y_val)
    print('test-auc=%.4f' % test_metric)