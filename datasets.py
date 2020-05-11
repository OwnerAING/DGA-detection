# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf


def parse_single_sampe(sample):
    record_defaults = [0] * 76
    parsed = tf.io.decode_csv(records=sample, record_defaults=record_defaults)
    x = tf.stack(parsed[0:75])
    y = tf.stack(parsed[75])
    # x = tf.one_hot(x, depth=37)
    # y = tf.one_hot(y, depth=2)
    return x, y


def csv_dataset_reader(filename, batch_size, epoch, has_header=True, shuffle_buffer_size=10000):

    if has_header:
        dataset = tf.data.TextLineDataset(r'./' + str(filename) + '.csv').skip(1)
    else:
        dataset = tf.data.TextLineDataset(r'./' + str(filename) + '.csv')
    
    dataset = dataset.repeat(epoch)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_single_sampe)
    dataset = dataset.batch(batch_size)
    return dataset


def train_val_test_data_split(filename, test_size=0.2, val_size=0.2):

	cols = ['domain%d' %d for d in range(0, 75)] + ['label']
	data = pd.read_csv(r'./' + str(filename) + '.csv')
	x = data.iloc[:, 0:-1]
	y = data.iloc[:, -1]

	x_train_and_val,x_test, y_train_and_val, y_test = train_test_split(x, y, test_size=test_size)
	x_train, x_val, y_train, y_val = train_test_split(x_train_and_val, y_train_and_val, test_size=val_size)

	train = pd.concat([x_train, y_train], axis = 1)
	train.columns = cols
	train.to_csv('./train_ord_encode.csv', index=False)
	test = pd.concat([x_test, y_test], axis = 1)
	test.columns = cols
	test.to_csv('./test_ord_encode.csv', index=False)

	val = pd.concat([x_val, y_val], axis = 1)
	val.columns = cols
	val.to_csv('./val_ord_encode.csv', index=False)