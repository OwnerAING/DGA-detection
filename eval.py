#  -*- coding: utf-8 -*-

import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, average_precision_score, roc_auc_score
import pandas as pd
import numpy as np


def parse_single_sampe(sample):
    record_defaults = [0] * 76
    parsed = tf.io.decode_csv(records=sample, record_defaults=record_defaults)
    x = tf.stack(parsed[0:75])
    # y = tf.stack(parsed[75])
    # x = tf.one_hot(x, depth=37)
    # y = tf.one_hot(y, depth=2)
    return x


def csv_dataset_reader(filename, has_header=True, shuffle_buffer_size=10000):

	dataset = tf.data.TextLineDataset(r'./' + str(filename) + '.csv') 
	if has_header:
		dataset = dataset.skip(1)
	# dataset = dataset.repeat(epoch)
	dataset = dataset.shuffle(shuffle_buffer_size)
	dataset = dataset.map(parse_single_sampe)
	# dataset = dataset.batch(batch_size)
	return dataset


def model_eval(model_path, x, y_label, threshold=0.5):
	result = {}
	model = tf.keras.models.load_model(model_path)
	y_pred_value = model.predict(x=x)
	y_pred = np.where(y_pred_value > threshold, 1, 0)
	result['cm'] = confusion_matrix(y_label, y_pred)
	result['f1_score'] = f1_score(y_label, y_pred)
	result['precision_score'] = precision_score(y_label, y_pred)
	result['recall_score'] = precision_score(y_label, y_pred)
	result['average_precision_score'] = average_precision_score(y_label, y_pred)
	result['roc_auc_score'] = roc_auc_score(y_label, y_pred)

	return result


testset = pd.read_csv('./test_ord_encode.csv')

x = testset.iloc[:, 0:75].values.astype(np.float32)
y = testset.iloc[:, -1].values


result = model_eval(r'./checkpoint', x, y)


for (x, y) in result.items():
	print((x, y))




#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#######              tf.keras.evaluate                   ########

##        865us/sample - loss: 0.0239 - accuracy: 0.9833       ##
##              [0.04613903951440131, 0.98328847]              ##

#######                                                  ########
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#######            customed model_eval                   #######

####                confusion matrix                        ####
##                 [[190987,   3229],                         ##
##                 [  2722, 159164]]                          ##

####          f1_score:  0.9816485187138235                 ####

####       precision_score:  0.9801161380108748             ####

####        recall_score:  0.9801161380108748               ####

####   average_precision_score:  0.9712800471538547         ####

####         roc_auc_score:  0.9832799399509491             ####

#######                                                 ########
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

