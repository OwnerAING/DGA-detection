# DGA-detection
DGA detection based on Deep Learning (CNN and GRU) (基于深度学习的DGA检测)

  This project implements the DGA detection algorithm based on CNN and GRU to replace the traditional manual feature machine learning model. This article improves the pre-training model (.pb). Its roc_auc_score is 0.9833
  The evaluation indicators of the model are as follows:

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

