#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import itertools
from collections import OrderedDict

def plot_confusion_matrix(cm, classes,
                          normalize=False, title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def read_log(log):
    """
    Read saved training log.
    """
    record = np.load(log)

    record_dict = OrderedDict()

    for k, v in record:
        record_dict[k] = v

    return record_dict


if __name__ == '__main__':
    # Plot training and validation errors
    # history = read_log('')

    # Compute metrics
    met = read_log('')
    acc = metrics.accuracy_score(y_true, y_pred)
    conf_mat = metrics.confusion_matrix(y_true, y_pred)
    rept = metrics.classification_report(y_true, y_pred,
                                         target_names=[str(i) for i in range(len(y_true))])

    fpr, tpr, thrshld = metrics.roc_curve(y_true, y_prob)

    prec = metrics.precision_score(y_true, y_pred, pos_label=1, average='binary')
    recl = metrics.recall_score(y_true, y_pred, pos_label=1, average='binary')
    f_sc = metrics.f1_score(y_true, y_pred, pos_label=1, average='binary')
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    print( 'Confusion matrix: Train ', train_conf_mat, 'Valid ', valid_conf_mat, 'Test ', test_conf_mat )
    print( 'Report: Train ', train_rept, 'Valid ', valid_rept, 'Test ', test_rept)

    # Plot non-normalized confusion matrix
    np.set_printoptions(precision=2)
    class_names = [str(i) for i in range(num_classes)]

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
    title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    title='Normalized confusion matrix')

    plt.show()
