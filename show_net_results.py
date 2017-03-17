#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
from scipy import interp
import sklearn.metrics as metrics
import itertools
from collections import OrderedDict
import scipy.io as sio

import matplotlib
matplotlib.use('pdf')
from matplotlib import rc
import matplotlib.pyplot as plt

def change_diag(diag):
    def passer(fn):
        def wrapper(*args, **kwargs):
            return fn(*args, diag_only=diag)
        return wrapper
    return passer

def change_cmap(cmap):
    def passer(fn):
        def wrapper(*args, **kwargs):
            return fn(*args, cmap=cmap)
        return wrapper
    return passer

# decorator, just for fun
@change_diag(True)
@change_cmap(plt.cm.jet) # Jet, Blues, Greys, PuBuGn
def plot_confusion_matrix(cm, num_classes, diag_only=False,
                          normalize=False, cmap=plt.cm.Blues):
    """
    Plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if diag_only: # only print diagonal texts
        for i in range(cm.shape[0]):
            plt.text(i, i, cm[i,i],
                    horizontalalignment="center",
                    color="gray" if cm[i,i] > cm.max()/2. else "white",
                    size=6)

    else:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if cm[i,j] > 0:
                plt.text(j, i, cm[i,j],
                        horizontalalignment="center",
                        color="gray" if cm[i,j] > cm.max()/2. else "white",
                        size=6)

    plt.colorbar()
    plt.ylabel('True label', size=18)
    plt.xlabel('Predicted label', size=18)
    plt.tick_params(
        axis='both', # x and y
        which='both',      # both major and minor ticks are affected
        top='off',         # ticks along the top edge are off
        bottom='off',      # ticks along the bottom edge are off
        left='off',
        right='off',
        labelleft='off',
        labelbottom='off') # labels along the bottom edge are off

    fig.set_tight_layout(True)
    fig.savefig('conf_mat.eps')
    plt.close()

def plot_speaker_dist(speaker_list, num_classes, save_to):
    sl = [int(s.split(' ')[2]) for s in open(speaker_list)]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(sl, num_classes, normed=0, cumulative=0, histtype='bar',
            align='mid', rwidth=0.8, color='steelblue', alpha=0.75)
    # aqua, navy, cornflowerblue, steelblue,
    # darkcyan, lightseagreen, teal,
    # slategray, darkslategray,
    # magenta, deeppink, darkorange

    ax.set_xlim((0, num_classes-1))
    ax.set_ylim((0, 1000))
    ax.set_xticks(np.arange(0, num_classes, 4))
    ax.set_yticks(np.arange(0, 1000, 100))

    ax.tick_params(
        axis='both', # x and y
        which='both',      # both major and minor ticks are affected
        labelsize=16,
        top='off',         # ticks along the top edge are off
        bottom='off',      # ticks along the bottom edge are off
        right='off',
        labelbottom='on') # labels along the bottom edge are off

    ax.set_xlabel('Speaker classes', size=18)
    ax.set_ylabel('Number of samples', size=18)

    fig.set_tight_layout(True)
    fig.savefig(save_to+'.eps')
    plt.close()

def read_log(log):
    """
    Read saved training log.
    """
    record = np.load(log)

    record_dict = OrderedDict()

    for k in record.files:
        record_dict[k] = record[k]

    return record_dict

def gen_label_mat(true, num_classes):
    """Generate a matrix of true labels, with rows as samples and cols
    as classes. Each entry is binary. For each row, the k-th entry being
    1 indicates this sample belongs to the k-th class.

    :true: (list[int]) list of true labels
    :num_classes: (int)
    """
    num_samp = len(true)
    label_mat = []

    for i in xrange(num_samp):
        tmp = np.zeros((num_classes,))
        tmp[true[i]] = 1
        label_mat.append(tmp.tolist())

    return label_mat


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    np.set_printoptions(threshold='nan')

    save_to_mat = 0 # save data to mat file
    plot_trn_hist = 0 # plot training history
    plot_conf_mat = 1 # plot confusion matrix
    plot_spk_dist = 1 # plot speaker distribution

    # Plot training and validation errors
    if plot_trn_hist:
        pass

    # Compute ROC
    objs = ['ey', 'breath']
    fig = plt.figure(figsize=(8, 6)) # size in inches
    colors = ['aqua', 'magenta']
    # aqua, navy, cornflowerblue, steelblue,
    # darkcyan, lightseagreen, teal,
    # slategray, darkslategray,
    # magenta, deeppink, darkorange

    for idx in xrange(2):
        obj = objs[idx]

        if obj == 'ey':
            met = read_log('./ey100_closeset_dstt_lstm_f5-3-3_s1-1_p2-1_d05_delta.npz.1.T-2000_metric.npz')
            num_classes = 53

        elif obj == 'breath':
            met = read_log('./br100_close_dstt_f8-3-3_s1-1_p2-1_d04_delta.npz.3.T-2000_metric.npz')
            num_classes = 44

        acc = met['acc']
        y_prob = met['prob']
        y_pred = met['pred']
        y_true = met['true']

        # Save to mat
        if save_to_mat:
            mat_to_save = {'y_prob':y_prob, 'y_pred':y_pred, 'y_true':y_true}
            sio.savemat(obj, mat_to_save, format='5')

        # Acc
        acc = metrics.accuracy_score(y_true, y_pred)
        print('acc: ', acc)

        # ROC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        y_true_mat = gen_label_mat(y_true, num_classes)

        # Compute fpr, tpr and auc for each class
        for c in xrange(num_classes):
            fpr[c], tpr[c], _ = metrics.roc_curve(y_true_mat[c], y_prob[c],
                                                pos_label=1)
            roc_auc[c] = metrics.auc(fpr[c], tpr[c])

        # Compute micro-average ROC and AUC
        fpr["micro"], tpr["micro"], _ = metrics.roc_curve(np.array(y_true_mat).ravel(),
                                                        np.array(y_prob).ravel())
        roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
        print('auc micro: ', roc_auc['micro'])

        # Compute macro-average ROC and AUC
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= num_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
        print('auc macro: ', roc_auc['macro'])

        # Plot ROC curve
        cl = colors[idx]
        plt.plot(fpr["macro"], tpr["macro"],
                label='/' + obj + '/' + ', area = {0:0.3f}'
                 ''.format(roc_auc["macro"]),
                color=cl, linestyle='-', linewidth=3)

    plt.plot([0, 1], [0, 1], color='black', linestyle='--', lw=3)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(np.arange(0, 1.1, 0.1),
               ['0', '0.1', '0.2', '0.3', '0.4', '0.5',
                      '0.6', '0.7', '0.8', '0.9', '1', '1.1'],
               fontsize=16)
    plt.yticks(np.arange(0, 1.1, 0.1),
               ['0', '0.1', '0.2', '0.3', '0.4', '0.5',
                      '0.6', '0.7', '0.8', '0.9', '1', '1.1'],
               fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.legend(loc="lower right", fontsize=18)

    fig.set_tight_layout(True)
    fig.savefig('roc.jpg')
    plt.close()

    # Confusion matrix
    if plot_conf_mat:
        conf_mat = metrics.confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(conf_mat, num_classes, diag_only=True)

    # Speaker distribution
    if plot_spk_dist:
        plot_speaker_dist('./ey_selected_100', 53, 'ey_hist')
        plot_speaker_dist('./breath_selected_100', 44, 'br_hist')
