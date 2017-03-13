#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import sys
import os
import time
import re
from collections import OrderedDict
import six.moves.cPickle as pickle

import numpy as np
import theano as T
import theano.tensor as tensor
from theano import config
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import itertools

import eybreath_data_prepare as edp

from w_net import LSTMBuilder, ConvolutionBuilder, DenseBuilder,\
    ActivationBuilder, PoolBuilder, ReshapeBuilder,\
    DropoutBuilder,\
    get_cost, pred_class,\
    sgd, rmsprop, adadelta

DEBUG_OUTPUT = 0

model = OrderedDict() # dict: layer_name (string) -> layer (func)

def get_layer(layer_name):
    l = model[layer_name]

    return l

def build_model(params, num_classes, dropout=False, time_encoder='lstm'):
    """
    Build CNN-LSTM model.

    :params: list of params, empty or reloaded from saved model
    """

    print('Using encoder: ', time_encoder)

    if params: # not empty
        if time_encoder == 'lstm':
            W_embd  = params[0]
            b_embd  = params[1]
            W_lstm  = params[2]
            U_lstm  = params[3]
            b_lstm  = params[4]
            W_dense = params[5]
            b_dense = params[6]

        elif time_encoder == 'tdnn':
            W_embd  = params[0]
            b_embd  = params[1]
            W_tdnn  = params[2]
            b_tdnn  = params[3]
            W_dense = params[4]
            b_dense = params[5]

        print('Params from previously saved model loaded successfully!')

    else: # init params
            W_embd  = None
            b_embd  = None
            W_conv  = None
            b_conv  = None
            W_tdnn  = None
            b_tdnn  = None
            W_lstm  = None
            U_lstm  = None
            b_lstm  = None
            W_lstm2  = None
            U_lstm2  = None
            b_lstm2  = None
            W_dense = None
            b_dense = None

    x = tensor.tensor4('x', dtype=config.floatX)
    y = tensor.scalar('y', dtype='int32')

    n_samp, n_ch, n_row, n_col = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

    iftrain = T.shared(np.asarray(0, dtype=config.floatX)) # used in inverted dropout

    embd = ConvolutionBuilder(x, (8, 1, 3, 3), prefix = 'embd',
                              stride=(1,1),
                              W=W_embd, b=b_embd, rand_scheme='standnormal')
    embd_a = ActivationBuilder(embd.output, 'relu')

    if time_encoder == 'lstm':
        pooled = PoolBuilder(embd_a.output, 'max', ds=(2,1))
        reshaped = ReshapeBuilder(pooled.output, prefix='reshape', shape=(3,0,(1,2)))

        lstm = LSTMBuilder(reshaped.output, 1840, prefix = 'lstm',
                           W=W_lstm, U=U_lstm, b=b_lstm,
                           out_idx='last', rand_scheme = 'orthogonal')
        if dropout:
            dropped = DropoutBuilder(lstm.output, 0.6, iftrain, 'dropout')
            dense = DenseBuilder(dropped.output, 1840, num_classes, prefix = 'dense',
                    W=W_dense, b=b_dense, rand_scheme='standnormal')

        else:
            dense = DenseBuilder(lstm.output, 920, num_classes, prefix = 'dense',
                    W=W_dense, b=b_dense, rand_scheme='standnormal')

    elif time_encoder == 'tdnn':
        tdnn = ConvolutionBuilder(embd_a.output, (5, 5, 1, 2), prefix = 'tdnn',
                              stride=(1,1),
                              W=W_tdnn, b=b_tdnn, rand_scheme='standnormal')
        reshaped = ReshapeBuilder(tdnn.output, prefix='reshape', shape=(1,2,3))
        pooled = PoolBuilder(reshaped.output, 'mean', axis=2)
        reshaped2 = ReshapeBuilder(pooled.output, prefix='reshape2')
        dense = DenseBuilder(reshaped2.output, 1570, num_classes, prefix = 'dense',
                             W=W_dense, b=b_dense, rand_scheme='standnormal')

    dense_a = ActivationBuilder(dense.output, 'softmax')
    cost = get_cost(dense_a.output, y)
    pred = dense_a.output.argmax()

    # if DEBUG_OUTPUT:
    tx = np.zeros((1,1,463,20), dtype = 'float32')
    f1 = T.function([x], embd_a.output)
    print('1. embd out: ', f1(tx).shape)

    f1_ = T.function([x], pooled.output)
    print('2. pool out: ', f1_(tx).shape)
    f2 = T.function([x], reshaped.output)
    print('3. reshape out: ', f2(tx).shape)
    f3 = T.function([x], lstm.output)
    print('4. lstm out:', f3(tx).shape)
    f4 = T.function([x], dropped.output)
    print('5. dropped out:', f4(tx).shape)
    f5 = T.function([x], dense.output)
    print('6. dense out: ', f5(tx).shape)
    f7 = T.function([x, y], cost)
    print('7. cost out: ', f7(tx, 0))
    f8 = T.function([x], pred)
    print('8. pred out: ', f8(tx))

    f_pred_prob = T.function([x], dense_a.output, name='f_pred_prob')
    f_cost = T.function([x, y], cost, name='f_cost')
    f_pred_class = T.function([x], pred, name='f_pred_class')

    if time_encoder == 'lstm':
        params = [
                embd.params[0], embd.params[1],
                lstm.params[0], lstm.params[1], lstm.params[2],
                dense.params[0], dense.params[1]
                ]

    elif time_encoder == 'tdnn':
        params = [
                embd.params[0], embd.params[1],
                tdnn.params[0], tdnn.params[1],
                dense.params[0], dense.params[1]
                ]

    # TODO: Improvement for clearity
    # save params as a dict
    # and grads = tensor.grad(cost, wrt=list(params.values()))
    grads = tensor.grad(cost, wrt = params)

    # TODO: save model config
    if time_encoder == 'lstm':
        model = {
                'embd': embd,
                'lstm': lstm,
                'dense': dense
                }

    elif time_encoder == 'tdnn':
        model = {
                'embd': embd,
                'tdnn': tdnn,
                'dense': dense
                }

    return x, y, f_pred_prob, f_cost, f_pred_class,\
        cost, params, grads, model, iftrain

def get_minibatches_idx(num_samples, batch_size, shuffle = False):
    samp_idx_list = np.arange(num_samples, dtype = "int32")

    if shuffle:
        np.random.shuffle(samp_idx_list)

    num_batches = num_samples // batch_size
    mini_batches = [ samp_idx_list[i * batch_size :
                                   (i + 1) * batch_size]
                    for i in range(num_batches) ]

    leftover = num_batches * batch_size
    if (leftover != num_samples):
        mini_batches.append(samp_idx_list[leftover :])

    return zip(range(len(mini_batches)), mini_batches)

def pred_error(f_pred_prob, f_pred_class, data, iterator, verbose=False):
    """
    Compute prediction error.

    :f_pred_class: Theano function computing the prediction
            :x (tensor4)
            <- class (int32)
    :data: [(x(list), y(int)), ...]
    :iterator: list: [index, [sample_indexes]] @get_minibatches_idx
    """
    valid_err = 0
    valid_cnt = 0
    y_prob = []
    y_pred = []
    y_true = []
    for _, valid_index in iterator:
        # TODO: bug TypeError: iteration over a 0-d array
        # y_prob: [ array[[]] ]
        y_prob.extend(f_pred_prob(data[i][0]) for i in valid_index)
        y_pred.extend(f_pred_class(data[i][0]) for i in valid_index)
        y_true.extend(data[i][1] for i in valid_index)

    acc = metrics.accuracy_score(y_true, y_pred)
    conf_mat = metrics.confusion_matrix(y_true, y_pred)
    # rept = metrics.classification_report(y_true, y_pred,
                                         # target_names=[str(i) for i in range(len(y_true))])

    # fpr, tpr, thrshld = metrics.roc_curve(y_true, y_prob)

    # More metrics
    # prec = metrics.precision_score(y_true, y_pred, pos_label=1, average='binary')
    # recl = metrics.recall_score(y_true, y_pred, pos_label=1, average='binary')
    # f_sc = metrics.f1_score(y_true, y_pred, pos_label=1, average='binary')
    # precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

     # return acc, conf_mat, rept, zip(fpr, tpr, thrshld)
    return acc, conf_mat

def pack_params(params_list):
    """
    Pack the parameters for saving.

    :params_list: a list of theano shared variables
    """
    params = OrderedDict()
    for i, v in enumerate(params_list):
        params[str(i)] = v.get_value()

    return params

def load_params(model_path):
    """
    Load params from saved model.

    :model_path: (string) path saving the model params
    <- list of params
    """
    # TODO: Better practice to avoid key-value conflicts
    # pass in params as dict
    # and check if loaded params have compatible keys.

    # loaded params is an OrderedDict
    p = np.load(model_path)
    pp = [
            p['0'],
            p['1'],
            p['2'],
            p['3'],
            p['4'],
            p['5'],
            p['6']
            ]

    return pp

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

def train_model(
    dpath = '../feat_constq',
    dataname='ey',
    datalist='./ey.interested',
    sample_threshold=100,
    num_classes=53,

    use_unknown=False,
    unknown_class=None,

    use_dct=False,

    distort=False,
    sigma=None,
    alpha=None,
    ds_limit=5,

    patience=100,  # Number of epoch to wait before early stop if no progress
    max_epochs=10000,  # The maximum number of epoch to run
    dispFreq=100,  # Display to stdout the training progress every N updates
    time_encoder='lstm', # lstm or tdnn
    optimizer=rmsprop,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    gamma=0.9, # for adaxxx
    validFreq=5000,  # Compute the validation error after this number of update.
    batch_size=1,  # The batch size during training.
    valid_batch_size=1,  # The batch size used for validation/test set.

    weight_decay=False,
    use_dropout=True,

    escape_train=False,
    num_tests=None,

    save_file='model.npz',
    saveFreq=2000,
    reload_model_path=None
):
    # Options for model
    model_options = locals().copy

    print('Building model')
    if reload_model_path: # if reload model
        print('loading model ...')
        params = load_params(reload_model_path)
        print('Done.')

    else: # init params
        params = []

    (x, y, f_pred_prob, _, f_pred_class, cost, params, grads, model, iftrain) = build_model(
        params, num_classes, dropout=use_dropout, time_encoder=time_encoder)

    # TODO:
    if weight_decay:
        pass

    f_grad_shared, f_update = optimizer(params, grads, x, y, cost,
                                        lr=lrate, gamma=gamma)

    print('Preparing data')
    if not escape_train: # normal training
        train, valid, test = edp.load_data(
            os.path.join(dpath, dataname), datalist,
            shuffle=True, num_tests=num_tests,
            use_unknown=use_unknown, unknown_class=unknown_class,
            use_dct=use_dct,
            distort=distort, sigma=sigma, alpha=alpha, ds_limit=ds_limit
        )

        num_trains = len(train)
        num_vals = len(valid)
        num_tests = len(test)

        kf_valid = get_minibatches_idx(num_vals, batch_size)
        kf_test = get_minibatches_idx(num_tests, batch_size)

        print("%d train examples" % num_trains)
        print("%d valid examples" % num_vals)
        print("%d test examples" % num_tests)

    else: # testing only
        assert  num_tests is not None, "Number of tests cannot be None!"
        test = edp.load_data(
            os.path.join(dpath, dataname), datalist,
            shuffle=True, num_tests=num_tests,
            use_unknown=use_unknown, unknown_class=unknown_class,
            use_dct=use_dct,
            distort=distort, sigma=sigma, alpha=alpha, ds_limit=ds_limit
        )

        kf_test = get_minibatches_idx(num_tests, batch_size)

        print("%d test examples" % num_tests)


    if not escape_train: # normal training
        print('Training')


        if validFreq == -1:
            validFreq = num_trains // batch_size

        if saveFreq == -1:
            saveFreq = num_trains // batch_size

        history_errs = []
        best_p = None # best set of params
        bad_counter = 0 # count non-decreasing trains
        uidx = 0  # the number of updates done
        estop = False  # early stop
        start_time = time.time()

        try:
            for eidx in range(max_epochs):
                n_samples = 0

                # Get new shuffled index for the training set.
                kf = get_minibatches_idx(num_trains, batch_size, shuffle=True)

                for _, train_index in kf:
                    uidx += 1
                    iftrain.set_value(1.)

                    # Select the random examples for this minibatch
                    x, y = train[train_index[0]]
                    n_samples += batch_size

                    # Update params and grads
                    cost = f_grad_shared(x, y)
                    f_update()

                    if np.isnan(cost) or np.isinf(cost):
                        print('bad cost detected: ', cost)
                        return 1., 1., 1.

                    # Display
                    if np.mod(uidx, dispFreq) == 0:
                        print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                    # Save
                    if save_file and np.mod(uidx, saveFreq) == 0:
                        print('Saving model...')
                        if best_p is None:
                            best_p = pack_params(params)
                        np.savez(save_file,
                            # history_errs=history_errs,
                                **best_p)
                        #pickle.dump(model_options,
                        #            open('%s.pkl' % save_file, 'wb'), -1)
                        print('Done.')

                    # Validation
                    if np.mod(uidx, validFreq) == 0:
                        iftrain.set_value(0.)

                        train_acc, _ = pred_error(f_pred_prob, f_pred_class, train, kf)
                        valid_acc, _ = pred_error(f_pred_prob, f_pred_class, valid, kf_valid)
                        test_acc, _ = pred_error(f_pred_prob, f_pred_class, test, kf_test)

                        train_err = 1 - train_acc
                        valid_err = 1 - valid_acc
                        test_err = 1 - test_acc

                        history_errs.append([valid_err, test_err])

                        if ( (best_p is None) or
                                (valid_err <= np.array(history_errs)[:,0].min()) ):
                            # pack best params for saving
                            best_p = pack_params(params)
                            bad_counter = 0

                        print('Train ', train_err, 'Valid ', valid_err,
                            'Test ', test_err)

                        if (len(history_errs) > patience and
                            valid_err >= np.array(history_errs)[:-patience, 0].min()):
                            bad_counter += 1
                            if bad_counter > patience:
                                print('Early Stop!')
                                estop = True
                                break

                print('Seen %d samples' % n_samples)

                if estop:
                    break

        except KeyboardInterrupt:
            print("Training interupted")

        end_time = time.time()

        iftrain.set_value(0.)

        kf_train_sorted = get_minibatches_idx(num_trains, batch_size)
        train_acc, train_conf_mat = pred_error(f_pred_prob, f_pred_class, train, kf_train_sorted)
        valid_acc, valid_conf_mat = pred_error(f_pred_prob, f_pred_class, valid, kf_valid)
        test_acc, test_conf_mat = pred_error(f_pred_prob, f_pred_class, test, kf_test)

        print( 'Acc: Train ', train_acc, 'Valid ', valid_acc, 'Test ', test_acc )
        print( 'Confusion matrix: Train ', train_conf_mat, 'Valid ', valid_conf_mat, 'Test ', test_conf_mat )
        # TODO
        # print( 'Report: Train ', train_rept, 'Valid ', valid_rept, 'Test ', test_rept)

        if save_file:
            # if best params not found
            if best_p is None:
                # save current params
                best_p = pack_params(params)
            np.savez(save_file,
                    #train_err=train_err, # TODO
                    #valid_err=valid_err,
                    #test_err=test_err,
                    #history_errs=history_errs,
                    **best_p)

        print('The code run for %d epochs, with %f sec/epochs' % (
            (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
        print( ('Training took %.1fs' %
                (end_time - start_time)), file=sys.stderr)

        return train_acc, valid_acc, test_acc

    else: # testing only
        test_acc, test_conf_mat = pred_error(f_pred_prob, f_pred_class, test, kf_test)

        print( 'Acc: Test ', test_acc )
        print( 'Confusion matrix: Test ', test_conf_mat )
        # print( 'Report: Test ', test_rept)

        np.set_printoptions(precision=2)
        class_names = [str(i) for i in range(num_classes)]
        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names,
        title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
        title='Normalized confusion matrix')

        plt.show()

        return test_acc


if __name__ == '__main__':
    train_model(
        dpath = '../feat_constq/',
        # dpath = '~/Downloads/Data/Rita/EyBreath/data/feat_constq/',
        dataname='breath',
        datalist='./breath_selected_100',
        sample_threshold=100,
        num_classes=44,
        # ey: 100:53, 200:22, 300:14, 500:4
        # br: 100:44, 200:20

        use_unknown=False,
        unknown_class=None,

        use_dct=False,

        distort=True,
        sigma=2,
        alpha=15,
        ds_limit=2,

        patience=10000,
        time_encoder='lstm',
        optimizer=adadelta,
        lrate=0.0001,
        gamma=0.9,
        use_dropout=True,
        validFreq=20000,

        # escape_train=True,
        # num_tests=1000,
        escape_train=False,
        num_tests=None,

        save_file='../npz/br100_close_et2_f8-3-3_s1-1_p2-1_d06_delta.npz',
        reload_model_path=None)
        # reload_model_path='../npz/br100_closeset_dstt_lstm_f5-3-3_s1-1_p2-1_d05_delta.npz')
