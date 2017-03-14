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

        # ey 1150, br 1840
        lstm = LSTMBuilder(reshaped.output, 1840, prefix = 'lstm',
                           W=W_lstm, U=U_lstm, b=b_lstm,
                           out_idx='last', rand_scheme = 'orthogonal')
        if dropout:
            dropped = DropoutBuilder(lstm.output, 0.4, iftrain, 'dropout')
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
    y_prob = []
    y_pred = []
    y_true = []
    for _, valid_index in iterator:
        y_pred.extend(f_pred_class(data[i][0]) for i in valid_index)
        y_true.extend(data[i][1] for i in valid_index)

    acc = metrics.accuracy_score(y_true, y_pred)

    return (1-acc)

def compute_metrics(f_pred_prob, f_pred_class, data, iterator, verbose=False):
    y_prob = []
    y_pred = []
    y_true = []
    for _, valid_index in iterator:
        prob_tmp = [f_pred_prob(data[i][0]) for i in valid_index]
        y_prob.extend( list(itertools.chain.from_iterable(prob_tmp)) )
        y_pred.extend(f_pred_class(data[i][0]) for i in valid_index)
        y_true.extend(data[i][1] for i in valid_index)

    acc = metrics.accuracy_score(y_true, y_pred)

    return acc, y_prob, y_pred, y_true

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
    optimizer=sgd,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
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

        history_errs = [] # record history of errors
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

                    iftrain.set_value(1.) # dropout

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
                                **best_p)
                        np.savez(save_file+'_err_history',
                                 history_errs=history_errs)
                        # TODO: save:
                        # model options
                        # pickle.dump(model_options,
                                   # open('%s.pkl' % save_file, 'wb'), -1)
                        print('Done.')

                    # Validation
                    if np.mod(uidx, validFreq) == 0:
                        iftrain.set_value(0.) # not dropout

                        train_err = pred_error(f_pred_prob, f_pred_class, train, kf)
                        valid_err = pred_error(f_pred_prob, f_pred_class, valid, kf_valid)
                        test_err = pred_error(f_pred_prob, f_pred_class, test, kf_test)

                        history_errs.append([train_err, valid_err, test_err])

                        if ( (best_p is None) or
                                (valid_err <= np.array(history_errs)[:,1].min()) ):
                            # pack best params for saving
                            best_p = pack_params(params)
                            bad_counter = 0

                        print('Train ', train_err, 'Valid ', valid_err,
                            'Test ', test_err)

                        if (len(history_errs) > patience and
                            valid_err >= np.array(history_errs)[:-patience, 1].min()):
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
        train_err = pred_error(f_pred_prob, f_pred_class, train, kf_train_sorted)
        valid_err = pred_error(f_pred_prob, f_pred_class, valid, kf_valid)
        test_err = pred_error(f_pred_prob, f_pred_class, test, kf_test)

        print( 'Err: Train ', train_err, 'Valid ', valid_err, 'Test ', test_err )
        history_errs.append([train_err, valid_err, test_err])

        if save_file:
            if best_p is None: # if best params not found
                best_p = pack_params(params) # save current params

            np.savez(save_file,
                    **best_p)
            np.savez(save_file+'_err_history',
                        history_errs=history_errs)

        print('The code run for %d epochs, with %f sec/epochs' % (
            (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
        print( ('Training took %.1fs' %
                (end_time - start_time)), file=sys.stderr)

        return train_acc, valid_acc, test_acc

    else: # testing only
        acc, prob, pred, true = compute_metrics(f_pred_prob, f_pred_class, test, kf_test)
        print( 'Test acc: ', acc)

        np.savez(save_file+'_metric',
                 acc=acc,
                 prob=prob,
                 pred=pred,
                 true=true
                 )

        return acc


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

        time_encoder='lstm',
        optimizer=adadelta,
        lrate=0.0001,
        gamma=0.9,

        batch_size=1,
        valid_batch_size=1,
        max_epochs=10000,
        patience=10000,
        dispFreq=100,
        validFreq=5000,

        weight_decay=False,
        use_dropout=True,

        distort=True,
        sigma=2,
        alpha=15,
        ds_limit=5,

        escape_train=True,
        num_tests=2000,
        # escape_train=False,
        # num_tests=None,

        saveFreq=2000,
        # save_file='../npz/br100_close_dstt_f8-3-3_s1-1_p2-1_d04_delta.npz.bkp',
        # reload_model_path='../npz/br100_close_dstt_f8-3-3_s1-1_p2-1_d04_delta.npz.bkp')

        save_file='../npz/br100_close_dstt_f8-3-3_s1-1_p2-1_d04_delta.npz.3.T-2000',
        reload_model_path='../npz/br100_close_dstt_f8-3-3_s1-1_p2-1_d04_delta.npz.3')

        # reload_model_path=None)
