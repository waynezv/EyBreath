#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import sys
import os
import time
import re
from collections import OrderedDict

import numpy as np

import theano as T
import theano.tensor as tensor
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, Dense, Convolution2D, \
    MaxPooling2D, AveragePooling1D, ZeroPadding2D, Dropout, \
    Flatten, LSTM, Input, Reshape
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

import eybreath_data_prepare as edp

DEBUG_OUTPUT = 0

# Set the random number generator's seed for consistency
SEED = 93492019
rng = np.random.RandomState(SEED)

"""
Keras model related.
"""
class NetBuilder:
    """
    Net builder with Keras.
    """
    def __init__(self, inp, lbl):
        nrow, ncol = inp.shape
        nout = lbl.shape[0]
        inp = inp.reshape((1, 1, nrow, ncol))
        lbl = lbl.reshape((1, nout))
        self.input = inp
        self.output = lbl

        self.model = Sequential()
        self.model.add(Convolution2D(
            10, nrow, 1,
            input_shape = (1, nrow, ncol),
            subsample = (1, 1)
        )
                       )
        self.model.add(Activation('relu'))
        self.model.add(Reshape((10, ncol)))
        self.model.add(LSTM(100))
        #self.model.add(AveragePooling1D(pool_length=2))
        self.model.add(Dense(nout))
        self.model.add(Activation('softmax'))

    def trainer(self, nepoch):
        self.model.compile(
            loss = 'categorical_crossentropy',
            optimizer = 'adam',
            metrics = ['accuracy']
        )
        if DEBUG_OUTPUT:
            print(self.model.layers[0].output_shape)
            print(self.model.layers[1].output_shape)
            print(self.model.layers[2].output_shape)
            print(self.model.layers[3].output_shape)
            print(self.model.layers[4].output_shape)
            print(self.model.layers[5].output_shape)

        hist = self.model.fit(
            self.input, self.output,
            nb_epoch = nepoch
        )

    def tester(self, x, y):
        nr, nc = x.shape
        no = y.shape[0]
        x = x.reshape((1, 1, nr, nc))
        y = y.reshape((1, no))
        eval = self.model.evaluate(x, y)

    def saver(self):
        pass

"""
Net related.
"""
def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)

class LSTMBuilder:
    """
    Net builder for LSTM.
    """
    def __init__(self, inp, prefix, rand_scheme = 'standnormal'):
        """
        :inp: (num_time_steps, num_samples, num_embedding_size)

        TODO:
        1. Batch & mask
        2. Output dim
        """

        n_tstep, n_samp, embd_size = inp.shape[0], inp.shape[1], inp.shape[2]
        embd_size = 15
        self.n_tstep = n_tstep
        self.n_samp = n_samp
        self.embd_size = embd_size

        W_val = np.concatenate([ortho_weight(embd_size),
                            ortho_weight(embd_size),
                            ortho_weight(embd_size),
                            ortho_weight(embd_size)], axis=1)
        W = T.shared(value = W_val, name = prefix + '_W', borrow = True)
        U_val = np.concatenate([ortho_weight(embd_size),
                            ortho_weight(embd_size),
                            ortho_weight(embd_size),
                            ortho_weight(embd_size)], axis=1)
        U = T.shared(value = U_val, name = prefix + '_U', borrow = True)
        b_val = np.zeros((4 * embd_size,), dtype = config.floatX)
        b = T.shared(value = b_val, name = prefix + '_b', borrow = True)

        self.input = inp
        self.W = W
        self.U = U
        self.b = b

        # Parallelize: process one batch: n_samp per n_tstep.
        self.inp_t = tensor.dot(self.input, self.W) + self.b
        '''
        rval, updates = T.scan(self._step,
                               sequences    = self.inp_t,
                               outputs_info = [
                                   tensor.zeros_like(self.inp_t),
                                   tensor.zeros_like(self.inp_t)
                                   ],
                               name = prefix + '_layers',
                               n_steps = n_tstep)
        '''

        rval, updates = T.scan(self._step,
                               sequences    = self.inp_t,
                               outputs_info = [
                                   tensor.alloc(
                                       np.asarray(0., dtype = config.floatX),
                                       n_samp,
                                       embd_size
                                       ),
                                   tensor.alloc(
                                       np.asarray(0., dtype = config.floatX),
                                       n_samp,
                                       embd_size
                                       )
                                   ],
                               name = prefix + '_layers',
                               n_steps = n_tstep)

        self.output = rval[0] # h: (n_tstep, n_samp, embd_size)

        self.f = T.function([inp], self.output, name = 'f_' + prefix)
        self.params = [self.W, self.U, self.b]

    def _step(self, x_, h_, c_):
        """
        :x_: W * x_t
        :h_: h_(t-1)
        :c_: c_(t-1)
        """
        preact = x_ + tensor.dot(h_, self.U)

        # Gates, outputs, states
        i = tensor.nnet.sigmoid(self._slice(preact, 0, self.embd_size))
        f = tensor.nnet.sigmoid(self._slice(preact, 1, self.embd_size))
        o = tensor.nnet.sigmoid(self._slice(preact, 2, self.embd_size))
        c = tensor.tanh(self._slice(preact, 3, self.embd_size))

        # States update
        c = f * c_ + i * c

        # Hidden updat
        h = o * tensor.tanh(c)

        return h, c

    def _slice(self, _x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

class DenseBuilder:
    """
    Net builder for Dense layer.
    """
    def __init__(self, inp, in_dim, out_dim, prefix,
                 W = None, b = None, rand_scheme = 'standnormal'):
        n_samp, m = inp.shape[0], inp.shape[1]

        if W == None:
            if rand_scheme == 'uniform':
                W_val = np.asarray(
                    rng.uniform(
                        low  = -np.sqrt(6. / (m + out_dim)),
                        high =  np.sqrt(6. / (m + out_dim)),
                        size = (m, out_dim)
                    ),
                    dtype = config.floatX
                )
            elif rand_scheme == 'standnormal':
                W_val = np.asarray(
                    rng.randn(in_dim, out_dim),
                    dtype = config.floatX
                )
            elif rand_scheme == 'orthogonal':
                pass
            elif rand_scheme == 'identity':
                pass
            W = T.shared(value = W_val, name = prefix + '_W', borrow = True)

        if b == None:
            b_val = np.zeros((out_dim,), dtype = config.floatX)
            b = T.shared(value = b_val, name = prefix + '_b', borrow = True)

        self.input = inp
        self.W = W
        self.b = b
        self.output = tensor.dot(self.input, W) + b

        self.f = T.function([inp], self.output, name = 'f_' + prefix)
        self.params = [self.W, self.b]


class ConvolutionBuilder:
    """
    Net builder for Convolution layer.
    """
    def __init__(self, inp, filter_shape, prefix, stride = (1, 1),
            W = None, b = None, rand_scheme = 'standnormal'):
        n_samp, n_ch, n_row, n_col = inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]
        n_filt, n_in_featmap, filt_hgt, filt_wdh = filter_shape

        if W == None:
            if rand_scheme == 'uniform':
                fan_in = n_in_featmap * filt_hgt * filt_wdh
                fan_out = n_filt * filt_hgt * filt_wdh // np.prod(stride)
                W_bound = np.sqrt(6. / (fan_in + fan_out))
                W_val = np.asarray(
                    rng.uniform(
                        low  = -W_bound,
                        high = W_bound,
                        size = filter_shape
                    ),
                    dtype = config.floatX
                )
            elif rand_scheme == 'standnormal':
                W_val = np.asarray(
                    rng.randn(n_filt, n_in_featmap, filt_hgt, filt_wdh),
                    dtype = config.floatX
                )
            elif rand_scheme == 'orthogonal':
                pass
            elif rand_scheme == 'identity':
                pass
            W = T.shared(value = W_val, name = prefix + '_W', borrow = True)

        if b == None:
            b_val = np.zeros((n_filt,), dtype = config.floatX)
            b = T.shared(value = b_val, name = prefix + '_b', borrow = True)

        self.input = inp
        self.W = W
        self.b = b
        self.output = conv2d(input = inp, filters = self.W, filter_shape = filter_shape,
                          subsample = stride) + self.b.dimshuffle('x', 0, 'x', 'x')

        self.f = T.function([inp], self.output, name = 'f_' + prefix)
        self.params = [self.W, self.b]


class PoolBuilder:
    """
    Net builder for pool layer.
    """
    def __init__(self, inp, pool_scheme):
        a = inp.shape[0]
        b = inp.shape[1]
        c = inp.shape[2]
        d = inp.shape[3]
        if pool_scheme == 'max':
            pass
        elif pool_scheme == 'mean':
            n_tstep = inp.shape[0].astype(config.floatX)
            self.output = inp.sum(axis=0) / n_tstep
        elif pool_scheme == 'random':
            pass

        self.f = T.function([inp], self.output, name = 'f_' + pool_scheme)


class ReshapeBuilder:
    def __init__(self, inp, reshape):
        a = inp.shape[0]
        b = inp.shape[1]
        c = inp.shape[2]
        d = inp.shape[3]
        #self.output  = inp.reshape((a, b, d)).dimshuffle(reshape)
        self.output  = inp.reshape((d, a, b))
        self.f = T.function([inp], self.output, name = 'f_reshape')


class ActivationBuilder:
    """
    Net builder for Activation layer.
    """
    def __init__(self, inp, activation):
        if activation == 'relu':
            self.output = tensor.nnet.relu(inp)
        elif activation == 'softmax':
            self.output = tensor.nnet.softmax(inp)
        elif activation == 'tanh':
            self.output = tensor.nnet.sigmoid(inp)
        elif activation == 'sigmoid':
            self.output = tensor.tanh(inp)

        self.f = T.function([inp], self.output, name = 'f_' + activation)


def get_cost(pred_prob, y):
    if pred_prob.ndim == 2:
        n_samp = pred_prob.shape[0]
    else:
        n_samp = 1

    off = 1e-8
    if pred_prob.dtype == 'float16':
        off = 1e-6

    cost = -tensor.log(pred_prob[tensor.arange(n_samp), y] + off).mean()
    return cost

def pred_class(pred_prob):
    if pred_prob.ndim == 2:
        return np.argmax(pred_prob, axis=1)
    return np.argmax(pred_prob)

model = OrderedDict() # dict: layer_name (string) -> layer (func)

def get_layer(layer_name):
    l = model[layer_name]
    return l

def build_model(time_encoder = 'lstm'):
    x = tensor.tensor4('x', dtype = config.floatX)
    y = tensor.scalar('y', dtype = 'int32')

    n_samp, n_ch, n_row, n_col = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    embd = ConvolutionBuilder(x, (15, 1, 463, 1), prefix = 'conv1')
    embd_a = ActivationBuilder(embd.output, 'relu')
    if time_encoder == 'lstm':
        reshaped = ReshapeBuilder(embd_a.output, (2, 0, 1))
        lstm = LSTMBuilder(reshaped.output, prefix = 'lstm')
        pooled = PoolBuilder(lstm.output, 'mean')
    elif time_encoder == 'tdnn':
        tdnn = ConvolutionBuilder(embd_a.output, (15, 15, 1, 3), prefix = 'tdnn')
    dense = DenseBuilder(pooled.output, 15, 53, prefix = 'dense1')
    dense_a = ActivationBuilder(dense.output, 'softmax')
    cost = get_cost(dense_a.output, y)
    pred = pred_class(dense_a.output)

    tx = np.zeros((1,1,463,20), dtype = 'float32')
    f1 = T.function([x], embd_a.output)
    print('conv out: ', f1(tx).shape)
    f2 = T.function([x], reshaped.output)
    print('reshape out: ', f2(tx).shape)
    f3 = T.function([x], lstm.output)
    print('lstm out:', f3(tx).shape)
    f4 = T.function([x], pooled.output)
    print('pool out: ', f4(tx).shape)
    f5 = T.function([x], dense_a.output)
    print('dense out: ', f5(tx).shape)
    f6 = T.function([x, y], cost)
    print('cost out: ', f6(tx, 5))
    f7 = T.function([x], pred)
    print('pred out: ', f7(tx))

    model = {
            'embd': embd,
            'lstm': lstm,
            'dense': dense
            }

    fpred_prob = T.function([x], dense_a.output, name='f_pred_prob')
    fpred = T.function([x], pred, name='f_pred_class')

    params = [
            embd.params[0], embd.params[1],
            lstm.params[0], lstm.params[1], lstm.params[2],
            dense.params[0], dense.params[1]
            ]

    grads = tensor.grad(cost, wrt = params)

    return x, y, fpred_prob, fpred, cost, params, grads, model


"""
Training related.
"""
def sgd(lr, params, grads, x, y, cost):
    """
    Stochastic Gradient Descent.
    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [T.shared(p.get_value() * 0., name='%s_grad' % str(k))
               for k, p in zip(range(len(params)), params)]
    gs_updt = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = T.function([x, y], cost, updates=gs_updt,
                                    name='sgd_f_grad_shared')

    para_updted = [(p, p - lr * g) for p, g in zip(params, gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = T.function([lr], [], updates=para_updted,
                               name='sgd_f_update')

    return f_grad_shared, f_update

def rmsprop(lr, params, grads, x, y, cost):
    """
    A variant of SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    pramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [T.shared(p.get_value() * np.asarray(0., dtype=config.floatX),
                                  name='%s_grad' % str(k))
                    for k, p in zip(range(len(params)), params)]
    running_grads = [T.shared(p.get_value() * np.asarray(0., dtype=config.floatX),
                                  name='%s_rgrad' % str(k))
                    for k, p in zip(range(len(params)), params)]
    running_grads2 = [T.shared(p.get_value() * np.asarray(0., dtype=config.floatX),
                                  name='%s_rgrad2' % str(k))
                      for k, p in zip(range(len(params)), params)]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = T.function([x, y], cost,
                               updates=zgup + rgup + rg2up,
                               name='rmsprop_f_grad_shared')

    updir = [T.shared(p.get_value() * np.asarray(0., dtype=config.floatX),
                      name='%s_updir' % k)
             for k, p in zip(range(len(params)), params)]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(params, updir_new)]
    f_update = T.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update

def get_minibatches_idx(num_samples, batch_size, shuffle = False):
    samp_idx_list = np.arange(num_samples, dtype = "int32")

    if shuffle:
        np.random.shuffle(samp_idx_list)

    num_batches = num_samples // batch_size
    mini_batches = [ samp_idx_list[i * batch_size :
                                   (i + 1) * batch_size]
                    for i in xrange(num_batches) ]
    leftover = num_batches * batch_size
    if (leftover != num_samples):
        mini_batches.append(samp_idx_list[leftover :])

    return zip(range(len(mini_batches)), mini_batches)

def pred_error(f_pred, data, iterator, verbose=False):
    """
    Compute the error.
    :f_pred: Theano function computing the prediction
            :x (tensor4)
            <- class (int32)
    :data: [(x(list), y(int)), ...]
    :iterator: list: [index, [sample_indexes]] @get_minibatches_idx
    """
    valid_err = 0
    valid_cnt = 0
    for _, valid_index in iterator:
        preds = np.array([f_pred(data[i][0]) for i in valid_index])
        targets = np.array([data[i][1] for i in valid_index])
        valid_err += np.count_nonzero(preds == targets)
        valid_cnt += len(valid_index)
    valid_err = 1. - np.asarray(valid_err, dtype = config.floatX) / valid_cnt

    return valid_err

def train_model(
    patience=50,  # Number of epoch to wait before early stop if no progress
    max_epochs=10000,  # The maximum number of epoch to run
    dispFreq=100,  # Display to stdout the training progress every N updates
    lrate=0.01,  # Learning rate for sgd (not used for adadelta and rmsprop)
    optimizer=sgd,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    validFreq=5000,  # Compute the validation error after this number of update.
    batch_size=1,  # The batch size during training.
    valid_batch_size=1,  # The batch size used for validation/test set.
):
    print('Preparing data')
    #dpath = './feat_constq'
    dpath = '/mingback/zhaowenbo/EyBreath/feat_constq'
    train, valid, test = edp.load_data(
        os.path.join(dpath ,'ey'), './ey.interested',
        shuffle = True, spk_smpl_thrd = 100
    )
    num_trains = len(train)
    num_vals = len(valid)
    num_tests = len(test)

    print('Building model')
    (x, y, f_pred_prob, f_pred, cost, params, grads, model) = build_model()

    f_cost = T.function([x, y], cost, name='f_cost')

    f_grad = T.function([x, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, params, grads,
                                        x, y, cost)

    print('Training')

    kf_valid = get_minibatches_idx(num_vals, batch_size)
    kf_test = get_minibatches_idx(num_tests, batch_size)

    print("%d train examples" % num_trains)
    print("%d valid examples" % num_vals)
    print("%d test examples" % num_tests)

    history_errs = []
    best_p = None
    bad_counter = 0

    if validFreq == -1:
        validFreq = len(train[0]) // batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in range(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(num_trains, batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1

                # Select the random examples for this minibatch
                #x, y = [train[t] for t in train_index]
                x, y = train[train_index[0]]
                n_samples += batch_size

                cost = f_grad_shared(x, y)
                f_update(lrate)

                if np.isnan(cost) or np.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if np.mod(uidx, dispFreq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                if np.mod(uidx, validFreq) == 0:
                    train_err = pred_error(f_pred, train, kf)
                    valid_err = pred_error(f_pred, valid, kf_valid)
                    test_err = pred_error(f_pred, test, kf_test)

                    history_errs.append([valid_err, test_err])

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

    kf_train_sorted = get_minibatches_idx(num_trains, batch_size)
    train_err = pred_error(f_pred, train, kf_train_sorted)
    valid_err = pred_error(f_pred, valid, kf_valid)
    test_err = pred_error(f_pred, test, kf_test)

    print( 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err )
    print('The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
    print( ('Training took %.1fs' %
            (end_time - start_time)), file=sys.stderr)
    return train_err, valid_err, test_err

def run_model():
    dpath = './feat_constq'
    read_instance(os.path.join(dpath ,'ey'), './ey.interested')

    num_samples = len(instance_dict)
    num_trains = int(np.floor(num_samples * 0.8))
    num_tests = num_samples - num_trains

    all_list = np.random.permutation(xrange(num_samples))
    train_list = all_list[: num_trains]
    dev_list = all_list[num_trains : num_samples]

    num_epoch = 10
    dev_j = 0
    while True:
        for i in xrange(num_trains):
            ins = instance_dict[train_list[i]]
            inp = np.asarray(ins.featvec, dtype = config.floatX)
            spk_id = ins.speaker_id
            lbl = np.asarray(np.zeros((805,)), dtype = config.floatX)
            lbl[spk_id] = 1.0

            nn = NetBuilder(inp, lbl)
            nn.trainer(num_epoch)
            if i % 50 == 0:
                tins = instance_dict[dev_list[dev_j]]
                tinp = np.asarray(tins.featvec, dtype = config.floatX)
                tspk_id = tins.speaker_id
                tlbl = np.asarray(np.zeros((805,)), dtype = config.floatX)
                tlbl[tspk_id] = 1.0
                nn.tester(tinp, tlbl)
                dev_j += 1


if __name__ == '__main__':
    # run_model()
    train_model()
