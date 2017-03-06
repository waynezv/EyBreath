#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import sys
import os
from collections import OrderedDict
from operator import mul

import numpy as np

import theano as T
import theano.tensor as tensor
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams

# Set the random number generator's seed for consistency
# Set seed for the random numbers
np.random.seed(1234)
rds = np.random.RandomState(1234)
# Generate a theano RandomStreams
rng = RandomStreams(rds.randint(999999))

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
    def __init__(self, inp, embd_size, prefix,
                 W=None, U=None, b=None, out_idx='all', rand_scheme = 'orthogonal'):
        """
        :inp: (num_time_steps, num_samples, num_embedding_size)

        TODO:
        1. Batch & mask
        2. Output dim
        """

        n_tstep, n_samp, _ = inp.shape[0], inp.shape[1], inp.shape[2]
        self.n_tstep = n_tstep
        self.n_samp = n_samp
        self.embd_size = embd_size
        if W is None:
            W_val = np.concatenate([ortho_weight(embd_size),
                                ortho_weight(embd_size),
                                ortho_weight(embd_size),
                                ortho_weight(embd_size)], axis=1)
        else:
            W_val = W

        W = T.shared(value = W_val, name = prefix + '_W', borrow = True)

        if U is None:
            U_val = np.concatenate([ortho_weight(embd_size),
                                ortho_weight(embd_size),
                                ortho_weight(embd_size),
                                ortho_weight(embd_size)], axis=1)
        else:
            U_val = U

        U = T.shared(value = U_val, name = prefix + '_U', borrow = True)

        if b is None:
            b_val = np.zeros((4 * embd_size,), dtype = config.floatX)
        else:
            b_val = b

        b = T.shared(value = b_val, name = prefix + '_b', borrow = True)

        self.input = inp
        self.W = W
        self.U = U
        self.b = b

        # Parallelize: process one batch: n_samp per n_tstep.
        self.inp_t = tensor.dot(self.input, self.W) + self.b

        # order to scan:
        # sequences (if any), prior result(s) (if needed), non-sequences (if any)
        # here can actually pass shared variables W U b to scan
        # as non-sequences to improve efficiency.
        rval, updates = T.scan(self._step,
                               sequences    = self.inp_t,
                               outputs_info = [
                                   tensor.alloc(
                                       np.asarray(0., dtype = config.floatX),
                                       n_samp,
                                       embd_size
                                       ),
                                   # initial state has the same shape and dtype as output
                                   tensor.alloc(
                                       np.asarray(0., dtype = config.floatX),
                                       n_samp,
                                       embd_size
                                       )
                                   ],
                               non_sequences=[self.U],
                               name = prefix + '_layers',
                               n_steps = n_tstep,
                               strict=False)

        if out_idx == 'all':
            self.output = rval[0] # h: (n_tstep, n_samp, embd_size)
        elif out_idx == 'last':
            self.output = rval[0][-1] # h: (n_samp, embd_size)

        self.f = T.function([inp], self.output, name = 'f_' + prefix)
        self.params = [self.W, self.U, self.b]

    def _step(self, x_, h_, c_, U):
        """
        :x_: W * x_t
        :h_: h_(t-1)
        :c_: c_(t-1)
        """
        preact = x_ + tensor.dot(h_, U)

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

        if W is None:
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
                    np.random.randn(in_dim, out_dim),
                    dtype = config.floatX
                )
            elif rand_scheme == 'orthogonal':
                pass
            elif rand_scheme == 'identity':
                pass
        else:
            W_val = W

        W = T.shared(value = W_val, name = prefix + '_W', borrow = True)

        if b is None:
            b_val = np.zeros((out_dim,), dtype = config.floatX)
        else:
            b_val = b
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

        if W is None:
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
                    np.random.randn(n_filt, n_in_featmap, filt_hgt, filt_wdh),
                    dtype = config.floatX
                )
            elif rand_scheme == 'orthogonal':
                pass
            elif rand_scheme == 'identity':
                pass
        else:
            W_val = W

        W = T.shared(value = W_val, name = prefix + '_W', borrow = True)

        if b is None:
            b_val = np.zeros((n_filt,), dtype = config.floatX)
        else:
            b_val = b

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
    def __init__(self, inp, pool_scheme, ds=(2,2), axis=None):
        a = inp.shape[0]
        b = inp.shape[1]
        c = inp.shape[2]
        shape_map = {0:a, 1:b, 2:c}

        if pool_scheme == 'max':
            self.output = pool.pool_2d(inp, ds, st=None, padding=(0,0), mode='max',
                    ignore_border=True)

        if pool_scheme == 'sum':
            self.output = pool.pool_2d(inp, ds, st=None, padding=(0,0), mode='sum')

        elif pool_scheme == 'mean':
            if axis != None:
                n_tstep = shape_map[axis].astype(config.floatX)
                self.output = inp.sum(axis=axis) / n_tstep
            else:
                self.output = None

        elif pool_scheme == 'random':
            pass

        self.f = T.function([inp], self.output, name = 'f_' + pool_scheme)

class DropoutBuilder:
    def __init__(self, inp, p, iftrain, prefix):
        # With probability p to set activations to 0
        mask = rng.binomial(inp.shape, n=1, p=1-p, dtype=inp.dtype)
        # Perform inverted dropout
        drop = tensor.switch(mask, inp, 0) / (1-p)
        self.output = tensor.switch(iftrain, drop, inp)
        self.f = T.function([inp], self.output, name = 'f_'+prefix)

class ReshapeBuilder:
    def __init__(self, inp, prefix, shape=None):
        shape_map = dict()

        for i in range(inp.ndim):
            shape_map[i] = inp.shape[i]

        if shape == None: # flatten by default
            self.output = inp.reshape((1, inp.size))
        else:
            new_shape = []
            for r in shape:
                if isinstance(r, (list, tuple)):
                    rmap = list(map(lambda x:shape_map[x], r))
                    new_shape.append(reduce(mul, rmap, 1))
                else:
                    new_shape.append(shape_map[r])
            self.output  = inp.reshape(tuple(new_shape))

        self.f = T.function([inp], self.output, name = 'f_'+prefix)


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
    # Deprecated:
    # cause dtype error in theano
    # reasons unknown
    if (pred_prob.ndim == 2):
        return np.argmax(pred_prob, axis=1)
    return np.argmax(pred_prob)

"""
Training related.
"""
def sgd(params, grads, x, y, cost,
        lr=0.0001, gamma=0.9, beta=(0.9,0.999), epsilon=1e-8):
    """
    Stochastic Gradient Descent.

    :params: list of params (theano shared variables)
    can also be OrderedDict (key:value = shared variable name, shared value)
    :grads: list of gradients
    :x, y: data (input, label)
    :cost: expected loss @get_cost
    :gamma: weighting factor in running average
    :beta: weighting factors for Adam
    :epsilon: smoothing factor
    """
    # Init: new set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [T.shared(p.get_value() * np.asarray(0., dtype=config.floatX),
                        name='%s_grad' % str(k))
               for k, p in enumerate(params)]

    # Update grads
    gs_updt = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = T.function([x, y], cost, updates=gs_updt,
                                    name='sgd_f_grad_shared')

    # Update params
    para_updt = [(p, p - lr * g) for p, g in zip(params, gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = T.function([], [], updates=para_updt,
                               name='sgd_f_update')

    return f_grad_shared, f_update

def adadelta(params, grads, x, y, cost,
        lr=0.0001, gamma=0.9, beta=(0.9,0.999), epsilon=1e-8):
    """
    An adaptive learning rate optimizer.

    Parameters
    ----------
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
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    # Init
    zipped_grads = [T.shared(p.get_value() * np.asarray(0., dtype=config.floatX),
                                  name='%s_grad' % str(k))
                   for k, p in enumerate(params)]
    # running average of delta theta
    running_up2 = [T.shared(p.get_value() * np.asarray(0., dtype=config.floatX),
                                 name='%s_rup2' % str(k))
                   for k, p in enumerate(params)]
    # runing average of grads' squrares
    running_grads2 = [T.shared(p.get_value() *  np.asarray(0., dtype=config.floatX),
                                    name='%s_rgrad2' % str(k))
                   for k, p in enumerate(params)]

    # Update grads
    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, gamma * rg2 + (1-gamma) * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = T.function([x, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    # Update params
    updir = [-tensor.sqrt(ru2 + epsilon) / tensor.sqrt(rg2 + epsilon) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]

    param_up = [(p, p + ud) for p, ud in zip(params, updir)]

    # ???
    ru2up = [(ru2, gamma * ru2 + (1-gamma) * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]

    f_update = T.function([], [], updates=ru2up + param_up,
                          on_unused_input='ignore',
                          name='adadelta_f_update')

    return f_grad_shared, f_update

def rmsprop(params, grads, x, y, cost,
        lr=0.0001, gamma=0.9, beta=(0.9,0.999), epsilon=1e-8):
    """
    A variant of SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
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

    # Init
    zipped_grads = [T.shared(p.get_value() * np.asarray(0., dtype=config.floatX),
                                  name='%s_grad' % str(k))
                    for k, p in enumerate(params)]
    # runing average of grads
    running_grads = [T.shared(p.get_value() * np.asarray(0., dtype=config.floatX),
                                  name='%s_rgrad' % str(k))
                    for k, p in enumerate(params)]
    # runing average of grads' squrares
    running_grads2 = [T.shared(p.get_value() * np.asarray(0., dtype=config.floatX),
                                  name='%s_rgrad2' % str(k))
                    for k, p in enumerate(params)]

    # Update grads
    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]

    rgup = [(rg, gamma * rg + (1-gamma) * g) for rg, g in zip(running_grads, grads)]

    rg2up = [(rg2, gamma * rg2 + (1-gamma) * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = T.function([x, y], cost,
                               updates=zgup + rgup + rg2up,
                               name='rmsprop_f_grad_shared')

    # Update pramas
    updir = [T.shared(p.get_value() * np.asarray(0., dtype=config.floatX),
                      name='%s_updir' % k)
             for k, p in enumerate(params)]

    # ???
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]

    param_up = [(p, p + udn[1]) for p, udn in zip(params, updir_new)]

    f_update = T.function([], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update

# TODO:
# 0. Adam
# Adagrad
# 1. Momentum
# 2. NAG
