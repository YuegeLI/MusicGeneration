# coding=utf-8
# Author: Yuege LI
# RNN for monophonic music

from __future__ import print_function

import glob
import os
import sys
import collections
import datetime
from collections import Counter

import numpy
import numpy as np
try:
    import pylab
except ImportError:
    print ("pylab isn't available. If you use its functionality, it will crash.")
    print("It can be installed with 'pip install -q Pillow'")

from midi.utils import midiread, midiwrite
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

numpy.random.seed(0xbeef)
rng = RandomStreams(seed=numpy.random.randint(1 << 30))
theano.config.warn.subtensor_merge_bug = False


def shared_normal(num_rows, num_cols, scale=1):
    '''Initialize a matrix shared variable with normally distributed
    elements.'''
    return theano.shared(numpy.random.normal(
        scale=scale, size=(num_rows, num_cols)).astype(theano.config.floatX))

def shared_zeros(*shape):
    '''Initialize a vector shared variable with zero elements.'''
    return theano.shared(numpy.zeros(shape, dtype=theano.config.floatX))


def build_rnn(n_visible, n_hidden_recurrent):

    W_xh = shared_normal(n_visible, n_hidden_recurrent, 0.0001) # input to hidden
    W_hh = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001) # hidden to hidden
    W_hy = shared_normal(n_hidden_recurrent, n_visible, 0.0001) # hidden to output
    b_h = shared_zeros(n_hidden_recurrent) # hidden bias
    b_y = shared_zeros(n_visible) # output bias

    params = W_xh, W_hh, W_hy, b_h, b_y

    x = T.matrix()  # a training sequence
    h0 = T.zeros((n_hidden_recurrent,))  # initial value for the RNN hidden
                                             # units
    target = T.matrix()

    act = T.tanh
    def recurrence(x_t, h_tm1):
        generate = x_t is None #v_t是None的话, generate是true
        if generate:
            x_t = rng.normal(size=(n_visible,))
        h_t = act(T.dot(x_t, W_xh) + T.dot(h_tm1, W_hh) + b_h)
        s_t = T.nnet.softmax(T.dot(h_t, W_hy) + b_y)[0]
        return h_t, s_t

    (h, sample), updates_train = theano.scan(lambda x_t, h_t, *_: recurrence(x_t, h_t), sequences=x,
                                             outputs_info=[h0, None], non_sequences=params)

    monitor = T.xlogx.xlogy0(target, sample) + T.xlogx.xlogy0(1 - target, 1 - sample)
    monitor = monitor.sum() / target.shape[0]
    cost = - T.xlogx.xlogy0(target, sample) - T.xlogx.xlogy0(1 - target, 1 - sample)
    cost = cost.sum() / target.shape[0]

    (h_g, generation), updates_generate = theano.scan(lambda h_tm1, *_: recurrence(None, h_tm1),
                                                      outputs_info=[h0, None], non_sequences=params, n_steps=200)

    return x, target, cost, monitor, sample, params, updates_train, generation, updates_generate


class Rnn:
    def __init__(self, n_hidden_recurrent = 40, lr = 0.001, r = (21, 109), dt = 0.3):

        self.r = r
        self.dt = dt
        self.lr = lr

        (x, target, cost, monitor, sample, params, updates_train, generation, updates_generate) = build_rnn(r[1] - r[0], n_hidden_recurrent)

        gradients = T.grad(cost, params)
        updates_train.update(((p, p - lr * g) for p, g in zip(params, gradients)))
        self.train_function = theano.function(inputs = [x, target], outputs = [cost, monitor, sample], updates = updates_train)

        self.generate_function = theano.function(inputs=[], outputs=[generation], updates = updates_generate)


    def train(self, files, batch_size=100, num_epochs=200):

        assert len(files) > 0, 'Training set is empty!' \
                               ' (did you download the data files?)'
        dataset = [midiread(f, self.r, self.dt).piano_roll.astype(theano.config.floatX) for f in files]

        def accuracy (v, v_sample):
            accs = []
            t, n = v.shape
            for time in range(t):
                tp = 0 # true positive
                fp = 0 # false positive
                fn = 0 # false negative
                for note in range(n):
                    if v[time][note] == 1 and v_sample[time][note] == 1:
                        tp += 1.
                    if v[time][note] == 0 and v_sample[time][note] == 1:
                        fp += 1.
                    if v[time][note] == 1 and v_sample[time][note] == 0:
                        fn += 1.
                if tp + fp + fn != 0:
                    a = tp / (tp + fp + fn)
                else:
                    a = 0
                accs.append(a)

            acc = numpy.mean(accs)
            return acc

        def sampling (sample): # 01
            ixes = []
            for i, s in enumerate(sample):
                n_step = []
                for n in xrange(20):
                    ix = np.random.choice(range(88), p=s) # 根据prob(p)随机选一个
                    n_step.append(ix)
                count = Counter(n_step)
                ix = count.most_common(1)[0][0]
                x = np.zeros((88,))
                x[ix] = 1
                ixes.append(x)
            return ixes

        try:
            print ('rnn_mono, dataset=nottingham_mono, lr=%f, epoch=%i' %(self.lr, num_epochs))

            for epoch in range(num_epochs):
                numpy.random.shuffle(dataset)
                costs = []
                monitors = []
                accs = []

                for s, sequence in enumerate(dataset):
                    for i in range(0, len(sequence), batch_size):
                        if i+batch_size+1 >= len(sequence):
                            break

                        v = sequence[i:i + batch_size]
                        targets = sequence[i + 1 : i + batch_size + 1]

                        (cost, monitor, sample) = self.train_function(v, targets)
                        costs.append(cost)
                        monitors.append(monitor)

                        sample = sampling(sample)
                        acc = accuracy(v, sample)
                        accs.append(acc)

                p = 'Epoch %i/%i  LL %f   ACC %f  Cost %f       time %s' \
                    % (epoch + 1, num_epochs, numpy.mean(monitors), numpy.mean(accs), numpy.mean(costs),
                       datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                print (p)

                if (epoch%10 == 0):
                    piano_roll = sampling(self.generate_function()[0])
                    midiwrite('sample/rnn_mono_%i.mid' %(epoch), piano_roll, self.r, self.dt)
                sys.stdout.flush()

        except KeyboardInterrupt:
            print('Interrupted by user.')


def test_rnn(batch_size=256, num_epochs=500):
    model = Rnn()
    re = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'data', 'Nottingham', 'mono', '*.mid')
    model.train(glob.glob(re), batch_size=batch_size, num_epochs=num_epochs)
    return model

if __name__ == '__main__':
    model = test_rnn()