# coding=utf-8
# Author: Yuege LI
# AttLSTM-RBM for polyphonic music generation

import glob
import os
import sys
import datetime

import numpy
try:
    import pylab
except ImportError:
    print "pylab isn't available, if you use their fonctionality, it will crash"
    print "It can be installed with 'pip install -q Pillow'"

from midi.utils import midiread, midiwrite
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

numpy.random.seed(0xbeef)
rng = RandomStreams(seed=numpy.random.randint(1 << 30))
theano.config.warn.subtensor_merge_bug = False


def fast_dropout(rng, x):
    """ Multiply activations by N(1,1) """
    mask = rng.normal(size=x.shape, avg=1., dtype=theano.config.floatX)
    return x * mask

def build_rbm(v, W, bv, bh, k):
    def gibbs_step(v):
        mean_h = T.nnet.sigmoid(T.dot( fast_dropout( rng, v) , W) + bh)
        h = rng.binomial(size=mean_h.shape, n=1, p=mean_h,
                         dtype=theano.config.floatX)
        mean_v = T.nnet.sigmoid(T.dot( fast_dropout(rng, h) , W.T) + bv)
        v = rng.binomial(size=mean_v.shape, n=1, p=mean_v,
                         dtype=theano.config.floatX)
        return mean_v, v

    chain, updates = theano.scan(lambda v: gibbs_step(v)[1], outputs_info=[v],
                                 n_steps=k)
    v_sample = chain[-1]

    mean_v = gibbs_step(v_sample)[0]
    monitor = T.xlogx.xlogy0(v, mean_v) + T.xlogx.xlogy0(1 - v, 1 - mean_v)
    monitor = monitor.sum() / v.shape[0]

    def free_energy(v):
        return -(v * bv).sum() - T.log(1 + T.exp(T.dot(v, W) + bh)).sum()
    cost = (free_energy(v) - free_energy(v_sample)) / v.shape[0]

    return v_sample, cost, monitor, updates


def shared_normal(num_rows, num_cols, scale=1):
    '''Initialize a matrix shared variable with normally distributed
elements.'''
    return theano.shared(numpy.random.normal(
        scale=scale, size=(num_rows, num_cols)).astype(theano.config.floatX))


def shared_zeros(*shape):
    '''Initialize a vector shared variable with zero elements.'''
    return theano.shared(numpy.zeros(shape, dtype=theano.config.floatX))


def build_attlstmrbm(n_visible, n_hidden, n_hidden_recurrent, length=10):

    W = shared_normal(n_visible, n_hidden, 0.01)
    bv = shared_zeros(n_visible)
    bh = shared_zeros(n_hidden)
    Wuh = shared_normal(n_hidden_recurrent, n_hidden, 0.0001)
    Wuv = shared_normal(n_hidden_recurrent, n_visible, 0.0001)
    Wvu = shared_normal(n_visible, n_hidden_recurrent, 0.0001)
    Wuu = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    bu = shared_zeros(n_hidden_recurrent)

    Wui = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    Wqi = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    Wci = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    bi = shared_zeros(n_hidden_recurrent)
    Wuf = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    Wqf = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    Wcf = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    bf = shared_zeros(n_hidden_recurrent)
    Wuc = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    Wqc = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    bc = shared_zeros(n_hidden_recurrent)
    Wqv = shared_normal(n_hidden_recurrent, n_visible, 0.0001)
    Wqh = shared_normal(n_hidden_recurrent, n_hidden, 0.0001)

    W0_1 = shared_normal(n_hidden_recurrent, length, 0.0001)
    W1_1 = shared_normal(n_hidden_recurrent, length, 0.0001)
    W2_1 = shared_normal(n_hidden_recurrent, length, 0.0001)
    W3_1 = shared_normal(n_hidden_recurrent, length, 0.0001)
    W4_1 = shared_normal(n_hidden_recurrent, length, 0.0001)
    W5_1 = shared_normal(n_hidden_recurrent, length, 0.0001)
    W6_1 = shared_normal(n_hidden_recurrent, length, 0.0001)
    W7_1 = shared_normal(n_hidden_recurrent, length, 0.0001)
    W8_1 = shared_normal(n_hidden_recurrent, length, 0.0001)
    W9_1 = shared_normal(n_hidden_recurrent, length, 0.0001)
    W_2 = shared_normal(n_hidden_recurrent, length, 0.0001)
    v_l = shared_normal(length, length, 0.0001) #matrix

    params = W, bv, bh, Wuh, Wuv, Wvu, Wuu, bu, Wui, Wqi, Wci, bi,
    Wuf, Wqf, Wcf, bf, Wuc, Wqc, bc,
    Wqv, Wqh,
    W0_1, W1_1, W2_1, W3_1, W4_1, W5_1, W6_1, W7_1, W8_1, W9_1, W_2, v_l
    # learned parameters as shared
    # variables

    v = T.matrix()  # a training sequence
    u0 = T.zeros((n_hidden_recurrent,))  # initial value for the RNN hidden
                                         # units
    h0_0 = T.zeros((n_hidden_recurrent,))
    h0_1 = T.zeros((n_hidden_recurrent,))
    h0_2 = T.zeros((n_hidden_recurrent,))
    h0_3 = T.zeros((n_hidden_recurrent,))
    h0_4 = T.zeros((n_hidden_recurrent,))
    h0_5 = T.zeros((n_hidden_recurrent,))
    h0_6 = T.zeros((n_hidden_recurrent,))
    h0_7 = T.zeros((n_hidden_recurrent,))
    h0_8 = T.zeros((n_hidden_recurrent,))
    h0_9 = T.zeros((n_hidden_recurrent,))
    c0 = T.zeros((n_hidden_recurrent,))

    sigma = lambda x: 1 / (1 + T.exp(-x))
    def recurrence(v_t, u_tm1, h_0, h_1, h_2, h_3, h_4, h_5, h_6, h_7, h_8, h_9, c_tm1):
        bv_t = bv + T.dot(u_tm1, Wuv) + T.dot( h_9, Wqv)
        bh_t = bh + T.dot(u_tm1, Wuh) + T.dot( h_9, Wqh)
        generate = v_t is None
        if generate:
            v_t, _, _, updates = build_rbm(T.zeros((n_visible,)), W, bv_t,
                                           bh_t, k=25)
        u_t = T.tanh(bu + T.dot(v_t, Wvu) + T.dot(u_tm1, Wuu))

        i_t = sigma(bi + T.dot(c_tm1, Wci) + T.dot(h_9, Wqi) + T.dot(u_t, Wui))
        f_t = sigma(bf + T.dot(c_tm1, Wcf) + T.dot(h_9, Wqf) + T.dot(u_t, Wuf))
        c_t = (f_t * c_tm1) + ( i_t * sigma( T.dot(h_9, Wuc) + T.dot( h_9, Wqc) + bc ))

        ua_t = T.dot(T.tanh(T.dot(h_0, W0_1) + T.dot(h_1, W1_1) + T.dot(h_2, W2_1) + T.dot(h_3, W3_1) + T.dot(h_4, W4_1) +
                            T.dot(h_5, W5_1) + T.dot(h_6, W6_1) + T.dot(h_7, W7_1) + T.dot(h_8, W8_1) + T.dot(h_9, W9_1) +
                            T.dot(c_t, W_2)), v_l)
        a = T.nnet.softmax(ua_t)
        at = a[0].transpose()
        d = T.dot(h_0, at[0]) + T.dot(h_1, at[1]) + T.dot(h_2, at[2]) + T.dot(h_3, at[3]) + T.dot(h_4, at[4]) + \
            T.dot(h_5, at[5]) + T.dot(h_6, at[6]) + T.dot(h_7, at[7]) + T.dot(h_8, at[8]) + T.dot(h_9, at[9])

        return ([v_t, u_t, h_1, h_2, h_3, h_4, h_5, h_6, h_7, h_8, h_9, d, c_t], updates) if generate \
            else [u_t, h_1, h_2, h_3, h_4, h_5, h_6, h_7, h_8, h_9, d, c_t, bv_t, bh_t]

    (u_t, h_1, h_2, h_3, h_4, h_5, h_6, h_7, h_8, h_9, d, c_t, bv_t, bh_t), updates_train = theano.scan(
        lambda v_t, u_tm1, h_tm0, h_tm1, h_tm2, h_tm3, h_tm4, h_tm5, h_tm6, h_tm7, h_tm8, h_tm9, c_tm1, *_:
        recurrence(v_t, u_tm1, h_tm0, h_tm1, h_tm2, h_tm3, h_tm4, h_tm5, h_tm6, h_tm7, h_tm8, h_tm9, c_tm1),
        sequences=v, outputs_info=[u0, h0_0, h0_1, h0_2, h0_3, h0_4, h0_5, h0_6, h0_7, h0_8, h0_9, c0, None, None], non_sequences=params)
    v_sample, cost, monitor, updates_rbm = build_rbm(v, W, bv_t[:], bh_t[:],
                                                     k=15)
    updates_train.update(updates_rbm)

    (v_t, u_t, h_1, h_2, h_3, h_4, h_5, h_6, h_7, h_8, h_9, d, c_t), updates_generate = theano.scan(
        lambda u_tm1, h_tm0, h_tm1, h_tm2, h_tm3, h_tm4, h_tm5, h_tm6, h_tm7, h_tm8, h_tm9, c_tm1, *_:
        recurrence(None, u_tm1, h_tm0, h_tm1, h_tm2, h_tm3, h_tm4, h_tm5, h_tm6, h_tm7, h_tm8, h_tm9, c_tm1),
        outputs_info=[None, u0, h0_0, h0_1, h0_2, h0_3, h0_4, h0_5, h0_6, h0_7, h0_8, h0_9, c0], non_sequences=params, n_steps=200)

    return (v, v_sample, cost, monitor, params, updates_train, v_t,
            updates_generate)


class AttLstmRbm:

    def __init__(self, n_hidden=150, n_hidden_recurrent=100, length=10, lr=0.0001,
                 r=(21, 109), dt=0.3):

        self.r = r
        self.dt = dt
        (v, v_sample, cost, monitor, params, updates_train, v_t,
         updates_generate) = build_attlstmrbm(r[1] - r[0], n_hidden,
                                           n_hidden_recurrent, length)

        gradient = T.grad(cost, params, consider_constant=[v_sample])
        updates_train.update(((p, p - lr * g) for p, g in zip(params,
                                                                gradient)))
        self.train_function = theano.function([v], (monitor, v_sample), updates=updates_train)
        self.generate_function = theano.function([], v_t, updates=updates_generate)

    def train(self, files, batch_size=100, num_epochs=200):

        assert len(files) > 0, 'Training set is empty!' \
                               ' (did you download the data files?)'
        dataset = [midiread(f, self.r,
                            self.dt).piano_roll.astype(theano.config.floatX)
                   for f in files]

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

        def generate(self, filename, show=True):
            piano_roll = self.generate_function()
            midiwrite(filename, piano_roll, self.r, self.dt)

        try:
            print ('attlstm_rbm, dataset=nottingham, lr=0.0001, epoch=500 \n')

            for epoch in xrange(num_epochs):
                numpy.random.shuffle(dataset)
                costs = []
                accs = []

                for s, sequence in enumerate(dataset):
                    for i in xrange(0, len(sequence), batch_size):
                        v = sequence[i:i + batch_size]

                        (cost, v_sample) = self.train_function(v)
                        costs.append(cost)

                        acc = accuracy(v, v_sample)
                        accs.append(acc)

                p = 'Epoch %i/%i    LL %f   ACC %f  time %s' % (epoch + 1, num_epochs, numpy.mean(costs), numpy.mean(accs), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                print (p)

                if (epoch % 50 == 0):
                    generate(self, 'sample/attlstm_rbm_%i.mid' % (epoch))

                sys.stdout.flush()

        except KeyboardInterrupt:
            print 'Interrupted by user.'


def test_attlstmrbm(batch_size=256, num_epochs=500):
    model = AttLstmRbm()
    re = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'data', 'Nottingham', 'train', '*.mid')
    model.train(glob.glob(re), batch_size=batch_size, num_epochs=num_epochs)
    return model

if __name__ == '__main__':
    model = test_attlstmrbm()
