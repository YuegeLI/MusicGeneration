# coding=utf-8
# LSTM-RBM for music generation

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


def build_lstmrbm(n_visible, n_hidden, n_hidden_recurrent):

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
    Wuo = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    Wqo = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    Wco = shared_normal(n_hidden_recurrent, n_hidden_recurrent, 0.0001)
    Wqv = shared_normal(n_hidden_recurrent, n_visible, 0.0001)
    Wqh = shared_normal(n_hidden_recurrent, n_hidden, 0.0001)
    bo = shared_zeros(n_hidden_recurrent) 


    params = W, bv, bh, Wuh, Wuv, Wvu, Wuu, bu, Wui, Wqi, Wci, bi,
    Wuf, Wqf, Wcf, bf, Wuc, Wqc, bc, Wuo, Wqo, Wco, bo , Wqv, Wqh
    # learned parameters as shared
    # variables

    v = T.matrix()  # a training sequence
    u0 = T.zeros((n_hidden_recurrent,))  # initial value for the RNN hidden
                                         # units
    q0 = T.zeros((n_hidden_recurrent,))
    c0 = T.zeros((n_hidden_recurrent,))

    sigma = lambda x: 1 / (1 + T.exp(-x))
    def recurrence(v_t, u_tm1, q_tm1, c_tm1):
        bv_t = bv + T.dot(u_tm1, Wuv) + T.dot( q_tm1, Wqv)
        bh_t = bh + T.dot(u_tm1, Wuh) + T.dot( q_tm1, Wqh)
        generate = v_t is None
        if generate:
            v_t, _, _, updates = build_rbm(rng.normal(size=(n_visible,)), W, bv_t,
                                           bh_t, k=25)
        u_t = T.tanh(bu + T.dot(v_t, Wvu) + T.dot(u_tm1, Wuu))
        
        i_t = sigma(bi + T.dot(c_tm1, Wci) + T.dot(q_tm1, Wqi) + T.dot(u_t, Wui))
        f_t = sigma(bf + T.dot(c_tm1, Wcf) + T.dot(q_tm1, Wqf) + T.dot(u_t, Wuf))
        c_t = (f_t * c_tm1) + ( i_t * sigma( T.dot(u_t, Wuc) + T.dot( q_tm1, Wqc) + bc ))
        o_t = sigma(bo + T.dot(c_t, Wco) + T.dot(q_tm1, Wqo) + T.dot(u_t, Wuo))
        q_t = o_t * T.tanh( c_t )
        
        return ([v_t, u_t, q_t, c_t], updates) if generate else [u_t, q_t, c_t, bv_t, bh_t]
    
    (u_t, q_t, c_t, bv_t, bh_t), updates_train = theano.scan(
        lambda v_t, u_tm1, q_tm1, c_tm1, *_: recurrence(v_t, u_tm1, q_tm1, c_tm1),
        sequences=v, outputs_info=[u0, q0, c0, None, None], non_sequences=params)
    v_sample, cost, monitor, updates_rbm = build_rbm(v, W, bv_t[:], bh_t[:],
                                                     k=15)
    updates_train.update(updates_rbm)

    # symbolic loop for sequence generation
    (v_t, u_t, q_t, c_t), updates_generate = theano.scan(
        lambda u_tm1, q_tm1, c_tm1, *_: recurrence(None, u_tm1, q_tm1, c_tm1),
        outputs_info=[None, u0, q0, c0], non_sequences=params, n_steps=200)

    return (v, v_sample, cost, monitor, params, updates_train, v_t,
            updates_generate)


class LstmRbm:

    def __init__(self, n_hidden=150, n_hidden_recurrent=100, lr=0.0001,
                 r=(21, 109), dt=0.3):

        self.lr = lr
        self.r = r
        self.dt = dt
        (v, v_sample, cost, monitor, params, updates_train, v_t,
         updates_generate) = build_lstmrbm(r[1] - r[0], n_hidden,
                                           n_hidden_recurrent)

        gradient = T.grad(cost, params, consider_constant=[v_sample])
        updates_train.update(((p, p - lr * g) for p, g in zip(params,
                                                                gradient)))
        self.train_function = theano.function([v], (monitor, v_sample),
                                               updates=updates_train)
        self.generate_function = theano.function([], v_t,
                                                 updates=updates_generate)

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

        try:
            print ('lstm_rbm, dataset=Nottingham, lr=%f, epoch=%i' %(self.lr, num_epochs))

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
                if (epoch%100 == 0 or epoch==num_epochs-1):
                    piano_roll = self.generate_function()
                    midiwrite('sample/lstm_rbm%i.mid' %(epoch), piano_roll, self.r, self.dt)
                sys.stdout.flush()

        except KeyboardInterrupt:
            print 'Interrupted by user.'


def test_lstmrbm(batch_size=100, num_epochs=500):
    model = LstmRbm()
    re = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'data', 'Nottingham', 'train', '*.mid')
    model.train(glob.glob(re), batch_size=batch_size, num_epochs=num_epochs)
    return model

if __name__ == '__main__':
    model = test_lstmrbm()
