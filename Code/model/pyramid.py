# coding=utf-8
# Author: Yuege LI
# Pyramid Model for polyphonic music

from __future__ import print_function

import glob
import os
import sys
import collections
import datetime

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
    '''Initialize a matrix shared variable with normally distributedelements.'''
    return theano.shared(numpy.random.normal(
        scale=scale, size=(num_rows, num_cols)).astype(theano.config.floatX))

def shared_zeros(*shape):
    '''Initialize a vector shared variable with zero elements.'''
    return theano.shared(numpy.zeros(shape, dtype=theano.config.floatX))


def build_rnn(n_in, n_hidden, n_y):

    target = T.matrix() # original melody, target of the RNN
    upsample = T.matrix() # input melody to the RNN, upsample=Upsampling(Downsampling(target))

    W_vh = shared_normal(n_in, n_hidden, 0.0001)
    W_hh = shared_normal(n_hidden, n_hidden, 0.0001)
    b_h = shared_zeros(n_hidden)
    W_hy = shared_normal(n_hidden, n_y, 0.0001)
    b_y = shared_zeros(n_y)

    params = W_vh, W_hh, b_h, W_hy, b_y

    h0 = T.zeros((n_hidden,))

    act = T.tanh
    def recurrence(v_t, h_tm1):
        generate = v_t is None
        if generate:
            v_t = rng.normal(size=(n_in,))
        h_t = act(T.dot(v_t, W_vh) + b_h + T.dot(h_tm1, W_hh))
        y_t = T.nnet.softmax(theano.dot(h_t, W_hy) + b_y)[0]
        return h_t, y_t

    (_, sharp), updates_train = theano.scan(
            lambda v_t, h_tm1, *_: recurrence(v_t, h_tm1), sequences=upsample,
            outputs_info=[h0, None], non_sequences=params)

    monitor = T.xlogx.xlogy0(target, sharp) + T.xlogx.xlogy0(1 - target, 1 - sharp)
    monitor = monitor.sum() / target.shape[0]
    cost = - T.xlogx.xlogy0(target, sharp) - T.xlogx.xlogy0(1 - target, 1 - sharp)
    cost = cost.sum() / target.shape[0]

    # used for the generation in last layer(layer 4), generating a melody with a random seed
    (_, generation), updates_generate = theano.scan(
            lambda h_tm1, *_: recurrence(None, h_tm1),
            outputs_info=[h0, None], non_sequences=params, n_steps=8)

    return target, upsample, params, updates_train, cost, monitor, sharp, generation, updates_generate


class Pyramid:

    def __init__(self, lr=0.01, r=(21, 109), dt=0.3):
        '''
        r: mapping to the key on the piano
        dt: sampling period when converting the MIDI files into piano-rolls, or equivalently
            the time difference between consecutive time steps.
        lr: learninig rate
        '''

        self.r = r
        self.dt = dt
        self.lr = lr

        # layer 0 => G_0
        (target_0, upsample_0, params_0, updates_train_0, cost_0, monitor_0, sharp_0, _, _) = build_rnn(r[1]-r[0], 70, r[1]-r[0])
        # update params
        gradients_0 = T.grad(cost_0, params_0)
        updates_train_0.update(((p, p - lr * g) for p, g in zip(params_0, gradients_0)))
        # training function
        self.sharp_fun_0 = theano.function(inputs=[target_0, upsample_0], outputs=[sharp_0, monitor_0], updates=updates_train_0)
        # generate function
        self.generate_0 = theano.function(inputs=[target_0, upsample_0], outputs=[sharp_0], updates=updates_train_0)
        # 按说generate function这里的input不应该有target... 但这里还没改出来,但对于训练整体应该没有影响

        # layer 1 => G_1
        (target_1, upsample_1, params_1, updates_train_1, cost_1, monitor_1, sharp_1, _, _) = build_rnn(r[1]-r[0], 30, r[1]-r[0])
        gradients_1 = T.grad(cost_1, params_1)
        updates_train_1.update(((p, p - lr * g) for p, g in zip(params_1, gradients_1)))
        self.sharp_fun_1 = theano.function(inputs=[target_1, upsample_1], outputs=[sharp_1, monitor_1], updates=updates_train_1)
        self.generate_1 = theano.function(inputs=[target_1, upsample_1], outputs=[sharp_1], updates=updates_train_1)

        # layer 2 => G_2
        (target_2, upsample_2, params_2, updates_train_2, cost_2, monitor_2, sharp_2, _, _) = build_rnn(r[1]-r[0], 10, r[1]-r[0])
        gradients_2 = T.grad(cost_2, params_2)
        updates_train_2.update(((p, p - lr * g) for p, g in zip(params_2, gradients_2)))
        self.sharp_fun_2 = theano.function(inputs=[target_2, upsample_2], outputs=[sharp_2, monitor_2], updates=updates_train_2)
        self.generate_2 = theano.function(inputs=[target_2, upsample_2], outputs=[sharp_2], updates=updates_train_2)

        # layer 3 => G_3
        (target_3, upsample_3, params_3, updates_train_3, cost_3, monitor_3, sharp_3, _, _) = build_rnn(r[1]-r[0], 5, r[1]-r[0])
        gradients_3 = T.grad(cost_3, params_3)
        updates_train_3.update(((p, p - lr * g) for p, g in zip(params_3, gradients_3)))
        self.sharp_fun_3 = theano.function(inputs=[target_3, upsample_3], outputs=[sharp_3, monitor_3], updates=updates_train_3)
        self.generate_3 = theano.function(inputs=[target_3, upsample_3], outputs=[sharp_3], updates=updates_train_3)

        # layer 4 => G_4
        (target_4, upsample_4, params_4, updates_train_4, cost_4, monitor_4, sharp_4, generation_4, updates_generate_4) = build_rnn(r[1]-r[0], 5, r[1]-r[0])
        gradients_4 = T.grad(cost_4, params_4)
        updates_train_4.update(((p, p - lr * g) for p, g in zip(params_4, gradients_4)))
        self.sharp_fun_4 = theano.function(inputs=[target_4, upsample_4], outputs=[sharp_4, monitor_4], updates=updates_train_4)
        self.generate_4 = theano.function(inputs=[], outputs=[generation_4], updates=updates_generate_4)


    def train(self, files, batch_size=128, num_epochs=200):

        def downsampling (sample):
            # only keep the note on even index position (0, 2, 4 ...)
            downsample = []
            for i, s in enumerate(sample):
                if i % 2 == 0:
                    downsample.append(s)
            return downsample

        def upsampling (sample, length):
            '''
            double each notes in sample
                sample: the melody to be upsampled
                length: the length of the original melody M, sample=Downsampling(M)
            '''
            upsample = []
            for s in sample:
                upsample.append(s)
                if (len(upsample)>=length):
                    # upsampling melody length cannot longer than original melody length
                    break
                upsample.append(s)
            return upsample

        def sampling (sample):
            # to be one hot (to be 0,1)
            s = T.matrix()
            b = rng.binomial(size=s.shape, n=1, p=s, dtype=theano.config.floatX)
            fun = theano.function(inputs=[s], outputs=[b])

            return fun(sample)[0]

        def accuracy (v, v_sample):
            # ACC
            accs = []
            t = len(v)
            n = len(v[0])
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


        def generate ():
            # sampling procedure, generating music

            # layer 4
            generate_sharp_4 = self.generate_4()[0]
            generate_sharp_4 = sampling(generate_sharp_4)

            # layer 3
            upsample_3 = upsampling(generate_sharp_4, 2*len(generate_sharp_4)+1)
            generate_sharp_3 = self.generate_3(upsample_3, upsample_3)[0]
            generate_sharp_3 = sampling(generate_sharp_3)

            # layer 2
            upsample_2 = upsampling(generate_sharp_3, 2*len(generate_sharp_3)+1)
            generate_sharp_2 = self.generate_2(upsample_2, upsample_2)[0]
            generate_sharp_2 = sampling(generate_sharp_2)

            # layer 1
            upsample_1 = upsampling(generate_sharp_2, 2*len(generate_sharp_2)+1)
            generate_sharp_1 = self.generate_1(upsample_1, upsample_1)[0]
            generate_sharp_1 = sampling(generate_sharp_1)

            # layer 0
            upsample_0 = upsampling(generate_sharp_1, 2*len(generate_sharp_1)+1)
            generate_sharp_0 = self.generate_0(upsample_0, upsample_0)[0]
            generate_sharp_0 = sampling(generate_sharp_0)

            return generate_sharp_0


        # load data
        dataset = [midiread(f, self.r, self.dt).piano_roll.astype(theano.config.floatX) for f in files]
        print ('pyramid model, dataset=Nottingham, lr=%f, epoch=%i' %(self.lr, num_epochs))

        for epoch in range(num_epochs):
            numpy.random.shuffle(dataset)
            monitors_0 = []
            accs_0 = []
            monitors_1 = []
            accs_1 = []
            monitors_2 = []
            accs_2 = []
            monitors_3 = []
            accs_3 = []
            monitors_4 = []
            accs_4 = []

            for s, sequence in enumerate(dataset):
                for i in range(0, len(sequence), batch_size):

                    batch_music = sequence[i:i + batch_size] # 128

                    # layer 0
                    downsample_0 = downsampling(batch_music) # 64
                    upsample_0 = upsampling(downsample_0, len(batch_music)) # 128

                    # layer 1
                    downsample_1 = downsampling(downsample_0) # 32
                    upsample_1 = upsampling(downsample_1, len(downsample_0)) # 64

                    # layer 2
                    downsample_2 = downsampling(downsample_1) # 16
                    upsample_2 = upsampling(downsample_2, len(downsample_1)) # 32

                    # layer 3
                    downsample_3 = downsampling(downsample_2) # 8
                    upsample_3 = upsampling(downsample_3, len(downsample_2)) # 16

                    # layer 0
                    sharp_0, monitor_0 = self.sharp_fun_0(batch_music, upsample_0)
                    monitors_0.append(monitor_0)
                    accs_0.append(accuracy(batch_music, sampling(sharp_0)))

                    # layer 1
                    sharp_1, monitor_1 = self.sharp_fun_1(downsample_0, upsample_1)
                    monitors_1.append(monitor_1)
                    accs_1.append(accuracy(downsample_0, sampling(sharp_1)))

                    # layer 2
                    sharp_2, monitor_2 = self.sharp_fun_2(downsample_1, upsample_2)
                    monitors_2.append(monitor_2)
                    accs_2.append(accuracy(downsample_1, sampling(sharp_2)))

                    # layer 3
                    sharp_3, monitor_3 = self.sharp_fun_3(downsample_2, upsample_3)
                    monitors_3.append(monitor_3)
                    accs_3.append(accuracy(downsample_2, sampling(sharp_3)))

                    # layer 4
                    if (len(downsample_3) == 1):
                        sharp_4, monitor_4 = self.sharp_fun_4(downsample_3, downsample_3)
                        accs_4.append(accuracy(downsample_3, sampling(sharp_4)))
                    else:
                        sharp_4, monitor_4 = self.sharp_fun_4(downsample_3[1:], downsample_3[:len(downsample_3)-1])
                        accs_4.append(accuracy(downsample_3[1:], sampling(sharp_3)))
                    monitors_4.append(monitor_4)


            p = 'Epoch %i/%i    layer0:LL %f ACC %f   layer1:LL %f ACC %f  layer2:LL %f ACC %f  layer3:LL %f ACC %f  ' \
                'layer4:LL %f ACC %f  time %s' % \
                (epoch + 1, num_epochs, numpy.mean(monitors_0), numpy.mean(accs_0), numpy.mean(monitors_1), numpy.mean(accs_1),
                 numpy.mean(monitors_2), numpy.mean(accs_2), numpy.mean(monitors_3), numpy.mean(accs_3), numpy.mean(monitors_4),
                 numpy.mean(accs_4),   datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print (p)

            if (epoch % 100 == 0 or epoch == num_epochs-1):

                piano_roll = generate()
                midiwrite('sample/pyramid_%i.mid' %(epoch), piano_roll, self.r, self.dt)


def test_pyramid(batch_size=128, num_epochs=500):
    model = Pyramid()
    re = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'data', 'Nottingham', 'train', '*.mid')
    model.train(glob.glob(re), batch_size=batch_size, num_epochs=num_epochs)
    return model

if __name__ == '__main__':
    model = test_pyramid()


