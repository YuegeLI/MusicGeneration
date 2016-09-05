from __future__ import print_function

import glob
import os
import sys
import datetime
import math

import numpy
try:
    import pylab
except ImportError:
    print ("pylab isn't available. If you use its functionality, it will crash.")
    print("It can be installed with 'pip install -q Pillow'")

from midi.utils import midiread, midiwrite
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

def load_dataset(N, name):
    re = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'data', name, 'train', '*.mid')
    files = glob.glob(re)
    dataset = [midiread(f, (21,109), 0.3).piano_roll.astype(theano.config.floatX) for f in files]

    all_gram = dict()
    for i in range(N):
        all_gram[i+1]=dict()

    for i in range(N):
        n = i+1
        print ('======'+str(n)+'======')
        n_gram = all_gram.get(n)
        for s, sequence in enumerate(dataset):
            print (s)
            count = dict()
            for l in range(len(sequence)-n+1):
                gram = sequence[l : l+n]
                if count.has_key(str(gram)):
                    count[str(gram)] += 1
                else:
                    count[str(gram)] = 1
            for k in count:
                if n_gram.has_key(k):
                    n_gram[k]=max(count.get(k), n_gram.get(k))
                else:
                    n_gram[k]=count.get(k)
    return all_gram


def load_generation(N, name):
    re_gen = os.path.join(os.path.split(os.path.dirname(__file__))[0], name, '*.mid')
    files_gen = glob.glob(re_gen)
    dataset_gen = [midiread(f, (21,109), 0.3).piano_roll.astype(theano.config.floatX) for f in files_gen]

    all_gram_gen = dict()
    for i in range(N):
        all_gram_gen[i+1]=dict()

    for i in range(N):
        n = i+1
        n_gram = all_gram_gen.get(n)
        for s, sequence in enumerate(dataset_gen):
            count = dict()
            for l in range(len(sequence)-n+1):
                gram = sequence[l : l+n]
                if count.has_key(str(gram)):
                    count[str(gram)] += 1
                else:
                    count[str(gram)] = 1
            for k in count:
                if n_gram.has_key(k):
                    n_gram[k]=max(count.get(k), n_gram.get(k))
                else:
                    n_gram[k]=count.get(k)
    return all_gram_gen

def generation_length():
    re_gen = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'generation', '*.mid')
    files_gen = glob.glob(re_gen)
    dataset_gen = [midiread(f, (21,109), 0.3).piano_roll.astype(theano.config.floatX) for f in files_gen]
    return len(dataset_gen[0])