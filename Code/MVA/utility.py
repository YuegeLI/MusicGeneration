# Author: Ying WEN

from midi.utils import midiread, midiwrite
import os
import glob
import numpy as np


R = (21, 109)
DT = 0.3


def get_file_path(style='Nottingham', data_type='mono'):
    """
        Read midi files from data directory
    """
    re = os.path.join(os.path.dirname(__file__),
                      'data', style, data_type, '*.mid')
    file_paths = glob.glob(re)
    return file_paths


def key2frequency(sample):
    """
        convert key to frequency, f = 2^((n - 49) / 12) * 440Hz
        note, the index begin from 0, therefore our code becomes 49 -> 48
    """
    return np.around(np.power(2, (sample - 48) / 12.) * 440., decimals=2)


def frequency2key(sample):
    """
        convert frequency to key, n = 12 * log2(f / 440Hz) + 49
        note, the index begin from 0, therefore our code becomes 49 -> 48
    """
    base = np.log2(sample / 440.)
    return np.around(12 * base + 48).astype(int)


def get_index(key, data_type='mono', max_key_number=4):
    """
        extract index number from 88 * 1 list
        work for monoxxx
    """
    if data_type == 'mono':
        key = key.tolist()
        if 1. in key:
            return np.array([key.index(1.)])
        else:
            return np.array([0.])
    else:
        index = np.array((key > 0).nonzero()[0][0:max_key_number])
        # print(index)
        if index.shape[0] < max_key_number:
            gap = (max_key_number - index.shape[0], 0)
            index = np.pad(index, pad_width=gap,
                           mode='constant', constant_values=-200.)
        return index

# print get_index(np.array([0,0,0,0,1,0,0,0,1,0]),data_type='mono')


def padding(d, max_length):
    d = d[0:max_length]
    if len(d) < max_length:
        d = d.tolist() + ([[0] * d[0].shape[0]] * (max_length - len(d)))
    return d


def load_dataset(style='Nottingham', data_type='train', max_length=256, max_key_number=1):
    """
        Load dataset from files
        for mono case: max_key_number would be 1
        for poly case: max_key_number default is 4
    """
    file_paths = get_file_path(style, data_type)
    dataset = [midiread(f, R, DT).piano_roll.astype(np.float32)
               for f in file_paths]
    dataset = [np.array([get_index(dataset[i][j], data_type, max_key_number) for j in xrange(
        len(dataset[i]))]) for i in xrange(len(dataset))]
    # print('min length:', sum([len(d) for d in dataset])/len(dataset))
    dataset = np.array(
        [np.array(padding(key2frequency(d), max_length)) for d in dataset])
    return dataset


def non_zero_avg(a):
    if a.sum(1) > 0:
        return a.sum(1) / (a != 0).sum(1)
    else:
        return 0.


# rint non_zero_avg(np.array([0,0,2,4]))


def moving_avg_low_pass_filter(sample, padding='SAME'):
    """
        Low pass moving average filter, y = 0.25*(x[t]+2*x[t-1]+x[t-2])
    """
    sample_filtered = np.zeros(sample.shape)

    for i in xrange(len(sample) - 1, 1, -1):
        sample_filtered[i] = 0.25 * \
            (sample[i] + 2 * non_zero_avg(sample[i - 1]) +
             non_zero_avg(sample[i - 2]))
    if padding == 'SAME':
        sample_filtered[1] = 0.25 * (sample[1] + 3 * non_zero_avg(sample[0]))
        sample_filtered[0] = sample[0]
    elif padding == 'ZERO':
        sample_filtered[1] = 0.25 * (sample[1] + 2 * non_zero_avg(sample[0]))
        sample_filtered[0] = 0.25 * sample[0]
    return sample_filtered


def write_to_midi_file(sample, file_path):
    """
        write midi list to midi format file
    """
    sample = frequency2key(sample)
    sample = convert_key_to_midi_list(sample)
    midiwrite(file_path, sample, R, DT)


def convert_key_to_midi_list(sample):
    """
        convert key list to midi list
    """
    midi_list = np.zeros([len(sample), 88])
    for i in xrange(len(sample)):
        midi_list[i][np.clip(sample[i], 0, 87)] = np.ones(sample[i].shape[0])
    return midi_list


def make_low_pass_datasets(dataset, low_pass_times=3, model_type='laplacian'):
    """
        make dataset from files
    """
    dataset = np.array([d.reshape(d.shape[0], 1, d.shape[1]) for d in dataset])
    datasets = [dataset]
    temp = dataset
    for _ in xrange(low_pass_times):
        temp = np.array([moving_avg_low_pass_filter(d) for d in temp])
        datasets.append(temp)
    high_pass_parts = []
    for i in xrange(0, len(datasets) - 1):
        if model_type == 'laplacian':
            Y = np.subtract(datasets[i], datasets[i - 1])
        elif model_type == 'gaussian':
            Y = np.copy(datasets[i])
        Y = np.array([y.reshape(y.shape[0], y.shape[2]) for y in Y])
        high_pass_parts.append(Y)
    low_pass_filtered_parts = datasets[1:]
    assert len(high_pass_parts) == len(low_pass_filtered_parts)

    return low_pass_filtered_parts, high_pass_parts


def get_datasets_for_generator(dataset):
    """
        make dataset from files
    """
    y = np.array([d[1:].reshape(d.shape[0] - 1, d.shape[2]) for d in dataset])
    X = np.array([np.array(d[0:len(d) - 1]) for d in dataset])
    # zero padding to keep dim
    y = np.array([np.append(d, [np.zeros(d[0].shape)], axis=0) for d in y])
    X = np.array([np.append(d, [np.zeros(d[0].shape)], axis=0) for d in X])
    return X, y


# test for all method
# dt = load_dataset()
# print dt.shape

# Xs, ys = make_low_pass_datasets(dt)
# print Xs.shape
# print ys.shape

# X,y = get_datasets_for_generator(Xs[-1])
# print X.shape
# print y.shape
