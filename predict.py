#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
from chainer import optimizers, cuda
import chainer
from make_data import *
import matplotlib.pyplot as plt
import config as c
import read_csv as rc

xp = np #cuda.cupy

def predict_sequence(model, input_seq, dummy):
    sequences_col = len(input_seq)
    model.reset_state()
    for i in range(sequences_col):
        x = chainer.Variable(xp.asarray(input_seq[i:i+1], dtype=np.float32)[:, np.newaxis])
        future = model(x, dummy)
    cpu_future = chainer.cuda.to_cpu(future.data)
    return cpu_future

def predict(seq, model, pre_length, initial_path, prediction_path):
    # initial sequence
    input_seq = seq[:,:2]
    output_seq0 = np.empty(0)
    output_seq1 = np.empty(0)
    output_seq2 = np.empty(0)

    # append an initial value
    model.train = False
    dummy = chainer.Variable(xp.asarray([[0,0,0]], dtype=np.float32))

    for i in range(pre_length):
        future = predict_sequence(model, input_seq, dummy)
        input_seq = np.delete(input_seq, 0, axis=0)
        input_seq = np.append(input_seq, [future[0,:2]], axis=0)
        output_seq0 = np.append(output_seq0, [future[0][0]])
        output_seq1 = np.append(output_seq1, [future[0][1]])
        output_seq2 = np.append(output_seq2, [future[0][2]])
    """
    plt.plot(seq[:,:].tolist())
    plt.show()
    """
    plt.plot(output_seq0.tolist())
    plt.plot(output_seq1.tolist())
    plt.plot(output_seq2.tolist())
    plt.show()

if __name__ == "__main__":
    # load model
    with open(c.MODEL_PATH, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        model = u.load()

    # Read data
    data_reader = rc.DataReader('sample.csv')
    data_reader.read()

    # make data
    data_maker = DataMaker(
        steps_per_cycle=c.STEPS_PER_CYCLE,
        number_of_cycles=c.NUMBER_OF_CYCLES)
    data = data_maker.make()

    sequences, answer = data_reader.make_mini_batch(
        mini_batch_size=c.MINI_BATCH_SIZE,
        length_of_sequence=c.LENGTH_OF_SEQUENCE)

    plt.plot(answer[45,:,:])
    plt.show()

    sample_index = 45
    predict(answer[sample_index,:,:], model, c.PREDICTION_LENGTH, c.INITIAL_PATH, c.PREDICTION_PATH)
