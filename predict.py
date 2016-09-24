#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
from chainer import optimizers, cuda
import chainer
from make_data import *
import matplotlib.pyplot as plt
import config as c

xp = np #cuda.cupy

def predict_sequence(model, input_seq, output_seq, dummy):
    sequences_col = len(input_seq)
    model.reset_state()
    for i in range(sequences_col):
        x = chainer.Variable(xp.asarray(input_seq[i:i+1], dtype=np.float32)[:, np.newaxis])
        future = model(x, dummy)
    cpu_future = chainer.cuda.to_cpu(future.data)
    return cpu_future

def predict(seq, model, pre_length, initial_path, prediction_path):
    # initial sequence
    input_seq = np.array(seq[:seq.shape[0]/4])
    output_seq = np.empty(0)

    # append an initial value
    output_seq = np.append(output_seq, input_seq[-1])

    model.train = False
    dummy = chainer.Variable(xp.asarray([0], dtype=np.float32)[:, np.newaxis])

    for i in range(pre_length):
        future = predict_sequence(model, input_seq, output_seq, dummy)
        input_seq = np.delete(input_seq, 0)
        input_seq = np.append(input_seq, future)
        output_seq = np.append(output_seq, future)

    with open(prediction_path, "w") as f:
        for (i, v) in enumerate(output_seq.tolist(), start=input_seq.shape[0]):
            f.write("{i} {v}\n".format(i=i-1, v=v))

    with open(initial_path, "w") as f:
        for (i, v) in enumerate(seq.tolist()):
            f.write("{i} {v}\n".format(i=i, v=v))

    plt.plot(seq.tolist())
    plt.plot([None for i in range(24)]+output_seq.tolist())
    plt.show()


if __name__ == "__main__":
    # load model
    with open(c.MODEL_PATH, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        model = u.load()

    # make data
    data_maker = DataMaker(
        steps_per_cycle=c.STEPS_PER_CYCLE,
        number_of_cycles=c.NUMBER_OF_CYCLES)
    data = data_maker.make()
    sequences = data_maker.make_mini_batch(data,
        mini_batch_size=c.MINI_BATCH_SIZE,
        length_of_sequence=c.LENGTH_OF_SEQUENCE)

    sample_index = 45
    predict(sequences[sample_index], model, c.PREDICTION_LENGTH, c.INITIAL_PATH, c.PREDICTION_PATH)
