#!/usr/bin/python
# -*- coding: utf-8 -*-

import datetime
from make_data import *
from lstm import *
import numpy as np
from chainer import optimizers, cuda
import time
import sys
import pickle
import random
import read_csv as rc
import config as c

xp = np #cuda.cupy

def compute_loss(model, sequences, answer):
    loss = 0
    rows, cols, v = answer.shape
    length_of_sequence = cols
    for i in range(cols - 1):
        x = chainer.Variable(
            xp.asarray(
                [sequences[j, i + 0] for j in range(rows)],
                dtype=np.float32
            )[:, np.newaxis]
        )
        t = chainer.Variable(
            xp.asarray(
                [answer[j, i + 1] for j in range(rows)],
                dtype=np.float32
            )
        )
        loss += model(x, t)
    return loss


if __name__ == "__main__":
    # ファイルから読み込むようにする
    data_reader = rc.DataReader('sample.csv')
    data_reader.read()

    # setup model
    model = LSTM(c.IN_UNITS, c.HIDDEN_UNITS, c.OUT_UNITS)
    # モデルをランダムに生成する（これってどうなのよ）
    for param in model.params():
        data = param.data
        data[:] = np.random.uniform(-0.1, 0.1, data.shape)
    # model.to_gpu()

    # setup optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    start = time.time()
    cur_start = start
    for epoch in range(c.TRAINING_EPOCHS):
        # batch を作成する
        sequences, answer = data_reader.make_mini_batch(
            mini_batch_size=c.MINI_BATCH_SIZE,
            length_of_sequence=c.LENGTH_OF_SEQUENCE)
        model.reset_state()
        model.zerograds()
        loss = compute_loss(model, sequences, answer)
        loss.backward()
        optimizer.update()

        if epoch != 0 and epoch % c.DISPLAY_EPOCH == 0:
            cur_end = time.time()
            # display loss
            print(
                "[{j}]training loss:\t{i}\t{k}[sec/epoch]".format(
                    j=epoch,
                    i=loss.data/(sequences.shape[1] - 1),
                    k=(cur_end - cur_start)/c.DISPLAY_EPOCH
                )
            )
            cur_start = time.time()
            sys.stdout.flush()

    end = time.time()

    # save model
    now = datetime.datetime.now()
    nowstr = now.strftime("%Y%m%d%H%M%S")
    pickle.dump(model, open("./"+nowstr+"_model.pkl", "wb"))

    print("{}[sec]".format(end - start))
