#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import math
import pandas as pd
import random
import matplotlib.pyplot as plt

class DataReader(object):

    def __init__(self, path):
        self.path = path

    def read(self):
        data = pd.read_csv(self.path)
        self.time = np.array(data.time, dtype=np.int32)
        self.ans = np.array(data.ans, dtype=np.float32)
        self.datas = np.array(data[data.columns[2:]], dtype=np.float32)
        print(self.datas.shape)

    def make_mini_batch(self, mini_batch_size, length_of_sequence):
        sequences = np.ndarray((mini_batch_size, length_of_sequence, self.datas.shape[1]), dtype=np.float32)
        answer = np.ndarray((mini_batch_size, length_of_sequence), dtype=np.float32)
        for i in range(mini_batch_size):
            index = random.randint(0, len(self.datas) - length_of_sequence)
            sequences[i] = self.datas[index:index+length_of_sequence,:]
            answer[i] = self.ans[index:index+length_of_sequence]
        return sequences, answer


if __name__ == "__main__":
    dr = DataReader('sample.csv')
    dr.read()
