#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import math
import random

random.seed(0)

class DataMaker(object):

    def __init__(self, steps_per_cycle, number_of_cycles):
        self.steps_per_cycle = steps_per_cycle
        self.number_of_cycles = number_of_cycles

    def make(self):
        return np.array(
            [math.sin(i * 2 * math.pi/self.steps_per_cycle)
            +math.sin(i * 4 * math.pi/self.steps_per_cycle)
            +math.sin(i * 6 * math.pi/self.steps_per_cycle)
            +math.sin(i * 8 * math.pi/self.steps_per_cycle)
            +math.sin(i * 16 * math.pi/self.steps_per_cycle)
            +math.sin(i * 32 * math.pi/self.steps_per_cycle)
            + i*0.1
             for i in range(self.steps_per_cycle)] * self.number_of_cycles)

    def make_mini_batch(self, data, mini_batch_size, length_of_sequence):
        sequences = np.ndarray((mini_batch_size, length_of_sequence), dtype=np.float32)
        for i in range(mini_batch_size):
            index = random.randint(0, len(data) - length_of_sequence)
            sequences[i] = data[index:index+length_of_sequence]
        return sequences
