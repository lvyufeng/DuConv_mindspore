import collections
import math
import os
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.nn as nn
import mindspore.ops as P

from mindspore import dtype as mstype
from mindspore import log as logger
from mindspore._checkparam import Validator as validator
from mindspore.common.tensor import Tensor
from mindspore.nn.learning_rate_schedule import (LearningRateSchedule,PolynomialDecayLR, WarmUpLR)
from mindspore.train.callback import Callback
def linear_warmup(warmup_steps, current_step):
    return min([1.0, float(current_step)/float(warmup_steps)])

def rsqrt_decay(warmup_steps, current_step):
    return float(max([current_step, warmup_steps])) ** -0.5

def rsqrt_hidden(hidden_size):
    return float(hidden_size) ** -0.5

def CustomWarmUpLR(d_model, learning_rate, warmup_steps, training_steps):
    lr = []
    # start_decay_step = 16000
    for current_step in range(1, training_steps+1):
        lr_value = learning_rate
        if current_step > 1001:
            a = float(current_step**-0.5)
            b = float((warmup_steps**-1.5) * current_step)
            c = min([a,b])
            lr_value = learning_rate * float(d_model** -0.5) * c
            lr_value = lr_value * learning_rate
        lr.append(lr_value)
        # else:
        #     if start_decay_step < warmup_steps:
        #         start_decay_step = warmup_steps
        #     cur_lr = 1.0 *  float(learning_rate)
        #     cur_lr = cur_lr * rsqrt_hidden(1024)
        #     cur_lr = cur_lr * linear_warmup(warmup_steps, current_step)
        #     cur_lr = cur_lr * rsqrt_decay(warmup_steps, current_step-start_decay_step+warmup_steps)
        #     if warmup_steps < current_step < start_decay_step:
        #         cur_lr = lr[-1]
        #     if current_step > warmup_steps:
        #         cur_lr = max([cur_lr, 0.0])
        #     lr.append(cur_lr*learning_rate)
    return lr 

# class CustomWarmUpLR(LearningRateSchedule):
#     """
#     apply the functions to  the corresponding input fields.
#     ·
#     """
#     def __init__(self, d_model, learning_rate, warmup_steps, max_train_steps):
#         super(CustomWarmUpLR, self).__init__()
#         if not isinstance(learning_rate, float):
#             raise TypeError("learning_rate must be float.")
#         validator.check_non_negative_float(learning_rate, "learning_rate", self.cls_name)
#         validator.check_positive_int(warmup_steps, 'warmup_steps', self.cls_name)
#         self.warmup_steps = warmup_steps
#         self.learning_rate = learning_rate
#         self.max_train_steps = max_train_steps
#         #self.base_lr = learning_rate
#         self.d_model = d_model
#         self.cast = P.Cast()

#     def construct(self, current_step):
#         a = float(current_step**-0.5)
#         b = float((self.warmup_steps**-1.5) * current_step)
#         if a < b:
#             c= a
#         else:
#             c =b
#         lr_value = self.learning_rate * float(self.d_model** -0.5) * c

#         # lr_value = self.base_lr * np.power(self.d_model, -0.5) * np.min([
#         #                np.power(current_step, -0.5),
#         #                np.power(self.warmup_steps, -1.5) * current_step])
#         # if current_step < self.warmup_steps:
#         #     warmup_percent = self.cast(current_step, mstype.float32)/ self.warmup_steps
#         # else:
#         #     warmup_percent = 1 - self.cast(current_step, mstype.float32)/ self.max_train_steps
#         #print("lr: ", self.learning_rate * warmup_percent)
#         # return self.learning_rate * warmup_percent
#         #print("lr_valuel", lr_value)
#         return lr_value * self.learning_rate


# """lr generator for deeptext"""
# import math

# def rsqrt_decay(warmup_steps, current_step):
#     return float(max([current_step, warmup_steps])) ** -0.5

# def linear_warmup_learning_rate(current_step, warmup_steps, base_lr, init_lr):
#     lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
#     learning_rate = float(init_lr) + lr_inc * current_step
#     return learning_rate

# def a_cosine_learning_rate(current_step, base_lr, warmup_steps, total_steps):
#     decay_steps = total_steps - warmup_steps
#     linear_decay = (total_steps - current_step) / decay_steps
#     cosine_decay = 0.5 * (1 + math.cos(math.pi * 2 * 0.47 * current_step / decay_steps))
#     decayed = linear_decay * cosine_decay + 0.00001
#     learning_rate = decayed * base_lr
#     return learning_rate

# def dynamic_lr(base_lr, num_epochs, warmup_step, warmup_ratio, base_step):
#     """dynamic learning rate generator"""
#     base_lr = base_lr
#     total_steps = int(base_step * num_epochs)
#     warmup_steps = int(warmup_step)
#     lr = []
#     for i in range(total_steps):
#         if i < warmup_steps:
#             lr.append(linear_warmup_learning_rate(i, warmup_steps, base_lr, base_lr * warmup_ratio))
#         else:
#             lr.append(a_cosine_learning_rate(i, base_lr, warmup_steps, total_steps))
#     return lr
