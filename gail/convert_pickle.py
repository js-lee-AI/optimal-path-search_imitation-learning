import pickle
import ndjson
import numpy as np
from utils.zfilter import ZFilter

import os
import gym
import pickle
import argparse
import numpy as np
from collections import deque

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from utils.utils import *
from utils.zfilter import ZFilter
from model import Actor, Critic, Discriminator
from train_model import train_actor_critic, train_discrim




with open('aa.ndjson') as f:
    data = ndjson.load(f)
draw = np.array(data[0]['drawing'], dtype=np.float64)
print(draw)
print(draw.shape)
print(draw[0][0].shape)

draw = np.reshape(draw, (3, 165))

new_raw = []
for i in range(len(draw[0])):
    # new_raw.append([draw[0][i], draw[1][i], draw[2][i], draw[0][i+1], draw[1][i+1], draw[2][i+1], draw[0][i+2], draw[1][i+2], draw[2][i+2], 255, 1])
    new_raw.append([draw[0][i], draw[1][i], draw[2][i], draw[0][i], draw[1][i], draw[2][i], draw[0][i],
                    draw[1][i], draw[2][i], 255, 1, 0, 1, 2])

new_raw = np.array(new_raw)
print(new_raw.shape)

all_data = np.zeros((50000, 14), dtype=np.float64)

for i in range(50000):
    all_data[i][:] = new_raw[i%164][:]

print(all_data.shape)
# all_data = np.reshape(all_data, 50000, 14)
# print(all_data.shape)
# new_data = np.reshape(new_data, (2, 2, 1))

# new_data = [draw[0][i], draw[1][i], draw[2][i]]
running_state = ZFilter((14,), clip=5)

zfilter_data = np.zeros((50000, 14), dtype=np.float64)
a = np.array(all_data[-1][:],dtype=float)

for i in range(50000):
      zfilter_data[i][:] = running_state(all_data[i][:])
print('zfilter data = ' ,all_data[-1][:],running_state(all_data[-1][:]))



#
with open('expert_demo.p','rb') as f:
    data2 = pickle.load(f)
print(data2,'\n',np.array(data2)[0].shape)

with open('aa_expert.p','rb') as f:
    data2 = pickle.load(f)
print(data2,'\n')

# xyt xyt xyt 255 1 데이터 변환
# [1,2,3,4,5,6,7,8,9,0,1 , 3, 5, 6]


# state 값 Z필터 씌움
# data3 = ZFilter((num_inputs,), clip=5)


# action 값 torch.normal 씌움


# 데이터 저장
# with open(r'aa_expert.p', 'wb') as f:
#     pickle.dump(zfilter_data, f)
