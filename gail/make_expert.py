import ndjson
import numpy as np
import random
import pickle
import torch
# load from file-like objects
import torch.optim as optim
from torch import nn
import argparse

temp_learner = []
temp_expert = []

parser = argparse.ArgumentParser(description='PyTorch GAIL')
parser.add_argument('--env_name', type=str, default="Hopper-v2",
                    help='name of the environment to run')
parser.add_argument('--load_model', type=str, default=None,
                    help='path to load the saved model')
parser.add_argument('--render', action="store_true", default=False,
                    help='if you dont want to render, set this to False')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discounted factor (default: 0.99)')
parser.add_argument('--lamda', type=float, default=0.98,
                    help='GAE hyper-parameter (default: 0.98)')
parser.add_argument('--hidden_size', type=int, default=100,
                    help='hidden unit size of actor, critic and discrim networks (default: 100)')
parser.add_argument('--learning_rate', type=float, default=3e-4,
                    help='learning rate of models (default: 3e-4)')
parser.add_argument('--l2_rate', type=float, default=1e-3,
                    help='l2 regularizer coefficient (default: 1e-3)')
parser.add_argument('--clip_param', type=float, default=0.2,
                    help='clipping parameter for PPO (default: 0.2)')
parser.add_argument('--discrim_update_num', type=int, default=2,
                    help='update number of discriminator (default: 2)')
parser.add_argument('--actor_critic_update_num', type=int, default=10,
                    help='update number of actor-critic (default: 10)')
parser.add_argument('--total_sample_size', type=int, default=2048,     ##2048
                    help='total sample size to collect before PPO update (default: 2048)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size to update (default: 64)')
parser.add_argument('--suspend_accu_exp', type=float, default=0.8,
                    help='accuracy for suspending discriminator about expert data (default: 0.8)')
parser.add_argument('--suspend_accu_gen', type=float, default=0.8,
                    help='accuracy for suspending discriminator about generated data (default: 0.8)')

parser.add_argument('--max_iter_num', type=int, default=5,
                    help='maximal number of main iterations (default: 4000)')
# parser.add_argument('--max_iter_num', type=int, default=4000,
                    # help='maximal number of main iterations (default: 4000)')


parser.add_argument('--seed', type=int, default=500,
                    help='random seed (default: 500)')
parser.add_argument('--logdir', type=str, default='logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()
class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        self._n = n

    @property
    def mean(self):
        return self._M

    @mean.setter
    def mean(self, M):
        self._M = M

    @property
    def sum_square(self):
        return self._S

    @sum_square.setter
    def sum_square(self, S):
        self._S = S

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape
class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)

        if self.demean:
            x = x - self.rs.mean

        if self.destd:
            x = x / (self.rs.std + 1e-8)

        if self.clip:
            x = np.clip(x, -self.clip, self.clip)

        return x

class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs, args):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_inputs, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, num_outputs)


        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = self.fc3(x)
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)
        return mu, std

def get_action(mu, std):
    action = torch.normal(mu, std)
    action = action.data.numpy()
    return action

def make_sample():
    result = [[20,20]]
    for i in range(21,120+1):
        result.append([i,20])
    for i in range(21,120+1):
        result.append([120,i])
    for i in range(119,19,-1):
        result.append([i,120])
    for i in range(119,20,-1):
        result.append([20,i])
    new_arr = []


    actor = Actor(2, 8, args)

    for i in range(len(result)-1):
        x1 = result[i][0]
        y1 = result[i][1]
        x2 = result[i+1][0]
        y2 = result[i+1][1]
        if (y1 == y2 and x1 < x2):
            new_arr.append([x1, y1, 0, 0, 0, 1, 0, 0, 0, 0])  ## Right
        elif (y1 == y2 and x1 > x2):
            new_arr.append([x1, y1, 0, 0, 1, 0, 0, 0, 0, 0])  ## Left
        elif (x1 == x2 and y1 > y2):
            new_arr.append([x1, y1, 1, 0, 0, 0, 0, 0, 0, 0])  ## Up
        elif (x1 == x2 and y1 < y2):
            new_arr.append([x1, y1, 0, 1, 0, 0, 0, 0, 0, 0])  ## Down
        elif (x1 < x2 and y1 > y2):
            new_arr.append([x1, y1, 0, 0, 0, 0, 1, 0, 0, 0])  ## UP | RIGHT
        elif (x1 < x2 and y1 < y2):
            new_arr.append([x1, y1, 0, 0, 0, 0, 0, 0, 1, 0])  ## DOWN | RIGHT
        elif (x1 > x2 and y1 > y2):
            new_arr.append([x1, y1, 0, 0, 0, 0, 0, 1, 0, 0])  ## UP | LEFT
        elif (x1 > x2 and y1 < y2):
            new_arr.append([x1, y1, 0, 0, 0, 0, 0, 0, 0, 1])  ## DONW | LEFT
    new_arr = np.array(new_arr)  ## Cast Numpy array
    running_state = ZFilter((10,), clip=5)  ## ZFilter : num_state(2) + num_action(8)
    zfilter_data = np.zeros((len(new_arr), 10), dtype=np.float64)
    for i in range(len(new_arr)):
        zfilter_data[i][:] = running_state(new_arr[i][:])  ## Cast ZFilter
        print("1: ",zfilter_data)
        state = [zfilter_data[i][0],zfilter_data[i][1]]
        mu, std = actor(torch.Tensor(state).unsqueeze(0))
        action = get_action(mu, std)[0]
        zfilter_data[i][2:10] =action[:]
        print("2: ",zfilter_data)
    result1 = []
    result1.append(zfilter_data)
    result1.append([result[0][0],result[0][1]])
    with open(r'Ree_expert.p', 'wb') as f:  ## Pickling
        pickle.dump(result1, f)


def make_expert_data(ndj_file_name):
    with open(ndj_file_name) as f:
        data = ndjson.load(f)
    # draw = data[0]["drawing"]
    draw = make_sample()
    new_arr = []


    for i in range(0,len(draw[0][0])-1):
        x1 = int(draw[0][0][i])
        y1 = int(draw[0][1][i])   ## draw[0][0] : x  ## draw[0][1] : y  ## draw[0][2] : t
        x2 = int(draw[0][0][i+1])
        y2 = int(draw[0][1][i+1])
        if(y1 == y2 and x1 < x2):
            new_arr.append([x1,y1,0,0,0,1,0,0,0,0]) ## Right
        elif(y1 == y2 and x1 > x2):
            new_arr.append([x1,y1,0,0,1,0,0,0,0,0]) ## Left
        elif(x1 == x2 and y1 > y2):
            new_arr.append([x1,y1,1,0,0,0,0,0,0,0]) ## Up
        elif(x1 == x2 and y1 < y2):
            new_arr.append([x1,y1,0,1,0,0,0,0,0,0]) ## Down
        elif(x1 < x2 and y1 > y2):
            new_arr.append([x1,y1,0,0,0,0,1,0,0,0]) ## UP | RIGHT
        elif(x1 <x2 and y1 < y2):
            new_arr.append([x1,y1,0,0,0,0,0,0,1,0]) ## DOWN | RIGHT
        elif(x1 > x2 and y1 > y2):
            new_arr.append([x1,y1,0,0,0,0,0,1,0,0]) ## UP | LEFT
        elif(x1 > x2 and y1 < y2):
            new_arr.append([x1,y1,0,0,0,0,0,0,0,1]) ## DONW | LEFT

    new_arr = np.array(new_arr) ## Cast Numpy array
    running_state = ZFilter((10,), clip=5) ## ZFilter : num_state(2) + num_action(8)
    zfilter_data = np.zeros((164, 10), dtype=np.float64)

    for i in range(164):
          zfilter_data[i][:] = running_state(new_arr[i][:]) ## Cast ZFilter
    result=[]
    result.append(zfilter_data)

    result.append([int(draw[0][0][0]),int(draw[0][1][0])])
    print(result[1][1])
    with open(r'Lee_expert.p', 'wb') as f: ## Pickling
        pickle.dump(result, f)

make_sample()

import ndjson
import numpy as np
import random
import pickle
# load from file-like objects

class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        self._n = n

    @property
    def mean(self):
        return self._M

    @mean.setter
    def mean(self, M):
        self._M = M

    @property
    def sum_square(self):
        return self._S

    @sum_square.setter
    def sum_square(self, S):
        self._S = S

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape
class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)

        if self.demean:
            x = x - self.rs.mean

        if self.destd:
            x = x / (self.rs.std + 1e-8)

        if self.clip:
            x = np.clip(x, -self.clip, self.clip)

        return x
def make_sample():
    result = [[20,20]]
    for i in range(21,120+1):
        result.append([i,20])
    for i in range(21,120+1):
        result.append([120,i])
    for i in range(119,19,-1):
        result.append([i,120])
    for i in range(119,20,-1):
        result.append([20,i])
    new_arr = []
    print(result)
    print(len(result))
    for i in range(len(result)-1):
        x1 = result[i][0]
        y1 = result[i][1]
        x2 = result[i+1][0]
        y2 = result[i+1][1]
        if (y1 == y2 and x1 < x2):
            new_arr.append([x1, y1, 0, 0, 0, 1, 0, 0, 0, 0])  ## Right
        elif (y1 == y2 and x1 > x2):
            new_arr.append([x1, y1, 0, 0, 1, 0, 0, 0, 0, 0])  ## Left
        elif (x1 == x2 and y1 > y2):
            new_arr.append([x1, y1, 1, 0, 0, 0, 0, 0, 0, 0])  ## Up
        elif (x1 == x2 and y1 < y2):
            new_arr.append([x1, y1, 0, 1, 0, 0, 0, 0, 0, 0])  ## Down
        elif (x1 < x2 and y1 > y2):
            new_arr.append([x1, y1, 0, 0, 0, 0, 1, 0, 0, 0])  ## UP | RIGHT
        elif (x1 < x2 and y1 < y2):
            new_arr.append([x1, y1, 0, 0, 0, 0, 0, 0, 1, 0])  ## DOWN | RIGHT
        elif (x1 > x2 and y1 > y2):
            new_arr.append([x1, y1, 0, 0, 0, 0, 0, 1, 0, 0])  ## UP | LEFT
        elif (x1 > x2 and y1 < y2):
            new_arr.append([x1, y1, 0, 0, 0, 0, 0, 0, 0, 1])  ## DONW | LEFT
    new_arr = np.array(new_arr)  ## Cast Numpy array
    running_state = ZFilter((2,), clip=5)  ## ZFilter : num_state(2) + num_action(8)
    zfilter_data = np.zeros((len(new_arr), 10), dtype=np.float64)
    for i in range(len(new_arr)):
        zfilter_data[i][:2] = running_state(new_arr[i][:2])  ## Cast ZFilter
        zfilter_data[i][2:] = new_arr[i][2:]
    result1 = []
    result1.append(zfilter_data)
    result1.append([result[0][0],result[0][1]])
    print(result1[0][222])
    with open(r'Ree1_expert.p', 'wb') as f:  ## Pickling
        pickle.dump(result1, f)


def make_expert_data(ndj_file_name):
    with open(ndj_file_name) as f:
        data = ndjson.load(f)
    # draw = data[0]["drawing"]
    draw = make_sample()
    new_arr = []
    for i in range(0,len(draw[0][0])-1):
        x1 = int(draw[0][0][i])
        y1 = int(draw[0][1][i])   ## draw[0][0] : x  ## draw[0][1] : y  ## draw[0][2] : t
        x2 = int(draw[0][0][i+1])
        y2 = int(draw[0][1][i+1])
        if(y1 == y2 and x1 < x2):
            new_arr.append([x1,y1,0,0,0,1,0,0,0,0]) ## Right
        elif(y1 == y2 and x1 > x2):
            new_arr.append([x1,y1,0,0,1,0,0,0,0,0]) ## Left
        elif(x1 == x2 and y1 > y2):
            new_arr.append([x1,y1,1,0,0,0,0,0,0,0]) ## Up
        elif(x1 == x2 and y1 < y2):
            new_arr.append([x1,y1,0,1,0,0,0,0,0,0]) ## Down
        elif(x1 < x2 and y1 > y2):
            new_arr.append([x1,y1,0,0,0,0,1,0,0,0]) ## UP | RIGHT
        elif(x1 <x2 and y1 < y2):
            new_arr.append([x1,y1,0,0,0,0,0,0,1,0]) ## DOWN | RIGHT
        elif(x1 > x2 and y1 > y2):
            new_arr.append([x1,y1,0,0,0,0,0,1,0,0]) ## UP | LEFT
        elif(x1 > x2 and y1 < y2):
            new_arr.append([x1,y1,0,0,0,0,0,0,0,1]) ## DONW | LEFT

    new_arr = np.array(new_arr) ## Cast Numpy array
    running_state = ZFilter((10,), clip=5) ## ZFilter : num_state(2) + num_action(8)
    zfilter_data = np.zeros((164, 10), dtype=np.float64)

    for i in range(164):
          zfilter_data[i][:] = running_state(new_arr[i][:]) ## Cast ZFilter
    result=[]
    result.append(zfilter_data)
    result.append([int(draw[0][0][0]),int(draw[0][1][0])])
    print(result[1][1])
    with open(r'Lee_expert.p', 'wb') as f: ## Pickling
        pickle.dump(result, f)

make_sample()

####################### 000001##############
# import ndjson
# import numpy as np
# import random
# import pickle
# # load from file-like objects
#
# class RunningStat(object):
#     def __init__(self, shape):
#         self._n = 0
#         self._M = np.zeros(shape)
#         self._S = np.zeros(shape)
#
#     def push(self, x):
#         x = np.asarray(x)
#         assert x.shape == self._M.shape
#         self._n += 1
#         if self._n == 1:
#             self._M[...] = x
#         else:
#             oldM = self._M.copy()
#             self._M[...] = oldM + (x - oldM) / self._n
#             self._S[...] = self._S + (x - oldM) * (x - self._M)
#
#     @property
#     def n(self):
#         return self._n
#
#     @n.setter
#     def n(self, n):
#         self._n = n
#
#     @property
#     def mean(self):
#         return self._M
#
#     @mean.setter
#     def mean(self, M):
#         self._M = M
#
#     @property
#     def sum_square(self):
#         return self._S
#
#     @sum_square.setter
#     def sum_square(self, S):
#         self._S = S
#
#     @property
#     def var(self):
#         return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)
#
#     @property
#     def std(self):
#         return np.sqrt(self.var)
#
#     @property
#     def shape(self):
#         return self._M.shape
# class ZFilter:
#     """
#     y = (x-mean)/std
#     using running estimates of mean,std
#     """
#
#     def __init__(self, shape, demean=True, destd=True, clip=10.0):
#         self.demean = demean
#         self.destd = destd
#         self.clip = clip
#
#         self.rs = RunningStat(shape)
#
#     def __call__(self, x, update=True):
#         if update: self.rs.push(x)
#
#         if self.demean:
#             x = x - self.rs.mean
#
#         if self.destd:
#             x = x / (self.rs.std + 1e-8)
#
#         if self.clip:
#             x = np.clip(x, -self.clip, self.clip)
#
#         return x
# def make_sample():
#     result = [[20,20]]
#     for i in range(21,120+1):
#         result.append([i,20])
#     for i in range(21,120+1):
#         result.append([120,i])
#     for i in range(119,19,-1):
#         result.append([i,120])
#     for i in range(119,20,-1):
#         result.append([20,i])
#     new_arr = []
#     print(result)
#     print(len(result))
#     for i in range(len(result)-1):
#         x1 = result[i][0]
#         y1 = result[i][1]
#         x2 = result[i+1][0]
#         y2 = result[i+1][1]
#         if (y1 == y2 and x1 < x2):
#             new_arr.append([x1, y1, 0, 0, 0, 1, 0, 0, 0, 0])  ## Right
#         elif (y1 == y2 and x1 > x2):
#             new_arr.append([x1, y1, 0, 0, 1, 0, 0, 0, 0, 0])  ## Left
#         elif (x1 == x2 and y1 > y2):
#             new_arr.append([x1, y1, 1, 0, 0, 0, 0, 0, 0, 0])  ## Up
#         elif (x1 == x2 and y1 < y2):
#             new_arr.append([x1, y1, 0, 1, 0, 0, 0, 0, 0, 0])  ## Down
#         elif (x1 < x2 and y1 > y2):
#             new_arr.append([x1, y1, 0, 0, 0, 0, 1, 0, 0, 0])  ## UP | RIGHT
#         elif (x1 < x2 and y1 < y2):
#             new_arr.append([x1, y1, 0, 0, 0, 0, 0, 0, 1, 0])  ## DOWN | RIGHT
#         elif (x1 > x2 and y1 > y2):
#             new_arr.append([x1, y1, 0, 0, 0, 0, 0, 1, 0, 0])  ## UP | LEFT
#         elif (x1 > x2 and y1 < y2):
#             new_arr.append([x1, y1, 0, 0, 0, 0, 0, 0, 0, 1])  ## DONW | LEFT
#     new_arr = np.array(new_arr)  ## Cast Numpy array
#     running_state = ZFilter((2,), clip=5)  ## ZFilter : num_state(2) + num_action(8)
#     zfilter_data = np.zeros((len(new_arr), 10), dtype=np.float64)
#     for i in range(len(new_arr)):
#         zfilter_data[i][:2] = running_state(new_arr[i][:2])  ## Cast ZFilter
#         zfilter_data[i][2:] = new_arr[i][2:]
#     result1 = []
#     result1.append(zfilter_data)
#     result1.append([result[0][0],result[0][1]])
#     print(result1[0][222])
#     with open(r'Ree1_expert.p', 'wb') as f:  ## Pickling
#         pickle.dump(result1, f)
#
#
# def make_expert_data(ndj_file_name):
#     with open(ndj_file_name) as f:
#         data = ndjson.load(f)
#     # draw = data[0]["drawing"]
#     draw = make_sample()
#     new_arr = []
#     for i in range(0,len(draw[0][0])-1):
#         x1 = int(draw[0][0][i])
#         y1 = int(draw[0][1][i])   ## draw[0][0] : x  ## draw[0][1] : y  ## draw[0][2] : t
#         x2 = int(draw[0][0][i+1])
#         y2 = int(draw[0][1][i+1])
#         if(y1 == y2 and x1 < x2):
#             new_arr.append([x1,y1,0,0,0,1,0,0,0,0]) ## Right
#         elif(y1 == y2 and x1 > x2):
#             new_arr.append([x1,y1,0,0,1,0,0,0,0,0]) ## Left
#         elif(x1 == x2 and y1 > y2):
#             new_arr.append([x1,y1,1,0,0,0,0,0,0,0]) ## Up
#         elif(x1 == x2 and y1 < y2):
#             new_arr.append([x1,y1,0,1,0,0,0,0,0,0]) ## Down
#         elif(x1 < x2 and y1 > y2):
#             new_arr.append([x1,y1,0,0,0,0,1,0,0,0]) ## UP | RIGHT
#         elif(x1 <x2 and y1 < y2):
#             new_arr.append([x1,y1,0,0,0,0,0,0,1,0]) ## DOWN | RIGHT
#         elif(x1 > x2 and y1 > y2):
#             new_arr.append([x1,y1,0,0,0,0,0,1,0,0]) ## UP | LEFT
#         elif(x1 > x2 and y1 < y2):
#             new_arr.append([x1,y1,0,0,0,0,0,0,0,1]) ## DONW | LEFT
#
#     new_arr = np.array(new_arr) ## Cast Numpy array
#     running_state = ZFilter((10,), clip=5) ## ZFilter : num_state(2) + num_action(8)
#     zfilter_data = np.zeros((164, 10), dtype=np.float64)
#
#     for i in range(164):
#           zfilter_data[i][:] = running_state(new_arr[i][:]) ## Cast ZFilter
#     result=[]
#     result.append(zfilter_data)
#     result.append([int(draw[0][0][0]),int(draw[0][1][0])])
#     print(result[1][1])
#     with open(r'Lee_expert.p', 'wb') as f: ## Pickling
#         pickle.dump(result, f)
#
# make_sample()

