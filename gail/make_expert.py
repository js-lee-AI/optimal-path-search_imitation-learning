import ndjson
import numpy as np
import cv2
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
    running_state = ZFilter((10,), clip=5)  ## ZFilter : num_state(2) + num_action(8)
    zfilter_data = np.zeros((len(new_arr), 10), dtype=np.float64)
    for i in range(len(new_arr)):
        zfilter_data[i][:] = running_state(new_arr[i][:])  ## Cast ZFilter
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

