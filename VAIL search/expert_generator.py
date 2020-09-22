import math
import pickle
import torch

canvas = []
ob1_y= 10
ob1_x = 10

ob2_y = 3      ####### 설정, 원의 중심좌표 (7,2) obs1_x = 7, obs1_y = 2
ob2_x = 3

ob3_y = 16
ob3_x = 16

ob4_y = 16
ob4_x = 2
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, 5)
#         # self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(16, 16, 5)
#         # self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc1 = nn.Linear(9216,120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#
#     def forward(self, x):
#         # x = self.pool(F.relu(self.conv1(x)))
#         # x = self.pool(F.relu(self.conv2(x)))
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         # x = x.view(-1, 16 * 5 * 5)
#         x = x.view(-1, 9216)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#
#         x = self.fc3(x)
#         return x
#
# # Hyper Parameters
# optimizer = torch.optim.Adam(rnn.parameters(), lr=LR, weight_decay=1e-5)   # optimize all cnn parameters
# class RNN(nn.Module):
#     def __init__(self):
#         super(RNN, self).__init__()
#
#         self.rnn = nn.GRU(         # if use nn.RNN(), it hardly learns
#             input_size=INPUT_SIZE,
#             hidden_size=256,         # rnn hidden unit
#             num_layers=2,           # number of rnn layer
#             batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
#         )
#
#         self.out = nn.Linear(256, 10)
#
#     def forward(self, x):
#         # x shape (batch, time_step, input_size)
#         # r_out shape (batch, time_step, output_size)
#         # h_n shape (n_layers, batch, hidden_size)
#         # h_c shape (n_layers, batch, hidden_size)
#         r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
#
#         # choose r_out at the last time step
#         out = self.out(r_out[:, -1, :])
#         return out
for i in range(20):
    tmp = []
    for j in range(20):
        tmp.append(0)
    canvas.append(tmp)

def obstac(a,b):
    for i in range(3):
        for j in range(3):
            canvas[a+i][b+j] = 1



obstac(ob1_x-1,ob1_y-1)

obstac(ob2_x-1,ob2_y-1)             ########장애물설정

obstac(ob3_x-1,ob3_y-1)

obstac(ob4_x-1,ob4_y-1)


for i in canvas:
    print(i)

def dd(s1,s2,o1,o2):
    return math.sqrt(math.pow((s1-o1),2) + math.pow((s2-o2),2))

result = []
act = 1


x = 19
y = 0

while(act!=0):
    act = int(input())
    canvas[x][y] = 8
    d1 = dd(x,y,ob1_x,ob1_y)
    d2 = dd(x,y,ob2_x,ob2_y)
    d3 = dd(x,y,ob3_x,ob3_y)
    d4 = dd(x,y,ob4_x,ob4_y)
    if(act ==1):
        result.append([x,y,d1,d2,d3,d4,1,0,0,0,0,0,0,0])
        x = x-1
    elif(act==2):
        result.append([x,y,d1,d2,d3,d4,0,1,0,0,0,0,0,0])
        x = x+1
    elif(act==3):
        result.append([x,y,d1,d2,d3,d4,0,0,1,0,0,0,0,0])
        y = y-1
    elif(act==4):
        result.append([x,y,d1,d2,d3,d4,0,0,0,1,0,0,0,0])
        y = y+1
    elif(act==5):
        result.append([x,y,d1,d2,d3,d4,0,0,0,0,1,0,0,0])
        y = y-1
        x = x-1
    elif(act==6):
        result.append([x,y,d1,d2,d3,d4,0,0,0,0,0,1,0,0])
        x= x-1
        y = y+1
    elif(act==7):
        result.append([x,y,d1,d2,d3,d4,0,0,0,0,0,0,1,0])
        x = x+1
        y = y+1
    elif(act==8):
        result.append([x,y,d1,d2,d3,d4,0,0,0,0,0,0,0,1])
        x= x+1
        y = y-1
    canvas[x][y] = 7
    for i in canvas:
        print(i)
result.append([19,0])
with open(r'Last1.p','wb') as f:
    pickle.dump(result, f)
print(result)

# 상 : 1 하 : 2 좌 : 3 우 : 4
# 좌상 5, 우상 6