import pickle
import numpy as np

with open('expert_demo.p','rb') as f:
    data2 = pickle.load(f)
print(data2,'\n',np.array(data2)[0].shape)


# xyt xyt xyt 255 1 데이터 변환
# [1,2,3,4,5,6,7,8,9,0,1 , 3, 5, 6]


# state 값 Z필터 씌움
# data3 = ZFilter((num_inputs,), clip=5)


# action 값 torch.normal 씌움


# 데이터 저장
# with open(r'aa_expert.p', 'wb') as f:
#     pickle.dump(zfilter_data, f)
