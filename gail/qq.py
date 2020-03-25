import pickle
import numpy as np

# with open('expert_demo.p', 'wb') as file:
#    for data in text_list:
#         pickle.dump(data, file)

#
# with open('expert_demo.p', 'rb') as file:
#     data_list = []
#     while True:
#         try:
#             data = pickle.load(file)
#         except EOFError:
#             break
#         data_list.append(data)
#
# print(data_list)

# with open('expert_demo.p','rb') as fr:
#     data = pickle.load(fr)
# print(data)



with open('expert_demo.p', 'rb') as file:    # james.p 파일을 바이너리 읽기 모드(rb)로 열기
    name = pickle.load(file)
print(name[0])
print(type(name[0]))
name2 = np.array(name[0])
print(name2.shape)

