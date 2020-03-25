import ndjson
import numpy as np
import cv2
import time

# load from file-like objects
with open('aa.ndjson') as f:
    data = ndjson.load(f)
draw = np.array(data[0]['drawing'])
print(draw[0][2])
## draw[0][0] : x
## draw[0][1] : y
## draw[0][2] : t

width = 1000
height = 1000
x_tmp = draw[0][0][0]
y_tmp = draw[0][1][0]
img = np.zeros((height, width, 1), np.uint8)
for i in range(len(draw[0][0])):
    if i == 20:
        time.sleep(1)
    x = draw[0][0][i]
    y = draw[0][1][i]
    cv2.line(img, (x, y), (x_tmp, y_tmp), (255, 0, 0), 3)
    x_tmp = x
    y_tmp = y
cv2.imshow("re",img)
cv2.waitKey(0);