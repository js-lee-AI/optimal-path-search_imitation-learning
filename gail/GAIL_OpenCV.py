import cv2
import numpy as np
import json

with open('aa.ndjson') as json_file:
    json_data = json.load(json_file)
    json_data = json_data['drawing']

print(json_data)

drawings = False  # True 이면 마우스가 눌린 상태입니다.
mode = True  # True이면 사각형을 그립니다. 'm'을 누르면 곡선으로 변경(토글)됩니다
ix, iy = -1, -1


# 마우스 콜백 함수
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawings, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawings = True
        ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawings = False
        if mode == True:
            cv2.circle(img, (x, y), 1, (0, 0, 0), -1)
        else:
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

img = np.zeros((512, 512, 3), np.uint8)
img = img + 255

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

# cv2.circle(img, (x, y), 1, (0, 0, 0), -1)


while (1):
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break

cv2.destroyAllWindows()