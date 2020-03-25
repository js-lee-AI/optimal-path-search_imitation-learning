import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image

np.random.seed(1)
PhotoImage = ImageTk.PhotoImage
UNIT = 3  # 픽셀 수
HEIGHT = 500  # 그리드 월드 가로
WIDTH = 500  # 그리드 월드 세로

width = 1

class Env(tk.Tk):

    def __init__(self, a, b):
        self.a = int(UNIT * a - UNIT/2)
        self.b = int(UNIT * b - UNIT/2)
        super(Env, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('GridWorld')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()
        self.canvas = self.build_canvas(self.a, self.b)
        self.texts = []

        print('self.a = ', self.a)
        print('self.b = ', self.b)
        ####
        self.previous = 0
        self.r = [] # 이전 상태 사각형 그리기

    def callback(event):
        canvas = event.widget
        x = canvas.canvasx(event.x)
        y = canvas.canvasy(event.y)
        canvas.find_closest(x, y)



    def build_canvas(self, a, b):
        canvas = tk.Canvas(self, bg='white',
                           height=HEIGHT * UNIT * 10, # 10곱함 임시로
                           width=WIDTH * UNIT * 10) # 10곱함 임시로





         ### 여러번 그리지 않게 조건문 생성

        # 그리드 생성
        # for c in range(0, WIDTH * UNIT, UNIT):  # 0~400 by 80
        #     x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
        #     canvas.create_line(x0, y0, x1, y1)
        # for r in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
        #     x0, y0, x1, y1 = 0, r, HEIGHT * UNIT, r
        #     canvas.create_line(x0, y0, x1, y1)

        # 캔버스에 이미지 추가
        self.rectangle = canvas.create_image(a, b, image=self.shapes[0])
        print('a = ', a)
        print('b = ', b)
        # self.triangle1 = canvas.create_image(250, 150, image=self.shapes[1])
        # self.triangle2 = canvas.create_image(150, 250, image=self.shapes[1])
        # self.circle = canvas.create_image(250, 250, image=self.shapes[2])



        label = tk.Label(self, text=str(width))
        label.pack()

        def scroll(event):
            global width
            if event.delta == 120:
                width += 1
            if event.delta == -120:
                width -= 1
            label.config(text=str(width))
        canvas.pack(expand=True, fill="both")

        canvas.bind("<MouseWheel>", scroll)
        return canvas


    ### 이전 상태 기록하고 그리는 메서드
    def previous_state(self, pre_x, pre_y):
        rects = self.canvas.create_image(pre_x, pre_y, image=self.shapes[0])
        self.canvas.pack()
        return self.r.append(rects)


    def load_images(self):
        rectangle = PhotoImage(
            Image.open("../img/rectangle.png").resize((UNIT, UNIT)))
        triangle = PhotoImage(
            Image.open("../img/triangle.png").resize((65, 65)))
        circle = PhotoImage(
            Image.open("../img/circle.png").resize((65, 65)))

        return rectangle, triangle, circle

    def text_value(self, row, col, contents, action, font='Helvetica', size=10,
                   style='normal', anchor="nw"):
        if action == 0:
            origin_x, origin_y = 7, 42
        elif action == 1:
            origin_x, origin_y = 85, 42
        elif action == 2:
            origin_x, origin_y = 42, 5
        else:
            origin_x, origin_y = 42, 77

        x, y = origin_y + (UNIT * col), origin_x + (UNIT * row)
        font = (font, str(size), style)
        text = self.canvas.create_text(x, y, fill="black", text=contents,
                                       font=font, anchor=anchor)
        return self.texts.append(text)

    def print_value_all(self, q_table):
        for i in self.texts:
            self.canvas.delete(i)
        # self.texts.clear()
        for x in range(HEIGHT):
            for y in range(WIDTH):
                for action in range(0, 4):
                    state = [x, y]
                    if str(state) in q_table.keys():
                        temp = q_table[str(state)][action]
                        # self.text_value(y, x, round(temp, 2), action)

    def coords_to_state(self, coords):
        x = int((coords[0] - 5) / 10)
        y = int((coords[1] - 5) / 10)
        return [x, y]

    def reset(self):
        self.update()
        time.sleep(0.5)
        x, y = self.canvas.coords(self.rectangle) # 시작점을 주면 됨.
        # print('dd = ', x, y)
        print('cc = ', self.a, self.b)
        # self.canvas.move(self.rectangle, UNIT / 2 - x, UNIT / 2 - y)

        self.canvas.move(self.rectangle, -x, -y)

        self.canvas.move(self.rectangle, self.a, self.b)

        print('사각형 좌표 = ', self.canvas.coords(self.rectangle))
        ### 이전 상태 초기화 ###
        for i in self.r:
            self.canvas.delete(i)
        # for i in range(self.previous):
        #     self.canvas.delete(self.a[i])
        # self.canvas.pack()
        # self.previous = 0


        self.render()
        return self.coords_to_state(self.canvas.coords(self.rectangle))

    def step(self, action):
        state = self.canvas.coords(self.rectangle)
        base_action = np.array([0, 0])
        self.render()


        # ### 이전 상태 색칠 ####
        # self.a[self.previous] = self.canvas.create_image(base_action[0], base_action[1], image=self.shapes[0])
        # self.canvas.pack()
        ### 이전 상태 개수 구함 ###
        self.previous += self.previous

        self.previous_state(state[0], state[1])


        if action == 0:  # 상
            if state[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # 하
            if state[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # 좌
            if state[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 3:  # 우
            if state[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT



        # 에이전트 이동
        self.canvas.move(self.rectangle, base_action[0], base_action[1])




        # 에이전트(빨간 네모)를 가장 상위로 배치
        self.canvas.tag_raise(self.rectangle)
        next_state = self.canvas.coords(self.rectangle)

        # 보상 함수 ####
        # if next_state == self.canvas.coords(self.circle):
        #     reward = 100
        #     done = True
        # elif next_state in [self.canvas.coords(self.triangle1),
        #                     self.canvas.coords(self.triangle2)]:
        #     reward = -100
        #     done = True
        # else:
        #     reward = 0
        #     done = False
        reward = 1 #####
        done = False ####

        next_state = self.coords_to_state(next_state)



        return next_state, reward, done, {}

    def render(self):
        time.sleep(0.03)
        self.update()

    # def previous_state_record(self):
    #     self.shapes2 = self.load_images()
    #     self.rectangle2[previous] = canvas.create_image(x_pre, y_pre, image=self.shapes[0])


class Previous_state_record(Env):
    def __init__(self, x_pre, y_pre):
        self.shapes2 = self.load_images()
        self.rectangle = self.canvas.create_image(x_pre, y_pre, image=self.shapes2[0])

    def reset(self):
        self.canvas.clear()

