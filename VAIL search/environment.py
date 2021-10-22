import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image


q = []
np.random.seed(1)
PhotoImage = ImageTk.PhotoImage
UNIT = 20  # 픽셀 수
HEIGHT = 20  # 그리드 월드 가로
WIDTH = 20  # 그리드 월드 세로
STEP_SIZE = 1
width = 1

noize = 0 * UNIT

obstacle_size = 1

obstacle1_xxyy = [9, 11, 9, 11]
obstacle2_xxyy = [12, 14, 9, 11]
obstacle3_xxyy = [15, 17, 9, 11]
obstacle4_xxyy = [18, 20, 9, 11]

class Env(tk.Tk):

    def __init__(self, b, a):
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
        self.line_graph = []


        # print('self.a = ', self.a)
        # print('self.b = ', self.b)
        ####
        self.previous = 0
        self.r = [] # 이전 상태 사각형 그리기

    def callback(event):
        canvas = event.widget
        x = canvas.canvasx(event.x)
        y = canvas.canvasy(event.y)
        canvas.find_closest(x, y)

    def draw_graph(self):
        for i in range(len(self.line_graph)-1):
            self.canvas.create_line(self.line_graph[i][0], self.line_graph[i][1],self.line_graph[i+1][0],self.line_graph[i+1][1], tag="line{}".format(str(i)), width=4 , fill='blue')



    def delete_graph(self):
        global q
        for i in range(len(self.line_graph)-1):
            self.canvas.delete("line{}".format(str(i)))
        self.line_graph = []
        q = []





    def build_canvas(self, b, a):
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

        print('a, b =', a, b)

        self.oval_obstacle = canvas.create_oval(obstacle1_xxyy[0]*UNIT, obstacle1_xxyy[2]*UNIT, obstacle1_xxyy[1]*UNIT,
                                                obstacle1_xxyy[3]*UNIT, fill='red')
        self.oval_obstacle = canvas.create_oval(obstacle2_xxyy[0]*UNIT, obstacle2_xxyy[2]*UNIT, obstacle2_xxyy[1]*UNIT,
                                                obstacle2_xxyy[3]*UNIT, fill='red')
        self.oval_obstacle = canvas.create_oval(obstacle3_xxyy[0]*UNIT, obstacle3_xxyy[2]*UNIT, obstacle3_xxyy[1]*UNIT,
                                                obstacle3_xxyy[3]*UNIT, fill='red')
        self.oval_obstacle = canvas.create_oval(obstacle4_xxyy[0]*UNIT, obstacle4_xxyy[2]*UNIT, obstacle4_xxyy[1]*UNIT,
                                                obstacle4_xxyy[3]*UNIT, fill='red')

        # self.oval_obstacle = canvas.create_oval(31*UNIT, 46*UNIT, 39*UNIT, 54*UNIT, fill='red')
        #
        # self.oval_obstacle = canvas.create_oval(61*UNIT, 46*UNIT, 69*UNIT, 54*UNIT, fill='red')

        # self.oval_obstacle = canvas.create_oval(50 * UNIT, 26 * UNIT, 60 * UNIT, 36 * UNIT, fill='red')
        # self.oval_obstacle = canvas.create_oval(20 * UNIT, 56 * UNIT, 30 * UNIT, 66 * UNIT, fill='red')
        # self.oval_obstacle = canvas.create_oval(20 * UNIT, 16 * UNIT, 30 * UNIT, 26 * UNIT, fill='red')
        # self.oval_obstacle = canvas.create_oval(35 * UNIT, 36 * UNIT, 45 * UNIT, 46 * UNIT, fill='red')

        # self.obstacle = canvas.create_image(48*UNIT, 49*UNIT, image=self.shapes[1])
        # self.obstacle = canvas.create_image(48*UNIT, 50*UNIT, image=self.shapes[1])
        # self.obstacle = canvas.create_image(48*UNIT, 51*UNIT, image=self.shapes[1])
        # self.obstacle = canvas.create_image(48*UNIT, 52*UNIT, image=self.shapes[1])
        # self.obstacle2 = canvas.create_image(49*UNIT, 48*UNIT, image=self.shapes[1])
        # self.obstacle3 = canvas.create_image(49*UNIT, 49*UNIT, image=self.shapes[1])
        # self.obstacle4 = canvas.create_image(49*UNIT, 50*UNIT, image=self.shapes[1])
        # self.obstacle2 = canvas.create_image(49 * UNIT, 51 * UNIT, image=self.shapes[1])
        # self.obstacle3 = canvas.create_image(49 * UNIT, 52 * UNIT, image=self.shapes[1])
        # self.obstacle2 = canvas.create_image(50*UNIT, 48*UNIT, image=self.shapes[1])
        # self.obstacle3 = canvas.create_image(50*UNIT, 49*UNIT, image=self.shapes[1])
        # self.obstacle4 = canvas.create_image(50*UNIT, 50*UNIT, image=self.shapes[1])
        # self.obstacle2 = canvas.create_image(50 * UNIT, 51 * UNIT, image=self.shapes[1])
        # self.obstacle3 = canvas.create_image(50 * UNIT, 52 * UNIT, image=self.shapes[1])
        # self.obstacle2 = canvas.create_image(51*UNIT, 48*UNIT, image=self.shapes[1])
        # self.obstacle3 = canvas.create_image(51*UNIT, 49*UNIT, image=self.shapes[1])
        # self.obstacle4 = canvas.create_image(51*UNIT, 50*UNIT, image=self.shapes[1])
        # self.obstacle2 = canvas.create_image(51 * UNIT, 51 * UNIT, image=self.shapes[1])
        # self.obstacle3 = canvas.create_image(51 * UNIT, 52 * UNIT, image=self.shapes[1])
        # self.obstacle2 = canvas.create_image(52*UNIT, 48*UNIT, image=self.shapes[1])
        # self.obstacle3 = canvas.create_image(52*UNIT, 49*UNIT, image=self.shapes[1])
        # self.obstacle4 = canvas.create_image(52*UNIT, 50*UNIT, image=self.shapes[1])
        # self.obstacle2 = canvas.create_image(52 * UNIT, 51 * UNIT, image=self.shapes[1])
        # self.obstacle3 = canvas.create_image(52 * UNIT, 52 * UNIT, image=self.shapes[1])



        # ### 파란색 사각형으로 범위 설정
        # for i in range(5, 100-5):
        #     self.obstacle3 = canvas.create_image(5 * UNIT, i * UNIT, image=self.shapes[3])
        # for i in range(5, 100-5):
        #     self.obstacle3 = canvas.create_image(i * UNIT, 95 * UNIT, image=self.shapes[3])
        # for i in range(100-5, 5, -1):
        #     self.obstacle3 = canvas.create_image(95 * UNIT, i * UNIT, image=self.shapes[3])
        # for i in range(100-5, 5, -1):
        #     self.obstacle3 = canvas.create_image(i * UNIT, 5 * UNIT, image=self.shapes[3])
        #
        # for i in range(41, 100-41):
        #     self.obstacle3 = canvas.create_image(41 * UNIT, i * UNIT, image=self.shapes[3])
        # for i in range(41, 100-41):
        #     self.obstacle3 = canvas.create_image(i * UNIT, 59 * UNIT, image=self.shapes[3])
        # for i in range(100-41, 41, -1):
        #     self.obstacle3 = canvas.create_image(59 * UNIT, i * UNIT, image=self.shapes[3])
        # for i in range(100-41, 41, -1):
        #     self.obstacle3 = canvas.create_image(i * UNIT, 41 * UNIT, image=self.shapes[3])




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
        self.line_graph.append([pre_x, pre_y])
        self.canvas.pack()
        return self.r.append(rects)


    def load_images(self):
        rectangle = PhotoImage(
            Image.open("./img/rectangle.png").resize((UNIT, UNIT)))
        obstacle = PhotoImage(
            Image.open("./img/obstacle.png").resize((UNIT*obstacle_size, UNIT*obstacle_size)))
        circle = PhotoImage(
            Image.open("./img/circle.png").resize((65, 65)))
        obstacle_r = PhotoImage(
            Image.open("./img/obstacle2.png").resize((UNIT*obstacle_size, UNIT*obstacle_size)))

        return rectangle, obstacle, circle, obstacle_r

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
        # x = int((coords[0] - 5) / 10)
        # y = int((coords[1] - 5) / 10)
        ##############################
        # x = int((coords[0] - UNIT) / 2*UNIT)
        # y = int((coords[1] - UNIT) / 2*UNIT)
        x = int((coords[0] + UNIT / 2) / UNIT)
        y = int((coords[1] + UNIT / 2) / UNIT)
        # print([x,y])
        return [x, y]

    def reset(self):


        self.update()
        # time.sleep(0.5)
        x, y = self.canvas.coords(self.rectangle) # 시작점을 주면 됨.
        self.canvas.move(self.rectangle, -x, -y)

        self.canvas.move(self.rectangle, self.a , self.b)
        # print('aaaa', self.a, self.b)
        # print('사각형 좌표 = ', self.canvas.coords(self.rectangle))
        ### 이전 상태 초기화 ###
        for i in self.r:
            self.canvas.delete(i)
        # time.sleep(0.5)
        self.delete_graph()
        # for i in range(self.previous):
        #     self.canvas.delete(self.a[i])
        # self.canvas.pack()
        # self.previous = 0


        self.render()
        a = self.coords_to_state(self.canvas.coords(self.rectangle))
        ### num input == 3 일때
        # a.append(np.sqrt(np.square(a[0] - 50) + np.square(a[1] - 50)))
        # d1, d2 append (num_input == 4)

        # def distance(state, obstacle):
        #     return distance(np.sqrt( np.square(state[0]-obstacle[0]) + np.square(state[1]-obstacle[1])))
        # d1 = distance(a,obstacle)
        # d2 = distance(a,obstacle)

        d1 = np.sqrt(np.square(a[0] - (obstacle1_xxyy[0] + obstacle1_xxyy[2])/2) + np.square(a[1] - (obstacle1_xxyy[1] + obstacle1_xxyy[3])/2))
        d2 = np.sqrt(np.square(a[0] - (obstacle2_xxyy[0] + obstacle2_xxyy[2])/2) + np.square(a[1] - (obstacle2_xxyy[1] + obstacle2_xxyy[3])/2))
        d3 = np.sqrt(np.square(a[0] - (obstacle3_xxyy[0] + obstacle3_xxyy[2])/2) + np.square(a[1] - (obstacle3_xxyy[1] + obstacle3_xxyy[3])/2))
        d4 = np.sqrt(np.square(a[0] - (obstacle4_xxyy[0] + obstacle4_xxyy[2])/2) + np.square(a[1] - (obstacle4_xxyy[1] + obstacle4_xxyy[3])/2))
        a.append(d1)
        a.append(d2)
        a.append(d3)
        a.append(d4)
        return a

    def step(self, action):
        state = self.canvas.coords(self.rectangle)
        base_action = np.array([0, 0])
        self.render()

        ### 이전 상태 개수 구함 ###
        self.previous += self.previous

        self.previous_state(state[0], state[1])


        if action == 0:  # 상
            if state[1] > UNIT * STEP_SIZE:
                base_action[1] -= UNIT * STEP_SIZE + noize
        elif action == 1:  # 하
            if state[1] < (HEIGHT - 1) * UNIT * STEP_SIZE:
                base_action[1] += UNIT * STEP_SIZE - noize
        elif action == 2:  # 좌
            if state[0] > UNIT * STEP_SIZE:
                base_action[0] -= UNIT * STEP_SIZE - noize
        elif action == 3:  # 우
            if state[0] < (WIDTH - 1) * UNIT * STEP_SIZE:
                base_action[0] += UNIT * STEP_SIZE - noize
        elif action == 4 : # 좌상
            if state[0] > UNIT * STEP_SIZE and state[1] > UNIT * STEP_SIZE:
                base_action[0] -= UNIT * STEP_SIZE - noize
                base_action[1] -= UNIT * STEP_SIZE - noize
        elif action == 5 : # 우상
            if state[0] < (WIDTH - 1) * STEP_SIZE * UNIT and state[1] > UNIT * STEP_SIZE:
                base_action[0] += UNIT * STEP_SIZE - noize
                base_action[1] -= UNIT * STEP_SIZE - noize
        elif action == 6 : # 우하
            if state[0] < (WIDTH - 1) * UNIT * STEP_SIZE and state[1] < (HEIGHT - 1) * UNIT * STEP_SIZE:
                base_action[0] += UNIT * STEP_SIZE - noize
                base_action[1] += UNIT * STEP_SIZE - noize
        elif action == 7 : # 좌하
            if state[0] > UNIT * STEP_SIZE and state[1] < (HEIGHT - 1) * UNIT * STEP_SIZE:
                base_action[0] -= UNIT * STEP_SIZE - noize
                base_action[1] += UNIT * STEP_SIZE - noize


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

        ## state 수정하는 곳 appned
        d1 = np.sqrt(np.square(next_state[0] - (obstacle1_xxyy[0] + obstacle1_xxyy[2])/2) +
                     np.square(next_state[1] - (obstacle1_xxyy[1] + obstacle1_xxyy[3])/2))

        d2 = np.sqrt(np.square(next_state[0] - (obstacle2_xxyy[0] + obstacle2_xxyy[2])/2) +
                     np.square(next_state[1] - (obstacle2_xxyy[1] + obstacle2_xxyy[3])/2))

        d3 = np.sqrt(np.square(next_state[0] - (obstacle3_xxyy[0] + obstacle3_xxyy[2])/2) +
                     np.square(next_state[1] - (obstacle3_xxyy[1] + obstacle3_xxyy[3])/2))

        d4 = np.sqrt(np.square(next_state[0] - (obstacle4_xxyy[0] + obstacle4_xxyy[2])/2) +
                     np.square(next_state[1] - (obstacle4_xxyy[1] + obstacle4_xxyy[3])/2))

        next_state.append(d1)
        next_state.append(d2)
        next_state.append(d3)
        next_state.append(d4)
        # next_state.append(np.sqrt(np.square(state[0]-50) + np.square(state[1]-50)))
        return next_state, reward, done, {}

    def render(self):
        # time.sleep(0.03)
        # time.sleep(20)
        self.update()

## 이전 상태들을 그림
class Previous_state_record(Env):
    def __init__(self, x_pre, y_pre):
        self.shapes2 = self.load_images()
        self.rectangle = self.canvas.create_image(x_pre, y_pre, image=self.shapes2[0])

    # 이전 상태 이미지 리셋
    def reset(self):
        self.canvas.clear()
