'''此为实现flappybird的环境'''
import cv2
import numpy as np
import pydirectinput
import time
import threading
from io_utils import grab_screen, get_white_pixel,key_check, pause_game ,ScreenCapturer
#from Buffer import Buffer_for_PPO , Buffer_episode_for_PPO

## 首先从preparation.py得到关键参数
scaling_size = 1.25 # 系统->屏幕->缩放

'''
这6个值比较重要
home_window,zero_window,start_window
start_pixel,terminal_pixel,current_white_pixel
'''
home_window = (0, 50, 311, 503)
single_game_window = (81, 291, 229, 342)
zero_window =  (103, 101, 193, 129)
start_window =  (147, 337, 160, 359)
bird_xy_window =  (56, 56, 100, 606)
restart_window = (41, 359, 135, 401)
terminal_window = (52, 74, 79, 96)
best_scale_home = 0.763157894736842
best_scale_single_game = 0.763157894736842
best_scale_zero = 0.763157894736842
best_scale_start = 0.8157894736842105
best_scale_restart = 0.763157894736842
best_scale_terminal = 0.8026315789473684
start_pixel = 255
terminal_pixel = 58
current_white_pixel = 6


class Env_flappybird:
    '''
    https://u.ali213.net/games/flapybird/index.html?game_code=179 # 无网下也能使用
    改为微信小程序：像素小鸟pro
    '''
    def __init__(self):
        self.stack_frame = False # 默认false
        #self.stack_num = 4
        self.discrete_env = True
        self.action_dim = 2
        

        ## 关键参数
        self.home_window = home_window
        self.single_game_window = single_game_window
        self.zero_window = zero_window
        self.start_window = start_window
        self.restart_window = restart_window
        self.terminal_window = terminal_window
        self.bird_xy_window = bird_xy_window

        ## 点击位置
        self.restart_window_xy = (int((restart_window[0]+restart_window[2])/2/scaling_size),int((restart_window[1]+restart_window[3])/2/scaling_size))
        self.click_xy = (int((home_window[0]+home_window[2])/2/scaling_size),int(home_window[3]/scaling_size + 20))

        ##  相对窗口（start,terminal 相对于home_windpw_x1y1）
        self.start_window_r = (start_window[0]-home_window[0],start_window[1]-home_window[1],start_window[2]-home_window[0],start_window[3]-home_window[1])
        #self.terminal_window_r = (terminal_window[0]-home_window[0],terminal_window[1]-home_window[1],terminal_window[2]-home_window[0],terminal_window[3]-home_window[1])
        self.zero_window_r = (zero_window[0]-home_window[0],zero_window[1]-home_window[1],zero_window[2]-home_window[0],zero_window[3]-home_window[1])
        self.bird_xy_window_r = (bird_xy_window[0]-home_window[0],bird_xy_window[1]-home_window[1],bird_xy_window[2]-home_window[0],bird_xy_window[3]-home_window[1])
        
        # wh：列行
        self.start_window_wh = (self.start_window_r[2]-self.start_window_r[0],self.start_window_r[3]-self.start_window_r[1])
        # start判断位置：2/3位置
        self.start_window_judge = (self.start_window_wh[1]*2//3,self.start_window_wh[0]*2//3)

        
        self.start_pixel = start_pixel
        self.terminal_pixel = terminal_pixel


        self.bird_width = self.home_window[2] - self.home_window[0]
        self.bird_height = self.home_window[3] - self.home_window[1]
        self.resize_ratio = 4 
        self.bird_resize_w = int(self.bird_width//self.resize_ratio) 
        self.bird_resize_h = int(self.bird_height//self.resize_ratio) 

        self.obs_dim = (1,self.bird_resize_h,self.bird_resize_w)
        # 读取背景图
        self.background = cv2.imread('background.png',0)
        #print(self.bird_resize_h)
        self.background = cv2.resize(self.background,(self.bird_resize_w,self.bird_resize_h))

        '''
        self.bird_x = (self.bird_xy_window_r[0]+self.bird_xy_window_r[2])//2//self.resize_ratio
        #print('bird_x:',self.bird_x) # 验证小鸟x位置识别正确
        self.cnt = 0
        '''

        

        
        
    def reset(self):
        self.mark = 0
        self.current_white_pixel = current_white_pixel 
        self.last_action = 1
        
        reset = False
        while not reset:
            screen_gray = cv2.cvtColor(grab_screen(self.home_window),cv2.COLOR_BGR2GRAY)
            #开始窗口
            start  = screen_gray[self.start_window_r[1]:self.start_window_r[3],self.start_window_r[0]:self.start_window_r[2]]

            if start[self.start_window_judge] == self.start_pixel:
                reset = True
                #print('game get ready')
            else:
                pydirectinput.click(self.restart_window_xy[0],self.restart_window_xy[1])
                time.sleep(1)

        
        pydirectinput.moveTo(self.click_xy[0],self.click_xy[1])
        time.sleep(0.2)
        
        return None 
    
    def first_step(self):
        pydirectinput.click()
        time.sleep(0.1)
        screen_gray = cv2.cvtColor(grab_screen(self.home_window),cv2.COLOR_BGRA2GRAY)
        self.resize_screen_0 = cv2.resize(screen_gray,(self.bird_resize_w,self.bird_resize_h))

        obs_screen = self.resize_screen_0 - self.background #self.resize_screen_0
        
        return obs_screen


    def step(self,action):
        #reward = 0
        ''' 先进行动作'''
        if action == 0:
            time.sleep(0.2)
            
        elif action ==1:
            pydirectinput.click() # 相当于0.1s
            time.sleep(0.1)

        ''' 收集图像'''
        # 保存截图

        screen_gray = cv2.cvtColor(grab_screen(self.home_window),cv2.COLOR_BGR2GRAY)

        #开始窗口
        start  = screen_gray[self.start_window_r[1]:self.start_window_r[3],self.start_window_r[0]:self.start_window_r[2]]
        zero = screen_gray[self.zero_window_r[1]:self.zero_window_r[3],self.zero_window_r[0]:self.zero_window_r[2]]
        self.resize_screen_1 = cv2.resize(screen_gray,(self.bird_resize_w,self.bird_resize_h))
        

        reward = 0
        w_p = get_white_pixel(zero)
        if w_p != self.current_white_pixel:
            reward = 1
            self.current_white_pixel = w_p
            self.mark += 1 # 分数+1
        

        terminated = start[self.start_window_judge] == self.terminal_pixel or self.mark > 20

        
        next_obs = self.resize_screen_1 - self.background 

        return next_obs,reward,terminated,False, None
    
    def random_action(self):
        return np.random.randint(2) #,np.random.randint(10)
      

if __name__ == '__main__':
    env = Env_flappybird()  # 
    np.random.seed(0)
    print(env.obs_dim)
