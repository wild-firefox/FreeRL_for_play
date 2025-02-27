# 在主页面像素小鸟 得到关键参数 以使用不同显示器的电脑
import cv2
import numpy as np
from io_utils import grab_screen, get_white_pixel
import pydirectinput
import time

scaling_size = 1.25 # 系统->屏幕->缩放

def multi_scale_template_matching(screen_gray, template, scales):
    best_match = None
    best_val = -1
    best_loc = None
    best_scale = 1.0

    for scale in scales:
        resized_template = cv2.resize(template, (int(template.shape[1] * scale), int(template.shape[0] * scale)))
        res = cv2.matchTemplate(screen_gray, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val > best_val:
            best_val = max_val
            best_match = resized_template
            best_loc = max_loc
            best_scale = scale

    return best_match, best_loc, best_scale

# 计算匹配区域
def calculate_window(template, loc):
    top_left = loc
    bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
    return top_left + bottom_right

# 用矩阵的方式显示图片
def show_image(screen, window_1, window_2=None , window_3=None):
    screen_copy = np.copy(screen)

    cv2.rectangle(screen_copy, window_1[:2], window_1[2:], (0, 0, 255), 2)
    if window_2 is not None:
        cv2.rectangle(screen_copy, window_2[:2], window_2[2:], (0, 0, 255), 2)
    if window_3 is not None:
        cv2.rectangle(screen_copy, window_3[:2], window_3[2:], (0, 0, 255), 2)

    # 显示结果
    cv2.imshow('screen', screen_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 获取当前屏幕截图
screen_0 = grab_screen()
screen_gray_0 = cv2.cvtColor(screen_0, cv2.COLOR_BGR2GRAY)

# 读取home模板
home = cv2.imread('./image/home.png', 0)
single_game = cv2.imread('./image/single_game.png', 0)
zero = cv2.imread('./image/zero.png', 0)
start = cv2.imread('./image/start.png', 0)
restart = cv2.imread('./image/restart.png', 0)
terminal = cv2.imread('./image/terminal.png', 0)
bird_xy = cv2.imread('./image/bird_xy.png', 0)

# ## 缩小模板 # 随机1-2的小数 模拟在不同显示器上的情况  # 缩小模板 模板图像尺寸必须小于屏幕图像尺寸
n = 2#np.random.uniform(1, 2)
home = cv2.resize(home,(int(home.shape[1]/n),int(home.shape[0]/n)) )
single_game = cv2.resize(single_game,(single_game.shape[1]//n,single_game.shape[0]//n))
zero = cv2.resize(zero,(int(zero.shape[1]/n),int(zero.shape[0]/n)) )
start = cv2.resize(start,(start.shape[1],start.shape[0]))
restart = cv2.resize(restart,(restart.shape[1]//n,restart.shape[0]//n))
terminal = cv2.resize(terminal,(terminal.shape[1]//n,terminal.shape[0]//n))
bird_xy = cv2.resize(bird_xy,(bird_xy.shape[1]//n,bird_xy.shape[0]//n))

# 定义缩放比例范围    
scales = np.linspace(0.5, 2, 20)  # 从0.5倍到2倍，分20个级别

# 多尺度模板匹配
best_home, best_loc_home, best_scale_home = multi_scale_template_matching(screen_gray_0, home, scales)
best_single_game, best_loc_single_game, best_scale_single_game = multi_scale_template_matching(screen_gray_0, single_game, scales)

# 计算匹配区域
home_window = calculate_window(best_home, best_loc_home)
print('home_window =', home_window)

single_game_window = calculate_window(best_single_game, best_loc_single_game)
print('single_game_window =', single_game_window)
#show_image(screen_0, home_window, single_game_window)


## 点击一下中心single_game

pydirectinput.click(int((single_game_window[0]+single_game_window[2])/2/scaling_size),int((single_game_window[1]+single_game_window[3])/2/scaling_size))
time.sleep(1)

## 截图
screen_1 = grab_screen()
screen_gray_1 = cv2.cvtColor(screen_1, cv2.COLOR_BGR2GRAY)

# 多尺度模板匹配
best_zero, best_loc_zero, best_scale_zero = multi_scale_template_matching(screen_gray_1, zero, scales)
beast_start, best_loc_start, best_scale_start = multi_scale_template_matching(screen_gray_1, start, scales)
best_bird_xy, best_loc_bird_xy, best_scale_bird_xy = multi_scale_template_matching(screen_gray_1, bird_xy, scales)

# 计算匹配区域
zero_window = calculate_window(best_zero, best_loc_zero)
print('zero_window = ', zero_window)


start_window = calculate_window(beast_start, best_loc_start)
print('start_window = ', start_window)

bird_xy_window = calculate_window(best_bird_xy, best_loc_bird_xy)
print('bird_xy_window = ', bird_xy_window)

## 像素点颜色
start_pixel = screen_gray_1[start_window[1]+int(start_window[3]-start_window[1])*2//3,start_window[0]+int(start_window[2]-start_window[0])*2//3]
zero_zone = screen_gray_1[zero_window[1]:zero_window[3],zero_window[0]:zero_window[2]]
current_white_pixel = get_white_pixel(zero_zone)

# 点击
pydirectinput.click()
time.sleep(0.1)

## 存储背景图 将小鸟的位置填充为背景色
screen_gray = cv2.cvtColor(grab_screen(home_window),cv2.COLOR_BGR2GRAY)
bird_xy_window_r = (bird_xy_window[0]-home_window[0],bird_xy_window[1]-home_window[1],bird_xy_window[2]-home_window[0],bird_xy_window[3]-home_window[1])
screen_gray_bird_xy = screen_gray[bird_xy_window_r[1]:bird_xy_window_r[3],bird_xy_window_r[0]:bird_xy_window_r[2]] 
screen_gray_bird_xy[screen_gray.shape[0]//3:screen_gray.shape[0]//3*2] = screen_gray_bird_xy[10][0] # 此为背景色

zero_window_r = (zero_window[0]-home_window[0],zero_window[1]-home_window[1],zero_window[2]-home_window[0],zero_window[3]-home_window[1])
screen_gray_zero_window = screen_gray[zero_window_r[1]:zero_window_r[3],zero_window_r[0]:zero_window_r[2]]
screen_gray_zero_window[:] = screen_gray_bird_xy[10][0] # 此为背景色  ##增加 分数0为涂成背景

## 将
cv2.imwrite('background.png',screen_gray)

#self.resize_screen_0 = cv2.resize(screen_gray,(self.bird_resize_w,self.bird_resize_h))
time.sleep(3)

## 截图
screen_2 = grab_screen()
screen_gray_2 = cv2.cvtColor(screen_2, cv2.COLOR_BGR2GRAY)

best_restart, best_loc_restart, best_scale_restart = multi_scale_template_matching(screen_gray_2, restart, scales)
restart_window = calculate_window(best_restart, best_loc_restart)
print('restart_window =', restart_window)

best_terminal, best_loc_terminal, best_scale_terminal = multi_scale_template_matching(screen_gray_2, terminal, scales)
terminal_window = calculate_window(best_terminal, best_loc_terminal)
print('terminal_window =', terminal_window)

# current_white_pixel_black = get_white_pixel(zero_zone)
# print('current_white_pixel_black =', current_white_pixel_black)

## 像素点颜色
terminal_pixel = screen_gray_2[start_window[1]+int(start_window[3]-start_window[1])*2//3,start_window[0]+int(start_window[2]-start_window[0])*2//3]

# 点击再来一局
pydirectinput.click(int((restart_window[0]+restart_window[2])/2/scaling_size),int((restart_window[1]+restart_window[3])/2/scaling_size))

# 显示结果
show_image(screen_0, home_window, single_game_window)
show_image(screen_1, zero_window, start_window,bird_xy_window)
show_image(screen_2, restart_window, terminal_window)

# 显示最佳缩放scale 相对于原来的图像
print('best_scale_home =', best_scale_home/n)
print('best_scale_single_game =', best_scale_single_game/n)
print('best_scale_zero =', best_scale_zero/n)
print('best_scale_start =', best_scale_start)
print('best_scale_restart =', best_scale_restart/n)
print('best_scale_terminal =', best_scale_terminal/n)

print('start_pixel =', start_pixel)
print('terminal_pixel =', terminal_pixel)

print('current_white_pixel =', current_white_pixel)




