'''此代码为集成控制键鼠的操作'''
'''pip install pydirectinput'''
import pydirectinput
import win32api as wapi
import time
keyList = []
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys

'''
暂停程序的代码:
使用示例：
paused = False
while True:
    paused = pause_game(paused)
'''
def pause_game(paused):
    pressed_keys = key_check()
    if 'T' in pressed_keys:
        if paused:
            paused = False
            print('start game')
            time.sleep(1)
        else:
            paused = True
            print('pause game')
            time.sleep(1)
    if paused:
        print('paused')
        while True:
            keys = key_check()
            # pauses game and can get annoying.
            if 'T' in keys:
                if paused:
                    paused = False
                    print('start game')
                    time.sleep(1)
                    break
                else:
                    paused = True
                    time.sleep(1)
    return paused

''' 键鼠操作示意'''

'''
# 按下并松开 'a' 键
pydirectinput.press('w')

#按住 'a' 键
pydirectinput.keyDown('a')
time.sleep(1)  # 等待 1 秒
# 松开 'a' 键
pydirectinput.keyUp('a')

# 输入字符串 "Hello, world!"
pydirectinput.typewrite('Hello, world!')

# 左键点击
pydirectinput.click()

# 右键点击
pydirectinput.click(button='right')

# 中键点击
pydirectinput.click(button='middle')

# 移动鼠标到屏幕坐标 (100, 200)
pydirectinput.moveTo(1000, 200)

# 相对当前位置向右移动 50 像素，向下移动 50 像素
pydirectinput.move(50, 50)

# 按住左键从 (100, 200) 拖动到 (300, 400)
pydirectinput.moveTo(100, 200)
pydirectinput.mouseDown()
pydirectinput.moveTo(300, 400)
pydirectinput.mouseUp()
'''

'''截图操作示意'''
'''
window_size = (160, 170, 640, 990) # flappybird 窗口
score_size = (370,240,430,310)
bird_size = (205,187,370,900)
'''
import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api

def grab_screen(region=None):
    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
    
    # 使用 np.frombuffer 而不是 np.fromstring
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)  # 注意确保高度和宽度顺序正确

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return img

'''
得分值是否变化，变化则给奖励
white_pixel: 20 显示为0 是开始值
white_pixel: 0  显示为空白 为结束值
white_pixel: 61 得重新进入游戏
'''
def get_white_pixel(img):
    white_pixel_sum = 0
    for white_pixel in img[19]:
        if white_pixel >= 240:
            white_pixel_sum +=1
    return white_pixel_sum

class ScreenCapturer:
    def __init__(self):
        self.hwin = win32gui.GetDesktopWindow()
        self.hwindc = win32gui.GetWindowDC(self.hwin)
        self.srcdc = win32ui.CreateDCFromHandle(self.hwindc)
        self.memdc = self.srcdc.CreateCompatibleDC()
        self.bmp = None
        self.current_size = (0, 0)

    def grab_screen(self, region=None):
        if region:
            left, top, right, bottom = region
            width = right - left + 1
            height = bottom - top + 1
        else:
            width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
            height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
            left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
            top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

        # 仅在尺寸变化时重新创建位图
        if (width, height) != self.current_size:
            self.bmp = win32ui.CreateBitmap()
            self.bmp.CreateCompatibleBitmap(self.srcdc, width, height)
            self.current_size = (width, height)

        self.memdc.SelectObject(self.bmp)
        self.memdc.BitBlt((0, 0), (width, height), self.srcdc, (left, top), win32con.SRCCOPY)
        
        # 获取位图数据
        signed_ints = self.bmp.GetBitmapBits(True)
        img = np.frombuffer(signed_ints, dtype=np.uint8)
        img.shape = (height, width, 4)  # BGRA格式
        
        return img

    def __del__(self):
        if self.bmp:
            win32gui.DeleteObject(self.bmp.GetHandle())
        self.memdc.DeleteDC()
        self.srcdc.DeleteDC()
        win32gui.ReleaseDC(self.hwin, self.hwindc)

# 初始化截图对象
screen_capturer = ScreenCapturer()
        
    
