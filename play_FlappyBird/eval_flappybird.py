import numpy as np
from env_flappybird import Env_flappybird
from io_utils import pause_game
import matplotlib.pyplot as plt
import os
from PPO_atari import PPO


def prepro(image_in):
    image=image_in.copy()
    image = image[2:112,0:76]  # crop
    image = image[::2,::2] # downsample by factor of 2
    image[image != 0] = 1  # 转为灰度图，除了黑色外其他都是白色
    return image.astype(np.float32).ravel()



# 读取模型
D = 55*38 #80 * 80
dim_info = [D,2]
is_continue = False
script_dir = os.path.dirname(os.path.abspath(__file__)) # 当前脚本文件夹
model_dir = script_dir + '\\results\\flappybird\\PPO_9'
policy = PPO.load(dim_info = dim_info , is_continue=is_continue ,model_dir=model_dir)



env = Env_flappybird()
paused = True
paused = pause_game(paused)
env.reset()
prev_x = None # used in computing the difference frame
episode_first_step = True



episode_rewards = []

for i in range(100): # 评估10次
    episode_first_step = True
    env.reset()
    episode_reward = 0 
    done = False
    while not done:
        paused = pause_game(paused)
        if episode_first_step  == True:
            obs  = env.first_step()
            cur_x = prepro(obs)
            obs = np.zeros(D) 
            prev_x = cur_x
            episode_first_step = False
    
        action = policy.evaluate_action(obs)
        #action = env.random_action()
        #print(action)
        next_obs, reward,terminated, truncated , info = env.step(action) 
        episode_reward += reward
        done = terminated or truncated
        cur_x = prepro(next_obs)
        next_obs = (cur_x - prev_x)#.reshape(1,dim_info[0][1] ,dim_info[0][2])
        obs = next_obs
    
    print(f"{i} episode_reward:",episode_reward)
    episode_rewards.append(episode_reward)

print('mean_reward:' ,np.mean(episode_rewards))

