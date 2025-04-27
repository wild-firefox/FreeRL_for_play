import numpy as np
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
import _pickle as pickle

import imageio
import os

## 此文件用来评估PPO_atari.py的结果
from PPO_atari import PPO

script_dir = os.path.dirname(os.path.abspath(__file__)) # 当前脚本文件夹
model_dir = os.path.join(script_dir,'./results/Pong-v0', 'PPO_8')

D = 75*80
dim_info = [D,2]
is_continue = False

policy = PPO.load(dim_info, is_continue,model_dir=model_dir)

# def sigmoid(x): 
#   return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

# def policy_forward(x):
#   h = np.dot(model['W1'], x)
#   h[h<0] = 0 # ReLU nonlinearity
#   logp = np.dot(model['W2'], h)
#   p = sigmoid(logp)
#   return p, h # return probability of taking action 2, and hidden state

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:185] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float32).ravel() # 展平


env = gym.make("Pong-v0", render_mode="rgb_array") # "rgb_array"时可以保存gif, "human"时可以显示窗口
obs, info = env.reset() ## 25.4.26 ：发现之前PPO收敛快一部分原因是因为reset的seed固定了
#print(observation)
done = False

episode_first_step = True
first_action = 0 
prev_x = None
frames = []

episode_reward = 0
step = 0

while not done:
    step += 1
    if episode_first_step  == True:
      obs  = env.step(first_action)[0] 
      cur_x = prepro(obs)
      obs = np.zeros(D) #cur_x - prev_x if prev_x is not None else np.zeros(D)
      prev_x = cur_x
      episode_first_step = False

    if step % 4 == 0:
      frame = env.render()
      frames.append(frame)

    action  = policy.evaluate_action(obs)   # 这里离散
    action_ = 2 if action == 0 else 3

    # x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    # prev_x = cur_x
    #aprob, h = policy_forward(x)   
    #action = 2 if aprob >= 0.5 else 3 # 使用 aprob 的值来决定动作
    #action = env.action_space.sample()  # 随机选择动作
    next_obs, reward, terminated, truncated, info = env.step(action_)
    cur_x = prepro(next_obs)
    next_obs = (cur_x - prev_x)
    obs = next_obs
    episode_reward += reward
    if reward != 0:
        print(f"Reward: {reward} ,{terminated} ,  {truncated} ")  # 仅在奖励非零时输出
    if terminated or truncated:
        obs, info = env.reset()
        done = True
        print("Reset")
env.close()
print(f"Episode reward: {episode_reward}")
imageio.mimsave(os.path.join(model_dir,"evaluate.gif"),frames,duration=1/60)