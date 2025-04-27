from pettingzoo.atari import pong_v3
import time

from IPPO import IPPO,prepro
import os
import numpy as np
import time

# 在文件顶部添加必要的导入
import imageio # pip install imageio
import numpy as np
from datetime import datetime

# 在while循环前添加
frames = []  # 用于存储动画帧
save_freq = 4  # 每隔多少步保存一帧，可以调整以减小文件大小

# mode='rgb_array'时为保存gif mode='human'时为查看游戏
env = pong_v3.parallel_env(render_mode="rgb_array",max_cycles=10000)

observations, infos = env.reset()

# 模型文件夹 - 读取
env_name = 'pong_v3'
folder_name = 'IPPO_22'#'IPPO_18'
script_dir = os.path.dirname(os.path.abspath(__file__)) # 当前脚本文件夹
results_dir =   os.path.join(script_dir,'./results')
model_dir = os.path.join(results_dir,env_name,folder_name) 
print(f'model_dir: {model_dir}')

dim_info = {agent_id: [6000,2] for agent_id in env.agents}
max_action = None
is_continue = False
trick={'adv_norm':True,
        'ObsNorm':False,
        'reward_norm':True,'reward_scaling':False,    # or
        'orthogonal_init':True,'adam_eps':True,'lr_decay':True, # 原代码中设置为False
        # 以上均在PPO_with_tricks.py中实现过
        'ValueClip':True,'huber_loss':True,
        'LayerNorm':True,'feature_norm':True,
                ## 增加
        'add_agent_id':True, 
        }

policy = IPPO.load(dim_info,is_continue = is_continue ,model_dir=model_dir,trick = trick,)

episode_first_step  =True
prev_x = None 
first_action = {agent_id: int(2) for agent_id in env.agents}
D = 75*80
env_agents = [agent_id for agent_id in env.agents]
episode_reward = {agent_id: 0 for agent_id in env_agents}
episode_score = {agent_id: 0 for agent_id in env_agents}
score_200 = True
action_env = {agent_id: 0 for agent_id in env_agents} ## 默认不动

done = {agent_id: False for agent_id in env.agents}
print( any(done.values()))
step = 0
max_episodes = 2 #测试两次
episode_num = 0
episode_step = 0
t = False
#while env.agents:
while episode_num < max_episodes:
    if episode_first_step  == True:
        obs  = env.step(first_action)[0] 
        cur_x = {agent_id: prepro(obs[agent_id]) for agent_id in env_agents}
        obs = {agent_id: np.zeros(D) for agent_id in env_agents}
        prev_x = cur_x
        episode_first_step = False
    # this is where you would insert your policy
    #actions = {agent: 4 for agent in env.agents}
    step += 1
    episode_step += 1
    actions  , _ = policy.select_action(obs)
    
    #action_  = { agent_id: int(actions[agent_id]) for agent_id in env_agents}
    action_ = { agent_id: 2 if actions[agent_id] == 0 else 3 for agent_id in env_agents} # 离散动作空间
    #action_ = {agent_id: 0 for agent_id in env_agents}
    #print(step, action_)
    #action_ = {agent_id: actions[i]+1 for i,agent_id in enumerate(env_agents)}
    #print(action_)
    #time.sleep(0.5) 
    #print(env.agents)
    #time.sleep(0.1)

    if step % 200 == 0 :
        if score_200:
            score_200 = False
            last_score = episode_score
        else: 
            # 若是分数不变化，就让score分高的一方发球 (环境没问题的话会自己发球:让赢球的一方发球)
            if last_score[env_agents[0]] == episode_score[env_agents[0]] and last_score[env_agents[1]] == episode_score[env_agents[1]]:
                # 右边是first,左边是second
                if episode_score[env_agents[0]] > episode_score[env_agents[1]]: # 右边分数高，向左发球
                    action_env = {agent_id: 1 for agent_id in env_agents}
                else:
                    action_env = {agent_id: 1 for agent_id in env_agents}
            #action_env = {agent_id: 5 for agent_id in env_agents}  4 是两个都往上走，发球 发球逻辑不变 5是都往下
                env.step(action_env) ## 发球得分不记录
                # if t == True:
                #     print('t1:',t)
                #     time.sleep(1)
                step += 1
                episode_step += 1
            last_score = episode_score
            

    next_obs, reward,terminated, truncated, infos = env.step(action_) 
    if step % save_freq == 0:
        # 获取当前渲染画面
        frame = env.render()
        # 对于某些环境，您可能需要使用这种方式获取渲染画面
        #frame = env.render(mode='rgb_array')
        if frame is not None:
            frames.append(frame)
    # if t == True:
    #     print('t1:',t)
    #     time.sleep(1)
    score = {agent_id: (0  if reward[agent_id] < 0 else reward[agent_id] ) for agent_id in env_agents }
    # if reward[env_agents[0]] != 0 or reward[env_agents[1]] != 0:
    #     print(step ,'reward:',reward) 
    #     time.sleep(1)
    #     t = True
    # else:
    #     t = False
        
    done = {agent_id: terminated[agent_id] or truncated[agent_id] for agent_id in env_agents}
    cur_x = {agent_id: prepro(next_obs[agent_id]) for agent_id in env_agents}
    next_obs = {agent_id: cur_x[agent_id] - prev_x[agent_id]  for agent_id in env_agents}
    obs = next_obs

    episode_reward = {agent_id: episode_reward[agent_id] + reward[agent_id] for agent_id in env_agents}
    episode_score = {agent_id: episode_score[agent_id] + score[agent_id] for agent_id in env_agents}
    #print(env.agents)
    if any(done.values()):
        print(terminated,truncated)
        print(episode_step,episode_score)
        obs,info = env.reset()
        episode_step = 0
        episode_num += 1
        episode_first_step = True
        episode_reward = {agent_id: 0 for agent_id in env_agents}
        episode_score = {agent_id: 0 for agent_id in env_agents}

        if frames:
            print("正在保存动画...")
            # 创建带有时间戳的文件名以避免覆盖
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"/pong_animation_{timestamp}.gif"
            # 保存为GIF
            imageio.mimsave(model_dir+filename, frames, fps=40)
            print(f"动画已保存为: {model_dir+filename}")
            frames.clear()  # 清空帧列表以便下次使用
        
        #break
env.close()
print(f'step: {step}')