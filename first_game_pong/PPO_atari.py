import os
# 设置OMP_WAIT_POLICY为PASSIVE，让等待的线程不消耗CPU资源 #确保在pytorch前设置
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE' #确保在pytorch前设置

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal,Categorical

from copy import deepcopy
import numpy as np
from Buffer import Buffer_for_PPO , Buffer_atari

import gymnasium as gym
import argparse

## 其他
import re
import time
from torch.utils.tensorboard import SummaryWriter

import ale_py
gym.register_envs(ale_py)


# from env_flappybird_c_single import Env_flappybird
# from io_utils import pause_game
'''
1: 
2: 改tanh
3:2 上改actor_lr 1e-3
4: 1上改actor_lr 1e-3
reward_norm_1：4 加上reward_norm
5: 4加上prev_x = cur_x
6：4将reward 都改成1 看是否收敛
7: 4 hidden改成400


'''

## 第一部分：定义Agent类

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=128, hidden_2=128):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.mean_layer = nn.Linear(hidden_2, action_dim)        # 此方法称为对角高斯函数 的主流方法1.对于每个action维度都有独立的方差 第二种方法 2.self.log_std_layer = nn.Linear(hidden_2, action_dim) log_std 是环境状态的函数
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))  # 方法参考 1.https://github.com/zhangchuheng123/Reinforcement-Implementation/blob/master/code/ppo.py#L134C29-L134C41
                                                                      # 2.    https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/main/5.PPO-continuous/ppo_continuous.py#L56
                                                                      # 3.    https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/core.py#L85
    def forward(self, obs ):
        x = F.relu(self.l1(obs))
        x = F.relu(self.l2(x))
        mean = self.mean_layer(x)
        mean = torch.tanh(self.mean_layer(x))  # 使得mean在-1,1之间

        log_std = self.log_std.expand_as(mean)  # 使得log_std与mean维度相同 输出log_std以确保std=exp(log_std)>0
        log_std = torch.clamp(log_std, -20, 2) # exp(-20) - exp(2) 等于 2e-9 - 7.4，确保std在合理范围内
        std = torch.exp(log_std)

        return mean, std

class Actor_discrete(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=400, hidden_2=400):
        super(Actor_discrete, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden_1)
        # self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, action_dim)

    def forward(self, obs ):
        x = F.relu(self.l1(obs)) #
        # x = F.relu(self.l2(x))
        a_prob = torch.softmax(self.l3(x), dim=1)
        return a_prob
'''
对图像的动作空间的处理
'''
class Actor_image(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=512, hidden_2=256,hidden_3=128):
        super(Actor_image, self).__init__()
        '''例
        obs_dim: (1, 180, 80) # (c, h, w)
        '''
        self.obs_w = obs_dim[2]
        self.obs_h = obs_dim[1]
        self.obs_c = obs_dim[0]



        self.conv_w = self.obs_w // 4 # 4是因为两次池化
        self.conv_h = self.obs_h // 4

        self.conv1 = nn.Conv2d(self.obs_c , 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.l1 = nn.Linear(32*self.conv_w*self.conv_h, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, hidden_3)
        self.mean_layer = nn.Linear(hidden_3, action_dim)        # 此方法称为对角高斯函数 的主流方法1.对于每个action维度都有独立的方差 第二种方法 2.self.log_std_layer = nn.Linear(hidden_2, action_dim) log_std 是环境状态的函数
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, obs ):
        x = F.relu(self.conv1(obs)) 
        x = F.max_pool2d(x, kernel_size=2, stride=2) 
        x = F.relu(self.conv2(x)) 
        x = F.max_pool2d(x, kernel_size=2, stride=2) 
        x = x.reshape(x.size(0), -1) 
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        mean = torch.tanh(self.mean_layer(x))  # 使得mean在-1,1之间

        log_std = self.log_std.expand_as(mean)  # 使得log_std与mean维度相同 输出log_std以确保std=exp(log_std)>0
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        return mean, std

class Actor_discrete_image(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=512, hidden_2=256,hidden_3=128):
        super(Actor_discrete_image, self).__init__()
        '''例
        obs_dim: (1, 180, 80) # (c, h, w)
        '''
        self.obs_w = obs_dim[2]
        self.obs_h = obs_dim[1]
        self.obs_c = obs_dim[0]


        self.conv_w = self.obs_w // 4 # 4是因为两次池化
        self.conv_h = self.obs_h // 4

        self.conv1 = nn.Conv2d(self.obs_c, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.l1 = nn.Linear(32*self.conv_w*self.conv_h, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2,hidden_3)
        self.l4 = nn.Linear(hidden_3, action_dim)

    def forward(self, obs ):
        x = F.relu(self.conv1(obs))  # 1x180x80 -> 16x180x80
        x = F.max_pool2d(x, kernel_size=2, stride=2) # 16x180x80 -> 16x90x40
        x = F.relu(self.conv2(x))  # 16x90x40 -> 32x90x40
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # 32x90x40 -> 32x45x20
        x = x.reshape(x.size(0), -1)  # 32x45x20 -> 1x28800 # shape(batch_size, -1)
        x = F.relu(self.l1(x)) # 
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        a_prob = torch.softmax(self.l4(x), dim=1)
        return a_prob
    
'''   
critic部分 与sac区别
区别:sac中critic输出Q1,Q2,而ppo中只输出V
'''    
class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_1=400, hidden_2=400):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden_1)
        #self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, 1)

    def forward(self, obs):       
        x = F.relu(self.l1(obs))
        #x = F.relu(self.l2(x))
        value = self.l3(x)
        return value

class Critic_image(nn.Module):
    def __init__(self, obs_dim, hidden_1=512, hidden_2=256, hidden_3 = 128):
        super(Critic_image, self).__init__()
        '''例
        obs_dim: (1, 180, 80) # (c, h, w)
        '''
        self.obs_w = obs_dim[2]
        self.obs_h = obs_dim[1]
        self.obs_c = obs_dim[0]

        self.conv_w = self.obs_w // 4 # 4是因为两次池化
        self.conv_h = self.obs_h // 4

        self.conv1 = nn.Conv2d(self.obs_c, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.l1 = nn.Linear(32*self.conv_w*self.conv_h, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, hidden_3)
        self.l4 = nn.Linear(hidden_3, 1)

    def forward(self, obs):
        x = F.relu(self.conv1(obs)) # 1x180x80 -> 16x180x80
        x = F.max_pool2d(x, kernel_size=2, stride=2) # 16x180x80 -> 16x90x40
        x = F.relu(self.conv2(x)) # 16x90x40 -> 32x90x40
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # 32x90x40 -> 32x45x20
        x = x.reshape(x.size(0), -1) # 32x45x20 -> 1x28800 # shape(batch_size, -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        value = self.l4(x)
        return value

# -----------------  Actor_Critic -----------------

class Actor_Critic(nn.Module):
    def __init__(self, obs_dim, num_actions):
        super(Actor_Critic, self).__init__()
        self.channels = 32
        self.kernel_size = 3
        self.stride = 2
        self.padding = 1

        input_channels = obs_dim[0]
        h = obs_dim[1]
        w = obs_dim[2]
        
        # 卷积层序列
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, self.channels, self.kernel_size, self.stride, self.padding, dilation=1, groups=1),
            nn.BatchNorm2d(self.channels),  
            nn.ReLU(),
            nn.Conv2d(self.channels, self.channels, self.kernel_size, self.stride, self.padding, dilation=1, groups=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(),
            nn.Conv2d(self.channels, self.channels, self.kernel_size, self.stride, self.padding, dilation=1, groups=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(),
            nn.Conv2d(self.channels, self.channels, self.kernel_size, self.stride, self.padding, dilation=1, groups=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU()
        )
        
        # 全连接层参数计算
        self.fc_input_dim = self._get_conv_output_dim((1, input_channels, h, w))  # 假设输入为84x84
        
        # 全连接层
        self.linear0 = nn.Linear(self.fc_input_dim, 200)
        self.actor = nn.Linear(200, num_actions)
        self.critic = nn.Linear(200, 1)
        
        # 初始化权重
        self._init_weights()

    def _get_conv_output_dim(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(input_shape)
            output = self.conv_layers(dummy_input)
            return int(torch.prod(torch.tensor(output.size()[1:])))  # channels * height * width

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.linear0(x))
        logits = self.actor(x)
        probs = F.softmax(logits, dim=-1)
        value = self.critic(x)
        return probs, value

class Actor_Critic_MLP(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden = 200):
        super(Actor_Critic_MLP, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden, )
        self.l2 = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

        # # xavier初始化
        # nn.init.xavier_normal_(self.l1.weight.data)
        # nn.init.xavier_normal_(self.l2.weight.data)


    def forward(self, obs):
        x = F.relu(self.l1(obs))
        probs = F.softmax(x, dim=-1)
        value = self.critic(x)
        return probs, value


class Agent:
    def __init__(self, obs_dim, action_dim, actor_lr, critic_lr, is_continue, device, image):
        
        # if image:
        #     # if is_continue:
        #     #     self.actor = Actor_image(obs_dim, action_dim, ).to(device)
        #     # else:
        #     #     self.actor = Actor_discrete_image(obs_dim, action_dim, ).to(device)
        #     # self.critic = Critic_image( obs_dim ).to(device)
        #     self.actor_critic = Actor_Critic_MLP(obs_dim, action_dim).to(device)
        #     self.actor_critic_optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=actor_lr)
        # else:
            # if is_continue:
            #     self.actor = Actor(obs_dim, action_dim, ).to(device)
            # else:
        self.actor = Actor_discrete(obs_dim, action_dim, ).to(device)
        self.critic = Critic( obs_dim ).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

    def update_actor_critic(self, actor_critic_loss):
        self.actor_critic_optimizer.zero_grad()
        actor_critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
        self.actor_critic_optimizer.step()



## 第二部分：定义DQN算法类
class PPO:
    def __init__(self, dim_info, is_continue, actor_lr, critic_lr, horizon, device, trick = None,image = False):

        obs_dim, action_dim = dim_info
        self.agent = Agent(obs_dim, action_dim,  actor_lr, critic_lr, is_continue, device,image)
        #self.buffer = Buffer_for_PPO(horizon, obs_dim, act_dim = action_dim if is_continue else 1, device = device,) #Buffer中说明了act_dim和action_dim的区别
        self.device = device
        self.is_continue = is_continue
        print('actor_type:continue') if self.is_continue else print('actor_type:discrete')

        self.horizon = int(horizon)
        self.trick = trick

    def select_action(self, obs):
        obs = torch.as_tensor(obs,dtype=torch.float32).unsqueeze(dim = 0).to(self.device) # 1xcxhxw #.reshape(1, -1).to(self.device) # 1xobs_dim
        # if self.is_continue: # dqn 无此项
        #     mean, std = self.agent.actor(obs)
        #     dist = Normal(mean, std)
        #     action = dist.sample()
        #     action_log_pi = dist.log_prob(action) # 1xaction_dim
        # else:
        dist = Categorical(probs=self.agent.actor(obs))
        action = dist.sample()
        action_log_pi = dist.log_prob(action)
        # to 真实值
        action = action.detach().cpu().numpy().squeeze(0) # 1xaction_dim ->action_dim
        action_log_pi = action_log_pi.detach().cpu().numpy().squeeze(0) # 1xaction_dim ->action_dim
        return action , action_log_pi
    
    def evaluate_action(self, obs):
        obs = torch.as_tensor(obs,dtype=torch.float32).reshape(1, -1).to(self.device)
        if self.is_continue:
            mean, _ = self.agent.actor(obs)
            action = mean.detach().cpu().numpy().squeeze(0)
        else:
            a_prob = self.agent.actor(obs).detach().cpu().numpy().squeeze(0) 
            action = np.argmax(a_prob)
        return action
    
    ## buffer相关
    '''PPO论文中提到
    计算优势函数 有两种方法1.generalized advantage estimation 2.finite-horizon estimators
    第2种实现方法在许多代码上的实现方法不一,有buffer中存入return和value值的方法,也有在buffer里不存，而在在更新时计算的方法。
    这里我们选择第1种,在buffer中不会存在上述争议。
    通常ppo的buffer中存储的是obs, action, reward, next_obs, done, log_pi ;
    比较1.不存储log_pi,而是在更新时计算出log_pi_old, 2.存储log_pi，将此作为log_pi_old 发现2更好 采用2
    '''
    # def add(self, obs, action, reward, next_obs, done, action_log_pi , adv_dones):
    #     self.buffer.add(obs, action, reward, next_obs, done, action_log_pi , adv_dones)
    
    ## ppo 无 buffer_sample  

    ## PPO算法相关
    '''
    论文：GENERALIZED ADVANTAGE ESTIMATION:https://arxiv.org/pdf/1506.02438 提到
    先更新critic会造成额外的偏差，所以PPO这里 先更新actor，再更新critic ,且PPO主要是策略更新的方法
    lmbda = 0 时 GAE 为 one-step TD ; lmbda = 1时，GAE 为 MC
    '''
    def learn(self, minibatch_size, gamma, lmbda ,clip_param, K_epochs, entropy_coefficient ,buffer):
        '''
        done : dead or win
        adv_done : dead or win or reach max step
        gae 公式：A_t = delta_t + gamma * lmbda * A_t+1 * (1 - adv_done) 
        '''
        obs, action, reward, next_obs, done , action_log_pi , adv_dones = buffer.all(self.device)
        #obs = buffer['obs']

        buffer_len = len(obs)
        # 计算GAE
        with torch.no_grad():  # adv and v_target have no gradient
            adv = torch.zeros(buffer_len,dtype=torch.float32)
            gae = 0
            vs = self.agent.critic(obs)
            vs_ = self.agent.critic(next_obs)
            td_delta = reward + gamma * (1.0 - done) * vs_ - vs
            td_delta = td_delta.reshape(-1).cpu().detach().numpy()
            adv_dones = adv_dones.reshape(-1).cpu().detach().numpy()
            for i in reversed(range(buffer_len)):
                gae = td_delta[i] + gamma * lmbda * gae * (1.0 - adv_dones[i])
                adv[i] = gae
            adv = adv.reshape(-1, 1).to(self.device) ## cuda
            v_target = adv + vs  
            # if self.trick['adv_norm']:  # Trick :advantage normalization
            adv = ((adv - adv.mean()) / (adv.std() + 1e-5)) 
        '''
        ## 计算log_pi_old 比较1
        mean , std = self.agent.actor(obs)
        dist = Normal(mean, std)
        log_pi_old = dist.log_prob(action).sum(dim = 1 ,keepdim = True) 
        action_log_pi = log_pi_old.detach() 
        '''
        minibatch_size = buffer_len 
        # Optimize policy for K epochs:
        for _ in range(K_epochs): 
            # 随机打乱样本 并 生成小批量
            shuffled_indices = np.random.permutation(buffer_len)
            indexes = [shuffled_indices[i:i + minibatch_size] for i in range(0, buffer_len, minibatch_size)]
            for index in indexes:
                # 先更新actor
                # if self.is_continue:
                #     mean, std = self.agent.actor(obs[index])
                #     dist_now = Normal(mean, std)
                #     dist_entropy = dist_now.entropy().sum(dim = 1, keepdim=True)  # mini_batch_size x action_dim -> mini_batch_size x 1
                #     action_log_pi_now = dist_now.log_prob(action[index]) # mini_batch_size x 1
                # else:
                dist_now = Categorical(probs=self.agent.actor(obs[index]))
                dist_entropy = dist_now.entropy().reshape(-1,1) # mini_batch_size  -> mini_batch_size x 1
                action_log_pi_now = dist_now.log_prob(action[index].reshape(-1)).reshape(-1,1) # mini_batch_size  -> mini_batch_size x 1
                '''
                公式： ratio = pi_now/pi_old = exp(log(a)-log(b))
                In multi-dimensional continuous action space，we need to sum up the log_prob
                Only calculate the gradient of 'a_logprob_now' in ratios
                '''
                # print('action_log_pi_now:',action_log_pi_now.shape)
                # print('action_log_pi:',action_log_pi[index].shape)
                # print('action_log_pi:',action_log_pi.shape)
                ratios = torch.exp(action_log_pi_now.sum(dim = 1, keepdim=True) - action_log_pi[index].sum(dim = 1, keepdim=True))  # shape(mini_batch_size X 1)
                surr1 = ratios * adv[index]  
                surr2 = torch.clamp(ratios, 1 - clip_param, 1 + clip_param) * adv[index]  
                actor_loss = -torch.min(surr1, surr2).mean() - entropy_coefficient * dist_entropy.mean()
                self.agent.update_actor(actor_loss)
                '''or  (mean -> 转换为标量)
                actor_loss = -torch.min(surr1, surr2) - entropy_coefficient * dist_entropy
                self.agent.update_actor(actor_loss.mean())
                '''

                # 再更新critic
                v_s = self.agent.critic(obs[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                self.agent.update_critic(critic_loss)
                #self.agent.update_actor_critic(actor_loss+critic_loss)
        
        ## 清空buffer
        buffer.clear()
    
    ## 保存模型
    def save(self, model_dir):
        torch.save(self.agent.actor.state_dict(), os.path.join(model_dir,"PPO.pt"))
    
    ## 加载模型
    @staticmethod 
    def load(dim_info, is_continue ,model_dir,trick=None):
        policy = PPO(dim_info,is_continue,0,0,0,device = torch.device("cpu"), trick = trick)
        policy.agent.actor.load_state_dict(torch.load(os.path.join(model_dir,"PPO.pt")))
        return policy

## 第三部分：main函数   
''' 这里不用离散转连续域技巧'''
def get_env_atari(env_name,is_dis_to_con = False):
    env = gym.make(env_name)
    if isinstance(env.observation_space, gym.spaces.Box):
        obs_dim = env.observation_space.shape #
    else:
        obs_dim = 1
    if isinstance(env.action_space, gym.spaces.Box): # 是否动作连续环境
        action_dim = env.action_space.shape[0]
        dim_info = [obs_dim,action_dim]
        max_action = env.action_space.high[0]
        is_continuous = True # 指定buffer和算法是否用于连续动作
        if is_dis_to_con :
            if action_dim == 1:
                dim_info = [obs_dim,16]  # 离散动作空间
                is_continuous = False
            else: # 多重连续动作空间->多重离散动作空间
                dim_info = [obs_dim,2**action_dim]  # 离散动作空间
                is_continuous = False
    else:
        action_dim = env.action_space.n
        dim_info = [obs_dim,action_dim]
        max_action = None
        is_continuous = False
    
    return env,dim_info, max_action, is_continuous #dqn中均转为离散域.max_action没用到

## make_dir
def make_dir(env_name,policy_name = 'DQN',trick = None):
    script_dir = os.path.dirname(os.path.abspath(__file__)) # 当前脚本文件夹
    env_dir = os.path.join(script_dir,'./results', env_name)
    os.makedirs(env_dir) if not os.path.exists(env_dir) else None
    print('trick:',trick)
    # 确定前缀
    if trick is None or not any(trick.values()):
        prefix = policy_name + '_'
    else:
        prefix = policy_name + '_'
        for key in trick.keys():
            if trick[key]:
                prefix += key + '_'
    # 查找现有的文件夹并确定下一个编号
    pattern = re.compile(f'^{prefix}\d+') # ^ 表示开头，\d 表示数字，+表示至少一个
    existing_dirs = [d for d in os.listdir(env_dir) if pattern.match(d)]
    max_number = 0 if not existing_dirs else max([int(d.split('_')[-1]) for d in existing_dirs if d.split('_')[-1].isdigit()])
    model_dir = os.path.join(env_dir, prefix + str(max_number + 1))
    os.makedirs(model_dir)
    return model_dir

def to_gray(image_in):
    image=image_in.copy()
    image[image != 0] = 1  # 0为白色，1为黑色 相当于直接归一了 0-255 -> 0-1
    return image
 
def prepro(image_in):
    image=image_in.copy()
    image = image[35:185] # crop
    image = image[::2,::2,0] # downsample by factor of 2
    image[image == 144] = 0  # 擦除背景 (background type 1)
    image[image == 109] = 0  # 擦除背景
    image[image != 0] = 1  # 转为灰度图，除了黑色外其他都是白色
    return image.astype(np.float32).ravel()

class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = self.std #####！！！！ 明天看其他是否有错
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False #是否更新均值和方差，在评估时，update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x
'''
env链接：https://ale.farama.org/
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 环境参数t
    parser.add_argument("--env_name", type = str,default='Pong-v0') #"ALE/Freeway-v5") 
    # 共有参数
    parser.add_argument("--seed", type=int, default=100) # 0 10 100tt
    parser.add_argument("--max_episodes", type=int, default=int(2000)) #500
    parser.add_argument("--save_freq", type=int, default=int(400//4))
    parser.add_argument("--start_steps", type=int, default=0) #ppo无此参数
    parser.add_argument("--random_steps", type=int, default=0)  #dqn 无此参数
    parser.add_argument("--learn_steps_interval", type=int, default=0)  # 这个算法不方便用
    parser.add_argument("--is_dis_to_con", type=bool, default=False) # dqn 默认为True
    # 新增
    parser.add_argument("--learn_episodes_interval", type=int, default=1)  
    # 训练参数
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    ## A-C参数t
    parser.add_argument("--actor_lr", type=float, default=1e-3) # 1e-3 #
    parser.add_argument("--critic_lr", type=float, default=1e-3) # 1e-3
    # PPO独有参数
    parser.add_argument("--horizon", type=int, default=128) #根据环境更改：max_episodes数量级在百单位时horizon = 2048 minibatch_size = 64 小于百单位时，horizon = 128  minibatch_size = 32
    parser.add_argument("--clip_param", type=float, default=0.2) 
    parser.add_argument("--K_epochs", type=int, default=10) 
    parser.add_argument("--entropy_coefficient", type=float, default=0.01)
    parser.add_argument("--minibatch_size", type=int, default=32) # 32
    parser.add_argument("--lmbda", type=float, default=0.95) # GAE参数
    # trick参数
    parser.add_argument("--policy_name", type=str, default='PPO')
    parser.add_argument("--trick", type=dict, default={'reward_norm':False,}) 
    parser.add_argument("--image", type=bool, default=True) # 是否使用图像

    # device参数
    parser.add_argument("--device", type=str, default='cpu') # cpu/cuda
    args = parser.parse_args()
    print(args)
    print('Algorithm:',args.policy_name)
    
    ## 环境配置
    env,dim_info,max_action,is_continue = get_env_atari(args.env_name,args.is_dis_to_con)
    
    # 具体要看此环境图形如何裁剪和处理
    if args.env_name == 'ALE/Freeway-v5':
        obs ,info =  env.reset()
        background = obs # reset当背景
        obs = obs[::2, ::2,0]  # 105x80
        stack_num = 4
        dim_info[0] = (stack_num,obs.shape[0],obs.shape[1]) # (4,105,80)
        
        first_action = 0 # 此环境中 无动作的动作为0
    elif args.env_name == 'Pong-v0':
        
        obs ,info =  env.reset()
        obs = prepro(obs)
        prev_x = None #obs
        stack_num = 1
        dim_info[0] = 75*80 #(stack_num,obs.shape[0],obs.shape[1])
        dim_info[1] = 2
        D = 75*80 #obs.shape[0] * obs.shape[1]
        first_action = 2 # 2 为向上



    print(f'Env:{args.env_name}  obs_dim:{dim_info[0]}  action_dim:{dim_info[1]}  max_action:{max_action}  max_episodes:{args.max_episodes}')

    ## 随机数种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ### cuda
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('Random Seed:',args.seed)

    ## 保存文件夹
    model_dir = make_dir(args.env_name,policy_name = args.policy_name ,trick=args.trick)
    print(f'model_dir: {model_dir}')
    writer = SummaryWriter(model_dir)

    ## device参数
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
    
    ## 算法配置
    policy = PPO(dim_info, is_continue, actor_lr = args.actor_lr, critic_lr = args.critic_lr, horizon = args.horizon, device = device, trick = args.trick, image = args.image)
    buffer = Buffer_atari()
    # buffer.stack_frame = True
    # buffer.stack_num = stack_num

    env_spec = gym.spec('Pong-v0')
    print('reward_threshold:',env_spec.reward_threshold if env_spec.reward_threshold else 'No Threshold = Higher is better')
    time_ = time.time()
    ## 训练
    episode_num = 0
    step = 0
    episode_reward = 0
    train_return = []
    running_reward = None

    obs ,info =  env.reset(seed=args.seed)
    #env.action_space.seed(seed=args.seed) if args.random_steps > 0 else None # 针对action复现:env.action_space.sample()
    episode_first_step  =True
    if args.trick['reward_norm']:  
        reward_norm = Normalization(shape=1)
    while episode_num < args.max_episodes:
        #paused = pause_game(paused)
    
        if episode_first_step  == True:
            obs  = env.step(first_action)[0] 
            cur_x = prepro(obs)
            obs = np.zeros(D) #cur_x - prev_x if prev_x is not None else np.zeros(D)
            prev_x = cur_x
            #obs = to_gray((obs - background)[::2, ::2, 0] ).reshape(1,dim_info[0][1] ,dim_info[0][2])
            #obs = np.stack([obs]*stack_num,axis = 0) # [105,80] -> [4,105,80]
            #obs = np.repeat(obs, stack_num, axis=0) # 如果是是[1,105,80] -> [4,105,80]
            #print('obs_shape:',obs.shape)
            episode_first_step = False
        step +=1
        #cur_x = prepro(obs)
        # x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        # prev_x = cur_x
        # 获取动作t
        #obs = obs.reshape(1,dim_info[0][1] ,dim_info[0][2])
        action , action_log_pi = policy.select_action(obs)   # 这里离散
        action_ = 2 if action == 0 else 3
        # 探索环境
        next_obs, reward,terminated, truncated , info = env.step(action_) 
        cur_x = prepro(next_obs)
        next_obs = (cur_x - prev_x)#.reshape(1,dim_info[0][1] ,dim_info[0][2])
        #prev_x = cur_x
        #next_obs = to_gray((next_obs - background)[::2, ::2, 0] ).reshape(1,dim_info[0][1] ,dim_info[0][2])
        # 帧[1,2,3,4] -> [2,3,4,5]
        #print(next_obs.shape)
        #next_obs = np.append(obs[1:],next_obs,axis = 0)
        #print('action:',action,'reward:',reward)
        #print(action,reward)
        done = terminated or truncated
        done_bool = terminated     ### truncated 为超过最大步数
        # if args.trick['reward_norm']:  
        #     reward_ = reward_norm(reward)[0]
        #     #print('reward:',reward , 'reward_:',reward_)
        # if args.trick['reward_norm']:
        #     buffer.add_buffer(obs, action, reward_, next_obs, done_bool, action_log_pi,done)
        # else:
        buffer.add_buffer(obs, action, reward, next_obs, done_bool, action_log_pi,done)
        #policy.add(obs, action, reward, next_obs, done_bool, action_log_pi,done)
        episode_reward += reward
        obs = next_obs
        #obs_screen = next_obs_screen
        # episode 结束t
        if done:
            
            episode_first_step = True 
            # 将buffer里的
            ## 显示t
            running_reward = episode_reward if running_reward is None else running_reward * 0.99 + episode_reward * 0.01
            writer.add_scalar('reward', episode_reward, episode_num + 1)
            if  (episode_num + 1) % 10 == 0:
                print("episode: {}, reward: {} , running_reward:{}".format(episode_num + 1, episode_reward , running_reward))
            # writer.add_scalar('episode_time', info['episode_time'], episode_num + 1)
            # writer.add_scalar('mark', info['mark'], episode_num + 1)
            train_return.append(episode_reward)

            ''' 在游戏时更新网络会影响游戏情况，故在游戏结束后更新网络'''
            # 满足episodes,更新网络
            if (episode_num + 1) % args.learn_episodes_interval == 0:
                policy.learn(args.minibatch_size, args.gamma, args.lmbda, args.clip_param, args.K_epochs, args.entropy_coefficient,buffer)
                #buffer.process_buffer()
                # print('learn')
                # obs = env.reset()
                #episode_first_step = True
                #episode_reward = 0

            episode_num += 1
            #obs,info = env.reset(seed=args.seed)
            obs , info= env.reset(seed=args.seed)
            episode_reward = 0
        
        # 保存模型t
        if episode_num % args.save_freq == 0:
            policy.save(model_dir)

    
    print('total_time:',time.time()-time_)
    policy.save(model_dir)
    ## 保存数据
    np.save(os.path.join(model_dir,f"{args.policy_name}_seed_{args.seed}.npy"),np.array(train_return))