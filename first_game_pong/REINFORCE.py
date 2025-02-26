import os
# 设置OMP_WAIT_POLICY为PASSIVE，让等待的线程不消耗CPU资源 #确保在pytorch前设置
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE' #确保在pytorch前设置

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
import numpy as np

import gymnasium as gym
import argparse

## 其他
import re
import time
from torch.utils.tensorboard import SummaryWriter
import ale_py
gym.register_envs(ale_py)

'''
1: 将pong130 修改为此文件 两处不同 learn_episodes_interval = 1 , Adam
2：1基础上 learn_episodes_interval = 10
3. 1基础上 增加bias state初始化默认
4  3基础上 增加1层隐藏层 200
9  改cnn 1e-4
15 改cnn 1e-3 (deepseek改版)
17 改cnn 1e-3 (dqn改版)
'''


'''
一个深度强化学习算法分三个部分实现：
1.Agent类:包括actor、critic、target_actor、target_critic、actor_optimizer、critic_optimizer、
2.DQN算法类:包括select_action,learn、save、load等方法,为具体的算法细节实现
3.main函数:实例化DQN类,主要参数的设置,训练、测试、保存模型等
'''
'''REINFORCE算法的实现
这里的REINFORCE参考https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5实现
'''
'''  参数修改 改三处 1.MLP的hidden  2.main中args 3.dis_to_con中的离散转连续空间维度 '''
## 第一部分：定义Agent类
class Policy_MLP(nn.Module):
    '''
    只有一层隐藏层的多层感知机。
    batch_size x obs_dim -> batch_size x hidden -> batch_size x action_dim
    公式：a = relu(W1*s + b1), q = W2*a + b2
    采用在离散动作空间上的softmax()函数来实现一个可学习的多项分布
    '''
    def __init__(self, obs_dim, action_dim, hidden = 200):
        super(Policy_MLP, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden, bias=False)
        self.l2 = nn.Linear(hidden, action_dim, bias=False)

        # xavier初始化
        nn.init.xavier_normal_(self.l1.weight.data)
        nn.init.xavier_normal_(self.l2.weight.data)


    def forward(self, obs):
        x = F.relu(self.l1(obs))
        return F.softmax(self.l2(x), dim=1)


class Agent:
    def __init__(self, obs_dim, action_dim, policy_net_lr , device):

        self.policy_net = Policy_MLP(obs_dim, action_dim).to(device)
        #self.Qnet_target = deepcopy(self.Qnet)

        self.policy_net_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr = policy_net_lr)
    
    def update_policy(self, loss):
        self.policy_net_optimizer.zero_grad()
        loss.backward()
        self.policy_net_optimizer.step()

## 第二部分：定义DQN算法类
class REINFORCE:
    def __init__(self, dim_info, is_continue, policy_net_lr, device, trick = None):
        obs_dim, action_dim = dim_info
        self.agent = Agent(obs_dim, action_dim, policy_net_lr, device)
        #self.buffer = Buffer(buffer_size, obs_dim, act_dim = action_dim if is_continue else 1, device = device) #Buffer中说明了act_dim和action_dim的区别
        self.device = device
        self.is_continue = is_continue

        # buffer
        self.rewards = []
        self.log_probs = []
        self.done = []

    def select_action(self, obs):
        '''
        输入的obs shape为(obs_dim) np.array reshape => (1,obs_dim) torch.tensor 
        '''
        obs = torch.as_tensor(obs,dtype=torch.float32).reshape(1, -1).to(self.device)

        probs = self.agent.policy_net(obs)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        self.log_probs.append(log_prob)

        return action.detach().cpu().numpy().squeeze(0) #,log_prob.detach().cpu().numpy().squeeze(0) ## ??
    
    def evaluate_action(self, obs):
        '''DQN的探索策略是ε-greedy, 评估时,在main中去掉ε就行。类似于确定性策略ddpg。'''
        return self.select_action(obs)

    ## buffer相关
    def add(self, reward, done):
        self.rewards.append(reward)
        self.done.append(done) 

    def all(self):
        return self.rewards, self.done ,self.log_probs
    

    ## 算法相关
    def learn(self, gamma):

        #obs, actions, rewards, next_obs, dones ,log_probs = self.sample(batch_size) 
        rewards, dones, log_probs = self.all()
        returns = []
        G = 0
        for t in reversed(range(len(rewards))):
            ## special for Pong
            #if rewards[t] != 0: G = 0
            G = rewards[t] + gamma * G * (1 - dones[t])  # 处理终止状态
            returns.insert(0, G)  # 逆序插入，保持正向顺序

        returns = torch.as_tensor(returns, dtype=torch.float32).to(self.device)

        # 归一化回报（减少方差，常见但非必需）
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for log_prob, g in zip(log_probs, returns):
            loss += -log_prob * g

        self.agent.update_policy(loss)

        self.rewards = []
        self.done = []
        self.log_probs = []


    ## 保存模型
    def save(self, model_dir):
        torch.save(self.agent.policy_net.state_dict(), os.path.join(model_dir,"REINFORCE.pt"))
    ## 加载模型
    @staticmethod 
    def load(dim_info, is_continue ,model_dir, trick = None):
        policy = REINFORCE(dim_info,is_continue,0,device = torch.device("cpu"),trick = trick)
        policy.agent.policy_net.load_state_dict(torch.load(os.path.join(model_dir,"REINFORCE.pt")))
        return policy

### 第三部分：main函数
## 环境配置

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

# 环境函数
def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:185] # crop # 改3
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float32).ravel() # 展平
''' 

'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 环境参数
    parser.add_argument("--env_name", type = str,default="Pong-v0") 
    # 共有参数
    parser.add_argument("--seed", type=int, default=0) # 0 10 100
    parser.add_argument("--max_episodes", type=int, default=int(5000))
    parser.add_argument("--save_freq", type=int, default=int(500//4))
    parser.add_argument("--start_steps", type=int, default=500)
    parser.add_argument("--random_steps", type=int, default=0)  #可选择是否使用 dqn论文无此参数
    parser.add_argument("--learn_steps_interval", type=int, default=1)
    parser.add_argument("--learn_episodes_interval", type=int, default=1) 
    #parser.add_argument("--is_dis_to_con", type=bool, default=True) # dqn 默认为True
    # 训练参数
    parser.add_argument("--gamma", type=float, default=0.99)
    ## REINFORCE参数
    parser.add_argument("--policy_net_lr", type=float, default=1e-3)
    # trick参数
    parser.add_argument("--policy_name", type=str, default='REINFORCE')
    parser.add_argument("--trick", type=str, default=None)
    # device参数
    parser.add_argument("--device", type=str, default='cpu')
    
    args = parser.parse_args()
    print(args)
    print('-'*50)
    print('Algorithm:',args.policy_name)
    ## 环境配置

    env = gym.make(args.env_name)
    dim_info = [75*80,2]
    max_action = None
    is_continue = False
    prev_x = None 
    running_reward = None

    print(f'Env:{args.env_name}  obs_dim:{dim_info[0]}  action_dim:{dim_info[1]}  max_action:{max_action}  max_episodes:{args.max_episodes}')

    ## 随机数种子(cpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ### cuda
    torch.cuda.manual_seed(args.seed) # 经过测试,使用cuda时,只加这句就能保证两次结果一致
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
    policy = REINFORCE(dim_info,is_continue,args.policy_net_lr,device)

    env_spec = gym.spec(args.env_name)
    print('reward_threshold:',env_spec.reward_threshold if env_spec.reward_threshold else 'No Threshold = Higher is better')
    time_ = time.time()
    ## 训练
    episode_num = 0
    step = 0
    episode_reward = 0
    train_return = []
    obs,info = env.reset(seed=args.seed)  # 针对obs复现
    env.action_space.seed(seed=args.seed) if args.random_steps > 0 else None # 针对action复现:env.action_space.sample()
    while episode_num < args.max_episodes:
        step +=1
        # 获取动作 区分动作action_为环境中的动作 action为要训练的动作
        cur_x = prepro(obs)
        x = cur_x - prev_x if prev_x is not None else np.zeros(dim_info[0])
        prev_x = cur_x

        action = policy.select_action(x)
        #print('action:',action)
        action_ = 2 if action == 0 else 3
        # 此时输出action为离散动作        
        # 探索环境
        next_obs, reward,terminated, truncated, infos = env.step(action_) 
        done = terminated or truncated
        done_bool = terminated     ### truncated 为超过最大步数
        policy.add(reward, done_bool)
        episode_reward += reward
        obs = next_obs
        # episode 结束
        if done:
            ## 显示
            running_reward = episode_reward if running_reward is None else running_reward * 0.99 + episode_reward * 0.01
            writer.add_scalar('reward', episode_reward, episode_num + 1)
            if  (episode_num + 1) % 10 == 0:
                print("episode: {}, reward: {} , running_reward:{}".format(episode_num + 1, episode_reward , running_reward))
                
            ## 保存
            if (episode_num + 1) % args.save_freq == 0:
                policy.save(model_dir)
            
            train_return.append(episode_reward)

            episode_num += 1
            obs,info = env.reset(seed=args.seed)
            episode_reward = 0
            prev_x = None
        
            # 满足episodes,更新网络
            if episode_num % args.learn_episodes_interval == 0:
                policy.learn(args.gamma)
        
        # 保存模型
        if episode_num % args.save_freq == 0:
            policy.save(model_dir)

    
    print('total_time:',time.time()-time_)
    policy.save(model_dir)
    ## 保存数据
    np.save(os.path.join(model_dir,f"{args.policy_name}_seed_{args.seed}.npy"),np.array(train_return))




