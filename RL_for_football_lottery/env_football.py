## 写环境的一些思路
## 比赛编号范围  当天13点到第二天13点，左开右闭的区间 一般可投注两天
'''
先获取一个相同的比赛编号的比赛（当天可投注的所有比赛）
其中除了单固可以为单场投注（既可以只选择一场也可以选择多场）外，其余均为多场投注


为简化动作的的设计，这里只最多只两场投注,且只玩胜平负游戏；（即当天能投注的所有情况如下：假设为1个单场A 2个多场B、C）
A AB AC BC
对A:动作有：
（单场投注的所有动作）
胜、平、负、胜平、胜负、平负、胜平负(7种),不投注,是否可投注 9种
对AB动作有：
胜1胜2,胜1平2,胜1负2,平1胜2,平1平2,平1负2,负1胜2,负1平2,负1负2
胜1平1胜2,胜1平1平2,胜1平1负2,胜1负1胜2,胜1负1平2,胜1负1负2,平1负1胜2,平1负1平2,平1负1负2,胜2平2胜1...
发现这样设计不好：不如在算法上改成多头/多智能体，其中一个动作选择一个比赛的所有动作，
例：[0,1] 第一场选择胜，第二场选择负
若是只有一场（单场投注），则将第二个动作屏蔽 action mask，使得是否可投注为是


环境逻辑：
首先获取比赛编号相同的所有比赛，按照时间排序
先检查是否有单场投注，先投注单场的
若无，则投注多场的，根据时间顺序，时间一样则按照编号排序
排列组合所有可投注的方式存到一个列表中，然后遍历此列表，每step一次遍取出下一个场可投注的方式，直到当天的数据遍历完，然后继续下一天的数据

环境设有随机种子，每次随机一个时间点开始，玩1000场结束。

为简便环境，这里只进行单场投注和两场组合投注(即，可以设计成两智能体的环境)

'''

'''
环境借鉴SMAC环境
这里设计为多智能体环境，使用多智能体算法来进行实验
动作：n x ([不可投注,胜, 平, 负, 胜平, 胜负, 平负, 胜平负, 不投注] 9个)
状态：n x ( 自身状态:[胜平负赔率(7*2*3),让球盘口赔率(4*2*3),大小盘口赔率(4*2*3)]  # 共90个信息 )
奖励：和现实足彩一致，投注成功则获得赔率乘以投注额的奖金，投注失败则损失投注额
'''

import json
import pandas as pd
import datetime
import random
from datetime import timedelta
import numpy as np


## 基础设定
NUM_GAMES = 1000  # 每个episode的游戏数量
NUM_ACTIONS = 9  # 每个单场投注的动作数量
NUM_AGENTS = 2  # 共两个智能体
PRICE_PER_BET = 2  # 每次投注的价格

### 可用动作


def get_can_bet(single_game,agent_id):
    """返回是否可投注
    默认在单场投注下，第二场不可投注
    single_game: 是否单场投注
    agent_id: 智能体编号
    """
    if single_game and agent_id ==  1:  
        return False
    return True

def get_avail_actions(single_game):
    """返回可用的投注动作字典
    '不可投注','胜', '平', '负', '胜平', '胜负', '平负', '胜平负', '不投注'
    '不可投注'： 0表示可投注，1表示不可投注
    """
    avail_actions = {}
    for i in range(NUM_AGENTS):
        if get_can_bet(single_game, i):
            avail_actions[i] = [0] + [1] * (NUM_ACTIONS - 1)  # 默认所有动作可用
        else:
            avail_actions[i] = [1] + [0] * (NUM_ACTIONS - 1)
    return avail_actions


def get_odds_info(full_data):
    """获取胜平负赔率信息"""
    odds_list = []
    
    try:
        # 获取初始和终赔赔率
        odds_data = full_data.get('odds', {})
        
        # 收集初始赔率
        initial_odds = odds_data.get('initial', {})
        for company in ['bifa', 'bet36', 'william', 'crown', 'ladbrokes', 'jingcai', 'jingcairangqiu']:
            if company in initial_odds:
                odds_list.extend([
                    float(initial_odds[company].get('win', 0)),
                    float(initial_odds[company].get('draw', 0)),
                    float(initial_odds[company].get('lose', 0))
                ])
            else:
                odds_list.extend([0.0, 0.0, 0.0])  # 如果没有该公司的赔率，填充0
        
        # 收集终赔赔率
        final_odds = odds_data.get('final', {})
        for company in ['bifa', 'bet36', 'william', 'crown', 'ladbrokes', 'jingcai', 'jingcairangqiu']:
            if company in final_odds:
                odds_list.extend([
                    float(final_odds[company].get('win', 0)),
                    float(final_odds[company].get('draw', 0)),
                    float(final_odds[company].get('lose', 0))
                ])
            else:
                odds_list.extend([0.0, 0.0, 0.0])  # 如果没有该公司的赔率，填充0
    except Exception as e:
        print(f"获取胜平负赔率时出错: {e}")
        return [0.0] * 42  # 7家公司 × 2种赔率(初始/终赔) × 3种结果(胜/平/负)
    
    return odds_list

def get_handicap_info(full_data):
    """获取让球盘口赔率信息"""
    handicap_list = []
    
    try:
        # 获取让球盘口数据
        handicap_data = full_data.get('handicap', {})
        
        # 收集初始让球盘口
        initial_handicap = handicap_data.get('initial', {})
        for company in ['bet36', 'william', 'crown', 'aomen']:
            if company in initial_handicap:
                handicap_list.extend([
                    float(initial_handicap[company].get('homeWaterLevel', 0)),
                    float(initial_handicap[company].get('handicapLine', 0)),
                    float(initial_handicap[company].get('awayWaterLevel', 0))
                ])
            else:
                handicap_list.extend([0.0, 0.0, 0.0])  # 如果没有该公司的让球盘口，填充0
        
        # 收集终盘让球盘口
        final_handicap = handicap_data.get('final', {})
        for company in ['bet36', 'william', 'crown', 'aomen']:
            if company in final_handicap:
                handicap_list.extend([
                    float(final_handicap[company].get('homeWaterLevel', 0)),
                    float(final_handicap[company].get('handicapLine', 0)),
                    float(final_handicap[company].get('awayWaterLevel', 0))
                ])
            else:
                handicap_list.extend([0.0, 0.0, 0.0])  # 如果没有该公司的让球盘口，填充0
    except Exception as e:
        print(f"获取让球盘口时出错: {e}")
        return [0.0] * 24  # 4家公司 × 2种盘口(初始/终盘) × 3个值(主水/盘口/客水)
    
    return handicap_list

def get_over_under_info(full_data):
    """获取大小球盘口赔率信息"""
    over_under_list = []
    
    try:
        # 获取大小球盘口数据
        over_under_data = full_data.get('Over_Under', {})
        
        # 收集初始大小球盘口
        initial_over_under = over_under_data.get('initial', {})
        for company in ['bet36', 'crown', 'aomen', 'ladbrokes']:
            if company in initial_over_under:
                over_under_list.extend([
                    float(initial_over_under[company].get('overWaterLevel', 0)),
                    float(initial_over_under[company].get('handicapLine', 0)),
                    float(initial_over_under[company].get('underWaterLevel', 0))
                ])
            else:
                over_under_list.extend([0.0, 0.0, 0.0])  # 如果没有该公司的大小球盘口，填充0
        
        # 收集终盘大小球盘口
        final_over_under = over_under_data.get('final', {})
        for company in ['bet36', 'crown', 'aomen', 'ladbrokes']:
            if company in final_over_under:
                over_under_list.extend([
                    float(final_over_under[company].get('overWaterLevel', 0)),
                    float(final_over_under[company].get('handicapLine', 0)),
                    float(final_over_under[company].get('underWaterLevel', 0))
                ])
            else:
                over_under_list.extend([0.0, 0.0, 0.0])  # 如果没有该公司的大小球盘口，填充0
    except Exception as e:
        print(f"获取大小球盘口时出错: {e}")
        return [0.0] * 24  # 4家公司 × 2种盘口(初始/终盘) × 3个值(大水/盘口/小水)
    
    return over_under_list


def get_self_obs(agent_id,data_dict):
    '''
    自身状态:[胜平负赔率(7*2*3),让球盘口赔率(4*2*3),大小盘口赔率(4*2*3)]  # 共90个信息  ## 暂不加入主客队信息和让球，纯赔率
    '''
    if data_dict['type'] == 'single':
        if agent_id == 0:
        # 单场投注
            match = data_dict['match']
            data = match['full_data']
            # 获取胜平负赔率
            odds_list = get_odds_info(data)
            # 获取让球盘口赔率
            handicap_list = get_handicap_info(data)
            # 获取大小球盘口赔率
            over_under_list = get_over_under_info(data)
            # 返回胜平负赔率
            return odds_list + handicap_list + over_under_list
        elif agent_id == 1:
            # 单场投注的第二个智能体不投注
            return [0] * 90
    elif data_dict['type'] == 'combo':

        match = data_dict['matches'][agent_id]
        data = match['full_data']
        
        odds_list1 = get_odds_info(data)
        handicap_list2 = get_handicap_info(data)
        over_under_list2 = get_over_under_info(data)
        
        return odds_list1 + handicap_list2 + over_under_list2
    
def get_obs(data_dict):
    '''
    获取当前比赛的观察信息
    data_dict: 当前比赛数据字典
    返回一个字典，包含所有智能体的观察信息
    '''
    obs = {}
    for agent_id in range(NUM_AGENTS):
        obs[agent_id] = np.array(get_self_obs(agent_id, data_dict), dtype=np.float32)
    return obs

def get_reward(data_dict, actions):
    '''
    根据投注结果计算奖励
    data_dict: 当前比赛数据字典
    actions: 智能体的动作字典 '不可投注','胜', '平', '负', '胜平', '胜负', '平负', '胜平负', '不投注'
    返回一个值
    '''
    # 将动作编号转换为对应的投注类型
    action_to_bet = {
        1: ['胜'],       # 胜
        2: ['平'],       # 平
        3: ['负'],       # 负
        4: ['胜', '平'],  # 胜平
        5: ['胜', '负'],  # 胜负
        6: ['平', '负'],  # 平负
        7: ['胜', '平', '负']  # 胜平负
    }
    
    # 计算投注的注数
    def calculate_bet_num(action):
        if action == 0 or action == 8:  # 不可投注或不投注
            return 0
        return len(action_to_bet.get(action, [])) #* PRICE_PER_BET
    
    # 检查投注是否命中
    def is_bet_hit(action, result):
        if action == 0 or action == 8:  # 不可投注或不投注
            return False
        return result in action_to_bet.get(action, [])
    
    if data_dict['type'] == 'single':
        # 单场投注
        match = data_dict['match']
        result = match['full_data']['results']['result']
        odds_result = match['full_data']['results']['oddsResult']
        
        action = actions[0]  # 只有第一个智能体投注
        
        # 计算投注花费
        cost = calculate_bet_num(action) * PRICE_PER_BET
        
        # 判断是否命中
        if is_bet_hit(action, result):
            # 赢了，获得赔率乘以投注额的奖金
            return odds_result * PRICE_PER_BET - cost
        else:
            # 输了，损失投注额
            return -cost
    
    elif data_dict['type'] == 'combo':
        # 两场组合投注
        results = [match['full_data']['results']['result'] for match in data_dict['matches']]
        odds_results = [match['full_data']['results']['oddsResult'] for match in data_dict['matches']]
        #print(odds_results)
        
        # 如果任何一方选择不投注或不可投注，则没有奖励
        if actions[0] == 0 or actions[1] == 0 or actions[0] == 8 or actions[1] == 8:
            return 0
            
        # 计算总花费
        total_cost = (calculate_bet_num(actions[0]) * calculate_bet_num(actions[1])) * PRICE_PER_BET
        
        # 判断两场比赛是否都命中
        hit1 = is_bet_hit(actions[0], results[0])
        hit2 = is_bet_hit(actions[1], results[1])
        
        if hit1 and hit2:
            # 两场都命中，获得两场赔率乘积乘以投注额的奖金
            bonus = sum([PRICE_PER_BET * odds for odds in odds_results])
            return bonus - total_cost
        else:
            # 至少一场没命中，损失所有投注额
            return -total_cost

class Env:
    '''
    动作：[不可投注,胜, 平, 负, 胜平, 胜负, 平负, 胜平负, 不投注] 9个
    '''
    def __init__(
        self,
        num_total_games= NUM_GAMES,  # 环境中的比赛数据
    ):
        # 读取JSON数据
        with open("football_data_2025_5_20.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)

        # 提取比赛状态并筛选已完成的比赛
        def extract_status(row):
            try:
                if isinstance(row['basicInfo'], dict) and 'status' in row['basicInfo']:
                    return row['basicInfo']['status']
                return None
            except:
                return None

        # 添加状态列
        df['status'] = df.apply(extract_status, axis=1)

        # 只保留状态为"完"的比赛
        df = df[df['status'] == '完']

        # 提取赔率结果并检查是否为数值类型
        def extract_odds_result(row):
            try:
                odds_result = row.get('results', {}).get('oddsResult')
                # 检查是否存在且为数值类型
                if odds_result is not None and isinstance(odds_result, (int, float)) :
                    # 如果是字符串形式的数字，转换为none
                    if isinstance(odds_result, str):
                        return None #float(odds_result)
                    return odds_result
                return None
            except Exception as e:
                print(f"提取赔率结果出错: {e}")
                return None

        # 添加赔率结果列
        df['odds_result'] = df.apply(extract_odds_result, axis=1)

        # 筛选赔率结果不为None的比赛
        df = df[df['odds_result'].notnull()]



        print(f"筛选后:完成比赛数量: {len(df)}")

        # 提取并转换比赛时间
        def extract_match_time(row):
            try:
                if isinstance(row['basicInfo'], dict) and 'matchTime' in row['basicInfo']:
                    return row['basicInfo']['matchTime']
                return None
            except:
                return None

        df['match_time'] = df.apply(extract_match_time, axis=1)
        df['match_time'] = pd.to_datetime(df['match_time'])

        self.df = df
        self.num_total_games = num_total_games
        
        # 设置日期范围
        self.start_date = datetime.datetime(2021, 1, 1)
        self.end_date = datetime.datetime(2025, 5, 20)

        # 计算总天数
        self.total_days = (self.end_date - self.start_date).days
        # agents个数
        self.num_agents = NUM_AGENTS  # 这里是两个智能体 
        self.agents_id = list(range(self.num_agents))  # 智能体编号列表

    def get_env_info(self):
        '''
        获取环境信息
        返回一个字典，包含环境的基本信息
        '''
        env_info = {
            "obs_shape": 90,  # 每个智能体的状态形状
            "action_shape": NUM_ACTIONS,  # 每个智能体的动作形状
            "n_agents": self.num_agents,  # 智能体数量
            "n_actions": NUM_ACTIONS,  # 每个智能体的动作数量
            "price_per_bet": PRICE_PER_BET,  # 每次投注的价格
            "num_total_games": self.num_total_games,  # 总比赛场数
        }
        return env_info
    
    def get_random_start_date(self):
        random_days = random.randint(0, self.total_days - 3)  # 至少留3天给窗口 只有-3 是正好可以满足1000场
        print(f"随机天数: {random_days}")
        random_date = self.start_date + timedelta(days=random_days)
        # 将时间设置为13:00
        random_date = random_date.replace(hour=13, minute=0, second=0, microsecond=0)
        return random_date
    
        # 获取指定时间窗口内的比赛
    def get_matches_in_window(self,start_time, window_days=2):
        end_time = start_time + timedelta(days=window_days)
        mask = (self.df['match_time'] >= start_time) & (self.df['match_time'] < end_time)
        return self.df[mask]
        
    def reset(self):
        # 随机选择一个起始日期
        current_date = self.get_random_start_date()
        print(f"随机选择的起始日期: {current_date.strftime('%Y-%m-%d %H:%M')}")
        all_bettable_games = []  # 存储所有可投注的组合
        window_count = 0

        while len(all_bettable_games) < self.num_total_games:
            window_count += 1
            # 获取当前窗口的比赛
            window_matches = self.get_matches_in_window(current_date)
            
            # print(f"\n窗口 {window_count}: {current_date.strftime('%Y-%m-%d %H:%M')} 到 {(current_date + timedelta(days=2)).strftime('%Y-%m-%d %H:%M')}")
            # print(f"该窗口内比赛数量: {len(window_matches)}")
            
            if len(window_matches) > 0:
                # 分离单固和普通比赛
                single_fixed_matches = []
                regular_matches = []
                
                for idx, match in window_matches.iterrows():
                    home_team = match['basicInfo'].get('homeTeam', 'Unknown')
                    away_team = match['basicInfo'].get('awayTeam', 'Unknown')
                    match_time = match['match_time']
                    is_single_fixed = match['basicInfo'].get('singleFixed') == '是'
                    
                    match_info = {
                        'id': idx,
                        'home_team': home_team,
                        'away_team': away_team,
                        'match_time': match_time,
                        'full_data': match  # 保存完整数据用于后续处理
                    }
                    
                    if is_single_fixed:
                        single_fixed_matches.append(match_info)
                        #print(f"  - [单固] {match_time.strftime('%Y-%m-%d %H:%M')}: {home_team} vs {away_team}")
                    else:
                        regular_matches.append(match_info)
                        #print(f"  - [普通] {match_time.strftime('%Y-%m-%d %H:%M')}: {home_team} vs {away_team}")
                
                # 处理单固比赛（单场投注）
                for single_match in single_fixed_matches:
                    all_bettable_games.append({
                        'type': 'single',
                        'match': single_match,
                        'date': current_date
                    })
                
                # 处理普通比赛（两场一组的组合投注）
                # 按比赛时间排序
                regular_matches.sort(key=lambda x: x['match_time'])
                
                # 生成两两组合
                from itertools import combinations
                for combo in combinations(regular_matches, 2):
                    all_bettable_games.append({
                        'type': 'combo',
                        'matches': [combo[0], combo[1]],
                        'date': current_date
                    })
                
                print(f"累计可投注组合数: {len(all_bettable_games)}")
                
                if len(all_bettable_games) >= self.num_total_games:
                    print(f"已达到目标场次 {self.num_total_games}，停止模拟")
                    break
            
            # 移动到下一天
            current_date = current_date + timedelta(days=1)
        
        # 截取至需要的场次数量
        self.all_bettable_games = all_bettable_games[:self.num_total_games]

        ## 初始化状态
        self.current_game_index = 0
        current_game_dict = self.all_bettable_games[self.current_game_index]

        obs = get_obs(current_game_dict)
        info = {
            "action_mask": get_avail_actions(current_game_dict['type'] == 'single'),
        }


        return obs , info  # 返回初始观察和信息字典

    def step(self, actions):
        '''
        actions: 智能体的动作字典
        '''
        actions = {agent_id: int(actions[agent_id]) for agent_id in self.agents_id}
        self.current_game_index += 1

        current_game_dict = self.all_bettable_games[self.current_game_index]



        obs = get_obs(current_game_dict)


        # 计算奖励
        reward_v = get_reward(current_game_dict, actions)
        reward = {agent_id: reward_v for agent_id in self.agents_id}

        ## 暂时不设置terminate,两者一致
        done = self.current_game_index >= self.num_total_games - 1
        terminal = {agent_id: done for agent_id in self.agents_id}
        truncated = terminal

        info = {
            "action_mask": get_avail_actions(current_game_dict['type'] == 'single'),
            "done": done,
        }


        return obs, reward, terminal, truncated, info  # 返回观察、奖励、终止状态和信息字典


if __name__ == "__main__":
    env = Env(num_total_games=NUM_GAMES)
    obs, info = env.reset()
    print("初始观察:", obs)
    print("初始信息:", info)

    reward_sum = 0

    for step in range(NUM_GAMES):
        actions = {agent_id: random.randint(0, NUM_ACTIONS - 1) for agent_id in env.agents_id}  # 随机动作
        action_mask = info["action_mask"]
        if action_mask:
            # 确保动作在可用范围内
            actions = {agent_id: action if action_mask[agent_id][action] == 1 else 0 for agent_id, action in actions.items()}

        obs, reward, terminal, truncated, info = env.step(actions)
        print(f"Step {step + 1}:")
        #print("观察:", obs)
        print("奖励:", reward)
        reward_sum += reward[0]
        #print("终止状态:", terminal)
        #print("信息:", info)
        if terminal[0]:  # 如果第一个智能体终止，则结束循环
            break
    print(f"总奖励: {reward_sum}")