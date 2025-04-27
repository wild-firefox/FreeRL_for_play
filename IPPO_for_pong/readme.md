IPPO for Pong
项目简介
这个项目是使用独立近端策略优化算法（Independent Proximal Policy Optimization, IPPO）训练智能体玩雅达利（Atari）经典游戏《Pong》。IPPO是PPO算法的多智能体变体。

## 安装环境和基础依赖指南
注意要在linux系统下安装环境。    
无linux系统的用户推荐使用WSL或Docker等虚拟机。  
见readme_install.ipynb

## 训练代码

```
IPPO.py
```


## 评估代码
```
eval_IPPO.py
```

两种模式变换
```python
# mode='rgb_array'时为保存gif mode='human'时为查看游戏
env = pong_v3.parallel_env(render_mode="rgb_array",max_cycles=10000)
```

## 训练记录展示

logdir后选择你的logdir

```bash
tensorboard --logdir=IPPO_for_pong/results/pong_v3/IPPO_22
```

## 最终效果展示

<img src="results/pong_v3/IPPO_22/pong_animation_20250426_145439.gif">

# 其他
eval_IPPO(test).py 和 IPPO(old).py为调试时的文件,可省略。

## 参考资料
1.[IPPO论文](https://arxiv.org/pdf/2011.09533)  
2.[pg-pong.py](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)  
3.[pg-pong.py作者博客](https://karpathy.github.io/2016/05/31/rl/)