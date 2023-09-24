import torch
import torch.nn as nn
import numpy as np


class Actor(nn.Module):
    """
    `input`: observation

    `output`: action
    """

    def __init__(self, obs_dim: list):
        super(Actor, self).__init__()

        if len(obs_dim) < 3:
            print("wrong obs_dim with the value {}".format(obs_dim))
            exit(1)
        # 5* 100 * 100 (obs_dim[0], obs_dim[1], obs_dim[2])
        self.res1 = nn.Sequential(
            nn.Conv2d(obs_dim[0], 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 5, 3, 1, 1)
        )
        self.conv1 = nn.Conv2d(5, 5, 1, 1)
        self.res2 = nn.Sequential(
            nn.Conv2d(5, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 5, 3, 1, 1)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(50000, 2048),
            nn.ReLU(),
            nn.Linear(2048, 256)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(496, 2),
            nn.Tanh()
        )

    def forward(self, obs, info):
        x = self.res1(obs)+obs
        x = self.conv1(x)  # 5*100*100
        x = self.res2(x)+x  # 5*100*100
        x = self.fc1(x.reshape(-1))  # 256
        y = torch.repeat_interleave(info, 80)  # 240
        x = torch.cat(x.reshape(-1), y.reshape(-1))  # 496
        x = self.fc2(x)
        return x


class MADDPG:
    def __init__(self, n_agents, obs_dims, act_dims, cuda_use):
        self.n_agents = n_agents
        self.agents = [None] * n_agents
        self.cuda_use = cuda_use
        self.action_space = act_dims[0]
        self.epsilon = 0.9
        self.gamma = 0.95
        self.train_cnt = 0
        self.maptensor = torch.tensor([180., 0., 0., 12.5])
        # 创建每个无人机的Actor和Critic网络
        for i in range(n_agents):
            self.agents[i] = {
                'actor': Actor(obs_dims, act_dims)
            }
        if self.cuda_use:
            for a in self.agents:
                a['actor'] = a['actor'].cuda()

        checkpoint = torch.load('model/maddpg/model.pkl')
        for i in range(n_agents):
            self.agents['actor'].load_state_dict(checkpoint[i])

    def select_actions(self, obs_s, obs_i, use_noise=False, epsilon=0.0):
        all_actions = []  # 所有智能体的动作列表
        with torch.no_grad():
            for i in range(self.n_agents):
                ep = np.random.random()
                actions = torch.tensor(np.random.random(
                    self.action_space)) * self.maptensor*2
                if ep > epsilon:
                    obs_s_t = torch.tensor(obs_s[i])
                    obs_i_t = torch.tensor(obs_i[i])
                    if self.cuda_use:
                        obs_s_t = obs_s_t.cuda()
                        obs_i_t = obs_i_t.cuda()
                    action_tensor = (self.agents[i]['actor'](
                        obs_s_t, obs_i_t)+1)*self.maptensor
                    # print(action_tensor)
                    # 将动作张量转换为 NumPy 数组
                    actions = action_tensor.cpu().detach().numpy()
                    if use_noise:
                        noise = 0.4 * np.random.random(len(actions))
                        actions += noise
                    actions = np.squeeze(actions)
                    # print(actions)
                all_actions.append(actions)  # 添加当前智能体的动作到列表中
        return np.array(all_actions)
