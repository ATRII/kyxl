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
        self.conv2 = nn.Sequential(
            nn.Conv2d(5, 5, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(5, 1, 3, 2, 1),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(625, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(124, 2),
            nn.Tanh()
        )

    def forward(self, obs, info):
        x = self.res1(obs)+obs
        x = self.conv1(x)  # 5*100*100
        x = self.res2(x) + x  # 5*100*100
        x = self.conv2(x)
        x = self.fc1(torch.flatten(x, 1, -1))
        y = torch.repeat_interleave(info, 20, dim=1)  # 60
        x = torch.cat((torch.flatten(x, 1, -1),
                      torch.flatten(y, 1, -1)), dim=1)  # 124
        x = self.fc2(x)
        return x


class MADDPG:
    def __init__(self, n_agents, obs_dims, act_dims, cuda_use):
        self.n_agents = n_agents
        self.agents = [None] * n_agents
        self.cuda_use = cuda_use
        self.action_space = act_dims
        self.gamma = 0.95
        self.train_cnt = 0
        self.gpu_enable = torch.cuda.is_available()
        # print(self.gpu_enable)
        self.maptensor = torch.tensor([180., 12.5])

        for i in range(n_agents):
            self.agents[i] = Actor(obs_dims)
        if self.cuda_use and self.gpu_enable:
            print('GPU Available!!')
            checkpoint = torch.load('model/maddpg/model_0001000.pkl')
            for i in range(n_agents):
                self.agents[i] = self.agents[i].cuda()
                self.agents[i].load_state_dict(checkpoint[i])
        else:
            checkpoint = torch.load(
                'model/maddpg/model_0001000.pkl', map_location=torch.device('cpu'))
            for i in range(n_agents):
                self.agents[i].load_state_dict(checkpoint[i])

    def select_actions(self, i, obs_s, obs_i):
        with torch.no_grad():
            ep = np.random.random()
            actions = torch.tensor(np.random.random(
                self.action_space)) * self.maptensor * 2
            obs_s_t = torch.tensor(obs_s).float().unsqueeze(0)
            obs_i_t = torch.tensor(obs_i).float().unsqueeze(0)
            if self.cuda_use and self.gpu_enable:
                obs_s_t = obs_s_t.cuda()
                obs_i_t = obs_i_t.cuda()
            action_tensor = (self.agents[i](
                obs_s_t, obs_i_t) + 1) * self.maptensor
            actions = action_tensor.cpu().detach().numpy()
            actions = np.squeeze(actions)

        return np.array(actions)
