import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
ACTION_MOVE = 2
UPDATE_INTERVAL = 200


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
        # print("x.shape1: ", x.shape)
        # x = self.fc1(x.reshape(-1))  # 64
        # print("x.shape1: ", torch.flatten(x).shape)
        x = self.fc1(torch.flatten(x, 1, -1))
        # print("x.shape2: ", x.shape)
        # print("y.shape1: ", info.shape)
        y = torch.repeat_interleave(info, 20, dim=1)  # 60
        # print("y.shape2: ", y.shape)
        x = torch.cat((torch.flatten(x, 1, -1),
                      torch.flatten(y, 1, -1)), dim=1)  # 124
        x = self.fc2(x)
        return x


class Critic(nn.Module):
    """
    `input`: observation + action

    `output`: Q
    """

    def __init__(self, obs_dim: list):
        super(Critic, self).__init__()
        if len(obs_dim) < 3:
            print("wrong obs_dim")
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
            nn.Linear(184, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, obs: torch.Tensor, info: torch.Tensor, act: torch.Tensor):
        x = self.res1(obs) + obs
        x = self.conv1(x)  # 5 * 100 * 100
        x = self.res2(x) + x  # 5 * 100 * 100
        x = self.conv2(x)
        x = self.fc1(torch.flatten(x, 1, -1))  # 64
        y = torch.repeat_interleave(info, 20, dim=1)  # 60
        x = torch.cat((torch.flatten(x, 1, -1),
                      torch.flatten(y, 1, -1)), dim=1)  # 124

        # print("action shape: ", act.shape)
        z = torch.repeat_interleave(torch.flatten(act, 1, -1), 3, dim=1)  # 60
        x = torch.cat((torch.flatten(x, 1, -1), z
                       ), dim=1)  # 184
        x = self.fc2(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    # 随机采样经验，并可选地将tensor放到GPU上
    def sample(self, batch_size, cuda_use=False):
        if len(self.memory) < batch_size:
            samples = self.memory
        else:
            samples = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states = map(
            np.asarray, zip(*samples))

        # states_screen = [i['screen'] for i in states]
        states_screen = [[i['screen'] for i in j] for j in states]
        states_info = [[i['info'] for i in j] for j in states]
        next_states_screen = [[i['screen'] for i in j] for j in next_states]
        next_states_info = [[i['info'] for i in j] for j in next_states]

        states_screen_t = torch.tensor(np.array(states_screen))
        # print("states_screen_t shape: ", states_screen_t.shape)
        states_info_t = torch.tensor(np.array(states_info))
        actions_t = torch.tensor(np.array(actions))
        # print("actions_t shape: ", actions_t.shape)
        rewards_t = torch.tensor(np.array(rewards))
        # next_screen_t = torch.from_numpy(
        #     np.vstack(next_states_screen))
        next_screen_t = torch.tensor(np.array(next_states_screen))
        next_info_t = torch.tensor(np.array(next_states_info))

        if cuda_use:
            states_screen_t = states_screen_t.cuda()
            states_info_t = states_info_t.cuda()
            actions_t = actions_t.cuda()
            rewards_t = rewards_t.cuda()
            next_screen_t = next_screen_t.cuda()
            next_info_t = next_info_t.cuda()
        return states_screen_t, states_info_t, actions_t, rewards_t, next_screen_t, next_info_t

    def __len__(self):
        return len(self.memory)


class MADDPG:
    def __init__(self, n_agents, obs_dims, act_dims, cuda_use, capacity, updmode):
        self.n_agents = n_agents
        self.agents = [None] * n_agents
        self.cuda_use = cuda_use
        self.memory_buff = ReplayBuffer(capacity)
        self.action_space = act_dims
        self.epsilon = 0.9
        self.gamma = 0.95
        self.train_cnt = 0
        self.updmode = updmode
        self.maptensor = torch.tensor([180., 12.5]).cuda()
        self.replace_target_iter = 1000
        self.learn_step_counter = 0
        self.losslist = []
        for i in range(n_agents):
            self.losslist.append([])
        # 创建每个无人机的Actor和Critic网络
        for i in range(n_agents):
            self.agents[i] = {
                'actor': Actor(obs_dims),
                'critic': Critic(obs_dims)
            }
        if self.cuda_use:
            for a in self.agents:
                a['actor'] = a['actor'].cuda()
                a['critic'] = a['critic'].cuda()
        # 定义优化器
        self.actor_optimizer = [torch.optim.Adam(
            agent['actor'].parameters(), lr=0.001) for agent in self.agents]
        self.critic_optimizer = [torch.optim.Adam(
            agent['critic'].parameters(), lr=0.001) for agent in self.agents]

        # 定义目标Actor和Critic网络
        self.target_agents = [
            {
                'actor': Actor(obs_dims),
                'critic': Critic(obs_dims)
            }
            for _ in range(n_agents)]
        for i in range(n_agents):
            self.target_agents[i]['actor'].load_state_dict(
                self.agents[i]['actor'].state_dict())
            self.target_agents[i]['critic'].load_state_dict(
                self.agents[i]['critic'].state_dict())

        if self.cuda_use:
            for d in self.target_agents:
                d['actor'] = d['actor'].cuda()
                d['critic'] = d['critic'].cuda()

    def sample(self, batch_size):
        states_screen_t, states_info_t, actions_t, rewards_t, next_screen_t, next_info_t = self.memory_buff.sample(
            batch_size, self.cuda_use)
        return (states_screen_t, states_info_t, actions_t, rewards_t, next_screen_t, next_info_t)

    def step_target(self, mode: str = "hard"):
        assert mode == "soft" or mode == "hard"
        if mode == "soft":
            for i in range(self.n_agents):
                for target_param, param in zip(self.target_agents[i]['actor'].parameters(),
                                               self.agents[i]['actor'].parameters()):
                    target_param.data.copy_(
                        self.gamma * target_param.data + (1 - self.gamma) * param.data)
                for target_param, param in zip(self.target_agents[i]['critic'].parameters(),
                                               self.agents[i]['critic'].parameters()):
                    target_param.data.copy_(
                        self.gamma * target_param.data + (1 - self.gamma) * param.data)
        if mode == "hard":
            for i in range(self.n_agents):
                for target_param, param in zip(self.target_agents[i]['actor'].parameters(),
                                               self.agents[i]['actor'].parameters()):
                    target_param.data.copy_(param.data)
                for target_param, param in zip(self.target_agents[i]['critic'].parameters(),
                                               self.agents[i]['critic'].parameters()):
                    target_param.data.copy_(param.data)

    def learn(self, batch_size):
        t0 = time.time()
        print("train cnt: ", self.learn_step_counter)
        if self.learn_step_counter % self.replace_target_iter == 0:
            step_counter_str = '%08d' % self.learn_step_counter
            torch.save({i: self.agents[i]['actor'].state_dict(
            ) for i in range(self.n_agents)}, 'model/maddpg/model_' + step_counter_str + '.pkl')
        self.learn_step_counter += 1
        obs_s_t, obs_i_t, act_t, rew_t, next_obs_s_t, next_obs_i_t = self.sample(
            batch_size)
        obs_s_t, obs_i_t, act_t, rew_t, next_obs_s_t, next_obs_i_t = obs_s_t.float(
        ), obs_i_t.float(), act_t.float(), rew_t.float(), next_obs_s_t.float(), next_obs_i_t.float()
        target_act_next_n = []
        for i in range(self.n_agents):
            target_act_next = self.target_agents[i]['actor'](
                next_obs_s_t[:, i], next_obs_i_t[:, i])
            target_act_next = target_act_next.cpu().detach().numpy()
            target_act_next_n.append(target_act_next)
        target_act_next_n = np.array(target_act_next_n)
        target_act_next_n = target_act_next_n.reshape((batch_size, -1))
        target_act_next_t = torch.from_numpy(target_act_next_n)
        # print("target_act_next_t shape: {}".format(target_act_next_t.shape))
        if self.cuda_use:
            target_act_next_t = target_act_next_t.cuda()
        target_q_n = []
        for i in range(self.n_agents):
            target_q = self.target_agents[i]['critic'](
                next_obs_s_t[:, i], next_obs_i_t[:, i], target_act_next_t)
            # print("CARE: ", target_q.shape)
            target_q = rew_t[:, i].reshape((-1, 1)).cpu().detach().numpy() + \
                self.gamma * target_q.cpu().detach().numpy()
            target_q_n.append(target_q)
        target_q_n = np.array(target_q_n)
        target_q_t = torch.tensor(target_q_n)
        if self.cuda_use:
            target_q_t = target_q_t.cuda()
        # print("target_q_t shape:", target_q_t.shape)
        act = act_t.reshape(batch_size, -1).cuda()
        # 更新Critic网络
        for i in range(self.n_agents):
            q = self.agents[i]['critic'](
                obs_s_t[:, i], obs_i_t[:, i], act)
            critic_loss = F.mse_loss(q, target_q_t[i])
            self.critic_optimizer[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizer[i].step()
        # Actor
        # for i in range(self.n_agents):
            # act_pred = self.agents[i]['actor'](obs_s_t, obs_i_t)
            # act_tt = []
            # for j in range(self.n_agents):
            #     # if j != i:
            #     act_j = self.agents[j]['actor'](obs_s_t[:, i], obs_i_t[:, i])
            #     act_tt.append(act_j)
            # act_tt = torch.cat(act_tt, dim=1).cuda()
            # print("act_tt.shape: ", act_tt.shape)
            # q = self.agents[i]['critic'](
            #     obs_s_t[:, i], obs_i_t[:, i], act_tt)
            q = self.agents[i]['critic'](
                obs_s_t[:, i], obs_i_t[:, i], act)
            actor_loss = -q.mean()
            actor_loss_scalar = actor_loss.item()
            # print("actor_loss", actor_loss_scalar)
            self.losslist[i].append(actor_loss_scalar)
            self.actor_optimizer[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizer[i].step()
        if self.updmode == "hard" and self.train_cnt % self.replace_target_iter != 0:
            return
        self.step_target(self.updmode)
    # obs_s: 10*5*100*100 obs_i:10*3
        t1 = time.time()-t0
        print("time: ", t1)

    def select_actions(self, ith, obs_s, obs_i, use_noise=False, epsilon=0.0):
        with torch.no_grad():
            ep = np.random.random()
            actions = torch.tensor(np.random.random(
                self.action_space)).cuda() * self.maptensor * 2
            if ep > epsilon:
                obs_s_t = torch.tensor(obs_s).float()
                obs_i_t = torch.tensor(obs_i).float()
                if use_noise:
                    noise = 0.4 * \
                        torch.tensor(np.random.random(len(actions))).cuda()
                    actions += noise
                if self.cuda_use:
                    obs_s_t = obs_s_t.cuda()
                    obs_i_t = obs_i_t.cuda()
                obs_s_t = obs_s_t.unsqueeze(0)
                obs_i_t = obs_i_t.unsqueeze(0)
                action_tensor = (self.agents[ith]['actor'](
                    obs_s_t, obs_i_t)+1.)*self.maptensor
                # print(action_tensor)
                actions = action_tensor.cpu().detach().numpy()
                actions = np.squeeze(actions)
                # print(type(actions))
                return actions
            else:
                return actions.cpu().detach().numpy()
