import copy
import numpy as np
from agent.fix_rule.agent import Agent
from interface import Environment
from train.maddpg import maddpg

MAP_PATH = 'maps/1000_1000_fighter10v10.map'
RENDER = True
MAX_EPOCH = 2000
BATCH_SIZE = 128
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
DETECTOR_NUM = 0
FIGHTER_NUM = 10
LEARN_INTERVAL = 100
MAX_STEP = 600

if __name__ == "__main__":
    rwd_list = []
    # create blue agent
    blue_agent = Agent()
    # get agent obs type
    red_agent_obs_ind = 'maddpg'
    blue_agent_obs_ind = blue_agent.get_obs_ind()
    # make env
    env = Environment(MAP_PATH, red_agent_obs_ind,
                      blue_agent_obs_ind, render=RENDER)
    # get map info
    size_x, size_y = env.get_map_size()
    red_detector_num, red_fighter_num, blue_detector_num, blue_fighter_num = env.get_unit_num()
    # set map info to blue agent
    blue_agent.set_map_info(
        size_x, size_y, blue_detector_num, blue_fighter_num)

    red_detector_action = []
    fighter_model = maddpg.MADDPG(
        FIGHTER_NUM, [5, 100, 100], 4, True, 100000, "hard")
    total_cnt = 0  # not related to epoch
    mean_rwd = 0
    tt_rwd = 0
    eps = 1
    for x in range(MAX_EPOCH):
        step_cnt = 0
        tt_rwd = 0
        env.reset()
        cur_step = 0
        while True:
            cur_step += 1
            if cur_step > MAX_STEP:
                break
            eps = np.exp(-total_cnt/2000)
            if total_cnt % 100 == 0:
                loss_list = np.array(fighter_model.losslist)
                np.save("./loss.npy", loss_list)
            obs_list = []
            action_list = []
            red_fighter_action = []
            # get obs
            if step_cnt == 0:
                red_obs_dict, blue_obs_dict = env.get_obs()
            # get action
            # get blue action
            blue_detector_action, blue_fighter_action = blue_agent.get_action(
                blue_obs_dict, step_cnt)
            # get red action
            obs_got_ind = [False] * red_fighter_num
            for y in range(red_fighter_num):
                true_action = np.array([0, 1, 0, 0], dtype=np.int32)
                if red_obs_dict['fighter'][y]['alive']:
                    obs_got_ind[y] = True
                    tmp_img_obs = red_obs_dict['fighter'][y]['screen']
                    tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)
                    tmp_info_obs = red_obs_dict['fighter'][y]['info']
                    # print(tmp_img_obs.dtype)
                    if (total_cnt < 100000):
                        tmp_action = fighter_model.select_actions(
                            y, tmp_img_obs, tmp_info_obs, True, eps)
                    else:
                        tmp_action = fighter_model.select_actions(
                            y, tmp_img_obs, tmp_info_obs, False, eps)
                    obs_list.append({'screen': copy.deepcopy(
                        tmp_img_obs), 'info': copy.deepcopy(tmp_info_obs)})
                    action_list.append(tmp_action)
                    # print("tmp_action.shape: ", tmp_action.shape)
                    # action formation
                    # TODO:TEST1
                    true_action[0] = np.floor(tmp_action[0])
                    true_action[1] = np.floor(tmp_action[1])
                    true_action[2] = np.floor(tmp_action[2])
                    true_action[3] = np.floor(tmp_action[3])
                else:
                    empty_obs = {'screen': np.zeros(
                        [5, 100, 100]), 'info': np.zeros(3)}
                    obs_list.append(empty_obs)
                    action_list.append(np.array([-1, -1, -1, -1]))
                red_fighter_action.append(true_action)
            red_fighter_action = np.array(red_fighter_action)
            # step
            env.step(red_detector_action, red_fighter_action,
                     blue_detector_action, blue_fighter_action)
            # get reward
            red_detector_reward, red_fighter_reward, red_game_reward, blue_detector_reward, blue_fighter_reward, blue_game_reward = env.get_reward()
            detector_reward = red_detector_reward + red_game_reward
            fighter_reward = red_fighter_reward + red_game_reward
            m_rwd = np.mean(fighter_reward)
            # print("mean reward: ", m_rwd)
            tt_rwd += m_rwd
            # save replay
            red_obs_dict, blue_obs_dict = env.get_obs()
            tmp_obs_list = []
            for y in range(red_fighter_num):
                # if obs_got_ind[y]:
                if red_obs_dict['fighter'][y]['alive']:
                    tmp_img_obs = red_obs_dict['fighter'][y]['screen']
                    tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)
                    tmp_info_obs = red_obs_dict['fighter'][y]['info']
                    tmp_obs_list.append({'screen': copy.deepcopy(
                        tmp_img_obs), 'info': copy.deepcopy(tmp_info_obs)})
                else:
                    empty_obs = {'screen': np.zeros(
                        [5, 100, 100]), 'info': np.zeros(3)}
                    tmp_obs_list.append(empty_obs)
            # 10*{5*100*100, 3}, 10*2, 10, 10*{5*100*100, 3}
            fighter_model.memory_buff.push(obs_list, np.array(action_list), fighter_reward,
                                           tmp_obs_list)

            # if done, perform a learn
            if env.get_done():
                # detector_model.learn()
                fighter_model.learn(BATCH_SIZE)
                break
            # if not done learn when learn interval
            if (total_cnt > 500) and (step_cnt % LEARN_INTERVAL == 0):
                # detector_model.learn()
                fighter_model.learn(BATCH_SIZE)
            step_cnt += 1
            total_cnt += 1
        mean_rwd = tt_rwd/step_cnt
        rwd_list.append(mean_rwd)
        rwd_n = np.array(rwd_list)
        if x % 20 == 0:
            np.save("./rwd.npy", rwd_n)
        # print("mean_rwd: ", mean_rwd)
        # writer.add_scalar('mean_reward', mean_rwd,
        #                   global_step=None, walltime=None)
