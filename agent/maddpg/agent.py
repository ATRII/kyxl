import os
from agent.base_agent import BaseAgent
from agent.maddpg import maddpg
import interface
from world import config
import copy
import random
import numpy as np

DETECTOR_NUM = 0
FIGHTER_NUM = 10
COURSE_NUM = 16
# long missile attack + short missile attack + no attack
ATTACK_IND_NUM = (DETECTOR_NUM + FIGHTER_NUM) * 2 + 1
ACTION_NUM = COURSE_NUM * ATTACK_IND_NUM


class Agent(BaseAgent):
    def __init__(self):
        """
        Init this agent
        :param size_x: battlefield horizontal size
        :param size_y: battlefield vertical size
        :param detector_num: detector quantity of this side
        :param fighter_num: fighter quantity of this side
        """
        BaseAgent.__init__(self)
        self.obs_ind = 'maddpg'
        if not os.path.exists('model/maddpg/model_0001000.pkl'):
            print('Error: agent maddpg model data not exist!')
            exit(1)
        self.fighter_model = maddpg.MADDPG(10, [5, 100, 100], 2, True)

    def set_map_info(self, size_x, size_y, detector_num, fighter_num):
        self.size_x = size_x
        self.size_y = size_y
        self.detector_num = detector_num
        self.fighter_num = fighter_num

    def __reset(self):
        pass

    def get_action(self, obs_dict, step_cnt):
        """
        get actions
        :param detector_obs_list:
        :param fighter_obs_list:
        :param joint_obs_dict:
        :param step_cnt:
        :return:
        """

        detector_action = []
        fighter_action = []
        for y in range(self.fighter_num):
            true_action = np.array([0, 1, 0, 0], dtype=np.int32)
            if obs_dict['fighter'][y]['alive']:
                true_action = np.array([0, 1, 0, 0], dtype=np.int32)
                tmp_img_obs = obs_dict['fighter'][y]['screen']
                tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)
                tmp_info_obs = obs_dict['fighter'][y]['info']
                tmp_action = self.fighter_model.select_actions(
                    y, tmp_img_obs, tmp_info_obs)
                # action formation
                true_action[0] = np.floor(tmp_action[0])
                true_action[3] = np.floor(tmp_action[1])
            fighter_action.append(copy.deepcopy(true_action))
        fighter_action = np.array(fighter_action)

        return detector_action, fighter_action
