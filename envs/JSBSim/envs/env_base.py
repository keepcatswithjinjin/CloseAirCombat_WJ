import os
import random
import sys

import gymnasium
from gymnasium.utils import seeding
import numpy as np
from typing import Dict, Any, Tuple, Union, List

from numpy import ndarray

from ..core.simulatior import AircraftSimulator, BaseSimulator
from ..tasks.task_base import BaseTask
from ..utils.utils import parse_config
from algorithms.SARS.rff_kernel_density import rff_kernel_density
from algorithms.SARS.augment_observations_with_noise import augment_observations_with_noise

class BaseEnv(gymnasium.Env):
    """
    A class wrapping the JSBSim flight dynamics module (FDM) for simulating
    aircraft as an RL environment conforming to the OpenAI Gym Env
    interface.

    An BaseEnv is instantiated with a Task that implements a specific
    aircraft control task with its own specific observation/action space and
    variables and agent_reward calculation.
    """
    metadata = {"render.modes": ["human", "txt"]}

    def __init__(self, config_name: str):
        # basic args
        self.config = parse_config(config_name)  # 从yaml中获取参数
        self.max_steps = getattr(self.config, 'max_steps', 100)  # type: int
        self.sim_freq = getattr(self.config, 'sim_freq', 60)  # type: int
        self.agent_interaction_steps = getattr(self.config, 'agent_interaction_steps', 12)  # type: int
        self.center_lon, self.center_lat, self.center_alt = \
            getattr(self.config, 'battle_field_center', (120.0, 60.0, 0.0))
        self._create_records = False
        # TODO new
        self.num_agent = 1
        self.Ds_buffer = [[] for _ in range(self.num_agent)]
        self.Df_buffer = [[] for _ in range(self.num_agent)]
        self.temp_buffer = [[] for _ in range(self.num_agent)]
        self.noisy_trajectories = []

        self.M = 500  # RFF 随机特征维数M  : 50会降低性能  应取500 1000 2000
        self.sigma = 0.2  # RFF 高斯核带宽 h
        self.N = 0  # 总计数
        self.a = 0.4  # 成型奖励的权重 0.4 0.6 0.8
        self.retention_rate = 0.6
        self.load()

    @property
    def num_agents(self) -> int:
        return self.task.num_agents

    @property
    def observation_space(self) -> gymnasium.Space:
        return self.task.observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        return self.task.action_space

    @property
    def agents(self) -> Dict[str, AircraftSimulator]:
        return self._jsbsims

    @property
    def time_interval(self) -> int:
        return self.agent_interaction_steps / self.sim_freq

    def load(self):
        self.load_task()
        self.load_simulator()
        self.seed()

    def load_task(self):
        self.task = BaseTask(self.config)

    def load_simulator(self):
        self._jsbsims = {}  # type: Dict[str, AircraftSimulator]
        for uid, config in self.config.aircraft_configs.items():
            self._jsbsims[uid] = AircraftSimulator(
                uid=uid,
                color=config.get("color", "Red"),
                model=config.get("model", "f16"),
                init_state=config.get("init_state"),
                origin=getattr(self.config, 'battle_field_center', (120.0, 60.0, 0.0)),
                sim_freq=self.sim_freq,
                num_missiles=config.get("missile", 0))
        # Different teams have different uid[0]
        _default_team_uid = list(self._jsbsims.keys())[0][0]
        self.ego_ids = [uid for uid in self._jsbsims.keys() if uid[0] == _default_team_uid]
        self.enm_ids = [uid for uid in self._jsbsims.keys() if uid[0] != _default_team_uid]

        # Link jsbsims
        for key, sim in self._jsbsims.items():
            for k, s in self._jsbsims.items():
                if k == key:
                    pass
                elif k[0] == key[0]:
                    sim.partners.append(s)
                else:
                    sim.enemies.append(s)

        self._tempsims = {}  # type: Dict[str, BaseSimulator]

    def add_temp_simulator(self, sim: BaseSimulator):
        self._tempsims[sim.uid] = sim

    def reset(self) -> np.ndarray:
        """Resets the state of the environment and returns an initial observation.

        Returns:
            obs (np.ndarray): initial observation
        """
        # reset sim
        self.current_step = 0
        for sim in self._jsbsims.values():
            sim.reload()
        self._tempsims.clear()
        # reset task
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs)

    def step(self, action: np.ndarray) -> Tuple[
        ndarray, ndarray, ndarray, Union[Dict[str, int], dict]]:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's observation. Accepts an action and
        returns a tuple (observation, reward_visualize, done, info).

        Args:
            action (np.ndarray): the agents' actions, allow opponent's action input

        Returns:
            (tuple):
                obs: agents' observation of the current environment
                rewards: amount of rewards returned after previous actions
                dones: whether the episode has ended, in which case further step() calls are undefined
                info: auxiliary information
        """
        self.current_step += 1
        info = {"current_step": self.current_step}
        # apply actions
        action = self._unpack(action)
        # print("actio_unpack", action, flush=True)
        for agent_id in self.agents.keys():
            # 返回下层控制动作
            a_action = self.task.normalize_action(self, agent_id, action[agent_id])
            # print("a_action", a_action, flush=True)
            self.agents[agent_id].set_property_values(self.task.action_var, a_action)
        # run simulation
        for _ in range(self.agent_interaction_steps):
            for sim in self._jsbsims.values():  # aircraft simulation
                sim.run()
            for sim in self._tempsims.values():  # missile simulation
                sim.run()
        # 不同任务中对交互后的agent的状态数据进行处理  包括导弹
        self.task.step(self)

        obs = self.get_obs()

        dones = {}
        for agent_id in self.agents.keys():
            done, info = self.task.get_termination(self, agent_id, info)
            dones[agent_id] = [done]

        rewards = {}
        for agent_id in self.agents.keys():
            reward, info = self.task.get_reward(self, agent_id, info)
            rewards[agent_id] = [reward]

        # return self._pack(obs), self._pack(rewards), self._pack(dones), info
        # print("rewards: ", rewards)   # rewards:  {'A0100': [9.143880041153126e-58]}
        # print("rewards: ", rewards["A0100"])  #rewards:  [1.6445475577171327e-60]
        # print("obs: ", obs)  #  {'A0100': array([ 1.40549983e-04, -3.31629529e-03, -1.20981263e-03,  1.64144502e+00,
        # 7.07679238e-02,  9.97492807e-01,  8.94750118e-03,  9.99959970e-01,
        # 7.29377415e-01, -1.44880734e-03,  1.13319705e-02,  4.96129541e-01])}

        # print("dones: ", dones)    # dones:  {'A0100': [False]}
        # print("info : ", info)  # info :  {'current_step': 1, 'termination': 64}
        # """
        # TODO new
        # obs_pack = self._pack(obs)
        # reward_pack = self._pack(rewards)
        for agent_idx in range(self.num_agents):  # 遍历每个代理
            agent_obs = obs["A0100"]
            self.temp_buffer[agent_idx].append(agent_obs)
            if 'termination' in info:  # 同一个环境终止条件是同一个
                # 拿到终止条件
                termination = info['termination']
                # 判断6个终止条件 达到任何一个条件  trajectory形成
                termination_flag = ((termination & (1 | 2 | 4 | 8 | 16 | 32 | 64)) != 0)  # 任意一位被置位
                if termination_flag:
                    unreach_heading_flag = (termination & 1)
                    # extreme_state_flag = ((termination >> 1) & 1)
                    # overload_flag = ((termination >> 2) & 1)
                    low_altitude_flag = ((termination >> 3) & 1)
                    # timeout_flag = ((termination >> 4) & 1)
                    # safe_return_flag = ((termination >> 5) & 1)
                    # reach_heading_flag = ((termination >> 6) & 1)
                    # deltaKeeping_flag = ((termination >> 7) & 1)

                    # 成功率
                    # if reach_heading_flag:
                    #     self.Ds_buffer[agent_idx].extend(self.temp_buffer[agent_idx])
                    # else:
                    #     self.Df_buffer[agent_idx].extend(self.temp_buffer[agent_idx])

                    #保留率
                    if random.random() < self.retention_rate:
                        # 失败率
                        if unreach_heading_flag | low_altitude_flag:
                            self.Df_buffer[agent_idx].extend(self.temp_buffer[agent_idx])
                        # elif reach_heading_flag:
                        #     # print(" Enforcement!", flush=True)
                        #     # TODO 样本增强
                        #     noisy_trajectories = self.augment_observations_with_noise(self.temp_buffer[agent_idx])
                        #     flattened_trajectories = np.vstack(noisy_trajectories)  # 100个轨迹（n * 12）
                        #     self.Ds_buffer[agent_idx].extend(flattened_trajectories)
                        #     self.Ds_buffer[agent_idx].extend(self.temp_buffer[agent_idx])
                        #     # self.Ds_buffer[agent_idx].extend(self.temp_buffer[agent_idx])
                        else:
                            self.Ds_buffer[agent_idx].extend(self.temp_buffer[agent_idx])
                        self.temp_buffer[agent_idx].clear()  # 清空对应暂存buffer
                    else:
                        self.temp_buffer[agent_idx].clear()  # 清空对应暂存buffer


                    rs_i, Ns_i_count, Nf_i_count = self.calculate_additional_reward(agent_obs,
                                                                                    self.Ds_buffer[agent_idx],
                                                                                    self.Df_buffer[agent_idx], self.M,
                                                                                    self.sigma)
                    if 'Ns_count' not in info:
                        info['Ns_count'] = 0
                    if 'Nf_count' not in info:
                        info['Nf_count'] = 0
                    if 'beta' not in info:
                        info['beta'] = 0

                    info['Ns_count'] = Ns_i_count
                    info['Nf_count'] = Nf_i_count
                    info['beta'] = rs_i
                    # print("Ns_i_count", Ns_i_count, flush=True)
                    # print("Nf_i_count", Nf_i_count, flush=True)
                    # print("rs_i : {}".format(rs_i), flush=True)
                    rewards["A0100"][0] = rewards["A0100"][0] - self.a * rs_i * 10  # √
                    # print("fix_rewards", rewards)
                    # print(info)
        # """
        return self._pack(obs), self._pack(rewards), self._pack(dones), info

    def augment_observations_with_noise(self, observations, noise_std=0.01, num_copies=30):
        """
        为轨迹中的每个观察值添加噪声，生成多条增强轨迹。

        参数:
        - observations: 原始轨迹中的观察值列表，每个元素是一个观察值（例如numpy数组）
        - noise_std: 噪声的标准差，控制噪声幅度
        - num_copies: 生成的增强轨迹数量

        返回:
        - augmented_trajectories: 增强后的轨迹列表，每条轨迹是一个带噪声的观察值序列
        """
        augmented_trajectories = []

        for _ in range(num_copies):
            noisy_trajectory = [
                obs + np.random.normal(0, noise_std, size=np.shape(obs)) for obs in observations
            ]
            augmented_trajectories.append(noisy_trajectory)

        return augmented_trajectories




    def calculate_additional_reward(self, obs, Ds_buffer, Df_buffer, M, sigma=0.2):
        # Ns_i_density 和 Nf_i_density 的计算
        # if len(Ds_buffer) > 0:
        Ns_i_density = rff_kernel_density(obs, Ds_buffer, M, sigma)
        # if len(Ds_buffer) > 0:
        Nf_i_density = rff_kernel_density(obs, Df_buffer, M, sigma)
        total_success = len(Ds_buffer)
        total_failure = len(Df_buffer)
        N = total_success + total_failure
        Ns_i_count = N * Ns_i_density
        Nf_i_count = N * Nf_i_density
        alpha = Ns_i_count + 1
        beta = Nf_i_count + 1

        # return np.random.beta(alpha, beta), total_success, total_failure  # 成功率
        return np.random.beta(beta, alpha), total_success, total_failure   # 失败率

    def clear_buffers(self):
        self.Ds_buffer.clear()
        self.Df_buffer.clear()


    def get_obs(self):
        """Returns all agent observations in a list.

        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        return dict([(agent_id, self.task.get_obs(self, agent_id)) for agent_id in self.agents.keys()])

    def get_state(self):
        """Returns the global state.

        NOTE: This functon should not be used during decentralised execution.
        """
        state = np.hstack([self.task.get_obs(self, agent_id) for agent_id in self.agents.keys()])
        return dict([(agent_id, state.copy()) for agent_id in self.agents.keys()])

    def close(self):
        """Cleans up this environment's objects

        NOTE: Environments automatically close() when garbage collected or when the
        program exits.
        """
        for sim in self._jsbsims.values():
            sim.close()
        for sim in self._tempsims.values():
            sim.close()
        self._jsbsims.clear()
        self._tempsims.clear()

    def render(self, mode="txt", filepath='./JSBSimRecording.txt.acmi'):
        """Renders the environment.

        The set of supported modes varies per environment. (And some

        environments do not support rendering at all.) By convention,

        if mode is:

        - human: print on the terminal
        - txt: output to txt.acmi files

        Note:

            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        :param mode: str, the mode to render with
        """
        if mode == "txt":
            if not self._create_records:
                with open(filepath, mode='w', encoding='utf-8-sig') as f:
                    f.write("FileType=text/acmi/tacview\n")
                    f.write("FileVersion=2.1\n")
                    f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")
                self._create_records = True
            with open(filepath, mode='a', encoding='utf-8-sig') as f:
                timestamp = self.current_step * self.time_interval
                f.write(f"#{timestamp:.2f}\n")
                for sim in self._jsbsims.values():
                    log_msg = sim.log()
                    if log_msg is not None:
                        f.write(log_msg + "\n")
                for sim in self._tempsims.values():
                    log_msg = sim.log()
                    if log_msg is not None:
                        f.write(log_msg + "\n")
        # TODO: real time rendering [Use FlightGear, etc.]
        else:
            raise NotImplementedError

    def seed(self, seed=None):
        """
        Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _pack(self, data: Dict[str, Any]) -> np.ndarray:
        """Pack seperated key-value dict into grouped np.ndarray"""
        ego_data = np.array([data[uid] for uid in self.ego_ids])
        enm_data = np.array([data[uid] for uid in self.enm_ids])
        if enm_data.shape[0] > 0:
            data = np.concatenate((ego_data, enm_data))  # type: np.ndarray
        else:
            data = ego_data  # type: np.ndarray
        try:
            assert np.isnan(data).sum() == 0
        except AssertionError:
            import pdb
            pdb.set_trace()
        # only return data that belongs to RL agents
        return data[:self.num_agents, ...]

    def _unpack(self, data: np.ndarray) -> Dict[str, Any]:
        """Unpack grouped np.ndarray into seperated key-value dict"""
        assert isinstance(data, (np.ndarray, list, tuple)) and len(data) == self.num_agents
        # unpack data in the same order to packing process
        unpack_data = dict(zip((self.ego_ids + self.enm_ids)[:self.num_agents], data))
        # fill in None for other not-RL agents
        for agent_id in (self.ego_ids + self.enm_ids)[self.num_agents:]:
            unpack_data[agent_id] = None
        return unpack_data

    @num_agents.setter
    def num_agents(self, value):
        self._num_agents = value
