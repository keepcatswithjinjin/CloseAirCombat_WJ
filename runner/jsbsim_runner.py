import sys
import time
import random

import torch
import logging
import numpy as np
from typing import List
from .base_runner import Runner, ReplayBuffer
from algorithms.SARS.rff_mapping import rff_mapping
from algorithms.SARS.log_reward_mapping import log_reward_mapping
from algorithms.SARS.rff_kernel_density import rff_kernel_density
from algorithms.SARS.calculate_additional_reward import calculate_additional_reward
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def _t2n(x):
    return x.detach().cpu().numpy()


class JSBSimRunner(Runner):

    def load(self):
        self.obs_space = self.envs.observation_space
        self.act_space = self.envs.action_space
        self.num_agents = self.envs.num_agents
        self.use_selfplay = self.all_args.use_selfplay

        # policy & algorithm
        if self.algorithm_name == "ppo":
            from algorithms.ppo.ppo_trainer import PPOTrainer as Trainer
            from algorithms.ppo.ppo_policy import PPOPolicy as Policy
        else:
            raise NotImplementedError
        self.policy = Policy(self.all_args, self.obs_space, self.act_space, device=self.device)
        self.trainer = Trainer(self.all_args, device=self.device)

        # buffer
        self.buffer = ReplayBuffer(self.all_args, self.num_agents, self.obs_space, self.act_space)

        if self.model_dir is not None:
            self.restore(self.use_best)

    def run(self):
        self.warmup()  # 重置环境
        start = time.time()
        self.total_num_steps = 0
        episodes = self.num_env_steps // self.buffer_size // self.n_rollout_threads
        for episode in tqdm(range(episodes), desc="total process Episode"):
        # for episode in range(episodes):
            heading_turns_list = []
            # 20240426 add by ash0^0
            end_step_list = []
            termination_list = []
            Ns_count_list = []
            Nf_count_list = []
            beta_list = []
            # 一次循环就是一次交互过程  循环结束后  会出现若干个任务结束的  trajectory : (s,a,r,s1,a1,r1........)
            for step in tqdm(range(self.buffer_size), desc="Collecting information step"):
            # for step in range(self.buffer_size):

                # Sample actions   values: 状态价值函数 s下期望的回报  用于计算优势函数
                # 采样动作用到 obs  更新obs时将obs装进buffer
                values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.collect(step)
                # print("actions : ", actions, flush=True)
                # reward and next obs  reward: 回报  Rt   用于计算优势函数
                obs, rewards, dones, infos = self.envs.step(actions)
                # Extra recorded information  每次循环将该次的info加入列表中 表示不同轨迹下的info 在一个buffer全部循环完之后
                # 再依次判断列表中的不同轨迹下该次的info情况  通过位运算判断标志位
                # 每次循环开始时info中的键值清零  这样才能正确根据状态位来判断
                for info in infos:
                    if 'heading_turn_counts' in info:
                        heading_turns_list.append(info['heading_turn_counts'])
                    if 'end_step' in info:
                        end_step_list.append(info['end_step'])
                    if 'termination' in info:
                        termination_list.append(info['termination'])
                    if 'Ns_count' in info:
                        Ns_count_list.append(info['Ns_count'])
                    if 'Nf_count' in info:
                        Nf_count_list.append(info['Nf_count'])
                    if 'beta' in info:
                        beta_list.append(info['beta'])
                data = obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic
                # insert data into buffer
                # 下一次采样动作时用到buffer中这一次的的obs
                self.insert(data)

            # compute return and update network
            # Inherit from parent class Runner
            self.compute()  # 计算优势函数中的一部分 ： rt + γ * V_θ(S_t+1)
            train_infos = self.train()

            # post process
            self.total_num_steps = (episode + 1) * self.buffer_size * self.n_rollout_threads

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                date_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))
                logging.info(
                    "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {},time {}.\n"
                    .format(self.all_args.scenario_name,
                            self.algorithm_name,
                            self.experiment_name,
                            episode,
                            episodes,
                            self.total_num_steps,
                            self.num_env_steps,
                            int(self.total_num_steps / (end - start)),
                            date_now))

                train_infos["average_episode_rewards"] = self.buffer.rewards.sum() / (self.buffer.masks == False).sum()
                logging.info("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                done_counts = len(end_step_list)
                if len(heading_turns_list):
                    train_infos["average_heading_turns"] = np.mean(heading_turns_list)
                    logging.info("average heading turns is {}".format(train_infos["average_heading_turns"]))
                if len(end_step_list):
                    train_infos["average_end_steps"] = np.mean(end_step_list)
                    logging.info("average end steps is {}".format(train_infos["average_end_steps"]))
                if len(Ns_count_list):
                    train_infos["average_Ns_counts"] = np.mean(Ns_count_list)
                if len(Nf_count_list):
                    train_infos["average_Nf_counts"] = np.mean(Nf_count_list)
                if len(beta_list):
                    train_infos["average_beta_distribution"] = np.mean(beta_list)

                if len(termination_list):
                    # termination = 1 | 2 | 4 | 8 | 16 | 32
                    # unreach_heading | extreme_state | overload | low_altitude | timeout | safe_return
                    unreach_heading_counts = 0
                    extreme_state_counts = 0
                    overload_counts = 0
                    low_altitude_counts = 0
                    timeout_counts = 0
                    safe_return_counts = 0
                    for termination in termination_list:
                        unreach_heading_counts += (termination & 1)
                        extreme_state_counts += ((termination >> 1) & 1)
                        overload_counts += ((termination >> 2) & 1)
                        low_altitude_counts += ((termination >> 3) & 1)
                        timeout_counts += ((termination >> 4) & 1)
                        safe_return_counts += ((termination >> 5) & 1)

                    # if done_counts == 0:
                    #     train_infos["unreach_heading_prop"] = 0
                    #     train_infos["extreme_state_prop"] = 0
                    #     train_infos["overload_prop"] = 0
                    #     train_infos["low_altitude_prop"] = 0
                    #     train_infos["timeout_prop"] = 0
                    #     train_infos["safe_return_prop"] = 0
                    #     train_infos["done_counts"] = done_counts
                    # else:
                    train_infos["unreach_heading_prop"] = unreach_heading_counts / done_counts
                    train_infos["extreme_state_prop"] = extreme_state_counts / done_counts
                    train_infos["overload_prop"] = overload_counts / done_counts
                    train_infos["low_altitude_prop"] = low_altitude_counts / done_counts
                    train_infos["timeout_prop"] = timeout_counts / done_counts
                    train_infos["safe_return_prop"] = safe_return_counts / done_counts
                    train_infos["done_counts"] = done_counts

                    logging.info("{:^20} | {:^20} | {:^20} | {:^20} | {:^20} | {:^20} | {:^20} | {:^20} | {:^20}"
                                 .format("done_counts", "unreach_heading_prop", "extreme_state_prop",
                                         "overload_prop", "low_altitude_prop", "timeout_prop", "average_Ns_counts",
                                         "average_Nf_counts", "average_beta_distribution"))
                    logging.info(
                        "{:^20} | {:^20.4f} | {:^20.4f} | {:^20.4f} | {:^20.4f} | {:^20.4f} | {:^20.4f} | {:^20.4f} | "
                        "{:^20.4f}"
                        .format(train_infos["done_counts"], train_infos["unreach_heading_prop"],
                                train_infos["extreme_state_prop"],
                                train_infos["overload_prop"], train_infos["low_altitude_prop"],
                                train_infos["timeout_prop"], train_infos["average_Ns_counts"],
                                train_infos["average_Nf_counts"], train_infos["average_beta_distribution"]))

                    # logging.info("{:^20} | {:^20} | {:^20} | {:^20} | {:^20} | {:^20}"
                    #              .format("done_counts", "extreme_state_prop",
                    #                      "overload_prop", "low_altitude_prop", "timeout_prop", "safe_return_prop"))
                    # logging.info("{:^20} | {:^20.4f} | {:^20.4f} | {:^20.4f} | {:^20.4f} | {:^20.4f}"
                    #              .format(train_infos["done_counts"],
                    #                      train_infos["extreme_state_prop"],
                    #                      train_infos["overload_prop"], train_infos["low_altitude_prop"],
                    #                      train_infos["timeout_prop"], train_infos["safe_return_prop"]))
                self.log_info(train_infos, self.total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and episode != 0 and self.use_eval:
                self.eval(self.total_num_steps)

            # save model
            if self.save_best:
                average_reward = self.buffer.rewards.sum() / (self.buffer.masks == False).sum()
                if average_reward > self.best_reward:
                    logging.info(
                        "save best model on episode {} : average episode rewards is {}".format(episode, average_reward))
                    self.best_reward = average_reward
                    self.save(episode, True)
            if (episode % self.save_interval == 0) or (episode == episodes - 1):
                self.save(episode)

            # clear buffer
            # if episode % 2 == 0:
            self.envs.clear_buffers()







    # def calculate_additional_reward_(self, obs, Ds_buffer, Df_buffer, M=500, sigma=0.2):
    #     beta_values = []
    #     # 获取当前环境的obs
    #     for i in range(len(Ds_buffer)):
    #         # 获取当前环境的obs
    #         current_obs = obs[i][0]
    #         # 对每个环境的 Ds_buffer 和 Df_buffer 分别进行计算
    #         beta, _, _ = self.calculate_additional_reward(current_obs, Ds_buffer[i], Df_buffer[i], M=M, sigma=sigma)
    #         beta_values.append(beta)
    #     return beta_values

    # def calculate_additional_reward(self, obs, Ds_buffer, Df_buffer, M=500, sigma=0.2):
    #     # Ns_i_density 和 Nf_i_density 的计算
    #     # if len(Ds_buffer) > 0:
    #     Ns_i_density = rff_kernel_density(obs, Ds_buffer, M, sigma)
    #     # if len(Ds_buffer) > 0:
    #     Nf_i_density = rff_kernel_density(obs, Df_buffer, M, sigma)
    #     total_success = len(Ds_buffer)
    #     total_failure = len(Df_buffer)
    #     N = total_success + total_failure
    #     Ns_i_count = N * Ns_i_density
    #     Nf_i_count = N * Nf_i_density
    #     alpha = Ns_i_count + 1
    #     beta = Nf_i_count + 1
    #     return np.random.beta(alpha, beta), total_success, total_failure

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        self.buffer.step = 0
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.policy.prep_rollout()
        values, actions, action_log_probs, rnn_states_actor, rnn_states_critic \
            = self.policy.get_actions(np.concatenate(self.buffer.obs[step]),
                                      np.concatenate(self.buffer.rnn_states_actor[step]),
                                      np.concatenate(self.buffer.rnn_states_critic[step]),
                                      np.concatenate(self.buffer.masks[step]))
        # split parallel data [N*M, shape] => [N, M, shape]
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states_actor = np.array(np.split(_t2n(rnn_states_actor), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def insert(self, data: List[np.ndarray]):
        obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic = data

        dones_env = np.all(dones.squeeze(axis=-1), axis=-1)

        rnn_states_actor[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_actor.shape[1:]),
                                                       dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_critic.shape[1:]),
                                                        dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        self.buffer.insert(obs, actions, rewards, masks, action_log_probs, values, rnn_states_actor, rnn_states_critic)

    @torch.no_grad()
    def eval(self, total_num_steps):
        logging.info("\nStart evaluation...")
        total_episodes, eval_episode_rewards = 0, []
        eval_cumulative_rewards = np.zeros((self.n_eval_rollout_threads, *self.buffer.rewards.shape[2:]),
                                           dtype=np.float32)

        eval_obs = self.eval_envs.reset()
        eval_masks = np.ones((self.n_eval_rollout_threads, *self.buffer.masks.shape[2:]), dtype=np.float32)
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states_actor.shape[2:]),
                                   dtype=np.float32)

        while total_episodes < self.eval_episodes:
            self.policy.prep_rollout()
            eval_actions, eval_rnn_states = self.policy.act(np.concatenate(eval_obs),
                                                            np.concatenate(eval_rnn_states),
                                                            np.concatenate(eval_masks), deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)

            eval_cumulative_rewards += eval_rewards
            eval_dones_env = np.all(eval_dones.squeeze(axis=-1), axis=-1)
            total_episodes += np.sum(eval_dones_env)
            eval_episode_rewards.append(eval_cumulative_rewards[eval_dones_env == True])
            eval_cumulative_rewards[eval_dones_env == True] = 0

            eval_masks = np.ones_like(eval_masks, dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), *eval_masks.shape[1:]),
                                                          dtype=np.float32)
            eval_rnn_states[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), *eval_rnn_states.shape[1:]), dtype=np.float32)

        eval_infos = {}
        eval_infos['eval_average_episode_rewards'] = np.concatenate(eval_episode_rewards).mean(
            axis=1)  # shape: [num_agents, 1]
        logging.info(" eval average episode rewards: " + str(np.mean(eval_infos['eval_average_episode_rewards'])))
        self.log_info(eval_infos, total_num_steps)
        logging.info("...End evaluation")

    @torch.no_grad()
    def render(self):
        logging.info("\nStart render ...")
        render_episode_rewards = 0
        render_obs = self.envs.reset()
        render_masks = np.ones((1, *self.buffer.masks.shape[2:]), dtype=np.float32)
        render_rnn_states = np.zeros((1, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
        self.envs.render(mode='txt', filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi')
        while True:
            self.policy.prep_rollout()
            render_actions, render_rnn_states = self.policy.act(np.concatenate(render_obs),
                                                                np.concatenate(render_rnn_states),
                                                                np.concatenate(render_masks),
                                                                deterministic=True)
            render_actions = np.expand_dims(_t2n(render_actions), axis=0)
            render_rnn_states = np.expand_dims(_t2n(render_rnn_states), axis=0)

            # Obser reward and next obs
            render_obs, render_rewards, render_dones, render_infos = self.envs.step(render_actions)
            if self.use_selfplay:
                render_rewards = render_rewards[:, :self.num_agents // 2, ...]
            render_episode_rewards += render_rewards
            self.envs.render(mode='txt', filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi')
            if render_dones.all():
                break
        render_infos = {}
        render_infos['render_episode_reward'] = render_episode_rewards
        logging.info("render episode reward of agent: " + str(render_infos['render_episode_reward']))

    def save(self, episode, best=False):

        policy_actor_state_dict = self.policy.actor.state_dict()
        policy_critic_state_dict = self.policy.critic.state_dict()
        if best:
            torch.save(policy_actor_state_dict, str(self.save_dir) + '/actor_best.pt')
            torch.save(policy_critic_state_dict, str(self.save_dir) + '/critic_best.pt')
        else:
            torch.save(policy_actor_state_dict, str(self.save_dir) + '/actor_latest.pt')
            torch.save(policy_critic_state_dict, str(self.save_dir) + '/critic_latest.pt')

    def restore(self, best=False):
        try:
            if best:
                policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_best.pt')
                policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_best.pt')
                logging.info("Use the best model file.")
            else:
                policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_latest.pt')
                policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_latest.pt')
                logging.info("Use the latest model file.")
        except FileNotFoundError:
            logging.error("Error: Model file not found.")
            sys.exit(1)

        self.policy.actor.load_state_dict(policy_actor_state_dict)
        self.policy.critic.load_state_dict(policy_critic_state_dict)
