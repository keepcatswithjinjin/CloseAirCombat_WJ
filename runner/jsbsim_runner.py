import sys
import time
import torch
import logging
import numpy as np
from typing import List
from .base_runner import Runner, ReplayBuffer
from algorithms.SARS.rff_mapping import rff_mapping
from algorithms.SARS.rff_kernel_density import rff_kernel_density
from tqdm import tqdm

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
        # add by wangjian 20240925
        num_agents = 1
        num_envs = 4

        for episode in tqdm(range(episodes), desc="total process Episode"):

            heading_turns_list = []
            # 20240426 add by ash0^0
            end_step_list = []
            termination_list = []

            """
            # 暂时/成功/失败状态缓存区
            Ds_buffer = []   # Ds
            Df_buffer = []   # Df
            # add by wangjian 20240905
            # 初始化不同环境和代理的缓冲区
            Ds_buffer = [[[] for _ in range(num_agents)] for _ in range(num_envs)]
            Df_buffer = [[[] for _ in range(num_agents)] for _ in range(num_envs)]
            temp_buffer = [[[] for _ in range(num_agents)] for _ in range(num_envs)]
            """

            # 一次循环就是一次交互过程  循环结束后  会出现若干个任务结束的  trajectory : (s,a,r,s1,a1,r1........)
            for step in tqdm(range(self.buffer_size), desc="Collecting information step"):
                # Sample actions   values: 状态价值函数 s下期望的回报  用于计算优势函数
                # 采样动作用到 obs  更新obs时将obs装进buffer
                values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.collect(step)
                # reward and next obs  reward: 回报  Rt   用于计算优势函数

                obs, rewards, dones, infos = self.envs.step(actions)

                """
                # =========================================S_A_S_R==============================================

                # SARS v1.0
                # 需要将收集的信息从obs_buffer[] 中将不同环境不同代理的奖励单独遍历进行计算
                for env_idx in range(num_envs):
                    for agent_idx in range(num_agents):  # 遍历每个代理
                        agent_obs = obs[env_idx][agent_idx] # 获取当前代理的obs
                        agent_reward = rewards[env_idx][agent_idx]  # 获取当前代理的奖励
                        temp_buffer[env_idx][agent_idx].append(agent_obs)

                        if 'termination' in infos[env_idx]:  # 同一个环境终止条件是同一个
                            # 拿到终止条件
                            termination = infos[env_idx]['termination']
                            # print("termination : {}, env:{}, agent:{}".format(termination, env_idx, agent_idx))
                            # 判断6个终止条件 达到任何一个条件  trajectory形成
                            termination_flag = ((termination & (1 | 2 | 4 | 8 | 16 | 32)) != 0)  # 任意一位被置位
                            if termination_flag:
                                # print("\n")
                                # print("\n")
                                # print("!!!!!!!!!!!!!!!!!termination!!!!!!!!!!!!!!!!!!!!!! env:{}, agent:{} ".format(env_idx, agent_idx))
                                # print("\n")
                                # print("\n")
                                # 拿到6个不同的终止标志位
                                unreach_heading_flag = (termination & 1)
                                extreme_state_flag = ((termination >> 1) & 1)
                                overload_flag = ((termination >> 2) & 1)
                                low_altitude_flag = ((termination >> 3) & 1)
                                timeout_flag = ((termination >> 4) & 1)
                                safe_return_flag = ((termination >> 5) & 1)
                                reach_heading_flag = ((termination >> 6) & 1)

                                # (1) 1v1 single_combat   env_num = 4  agent_num = 2
                                # flag: safe_return(安全返回 打掉敌机)

                                # (2) 1 single_control_heading env_num = 4 agent_num = 1
                                # flag: low_altitude / unreached_heading...

                                # 根据不同的任务设置不同的终止标志  将 成功/失败 的obs加入 Ds_buffer/Df_buffer
                                if unreach_heading_flag | overload_flag | timeout_flag:
                                    # Ds_buffer[env_idx][agent_idx].extend(temp_buffer[env_idx][agent_idx])  # 将成功的obs加入成功缓冲区
                                    Df_buffer[env_idx][agent_idx].extend(temp_buffer[env_idx][agent_idx])
                                elif reach_heading_flag:
                                    # Df_buffer[env_idx][agent_idx].extend(temp_buffer[env_idx][agent_idx])  # 将失败的obs加入失败缓冲区
                                    Ds_buffer[env_idx][agent_idx].extend(temp_buffer[env_idx][agent_idx])
                                else:
                                    Df_buffer[env_idx][agent_idx].extend(temp_buffer[env_idx][agent_idx])
                                temp_buffer[env_idx][agent_idx].clear()  # 清空对应暂存buffer

                        total_success = sum(len(buffer) for buffer in Ds_buffer[env_idx][agent_idx])
                        total_failure = sum(len(buffer) for buffer in Df_buffer[env_idx][agent_idx])
                        # print("total_success:{}, total_failure:{}, env:{}, agent:{}".format(total_success, total_failure, env_idx, agent_idx))
                        N = total_success + total_failure
                        # print("N : {}".format(N))
                        M = 500  # RFF 随机特征维数M  : 50会降低性能  应取500 1000 2000
                        sigma = 0.2  # RFF 高斯核带宽 h
                        # 在计算密度之前检查缓冲区
                        if len(Ds_buffer[env_idx][agent_idx]) > 0:

                            Ns_i_density = rff_kernel_density(agent_obs, Ds_buffer[env_idx][agent_idx], M, sigma)

                            # gpu--->cpu
                            # Ns_i_density = rff_kernel_density(agent_obs, Ds_buffer[env_idx][agent_idx], M, sigma).cpu()
                            flag_s = 1
                            # print("env_idx:{}, agent_idx:{}, Ns_i_density:{}".format(env_idx, agent_idx, Ns_i_density))
                        else:
                            Ns_i_density = 0  # 或者其他处理方式，例如返回默认值
                            flag_s = 0

                        if len(Df_buffer[env_idx][agent_idx]) > 0:

                            Nf_i_density = rff_kernel_density(agent_obs, Df_buffer[env_idx][agent_idx], M, sigma)

                            # gpu--->cpu
                            # Nf_i_density = rff_kernel_density(agent_obs, Df_buffer[env_idx][agent_idx], M, sigma).cpu()
                            flag_f = 1
                            # print("env_idx:{}, agent_idx:{}, Nf_i_density:{}".format(env_idx, agent_idx, Nf_i_density))

                        else:
                            Nf_i_density = 0  # 或者其他处理方式
                            flag_f = 0
                        if flag_s + flag_f > 0:
                            # tensor * int ---> int
                            Ns_i_count = N * Ns_i_density
                            Nf_i_count = N * Nf_i_density

                            # print("env:{}, agent:{}, Ns_i_count:{}".format(env_idx, agent_idx, Ns_i_count))
                            # print("env:{}, agent:{}, Nf_i_count:{}".format(env_idx, agent_idx, Nf_i_count))

                            alpha = Ns_i_count + 1
                            beta = Nf_i_count + 1

                            rs_i = np.random.beta(alpha, beta)


                            # print("rs_i_env_idx:{}, rs_i_agent_idx:{}, rs_i:{}".format(env_idx, agent_idx, rs_i))

                            a = 0.4  # 成型奖励的权重 0.4 0.6 0.8
                            rewards[env_idx][agent_idx] = agent_reward + a * rs_i  # 更新当前代理的奖励

                #=========================================S_A_S_R==============================================
                """

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

                data = obs, actions, rewards, dones, action_log_probs, values, rnn_states_actor, rnn_states_critic

                # insert data into buffer
                # 下一次采样动作时用到buffer中这一次的的obs
                self.insert(data)

            # compute return and update network
            # Inherit from parent class Runner
            self.compute()   # 计算优势函数中的一部分 ： rt + γ * V_θ(S_t+1)
            train_infos = self.train()

            # post process
            self.total_num_steps = (episode + 1) * self.buffer_size * self.n_rollout_threads

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                date_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))
                logging.info("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {},time {}.\n"
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
                        # unreach_heading_counts += (termination & 1)
                        extreme_state_counts += ((termination >> 1) & 1)
                        overload_counts += ((termination >> 2) & 1)
                        low_altitude_counts += ((termination >> 3) & 1)
                        timeout_counts += ((termination >> 4) & 1)
                        safe_return_counts += ((termination >> 5) & 1)
                    # train_infos["unreach_heading_prop"] = unreach_heading_counts / done_counts
                    train_infos["extreme_state_prop"] = extreme_state_counts / done_counts
                    train_infos["overload_prop"] = overload_counts / done_counts
                    train_infos["low_altitude_prop"] = low_altitude_counts / done_counts
                    train_infos["timeout_prop"] = timeout_counts / done_counts
                    train_infos["safe_return_prop"] = safe_return_counts / done_counts
                    train_infos["done_counts"] = done_counts
                    logging.info("{:^20} | {:^20} | {:^20} | {:^20} | {:^20} | {:^20} | {:^20}"
                                 .format("done_counts", "unreach_heading_prop", "extreme_state_prop",
                                         "overload_prop", "low_altitude_prop", "timeout_prop", "safe_return_prop"))
                    logging.info("{:^20} | {:^20.4f} | {:^20.4f} | {:^20.4f} | {:^20.4f} | {:^20.4f} | {:^20.4f}"
                                 .format(train_infos["done_counts"], train_infos["unreach_heading_prop"], train_infos["extreme_state_prop"],
                                         train_infos["overload_prop"], train_infos["low_altitude_prop"], train_infos["timeout_prop"], train_infos["safe_return_prop"]))

                self.log_info(train_infos, self.total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and episode != 0 and self.use_eval:
                self.eval(self.total_num_steps)

            # save model
            if self.save_best:
                average_reward = self.buffer.rewards.sum() / (self.buffer.masks == False).sum()
                if average_reward > self.best_reward:
                    logging.info("save best model on episode {} : average episode rewards is {}".format(episode, average_reward))
                    self.best_reward = average_reward
                    self.save(episode, True)
            if (episode % self.save_interval == 0) or (episode == episodes - 1):
                self.save(episode)

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

        rnn_states_actor[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_actor.shape[1:]), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), *rnn_states_critic.shape[1:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        self.buffer.insert(obs, actions, rewards, masks, action_log_probs, values, rnn_states_actor, rnn_states_critic)

    @torch.no_grad()
    def eval(self, total_num_steps):
        logging.info("\nStart evaluation...")
        total_episodes, eval_episode_rewards = 0, []
        eval_cumulative_rewards = np.zeros((self.n_eval_rollout_threads, *self.buffer.rewards.shape[2:]), dtype=np.float32)

        eval_obs = self.eval_envs.reset()
        eval_masks = np.ones((self.n_eval_rollout_threads, *self.buffer.masks.shape[2:]), dtype=np.float32)
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)

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
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), *eval_masks.shape[1:]), dtype=np.float32)
            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), *eval_rnn_states.shape[1:]), dtype=np.float32)

        eval_infos = {}
        eval_infos['eval_average_episode_rewards'] = np.concatenate(eval_episode_rewards).mean(axis=1)  # shape: [num_agents, 1]
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
