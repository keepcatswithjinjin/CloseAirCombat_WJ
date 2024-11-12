import os

from .env_base import BaseEnv
from ..tasks.heading_task import HeadingTask
from ..tasks.ren_heading_task import RenHeadingTask

# TODO 新加的
from concurrent.futures import ThreadPoolExecutor, as_completed
from algorithms.SARS.rff_kernel_density import rff_kernel_density
import numpy as np


class SingleControlEnv(BaseEnv):
    """
    SingleControlEnv is an fly-control env for single agent with no enemy fighters.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        self.num_agent = 1
        self.Ds_buffer = [[] for _ in range(self.num_agent)]
        self.Df_buffer = [[] for _ in range(self.num_agent)]
        self.temp_buffer = [[] for _ in range(self.num_agent)]
        assert len(self.agents.keys()) == 1, f"{self.__class__.__name__} only supports 1 aircraft!"
        self.init_states = None

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'heading':
            self.task = HeadingTask(self.config)
        elif taskname == 'ren_heading':
            self.task = RenHeadingTask(self.config)
        else:
            raise NotImplementedError(f'Unknown taskname: {taskname}')

    def reset(self):
        self.current_step = 0
        self.reset_simulators()
        self.heading_turn_counts = 0
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs)

    def reset_simulators(self):
        if self.init_states is None:
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]
        init_heading = self.np_random.uniform(0., 180.)
        init_altitude = self.np_random.uniform(14000., 30000.)
        init_velocities_u = self.np_random.uniform(400., 1200.)
        for init_state in self.init_states:
            init_state.update({
                'ic_psi_true_deg': init_heading,
                'ic_h_sl_ft': init_altitude,
                'ic_u_fps': init_velocities_u,
                'target_heading_deg': init_heading,
                'target_altitude_ft': init_altitude,
                'target_velocities_u_mps': init_velocities_u * 0.3048,
            })
        for idx, sim in enumerate(self.agents.values()):
            sim.reload(self.init_states[idx])
        self._tempsims.clear()

    def clear_buffers(self):
        # self.Ds_buffer = [[] for _ in range(self.num_agent)]
        # self.Df_buffer = [[] for _ in range(self.num_agent)]
        for agent_idx in range(self.num_agent):
            # 保留 Ds_buffer 每个子列表中的最新一半数据
            # print("Ds_before : {} , pid : {}".format(len(self.Ds_buffer[agent_idx]), os.getpid()), flush=True)
            # print("Df_before : {} , pid : {}".format(len(self.Df_buffer[agent_idx]), os.getpid()), flush=True)
            self.Ds_buffer[agent_idx] = self.Ds_buffer[agent_idx][len(self.Ds_buffer[agent_idx]) // 2:]
            self.Df_buffer[agent_idx] = self.Df_buffer[agent_idx][len(self.Df_buffer[agent_idx]) // 2:]
            # print("Ds : {} , pid : {}".format(len(self.Ds_buffer[agent_idx]), os.getpid()), flush=True)
            # print("Df : {} , pid : {}".format(len(self.Df_buffer[agent_idx]), os.getpid()), flush=True)
        return True


