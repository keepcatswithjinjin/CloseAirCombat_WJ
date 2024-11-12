import numpy as np
from typing import List, Tuple
from gymnasium import spaces
from .task_base import BaseTask
from ..core.catalog import Catalog as c
from ..reward_functions import AltitudeReward, HeadingReward, TimeoutReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, UnreachHeading, DeltaKeeping
from ..utils.utils import body2ned, ned2body

class HeadingTask(BaseTask):
    '''
    Control target heading with discrete action space
    '''
    def __init__(self, config):

        self.type = "body"

        super().__init__(config)
        # self.use_noise = getattr(self.config, 'use_noise', False)
        self.use_noise = False
        self.use_ekf = getattr(self.config, 'use_ekf', False)
        self.use_data_loss = getattr(self.config, 'use_data_loss', False)
        self.use_data_loss = True
        self.data_loss_prop = getattr(self.config, 'data_loss_prop', 0.02)  # type: float
        self.std = [50., 0., 10., 10., 10.]    # altitude  (unit: m),  attitude_psi_deg (unit: deg) ,
                                               # v_body_x   (unit: m/s), v_body_y   (unit: m/s), v_body_z   (unit: m/s)

        self.reward_functions = [
            HeadingReward(self.config),
            AltitudeReward(self.config),
            TimeoutReward(self.config),
        ]
        self.termination_conditions = [
            UnreachHeading(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),

            # DeltaKeeping(self.config)
        ]

    @property
    def num_agents(self):
        return 1

    def load_variables(self):
        if self.type == "ned_direct" or self.type == "ned2body":
            velocity_vars = [
                c.velocities_v_north_mps,
                c.velocities_v_east_mps,
                c.velocities_v_down_mps
            ]
        else:
            velocity_vars = [
                c.velocities_u_mps,     # v_body_x
                c.velocities_v_mps,     # v_body_y
                c.velocities_w_mps      # v_body_z
            ]

        self.state_var = [
            c.delta_altitude,       # 0. delta_h   (unit: m)
            c.delta_heading,        # 1. delta_heading  (unit: °)
            c.delta_velocities_u,   # 2. delta_v   (unit: m/s)
            c.position_h_sl_m,      # 3. altitude  (unit: m)
            c.attitude_roll_rad,    # 4. roll      (unit: rad)
            c.attitude_pitch_rad    # 5. pitch     (unit: rad)
        ] + velocity_vars + [
            c.velocities_vc_mps,    # 9. vc        (unit: m/s)
            c.attitude_psi_rad      # 10.yaw       (unit: rad)
        ]


        self.action_var = [
            c.fcs_aileron_cmd_norm,             # [-1., 1.]
            c.fcs_elevator_cmd_norm,            # [-1., 1.]
            c.fcs_rudder_cmd_norm,              # [-1., 1.]
            c.fcs_throttle_cmd_norm,            # [0.4, 0.9]
        ]
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(12,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])


    def get_termination(self, env, agent_id, info={}) -> Tuple[bool, dict]:
        """
        Aggregate termination conditions

        Args:
            env: environment instance
            agent_id: current agent id
            info: additional info

        Returns:
            (tuple):
                done(bool): whether the episode has terminated
                info(dict): additional info
        """
        done = False
        success = True
        info['termination'] = 0
        for condition in self.termination_conditions:
            d, s, info = condition.get_termination(self, env, agent_id, info)
            done = done or d
            success = success and s

            if done:
                info['heading_turn_counts'] = env.heading_turn_counts
                info['end_step'] = env.current_step
                break
        if info['termination'] == 0:
            del info['termination']
        return done, info

    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space.

        observation(dim 12):
            0. ego delta altitude      (unit: km)
            1. ego delta heading       (unit rad)
            2. ego delta velocities_u  (unit: mh)
            3. ego_altitude            (unit: 5km)
            4. ego_roll_sin
            5. ego_roll_cos
            6. ego_pitch_sin
            7. ego_pitch_cos
            8. ego v_body_x            (unit: mh)
            9. ego v_body_y            (unit: mh)
            10. ego v_body_z           (unit: mh)
            11. ego_vc                 (unit: mh)
        """
        obs = np.array(env.agents[agent_id].get_property_values(self.state_var))
        norm_obs = np.zeros(12)
        if self.use_noise:
            # delta_altitude = target_altitude - position_h_sl
            noise = np.random.normal(0, self.std)
            obs[0] -= noise[0]  # 0. ego delta altitude (unit: m)
            obs[1] += noise[1]  # 1. ego delta heading  (unit deg)
            obs[2] -= noise[2]  # 2. ego delta velocities_u (unit: m)
            obs[3] += noise[0]  # 3. ego_altitude   (unit: m)
            obs[6] += noise[2]  # 8. ego_v_north    (unit: m)
            obs[7] += noise[3]  # 9. ego_v_east     (unit: m)
            obs[8] += noise[4]  # 10. ego_v_down    (unit: m)
        if self.use_data_loss:
            not_loss = np.array(np.random.binomial(1, 1-self.data_loss_prop, size=11))
            obs = obs * not_loss

        norm_obs[0] = obs[0] / 1000         # 0. ego delta altitude (unit: 1km)
        norm_obs[1] = obs[1] / 180 * np.pi  # 1. ego delta heading  (unit rad)
        norm_obs[2] = obs[2] / 340          # 2. ego delta velocities_u (unit: mh)
        norm_obs[3] = obs[3] / 5000         # 3. ego_altitude   (unit: 5km)
        norm_obs[4] = np.sin(obs[4])        # 4. ego_roll_sin
        norm_obs[5] = np.cos(obs[4])        # 5. ego_roll_cos
        norm_obs[6] = np.sin(obs[5])        # 6. ego_pitch_sin
        norm_obs[7] = np.cos(obs[5])        # 7. ego_pitch_cos

        if self.type == "body2ned":
            v_body = obs[6:9]
            v_ned = body2ned(v_body, obs[4], obs[5], obs[10])
            norm_obs[8] = v_ned[0] / 340          # 8. ego_v_north    (unit: mh)
            norm_obs[9] = v_ned[1] / 340          # 9. ego_v_east     (unit: mh)
            norm_obs[10] = v_ned[2] / 340         # 10. ego_v_down    (unit: mh)
        elif self.type == "ned2body":
            v_ned = obs[6:9]
            v_body = ned2body(v_ned, obs[4], obs[5], obs[10])
            norm_obs[8] = v_body[0] / 340          # 8. ego_v_north    (unit: mh)
            norm_obs[9] = v_body[1] / 340          # 9. ego_v_east     (unit: mh)
            norm_obs[10] = v_body[2] / 340         # 10. ego_v_down    (unit: mh)
        else:
            norm_obs[8] = obs[6] / 340  # 8. ego_v_north    (unit: mh)
            norm_obs[9] = obs[7] / 340  # 9. ego_v_east     (unit: mh)
            norm_obs[10] = obs[8] / 340  # 10. ego_v_down    (unit: mh)

        norm_obs[11] = obs[9] / 340         # 11. ego_vc        (unit: mh)

        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        """Convert discrete action index into continuous value.
        """
        norm_act = np.zeros(4)
        norm_act[0] = action[0] * 2. / (self.action_space.nvec[0] - 1.) - 1.
        norm_act[1] = action[1] * 2. / (self.action_space.nvec[1] - 1.) - 1.
        norm_act[2] = action[2] * 2. / (self.action_space.nvec[2] - 1.) - 1.
        norm_act[3] = action[3] * 0.5 / (self.action_space.nvec[3] - 1.) + 0.4
        return norm_act

