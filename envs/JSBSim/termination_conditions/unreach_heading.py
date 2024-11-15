import math
from ..core.catalog import Catalog as c
from .termination_condition_base import BaseTerminationCondition


class UnreachHeading(BaseTerminationCondition):
    """
    UnreachHeading [0, 1]
    End up the simulation if the aircraft didn't reach the target heading or attitude in limited time.
    """

    def __init__(self, config):
        super().__init__(config)
        uid = list(config.aircraft_configs.keys())[0]
        aircraft_config = config.aircraft_configs[uid]
        self.safe_altitude = 14000.  # Need to be greater than the safe altitude set in the altitude reward function 4km
        self.max_heading_increment = aircraft_config['max_heading_increment']
        self.max_altitude_increment = aircraft_config['max_altitude_increment']
        self.max_velocities_u_increment = aircraft_config['max_velocities_u_increment']
        self.check_interval = aircraft_config['check_interval']
        self.increment_size = [0.2, 0.4, 0.6, 0.8, 1.0] + [1.0] * 10

    def get_termination(self, task, env, agent_id, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if the aircraft didn't reach the target heading in limited time.

        Args:
            task: task instance
            env: environment instance

        Returns:Q
            (tuple): (done, success, info)
        """
        done = False
        success = False
        cur_step = info['current_step']
        check_time = env.agents[agent_id].get_property_value(c.heading_check_time)
        # print("check_time : {}".format(check_time))
        # check heading when simulation_time exceed check_time
        if env.agents[agent_id].get_property_value(c.simulation_sim_time_sec) >= check_time:
            # if math.fabs(env.agents[agent_id].get_property_value(c.delta_heading)) > 10 or int(math.fabs(env.agents[agent_id].get_property_value(c.delta_altitude))) < 500 or int(math.fabs(env.agents[agent_id].get_property_value(c.delta_velocities_u))) < 10:
            if math.fabs(env.agents[agent_id].get_property_value(c.delta_heading)) > 10:

                # 没有保持稳定时 根据其他的条件还可以判断是否成功
                # > 10 但是小于15 持续了多长时间或者其他两个条件也满足  高度和速度  则判定成功
                done = True
                # if current target heading is reached, random generate a new target heading
            else:
                # if math.fabs(env.agents[agent_id].get_property_value(c.delta_heading)) < 5:
                #     info['termination'] += 64
                info['termination'] += 64
                delta = self.increment_size[env.heading_turn_counts]
                delta_heading = env.np_random.uniform(-delta, delta) * self.max_heading_increment
                delta_altitude = env.np_random.uniform(-delta, delta) * self.max_altitude_increment
                delta_velocities_u = env.np_random.uniform(-delta, delta) * self.max_velocities_u_increment
                new_heading = env.agents[agent_id].get_property_value(c.target_heading_deg) + delta_heading
                new_heading = (new_heading + 360) % 360
                new_altitude = env.agents[agent_id].get_property_value(c.target_altitude_ft) + delta_altitude
                # Make sure the new target altitude is higher than the safe altitude
                if new_altitude <= self.safe_altitude:
                    new_altitude = self.safe_altitude + abs(delta_altitude)
                new_velocities_u = env.agents[agent_id].get_property_value(c.target_velocities_u_mps) + delta_velocities_u
                env.agents[agent_id].set_property_value(c.target_heading_deg, new_heading)
                env.agents[agent_id].set_property_value(c.target_altitude_ft, new_altitude)
                env.agents[agent_id].set_property_value(c.target_velocities_u_mps, new_velocities_u)
                env.agents[agent_id].set_property_value(c.heading_check_time, check_time + self.check_interval)
                env.heading_turn_counts += 1

                self.log(f'current_step:{cur_step} target_heading:{new_heading} '
                         f'target_altitude_ft:{new_altitude} target_velocities_u_mps:{new_velocities_u}')


        if done:
            self.log(f'agent[{agent_id}] unreached heading. Total Steps={env.current_step}')
            info['termination'] += 1
            # info['heading_turn_counts'] = env.heading_turn_counts
        success = False
        return done, success, info
