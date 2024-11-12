import math

from .termination_condition_base import BaseTerminationCondition
from ..core.catalog import Catalog as c


class DeltaKeeping(BaseTerminationCondition):
    """
    delta
    End up the simulation if the aircraft is on an extreme state.
    高度差和速度差在一定范围内时 成功！
    """

    def __init__(self, config):
        super().__init__(config)

    def get_termination(self, task, env, agent_id, info={}):
        """
        angle

        Returns:
            (tuple): (done, success, info)
        """
        done = False
        cur_step = info['current_step']
        check_time = 20

        if env.agents[agent_id].get_property_value(c.simulation_sim_time_sec) >= check_time:
            if (math.fabs(env.agents[agent_id].get_property_value(c.delta_altitude)) < 500) & (math.fabs(env.agents[agent_id].get_property_value(c.delta_velocities_u)) < 10):
                done = True
            info['termination'] += 128
            # print("=====yes yes yes=====", flush=True)

        if done:
            self.log(f'agent[{agent_id}] unreached heading. Total Steps={env.current_step}')
            # info['heading_turn_counts'] = env.heading_turn_counts
        success = True
        return done, success, info
