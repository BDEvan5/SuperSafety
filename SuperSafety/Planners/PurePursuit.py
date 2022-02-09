from matplotlib import pyplot as plt
from SuperSafety.Utils.Trajectory import Trajectory 
import numpy as np
from SuperSafety.Utils import pure_pursuit_utils as pp_utils
from SuperSafety.Utils.utils import *

class PurePursuit:
    def __init__(self, conf, name="PurePursuit"):
        self.name = name
        
        self.trajectory = Trajectory(conf.map_name)

        self.lookahead = conf.lookahead
        self.vgain = conf.v_gain
        self.wheelbase =  conf.l_f + conf.l_r
        self.max_steer = conf.max_steer

        path = os.getcwd() + f"/{conf.vehicle_path}" + self.name
        init_file_struct(path)

    def plan(self, obs):
        ego_idx = obs['ego_idx']
        pose_th = obs['poses_theta'][ego_idx] 
        p_x = obs['poses_x'][ego_idx]
        p_y = obs['poses_y'][ego_idx]
        v_current = obs['linear_vels_x'][ego_idx]

        pos = np.array([p_x, p_y], dtype=np.float)

        v_min_plan = 1
        if v_current < v_min_plan:
            return np.array([0, 3]) #TODO:  change this.

        lookahead_point = self.trajectory.get_current_waypoint(pos, self.lookahead)

        speed, steering_angle = pp_utils.get_actuation(pose_th, lookahead_point, pos, self.lookahead, self.wheelbase)
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)
        speed *= self.vgain

        # speed = calculate_speed(steering_angle)
        # speed = np.clip(speed, 0, 3)
        # speed = self.max_v
        return np.array([steering_angle, speed])



