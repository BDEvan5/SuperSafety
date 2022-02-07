import numpy as np
from SuperSafety.Utils.utils import *


from matplotlib import pyplot as plt
import os, shutil


class RandomPlanner:
    def __init__(self, conf, name="RandoPlanner"):
        self.d_max = conf.max_steer # radians  
        self.name = name
        
        self.speed_dif = (conf.max_v - conf.min_kernel_v) / 2
        self.min_speed = conf.min_kernel_v

        path = os.getcwd() + f"/{conf.vehicle_path}" + self.name 
        init_file_struct(path)
        self.path = path
        np.random.seed(1)

    def plan(self, obs):
        steering = np.random.normal(0, 0.1)
        steering = np.clip(steering, -self.d_max, self.d_max)
        v = np.random.random() * self.speed_dif + self.min_speed
        # v = 5
        #TODO: make speed random too
        return np.array([steering, v])


class ConstantPlanner:
    def __init__(self, name="StraightPlanner", value=0):
        self.steering_value = value
        self.v = 5        
        self.name = name

        path = os.getcwd() + "/Data/Vehicles/" + self.name 
        init_file_struct(path)
        self.path = path

    def plan(self, obs):
        return np.array([self.steering_value, self.v])


