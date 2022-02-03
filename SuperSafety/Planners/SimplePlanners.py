import numpy as np
from SupervisorySafetySystem.NavUtils import pure_pursuit_utils as pp_utils
from SupervisorySafetySystem.NavUtils.speed_utils import calculate_speed
import csv 
from SupervisorySafetySystem import LibFunctions as lib


from matplotlib import pyplot as plt
import os, shutil
from SupervisorySafetySystem.NavUtils.Trajectory import Trajectory


class RandomPlanner:
    def __init__(self, conf, name="RandoPlanner"):
        self.d_max = conf.max_steer # radians  
        self.name = name
        
        path = os.getcwd() + f"/{conf.vehicle_path}" + self.name 
        # path = os.getcwd() + "/EvalVehicles/" + self.name 
        if os.path.exists(path):
            try:
                os.rmdir(path)
            except:
                shutil.rmtree(path)
        os.mkdir(path)
        self.path = path
        np.random.seed(1)

    def plan(self, obs):
        steering = np.random.normal(0, 0.1)
        steering = np.clip(steering, -self.d_max, self.d_max)
        # v = calculate_speed(steering)
        v = 1.5
        #TODO: make speed random too
        return np.array([steering, v])


class ConstantPlanner:
    def __init__(self, name="StraightPlanner", value=0):
        self.steering_value = value
        self.v = 4        
        self.name = name

        path = os.getcwd() + "/PaperData/Vehicles/" + self.name 
        # path = os.getcwd() + "/EvalVehicles/" + self.name 
        if os.path.exists(path):
            try:
                os.rmdir(path)
            except:
                shutil.rmtree(path)
        os.mkdir(path)
        self.path = path

    def plan_act(self, obs):
        return np.array([self.steering_value, self.v])


