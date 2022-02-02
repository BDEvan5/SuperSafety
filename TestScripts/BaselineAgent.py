from SuperSafety.Planners.AgentPlanner import TrainVehicle, TestVehicle
from SuperSafety.f110_gym.f110_env import F110Env
from SuperSafety.Utils.utils import *

from TrainTest import *



def train_baseline(n):
    conf = load_conf("simulator_config")
    agent_name = f"Baseline_{n}"
    env = F110Env(map=conf.map_name)
    planner = TrainVehicle(agent_name, conf)

    train_baseline_vehicle(env, planner, conf)




if __name__ == '__main__':
    train_baseline(1)