from SuperSafety.Planners.AgentPlanner import TrainVehicle, TestVehicle
from SuperSafety.f110_gym.f110_env import F110Env
from SuperSafety.Utils.utils import *
from SuperSafety.Supervisor.SupervisorySystem import LearningSupervisor

from TrainTest import *



def train_safe_agent(n):
    conf = load_conf("kernel_config")
    agent_name = f"KernelSSS_{n}"
    env = F110Env(map=conf.map_name)
    planner = TrainVehicle(agent_name, conf)
    safe_planner = LearningSupervisor(planner, conf)

    train_kernel_vehicle(env, safe_planner, conf)

def eval_safe_agent(n):

    conf = load_conf("std_config")
    agent_name = f"KernelSSS_{n}"
    env = F110Env(map=conf.map_name)
    planner = TestVehicle(agent_name, conf)

    evaluate_vehicle(env, planner, conf, True)

if __name__ == '__main__':
    train_safe_agent(1)
    # eval_safe_agent(1)

