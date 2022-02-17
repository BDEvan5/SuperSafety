from SuperSafety.Planners.AgentPlanner import TrainVehicle, TestVehicle
from SuperSafety.f110_gym.f110_env import F110Env
from SuperSafety.Utils.utils import *
from SuperSafety.Supervisor.SupervisorySystem import LearningSupervisor, Supervisor
from copy import copy

from TrainTest import *

MAP_NAME = "columbia_small"

def baseline(conf, env, n):
    conf.rk = 0
    agent_name = f"Baseline_{n}"
    planner = TrainVehicle(agent_name, conf)
    train_baseline_vehicle(env, planner, conf, False)

    planner = TestVehicle(agent_name, conf)
    eval_dict = evaluate_vehicle(env, planner, conf, False)
    
    config_dict = vars(conf)
    config_dict['test_number'] = n
    config_dict['Wo'] = eval_dict
    config_dict['agent_name'] = agent_name
    config_dict['eval_name'] = "repeat"
    config_dict['vehicle'] = "Base2"

    save_conf_dict(config_dict)


def kernel_sss(conf, env, n):
    conf.rk = 0.04
    agent_name = f"KernelSSS_{n}"
    planner = TrainVehicle(agent_name, conf)
    safe_planner = LearningSupervisor(planner, conf)
    train_kernel_vehicle(env, safe_planner, conf)

    planner = TestVehicle(agent_name, conf)
    eval_wo = evaluate_vehicle(env, planner, conf, False)

    planner = TestVehicle(agent_name, conf)
    safe_planner = Supervisor(planner, conf)
    eval_sss = evaluate_kernel_vehicle(env, safe_planner, conf, False)

    config_dict = vars(conf)
    config_dict['test_number'] = n
    config_dict['Wo'] = eval_wo
    config_dict['SSS'] = eval_sss
    config_dict['agent_name'] = agent_name
    config_dict['eval_name'] = "repeat"
    config_dict['vehicle'] = "KernelSSS2"


    save_conf_dict(config_dict)

def run_repeatability():
    conf = load_conf("config_file")
    conf.map_name = MAP_NAME
    env = F110Env(map=conf.map_name)

    for i in range (100, 110):
        # baseline(copy(conf), env, i)
        kernel_sss(copy(conf), env, i)

if __name__ == "__main__":
    run_repeatability()