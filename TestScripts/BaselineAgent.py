from SuperSafety.Planners.AgentPlanner import TrainVehicle, TestVehicle
from SuperSafety.f110_gym.f110_env import F110Env
from SuperSafety.Utils.utils import *

from TrainTest import *



def train_baseline(n):
    conf = load_conf("std_config")
    agent_name = f"Baseline_{n}"
    env = F110Env(map=conf.map_name)
    planner = TrainVehicle(agent_name, conf)

    train_baseline_vehicle(env, planner, conf)

def eval_baseline(n):

    conf = load_conf("std_config")
    agent_name = f"Baseline_{n}"
    env = F110Env(map=conf.map_name)
    planner = TestVehicle(agent_name, conf)

    evaluate_vehicle(env, planner, conf, True)


def train_test_baseline(n):
    conf = load_conf("std_config")
    agent_name = f"Baseline_{n}"
    env = F110Env(map=conf.map_name)
    conf.r1 = 0.012
    conf.r2 = 0.006
    planner = TrainVehicle(agent_name, conf)

    train_baseline_vehicle(env, planner, conf, False)

    planner = TestVehicle(agent_name, conf)
    eval_dict = evaluate_vehicle(env, planner, conf, True)
    
    config_dict = vars(conf)
    config_dict['test_number'] = n
    config_dict['Wo'] = eval_dict
    config_dict['agent_name'] = agent_name
    config_dict['eval_name'] = "Performance"

    save_conf_dict(config_dict)


if __name__ == '__main__':
    # train_baseline(1)
    eval_baseline(100)
    # train_test_baseline(306)

    # train_test_baseline(100)
    # for i in range(400, 410):
    #     train_test_baseline(i)
