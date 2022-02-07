from SuperSafety.Planners.AgentPlanner import TrainVehicle, TestVehicle
from SuperSafety.f110_gym.f110_env import F110Env
from SuperSafety.Utils.utils import *
from SuperSafety.Supervisor.SupervisorySystem import LearningSupervisor, Supervisor

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

def run_safe_train_eval(n):
    conf = load_conf("kernel_config")
    conf.r1 = 0.012
    conf.r2 = 0.006
    conf.rk = 0.01
    agent_name = f"KernelSSS_{n}"
    env = F110Env(map=conf.map_name)

    planner = TrainVehicle(agent_name, conf)
    safe_planner = LearningSupervisor(planner, conf)
    train_kernel_vehicle(env, safe_planner, conf)

    planner = TestVehicle(agent_name, conf)
    eval_wo = evaluate_vehicle(env, planner, conf, True)

    planner = TestVehicle(agent_name, conf)
    safe_planner = Supervisor(planner, conf)
    eval_sss = evaluate_vehicle(env, safe_planner, conf, True)

    config_dict = vars(conf)
    config_dict['test_number'] = n
    config_dict['Wo'] = eval_wo
    config_dict['SSS'] = eval_sss
    config_dict['agent_name'] = agent_name
    config_dict['eval_name'] = "Performance"

    save_conf_dict(config_dict)


if __name__ == '__main__':
    # train_safe_agent(5)
    # eval_safe_agent(405)
    run_safe_train_eval(406)

