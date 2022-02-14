from SuperSafety.Planners.AgentPlanner import TrainVehicle, TestVehicle
from SuperSafety.f110_gym.f110_env import F110Env
from SuperSafety.Utils.utils import *
from SuperSafety.Supervisor.SupervisorySystem import LearningSupervisor, Supervisor
from copy import copy

from TrainTest import *





MAP_NAME = "columbia_small"

def execute_kernel_run(run_name):
    runs = load_yaml_dict(run_name)
    base_config = load_yaml_dict(runs['base_config_name'])
    n = runs['n']

    for run in runs['runs']:
        conf = copy(base_config)
        conf['eval_name'] = runs['run_name']
        conf['base_config_name'] = runs['base_config_name']
        for param in run.keys():
            conf[param] = run[param]

        conf = Namespace(**conf)
        agent_name = f"KernelSSS_{n}_{conf.name}"
        env = F110Env(map=conf.map_name)

        planner = TrainVehicle(agent_name, conf)
        safety_planner = LearningSupervisor(planner, conf)
        train_kernel_vehicle(env, safety_planner, conf, show=False)

        planner = TestVehicle(agent_name, conf)
        eval_dict_wo = evaluate_vehicle(env, planner, conf, False)
    
        safety_planner = Supervisor(planner, conf)
        eval_dict_sss = evaluate_kernel_vehicle(env, safety_planner, conf, False)

        save_dict = vars(conf)
        save_dict['test_number'] = n
        save_dict['Wo'] = eval_dict_wo
        save_dict['SSS'] = eval_dict_sss
        save_dict['agent_name'] = agent_name
        save_conf_dict(save_dict)

def eval_kernel_run(run_name):
    runs = load_yaml_dict(run_name)
    base_config = load_yaml_dict(runs['base_config_name'])
    n = runs['n']
    base_config['test_n'] = 100

    for run in runs['runs']:
        conf = copy(base_config)
        conf['eval_name'] = runs['run_name']
        conf['base_config_name'] = runs['base_config_name']
        for param in run.keys():
            conf[param] = run[param]

        conf = Namespace(**conf)
        agent_name = f"KernelSSS_{n}_{conf.name}"
        env = F110Env(map=conf.map_name)

        planner = TestVehicle(agent_name, conf)
        eval_dict_wo = evaluate_vehicle(env, planner, conf, False)
    
        safety_planner = Supervisor(planner, conf)
        eval_dict_sss = evaluate_kernel_vehicle(env, safety_planner, conf, False)

        save_dict = vars(conf)
        save_dict['test_number'] = n
        save_dict['Wo'] = eval_dict_wo
        save_dict['SSS'] = eval_dict_sss
        save_dict['agent_name'] = agent_name
        save_conf_dict(save_dict)

if __name__ == "__main__":
    # execute_kernel_run("reward_run")
    eval_kernel_run("reward_run")