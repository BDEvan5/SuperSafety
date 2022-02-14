from SuperSafety.Planners.SimplePlanners import RandomPlanner, ConstantPlanner
from SuperSafety.f110_gym.f110_env import F110Env
from SuperSafety.Utils.utils import *
from TrainTest import *
from SuperSafety.Supervisor.SupervisorySystem import Supervisor
from SuperSafety.Supervisor.KernelGenerator import build_track_kernel
from SuperSafety.Supervisor.DynamicsBuilder import build_dynamics_table


def generate_kernels():
    conf = load_conf("config_file")
    build_dynamics_table(conf)

    for track in ['porto', 'columbia_small']:
        conf.map_name = track
        build_track_kernel(conf)




def run_random_test(n=1):
    conf = load_conf("kernel_config")
    conf.map_name = "porto"

    agent_name = f"RandoResult_{conf.map_name}_{conf.kernel_mode}_{n}"
    planner = RandomPlanner(conf, agent_name)
    env = F110Env(map=conf.map_name)

    safety_planner = Supervisor(planner, conf)

    eval_dict = evaluate_vehicle(env, safety_planner, conf, True)
    
    config_dict = vars(conf)
    config_dict['test_number'] = n
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)




def run_test_set(n=1):
    conf = load_conf("kernel_config")
    conf.test_n = 100

    conf.map_name = "porto"
    conf.stheta = -0.4
    env = F110Env(map=conf.map_name)

    agent_name = f"RandoResult_{conf.map_name}_{conf.kernel_mode}_{n}"
    planner = RandomPlanner(conf, agent_name)
    safety_planner = Supervisor(planner, conf)

    eval_dict = evaluate_kernel_vehicle(env, safety_planner, conf, False)
    
    config_dict = vars(conf)
    config_dict['eval_name'] = "KernelGen"
    config_dict['test_number'] = n
    config_dict['agent_name'] = agent_name
    config_dict['SSS'] = eval_dict

    save_conf_dict(config_dict)

    conf.map_name = "columbia_small"
    conf.stheta = 0
    env = F110Env(map=conf.map_name)

    agent_name = f"RandoResult_{conf.map_name}_{conf.kernel_mode}_{n}"
    planner = RandomPlanner(conf, agent_name)
    safety_planner = Supervisor(planner, conf)

    eval_dict = evaluate_kernel_vehicle(env, safety_planner, conf, False)
    
    config_dict = vars(conf)
    config_dict['eval_name'] = "KernelGen"
    config_dict['test_number'] = n
    config_dict['agent_name'] = agent_name
    config_dict['SSS'] = eval_dict

    save_conf_dict(config_dict)





def run_test_f1(n=1):
    conf = load_conf("kernel_config")
    conf.test_n = 10
    # conf.n_dx = 60

    conf.map_name = 'f1_aut_wide'
    env = F110Env(map=conf.map_name)

    agent_name = f"RandoResult_{conf.map_name}_{conf.kernel_mode}_{n}"
    planner = RandomPlanner(conf, agent_name)
    safety_planner = Supervisor(planner, conf)

    eval_dict = evaluate_kernel_vehicle(env, safety_planner, conf, True)
    
    config_dict = vars(conf)
    config_dict['eval_name'] = "KernelGen"
    config_dict['test_number'] = n
    config_dict['agent_name'] = agent_name
    config_dict['SSS'] = eval_dict

    save_conf_dict(config_dict)




if __name__ == "__main__":
    generate_kernels()

    # run_random_test(1)
    # run_test_set(2)
    # run_test_f1(1)

