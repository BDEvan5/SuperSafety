from SuperSafety.Planners.SimplePlanners import RandomPlanner, ConstantPlanner
from SuperSafety.f110_gym.f110_env import F110Env
from SuperSafety.Utils.utils import *
from TrainTest import *
from SuperSafety.Supervisor.SupervisorySystem import Supervisor

def run_random_test(n=1):
    conf = load_conf("kernel_config")

    agent_name = f"RandoResult_{conf.map_name}_{conf.kernel_mode}_{n}"
    planner = RandomPlanner(conf, agent_name)
    env = F110Env(map=conf.map_name)

    safety_planner = Supervisor(planner, conf)

    eval_dict = evaluate_vehicle(env, safety_planner, conf, True)
    
    config_dict = vars(conf)
    config_dict['test_number'] = n
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)




if __name__ == "__main__":
    run_random_test(1)


