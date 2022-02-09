from SuperSafety.f110_gym.f110_env import F110Env
from SuperSafety.Utils.utils import *

from TrainTest import *
from SuperSafety.Planners.PurePursuit import PurePursuit
from SuperSafety.Planners.follow_the_gap import FollowTheGap
from SuperSafety.Supervisor.SupervisorySystem import Supervisor

def test():
    conf = load_conf("std_config")
    conf.map_name = 'f1_aut_wide'
    env = F110Env(map=conf.map_name)

    # planner = PurePursuit(conf)
    planner = FollowTheGap(conf, "FGM")

    evaluate_vehicle(env, planner, conf, True)


def test_kernel():
    conf = load_conf("kernel_config")
    # conf.map_name = 'f1_aut_wide'
    env = F110Env(map=conf.map_name)

    planner = PurePursuit(conf)
    safe_planner = Supervisor(planner, conf)
    # planner = FollowTheGap(conf, "FGM")

    # evaluate_kernel_vehicle(env, safe_planner, conf, True)
    evaluate_vehicle(env, planner, conf, True)



def run_pp_set(n=1):
    conf = load_conf("std_config")
    conf.test_n = 1

    for track in ['porto', 'columbia_small', 'f1_aut_wide']:
        conf.map_name = track
        env = F110Env(map=conf.map_name)

        agent_name = f"PurePursuit_{conf.map_name}_{n}"
        planner = PurePursuit(conf, agent_name)

        eval_dict = evaluate_vehicle(env, planner, conf, False)
        
        config_dict = vars(conf)
        config_dict['test_number'] = n
        config_dict['eval_name'] = "benchmark"
        config_dict['agent_name'] = agent_name
        config_dict['Wo'] = eval_dict

        save_conf_dict(config_dict)


def run_fgm_set(n=1):
    conf = load_conf("std_config")
    conf.test_n = 1

    for track in ['porto', 'columbia_small', 'f1_aut_wide']:
        conf.map_name = track
        env = F110Env(map=conf.map_name)

        agent_name = f"FGM_{conf.map_name}_{n}"
        planner = FollowTheGap(conf, agent_name)

        eval_dict = evaluate_vehicle(env, planner, conf, False)
        
        config_dict = vars(conf)
        config_dict['test_number'] = n
        config_dict['eval_name'] = "benchmark"
        config_dict['agent_name'] = agent_name
        config_dict['Wo'] = eval_dict

        save_conf_dict(config_dict)



if __name__ == "__main__":
    run_pp_set(1)
    run_fgm_set(1)
    # test()
    # test_kernel()