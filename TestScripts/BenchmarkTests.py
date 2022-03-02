from SuperSafety.f110_gym.f110_env import F110Env
from SuperSafety.Utils.utils import *

from TrainTest import *
from SuperSafety.Planners.PurePursuit import PurePursuit
from SuperSafety.Planners.follow_the_gap import FollowTheGap
from SuperSafety.Supervisor.SupervisorySystem import Supervisor, LearningSupervisor
from SuperSafety.Planners.AgentPlanner import TrainVehicle, TestVehicle

# MAP_NAME = "porto"
MAP_NAME = "columbia_small"

def pure_pursuit_tests(n=1):
    conf = load_conf("config_file")
    conf.map_name = MAP_NAME
    conf.test_n = 5

    runs = zip(['porto', 'columbia_small'], [-0.4, 0])
    for track, stheta in runs:
        conf.map_name = track
        conf.stheta = stheta
        env = F110Env(map=conf.map_name)

        agent_name = f"PurePursuit_{conf.map_name}_{n}"
        planner = PurePursuit(conf, agent_name)

        eval_dict_wo = evaluate_vehicle(env, planner, conf, True)

        # safety_planner = Supervisor(planner, conf)
        # eval_dict_sss = evaluate_kernel_vehicle(env, safety_planner, conf, False)
        
        config_dict = vars(conf)
        config_dict['test_number'] = n
        config_dict['eval_name'] = "benchmark"
        config_dict['agent_name'] = agent_name
        config_dict['Wo'] = eval_dict_wo
        # config_dict['SSS'] = eval_dict_sss
        config_dict['vehicle'] = "PP"


        save_conf_dict(config_dict)
        env.close_rendering()   


def follow_the_gap_tests(n=1):
    conf = load_conf("config_file")
    conf.map_name = MAP_NAME
    conf.test_n = 2

    runs = zip(['porto', 'columbia_small'], [-0.4, 0])
    for track, stheta in runs:
        conf.map_name = track
        conf.stheta = stheta
        env = F110Env(map=conf.map_name)

        agent_name = f"FGM_{conf.map_name}_{n}"
        planner = FollowTheGap(conf, agent_name)

        eval_dict = evaluate_vehicle(env, planner, conf, False)
        
        safety_planner = Supervisor(planner, conf)
        eval_dict_sss = evaluate_kernel_vehicle(env, safety_planner, conf, False)
        
        config_dict = vars(conf)
        config_dict['test_number'] = n
        config_dict['eval_name'] = "benchmark"
        config_dict['agent_name'] = agent_name
        config_dict['Wo'] = eval_dict
        config_dict['SSS'] = eval_dict_sss
        config_dict['vehicle'] = "FGM"

        save_conf_dict(config_dict)


def benchmark_baseline_tests(n):
    conf = load_conf("config_file")
    conf.rk = 0
    runs = zip(['porto', 'columbia_small'], [-0.4, 0])
    # runs = zip(['columbia_small'], [0])
    for track, stheta in runs:
        conf.map_name = track
        conf.stheta = stheta
        agent_name = f"Baseline_{n}_{track}"
        env = F110Env(map=conf.map_name)

        planner = TrainVehicle(agent_name, conf)
        train_baseline_vehicle(env, planner, conf, False)

        planner = TestVehicle(agent_name, conf)
        eval_dict = evaluate_vehicle(env, planner, conf, True)
        
        config_dict = vars(conf)
        config_dict['test_number'] = n
        config_dict['Wo'] = eval_dict
        config_dict['agent_name'] = agent_name
        config_dict['eval_name'] = "benchmark"
        config_dict['vehicle'] = f"Base{n}"

        save_conf_dict(config_dict)
        env.close_rendering()


def benchmark_sss_tests(n):
    conf = load_conf("config_file")
    runs = zip(['porto', 'columbia_small'], [-0.4, 0])
    for track, stheta in runs:
        conf.map_name = track
        conf.stheta = stheta
        agent_name = f"KernelSSS_{n}_{track}"
        env = F110Env(map=conf.map_name)

        planner = TrainVehicle(agent_name, conf)
        safe_planner = LearningSupervisor(planner, conf)
        train_kernel_vehicle(env, safe_planner, conf)

        planner = TestVehicle(agent_name, conf)
        eval_wo = evaluate_vehicle(env, planner, conf, True)

        planner = TestVehicle(agent_name, conf)
        safe_planner = Supervisor(planner, conf)
        eval_sss = evaluate_kernel_vehicle(env, safe_planner, conf, True)

        config_dict = vars(conf)
        config_dict['test_number'] = n
        config_dict['Wo'] = eval_wo
        config_dict['SSS'] = eval_sss
        config_dict['agent_name'] = agent_name
        config_dict['eval_name'] = "benchmark"
        config_dict['vehicle'] = f"KernelSSS{n}"


        save_conf_dict(config_dict)
        env.close_rendering() #TODO: add in everywhere


if __name__ == "__main__":
    pure_pursuit_tests(1)
    follow_the_gap_tests(1)
    benchmark_sss_tests(1)
    benchmark_baseline_tests(1)