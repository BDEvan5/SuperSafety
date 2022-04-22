from SuperSafety.f110_gym.f110_env import F110Env
from SuperSafety.Utils.utils import *
from TrainTest import *
from SuperSafety.Supervisor.SupervisorySystem import Supervisor
from SuperSafety.Supervisor.KernelGenerator import build_track_kernel
from SuperSafety.Supervisor.DynamicsBuilder import build_dynamics_table
from SuperSafety.Supervisor.SupervisorySystem import Supervisor, LearningSupervisor
from SuperSafety.Planners.AgentPlanner import TrainVehicle, TestVehicle
from SuperSafety.Planners.PurePursuit import PurePursuit



def generate_kernels():
    conf = load_conf("levine_conf")
    # build_dynamics_table(conf)

    build_track_kernel(conf)


class RandomPlanner:
    def __init__(self, conf, name="RandoPlanner"):
        self.d_max = conf.max_steer # radians  
        self.name = name
        self.speed = conf.vehicle_speed

        path = os.getcwd() + f"/{conf.vehicle_path}" + self.name 
        init_file_struct(path)
        self.path = path
        np.random.seed(conf.random_seed)

    def plan(self, obs):
        steering = np.random.uniform(-self.d_max, self.d_max)
        return np.array([steering, self.speed])


def run_random_test(n=1):
    conf = load_conf("levine_conf")

    agent_name = f"RandoResultL_{conf.map_name}_{conf.kernel_mode}_{n}"
    planner = RandomPlanner(conf, agent_name)
    env = F110Env(map=conf.map_name)

    safety_planner = Supervisor(planner, conf)

    eval_dict = evaluate_vehicle(env, safety_planner, conf, True)
    
    config_dict = vars(conf)
    config_dict['test_number'] = n
    config_dict['agent_name'] = agent_name
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)


def benchmark_sss_tests(n):
    conf = load_conf("levine_conf")

    agent_name = f"LevineKernelSSS_{n}_{conf.map_name}"
    env = F110Env(map=conf.map_name)

    planner = TrainVehicle(agent_name, conf)
    safe_planner = LearningSupervisor(planner, conf)
    train_kernel_vehicle(env, safe_planner, conf, True)

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
    config_dict['eval_name'] = "levine"
    config_dict['vehicle'] = f"KernelSSS{n}"


    save_conf_dict(config_dict)
    env.close_rendering() #TODO: add in everywhere

def test_sss(n):

    conf = load_conf("levine_conf")

    agent_name = f"LevineKernelSSS_{n}_{conf.map_name}"
    env = F110Env(map=conf.map_name)

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
    config_dict['eval_name'] = "levine"
    config_dict['vehicle'] = f"KernelSSS{n}"


    save_conf_dict(config_dict)
    env.close_rendering() #TODO: add in everywhere


def benchmark_baseline_tests(n):
    conf = load_conf("levine_conf")
    conf.rk = 0
    agent_name = f"Baseline_{n}_{conf.map_name}"
    env = F110Env(map=conf.map_name)

    planner = TrainVehicle(agent_name, conf)
    train_baseline_vehicle(env, planner, conf, False)

    planner = TestVehicle(agent_name, conf)
    eval_dict = evaluate_vehicle(env, planner, conf, True)
    
    config_dict = vars(conf)
    config_dict['test_number'] = n
    config_dict['Wo'] = eval_dict
    config_dict['agent_name'] = agent_name
    config_dict['eval_name'] = "levine"
    config_dict['vehicle'] = f"Base{n}"

    save_conf_dict(config_dict)
    env.close_rendering()

def pure_pursuit_tests(n):
    conf = load_conf("levine_conf")
    conf.rk = 0
    agent_name = f"Baseline_{n}_{conf.map_name}"
    env = F110Env(map=conf.map_name)

    agent_name = f"PurePursuit_{conf.map_name}_{n}"
    planner = PurePursuit(conf, agent_name)

    eval_dict = evaluate_vehicle(env, planner, conf, True)
    
    safe_planner = Supervisor(planner, conf)
    eval_sss = evaluate_kernel_vehicle(env, safe_planner, conf, True)

    config_dict = vars(conf)
    config_dict['test_number'] = n
    config_dict['Wo'] = eval_dict
    config_dict['SSS'] = eval_sss
    config_dict['agent_name'] = agent_name
    config_dict['eval_name'] = "levine"
    config_dict['vehicle'] = f"Base{n}"

    save_conf_dict(config_dict)
    env.close_rendering()




if __name__ == "__main__":
    # generate_kernels()
    # run_random_test(1)
    benchmark_sss_tests(1)
    # test_sss(1)
    # benchmark_baseline_tests(1)
    # pure_pursuit_tests(1)

