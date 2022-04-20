from SuperSafety.f110_gym.f110_env import F110Env
from SuperSafety.Utils.utils import *
from TrainTest import *
from SuperSafety.Supervisor.SupervisorySystem import Supervisor
from SuperSafety.Supervisor.KernelGenerator import build_track_kernel
from SuperSafety.Supervisor.DynamicsBuilder import build_dynamics_table


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
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)


if __name__ == "__main__":
    # generate_kernels()
    run_random_test(1)
