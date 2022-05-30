from SuperSafety.f110_gym.f110_env import F110Env
from SuperSafety.Utils.utils import *
from TrainTest import *
from SuperSafety.Supervisor.SupervisorySystem import Supervisor
from SuperSafety.Supervisor.KernelGenerator import build_track_kernel
from SuperSafety.Supervisor.DynamicsBuilder import build_dynamics_table


def generate_kernels():
    conf = load_conf("config_file")
    build_dynamics_table(conf)

    # conf.map_name = "columbia_small"
    # build_track_kernel(conf)

    for track in ['porto', 'columbia_small']:
        conf.map_name = track
        build_track_kernel(conf)

def generate_kernel_single():
    conf = load_conf("config_file")
    build_dynamics_table(conf)

    # conf.map_name = 'example_map'
    # conf.map_name = 'race_track'
    # conf.map_name = 'f1_aut_wide'
    # conf.map_name = 'columbia_small'
    build_track_kernel(conf)





def run_random_test_single(n=1):
    conf = load_conf("config_file")
    # conf.map_name = "porto"
    # conf.map_name = "race_track"
    conf.map_name = "f1_aut_wide"

    agent_name = f"RandoResult_{conf.map_name}_{conf.kernel_mode}_{n}"
    planner = RandomPlanner(conf, agent_name)
    env = F110Env(map=conf.map_name)

    safety_planner = Supervisor(planner, conf)

    eval_dict = evaluate_vehicle(env, safety_planner, conf, True)
    
    config_dict = vars(conf)
    config_dict['test_number'] = n
    config_dict.update(eval_dict)

    save_conf_dict(config_dict)




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
    conf = load_conf("config_file")
    conf.test_n = 100


    runs = zip(['porto', 'columbia_small'], [-0.4, 0])
    for track, stheta in runs:
        conf.map_name = track
        conf.stheta = stheta
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



if __name__ == "__main__":
    # generate_kernels()
    # generate_kernel_single()
    run_random_test_single(1)

    # run_random_test(1)

