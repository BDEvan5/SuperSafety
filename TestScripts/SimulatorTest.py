from SuperSafety.f110_gym.f110_env import F110Env
from SuperSafety.Utils.utils import *

sim_conf = load_conf("simulator_config")
env = F110Env(map=sim_conf.map_name)


