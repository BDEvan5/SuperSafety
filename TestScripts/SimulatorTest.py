from SuperSafety.f110_gym.f110_env import F110Env
from SuperSafety.Utils.utils import *

from SuperSafety.Planners.PurePursuit import PurePursuit
from TrainTest import *

conf = load_conf("simulator_config")
env = F110Env(map=conf.map_name)

planner = PurePursuit(conf)

evaluate_vehicle(env, planner, conf, True)



