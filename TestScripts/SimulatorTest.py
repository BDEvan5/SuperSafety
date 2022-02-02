from SuperSafety.f110_gym.f110_env import F110Env
from SuperSafety.Utils.utils import *

from TrainTest import *
from SuperSafety.Planners.PurePursuit import PurePursuit
from SuperSafety.Planners.follow_the_gap import FollowTheGap

conf = load_conf("simulator_config")
env = F110Env(map=conf.map_name)

planner = PurePursuit(conf)
# planner = FollowTheGap("FGM", conf)

evaluate_vehicle(env, planner, conf, True)



