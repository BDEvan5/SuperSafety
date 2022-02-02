import time
import numpy as np
# Test Functions
def evaluate_vehicle(env, vehicle, conf, show=False):
    crashes = 0
    completes = 0
    lap_times = [] 
    laptime = 0.0
    start = time.time()

    for i in range(conf.test_n):
        obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
        lap_time = 0.0
        while not done:
            action = vehicle.plan(obs)
            sim_steps = conf.sim_steps
            while sim_steps > 0 and not done:
                obs, step_reward, done, _ = env.step(action[None, :])
                sim_steps -= 1

            laptime += step_reward
            if show:
                env.render(mode='human_fast')

        r = find_conclusion(obs, start)

        if r == -1:
            crashes += 1
        else:
            completes += 1
            lap_times.append(laptime)

    success_rate = (completes / (completes + crashes) * 100)
    if len(lap_times) > 0:
        avg_times, std_dev = np.mean(lap_times), np.std(lap_times)
    else:
        avg_times, std_dev = 0, 0


    print(f"Crashes: {crashes}")
    print(f"Completes: {completes} --> {success_rate:.2f} %")
    print(f"Lap times Avg: {avg_times} --> Std: {std_dev}")

    eval_dict = {}
    eval_dict['name'] = vehicle.name
    eval_dict['success_rate'] = float(success_rate)
    eval_dict['avg_times'] = float(avg_times)
    eval_dict['std_dev'] = float(std_dev)

    print(f"Finished running test and saving file with results.")

    return eval_dict


def find_conclusion(s_p, start):
    laptime = s_p['lap_times'][0]
    if s_p['collisions'][0] == 1:
        print(f'Collision --> Sim time: {laptime:.2f} Real time: {(time.time()-start):.2f}')
        return -1
    elif s_p['lap_counts'][0] == 1:
        print(f'Complete --> Sim time: {laptime:.2f} Real time: {(time.time()-start):.2f}')
        return 1
    else:
        print("No conclusion: Awkward palm trees")
        # print(s_p)
    return 0

