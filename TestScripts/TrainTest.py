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
        while not done:
            action = vehicle.plan(obs)
            sim_steps = conf.sim_steps
            while sim_steps > 0 and not done:
                obs, step_reward, done, _ = env.step(action[None, :])
                sim_steps -= 1

            if show:
                env.render(mode='human_fast')

        r = find_conclusion(obs, start)

        if r == -1:
            crashes += 1
        else:
            completes += 1
            lap_times.append(env.lap_times[0])

    success_rate = (completes / (completes + crashes) * 100)
    if len(lap_times) > 0:
        avg_times, std_dev = np.mean(lap_times), np.std(lap_times)
    else:
        avg_times, std_dev = 0, 0


    print(f"Crashes: {crashes}")
    print(f"Completes: {completes} --> {success_rate:.2f} %")
    print(f"Lap times Avg: {avg_times} --> Std: {std_dev}")

    eval_dict = {}
    eval_dict['success_rate'] = float(success_rate)
    eval_dict['avg_times'] = float(avg_times)
    eval_dict['std_dev'] = float(std_dev)

    print(f"Finished running test and saving file with results.")

    return eval_dict

def evaluate_kernel_vehicle(env, vehicle, conf, show=False):
    crashes = 0
    completes = 0
    lap_times = [] 
    start = time.time()
    interventions = []

    for i in range(conf.test_n):
        obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
        while not done:
            action = vehicle.plan(obs)
            sim_steps = conf.sim_steps
            while sim_steps > 0 and not done:
                obs, step_reward, done, _ = env.step(action[None, :])
                sim_steps -= 1

            if show:
                env.render(mode='human_fast')

        r = find_conclusion(obs, start)
        interventions.append(vehicle.interventions)
        vehicle.interventions = 0
        if r == -1:
            crashes += 1
        else:
            completes += 1
            lap_times.append(env.lap_times[0])

    success_rate = (completes / (completes + crashes) * 100)
    if len(lap_times) > 0:
        avg_times, std_dev = np.mean(lap_times), np.std(lap_times)
        avg_interventions = np.mean(interventions)
    else:
        avg_times, std_dev = 0, 0


    print(f"Crashes: {crashes}")
    print(f"Completes: {completes} --> {success_rate:.2f} %")
    print(f"Lap times Avg: {avg_times} --> Std: {std_dev}")
    print(f"Interventions Avg: {avg_interventions}")

    eval_dict = {}
    eval_dict['success_rate'] = float(success_rate)
    eval_dict['avg_times'] = float(avg_times)
    eval_dict['std_dev'] = float(std_dev)
    eval_dict['avg_interventions'] = float(avg_interventions)

    print(f"Finished running test and saving file with results.")

    return eval_dict


def train_baseline_vehicle(env, vehicle, conf, show=False):
    start_time = time.time()
    state, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    print(f"Starting Baseline Training: {vehicle.name}")
    crash_counter = 0

    ep_steps = 0 
    lap_counter = 0
    for n in range(conf.train_n):
        state['reward'] = set_reward(state)
        action = vehicle.plan(state)
        sim_steps = conf.sim_steps
        while sim_steps > 0 and not done:
            s_prime, r, done, _ = env.step(action[None, :])
            sim_steps -= 1

        state = s_prime
        vehicle.agent.train(2)
        if show:
            env.render('human_fast')
        
        if done or ep_steps > conf.max_steps:
            s_prime['reward'] = set_reward(s_prime) 
            vehicle.done_entry(s_prime)

            print(f"{n}::Lap done {lap_counter} -> FinalR: {s_prime['reward']} -> LapTime {env.lap_times[0]:.2f} -> TotalReward: {vehicle.t_his.rewards[vehicle.t_his.ptr-1]:.2f}")
            lap_counter += 1
            ep_steps = 0 
            if state['reward'] == -1:
                crash_counter += 1

            state, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
            
        ep_steps += 1

    vehicle.t_his.print_update(True)
    vehicle.t_his.save_csv_data()
    vehicle.agent.save(vehicle.path)

    train_time = time.time() - start_time
    print(f"Finished Training: {vehicle.name} in {train_time} seconds")
    print(f"Crashes: {crash_counter}")

    return train_time, crash_counter


def train_kernel_vehicle(env, vehicle, conf, show=False):
    start_time = time.time()
    state, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    print(f"Starting KernelSSS Training: {vehicle.name}")
    crash_counter = 0

    ep_steps = 0 
    lap_counter = 0
    for n in range(conf.train_n):
        state['reward'] = set_reward(state) #theoretically always zero in this position
        action = vehicle.plan(state)
        sim_steps = conf.sim_steps
        while sim_steps > 0 and not done:
            s_prime, r, done, _ = env.step(action[None, :])
            sim_steps -= 1

        state = s_prime
        vehicle.planner.agent.train(2)
        # env.render('human_fast')
        
        if s_prime['collisions'][0] == 1:
            print(f"COLLISION:: Lap done {lap_counter} -> {env.lap_times[0]} -> Inters: {vehicle.ep_interventions}")
            state, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
            lap_counter += 1
            s_prime['reward'] = set_reward(s_prime) # -1 in this position
            vehicle.done_entry(s_prime, env.lap_times[0])
            ep_steps = 0 


        if done or ep_steps > conf.max_steps:
            print(f"{n}::Lap done {lap_counter} -> {env.lap_times[0]} -> Inters: {vehicle.ep_interventions}")
            lap_counter += 1
            s_prime['reward'] = set_reward(s_prime) # always lap finished=1 at this position
            vehicle.lap_complete(env.lap_times[0])
            if show:
                env.render(wait=False)

            env.data_reset()
            done = False
            ep_steps = 0 
            
        ep_steps += 1

    vehicle.planner.t_his.print_update(True)
    vehicle.planner.t_his.save_csv_data()
    vehicle.planner.agent.save(vehicle.planner.path)
    vehicle.save_intervention_list()

    train_time = time.time() - start_time
    print(f"Finished Training: {vehicle.planner.name} in {train_time} seconds")
    print(f"Crashes: {crash_counter}")

    return train_time, crash_counter



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


def set_reward(s_p):
    if s_p['collisions'][0] == 1:
        return -1
    elif s_p['lap_counts'][0] == 1:
        return 1 + (30 - s_p['lap_times'][0]) 
    return 0

