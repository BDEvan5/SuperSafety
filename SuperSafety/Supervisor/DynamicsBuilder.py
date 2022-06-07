
from SuperSafety.Utils.utils import limit_phi, load_conf

from SuperSafety.Supervisor.Dynamics import run_dynamics_update

import numpy as np
from matplotlib import pyplot as plt

import numpy as np
from numba import njit


class Modes:
    def __init__(self, conf) -> None:
        self.time_step = conf.kernel_time_step
        self.nq_steer = conf.nq_steer
        self.max_steer = conf.max_steer
        vehicle_speed = conf.vehicle_speed

        ds = np.linspace(-self.max_steer, self.max_steer, self.nq_steer)
        vs = vehicle_speed * np.ones_like(ds)
        self.qs = np.stack((ds, vs), axis=1)

        self.n_modes = len(self.qs)

    def get_mode_id(self, delta):
        d_ind = np.argmin(np.abs(self.qs[:, 0] - delta))
        
        return int(d_ind)

    def action2mode(self, action):
        id = self.get_mode_id(action[0])
        return self.qs[id]

    def __len__(self): return self.n_modes


class SingleMode:
    def __init__(self, conf) -> None:
        self.qs = np.array([[0, conf.vehicle_speed]])
        self.n_modes = 1

    def get_mode_id(self, delta):
        return 0

    def action2mode(self, action):
        return self.qs[0]

    def __len__(self): return self.n_modes


def generate_dynamics_entry(state, action, m, time, resolution, phis):
    dyns = np.zeros(4)
    new_state = run_dynamics_update(state, action, time)
    dx, dy, phi, vel, steer = new_state[0], new_state[1], new_state[2], new_state[3], new_state[4]
    new_q = m.get_mode_id(steer)

    phi = limit_phi(phi)
    new_k = int(round((phi + np.pi) / (2*np.pi) * (len(phis)-1)))
    dyns[2] = min(max(0, new_k), len(phis)-1)
    dyns[0] = int(round(dx * resolution))                  
    dyns[1] = int(round(dy * resolution))                  
    dyns[3] = int(new_q)       

    return dyns           




# @njit(cache=True)
def build_viability_dynamics(state_m, act_m, conf):
    phis = np.linspace(-np.pi, np.pi, conf.n_phi)

    ns = conf.n_intermediate_pts
    dt = conf.kernel_time_step / ns

    dynamics = np.zeros((len(phis), len(state_m), len(act_m), ns, 4), dtype=np.int)
    invalid_counter = 0
    for i, p in enumerate(phis):
        for j, state_mode in enumerate(state_m.qs): # searches through old q's
            state = np.array([0, 0, p, state_mode[1], state_mode[0]])
            for k, action in enumerate(act_m.qs): # searches through actions

                for l in range(ns):
                    dynamics[i, j, k, l] = generate_dynamics_entry(state.copy(), action, state_m, dt*(l+1), conf.n_dx, phis)                                
                
    print(f"Invalid transitions: {invalid_counter}")
    print(f"Dynamics Table has been built: {dynamics.shape}")

    return dynamics




def build_dynamics_table(sim_conf):
    # if sim_conf.no_steer:
    #     state_m = SingleMode(sim_conf)
    # else: state_m = Modes(sim_conf)
    state_m = Modes(sim_conf)
    act_m = Modes(sim_conf)

    if sim_conf.kernel_mode == "viab":
        dynamics = build_viability_dynamics(state_m, act_m, sim_conf)
    else:
        raise ValueError(f"Unknown kernel mode: {sim_conf.kernel_mode}")


    np.save(f"{sim_conf.dynamics_path}{sim_conf.kernel_mode}_dyns.npy", dynamics)



if __name__ == "__main__":
    conf = load_conf("config_file")

    build_dynamics_table(conf)

