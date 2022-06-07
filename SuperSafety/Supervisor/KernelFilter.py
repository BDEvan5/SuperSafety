import numpy as np
from matplotlib import pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import yaml
from PIL import Image
from SuperSafety.Utils.utils import load_conf



class TrackKernel:
    def __init__(self, sim_conf, plotting=False, kernel_name=None):
        if kernel_name is None:
            kernel_name = f"{sim_conf.kernel_path}Kernel_{sim_conf.kernel_mode}_{sim_conf.map_name}.npy"
        else:
            kernel_name = f"{sim_conf.kernel_path}{kernel_name}"
        self.kernel = np.load(kernel_name)
        self.o_kernel = np.load(kernel_name)

        # self.o_map = np.copy(track_img)    
        self.fig, self.axs = plt.subplots(2, 2)
        self.fig1, self.axs1 = plt.subplots(2, 2)
        self.phis = np.linspace(-np.pi, np.pi, sim_conf.n_phi)

    def filter_kernel(self):
        print(f"Starting to filter: {np.count_nonzero(self.kernel)} -->{np.size(self.kernel)- np.count_nonzero(self.kernel)} --> {self.kernel.shape}")
        xs, ys, ths, ms = self.kernel.shape
        new_kernel = np.zeros((xs, ys, ths, 1), dtype=bool)
        new_kernel = filter_kernel(self.kernel, new_kernel)
        # self.kernel = np.bitwise_xor(self.kernel, new_kernel)
        self.kernel = new_kernel
        print(f"finished filtering: {np.count_nonzero(self.kernel)} --> {self.kernel.shape}")


    def view_kernel_angles(self, show=True):
        self.fig, self.axs = plt.subplots(2, 2)
        self.axs[0, 0].cla()
        self.axs[1, 0].cla()
        self.axs[0, 1].cla()
        self.axs[1, 1].cla()

        half_phi = int(20)
        quarter_phi = int(10)

        mode_ind = 0

        self.axs[0, 0].imshow(self.kernel[:, :, 0, mode_ind].T , origin='lower')
        self.axs[0, 0].set_title(f"Kernel phi: {self.phis[0]}")
        self.axs[1, 0].imshow(self.kernel[:, :, half_phi, mode_ind].T, origin='lower')
        self.axs[1, 0].set_title(f"Kernel phi: {self.phis[half_phi]}")
        self.axs[0, 1].imshow(self.kernel[:, :, -quarter_phi, mode_ind].T , origin='lower')
        self.axs[0, 1].set_title(f"Kernel phi: {self.phis[-quarter_phi]}")
        self.axs[1, 1].imshow(self.kernel[:, :, quarter_phi, mode_ind].T , origin='lower')
        self.axs[1, 1].set_title(f"Kernel phi: {self.phis[quarter_phi]}")

        plt.pause(0.0001)
        # plt.pause(1)

        if show:
            plt.show()

    def view_kernel_angles1(self, show=True):
        self.axs1[0, 0].cla()
        self.axs1[1, 0].cla()
        self.axs1[0, 1].cla()
        self.axs1[1, 1].cla()

        half_phi = int(20)
        quarter_phi = int(10)

        mode_ind = 0

        self.axs1[0, 0].imshow(self.kernel[:, :, 0, mode_ind].T , origin='lower')
        self.axs1[0, 0].set_title(f"Kernel phi: {self.phis[0]}")
        self.axs1[1, 0].imshow(self.kernel[:, :, half_phi, mode_ind].T, origin='lower')
        self.axs1[1, 0].set_title(f"Kernel phi: {self.phis[half_phi]}")
        self.axs1[0, 1].imshow(self.kernel[:, :, -quarter_phi, mode_ind].T , origin='lower')
        self.axs1[0, 1].set_title(f"Kernel phi: {self.phis[-quarter_phi]}")
        self.axs1[1, 1].imshow(self.kernel[:, :, quarter_phi, mode_ind].T , origin='lower')
        self.axs1[1, 1].set_title(f"Kernel phi: {self.phis[quarter_phi]}")

        plt.pause(0.0001)
        # plt.pause(1)

        # if show:
        #     plt.show()

@njit(cache=True)
def filter_kernel(kernel, new_kernel):
    xs, ys, ths, ms = kernel.shape
    assert ms > 2, "Single Use kernels..."
    for i in range(xs):
        for j in range(ys):
            for k in range(ths):
                new_kernel[i, j, k, 0] = kernel[i, j, k, :].any()
            
    return new_kernel


if __name__ == "__main__":
    sim_conf = load_conf("config_file")
    k = TrackKernel(sim_conf)
    k.view_kernel_angles1()
    k.filter_kernel()
    name = f"Kernel_filter_{sim_conf.map_name}"
    np.save(f"{sim_conf.kernel_path}{name}.npy", k.kernel)
    print(f"Saved kernel to file: {name}")
    k.view_kernel_angles()

