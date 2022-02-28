import numpy as np
from matplotlib import pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import yaml
from PIL import Image
from SuperSafety.Utils.utils import load_conf
from SuperSafety.Supervisor.KernelGenerator import viability_loop, check_viable_state, prepare_track_img, shrink_img

class KernelGenerator:
    def __init__(self, track_img, sim_conf):
        self.track_img = track_img
        self.sim_conf = sim_conf
        self.n_dx = int(sim_conf.n_dx)
        self.t_step = sim_conf.kernel_time_step
        self.n_phi = sim_conf.n_phi
        self.max_steer = sim_conf.max_steer 
        self.L = sim_conf.l_f + sim_conf.l_r

        self.n_x = self.track_img.shape[0]
        self.n_y = self.track_img.shape[1]
        self.xs = np.linspace(0, self.n_x/self.n_dx, self.n_x) 
        self.ys = np.linspace(0, self.n_y/self.n_dx, self.n_y)
        self.phis = np.linspace(-np.pi, np.pi, self.n_phi)
        
        self.n_modes = sim_conf.nq_steer 
        self.qs = np.linspace(-self.max_steer, self.max_steer, self.n_modes)

        self.o_map = np.copy(self.track_img)    
        self.fig, self.axs = plt.subplots(2, 2)

        self.kernel = np.zeros((self.n_x, self.n_y, self.n_phi, self.n_modes))
        self.previous_kernel = np.copy(self.kernel)

        self.kernel[:, :, :, :] = self.track_img[:, :, None, None] * np.ones((self.n_x, self.n_y, self.n_phi, self.n_modes))
        
        self.dynamics = np.load(f"{sim_conf.dynamics_path}{sim_conf.kernel_mode}_dyns.npy")
        print(f"Dynamics Loaded: {self.dynamics.shape}")

        self.iteration = 0

    def get_filled_kernel(self):
        filled = np.count_nonzero(self.kernel)
        total = self.kernel.size
        print(f"Filled: {filled} / {total} -> {filled/total}")
        return filled/total

    def save_build_picture(self):
        plt.figure(2)
        plt.clf()
        mode = 4 
        angle = int(len(self.phis)/2)
        plt.imshow(self.kernel[:, :, angle, mode].T + self.o_map.T, origin='lower')
        path = "Data/Images/"
        plt.xticks([])
        plt.yticks([])
        plt.savefig(path + f"Kernel_{self.sim_conf.map_name}_{self.iteration}.png")


    def calculate_kernel(self, n_loops=20):
        for z in range(n_loops):
            print(f"Running loop: {z}")
            self.iteration +=1
            if np.all(self.previous_kernel == self.kernel):
                print("Kernel has not changed: convergence has been reached")
                break
            self.previous_kernel = np.copy(self.kernel)
            self.kernel = viability_loop(self.kernel, self.dynamics)

            self.get_filled_kernel()
            self.save_build_picture()

        return self.get_filled_kernel()



class VeiwKernel:
    def __init__(self, conf, track_img):
        kernel_name = f"{conf.kernel_path}Kernel_{conf.kernel_mode}_{conf.map_name}.npy"
        self.kernel = np.load(kernel_name)
        self.conf = conf

        self.o_map = np.copy(track_img)    
        self.fig, self.axs = plt.subplots(2, 2)

        self.phis = np.linspace(-conf.phi_range/2, conf.phi_range/2, conf.n_phi)

        self.qs = np.linspace(-conf.max_steer, conf.max_steer, conf.nq_steer)
        # self.view_speed_build(True)
     
    def view_speed_build(self, show=True):
        self.axs[0, 0].cla()
        self.axs[1, 0].cla()
        self.axs[0, 1].cla()
        self.axs[1, 1].cla()

        phi_ind = int(len(self.phis)/2)

        inds = np.array([3, 4, 7, 8], dtype=int)

        self.axs[0, 0].imshow(self.kernel[:, :, phi_ind, inds[0]].T + self.o_map.T, origin='lower')
        self.axs[0, 0].set_title(f"Kernel Mode: {self.qs[inds[0]]}")
        self.axs[1, 0].imshow(self.kernel[:, :, phi_ind, inds[1]].T + self.o_map.T, origin='lower')
        self.axs[1, 0].set_title(f"Kernel Mode: {self.qs[inds[1]]}")
        self.axs[0, 1].imshow(self.kernel[:, :, phi_ind, inds[2]].T + self.o_map.T, origin='lower')
        self.axs[0, 1].set_title(f"Kernel Mode: {self.qs[inds[2]]}")

        self.axs[1, 1].imshow(self.kernel[:, :, phi_ind, inds[3]].T + self.o_map.T, origin='lower')
        self.axs[1, 1].set_title(f"Kernel Mode: {self.qs[inds[3]]}")

        plt.pause(0.0001)
        plt.pause(1)

        if show:
            plt.show()
        
    def make_picture(self, show=True):
        self.axs[0, 0].cla()
        self.axs[1, 0].cla()
        self.axs[0, 1].cla()
        self.axs[1, 1].cla()

        half_phi = int(len(self.phis)/2)
        quarter_phi = int(len(self.phis)/4)


        self.axs[0, 0].set(xticks=[])
        self.axs[0, 0].set(yticks=[])
        self.axs[1, 0].set(xticks=[])
        self.axs[1, 0].set(yticks=[])
        self.axs[0, 1].set(xticks=[])
        self.axs[0, 1].set(yticks=[])
        self.axs[1, 1].set(xticks=[])
        self.axs[1, 1].set(yticks=[])

        self.axs[0, 0].imshow(self.kernel[:, :, 0].T + self.o_map.T, origin='lower')
        self.axs[1, 0].imshow(self.kernel[:, :, half_phi].T + self.o_map.T, origin='lower')
        self.axs[0, 1].imshow(self.kernel[:, :, -quarter_phi].T + self.o_map.T, origin='lower')
        self.axs[1, 1].imshow(self.kernel[:, :, quarter_phi].T + self.o_map.T, origin='lower')
        
        plt.pause(0.0001)
        plt.pause(1)
        plt.savefig(f"{self.sim_conf.kernel_path}Kernel_build_{self.sim_conf.kernel_mode}.svg")

        if show:
            plt.show()

    def save_kernel(self, name):

        self.view_speed_build(False)
        plt.savefig(f"{self.sim_conf.kernel_path}KernelSpeed_{name}_{self.sim_conf.kernel_mode}.png")

        self.view_angle_build(False)
        plt.savefig(f"{self.sim_conf.kernel_path}KernelAngle_{name}_{self.sim_conf.kernel_mode}.png")

    def save_mode_picture(self):
        plt.figure(3)
        plt.clf()
        angle = int(len(self.phis)/2)
        modes = [0, 8]

        for m in modes:
            plt.imshow(self.kernel[:, :, angle, m].T + self.o_map.T, origin='lower')
            plt.savefig(f"Data/Images/Mode_{m}_{self.conf.map_name}.png")


    def save_angle_picture(self):
        plt.figure(3)
        plt.clf()
        mode = 4
        # angles = [0, 10, 15, 20, 25, 30, 40]
        # angles = [12, 16, 20, 24]
        # angles = [12, 15, 18, 21]
        angles = [12, 14, 16, 18]

        for a in angles:
            plt.imshow(self.kernel[:, :, a, mode].T + self.o_map.T, origin='lower')
            plt.savefig(f"Data/Images/Angle_{a}_{self.conf.map_name}.png")


from PIL import Image

def assemble_growth_stages_picture():
    numbers = np.array([1, 3, 8, 25])
    map_name = "porto"

    
    y1 = 165
    x1 = 410

    size = 160
    b = 5
    b2 = size+b*3
    i_size = size*2 + b*4
    img = np.zeros((i_size, i_size, 4), dtype=np.uint8)
    pts = [[b, b], [b2, b], [b, b2], [b2, b2]]

    for i, n in enumerate(numbers):
        img_no = Image.open(f"Data/Images/Kernel_{map_name}_{n}.png")
        img_n = np.asarray_chkfinite(img_no)
        img_crop = img_n[y1:y1+size, x1:x1+size]
        img[pts[i][0]:pts[i][0]+size, pts[i][1]:pts[i][1]+size] = img_crop

    img = Image.fromarray(img)
    # img.show()
    img.save(f"Data/Images/Kernel_{map_name}_growth.png")


def assemble_angles_picture():
    # angles = np.array([10, 15, 20, 25])
    # angles = [12, 16, 20, 24]
    # angles = [12, 15, 18, 21]
    angles = [12, 14, 16, 18]
    map_name = "columbia_small"
    
    y1 = 160
    x1 = 80

    size = 250
    b = 5
    b2 = size+b*3
    i_size = size*2 + b*4
    img = np.zeros((i_size, i_size, 4), dtype=np.uint8)
    pts = [[b, b], [b2, b], [b, b2], [b2, b2]]

    for i, n in enumerate(angles):
        img_no = Image.open(f"Data/Images/Angle_{n}_{map_name}.png")
        img_n = np.asarray_chkfinite(img_no)
        img_crop = img_n[y1:y1+size, x1:x1+size]
        img[pts[i][0]:pts[i][0]+size, pts[i][1]:pts[i][1]+size] = img_crop

    img = Image.fromarray(img)
    # img.show()
    img.save(f"Data/Images/Kernel_{map_name}_angles.png")



def assemble_modes_picture():
    modes = [8, 0]
    map_name = "columbia_small"
    
    y1 = 160
    x1 = 80

    size = 250
    b = 5
    b2 = size+b*3
    i_size = size*2 + b*4
    i_size2 = size + b*2
    img = np.zeros((i_size2, i_size, 4), dtype=np.uint8)
    pts = [[b, b], [b, b2]]

    for i, n in enumerate(modes):
        img_no = Image.open(f"Data/Images/Mode_{n}_{map_name}.png")
        img_n = np.asarray_chkfinite(img_no)
        img_crop = img_n[y1:y1+size, x1:x1+size]
        img[pts[i][0]:pts[i][0]+size, pts[i][1]:pts[i][1]+size] = img_crop

    img = Image.fromarray(img)
    # img.show()
    img.save(f"Data/Images/Kernel_{map_name}_modes.png")



def build_track_kernel(conf):
    img = prepare_track_img(conf) 
    img, img2 = shrink_img(img, conf.track_shrink_pixels)
    kernel = KernelGenerator(img2, conf)
    kernel.calculate_kernel(100)



def view_kernel():
    conf = load_conf("config_file")
    img = prepare_track_img(conf) 
    img, img2 = shrink_img(img, 5)
    k = VeiwKernel(conf, img2)
    k.save_mode_picture()
    k.save_angle_picture()

if __name__ == "__main__":

    # conf = load_conf("config_file")
    # conf.map_name = "porto"
    # build_track_kernel(conf)
# 
    assemble_growth_stages_picture()
    # view_kernel()
    # assemble_angles_picture()
    # assemble_modes_picture()



