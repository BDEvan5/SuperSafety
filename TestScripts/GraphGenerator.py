import enum
from turtle import width
import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib

def turn_on_pgf():
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })

def moving_average(data, period):
    ret = np.convolve(data, np.ones(period), 'same') / period
    # for i in range(period):
    #     t = np.mean
    #     t = np.convolve(data, np.ones(i+1), 'valid') / (i+1)
    #     ret[i] = t[0]
    #     ret[-i-1] = t[-1]
    return ret

def generate_training_graph():
    # agent = "Baseline_304"
    agent = "Baseline_309"
    path = f"Data/Vehicles/{agent}/training_data.csv"

    # data = np.load(path, allow_pickle=True)
    with open(path) as file:
        reader = csv.reader(file, quoting = csv.QUOTE_NONNUMERIC)
        rewards = []
        for lines in reader:  
            if lines[1] == 0:
                break
            rewards.append(lines)
    data = np.array(rewards)[:, 1]
    # data = np.array(reward)


    plt.figure(1, figsize=(3.6, 3.2))
    plt.plot(data, color='darkblue')
    avg = moving_average(data, 20)
    plt.plot(avg, color='red', linewidth=2)
    plt.xlabel('Episode Number')
    plt.ylabel('Reward')
    plt.gca().set_aspect(10)
    plt.grid(b=True)

    plt.tight_layout()

    plt.savefig(f"Data/Vehicles/{agent}/baseline_training.png", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"Data/Graphs/baseline_training.png", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"Data/Vehicles/{agent}/baseline_training.pgf", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"Data/Graphs/baseline_training.pgf", bbox_inches='tight', pad_inches=0.1)



def generate_kernel_graph():
    agent = "KernelSSS_100"
    path = f"Data/Vehicles/{agent}/training_data.csv"

    # data = np.load(path, allow_pickle=True)
    with open(path) as file:
        reader = csv.reader(file, quoting = csv.QUOTE_NONNUMERIC)
        rewards = []
        for lines in reader:  
            if lines[1] == 0:
                break
            rewards.append(lines)
    data = np.array(rewards)[:, 1]
    # data = np.array(reward)


    plt.figure(1, figsize=(3.6, 3.2))
    plt.plot(data, '.', color='darkblue', markersize=12)
    plt.plot(data, color='red', linewidth=2)
    # avg = moving_average(data, 20)
    # plt.plot(avg, color='red', linewidth=2)
    plt.xlabel('Episode Number')
    plt.ylabel('Reward')
    plt.gca().set_aspect(0.01)
    plt.grid(b=True)

    plt.tight_layout()

    plt.savefig(f"Data/Vehicles/{agent}/kernel_training.png", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"Data/Graphs/kernel_training.png", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"Data/Vehicles/{agent}/kernel_training.pgf", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"Data/Graphs/kernel_training.pgf", bbox_inches='tight', pad_inches=0.1)

    # plt.show()


def generate_kernel_graph_step():
    agent = "KernelSSS_1_porto"
    path = f"Data/Vehicles/{agent}/step_data.csv"

    # data = np.load(path, allow_pickle=True)
    with open(path) as file:
        reader = csv.reader(file, quoting = csv.QUOTE_NONNUMERIC)
        rewards = []
        for lines in reader:  
            if lines[1] == 0:
                break
            rewards.append(lines)
    data = np.array(rewards)[:, 1]
    # data = np.array(reward)

    # new_data = moving_average(data, 10)
    d =  np.zeros(2500)
    d[:len(data)] = data
    n = 20
    d = d.reshape((-1, n))
    new_data = np.cumsum(d, axis=1)[:, -1]
    xs = np.linspace(0, 2500, int(2500/n))

    d2 = moving_average(new_data, 10)

    plt.figure(1, figsize=(3.6, 3.2))
    # plt.plot(data, '.', color='red', markersize=2)
    plt.plot(xs, new_data, color='darkblue', linewidth=2)
    plt.plot(xs, d2, color='red', linewidth=2)
    # avg = moving_average(data, 20)
    # plt.plot(avg, color='red', linewidth=2)
    plt.xlabel('Step Number')
    plt.ylabel('Reward')
    # plt.gca().set_aspect(0.01)
    plt.grid(b=True)

    plt.tight_layout()

    plt.savefig(f"Data/Vehicles/{agent}/kernel_step_training.png", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"Data/Graphs/kernel_step_training.png", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"Data/Vehicles/{agent}/kernel_step_training.pgf", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"Data/Graphs/kernel_step_training.pgf", bbox_inches='tight', pad_inches=0.1)

    # plt.show()

def generate_baseline_graph_step():
    agent = "Baseline_1_columbia_small"
    path = f"Data/Vehicles/{agent}/step_data.csv"

    # data = np.load(path, allow_pickle=True)
    with open(path) as file:
        reader = csv.reader(file, quoting = csv.QUOTE_NONNUMERIC)
        rewards = []
        for lines in reader:  
            if lines[1] == 0:
                break
            rewards.append(lines)
    data = np.array(rewards)[:, 1]
    # data = np.array(reward)

    # new_data = moving_average(data, 10)
    # d =  np.zeros(15300)
    # d[:len(data)] = data
    # n = 50
    # d = d.reshape((-1, n))
    # new_data = np.cumsum(d, axis=1)[:, -1]
    # xs = np.linspace(0, 15300, int(15300/n))
    new_data = np.cumsum(data) + 120 * np.ones_like(data)
    xs = np.linspace(0, len(data), len(data))

    d2 = moving_average(new_data, 10)

    plt.figure(1, figsize=(3.6, 3.2))
    # plt.plot(data, '.', color='red', markersize=2)
    plt.plot(xs, new_data, color='darkblue', linewidth=2)
    plt.plot(xs, d2, color='red', linewidth=2)
    # avg = moving_average(data, 20)
    # plt.plot(avg, color='red', linewidth=2)
    plt.xlabel('Step Number')
    plt.ylabel('Reward')
    # plt.gca().set_aspect(0.01)
    plt.grid(b=True)

    plt.tight_layout()

    plt.savefig(f"Data/Vehicles/{agent}/kernel_step_training.png", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"Data/Graphs/kernel_step_training.png", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"Data/Vehicles/{agent}/kernel_step_training.pgf", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"Data/Graphs/kernel_step_training.pgf", bbox_inches='tight', pad_inches=0.1)

    # plt.show()

def generate_kernel_graph_step_zoom():
    agent = "KernelSSS_1_porto"
    path = f"Data/Vehicles/{agent}/step_data.csv"

    # data = np.load(path, allow_pickle=True)
    with open(path) as file:
        reader = csv.reader(file, quoting = csv.QUOTE_NONNUMERIC)
        rewards = []
        for lines in reader:  
            if lines[1] == 0:
                break
            rewards.append(lines)
    data = np.array(rewards)[:, 1]
    # data = np.array(reward)

    new_data = moving_average(data, 10)


    plt.figure(1, figsize=(3.6, 3.2))
    plt.plot(data, '.', color='red', markersize=2)
    plt.plot(new_data, color='darkblue', linewidth=2)
    # avg = moving_average(data, 20)
    # plt.plot(avg, color='red', linewidth=2)
    plt.xlabel('Step Number')
    plt.ylabel('Reward')
    plt.ylim([-0.046, -0.03])
    # plt.gca().set_aspect(0.01)
    plt.grid(b=True)

    plt.tight_layout()

    plt.savefig(f"Data/Vehicles/{agent}/kernel_stepZoom_training.png", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"Data/Graphs/kernel_stepZoom_training.png", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"Data/Vehicles/{agent}/kernel_stepZoom_training.pgf", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"Data/Graphs/kernel_stepZoom_training.pgf", bbox_inches='tight', pad_inches=0.1)

    # plt.show()


def generate_laptime_graph():
    agent = "KernelSSS_100"
    path = f"Data/Vehicles/{agent}/{agent}_laptime_list.csv"

    # data = np.load(path, allow_pickle=True)
    with open(path) as file:
        reader = csv.reader(file, quoting = csv.QUOTE_NONNUMERIC)
        rewards = []
        for lines in reader:  
            if lines[1] == 0:
                break
            rewards.append(lines)
    data = np.array(rewards)[:, 1]
    # data = np.array(reward)


    plt.figure(1, figsize=(3.6, 3.2))
    plt.plot(data, '.', color='darkblue', markersize=12)
    plt.plot(data, color='red', linewidth=2)
    # avg = moving_average(data, 20)
    # plt.plot(avg, color='red', linewidth=2)
    plt.xlabel('Episode Number')
    plt.ylabel('Lap Time (s)')
    plt.gca().set_aspect(2)
    plt.grid(b=True)

    plt.tight_layout()

    plt.savefig(f"Data/Vehicles/{agent}/kernel_laptime.png", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"Data/Vehicles/{agent}/kernel_laptime.pgf", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"Data/Graphs/kernel_laptime.png", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"Data/Graphs/kernel_laptime.pgf", bbox_inches='tight', pad_inches=0.1)

    # plt.show()
from matplotlib.ticker import MultipleLocator
def generate_dynamics_blocks():
    dynamics = np.load(f"Data/Dynamics/viab_dyns.npy")
    print(f"Dynamics Loaded: {dynamics.shape}")

    ds = np.linspace(-0.4, 0.4, 9)
    for mode in range(9):
        dyns = dynamics[30, mode, :, 1, :]
        print(f"Dynamics Loaded: {dyns.shape}")
        print(dyns)
        angles = np.linspace(-np.pi, np.pi, 41)
        a = [angles[i] for i in dyns[:, 2]]

        spacing = 2.5
        minorLocator = MultipleLocator(spacing)

        plt.figure(1, figsize=(1.9, 1.9))
        plt.clf()
        plt.ylim([-2, 28])
        plt.gca().xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, ls='--', which='both')
        plt.plot(dyns[:, 0], dyns[:, 1], '.', color='darkblue', markersize=10)
        plt.plot(0, 0, '.', color='red', markersize=10)
        l = 2
        plt.arrow(0, 0, 0, l, head_width=0.24, head_length=0.5, fc='red', ec='red', width=0.06)
        dx = np.cos(a) * l
        dy = np.sin(a) * l
        for i in range(9):
            plt.arrow(dyns[i, 0], dyns[i, 1], dx[i], dy[i], head_width=0.24, head_length=0.5, width=0.06, fc='darkblue', ec='darkblue')
        # plt.title(f"Speed: {2}, Steering: {ds[mode]}")

        # plt.gca().set_aspect(0.25)

        # plt.xlabel('Position - dx')
        # plt.ylabel('Position - dy')

        plt.tight_layout()
        plt.savefig("Data/Modes/mode_" + str(mode) + ".png", bbox_inches='tight', pad_inches=0.0)
        plt.savefig("Data/Modes/mode_" + str(mode) + ".pgf", bbox_inches='tight', pad_inches=0.0)

import matplotlib.patches as patches
def generate_dynamics_obs_blocks():
    dynamics = np.load(f"Data/Dynamics/viab_dyns.npy")
    print(f"Dynamics Loaded: {dynamics.shape}")
    
    obsx = [2.5, 8]
    obsy = [20, 30]

    ds = np.linspace(-0.4, 0.4, 9)
    for mode in range(9):
        dyns = dynamics[30, mode, :, 1, :]
        print(f"Dynamics Loaded: {dyns.shape}")
        print(dyns)
        angles = np.linspace(-np.pi, np.pi, 41)
        a = [angles[i] for i in dyns[:, 2]]

        spacing = 2.5
        minorLocator = MultipleLocator(spacing)
        l = 2

        plt.figure(1, figsize=(1.9, 1.9))
        plt.clf()
        plt.ylim([-2, 28])
        plt.gca().xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, ls='--', which='both')

        # plt.plot(dyns[:, 0], dyns[:, 1], '.', color='darkblue', markersize=10)
        plt.plot(0, 0, '.', color='darkblue', markersize=10)
        plt.arrow(0, 0, 0, l, head_width=0.24, head_length=0.5, fc='darkblue', ec='darkblue', width=0.06)

        rect = patches.Rectangle((2.5, 20), 6, 10, linewidth=1,
                         facecolor="grey", edgecolor='black')
        plt.gca().add_patch(rect)

        dx = np.cos(a) * l
        dy = np.sin(a) * l
        for i in range(9):
            x, y = dyns[i, 0], dyns[i, 1]

            if x > 2.5:
                plt.plot(dyns[i, 0], dyns[i, 1], 'X', color='red', markersize=7)
                plt.arrow(dyns[i, 0], dyns[i, 1], dx[i], dy[i], head_width=0.24, head_length=0.5, width=0.06, fc='red', ec='red')
            else:
                plt.plot(dyns[i, 0], dyns[i, 1], '.', color='green', markersize=10)
                plt.arrow(dyns[i, 0], dyns[i, 1], dx[i], dy[i], head_width=0.24, head_length=0.5, width=0.06, fc='green', ec='green')



        plt.tight_layout()
        plt.savefig("Data/Modes/mode_obs_" + str(mode) + ".png", bbox_inches='tight', pad_inches=0.0)
        plt.savefig("Data/Modes/mode_obs_" + str(mode) + ".pgf", bbox_inches='tight', pad_inches=0.0)

    # plt.show()

from PIL import Image

def assemble_modes_picture():
    modes = [0, 4]
    
    y1 = 160
    x1 = 80

    size = 250
    b = 25
    b2 = size+b*3
    i_size = size*2 + b*4
    i_size2 = size + b*2
    img = np.zeros((i_size2, i_size, 4), dtype=np.uint8)
    pts = [[b, b], [b, b2]]

    for i, n in enumerate(modes):
        img_no = Image.open(f"Data/Modes/mode_{n}.png")
        img_n = np.asarray_chkfinite(img_no)
        img_crop = img_n[y1:y1+size, x1:x1+size]
        img[pts[i][0]:pts[i][0]+size, pts[i][1]:pts[i][1]+size] = img_crop

    img = Image.fromarray(img)
    # img.show()
    img.save(f"Data/Modes/modes_assem.png")

def build_repeat_barplot():
    path = f"Data/Results/Data_repeat.csv"

    # data = np.load(path, allow_pickle=True)
    with open(path) as file:
        reader = csv.reader(file, quoting = csv.QUOTE_ALL)
        # reader = csv.reader(file, quoting = csv.QUOTE_NONNUMERIC)
        rewards = []
        for lines in reader:  
            if lines[1] == 0:
                break
            rewards.append(lines)
    data = np.array(rewards)

    new_data = np.zeros_like(data, dtype=np.float)
    for i, entry in enumerate(data):
        for j, col in enumerate(entry):
            try:
                f = float(col)
                new_data[i, j] = f
            except: pass

    print(data)
    print(new_data)

turn_on_pgf()
# generate_laptime_graph()
# generate_kernel_graph()
# generate_training_graph()
# generate_dynamics_blocks()
# build_repeat_barplot()
# generate_dynamics_obs_blocks()

# generate_kernel_graph_step()
# generate_kernel_graph_step_zoom()
generate_baseline_graph_step()