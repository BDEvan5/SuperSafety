import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

def moving_average(data, period):
    ret = np.convolve(data, np.ones(period), 'same') / period
    for i in range(period):
        t = np.convolve(data, np.ones(i+1), 'valid') / (i+1)
        ret[i] = t[0]
        ret[-i-1] = t[-1]
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

    plt.tight_layout()

    plt.savefig(f"Data/Vehicles/{agent}/baseline_training.png", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"Data/Vehicles/{agent}/baseline_training.pgf", bbox_inches='tight', pad_inches=0.1)



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

    plt.tight_layout()

    plt.savefig(f"Data/Vehicles/{agent}/kernel_training.png", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"Data/Vehicles/{agent}/kernel_training.pgf", bbox_inches='tight', pad_inches=0.1)

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

    plt.tight_layout()

    plt.savefig(f"Data/Vehicles/{agent}/kernel_laptime.png", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"Data/Vehicles/{agent}/kernel_laptime.pgf", bbox_inches='tight', pad_inches=0.1)

    # plt.show()


# generate_laptime_graph()
# generate_kernel_graph()
generate_training_graph()


