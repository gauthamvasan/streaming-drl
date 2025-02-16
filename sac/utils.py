import json
import numpy as np
import matplotlib.pyplot as plt
from pixel_ant import VisualAntReacher

def make_env(args):
    action_repeat= args.action_repeat
    env = VisualAntReacher(action_repeat=action_repeat)
    env.name = "VisualAntReacher"
    return env


def smoothed_curve(returns, ep_lens, x_tick=5000, window_len=5000):
    """
    Args:
        returns: 1-D numpy array with episodic returs
        ep_lens: 1-D numpy array with episodic returs
        x_tick (int): Bin size
        window_len (int): Length of averaging window
    Returns:
        A numpy array
    """
    rets = []
    x = []
    cum_episode_lengths = np.cumsum(ep_lens)

    if cum_episode_lengths[-1] >= x_tick:
        y = cum_episode_lengths[-1] + 1
        steps_show = np.arange(x_tick, y, x_tick)

        for i in range(len(steps_show)):
            rets_in_window = returns[(cum_episode_lengths > max(0, x_tick * (i + 1) - window_len)) *
                                     (cum_episode_lengths < x_tick * (i + 1))]
            if rets_in_window.any():
                rets.append(np.mean(rets_in_window))
                x.append((i+1) * x_tick)

    return np.array(rets), np.array(x)
    
def smoothed_plot(data, x_tick=1000, window_len=1000):
    """
    Args:
        data: Numpy 2-D array. Row 0 contains episode lengths/timesteps and row 1 contains episodic returns
    Returns:
    """
    color = "tab:blue"
    returns = data[1]
    ep_lens = data[0]

    rets, t = smoothed_curve(returns=returns, ep_lens=ep_lens, x_tick=x_tick, window_len=window_len)
    plt.plot(t, rets, color=color, linewidth=2)
    # plt.fill_between(x, rets - std_errs, rets + std_errs, alpha=0.6)
    plt.xlabel('Timesteps', fontweight='bold', fontsize=14)
    h = plt.ylabel("Return", labelpad=25, fontweight='bold', fontsize=14)
    h.set_rotation(0)
    plt.pause(0.001)
    plt.grid()
    plt.tight_layout()
    plt.show()


def learning_curve(rets, ep_lens, save_path, x_tick=10000, window_len=10000):
    if len(rets) > 0:
        plot_rets, plot_x = smoothed_curve(np.array(rets), np.array(ep_lens), x_tick=x_tick, window_len=window_len)
        if len(plot_rets):
            plt.clf()
            plt.plot(plot_x, plot_rets)
            plt.pause(0.001)
            plt.savefig(save_path, dpi=200)


def save_returns(rets, ep_lens, save_path):
    """ Save learning curve data as a numpy text file 

    Args:
        rets (list/array): A list or array of episodic returns
        ep_lens (list/array):  A list or array of episodic length
        savepath (str): Save path
    """
    data = np.zeros((2, len(rets)))
    data[0] = ep_lens
    data[1] = rets
    np.savetxt(save_path, data)


class NpEncoder(json.JSONEncoder):
    """ 
    JSON does not like Numpy elements. Convert to native python datatypes for json dump.  
    Ref: https://bobbyhadz.com/blog/python-typeerror-object-of-type-int64-is-not-json-serializable
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_args(args, save_path):
    """ Save hyper-parameters as a json file """
    hyperparas_dict = vars(args)
    hyperparas_dict["device"] = str(hyperparas_dict["device"])
    json.dump(hyperparas_dict, open(save_path, 'w'), indent=4, cls=NpEncoder)


if __name__ == '__main__':
    # Smooth plot
    x_tick = window_len = 5000
    fp = "results/ppo_visual_reacher_bs-2048_0.txt"
    data = np.loadtxt(fp)
    smoothed_plot(data, x_tick, window_len)