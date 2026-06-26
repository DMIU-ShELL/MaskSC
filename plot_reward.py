import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import os
import glob
import argparse
from collections import OrderedDict
from copy import deepcopy
import seaborn as sns
import csv
import pickle

def save_plot_data_to_csv(master, output_dir='./log/plots/plot_data/'):
    os.makedirs(output_dir, exist_ok=True)

    for exp_name, data in master.items():
        filename = f"{output_dir}/{exp_name.replace(' ', '_')}_curve.csv"
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Iteration', 'MeanReturn', 'ConfidenceInterval'])
            for x, y, cfi in zip(data['xdata'], data['ydata'], data['ydata_cfi']):
                writer.writerow([x, y, cfi])

def export_cumulative_to_csv(master, cumulative_return, cumulative_cfi, xdata, output_path='log/plots/summed_return.csv'):
    import csv
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Iteration', 'TotalAverageReturn', 'TotalConfidenceInterval'])
        for epoch, avg, cfi in zip(xdata, cumulative_return, cumulative_cfi):
            writer.writerow([epoch, avg, cfi])

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def cfi_delta(data, conf_int_param=0.95): # confidence interval
    mean = np.mean(data, axis=1)
    if data.ndim == 1:
        std_error_of_mean = st.sem(data, axis=1)
        lb, ub = st.t.interval(conf_int_param, df=len(data)-1, loc=mean, scale=std_error_of_mean)
        cfi_delta = ub - mean
    elif data.ndim == 2:
        std_error_of_mean = st.sem(data, axis=1)
        #print(std_error_of_mean)
        lb, ub = st.t.interval(conf_int_param, df=data.shape[0]-1, loc=mean, scale=std_error_of_mean)
        cfi_delta = ub - mean
        cfi_delta[np.isnan(cfi_delta)] = 0.
    else:
        raise ValueError('`data` with > 2 dim not expected. Expect either a 1 or 2 dimensional tensor.')
    return cfi_delta

def plot(master, title='', xaxis_label='Iteration', yaxis_label='Return'):
    #fig = plt.figure(figsize=(25, 6))  # For wide graph
    fig = plt.figure(figsize=(30, 6))
    ax = fig.subplots()

    ax.set_xlabel(xaxis_label)
    ax.xaxis.label.set_fontsize(35) # Originally 30
    ax.set_ylabel(yaxis_label)
    ax.yaxis.label.set_fontsize(35) # Originally 30
    ax.set_ylim(0, 1.0)
    # axis ticks
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.tick_params(axis='both', which='major', labelsize=35)
    # remove right and top spines
    #ax.spines['right'].set_visible(False)
    #ax.spines['top'].set_visible(False)
    # set left and bottom spines at (0, 0) co-ordinate
    #ax.spines['left'].set_position(('data', 0.0))
    #ax.spines['right'].set_position(('data', 0.0))
    # draw dark line at the (0, 0) co-ordinate
    #ax.axhline(y=-0.1, color='k')
    #ax.axvline(x=0, color='k')
    # set grid lines
    ax.grid(True, which='both')
        
    for method_name, result_dict in master.items():
        
        xdata = result_dict['xdata']
        ydata = result_dict['ydata']
        cfi = result_dict['ydata_cfi']
        plot_colour = result_dict['plot_colour']
        ax.plot(xdata, ydata, linewidth=3, label=method_name, alpha=0.5)
        ax.fill_between(xdata, ydata - cfi, ydata + cfi, alpha=0.2)
    # legend
    ax.legend(loc='lower center', prop={'size': 25}, ncols=7, bbox_to_anchor=(0.5, -0.51))
    return fig

def staircase_transform(master, task_len=100):
    """
    Transform per-iteration returns into a staircase curve by offsetting each task
    segment by the cumulative final performance of all previous tasks.
    """
    master_stair = deepcopy(master)
    for method_name, result_dict in master.items():
        y = np.asarray(result_dict['ydata'])
        if y.size == 0:
            continue
        y_stair = y.copy()

        n = len(y)
        n_tasks = (n + task_len - 1) // task_len

        final_values = []
        for t in range(n_tasks):
            end_idx = min((t + 1) * task_len, n) - 1
            final_values.append(y[end_idx])

        offsets = np.concatenate(([0.0], np.cumsum(final_values[:-1])))

        for t in range(n_tasks):
            start = t * task_len
            end = min((t + 1) * task_len, n)
            y_stair[start:end] = y[start:end] + offsets[t]

        master_stair[method_name]['ydata'] = y_stair
    return master_stair

def plot_staircase(master, title='', xaxis_label='Iteration', yaxis_label='Cumulative Task Return'):
    fig = plt.figure(figsize=(7, 4))
    ax = fig.subplots()

    ax.set_xlabel(xaxis_label)
    ax.xaxis.label.set_fontsize(20)
    ax.set_ylabel(yaxis_label)
    ax.yaxis.label.set_fontsize(20)

    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, which='both')

    max_y = 0.0
    for method_name, result_dict in master.items():
        xdata = result_dict['xdata']
        ydata = result_dict['ydata']
        cfi = result_dict['ydata_cfi']
        ax.plot(xdata, ydata, linewidth=3, label=method_name, alpha=0.5)
        ax.fill_between(xdata, ydata - cfi, ydata + cfi, alpha=0.2)
        if len(ydata) > 0:
            max_y = max(max_y, np.max(ydata))

    if max_y > 0:
        ax.set_ylim(0, max_y * 1.05)

    ax.legend(loc='lower center', prop={'size': 14}, ncols=3, bbox_to_anchor=(0.5, -0.8))
    return fig

def plot_sum(fig, ax, master, title='', xaxis_label='Iteration', yaxis_label='Summed Return'):
    """
    This function creates a visualization of the cumulative return (sum of average returns) 
    across iterations for multiple experiments in the provided data structure.

    Args:
        master (dict): A dictionary containing data for each experiment.
            - Key: Name of the experiment (method_name)
            - Value: A dictionary containing:
                - xdata (np.array): Iteration numbers.
                - ydata (np.array): Average return across seed runs for each iteration.
                - ydata_cfi (np.array): Confidence interval for the average return at each iteration.
                - plot_colour (str): Color to be used for plotting the experiment's results.
        title (str, optional): Title for the plot (defaults to '').
        xaxis_label (str, optional): Label for the x-axis (defaults to 'Iteration').
        yaxis_label (str, optional): Label for the y-axis (defaults to 'Cumulative Return').
    """

    # Initialize cumulative return (zeros for the same shape as one experiment's ydata)
    cumulative_return = np.zeros_like(master[list(master.keys())[0]]['ydata'])
    cumulative_cfi = np.zeros_like(cumulative_return)


    # Create output directory
    os.makedirs('./log/plots/summed_returns/', exist_ok=True)

    # Loop through experiments and plot individual lines with confidence interval fill
    for method_name, result_dict in master.items():
        xdata = result_dict['xdata']
        ydata = result_dict['ydata']
        cfi = result_dict['ydata_cfi']
        plot_colour = result_dict['plot_colour']

        #ax.plot(xdata, ydata, linewidth=3, label=method_name, alpha=0.5)
        #ax.fill_between(xdata, ydata - cfi, ydata + cfi, alpha=0.2, color=plot_colour)

        # Add current experiment's average return to cumulative return
        cumulative_return += result_dict['ydata']
        # Add current experiment's CFI to cumulative CFI (element-wise)
        cumulative_cfi += result_dict['ydata_cfi']

    output_path = f'./log/plots/summed_returns/{title.replace(" ", "_")}_summed.csv'
    with open(output_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Iteration', 'SummedAverageReturn'])
        for epoch, value in zip(xdata, cumulative_return):
            writer.writerow([epoch, value])
        
        writer.writerow(['max', max(cumulative_return)])
        writer.writerow(['min', min(cumulative_return)])

        # Compute milestone epochs
        max_perf = 14
        milestones = {
            'first_nonzero_epoch': next((i for i, val in enumerate(cumulative_return) if val > 0), None),
            'epoch_25_percent': next((i for i, val in enumerate(cumulative_return) if val >= 0.25 * max_perf), None),
            'epoch_50_percent': next((i for i, val in enumerate(cumulative_return) if val >= 0.50 * max_perf), None),
            'epoch_75_percent': next((i for i, val in enumerate(cumulative_return) if val >= 0.75 * max_perf), None),
        }

        # Append milestone data to CSV
        writer.writerow(['first_epoch_gt_0', milestones['first_nonzero_epoch']])
        writer.writerow(['epoch_25_percent', milestones['epoch_25_percent']])
        writer.writerow(['epoch_50_percent', milestones['epoch_50_percent']])
        writer.writerow(['epoch_75_percent', milestones['epoch_75_percent']])

    # Plot the cumulative return line
    ax.plot(xdata, cumulative_return, linewidth=3, label=title)  # Adjust color as needed
    ax.fill_between(xdata, cumulative_return - cumulative_cfi/2, cumulative_return + cumulative_cfi/2, alpha=0.2)  # Adjust color as needed

    # Legend
    ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.5),  # tweak -0.2 to fit under x-axis label
        ncol=3, 
        prop={'size': 25}
    )

    return fig, ax

def plot_seperate(fig, axs, master, plot_name, title='', xaxis_label='Iteration', yaxis_label='Summed Return'):
    for ax, (method_name, result_dict) in zip(axs.flat, master.items()):
        xdata = result_dict['xdata']
        ydata = result_dict['ydata']
        cfi = result_dict['ydata_cfi']
        plot_colour = result_dict['plot_colour']

        ax.plot(xdata, ydata, linewidth=3, label=plot_name, alpha=0.5)
        ax.fill_between(xdata, ydata - cfi, ydata + cfi, alpha=0.2)
        ax.set_title(method_name)
        ax.grid(True)
        #ax.set_ylim(0.0, 1.05)
        #ax.set_xlim(-0.05, 200)

    #axs.flat[0].legend(
    #    loc='upper center', 
    #    bbox_to_anchor=(0.5, 2.0),  # tweak -0.2 to fit under x-axis label
    #    ncol=5, 
    #    prop={'size': 15}
    #)

    supxlabel = fig.supxlabel('Iteration')
    supylabel = fig.supylabel('Return')

    return fig, axs

def plot_box(fig, ax, master, title='', xaxis_label='Iteration', yaxis_label='Summed Return'):
    return

def assess_policy_stability(rewards, window_size=10, threshold_ratio=0.95):
    """
    This function assesses the stability of an RL policy based on moving average reward and a threshold ratio of the maximum reward.

    Args:
        rewards: A list of rewards obtained during training (length 199 in your case).
        window_size: The window size for calculating the moving average reward.
        threshold_ratio: The threshold ratio of the maximum observed reward for assessing stability.

    Returns:
        A dictionary containing:
            - stable_timestep: The timestep at which the policy is considered stable (None if not found).
            - sample_efficiency: None (not calculated in this version).
    """
    results = {"stable_timestep": None}

    # Calculate maximum reward (assuming all rewards are positive)
    max_reward = np.max(rewards)

    # Calculate moving average reward
    moving_average_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')

    # Identify potential stable periods
    stable_periods = []
    threshold_reward = max_reward * threshold_ratio  # Dynamic threshold based on max_reward
    for i in range(len(moving_average_rewards) - 1):
        if moving_average_rewards[i] >= threshold_reward and moving_average_rewards[i + 1] >= threshold_reward:
            stable_periods.append((i, i + 1))

    # Check if any stable periods exist
    if not stable_periods:
        return results

    # Identify the most recent stable period with the longest duration
    longest_stable_period = max(stable_periods, key=lambda p: p[1] - p[0])
    results["stable_timestep"] = longest_stable_period[0]

    return results

def plot_tra(master, title='', xaxis_label='', yaxis_label=''):
    results = {}
    names = []
    for experiment_name, experiment_data in master.items():
        results[experiment_name] = {}
        for name, data, in experiment_data.items():
            max_reward = np.amax(data['ydata'])

            rewards = data['ydata']
            for i, reward in enumerate(rewards):
                if reward >= 0.8 * max_reward:
                    max_index = data['xdata'][i]
                    break
            #print(experiment_name, name, max_index)

            if max_index == 0: max_index = np.amax(data['xdata'])

            names.append(name)
            results[experiment_name][name] = {
                "max_y" : max_reward,
                "x_at_max_y" : max_index,
                "max_index_diff" : None
            }

    x_data = []
    y_data = []

    names = list(OrderedDict.fromkeys(names))
    #names = list(sorted(set(names)))
    experiments = list(results.keys())

    #print(names)

    for name in names:
        for exp1 in experiments:
            if exp1 == 'Isolated agents':
                for exp2 in experiments:
                    if exp2 == 'C3L':
                        value1 = results[exp1][name]["x_at_max_y"]
                        value2 = results[exp2][name]["x_at_max_y"]
                        diff = value1 - value2
                        if diff >= 0:
                            #print(exp1, exp2, name, diff, value1, value2)
                            results[exp1][name]["max_index_diff"] = diff
                            y_data.append(diff)
                            x_data.append(name)

    #print(x_data)
    #print(y_data)

    fig = plt.figure(figsize=(30, 6))
    ax = fig.subplots()
    
    ax.bar(x_data, y_data)

    # Calculate the y-axis offset for text placement (adjust as needed)
    y_offset = 0.1

    # Loop through data and add text annotations above each bar
    for i, value in enumerate(y_data):
        ax.text(x_data[i], value + y_offset, str(value), ha='center', va='bottom', fontsize=12)  # Adjust ha, va, and fontsize as needed
    

    ax.set_xlabel("Task")
    ax.set_ylabel("Time Reduction Advantage (TRA)")
    ax.tick_params(axis='x', rotation=90)
    fig.savefig(f'./log/plots/tra.pdf', dpi=256, format='pdf', bbox_inches='tight')

# NEURIPS
neurips_mctgraph = {
    'Finetuning (PPO)' : {
        'Finetuning (PPO)' : 'raw/ct28-interleaved-PPO/Titerationreward/',
    },

    'Mask-RI' : {
        'Mask-RI' : 'raw/ct28-interleaved-MaskRI/Titerationreward/',
    },

    'Mask-SC-oracle' : {
        'Mask-SC-oracle' : 'raw/ct28-interleaved-MaskSC-oracle/Titerationreward/',
    },

    'oEWC (MH)' : {
        'oEWC (MH)' : 'raw/ct28-interleaved-EWC-MH/Titerationreward/',
    },

    'SI (MH)' : {
        'SI (MH)' : 'raw/ct28-interleaved-SI-MH/Titerationreward/',
    },

    'CLEAR' : {
        'CLEAR' : 'raw/ct28-interleaved-CLEAR/Titerationreward/',
    },

    'SER' : {
        'SER' : 'raw/ct28-interleaved-SER/Titerationreward/',
    },

    'Mask-LC' : {
        'Mask-LC' : 'raw/ct28-interleaved-MaskLC/Titerationreward/',
    },

    'Mask-BLC' : {
        'Mask-BLC' : 'raw/ct28-interleaved-MaskBLC/Titerationreward/',
    },

    'CLHNet' : {
        'CLHNet' : 'raw/ct28-interleaved-CLHNET/Titerationreward/',
    },

    'PNN' : {
        'PNN' : 'raw/ct28-interleaved-PNN/Titerationreward/',
    },

    'CKA-RL' : {
        'CKA-RL' : 'raw/ct28-interleaved-CKA/Titerationreward/',
    },

    'SDW' : {
        'SDW' : 'raw/ct28-interleaved-SDW/Titerationreward/',
    },

    'Mask-SC (k=3)' : {
        'Mask-SC (k=3)' : 'raw/ct28-interleaved-MaskSC-3/Titerationreward/',
    }
}

neurips_mctgraph_causation_ablation = {
    #'Mask-RI (No Priors)' : {
    #    'Mask-RI (No Priors)' : 'raw/ct28-interleaved-MaskRI/Titerationreward/',
    #},

    'Mask-LC' : {
        'Mask-LC' : 'raw/ct28-interleaved-MaskLC/Titerationreward/',
    },

    'Random-3' : {
        'Random-3' : 'raw/ct28-interleaved-MaskSC-random-3/Titerationreward/',
    },

    'Mask-SC (oracle)' : {
        'Mask-SC (oracle)' : 'raw/ct28-interleaved-MaskSC-oracle/Titerationreward/',
    },

    'Mask-SC (k=3)' : {
        'Mask-SC (k=3)' : 'raw/ct28-interleaved-MaskSC-3/Titerationreward/',
    },
}

neurips_mctgraph_grid_search = {
    'Mask-SC (k=1)' : {
        'Mask-SC (k=1)' : 'raw/ct28-interleaved-MaskSC-1/Titerationreward/',
    },
    
    'Mask-SC (k=3)' : {
        'Mask-SC (k=3)' : 'raw/ct28-interleaved-MaskSC-3/Titerationreward/',
    },

    'Mask-SC (k=5)' : {
        'Mask-SC (k=5)' : 'raw/ct28-interleaved-MaskSC-5/Titerationreward/',
    },

    'Mask-SC (k=7)' : {
        'Mask-SC (k=7)' : 'raw/ct28-interleaved-MaskSC-7/Titerationreward/',
    },

    'Mask-SC (uncapped)' : {
        'Mask-SC (uncapped)' : 'raw/ct28-interleaved-MaskSC-all/Titerationreward/',
    },
}

neurips_mctgraph_grid_search_perf = {
    'Mask-SC-p (k=1)' : {
        'Mask-SC-p (k=1)' : 'raw/ct28-interleaved-MaskSC-perf-1/Titerationreward/',
    },
    
    'Mask-SC-p (k=3)' : {
        'Mask-SC-p (k=3)' : 'raw/ct28-interleaved-MaskSC-perf-3/Titerationreward/',
    },

    'Mask-SC-p (k=5)' : {
        'Mask-SC-p (k=5)' : 'raw/ct28-interleaved-MaskSC-perf-5/Titerationreward/',
    },

    'Mask-SC-p (k=7)' : {
        'Mask-SC-p (k=7)' : 'raw/ct28-interleaved-MaskSC-perf-7/Titerationreward/',
    },

    'Mask-SC-p (uncapped)' : {
        'Mask-SC-p (uncapped)' : 'raw/ct28-interleaved-MaskSC-perf-all/Titerationreward/',
    },

    'Mask-SC (k=3)' : {
        'Mask-SC (k=3)' : 'raw/ct28-interleaved-MaskSC-3/Titerationreward/',
    },
}

neurips_mctgraph_fdet_fsel_gridsearch = {
    'f_det=1 f_sel=1' : {
        'f_det=1 f_sel=1' : 'raw/ct28-interleaved-MaskSC-3/Titerationreward/',
    },
    
    'f_det=5 f_sel=1' : {
        'f_det=5 f_sel=1' : 'raw/ct28-interleaved-MaskSC-3-fdet_5/Titerationreward/',
    },

    'f_det=10 f_sel=1' : {
        'f_det=10 f_sel=1' : 'raw/ct28-interleaved-MaskSC-3-fdet_10/Titerationreward/',
    },

    'f_det=1 f_sel=5' : {
        'f_det=1 f_sel=5' : 'raw/ct28-interleaved-MaskSC-3-fsel_5/Titerationreward/',
    },

    'f_det=1 f_sel=10' : {
        'f_det=1 f_sel=10' : 'raw/ct28-interleaved-MaskSC-3-fsel_10/Titerationreward/',
    }
}

neurips_minigrid = {
    'Finetuning (PPO)' : {
        'Finetuning (PPO)' : 'raw/mg16-interleaved-PPO/Titerationreward/',
    },

    'Mask-RI' : {
        'Mask-RI' : 'raw/mg16-interleaved-MaskRI/Titerationreward/',
    },

    'Mask-SC-oracle' : {
        'Mask-SC-oracle' : 'raw/mg16-interleaved-MaskSC-oracle/Titerationreward/',
        
    },

    'oEWC (MH)' : {
        'oEWC (MH)' : 'raw/mg16-interleaved-EWC-MH/Titerationreward/',
    },

    'SI (MH)' : {
        'SI (MH)' : 'raw/mg16-interleaved-SI-MH/Titerationreward/',
    },

    'CLEAR' : {
        'CLEAR' : 'raw/mg16-interleaved-CLEAR/Titerationreward/',
    },

    'SER' : {
        'SER' : 'raw/mg16-interleaved-SER/Titerationreward/',
    },

    'Mask-LC' : {
        'Mask-LC' : 'raw/mg16-interleaved-MaskLC/Titerationreward/',
    },

    'Mask-BLC' : {
        'Mask-BLC' : 'raw/mg16-interleaved-MaskBLC/Titerationreward/',
    },

    'CLHNet' : {
        'CLHNet' : 'raw/mg16-interleaved-CLHNET/Titerationreward/',
    },

    'PNN' : {
        'PNN' : 'raw/mg16-interleaved-PNN/Titerationreward/',
    },

    'CKA-RL' : {
        'CKA-RL' : 'raw/mg16-interleaved-CKA/Titerationreward/',
    },

    'SDW' : {
        'SDW' : 'raw/mg16-interleaved-SDW/Titerationreward/',
    },

    'Mask-SC (k=4)' : {
        'Mask-SC (k=4)' : 'raw/mg16-interleaved-MaskSC-4/Titerationreward/',
    }
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_name', help='paths to the experiment folder for single'\
        'agent lifelong learning (support paths to multiple seeds)', type=str, default=None)
    parser.add_argument('--exp_name', help='name of experiment', default='metrics_plot')
    parser.add_argument('--num_agents', help='number of agents in the experiment', type=int, nargs='+', default=1)
    parser.add_argument('--interval', help='interval', type=int, default=1)
    parser.add_argument('--staircase_from_pickle', help='path to master*.pkl for staircase plot', type=str, default=None)
    parser.add_argument('--task_len', help='number of iterations per task for staircase plot', type=int, default=100)
    parser.add_argument('--staircase_out', help='output path for staircase plot pdf', type=str, default=None)
    args = parser.parse_args()

    if args.staircase_from_pickle:
        with open(args.staircase_from_pickle, 'rb') as f:
            master = pickle.load(f)
        master_stair = staircase_transform(master, task_len=args.task_len)
        fig = plot_staircase(master_stair, yaxis_label='Cumulative Task Return')
        if args.staircase_out:
            output_path = args.staircase_out
        else:
            base = os.path.splitext(os.path.basename(args.staircase_from_pickle))[0]
            output_path = f'./log/plots/staircase_{base}.pdf'
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        fig.savefig(output_path, dpi=256, format='pdf', bbox_inches='tight')
        print(f'Saved staircase plot to {output_path}')
        raise SystemExit(0)

    MYPATHS = neurips_mctgraph_fdet_fsel_gridsearch

    fig2 = plt.figure(figsize=(15, 6.5), constrained_layout=True)
    ax2 = fig2.subplots()

    # Set up axis labels, fonts, and limits
    ax2.set_xlabel('Iteration')
    ax2.xaxis.label.set_fontsize(29)
    ax2.set_ylabel('Summed Return')
    ax2.yaxis.label.set_fontsize(29)
    ax2.set_ylim(-0.5, 28.0)
    ax2.set_xlim(-1.0, 200.0)

    # Axis ticks and grid
    ax2.xaxis.tick_bottom()
    ax2.yaxis.tick_left()
    ax2.tick_params(axis='both', which='major', labelsize=25)
    ax2.grid(True, which='both')

    # Remove right and top spines
    #ax2.spines['right'].set_visible(False)
    #ax2.spines['top'].set_visible(False)

    # Set left and bottom spines at (0, 0) co-ordinate
    #ax2.spines['left'].set_position(('data', 0.0))
    #ax2.spines['bottom'].set_position(('data', 0.0))

    # Draw dark lines at (0, 0)
    #ax2.axhline(y=0, color='k')
    #ax2.axvline(x=0, color='k')

    fig3 = deepcopy(fig2)
    ax3 = deepcopy(ax2)

    master = {}
    master2 = {}

    # Store data for box plot
    boxplot_data = []
    boxplot_labels = []
    #interval_steps = [0, 25, 50, 75, 100, 125, 150, 175, 199]  # Example intervals for box plots
    #interval_steps = [0, 50, 100, 150, 199]
    interval_steps = list(range(0, 200, 1))

    fig4, axs4 = plt.subplots(7, 2, figsize=(10, 12.5), layout='constrained')

    for plot_name, paths in MYPATHS.items():
        print('NAMES:', plot_name, 'PATHS:', paths)
        master2[plot_name] = {}

        for name, path in paths.items():
            print(path)
            data = pd.DataFrame()
            experiment_summed_rewards = []
            for i, filepath in enumerate(sorted(glob.glob(f'{path}*.csv'))):
                # Load data into a pandas dataframe
                df = pd.read_csv(filepath)
                # Select data from second column for each seed run
                data.loc[:, i] = df['Value']
                print(data)

            master[name] = {}
            master[name]['xdata'] = np.arange(data.shape[0])
            master[name]['ydata'] = np.mean(data, axis=1)
            master[name]['ydata_cfi'] = cfi_delta(data)
            master[name]['plot_colour'] = 'green'

            master2[plot_name][name] = {}
            master2[plot_name][name]['xdata'] = np.arange(data.shape[0])
            master2[plot_name][name]['ydata'] = np.mean(data, axis=1)
            master2[plot_name][name]['ydata_cfi'] = cfi_delta(data)
            master2[plot_name][name]['plot_colour'] = 'green'
            
            
            # For each seed, calculate the sum of rewards at the specified interval steps
            for seed in data.columns:
                # Get the rewards for the specific seed
                seed_data = data[seed]

                # Sum the rewards at the specified interval steps for this seed
                values = []
                for step in interval_steps:
                    if step < len(seed_data):
                        #print(seed_data[step])
                        values.append(seed_data[step])
                summed_seed_reward = np.average(values)
                #summed_seed_reward = np.average([seed_data[step] for step in interval_steps if step < len(seed_data)])

                # Append the summed reward for this seed and task to the overall experiment rewards
                experiment_summed_rewards.append(summed_seed_reward)

            # Once all tasks in the experiment are summed, append the data for boxplot
            boxplot_data.extend(experiment_summed_rewards)  # Add all the summed rewards for this experiment
            boxplot_labels.extend([plot_name] * len(experiment_summed_rewards))  # Label with the experiment name

        """
            for step in interval_steps:
                if step < len(master[name]['ydata']):  # Ensure the step is within bounds
                    boxplot_data.append(master[name]['ydata'][step])  # Summed reward at the step
                    boxplot_labels.append(f'{name[0]}{name[-1]}_step_{step}')  # Label for the box plot"""



        if not os.path.exists('./log/plots/'): os.makedirs('./log/plots/')
        fig1 = plot(master, yaxis_label='Return')
        fig1.savefig(f'./log/plots/{plot_name}.pdf', dpi=256, format='pdf', bbox_inches='tight')

        fig2, ax2 = plot_sum(fig2, ax2, master, title=plot_name, yaxis_label='Instant Cumulative Return')

        fig4, axs4 = plot_seperate(fig4, axs4, master, plot_name)
        #save_plot_data_to_csv(master)
        with open(f'./log/plots/master{plot_name}.pkl', 'wb') as f:
            pickle.dump(master, f)

        #fig3, ax3 = plot_box(fig3, ax3, master, title=plot_name, yaxis_label='Summed Return')

    print(len(boxplot_data))
    # Convert boxplot_data to the format required by seaborn
    boxplot_data_df = pd.DataFrame({
        "Average Reward": boxplot_data,
        "Experiment": boxplot_labels
    })

    def remove_duplicates(original_list):
        unique_list = []
        for item in original_list:
            if item not in unique_list:
                unique_list.append(item)
        return unique_list
    
    botplot_ticks = remove_duplicates(boxplot_labels)
    # Create a new figure for box plots
    fig_box, ax_box = plt.subplots(figsize=(8, 6))
    #plt.boxplot(boxplot_data)
    sns.boxplot(x="Experiment", y="Average Reward", data=boxplot_data_df, ax=ax_box)
    ax_box.set_xticklabels(botplot_ticks, rotation=45, ha='right')
    ax_box.set_xlabel('Experiment')
    ax_box.set_ylabel('Reward distrubtion across all tasks')

    # Save figures
    fig_box.savefig('./log/plots/boxplot_comparison.pdf', dpi=256, bbox_inches='tight')  # Save the figure
    fig2.savefig(f'./log/plots/cumulative.pdf', dpi=256, format='pdf', bbox_inches='tight')
    fig4.savefig(f'./log/plots/individual.pdf', dpi=256, format='pdf', bbox_inches='tight')

    with open("./log/plots/cumulative.pkl", "wb") as f:
        pickle.dump(fig2, f)

    # Plot TRA metric
    plot_tra(master2)
