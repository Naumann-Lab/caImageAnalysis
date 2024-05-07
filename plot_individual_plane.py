"""
Functions to plot a bunch of relevant graphs for each plane. Can be ran through plot_individual_plane_runningscipt.py

@Zichen He 20240313
"""
import constants
from utilities import clustering, arrutils
#from fishy import WorkingFish, BaseFish, VizStimVolume

import pandas as pd
import cmasher as cmr
import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib import pyplot as plt
# from tqdm.auto import tqdm
import numpy as np
import seaborn as sns
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from fishy import BaseFish
from datetime import datetime as dt
hzReturner = BaseFish.hzReturner


def analyze_tail(frametimes_df, stimulus_df, tail_df, stimuli_s = 5, strength_boundary = 0.3):
    """
    capture tail events happened in the current inputs. Also produce a graph for the raw trace, smoothed trace, and positive/negative smoothed trace for the bouts
        frametimes_df: the dataframe for the frames and corresponding real times
        stimulus_df: the dataframe contain all the stimulus and their onset frames
        tail_df: the dataframe of the tail movement and their corresponding frames
        stimuli_s: the second that the stimuli was on
        strength_boundary: the minimal std for a tail to be counted as on
    Return:
        tail_byframe: a ndarray containing the calibrated tail angle by each frame
        tail_bout_df: a dataframe contains all the bout information, which includes
            cont_tuples: a list that contains the frames for [on_frame, off_frame] of each bout
            tail_strength: a list that contains the strength of the tail for each bout, calculated by standard deviation
            tail_angle_pos: a list that contains all the bouts and their corresponding max positive tail angle
            tail_angle_neg:a list that contains all the bouts and their corresponding max negative tail angle
            tail_duration_s: a list that contains all the bouts and their corresponding duration in seconds
            tail_stimuli: a list that contains all the stimuli for each bouts, if they are happened within the stimuli
             presentation. If not, the stimuli is labeled "spontaneous"
    """
    hz = hzReturner(frametimes_df)
    #min_on_hz = min_on_s * hz
    fig, ax = plt.subplots(3, 1, dpi = 400, figsize = (20, 6))

    ax[0].plot(list(tail_df.tail_sum), linewidth = 0.5, color = 'black')
    ax[0].set_ylim([-np.pi, np.pi])
    ax[0].set_xlim([0, tail_df.shape[0]])
    ax[0].set_xticks([0, tail_df.shape[0]])
    # supplement the non-tailtracked fish
    tail_frame = np.unique(tail_df.frame)
    non_tailframe = [i for i in range(0, np.max(frametimes_df.index) + 1) if i not in tail_frame]
    notailtrack_frame_df = pd.DataFrame(data=[] * len(non_tailframe), columns=tail_df.columns)
    notailtrack_frame_df['frame'] = non_tailframe
    tail_df = pd.concat([tail_df, notailtrack_frame_df]).sort_values(by=['frame'])
    #calibrate to mean
    baseline = np.mean(tail_df.tail_sum)
    tail_df.tail_sum = np.subtract(tail_df.tail_sum, baseline)


    #collect positive and negative tail movement
    pos = tail_df.drop(['t_dt'], axis = 1)
    pos.tail_sum = np.where(pos.tail_sum > 0,pos.tail_sum, 0)
    pos_byframe = np.array((pos.groupby(['frame']).mean().tail_sum)).flatten()
    neg = tail_df.drop(['t_dt'], axis = 1)
    neg.tail_sum = np.where(neg.tail_sum < 0, neg.tail_sum, 0)
    neg_byframe = np.array((neg.groupby(['frame']).mean().tail_sum)).flatten()
    tail_byframe = np.add(pos_byframe, neg_byframe)

    ax[1].plot(pos_byframe, linewidth = 0.5, color = 'maroon')
    ax[1].plot(neg_byframe, linewidth = 0.5, color = 'midnightblue')
    ax[1].set_ylim([-np.pi, np.pi])
    #collect tail strength
    std_byframe = np.array((tail_df.drop(['t_dt'], axis = 1).groupby(['frame']).std().tail_sum)).flatten()
    ax[2].plot(std_byframe, linewidth = 0.5, color = 'black')
    ax[2].set_ylim([0, 1])
    bout_on = std_byframe > strength_boundary
    bout_on = [int(x) for x in bout_on]
    on_index = np.where(np.diff(bout_on) == 1)[0]
    off_index = np.where(np.diff(bout_on) == -1)[0]
    if len(on_index) != 0 and len(off_index) != 0:
        if on_index[0] > off_index[0]:
            on_index = np.concatenate([[0], on_index])
        if on_index[-1] > off_index[-1]:
            off_index = np.concatenate([off_index, [len(std_byframe) - 1]])
        on_tuples = [(on, off) for on, off in zip(on_index, off_index) if off - on < len(std_byframe)]
    cont_on_index = []
    cont_off_index = []
    if len(on_tuples) > 0:
        cont_on_index = [on_tuples[0][0]]
        cont_cutoff_hz = 1
        if len(on_tuples) > 1:
            big_interval = np.array([on_tuples[i][0] - on_tuples[i - 1][1] for i in range(1, len(on_tuples))]) > cont_cutoff_hz
            for i in range(0, len(big_interval)):
                if big_interval[i]:
                    cont_on_index = cont_on_index + [on_tuples[i + 1][0]]
                    cont_off_index = cont_off_index + [on_tuples[i][1]]
        cont_off_index = cont_off_index + [on_tuples[-1][1]]
    cont_tuples = [(on, off) for on, off in zip(cont_on_index, cont_off_index)]
    #cont_tuples = on_tuples
    ax[2].axhline(strength_boundary, linewidth = 1, linestyle = ':', color = 'purple')
    tail_strength = np.full(len(cont_tuples), np.nan)
    tail_angle_pos = np.full(len(cont_tuples), np.nan)
    tail_angle_neg = np.full(len(cont_tuples), np.nan)
    tail_duration_s = np.full(len(cont_tuples), np.nan)
    tail_stimuli = ['spontaneous'] *len(cont_tuples)
    for i in range(len(cont_tuples)):
        ax[2].axvspan(cont_tuples[i][0], cont_tuples[i][1], color = 'pink', alpha = 0.5)
        tail_strength[i] = np.mean(std_byframe[cont_tuples[i][0]:cont_tuples[i][1] + 1])
        tail_angle_pos[i] = np.nanmax(pos_byframe[cont_tuples[i][0]:cont_tuples[i][1] + 1])
        tail_angle_neg[i] = np.nanmin(neg_byframe[cont_tuples[i][0]:cont_tuples[i][1] + 1])
        tail_duration_s[i] = np.divide(cont_tuples[i][1] - cont_tuples[i][0], hz)
        if not stimulus_df.empty:
            if cont_tuples[i][0] > stimulus_df.iloc[0]['frame']:
                stimulus_responding = stimulus_df.loc[stimulus_df['frame'] <= cont_tuples[i][0]].iloc[-1]#find the nearest stimuli before and see if tail happens within the stimulus
                if stimulus_responding['frame'] + stimuli_s * hz >= cont_tuples[i][0]:#if tail starts before the stimulus ends
                    tail_stimuli[i] = stimulus_responding['stim_name']
    #plot stimulus
    if not stimulus_df.empty:
        for i, stim_row in stimulus_df.iterrows():
            c = constants.allcolor_dict[stim_row['stim_name']]
            ax[1].axvspan(stim_row['frame'],stim_row['frame'] + 5*hz/2, color = c[0], alpha = 0.2)
            ax[1].axvspan(stim_row['frame'] + 5*hz/2 + 0.1, stim_row['frame'] + 5*hz, color = c[1], alpha = 0.2)

    #make plot prettier
    ax[0].set_xlabel('tail frames')
    ax[0].set_ylabel('raw tail angle (rad)')
    ax[0].set_yticks([3.14, 0, -3.14])
    ax[1].set_ylabel('smooth tail angle (rad)')
    ax[1].set_xticks([])
    ax[1].set_yticks([3.14, 0, -3.14])
    ax[1].set_xlim([0, np.max(tail_frame)])
    ax[2].set_ylabel('tail strength (std)')
    ax[2].set_xlabel('frames')
    ax[2].set_yticks([1, 0])
    ax[2].sharex(ax[1])

    tail_bout_df = pd.DataFrame(
        {'cont_tuples': cont_tuples, 'tail_strength': tail_strength, 'tail_angle_pos': tail_angle_pos,
         'tail_angle_neg': tail_angle_neg, 'tail_duration_s': tail_duration_s, 'tail_stimuli': tail_stimuli})
    return tail_byframe, tail_bout_df

def tail_angle_all(stimulus_df, tail_bout_df, stimuli_s = 5):
    """
    Plot the tail response against all angles, including spontanesous response
        stimulus_df: the dataframe contain all the stimulus and their onset frames
        tail_bout_df: the dataframe of the tail bouts movement details, which includes cont_tuples, tail_strength, tail_angle_pos, tail_angle_neg, tail_duration_s, tail_stimuli
        stimuli_s: the second that the stimuli was on
    """

    fig, ax = plt.subplots(4, 4, figsize = (10, 8), dpi = 240, gridspec_kw={'width_ratios': [1, 1, 1, 1]})

    #find out how many time each stimuli is presented and responded to
    gs = ax[0, 0].get_gridspec()
    for axes in ax[0:, 0]:
        axes.remove()
    ax_hist = fig.add_subplot(gs[0:, 0])
    all_stimuli_list = list(constants.deg_dict.keys()) + ['spontaneous']
    all_stimuli_color_list = []
    for s in all_stimuli_list:
        if s in constants.monocular_dict.keys():
            all_stimuli_color_list = all_stimuli_color_list + [constants.monocular_dict[s]]
    all_stimuli_color_list = all_stimuli_color_list + [constants.monocular_dict['left']] * 2 + [constants.monocular_dict['right']] * 2 + ['brown'] * 3

    stimuli_presentation = {stimuli: 0 for stimuli in all_stimuli_list}
    stimuli_responding = {stimuli: 0 for stimuli in all_stimuli_list}
    stimuli_responding_duration_s = {stimuli: [] for stimuli in all_stimuli_list}
    for stimuli in all_stimuli_list:
        stimuli_presentation[stimuli] = list(stimulus_df.stim_name).count(stimuli)
        stimuli_responding[stimuli] = tail_bout_df[tail_bout_df.tail_stimuli == stimuli].shape[0]
        stimuli_responding_duration_s[stimuli] = list(tail_bout_df[tail_bout_df.tail_stimuli == stimuli].tail_duration_s)
    ax_hist.barh(all_stimuli_list, list(stimuli_presentation.values()), color = 'grey', alpha = 0.5)
    ax_hist.barh(all_stimuli_list, list(stimuli_responding.values()), color = all_stimuli_color_list, alpha = 0.8)
    ax_hist.set_xlabel('bout/stim occurance')

    gs = ax[0, 1].get_gridspec()
    for axes in ax[0:, 1]:
        axes.remove()
    ax_time = fig.add_subplot(gs[0:, 1])
    for i in range(0, len(all_stimuli_list)):
        y = all_stimuli_list[i]
        x = stimuli_responding_duration_s[y]
        ax_time.scatter(y= np.add([i] * len(x), np.random.uniform(-0.1, 0.1, len(x))), x= x, color = all_stimuli_color_list[i], alpha = 0.5, s = 2)
    ax_time.set_yticklabels([])
    ax_time.set_xlim([0, 7.5])
    ax_time.set_xticks([0, 5, 7.5])
    ax_time.set_xticklabels([0, 5, 7.5])
    ax_time.axvline(stimuli_s, linestyle = ':', color = 'grey', linewidth = 1)
    ax_time.set_xlabel('bout duration (s)')

    #separate binocular visually evoked response and spontaneous response
    bi_response_index = []
    spon_response_index = []
    for index in range(0, tail_bout_df.tail_stimuli.shape[0]):
        if tail_bout_df.tail_stimuli[index] in constants.monocular_dict.keys():
           bi_response_index = bi_response_index + [index]
        elif not tail_bout_df.tail_stimuli[index] in constants.deg_dict.keys():
            spon_response_index = spon_response_index + [index]
    bi_cont_tuples = [tail_bout_df.cont_tuples[index] for index in bi_response_index]
    bi_tail_strength = [tail_bout_df.tail_strength[index] for index in bi_response_index]
    bi_tail_angle_pos = [tail_bout_df.tail_angle_pos[index] for index in bi_response_index]
    bi_tail_angle_neg = [tail_bout_df.tail_angle_neg[index] for index in bi_response_index]
    bi_tail_duration_s = [tail_bout_df.tail_duration_s[index] for index in bi_response_index]
    bi_tail_stimuli = [tail_bout_df.tail_stimuli[index] for index in bi_response_index]
    bi_tail_stimuli_color = [constants.monocular_dict.get(stimuli) for stimuli in bi_tail_stimuli]

    spon_cont_tuples = [tail_bout_df.cont_tuples[index] for index in spon_response_index]
    spon_tail_strength = [tail_bout_df.tail_strength[index] for index in spon_response_index]
    spon_tail_angle_pos = [tail_bout_df.tail_angle_pos[index] for index in spon_response_index]
    spon_tail_angle_neg = [tail_bout_df.tail_angle_neg[index] for index in spon_response_index]
    spon_tail_duration_s = [tail_bout_df.tail_duration_s[index] for index in spon_response_index]

    #plot all binocular responses
    gs = ax[0, 2].get_gridspec()
    ax[0, 2].remove()
    ax_bip = fig.add_subplot(gs[0, 2], projection = 'polar')
    ax_bip.bar(bi_tail_angle_pos, bi_tail_strength, color = bi_tail_stimuli_color , width = 0.02, alpha = 0.5, zorder = 10)
    gs = ax[1, 2].get_gridspec()
    ax[1, 2].remove()
    ax_bin = fig.add_subplot(gs[1, 2], projection = 'polar')
    ax_bin.bar(bi_tail_angle_neg, bi_tail_strength, color = bi_tail_stimuli_color , width = 0.02, alpha = 0.5, zorder = 10)
    gs = ax[2, 2].get_gridspec()
    ax[2, 2].remove()
    ax_bia = fig.add_subplot(gs[2, 2], projection = 'polar')
    ax_bia.bar(np.add(bi_tail_angle_neg,bi_tail_angle_pos), bi_tail_strength, color = bi_tail_stimuli_color , width = 0.02, alpha = 0.5, zorder = 10)
    gs = ax[3, 2].get_gridspec()
    ax[3, 2].remove()
    ax_bid = fig.add_subplot(gs[3, 2], projection = 'polar')
    ax_bid.bar([np.deg2rad(constants.deg_dict[s]) for s in constants.monocular_dict.keys()],
               len(constants.monocular_dict.values()) * [1],
               color = constants.monocular_dict.values() , width = 0.75, alpha = 0.3, zorder = 10)
    #plot spontaneous responses
    gs = ax[0, 3].get_gridspec()
    ax[0, 3].remove()
    ax_sponp = fig.add_subplot(gs[0, 3], projection = 'polar')
    ax_sponp.bar(spon_tail_angle_pos, spon_tail_strength, color = 'brown' , width = 0.02, alpha = 0.5, zorder = 10)
    gs = ax[1, 3].get_gridspec()
    ax[1, 3].remove()
    ax_sponn = fig.add_subplot(gs[1, 3], projection = 'polar')
    ax_sponn.bar(spon_tail_angle_neg, spon_tail_strength, color = 'brown' , width = 0.02, alpha = 0.5, zorder = 10)
    gs = ax[2, 3].get_gridspec()
    ax[2, 3].remove()
    ax_spona = fig.add_subplot(gs[2, 3], projection = 'polar')
    ax_spona.bar(np.add(spon_tail_angle_neg,spon_tail_angle_pos), spon_tail_strength, color = 'brown' , width = 0.04, alpha = 0.5, zorder = 10)
    ax[3, 3].axis('off')

    #make prettier
    def fix_ax(ax):
        """
        Make axis look prettier
        """
        ax.set_theta_offset(np.deg2rad(90))
        ax.set_theta_direction('clockwise')
        ax.set_ylim([0, 0.8])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    fix_ax(ax_bip)
    ax_bip.axvspan(np.deg2rad(180), np.deg2rad(360), color = 'lightgrey', alpha = 0.3)
    fix_ax(ax_bin)
    ax_bin.axvspan(np.deg2rad(0), np.deg2rad(180), color = 'lightgrey', alpha = 0.3)
    fix_ax(ax_bia)
    fix_ax(ax_bid)
    fix_ax(ax_sponp)
    ax_sponp.axvspan(np.deg2rad(180), np.deg2rad(360), color = 'lightgrey', alpha = 0.3)
    fix_ax(ax_sponn)
    ax_sponn.axvspan(np.deg2rad(0), np.deg2rad(180), color = 'lightgrey', alpha = 0.3)
    fix_ax(ax_spona)

    ax_bip.set_title('binocular')
    ax_sponp.set_title('spontaneous')
    ax_bip.set_ylabel('pos max\ntail angle')
    ax_bin.set_ylabel('neg max\ntail angle')
    ax_bia.set_ylabel('sum max\ntail angle')
    ax_bid.set_ylabel('stimuli\nangle')

def tail_angle_binocular(tail_bout_df, stimuli_s = 5):
    """
    Plot the tail response against all angles, including spontanesous response
        tail_bout_df: the dataframe of the tail bouts movement details, which includes cont_tuples, tail_strength, tail_angle_pos, tail_angle_neg, tail_duration_s, tail_stimuli
        stimuli_s: the second that the stimuli was on
    """
    #plot for each stimuli
    stimuli_num = len(constants.monocular_dict.keys())
    figall, ax = plt.subplots(5, stimuli_num, figsize = (15, 9), dpi = 400, gridspec_kw={'height_ratios': [2, 2, 2, 2, 2], 'hspace': 0})

    #plot stimuli specific response
    stimuli_n = 0
    for stimuli in list(constants.monocular_dict.keys()):# + ['all binocular']:
        response_df = tail_bout_df[tail_bout_df.tail_stimuli == stimuli]
        response_cont_tuples = list(response_df.cont_tuples)
        response_tail_strength = list(response_df.tail_strength)
        response_tail_angle_pos = list(response_df.tail_angle_pos)
        response_tail_angle_neg = list(response_df.tail_angle_neg)
        response_tail_duration_s = list(response_df.tail_duration_s)
        stimuli_color = constants.monocular_dict.get(stimuli)
        count = response_df.shape[0]

        ax_hist = figall.add_subplot(ax[0, stimuli_n])
        ax_hist.bar(0, count, color = stimuli_color, alpha = 0.8, width = 0.3)
        ax_hist.set_ylim([0, 80])
        ax_hist.set_xlim([-0.5, 1.5])
        ax_hist.set_yticks([])
        ax_hist.set_xticks([])
        ax_hist.set_xticklabels([])
        ax_hist.spines[['top', 'right', 'bottom', 'left']].set_visible(False)

        ax_scatter = figall.add_subplot(ax[0, stimuli_n])
        ax_scatter.scatter(np.add([1] * count, np.random.uniform(-0.1, 0.1, count)), np.divide(response_tail_duration_s, 7.5/80), color = stimuli_color, alpha = 0.5, s = 3, zorder = 1)
        #ax_scatter.set_ylim([0, 7.5])
        ax_scatter.set_xlim([-0.5, 1.5])
        ax_scatter.axhline(stimuli_s/7.5*80, xmin = 0.5, xmax =2, linestyle = ':', color = 'grey', linewidth = 1)
        ax_scatter.set_yticks([])
        ax_scatter.set_xticks([])
        ax_scatter.set_xticklabels([])
        ax_scatter.spines[['top', 'right', 'bottom', 'left']].set_visible(False)

        gs = ax[1, stimuli_n].get_gridspec()
        ax[1, stimuli_n].remove()
        axp = figall.add_subplot(gs[1, stimuli_n], projection = 'polar')
        axp.bar(response_tail_angle_pos, response_tail_strength, color = 'darkgrey' , width = 0.1, alpha = 0.5, zorder = 1)

        gs = ax[2, stimuli_n].get_gridspec()
        ax[2, stimuli_n].remove()
        axn = figall.add_subplot(gs[2, stimuli_n], projection = 'polar')
        axn.bar(response_tail_angle_neg, response_tail_strength, color = 'darkgrey', width = 0.1, alpha = 0.5, zorder = 1)

        gs = ax[3, stimuli_n].get_gridspec()
        ax[3, stimuli_n].remove()
        axa = figall.add_subplot(gs[3, stimuli_n], projection = 'polar')
        axa.bar(np.add(response_tail_angle_neg,response_tail_angle_pos), response_tail_strength, color = 'darkgrey' , width = 0.1, alpha = 0.5, zorder = 1)
        #plot the mean tail response
        if len(response_tail_strength) > 0:
            axp.bar(np.nanmean(response_tail_angle_pos), np.nanmean(response_tail_strength), color = stimuli_color , width = 0.1, alpha = 1, zorder = 2)
            axn.bar(np.nanmean(response_tail_angle_neg), np.nanmean(response_tail_strength), color = stimuli_color , width = 0.1, alpha = 1, zorder = 2)
            axa.bar(np.nanmean(np.add(response_tail_angle_neg,response_tail_angle_pos)), np.nanmean(response_tail_strength), color = stimuli_color , width = 0.1, alpha =1, zorder = 2)
        #plot stimuli
        gs = ax[4, stimuli_n].get_gridspec()
        ax[4, stimuli_n].remove()
        axd = figall.add_subplot(gs[4, stimuli_n], projection = 'polar')
        axd.bar(np.deg2rad(constants.deg_dict.get(stimuli)), 1, color = stimuli_color , width = 0.5, alpha = 0.3, zorder = 1)

        #make prettier
        def fix_ax(ax):
            """
            Make axis look prettier
            """
            ax.set_theta_offset(np.deg2rad(90))
            ax.set_theta_direction('clockwise')
            ax.set_ylim([0, 0.6])
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        fix_ax(axp)
        axp.axvspan(np.deg2rad(180), np.deg2rad(360), color = 'lightgrey', alpha = 0.3)
        fix_ax(axn)
        axn.axvspan(np.deg2rad(0), np.deg2rad(180), color = 'lightgrey', alpha = 0.3)
        fix_ax(axa)
        fix_ax(axd)
        axd.set_xlabel(str(stimuli))
        if stimuli_n == 0:
            ax_hist.set_ylabel('bout/stim\nbout duration (s)')
            axp.set_ylabel('pos max\ntail angle')
            axn.set_ylabel('neg max\ntail angle')
            axa.set_ylabel('sum max\ntail angle')
            axd.set_ylabel('stimuli\nangle')
        stimuli_n += 1

def plot_trace_loc(frametimes_df, stimulus_df, refImg, trace, tracename, loc, minbar = 0, maxbar = 1, byregion = False,
                   tail_byframe = np.ndarray([])):
    """
    Plot the cell traces heatmap with their corresponding location on the ref image
        frametimes_df: the dataframe for frames and corresponding time
        stimulus_df: the dataframe for stimulus and corresponding frames
        refImg: the dataframe containing information to plot reference image
        trace: the cell trace DataFrame to be plotted, if region == True, it's a dataframe containing all traces
        tracename: the cell trace name, used to label the graph
        loc: the cell ROIs DataFrame. if region == True, it's a dataframe containing all traces, it's a dataframe that containing all traces. Noted the the sequence should match the trace sequence
        minbar: the minimal value of the cbar, default 0
        maxbar: the max value of the cbar, dfault 1
        byregion: weather we are plotting cell by region
        tail_byframe: a ndarray for the tail radius angle by frame
    Return: Noneyhgh
    """
    hz = hzReturner(frametimes_df)
    #preparing figure space
    if byregion == True:
        region_count = len(trace.keys()) + 1
    else:
        region_count = 2
    fig, ax = plt.subplots(region_count, 5, sharex = 'col', figsize = (15, 10),
                               gridspec_kw={'height_ratios': [1 * (region_count - 1)] + [20] * (region_count - 1),
                                            'hspace': 0,
                                            'width_ratios': [0.4, 1, 20, 1, 20], 'wspace':0}, dpi = 240)
    #turn off unnecessary axis
    ax[0, 0].axis('off')
    ax[0, 1].axis('off')
    ax[0, 2].axis('off')
    ax[0, 3].axis('off')
    ax[0, 4].axis('off')
    #plot stimulus and tail
    if not stimulus_df.empty:
        for i, stim_row in stimulus_df.iterrows():
            c = constants.allcolor_dict[stim_row['stim_name']]
            ax[0, 2].axvspan(stim_row['frame'],stim_row['frame'] + 5*hz/2, color = c[0], alpha = 0.2)
            ax[0, 2].axvspan(stim_row['frame'] + 5*hz/2 + 0.1, stim_row['frame'] + 5*hz, color = c[1], alpha = 0.2)
        #ax[0, 2].set_ylim([-1, 1])
    if not tail_byframe.size == 0:
        ax[0, 2].plot(tail_byframe, linewidth = 0.5, color = 'black')
    #ax[0, 2].set_xlim([50, 150])
    #decide heatbar ax
    gs = ax[1, 0].get_gridspec()
    for axes in ax[1:region_count, 0]:
        axes.remove()
    ax_cbar = fig.add_subplot(gs[1:region_count, 0])
    #decide ref image scatter plot axis
    gs = ax[1, 4].get_gridspec()
    for axes in ax[1:, 4]:
        axes.remove()
    ax_show = fig.add_subplot(gs[1:, 4])
    ax_show.imshow(refImg, cmap = 'grey', alpha = 0.8, vmax = 100)
    #for each region, plot heatmap
    for region_row in range(1, region_count):
        if byregion == True:
            regionname = list(trace.keys())[region_row - 1]
            trace_to_plot = trace[regionname]
            colormap = constants.cmaplist[regionname]
            location = loc[regionname]
        else:
            trace_to_plot = trace
            regionname = ''
            colormap = 'Wistia'
            location = loc
        ax[region_row, 1].axis('off')
        #plot heatmap
        ax_heatmap = ax[region_row, 2]
        if region_row == 1:
            sns.heatmap(trace_to_plot, ax = ax_heatmap, vmin = minbar, vmax = maxbar,
                        cmap = 'viridis', cbar_ax = ax_cbar,
                        cbar_kws =dict(location="left", shrink = 0.3, label = tracename))
        else:
            sns.heatmap(trace_to_plot, ax = ax_heatmap, vmin = minbar, vmax = maxbar,
                        cmap = 'viridis', cbar = False)
            ax_heatmap.axhline(y = 0, xmax = len(frametimes_df), color='white', linewidth=3, linestyle = ':')
        ax_heatmap.set(yticklabels=[])
        ax_heatmap.tick_params(left=False)
        ax_heatmap.set_ylabel(regionname +' cell', labelpad = -5)
        #ax_heatmap.set_xlim([50, 150])
        #plot scatter plot for yaxis
        ax_scatter = ax[region_row, 3]
        ax_scatter.scatter([0] * trace_to_plot.shape[0], range(trace_to_plot.shape[0]),
                                        c = location['ypos'], s = 500,
                               marker = 'o' , cmap = colormap)
        ax_scatter.axis('off')
        ax_scatter.sharey(ax[region_row, 2])
        #plot scatter plot on ref image
        ax_show.scatter(location['xpos'], location['ypos'], c = location['ypos'],
                        s = 3, cmap = colormap)
        region_row += 1
    #ax[region_count - 1, 2].set_xticks(range(0, len(frametimes_df), 200))
    #ax[region_count - 1, 2].set_xticklabels(range(0, len(frametimes_df), 200))
    ax[region_count - 1, 2].set_xlabel('frames')
    ax_show.set_ylabel('cell location')
    ax_show.set_xticks([])
    ax_show.set_yticks([])

def plot_trace_loc_example(frametimes_df, stimulus_df, refImg, trace, tracename, loc, minbar = 0, maxbar = 1,
                           byregion = False, tail_byframe = np.ndarray([])):
    """
    Randomly select 30 cells, plot the cell traces line graph with their corresponding location on the ref image
        frametimes_df: the dataframe for frames and corresponding time
        stimulus_df: the dataframe for stimulus and corresponding frames
        refImg: the dataframe containing information to plot reference image
        trace: the cell trace DataFrame to be plotted, if region == True, it's a dataframe containing all traces
        tracename: the cell trace name, used to label the graph
        loc: the cell ROIs DataFrame. if region == True, it's a dataframe containing all traces, it's a dataframe that containing all traces. Noted the the sequence should match the trace sequence
        minbar: the minimal value of the cbar, default 0
        maxbar: the max value of the cbar, dfault 1
        byregion: weather we are plotting cell by region
        tail_byframe: a ndarray for the tail radius angle by frame
    Return: None
    """
    hz = hzReturner(frametimes_df)
    #preparing figure space
    if byregion == True:
        region_count = len(trace.keys())
    else:
        region_count = 1
    fig, ax = plt.subplots(region_count + 1, 3, figsize = (15, 8),
                           gridspec_kw={'height_ratios': [1] + [20] * region_count, 'width_ratios': [20, 1, 16],
                                        'hspace': 0.05, 'wspace':0}, dpi = 240)
    #if byregion == False:
    #    ax = np.array([ax, [0, 0, 0]])
    #plot tail
    if not tail_byframe.size == 0:
        ax[0, 0].plot(tail_byframe, linewidth = 0.5, color = 'black')
        ax[0, 0].sharex(ax[1, 0])
    ax[0, 0].axis('off')
    ax[0, 1].axis('off')
    ax[0, 2].axis('off')
    #plot imshow
    gs = ax[1, 2].get_gridspec()
    for axes in ax[1:, 2]:
        try:
            axes.remove()
        except: pass
    ax_show = fig.add_subplot(gs[1:, 2])
    ax_show.imshow(refImg, cmap = 'grey', alpha = 0.8, vmax = 100)
    for region_row in range(region_count):
        if byregion == True:
            regionname = list(trace.keys())[region_row]
            trace_to_plot = trace[regionname]
            sample_size = 10
            colormap = constants.cmaplist[regionname]
            location = loc[regionname]
        else:
            trace_to_plot = trace
            regionname = ''
            sample_size = 30
            colormap = 'Wistia'
            location = loc
        cell_selection = np.sort(np.random.randint(0, trace_to_plot.shape[0], size = sample_size))
        ax_lineplot = ax[region_row + 1, 0]
        ax_lineplot.invert_yaxis()
        #for each cell, plot lineplot
        row = 0
        for i in cell_selection:
            trace_to_plot_trim = np.divide(np.subtract(np.clip(trace_to_plot.iloc[i], minbar, maxbar), minbar),
                                           maxbar - minbar)
            ax_lineplot.plot(np.add(-trace_to_plot_trim, row), linewidth = 1, color = 'black')
            row += 1
        #plot stimulus background
        if not stimulus_df.empty:
            for i, stim_row in stimulus_df.iterrows():
                c = constants.allcolor_dict[stim_row['stim_name']]
                ax_lineplot.axvspan(stim_row['frame'],stim_row['frame'] + 5*hz/2, color = c[0], alpha = 0.2)
                ax_lineplot.axvspan(stim_row['frame'] + 5*hz/2 + 0.1, stim_row['frame'] + 5*hz, color = c[1], alpha = 0.2)
        ax_lineplot.set_ylabel(regionname + ' cells ' + tracename)
        if region_row < region_count - 1:
            ax_lineplot.xaxis.set_ticklabels([])
            ax_lineplot.set_xticks([])
        #plot scatter location
        ax_scatter = ax[region_row + 1, 1]
        ax_scatter.scatter([0] * len(cell_selection), range(len(cell_selection)),
                c = location.iloc[cell_selection]['ypos'], s = 100, marker = 'o' , cmap = colormap)
        ax_scatter.set_xlim([0, 0.01])
        ax_scatter.axis('off')
        ax_scatter.sharey(ax_lineplot)
        #plot location on ref.img
        ax_show.scatter(location.iloc[cell_selection]['xpos'], location.iloc[cell_selection]['ypos'],
                    c = location.iloc[cell_selection]['ypos'], cmap = colormap)
    ax_lineplot.set_xlabel('frames')
    ax_lineplot.set_xticks(np.arange(0, len(frametimes_df), 200))
    ax_show.set_ylabel('cell location')
    ax_show.axes.get_xaxis().set_ticks([])
    ax_show.axes.get_yaxis().set_ticks([])


def corr(trace, region_trace, tracename):
    """
    Make the correlation graphs for all cells and by region
        trace: the cell trace DataFrame to be plotted, if region == True, it's a dataframe containing all traces
        tracename: the cell trace name, used to label the graph
    Return:
        cof: the correlation coefficient matrix for all cells, sorted by y location
        region_cof: the correlation coefficient matrix for each region in a dictionary, sorted by y location
    """
    #preparing figure space
    region_count = len(region_trace.keys())
    fig, ax= plt.subplots(region_count, 3, gridspec_kw={'width_ratios': [3, 1, 1], 'hspace': 0.2, 'wspace':0},
            figsize = (12, 5), dpi = 240)
    #add gridspace for all neuron correlation plot
    gs = ax[0, 0].get_gridspec()
    for axes in ax[0:, 0]:
        axes.remove()
    ax_all = fig.add_subplot(gs[0:, 0])
    cof = np.corrcoef(trace)
    sns.heatmap(cof, cmap = 'RdBu_r', vmin = -1, vmax = 1, square = True, xticklabels = False, yticklabels = False,
            ax = ax_all, cbar_kws =dict(location="left", label = 'r2', ticks = [-1, 1]))
    ax_all.set_ylabel('all cells ' + tracename)
    i = 0
    region_cof = {key: None for key in region_trace.keys()}
    region_row = 0
    # region_length = 0
    for region in region_trace.keys():
        #plot heatmap for each region
        region_heat_ax = ax[region_row, 1]
        region_cof[region] = np.corrcoef(region_trace[region])
        sns.heatmap(region_cof[region], ax = region_heat_ax, cmap = 'RdBu_r', square = True, xticklabels = False,
                    yticklabels = False, cbar = False, vmin = -1, vmax = 1)
        region_heat_ax.set_ylabel(region)
        # region_length += region_trace[region].shape[0]
        # ax_all.axvline(region_length)
        # ax_all.axhline(region_length)
        #plot cof distribution for each region
        region_hist_ax = ax[region_row, 2]
        sns.histplot(region_cof[region].flatten(), ax = region_hist_ax, element = 'step', fill = True,
                     color = constants.cmaplist[region](0.5), stat = 'percent', shrink = 0.8)
        region_hist_ax.set_ylabel('%cell')
        region_hist_ax.set_yticks([])
        if region_row == region_count - 1:
            region_hist_ax.set_xticks([-1, 0, 1])
            region_hist_ax.set_xticklabels([-1, 0, 1])
            region_hist_ax.set_xlabel('r2')
        else:
            region_hist_ax.set_xticks([])
        region_row += 1
    return cof, region_cof

def corr_clustering(frametimes_df, stimulus_df, refImg, region_trace, tracename, loc, minbar = 0, maxbar = 1,
                    tail_byframe = np.ndarray([])):
    """
    Perform hierarchical clustering according to the correlation matrix input.
    (@https://github.com/TheLoneNut/CorrelationMatrixClustering/blob/master/CorrelationMatrixClustering.ipynb)
    Note that the index handling is very messy in this function. If grabbing any variable from here, be SURE to check
    if the indexing is correct.
        frametimes_df: the dataframe for frames and corresponding time
        refImg: the dataframe containing information to plot reference image
        trace: the dictionary containing all regions, cell trace DataFrame to be plotted
        stimulus_df: the dataframe for stimulus and corresponding frames
        tracename: the cell trace name, used to label the graph
        loc: loc:  it's a dataframe containing all regions and their cell ROIs
            it's a dataframe that containing all traces. Noted the the sequence should match the trace sequence
        minbar: the minimal value of the cbar, default 0
        maxbar: the max value of the cbar, dfault 1
        tail_byframe: a ndarray for the tail radius angle by frame
    Return:
        cluster_dict: the dictionary that contains all clusters and their corresponding cell index as in the trace input
    """
    hz = hzReturner(frametimes_df)
    #rebuild trace and ROIs from regions
    trace = pd.concat(region_trace.values())
    trace = trace.loc[~trace.index.duplicated(keep='first')]
    loc = pd.concat(loc.values())
    loc = loc.loc[~loc.index.duplicated(keep='first')]
    trace_reindex = trace.reset_index(drop=False)
    cof_reindex = np.corrcoef(trace_reindex.drop('index', axis=1))
    cbar = plt.get_cmap('rainbow_r')
    fig, ax = plt.subplots(4, 4, dpi=400, figsize=(12, 8), sharex = 'col',
        gridspec_kw={'height_ratios': [3, 20, 20, 1], 'hspace': 0.1,'wspace': 0.05, 'width_ratios': [20, 20, 3, 25]})
    ax[0, 0].axis('off')
    ax[0, 1].axis('off')
    ax[0, 2].axis('off')
    ax[0, 3].axis('off')
    ax[3, 2].axis('off')
    # plot stimulus on the top of heatmap
    ax_stimulus = ax[0, 1]
    if not stimulus_df.empty:
        for i, stim_row in stimulus_df.iterrows():
            c = constants.allcolor_dict[stim_row['stim_name']]  # constants.allcolor_dict[stim_row['stim_name']]
            ax_stimulus.axvspan(stim_row['frame'], stim_row['frame'] + 5 * hz / 2, color=c[0], alpha=0.2)
            ax_stimulus.axvspan(stim_row['frame'] + 5 * hz / 2 + 0.1, stim_row['frame'] + 5 * hz, color=c[1], alpha=0.2)
    if not tail_byframe.size == 0:
        ax_stimulus.plot(tail_byframe, linewidth = 0.5, color = 'black')
    # plot pre-proessed correlation
    ax_pre_cor = ax[1, 0]
    sns.heatmap(cof_reindex, cmap='RdBu_r', vmin=-1, vmax=1, square=True, xticklabels=False, yticklabels=False,
                ax=ax_pre_cor, cbar_ax=ax[3, 0], cbar_kws=dict(location="bottom", label='r2', ticks=[-1, 1]))
    ax_pre_cor.set_ylabel('original')
    # plot pre-processed heatmap
    ax_pre_heatmap = ax[1, 1]
    sns.heatmap(trace, ax=ax_pre_heatmap, vmin=minbar, vmax=maxbar, cmap='viridis', cbar_ax=ax[3, 1],
                cbar_kws=dict(location="bottom", label=tracename, ticks=[minbar, maxbar]))
    #ax_pre_heatmap.sharex(ax_stimulus)
    ax_pre_heatmap.axis('off')
    ax[1, 2].axis('off')

    # clustering...
    pdistance = pdist(cof_reindex)  # pdist(cof)

    distance = linkage(pdistance, method='complete')
    label = fcluster(distance, 0.5 * pdistance.max(), 'distance')
    columns = list((np.argsort(label)))
    cluster_sort_normf = trace_reindex.reindex(columns)
    cluster_sort_normf = cluster_sort_normf.set_index('index')
    cluster_sort_cof = np.corrcoef(cluster_sort_normf)
    cluster_num = max(label)
    cluster_dict = {cluster: None for cluster in range(1, cluster_num + 1)}
    colorlist = {cluster: None for cluster in range(1, cluster_num + 1)}
    for cluster in cluster_dict.keys():
        row = [x for x in range(0, len(label)) if label[x] == cluster]
        cell_index_list = trace_reindex.iloc[row]['index']
        cluster_dict[cluster] = cell_index_list
        colorlist[cluster] = cbar(cluster / cluster_num)

    # plot clustered correlation
    ax_sort_cor = ax[2, 0]
    sns.heatmap(cluster_sort_cof, cmap='RdBu_r', vmin=-1, vmax=1, xticklabels=False, yticklabels=False,
                ax=ax_sort_cor, cbar=False, square=True)
    ax_sort_cor.set_ylabel('clustered')
    # plot clustered heatmap
    ax_sort_heatmap = ax[2, 1]
    sns.heatmap(cluster_sort_normf, ax=ax_sort_heatmap, vmin=minbar, vmax=maxbar, cmap='viridis', cbar=False)
    ax_sort_heatmap.axis('off')
    #ax_sort_heatmap.sharex(ax_stimulus)
    # plot cluster label for clustered heatmap
    ax_scatter = ax[2, 2]
    ini_index = 0
    for cluster in cluster_dict.keys():
        end_index = ini_index + len(cluster_dict[cluster])
        ax_sort_heatmap.axhline(end_index, color='white', linewidth=0.8, linestyle=':')
        ax_scatter.add_patch(
            patches.Rectangle((0, ini_index), 1, end_index - ini_index, color=colorlist[cluster], alpha=0.8))
        ini_index = end_index
    ax_scatter.axis('off')
    ax_scatter.invert_yaxis()
    ax_scatter.sharey(ax_sort_heatmap)
    # plot scatter plot
    gs = ax[0, 3].get_gridspec()
    for axes in ax[0:, 3]:
        axes.remove()
    ax_imshow = fig.add_subplot(gs[0:, 3])
    ax_imshow.imshow(refImg, cmap='grey', alpha=0.8, vmax=100)
    ax_imshow.axis('off')
    for cluster in cluster_dict.keys():
        ax_imshow.scatter(loc.loc[cluster_dict[cluster]]['xpos'], loc.loc[cluster_dict[cluster]]['ypos'],
                          s=3,
                          color=colorlist[cluster], alpha=1)
    return cluster_dict


def plot_trace_stimuli(frametimes_df, offsets, stim_dict, tracename, loc, minbar = 0, maxbar = 1, byregion = False):
    """
    Plot the mean cell traces heatmap in the response window for each stimuli, with their corresponding y location on the right
        frametimes_df: the dataframe for frames and corresponding time
        offsets: the tuple of offsent of stimulus window
        stim_dict: the dictionmary of stimulus with their corresponiding cell responses. if region == True, each stimulus key is corresponding to a dictionary containing all regions
        tracename: the cell trace name, used to label the graph
        loc: the cell ROIs DataFrame. if region == True, it's a dataframe containing all traces, it's a dataframe that containing all traces. Noted the the sequence should match the trace sequence
        minbar: the minimal value of the cbar, default 0
        maxbar: the max value of the cbar, dfault 1
        byregion: weather we are plotting cell by region
    Return: None
    """
    hz = hzReturner(frametimes_df)
    #initiate figure plotting space
    if byregion == True:
        region_count = len(stim_dict['forward'].keys()) + 1
    else:
        region_count = 2
    stim_count = len(stim_dict.keys()) + 2
    fig, ax= plt.subplots(region_count, stim_count, figsize = (18, 8), dpi = 240,
                              gridspec_kw={'height_ratios': [1 * (region_count - 1)] + [20] * (region_count -1),
                                           'width_ratios': [1] + [5] * (stim_count -2) + [1],
                                            'hspace': 0, 'wspace': 0.05})
    #turn off unnecessary axis
    ax[0,0].axis('off')
    ax[0,stim_count - 1].axis('off')
    #prepare cbar axis
    gs = ax[1, 0].get_gridspec()
    for axes in ax[1:, 0]:
        axes.axis('off')
        axes.remove()
    ax_cbar = fig.add_subplot(gs[1:, 0])
    stim_col = 1
    for stim in constants.dir_sort: #sort stim bar according to a sequence that makes more sense
        #plot stimulus traces
        c = constants.allcolor_dict[stim]
        ax[0, stim_col].axvspan(0, 0 + 5*hz/2, facecolor = c[0], alpha = 0.5, )
        ax[0, stim_col].axvspan(0 + 5*hz/2 + 0.05, 0 + 5*hz, facecolor = c[1], alpha = 0.5)
        ax[0, stim_col].sharex(ax[1, stim_col])
        ax[0, stim_col].axis('off')
        #plot heatmap for each region
        for region_row in range(1, region_count):
            if byregion == True:
                regionname = list(stim_dict[stim].keys())[region_row - 1]
                trace_to_plot = stim_dict[stim][regionname]
                location = loc[regionname]
                colormap = constants.cmaplist[regionname]
            else:
                regionname = 'all'
                trace_to_plot = stim_dict[stim]
                location = loc
                colormap = 'gist_yarg'
            ax_heatmap = ax[region_row, stim_col]
            if region_row == region_count - 1 and stim_col == 1:#if first heatmap on the last row, plot cbar and x axis
                sns.heatmap(trace_to_plot, ax = ax_heatmap,
                    cmap = 'viridis', vmax = maxbar, vmin = minbar, yticklabels = False, cbar_ax = ax_cbar,
                    xticklabels = False, cbar_kws =dict(location="left", shrink = 0.3, label = 'mean ' + tracename))
                ax_heatmap.set_xticks([0, offsets[1] - offsets[0]], labels = [0, offsets[1] - offsets[0]])
                ax_heatmap.set_xlabel('stim on (frame)')
            else:
                sns.heatmap(trace_to_plot, ax = ax_heatmap,
                    cmap = 'viridis', vmax = maxbar, vmin = minbar, cbar = False, yticklabels = False,
                    xticklabels = False)
            #plot white dash lines between regions
            if region_row != 1:
                ax_heatmap.axhline(y = 0, xmax = offsets[1] - offsets[0], color='white', linewidth=1, linestyle = ':')
            #if last region row, plot region scatter
            if stim_col == stim_count - 2:
                ax_scatter = ax[region_row, stim_count - 1]
                ax_scatter.scatter([0] * trace_to_plot.shape[0], range(trace_to_plot.shape[0]), c = location['ypos'],
                                   s = 500, marker = 'o' , cmap = colormap)
                ax_scatter.set_xticks([])
                ax_scatter.set_yticks([])
                plt.setp(ax_scatter.spines.values(), visible=False)
                ax_scatter.set_ylabel(regionname + ' cells (cbar: ypos)')
                ax_scatter.yaxis.set_label_position("right")
                ax_scatter.sharey(ax_heatmap)
        stim_col += 1

def plot_tuning_distribution(booldf, tracename, loc, byregion = False):
    """
    Plot the distribution of direction tunning cell by yloc
        booldf: the booldf DataFrame to be plotted, if region == True, it's a dictionary contain all region and their booldf
        tracename: the cell trace name, used to label the graph
        loc: the cell ROIs DataFrame. if region == True, it's a ddictionary that containing all regions and their ROIs. Noted the the sequence should match the trace sequence
        byregion: weather we are plotting cell by region
    Return: None
    """
    #region_separation_end_index = np.cumsum([x.shape[0] - 1 for x in region_booldf_ysort.values()])
    if byregion == True:
        stim_count = booldf[list(booldf.keys())[0]].shape[1]
    else:
        stim_count = booldf.shape[1]
    fig, ax = plt.subplots(2, stim_count, figsize = (12, 5), dpi = 400,
                              gridspec_kw={'height_ratios': [1, 20], 'hspace': 0, 'wspace': 0.1})
    stim_col = 0
    for stim in constants.dir_sort: #sort stimulus in a way that makes more sense
        #shade stimuli on top
        c = constants.allcolor_dict[stim]
        ax[0, stim_col].axvspan(0, 2, facecolor = c[0], alpha = 0.5)
        ax[0, stim_col].axvspan(2.05, 4, facecolor = c[1], alpha = 0.5)
        ax[0, stim_col].axis('off')
        ax[0, stim_col].sharex(ax[1, stim_col])
        #plot histgram
        ax_hist = ax[1, stim_col]
        region_row = 1
        if byregion == True:
            ymax = 0
            ymin = 0
            for region in booldf.keys():
                sns.histplot(loc[region].loc[booldf[region][stim] == True], y = 'ypos', element = 'step',
                             bins = (loc[region]['ypos'].max() - loc[region]['ypos'].min())//15,
                             binrange = (loc[region]['ypos'].min(), loc[region]['ypos'].max()),
                             ax = ax_hist, fill = True, color = constants.cmaplist[region](0.5), alpha = 0.5)
                ymax = max(ymax, loc[region]['ypos'].max())
                ymin = min(ymin, loc[region]['ypos'].min())
                region_row += 1
        else:
            ymax = loc['ypos'].max()
            ymin = loc['ypos'].min()
            sns.histplot(loc.loc[booldf[stim] == True], y = 'ypos', element = 'step',
                         bins = (ymax - ymin)//15,
                         ax = ax_hist, fill = True, color = 'grey', alpha = 0.5)
        ax_hist.invert_yaxis()
        ax_hist.spines['top'].set_visible(False)
        ax_hist.spines['right'].set_visible(False)
        ax_hist.set_xlim(0, 20)
        ax_hist.set_ylim(ymax, ymin)
        if stim_col == 0:
            ax_hist.set_yticks([])
            ax_hist.set_ylabel('y location')
            ax_hist.set_xticks([0, 20])
            ax_hist.set_xlabel('Cell Count \n (by ' + tracename + ')')
        else:
            ax_hist.spines['bottom'].set_visible(False)
            ax_hist.set_yticks([])
            ax_hist.set_ylabel('')
            ax_hist.set_xticks([])
            ax_hist.set_xlabel('')
        stim_col = stim_col + 1


def plot_tuning_distribution_x(tracename, booldf, loc, degree_dict, response_dict, region_booldf,
                               region_loc, region_degree_dict, region_response_dict, refImg):
    """
    Plot the distribution of direction tunning cell by xloc
        booldf: the booldf DataFrame to be plotted
        region_booldf: the region dictionary that contains booldf for each region
        loc: the cell ROIs DataFrame. if region == True, it's a ddictionary that containing all regions and their ROIs. Noted the the sequence should match the trace sequence
        region_loc: the region dictionary that contains loc for each region
        degree_dict: the dictionary that contains all the stimuli and how the response in response_Dict is matched to each stimuli. if byregion == True, it's a dictionary containing all the regions.
        region_degree_dict: the region dictionary that contains degree_dict for each region
        response_dict: the dictionary that contains all the stimuli, and the corresponding average response from each cell to that stimuli. if byregion == True, it's a dictionary containing all the regions.
        region_response_dict: the region dictionary that contains response_dict for each region
        refImg: the reference image to plot
    Return: None
    """
    stim_count = len(constants.allcolor_dict.keys())
    region_count = 1 + len(region_booldf.keys()) + 2
    fig, ax = plt.subplots(region_count, stim_count, figsize = (20, 7), dpi = 400,
                              gridspec_kw={'height_ratios': [1 * region_count] + [10] * (region_count - 2) + [40],
                                           'hspace': 0.1, 'wspace': 0.1})
    stim_col = 0
    for stim in constants.allcolor_dict.keys(): #sort stimulus in a way that makes more sense
        #shade stimuli on top
        c = constants.allcolor_dict[stim]
        ax[0, stim_col].axvspan(0, 2, facecolor = c[0], alpha = 0.5)
        ax[0, stim_col].axvspan(2.05, 4, facecolor = c[1], alpha = 0.5)
        ax[0, stim_col].axis('off')
        ax[0, stim_col].set_xlim(0, 20)
        for region_row in range(1, region_count):
            if region_row < region_count - 1 and region_row > 1:
                regionname = list(region_booldf.keys())[region_row - 2]
                booldf_to_plot = region_booldf[regionname][stim]
                loc_to_plot = region_loc[regionname]
                degree_dict_to_plot = region_degree_dict[regionname]
                try:
                    response_index = list(degree_dict_to_plot.values())[0].index(constants.deg_dict[stim])
                    response_dict_to_plot = {key: region_response_dict[regionname][key] for key in region_response_dict[regionname].keys()}
                except:
                    response_dict_to_plot = {}
                for neuron in response_dict_to_plot.keys():
                    response_dict_to_plot[neuron] = response_dict_to_plot[neuron][response_index]
                region_c = constants.cmaplist[regionname](0.5)
            elif region_row == 1 or region_row == region_count - 1:
                regionname = 'all'
                booldf_to_plot = booldf[stim]
                loc_to_plot = loc
                degree_dict_to_plot = degree_dict
                try:
                    response_index = list(degree_dict_to_plot.values())[0].index(constants.deg_dict[stim])
                    response_dict_to_plot = {key: response_dict[key] for key in response_dict.keys()}
                except:
                    response_dict_to_plot = {}
                for neuron in response_dict_to_plot.keys():
                    response_dict_to_plot[neuron] = response_dict_to_plot[neuron][response_index]
                region_c = 'grey'
            #plot histogram
            if region_row < region_count - 1:
                ax_hist = ax[region_row, stim_col]
                sns.histplot(loc_to_plot[booldf_to_plot == True], x = 'xpos', element = 'step',
                                 bins = (refImg.shape[1] - 0)//30, binrange = (0, refImg.shape[1]),
                                 ax = ax_hist, fill = True, color = region_c, alpha = 0.5)
                ax_hist.spines['top'].set_visible(False)
                ax_hist.spines['right'].set_visible(False)
                ax_hist.set_ylim(0, 20)
                ax_hist.set_xlim(0, refImg.shape[1])
                ax_hist.set_xticks([])
                if stim_col == 0:
                    ax_hist.set_yticks([0, 20])
                    ax_hist.set_ylabel(regionname + ' #cell \n (by ' + tracename + ')')
                    if region_row == region_count - 2:
                        ax_hist.set_xlabel('xpos')
                    else:
                        ax_hist.set_xlabel('')
                else:
                    ax_hist.spines['bottom'].set_visible(False)
                    ax_hist.set_yticks([])
                    ax_hist.set_ylabel('')
                    ax_hist.set_xlabel('')
            elif region_row == region_count - 1:
                #plot refimg
                ax_scatter = ax[region_row, stim_col]
                ax_scatter.imshow(refImg, cmap = 'grey', alpha = 0.8, vmax = 100)
                responding_loc = loc[booldf_to_plot == True]
                responding_response = [response_dict_to_plot[key] for key in
                                       booldf_to_plot[booldf_to_plot == True].index]
                c_scatter = c[0]
                if c_scatter == [0, 0, 0]:
                    c_scatter = c[1]
                try:
                    ax_scatter.scatter(responding_loc['xpos'], responding_loc['ypos'],
                                       c = [c_scatter] * responding_loc.shape[0],
                                       alpha = np.divide(responding_response, np.nanmax(responding_response)), s = 3)
                except: pass
                ax_scatter.set_yticks([])
                ax_scatter.set_xticks([])
        stim_col = stim_col + 1

def plot_degree_tuned_distribution(all_degree, all_degreeval, region_degree, region_degreeval, tracename):
    """
    Plot the weighted degree tuning across all neurons for each region
        all_degree: the list that contains the degree_tuned for all neurons
        all_degreeval: the list that contains the degree_tuned weights for all neurons, note that its sequene should align with that of all_degree
        region_degree: the dictionary that contains the degree list for all regions
        region_degreeval: the dictionary that contains the degree weight list for all regions
        tracename: the type of signal used to plot
    Return: None
    """
    fig = plt.figure(figsize = (12, 12), dpi = 240)
    fig.subplots_adjust(wspace=.5)
    fig.subplots_adjust(hspace=.5)
    bins =list(range(0, 361, 30))
    region_row = 0
    region_list = ['all'] + list(region_degree.keys())
    for region in region_list:
        if region_row != 0:
            degree = region_degree[region]
            degreeval = region_degreeval[region]
            color = constants.cmaplist[region](0.4)
        else:
            degree = all_degree
            degreeval = all_degreeval
            color = 'grey'
        #plot histogram to summurize the degree distribution of all neurons
        ax_hist = fig.add_subplot(len(region_list), 3, region_row * 3 + 2, projection = 'polar')
        #calibrate -180 to 0 into 180 to 360
        for i in range(0, len(degree)):
            if np.isnan(degree[i]):
                degreeval[i] = np.nan
            elif degree[i] < 0:
                degree[i] = degree[i] + 360
        degree = [x for x in degree if not np.isnan(x)]
        degreeval = [x for x in degreeval if not np.isnan(x)]
        ax_hist.set_theta_offset(np.deg2rad(90))
        ax_hist.set_theta_direction('clockwise')
        ax_hist.hist(np.deg2rad(degree), np.deg2rad(bins), weights = degreeval, zorder = 10,
            color = color, edgecolor = color, alpha = 0.8)
        ax_hist.set_yticklabels([])
        ax_hist.set_yticks([])
        #plot histogram to summurize the degree distribution for individual neurons
        ax_scatter = fig.add_subplot(len(region_list), 3, region_row * 3 + 3, projection = 'polar')
        ax_scatter.set_theta_offset(np.deg2rad(90))
        ax_scatter.set_theta_direction('clockwise')
        # ax_scatter.scatter(np.deg2rad(degree), degreeval, s = np.multiply(degreeval, 100),
        #        c = np.deg2rad(degree), cmap = circmp, edgecolor = 'grey', alpha = 0.6, zorder = 10, linewidth = 0)
        ax_scatter.bar(np.deg2rad(degree), degreeval, #s = np.multiply(degreeval, 100),
                color = constants.circmp(np.divide(degree, 360)), alpha = 0.6, zorder = 10, width = 0.1)
        ax_scatter.set_yticks([])
        ax_scatter.set_yticklabels([])
        #plot cell category
        ax_label = fig.add_subplot(len(region_list), 3, region_row * 3 + 1)
        ax_label.axvline(0, c = color, linewidth = 60)
        ax_label.set_xlim(-10, 0)
        ax_label.set_xticks([])
        ax_label.set_yticks([])
        plt.setp(ax_label.spines.values(), visible=False)
        ax_label.text(-0.8, 0.5, region + ' cells \n (by ' + tracename + ')',
                      rotation = 'vertical', verticalalignment = 'center', horizontalalignment = 'center')
        region_row += 1

def plot_tuning_specificity(all_degree_dict, all_response_dict, all_booldf,
                            region_degree_dict, region_response_dict, region_booldf, tracename):
    """
    For neuron determined to be tuned to each stimuli, plot
        1) if they are tuned to other stimuli
        2) their mean response trace to other stimuli
        all_degree_dict: the dictoinary that contains all for all neurons
        all_degreeval_dict: the dictionary that contains the corresponding degree mean response for all neurons, note that its sequene should align with that of all_degree
        all_booldf: the dictionary that contains the booldf for all region
        region_degree_dict: the dictionary that contains the degree_dict for all regions
        region_degreeval_dict: the dictionary that contains the degree_dict for all regions
        region_booldf: the dictionary that contains the booldf for all region
        tracename: the type of signal used to plot
    Return: None
    """
    fig = plt.figure(figsize = (14, 10), dpi = 480)
    fig_row = (len(list(region_degree_dict.keys())) + 1) * 2
    fig_col = len(constants.monocular_dict.keys()) + 1
    region_list = ['all'] + list(region_degree_dict.keys())
    region_row = 0
    for region in region_list:
        stim_col = 2
        if region_row == 0:
            degree_dict = all_degree_dict
            response_dict = all_response_dict
            booldf = all_booldf
            color = 'grey'
        else:
            degree_dict = region_degree_dict[region]
            response_dict = region_response_dict[region]
            booldf = region_booldf[region]
            color = constants.cmaplist[region](0.4)
        for stim in constants.monocular_dict.keys():
            #count how many neurons are responding to other stimuli
            booldf_responding = booldf[booldf[stim] == True]
            index_n = booldf_responding.index
            responding_count = booldf_responding.sum()
            degree_list = []
            response_count_list = []
            for stim_other in constants.monocular_dict.keys():
                degree_list.append(constants.deg_dict[stim_other])
                response_count_list.append(responding_count[stim_other])
            degree_list.append(degree_list[0]) #circle back
            response_count_list.append(response_count_list[0]) #circle backh
            ax_count = fig.add_subplot(fig_row, fig_col, fig_col * region_row * 2 + stim_col, projection = 'polar')
            ax_count.set_yticks([])
            ax_count.set_yticklabels([])
            ax_count.set_xticklabels([])
            ax_count.set_theta_offset(np.deg2rad(90))
            ax_count.set_theta_direction('clockwise')
            ax_count.plot(np.deg2rad(degree_list), response_count_list, c = constants.circmp(constants.deg_dict[stim]/360))
            ax_count.fill_between(np.deg2rad(degree_list), response_count_list,
                                alpha = 0.1, color = constants.circmp(constants.deg_dict[stim]/360))
            #make individual neuron degree tuning graph
            ax_line = fig.add_subplot(fig_row, fig_col, fig_col * (region_row * 2 + 1) + stim_col, projection = 'polar')
            ax_line.set_theta_offset(np.deg2rad(90))
            ax_line.set_theta_direction('clockwise')
            ax_line.set_yticks([])
            ax_line.set_yticklabels([])
            ax_line.set_xticks([])
            ax_line.set_xticklabels([])
            for neuron in index_n:#degree_dict.keys():
                #sort angle from smallest to largest
                monoc_degree_dict = []
                monoc_response_dict = []
                for i in range(0, len(degree_dict[neuron])):
                    if degree_dict[neuron][i] < 361:
                        monoc_degree_dict.append(degree_dict[neuron][i])
                        monoc_response_dict.append(response_dict[neuron][i])
                degree_sort_index = np.argsort(monoc_degree_dict)
                response_sort = np.array(monoc_response_dict)[degree_sort_index]
                degree_sort = np.array(monoc_degree_dict)[degree_sort_index]
                response_sort = np.append(response_sort, response_sort[0])
                degree_sort = np.append(degree_sort, degree_sort[0])
                ax_line.plot(np.deg2rad(degree_sort), response_sort, alpha = 0.2, c = 'grey')
                ax_line.axvline(np.deg2rad(constants.deg_dict[stim]), c = constants.circmp(constants.deg_dict[stim]/360),
                                   alpha = 0.1, linewidth = 2, zorder = -5)
                ax_line.fill_between(np.deg2rad(degree_sort), response_sort, alpha = 0.1, color = 'grey')
            stim_col += 1
        #plot cell category
        ax_label = fig.add_subplot(fig_row, fig_col, region_row * 2 * fig_col + 1)
        ax_label.axvline(0, c = color, linewidth = 60)
        ax_label.set_xlim(-10, 0)
        ax_label.set_xticks([])
        ax_label.set_yticks([])
        plt.setp(ax_label.spines.values(), visible=False)
        ax_label.text(-2, 0.5, region + '\n (by ' + tracename + ')',
                      rotation = 'vertical', verticalalignment = 'center', horizontalalignment = 'center')
        ax_label_2 = fig.add_subplot(fig_row, fig_col, region_row * 2 * fig_col + fig_col + 1)
        ax_label_2.axvline(0, c = color, linewidth = 60)
        ax_label_2.set_xlim(-10, 0)
        ax_label_2.set_xticks([])
        ax_label_2.set_yticks([])
        plt.setp(ax_label_2.spines.values(), visible=False)
        ax_label_2.text(-2, 0.5, region + '\n (by ' + tracename + ')',
                      rotation = 'vertical', verticalalignment = 'center', horizontalalignment = 'center')
        region_row += 1

def cluster_on(frametimes_df, min_on_s, trace):
    """
    pull out traces when the cell is "on", according to a cluster method that fit the neuron fluorscence into "silent"
    cluster and "on" cluster. Cell traces that fire for more than 5 seconds are classified as on. Note that the input needs
    to be raw fluorscence and are smoothed with a 30 frame window and normalized.
        frametimes_df: the dataframe for all frames and their corresponding raw time
        min_on_s: the minimal on seconds to be counted as a spike (recommend: GCaMP6s 5 seconds; GCaMP6f: 2 seconds)
        trace: dictionary containing dataframe containing all traces for all regions, already noramlized and smoothed
    Return:
        mean_spike_trace: an ndarray that contains all cells, and their corresponding mean "on" traces
        mean_spike_duration: an ndarray that contains all cells's mean duration for staying on
    """
    mean_spike_trace = np.full((trace.shape[0], frametimes_df.shape[0]), np.nan)
    mean_spike_duration = np.full(trace.shape[0], np.nan)
    hz = hzReturner(frametimes_df)
    min_on_hz = min_on_s * hz
    for row in range(0, trace.shape[0]):
        gm = GaussianMixture(n_components=2, random_state=0).fit(pd.DataFrame(trace[row]))
        if gm.means_[0] > gm.means_[1]:
            calm = 1
        else:
            calm = 0
        boundary = (gm.means_[calm] + 1.8 * np.sqrt(gm.covariances_[calm]))[0][0]
        above = trace[row] > boundary
        above = [int(x) for x in above]
        #get on index, get off index
        on_index = np.where(np.diff(above) == 1)[0]
        off_index = np.where(np.diff(above) == -1)[0]
        if len(on_index) != 0 and len(off_index) != 0:
            if on_index[0] > off_index[0]:
                on_index = np.concatenate([[0], on_index])
            if on_index[-1] > off_index[-1]:
                off_index = np.concatenate([off_index, [len(list(trace[row])) - 1]])
            on_tuples = [(on, off) for on, off in zip(on_index, off_index) if off - on > min_on_hz]
            on_durations = [tuple[1] - tuple[0] for tuple in on_tuples]
            on_signals = np.full((len(on_tuples), np.max(on_durations)), boundary)
            for i in range(0, len(on_tuples)):
                on_signals[i][0:on_durations[i]]= trace[row][on_tuples[i][0]:on_tuples[i][1]]
        mean_spike_trace[row][0:np.max(on_durations)] = np.mean(on_signals, axis = 0)
        mean_spike_duration[row] = np.mean(on_durations)
    return mean_spike_trace, mean_spike_duration

def mean_on(frametimes_df, min_on_s, trace):
    """
    pull out traces when the cell is "on", according the on threshold defined by the mean fluorscence. Cell traces that fire
    for more than 5 seconds are classified as on. Note that the input needs to be raw fluorscence and are smoothed with a 30
    frame window and normalized.
        frametimes_df: the dataframe for all frames and their corresponding raw time
        min_on_s: the minimal on seconds to be counted as a spike (recommend: GCaMP6s 5 seconds; GCaMP6f: 2 seconds)
        trace: dictionary containing dataframe containing all traces for all regions, already noramlized and smoothed
    Return:
        mean_spike_trace: an ndarray that contains all cells, and their corresponding mean "on" traces
        mean_spike_duration: an ndarray that contains all cells's mean duration for staying on
        on_tuples: the on and off index of the trials
    """
    mean_spike_trace = np.full((trace.shape[0], frametimes_df.shape[0]), np.nan)
    mean_spike_duration = np.full(trace.shape[0], np.nan)
    hz = hzReturner(frametimes_df)
    min_on_hz = min_on_s * hz
    for row in range(0, trace.shape[0]):
        boundary = np.mean(trace[row])
        above = trace[row] > boundary
        above = [int(x) for x in above]
        #get on index, get off index
        on_index = np.where(np.diff(above) == 1)[0]
        off_index = np.where(np.diff(above) == -1)[0]
        if len(on_index) != 0 and len(off_index) != 0:
            if on_index[0] > off_index[0]:
                on_index = np.concatenate([[0], on_index])
            if on_index[-1] > off_index[-1]:
                off_index = np.concatenate([off_index, [len(list(trace[row])) - 1]])
            on_tuples = [(on, off) for on, off in zip(on_index, off_index) if off - on > min_on_hz and off - on < frametimes_df.shape[0]]
            if len(on_tuples) != 0:
                on_durations = [tuple[1] - tuple[0] for tuple in on_tuples]
                on_signals = np.full((len(on_tuples), np.max(on_durations)), boundary)
                for i in range(0, len(on_tuples)):
                    on_signals[i][0:on_durations[i]]= trace[row][on_tuples[i][0]:on_tuples[i][1]]
                mean_spike_trace[row][0:np.nanmax(on_durations)] = np.nanmean(on_signals, axis=0)
                mean_spike_duration[row] = np.nanmean(on_durations)

    return mean_spike_trace, mean_spike_duration

def plot_on(frametimes_df, region_trace, tracename, loc, on_method, min_on_s, cutoff_s, refImg, smooth = True):
    """
    Plot the mean "on" trace for each neuron. Note that the traces are normalized and smoothed when selecting "on periods".
     The smoothing factor is determined by frame rate * 10
        frametimes_df: the dataframe for all frames and their corresponding raw time
        region_trace: dictionary containing dataframe containing all traces for all regions, raw F!
        tracename: the cell trace name, used to label the graph
        loc: a dataframe containing all regions and their corresponding ROIs for each cell
        on_method: the method to selecton "on periods" ('cluster', 'mean')
        min_on_s: the minimal on seconds to be counted as a spike (recommend: GCaMP6s 5 seconds; GCaMP6f: 2 seconds)
        cutoff_s: the cut off seconds to differentiate on/off and peaky neurons
        refImg: the dataframe to plot the original fish plane
        smooth: boolean that determines if the cell trace will be smoothed
    """
    hz = hzReturner(frametimes_df)
    cbar = plt.get_cmap('rainbow')

    #preparing figure space
    region_count = len(region_trace.keys())
    fig, ax= plt.subplots(region_count + 1, 4,
                          gridspec_kw={'hspace': 0, 'wspace': 0.2, 'height_ratios':  [20] * region_count + [1]},
            figsize = (15, 6), dpi = 240, sharex = 'col')
    # add gridspace for all neuron correlation plot
    gs = ax[0, 3].get_gridspec()
    for axes in ax[0:, 3]:
        axes.remove()
    ax_scatter = fig.add_subplot(gs[0:, 3])
    ax_scatter.imshow(refImg, cmap='grey', alpha=0.8, vmax=100)
    ax_scatter.set_yticks([])
    ax_scatter.set_xticks([])
    # plot cbar
    color_cutoff = 60
    for i in range(0, 3):
        ax_cbar = ax[region_count, i]
        #if heatmap ax cbar, transfer to frames
        if i == 1:
            color_cutoff_hz = int(color_cutoff * hz)
            plotting_frame = int(np.floor(150 * hz) + 1)
            ax_cbar.scatter(np.linspace(0, color_cutoff_hz, color_cutoff_hz + 1), [0] * (color_cutoff_hz + 1),
                            c=np.linspace(0, color_cutoff_hz, color_cutoff_hz + 1), cmap=cbar)
            ax_cbar.scatter(np.linspace(color_cutoff_hz + 1, plotting_frame,
                                        plotting_frame - color_cutoff_hz),
                            [0] * (plotting_frame - color_cutoff_hz), c='red')
        #else, stay with seconds
        else:
            ax_cbar.scatter(np.linspace(0, color_cutoff, color_cutoff + 1), [0] * (color_cutoff + 1),
                        c=np.linspace(0, color_cutoff, color_cutoff + 1), cmap=cbar)
            ax_cbar.scatter(np.linspace(color_cutoff + 1, 150, 150 - color_cutoff), [0] * (150 - color_cutoff), c='red')
        ax_cbar.spines['top'].set_color('white')
        ax_cbar.spines['right'].set_color('white')
        ax_cbar.spines['bottom'].set_color('white')
        ax_cbar.spines['left'].set_color('white')
        ax_cbar.set_yticks([])
        ax_cbar.sharex(ax[0, i])
        ax_cbar.set_xlabel('time (s)')
    mean_on_trace = {key: np.empty((region_trace[key].shape[0], frametimes_df.shape[0])) for key in region_trace.keys()}
    mean_on_duration = {key: np.empty(region_trace[key].shape[0]) for key in region_trace.keys()}
    region_row = 0
    for region in region_trace.keys():
        #process data for each region
        trace = region_trace[region]
        for neuron in trace.index:
            if smooth:
                trace.loc[neuron, :] = arrutils.pretty(trace.loc[neuron, :], int(10 * hz))
        trace = arrutils.norm_0to1(trace.to_numpy())
        if on_method == 'cluster':
            mean_on_trace[region], mean_on_duration[region] = cluster_on(frametimes_df, min_on_s, trace)
        elif on_method == 'mean':
            mean_on_trace[region], mean_on_duration[region] = mean_on(frametimes_df, min_on_s, trace)
        #transfer everything to seconds
        mean_on_duration[region] = np.divide(mean_on_duration[region], hz)
        #sort neuron by duration
        sort_duration = np.argsort(mean_on_duration[region])
        sort_mean_on_trace = mean_on_trace[region][sort_duration]
        # plot heatmap
        region_heat_ax = ax[region_row, 1]
        sns.heatmap(sort_mean_on_trace[:, :plotting_frame], ax=region_heat_ax, cmap='viridis',
                    vmin=0, vmax=1, cbar=False)
        region_heat_ax.set_yticks([])
        region_heat_ax.set_xticks([])
        # plot histogram for each region
        region_hist_ax = ax[region_row, 2]
        peaky = mean_on_duration[region][np.where(mean_on_duration[region] < cutoff_s)[0]]
        onoff = mean_on_duration[region][np.where(mean_on_duration[region] >= cutoff_s)[0]]
        bins = np.linspace(0, 150, 11)
        peaky = np.divide(np.histogram(peaky, bins)[0], len(mean_on_duration[region])) *100#transfer to percentage
        onoff = np.divide(np.histogram(onoff, bins)[0], len(mean_on_duration[region])) *100
        bins_center = np.linspace(7.5, 142.5, 10)
        region_hist_ax.bar(bins_center, peaky, color = 'deepskyblue', alpha = 0.5, edgecolor = 'deepskyblue',
                           width = 15)
        region_hist_ax.bar(bins_center, onoff, color='coral', alpha=0.5, edgecolor='coral',
                          width = 15)
        region_hist_ax.set_ylim(0, 70)
        region_hist_ax.set_yticks([0, 70])
        region_hist_ax.axvline(cutoff_s, linestyle = ':', color = 'black', linewidth = 1)
        region_hist_ax.spines['top'].set_visible(False)
        region_hist_ax.spines['right'].set_visible(False)
        region_hist_ax.spines['bottom'].set_visible(False)
        #plot line plot for each region, plot after the other graphs because the duration is clipped
        mean_on_duration[region] = np.clip(mean_on_duration[region], 0, color_cutoff)/color_cutoff
        region_line_ax = ax[region_row, 0]
        for neuron in range(0, mean_on_trace[region].shape[0]):
            region_line_ax.plot(np.arange(0, 150, 1/hz), mean_on_trace[region][neuron][:plotting_frame],
                                c = cbar(mean_on_duration[region][neuron]), linewidth = 0.1)
        region_line_ax.axvline(cutoff_s, linestyle = ':', color = 'black', linewidth =1)
        region_line_ax.set_xlim(0, 150)
        region_line_ax.set_ylim(0, 1)
        region_line_ax.set_xticks([])
        region_line_ax.set_yticks([0, 1])
        region_line_ax.set_ylabel(region)
        region_line_ax.spines['top'].set_visible(False)
        region_line_ax.spines['right'].set_visible(False)
        region_line_ax.spines['bottom'].set_visible(False)
        if region_row == 0:
            region_line_ax.set_yticklabels([0, 1])
            region_heat_ax.set_ylabel(tracename)
            region_hist_ax.set_yticklabels([0, 70])
            region_hist_ax.set_ylabel('cell%')
        else:
            region_line_ax.set_yticklabels([])
            region_hist_ax.set_ylabel('')
            region_hist_ax.set_yticklabels([])
        ax_scatter.scatter(loc[region]['xpos'], loc[region]['ypos'], c = mean_on_duration[region], s = 0.1, cmap = cbar)
        region_row += 1
    return color_cutoff, mean_on_duration


def mean_on_cont(frametimes_df, min_on_s, interval_s, trace):
    """
    pull out traces when the cell is "on", according the on threshold defined by the mean fluorscence. Cell traces that fire
    for more than 5 seconds are classified as on. Note that the input needs to be raw fluorscence and are smoothed with a 30
    frame window and normalized.
        frametimes_df: the dataframe for all frames and their corresponding raw time
        min_on_s: the min seconds a spike should be on
        interval_s: the min seconds that two on events need to be apart
        trace: dictionary containing dataframe containing all traces for all regions, already noramlized and smoothed
    Return:
        all_spiking: an ndarray that contains all cells, and wheather they are spiking or not at the current frame (0 and 1)
        mean_spike_duration: an ndarray that contains all cells's mean duration for staying on
    """
    all_spiking = np.full((trace.shape[0], frametimes_df.shape[0]), 0)
    mean_spike_duration = np.full(trace.shape[0], 0)
    hz = hzReturner(frametimes_df)
    min_on_time = min_on_s * hz
    for row in range(0, trace.shape[0]):
        boundary = np.nanmean(trace[row])
        above = trace[row] > boundary
        above = [int(x) for x in above]
        #get on index, get off index
        on_index = np.where(np.diff(above) == 1)[0]
        off_index = np.where(np.diff(above) == -1)[0]
        if len(on_index) != 0 and len(off_index) != 0:
            if on_index[0] > off_index[0]:
                on_index = np.concatenate([[0], on_index])
            if on_index[-1] > off_index[-1]:
                off_index = np.concatenate([off_index, [len(list(trace[row])) - 1]])
            on_tuples = [(on, off) for on, off in zip(on_index, off_index) if off - on > min_on_time]
            #find continuous spikes
        cont_on_index = []
        cont_off_index = []
        if len(on_tuples) > 0:
            cont_on_index = [on_tuples[0][0]]
            cont_cutoff_hz = interval_s * hz
            if len(on_tuples) > 1:
                big_interval = [on_tuples[i][0] - on_tuples[i - 1][1] for i in range(1, len(on_tuples))] > cont_cutoff_hz
                for i in range(0, len(big_interval)):
                    if big_interval[i]:
                        cont_on_index = cont_on_index + [on_tuples[i + 1][0]]
                        cont_off_index = cont_off_index + [on_tuples[i][1]]
            cont_off_index = cont_off_index + [on_tuples[-1][1]]
        cont_tuples = [(on, off) for on, off in zip(cont_on_index, cont_off_index)]
        cont_durations = [tuple[1] - tuple[0] for tuple in cont_tuples]
        if len(cont_tuples) != 0:
            for i in range(0, len(cont_tuples)):
                all_spiking[row][cont_tuples[i][0]:cont_tuples[i][1]] = 1
            mean_spike_duration[row] = np.nanmax(cont_durations)
    return all_spiking, mean_spike_duration

def peak_on_cont(frametimes_df, min_on_s, interval_s, trace):
    """
    pull out traces when the cell is "on", according the on threshold defined by the find_peak method. Cell traces that fire
    for more than 5 seconds are classified as on. Note that the input needs to be raw fluorscence and are smoothed with a 30
    frame window and normalized.
        frametimes_df: the dataframe for all frames and their corresponding raw time
        min_on_s: the min seconds a spike should be on
        interval_s: the min seconds that two on events need to be apart
        trace: dictionary containing dataframe containing all traces for all regions, already noramlized and smoothed
    Return:
        all_spiking: an ndarray that contains all cells, and wheather they are spiking or not at the current frame (0 and 1)
        mean_spike_duration: an ndarray that contains all cells's mean duration for staying on
    """
    all_spiking = np.full((trace.shape[0], frametimes_df.shape[0]), 0)
    mean_spike_duration = np.full(trace.shape[0], 0)
    hz = hzReturner(frametimes_df)
    min_on_hz = min_on_s * hz
    for row in range(0, trace.shape[0]):
        #get on index, get off index
        on_index = find_peaks(trace[row], height=0.05, distance= 1)[0]
        off_index = np.add(on_index, 1)
        if len(on_index) != 0 and len(off_index) != 0:
            if on_index[0] > off_index[0]:
                on_index = np.concatenate([[0], on_index])
            if on_index[-1] > off_index[-1]:
                off_index = np.concatenate([off_index, [len(list(trace[row])) - 1]])
            on_tuples = [(on, off) for on, off in zip(on_index, off_index) if off - on > min_on_hz]
            #find continuous spikes
        cont_on_index = []
        cont_off_index = []
        if len(on_tuples) > 0:
            cont_on_index = [on_tuples[0][0]]
            cont_cutoff_hz = interval_s * hz
            if len(on_tuples) > 1:
                big_interval = [on_tuples[i][0] - on_tuples[i - 1][1] for i in range(1, len(on_tuples))] > cont_cutoff_hz
                for i in range(0, len(big_interval)):
                    if big_interval[i]:
                        cont_on_index = cont_on_index + [on_tuples[i + 1][0]]
                        cont_off_index = cont_off_index + [on_tuples[i][1]]
            cont_off_index = cont_off_index + [on_tuples[-1][1]]
        cont_tuples = [(on, off) for on, off in zip(cont_on_index, cont_off_index)]
        cont_durations = [tuple[1] - tuple[0] for tuple in cont_tuples]
        if len(cont_tuples) != 0:
            for i in range(0, len(cont_tuples)):
                all_spiking[row][cont_tuples[i][0]:cont_tuples[i][1]] = 1
            mean_spike_duration[row] = np.nanmax(cont_durations)
    return all_spiking, mean_spike_duration

def plot_on_cont(frametimes_df, region_trace, tracename, loc, on_method, min_on_s, cutoff_s, interval_s, refImg, smooth = False):
    """
    Plot the mean "on" trace for each neuron. Note that the traces are normalized and smoothed when selecting "on periods".
     The smoothing factor is determined by frame rate * 10
        frametimes_df: the dataframe for all frames and their corresponding raw time
        region_trace: dictionary containing dataframe containing all traces for all regions, norm F!
        tracename: the cell trace name, used to label the graph
        loc: a dataframe containing all regions and their corresponding ROIs for each cell
        on_method: the method to selecton "on periods" (mean', 'peak')
        min_on_s: the min seconds a spike should be on
        cutoff_s: the cut off seconds to differentiate on/off and peaky neurons
        interval_s: the minimal interval in s in second between differet on periods
        refImg: the dataframe to plot the original fish plane
        smooth: boolean that determines if the cell trace will be smoothed
    """
    hz = hzReturner(frametimes_df)
    cbar = plt.get_cmap('rainbow')

    #preparing figure space
    region_count = len(region_trace.keys())
    fig, ax= plt.subplots(region_count + 1, 4,
                          gridspec_kw={'hspace': 0, 'wspace': 0.2, 'height_ratios':  [20] * region_count + [1]},
            figsize = (15, 6), dpi = 240, sharex = 'col')
    # add gridspace for all neuron correlation plot
    gs = ax[0, 3].get_gridspec()
    for axes in ax[0:, 3]:
        axes.remove()
    ax_scatter = fig.add_subplot(gs[0:, 3])
    ax_scatter.imshow(refImg, cmap='grey', alpha=0.8, vmax=100)
    ax_scatter.set_yticks([])
    ax_scatter.set_xticks([])
    # plot cbar
    color_cutoff = 60
    plotting_cutoff = 150
    ax[region_count, 1].axis('off')
    for i in [0, 2]:
        ax_cbar = ax[region_count, i]
        ax_cbar.scatter(np.linspace(0, color_cutoff, color_cutoff + 1), [0] * (color_cutoff + 1),
                        c=np.linspace(0, color_cutoff, color_cutoff + 1), cmap=cbar, s = 10)
        ax_cbar.scatter(np.linspace(color_cutoff + 1, plotting_cutoff, plotting_cutoff - color_cutoff),
                        [0] * (plotting_cutoff - color_cutoff),
                        c='red', s=10)
        ax_cbar.spines[['top', 'right', 'bottom', 'left']].set_color('white')
        ax_cbar.set_yticks([])
        ax_cbar.sharex(ax[0, i])
        ax_cbar.set_xlabel('max on duration (s)')
    on_trace = {key: np.empty((region_trace[key].shape[0], frametimes_df.shape[0])) for key in region_trace.keys()}
    mean_on_duration = {key: np.empty(region_trace[key].shape[0]) for key in region_trace.keys()}
    region_row = 0
    for region in region_trace.keys():
        #process data for each region
        trace = region_trace[region]
        for neuron in trace.index:
            if smooth:
                trace.loc[neuron, :] = arrutils.pretty(trace.loc[neuron, :], int(10 * hz))
        trace = np.array(trace)
        if on_method == 'mean_cont':
            on_trace[region], mean_on_duration[region] = mean_on_cont(frametimes_df, min_on_s, interval_s, trace)
        elif on_method == 'peak_cont':
            on_trace[region], mean_on_duration[region] = peak_on_cont(frametimes_df, min_on_s, interval_s, trace)
        #transfer everything to seconds
        mean_on_duration[region] = np.divide(mean_on_duration[region], hz)
        # sort neuron by duration
        sort_duration = np.argsort(mean_on_duration[region])
        sort_on_trace = on_trace[region][sort_duration]
        sort_mean_on_duration = mean_on_duration[region][sort_duration]
        sort_mean_on_duration = np.clip(sort_mean_on_duration, 0, plotting_cutoff)
        mean_on_duration[region] = np.clip(mean_on_duration[region], 0, color_cutoff) / color_cutoff
        # plot heatmap
        region_heat_ax = ax[region_row, 1]
        sns.heatmap(on_trace[region], ax=region_heat_ax, cmap='viridis', vmin=0, vmax=1, cbar=False)
        region_heat_ax.set_yticks([])
        region_heat_ax.set_xticks([])
        # plot histogram for each region
        region_hist_ax = ax[region_row, 2]
        peaky = sort_mean_on_duration[np.where(sort_mean_on_duration < cutoff_s)[0]]
        onoff = sort_mean_on_duration[np.where(sort_mean_on_duration >= cutoff_s)[0]]
        bins = np.linspace(0, plotting_cutoff, 11)
        peaky = np.divide(np.histogram(peaky, bins)[0], len(sort_mean_on_duration)) *100#transfer to percentage
        onoff = np.divide(np.histogram(onoff, bins)[0], len(sort_mean_on_duration)) *100
        bins_center = np.linspace(plotting_cutoff/20, plotting_cutoff - plotting_cutoff/20, 10)
        region_hist_ax.bar(bins_center, peaky, color = 'deepskyblue', alpha = 0.5, edgecolor = 'deepskyblue',
                           width = plotting_cutoff/10)
        region_hist_ax.bar(bins_center, onoff, color='coral', alpha=0.5, edgecolor='coral',
                          width = plotting_cutoff/10)
        region_hist_ax.set_ylim(0, 70)
        region_hist_ax.set_yticks([0, 70])
        region_hist_ax.axvline(cutoff_s, linestyle = ':', color = 'black', linewidth = 1)
        region_hist_ax.spines[['top', 'right', 'bottom']].set_visible(False)
        #plot line plot for each region, plot after the other graphs because the duration is clipped and normalized
        region_line_ax = ax[region_row, 0]
        for neuron in range(0, sort_on_trace.shape[0]):
            region_line_ax.plot(np.arange(0, int(sort_mean_on_duration[neuron]), 1), [neuron] * int(sort_mean_on_duration[neuron]),
                                c = cbar(np.min([color_cutoff, sort_mean_on_duration[neuron]])/color_cutoff), linewidth = 1)
        region_line_ax.axvline(cutoff_s, linestyle = ':', color = 'black', linewidth =1)
        region_line_ax.set_xlim(0, plotting_cutoff)
        region_line_ax.sharey(region_heat_ax)
        region_line_ax.set_xticks([])
        region_line_ax.set_ylabel(region + ' max on duration (s)')
        region_line_ax.spines[['top', 'right', 'bottom']].set_visible(False)
        if region_row == 0:
            region_heat_ax.set_ylabel(tracename + ' on segments')
            region_hist_ax.set_yticklabels([0, 70])
            region_hist_ax.set_ylabel('cell%')
        elif region_row == region_count - 1:
            region_heat_ax.set_xlabel('time (s)')
        ax_scatter.scatter(loc[region]['xpos'], loc[region]['ypos'],
                           c = mean_on_duration[region], s = 0.1, cmap = cbar)
        region_row += 1
    return color_cutoff, mean_on_duration


def firingrate(frametimes_df, min_on_s, trace, trial_s):
    """
    Find the peaks of individual firing event. Cell traces that fire for more than 5 seconds and with a peak fluorscence
    > 0.2 are classified as a firing event. Note that the input needs to be raw fluorscence normalized without smoothing.
        frametimes_df: the dataframe for all frames and their corresponding raw time
        min_on_s: the min seconds a spike should be on
        trace: dictionary containing dataframe containing all traces for all regions, already noramlized and hopefull
         no need to smooth
         trial_s: the seonds per trial
    Return:
        all_peaks_index: an ndarray that contains all cells, with the frame with a peak labeled 1 and the rest 0
        all_fr_min: an ndarray that contains all cells, and their rolling firing rate per trial in each frame
        all_fr_min_mean: an ndarray that contains all cells, and their mean firing rate/min only when they are responding
    """
    all_peaks = np.empty((trace.shape[0], frametimes_df.shape[0]))
    all_fr_min = np.empty((trace.shape[0], frametimes_df.shape[0]))
    hz = hzReturner(frametimes_df)
    min_on_time = min_on_s * hz
    for row in range(0, trace.shape[0]):
        peaks_index = find_peaks(trace[row], height = 0.2, distance = min_on_time + 1)[0]
        peaks = np.full(len(trace[row]), 0)
        peaks[peaks_index] = 1
        #print(peaks_index)
        #peaks = [1 for frame in range(0, len(trace[row])) if frame in peaks_index else 0]
        frames_pertrial = int(hz * trial_s)
        fr_min = np.cumsum(peaks)
        fr_min[frames_pertrial:] = fr_min[frames_pertrial:] - fr_min[:-frames_pertrial]
        all_peaks[row] = peaks
        all_fr_min[row] = fr_min
    all_fr_min_mean = np.nanmean(np.ma.masked_equal(all_fr_min, 0), axis = 1)#np.max(all_fr_min, axis = 1)
    return all_peaks, all_fr_min, all_fr_min_mean

def plot_fr(frametimes_df, region_trace, tracename, loc, min_on_s, cutoff_fr, trial_s, refImg):
    """
    Plot the peaks in calcium signals that symbolizing firing events and the relative firing rate of each cell.
        frametimes_df: the dataframe for all frames and their corresponding raw time
        region_trace: dictionary containing dataframe containing all traces for all regions, normalized F!
        tracename: the cell trace name, used to label the graph
        loc: a dataframe containing all regions and their corresponding ROIs for each cell
        on_method: the method to selecton "on periods" ('cluster', 'mean')
        cutoff_fr: the cut off spikes/min to differentiate on/off and peaky neurons
        trial_s: the seonds per trial
        refImg: the dataframe to plot the original fish plane
    """
    hz = hzReturner(frametimes_df)
    cbar = plt.get_cmap('rainbow')

    # preparing figure space
    region_count = len(region_trace.keys())
    fig, ax = plt.subplots(region_count + 1, 4,
                           gridspec_kw={'hspace': 0, 'wspace': 0.2, 'height_ratios': [20] * region_count + [1]},
                           figsize=(15, 6), dpi=240, sharex='col')
    # add gridspace for all neuron correlation plot
    gs = ax[0, 3].get_gridspec()
    for axes in ax[0:, 3]:
        axes.remove()
    ax_scatter = fig.add_subplot(gs[0:, 3])
    ax_scatter.imshow(refImg, cmap='grey', alpha=0.8, vmax=100)
    ax_scatter.set_yticks([])
    ax_scatter.set_xticks([])
    # plot cbar
    color_cutoff = 8
    plotting_cutoff = 8
    ax[region_count, 0].axis('off')
    ax[region_count, 1].axis('off')
    ax_cbar = ax[region_count, 2]
    ax_cbar.scatter(np.linspace(0, color_cutoff, color_cutoff + 1), [0] * (color_cutoff + 1),
                    c=np.linspace(0, color_cutoff, color_cutoff + 1), cmap=cbar, s=10)
    ax_cbar.scatter(np.linspace(color_cutoff + 1, plotting_cutoff, plotting_cutoff - color_cutoff),
                    [0] * (plotting_cutoff - color_cutoff),
                    c='red', s=10)
    ax_cbar.spines['top'].set_color('white')
    ax_cbar.spines['right'].set_color('white')
    ax_cbar.spines['bottom'].set_color('white')
    ax_cbar.spines['left'].set_color('white')
    ax_cbar.set_yticks([])
    ax_cbar.sharex(ax[0, 2])
    ax_cbar.set_xlabel('firing rate (spikes/trial)')

    all_peaks = {key: np.empty((region_trace[key].shape[0], frametimes_df.shape[0])) for key in region_trace.keys()}
    all_fr_min = {key: np.empty((region_trace[key].shape[0], frametimes_df.shape[0])) for key in region_trace.keys()}
    all_fr_min_mean = {key: np.empty(region_trace[key].shape[0]) for key in region_trace.keys()}
    region_row = 0
    for region in region_trace.keys():
        # process data for each region
        trace = region_trace[region].to_numpy()
        all_peaks[region], all_fr_min[region], all_fr_min_mean[region] = firingrate(frametimes_df, min_on_s, trace, trial_s)
        # sort neuron by max
        sort_duration = np.argsort(all_fr_min_mean[region])
        sort_peaks = all_peaks[region][sort_duration]
        sort_fr_min = all_fr_min[region][sort_duration]
        sort_fr_min_mean = all_fr_min_mean[region][sort_duration]
        sort_fr_min_mean = np.clip(sort_fr_min_mean, 0, plotting_cutoff)
        all_fr_min_mean[region] = np.divide(np.clip(all_fr_min_mean[region], 0, color_cutoff), color_cutoff)
        # plot heatmap
        region_heat_ax = ax[region_row, 1]
        sns.heatmap(sort_fr_min, ax=region_heat_ax, cmap='rainbow', vmin=0, vmax=color_cutoff, cbar=False)
        region_heat_ax.set_yticks([])
        region_heat_ax.set_xticks([])
        # plot histogram for each region
        region_hist_ax = ax[region_row, 2]
        peaky = sort_fr_min_mean[np.where(sort_fr_min_mean < cutoff_fr)[0]]
        onoff = sort_fr_min_mean[np.where(sort_fr_min_mean >= cutoff_fr)[0]]
        bins = np.linspace(0, plotting_cutoff, 11)
        peaky = np.divide(np.histogram(peaky, bins)[0], len(sort_fr_min_mean)) * 100  # transfer to percentage
        onoff = np.divide(np.histogram(onoff, bins)[0], len(sort_fr_min_mean)) * 100
        bins_center = np.linspace(plotting_cutoff/20, plotting_cutoff - plotting_cutoff/20, 10)
        region_hist_ax.bar(bins_center, peaky, color='deepskyblue', alpha=0.5, edgecolor='deepskyblue',
                           width=plotting_cutoff/10)
        region_hist_ax.bar(bins_center, onoff, color='coral', alpha=0.5, edgecolor='coral',
                           width=plotting_cutoff/10)
        region_hist_ax.set_xlim(0, plotting_cutoff)
        region_hist_ax.set_ylim(0, 70)
        region_hist_ax.set_yticks([0, 70])
        region_hist_ax.axvline(cutoff_fr, linestyle=':', color='black', linewidth=1)
        region_hist_ax.spines[['top', 'right', 'bottom']].set_visible(False)
        # plot dot plot for each region
        region_line_ax = ax[region_row, 0]
        for neuron in range(0, len(sort_peaks)):
            peaks_index = np.argwhere(sort_peaks[neuron] == 1)
            region_line_ax.scatter(peaks_index, [neuron] * len(peaks_index),
                                   color = cbar(np.min([color_cutoff, sort_fr_min_mean[neuron]])/color_cutoff), s = 0.05)
        region_line_ax.set_xticks([])
        region_line_ax.set_yticks([])
        region_line_ax.set_ylabel(region + ' spikes')
        region_line_ax.sharey(region_heat_ax)
        region_line_ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
        #region_line_ax.spines['right'].set_visible(False)
        if region_row == 0:
            region_heat_ax.set_ylabel('spikes/min')
            region_hist_ax.set_yticklabels([0, 70])
            region_hist_ax.set_ylabel('cell%')
        elif region_row == region_count - 1:
            region_line_ax.set_xlabel('time (frame)')
            region_heat_ax.set_xlabel('time (frame)')
        ax_scatter.scatter(loc[region]['xpos'], loc[region]['ypos'], c=all_fr_min_mean[region], s=1, cmap=cbar)
        region_row += 1
    return color_cutoff, all_fr_min_mean


