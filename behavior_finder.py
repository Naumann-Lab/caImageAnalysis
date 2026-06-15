import numpy as np
import pandas as pd
import seaborn as sns
import sys, os, shutil
sys.path.append(r'C://Users//Zichen//anaconda3//envs//2ptank//Lib//site-packages//')
import utils
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colormaps

pink = "#FB008C"
green = "#72D100"
cmap_conv = LinearSegmentedColormap.from_list("conv",[green, (1, 1, 1), pink])
cmap_conj = colormaps.get_cmap('coolwarm_r')

plt.rcParams.update({'font.size': 13})

def saccade_finder(fish, time_window_s = 1, min_interval_s = .5, min_saccade_s = 0.1, startframe = 0, endframe = 10000,
                   lowpass = True, std_scale = 1):
    """
    Find the saccades, which can be composed of one big eye movement or both eyes moving
    :param fish:
    :param time_window_s: the time window that used to calculate a moving std
    :param min_iterval_s:
    :param min_saccade_s:
    :return:
    """
    fig, ax = plt.subplots(3, 1, dpi = 240, figsize = (12, 5))
    eye_frame_s = np.diff(fish.eye_df['real_time_s']).mean()#mean frequency for the eyes
    frame_window = int(time_window_s/eye_frame_s)

    #run the lowpass filter
    es_pass = []
    if np.mean(fish.eye_df.th_e0) <= np.mean(fish.eye_df.th_e1):
        eyelist = ['th_e0', 'th_e1']
        print('usual 2P setup')
    else:
        eyelist = ['th_e0', 'th_e1']#['th_e1', 'th_e0']
        print('eyes may be flipped')
    for e, eye in enumerate(eyelist):
        ax[e].plot(fish.eye_df.loc[:, eye][startframe:endframe], linewidth = .5, color = 'dimgrey')
        if lowpass:
            e_pass = utils.butter_lowpass_filter(fish.eye_df.loc[:, eye], cutoff = 1, fs = 1/eye_frame_s)#1-5
            ax[e].plot(range(startframe, endframe), e_pass[startframe:endframe], linewidth = 1, color = 'black')
            fish.eye_df.loc[:, eye + '_pass'] = e_pass
            es_pass.append(e_pass)
        else:
            es_pass = [list(fish.eye_df['th_e0']), list(fish.eye_df['th_e1'])]
        ax[e].set_ylim([70, 130])#2p: 70, 130; behavior rig:-110, -60
    ax[0].set_ylabel('left eye angle (°)')
    ax[1].set_ylabel('right eye angle (°)')

    #get the rolling std
    e0_stde = pd.Series(es_pass[0]).rolling(window = frame_window).std()
    e1_stde = pd.Series(es_pass[1]).rolling(window=frame_window).std()
    e_std = np.add(e0_stde, e1_stde)
    ax[2].plot(e_std[startframe:endframe], linewidth = 2, color = 'black')
    boundary = boundary = np.mean(e_std) + std_scale * np.std(e_std)
    ax[2].axhline(boundary, color = 'black', linewidth = 1, linestyle = ':')

    #get saccades higher than the std
    saccade_on = e_std > boundary
    saccade_on = [int(x) for x in saccade_on]
    on_index = np.where(np.diff(saccade_on) == 1)[0]
    off_index = np.where(np.diff(saccade_on) == -1)[0]
    on_tuples = []
    if len(on_index) != 0 and len(off_index) != 0:
        if on_index[0] > off_index[0]:
            on_index = np.concatenate([[0], on_index])
        if on_index[-1] > off_index[-1]:
            off_index = np.concatenate([off_index, [len(fish.eye_df) - 1]])
        on_tuples = [(on, off) for on, off in zip(on_index, off_index) if off - on < len(fish.eye_df) and off > on]
    cont_on_index = []
    cont_off_index = []
    if len(on_tuples) > 0:
        cont_on_index = [on_tuples[0][0]]
        if len(on_tuples) > 1:
            big_interval = np.array([on_tuples[i][0] - on_tuples[i - 1][1] for i in range(1, len(on_tuples))]) \
                                   > (min_interval_s/eye_frame_s)
            for i in range(0, len(big_interval)):
                if big_interval[i]:
                    cont_on_index = cont_on_index + [on_tuples[i + 1][0]]
                    cont_off_index = cont_off_index + [on_tuples[i][1]]
        cont_off_index = cont_off_index + [on_tuples[-1][1]]
    cont_tuples = [(on, off) for on, off in zip(cont_on_index, cont_off_index) if off - on > (min_saccade_s/eye_frame_s)]
    cont_tuples_realtime_s = [(fish.eye_df.real_time_s.iloc[tu[0]], fish.eye_df.real_time_s.iloc[tu[1]]) for tu in cont_tuples]

    #get saccade information
    eye_fromstart_s = np.full(len(cont_tuples), np.nan)
    e0_change = np.full(len(cont_tuples), np.nan)
    e1_change = np.full(len(cont_tuples), np.nan)
    eye_stimuli = np.full(len(cont_tuples), np.nan, dtype = object)
    eye_realstimuli = np.full(len(cont_tuples), np.nan, dtype = object)
    def find_eyechange(e, pre_e):
        """Find the max movement of the eye"""
        e = list(e)
        mine = min(e)
        maxe = max(e)
        minchange = mine - pre_e
        maxchange = maxe - pre_e
        if np.abs(minchange) > np.abs(maxchange):
            return minchange
        else:
            return maxchange
    for i in range(len(cont_tuples)):
        e0 = fish.eye_df.th_e0_pass[cont_tuples[i][0]:cont_tuples[i][1]]
        e1 = fish.eye_df.th_e1_pass[cont_tuples[i][0]:cont_tuples[i][1]]
        pre_e0 = fish.eye_df.th_e0_pass[cont_tuples[i][0]-5:cont_tuples[i][0]].mean()
        pre_e1 = fish.eye_df.th_e1_pass[cont_tuples[i][0]-5:cont_tuples[i][0]].mean()
        e0_change[i] = find_eyechange(e0, pre_e0)
        e1_change[i] = find_eyechange(e1, pre_e1)
        stim_delay_s = np.subtract(cont_tuples_realtime_s[i][0], fish.stimulus_df.real_starttime_s)
        stim_delay_s = [t for t in stim_delay_s if t>0]
        if len(stim_delay_s) > 0:#there is stimulus
            stim_nearest_row = np.argmin(stim_delay_s)#all the stimulus happening before this tail
            stim_delay_s = stim_delay_s[stim_nearest_row]
            stim_name = fish.stimulus_df['stim_name'].iloc[stim_nearest_row]
            stim_duration = fish.stimulus_df['duration'].iloc[stim_nearest_row]
            stim_stationarytime = fish.stimulus_df['stationary_time'].iloc[stim_nearest_row]
            if type(stim_name) == list or '[' in stim_name: #overlap stimulus
                if stim_delay_s < min(stim_stationarytime):
                    realstim_name = 'stationary'
                elif stim_delay_s < max(stim_stationarytime):
                    realstim_name = stim_name[1]
                else:
                    realstim_name = stim_name
            else:#one stimulus
                if stim_delay_s < stim_stationarytime:
                    realstim_name = 'stationary'
                else:
                    realstim_name = stim_name
        else:
            stim_delay_s = np.nan
            stim_name = np.nan
            realstim_name= np.nan
        eye_fromstart_s[i] = stim_delay_s
        eye_stimuli[i] = stim_name
        eye_realstimuli[i] = str(realstim_name)
    eye_saccade_df = pd.DataFrame(
            {'cont_tuples_eyeindex': cont_tuples, 'cont_tuples_realtime_s': cont_tuples_realtime_s,
             'e0_change': e0_change, 'e1_change': e1_change, 'eye_stimuli': eye_stimuli, 'eye_realstimuli': eye_realstimuli, 'eye_fromstart_s': eye_fromstart_s
             })
    if np.mean(fish.eye_df.th_e0) <= np.mean(fish.eye_df.th_e1):
        eye_saccade_df.loc[:, "e_conv_change"] = eye_saccade_df.e1_change  - eye_saccade_df.e0_change#2p: big = big e1 and small e0 = converge
    else:
        eye_saccade_df.loc[:, "e_conv_change"] = eye_saccade_df.e1_change  - eye_saccade_df.e0_change#eye_saccade_df.e0_change - eye_saccade_df.e1_change
    eye_saccade_df.loc[:, "e_conj_change"] = eye_saccade_df.e1_change  + eye_saccade_df.e0_change#2p: big = right = big e1 and big e0
    fish.saccade_df = eye_saccade_df
    #color code all the saccades
    for n, saccade in eye_saccade_df.iterrows():
        cont_tuple = saccade['cont_tuples_eyeindex']
        if cont_tuple[1] < endframe and cont_tuple[0] > startframe:
            ax[2].axvspan(cont_tuple[0], cont_tuple[1], ymin = 0, ymax = 0.6, color= 'lightgrey', alpha = 0.8)#converging index
            conv_mapping = np.clip(saccade.e_conv_change, -10, 10)
            c = cmap_conv((conv_mapping +10)/20)
            ax[2].scatter((cont_tuple[0] + cont_tuple[1])/2, 8, color= c, alpha = 1)#converging index
            conj_mapping = np.clip(saccade.e_conj_change, -20, 20)
            c = cmap_conj((conj_mapping + 20)/40)
            ax[2].scatter((cont_tuple[0] + cont_tuple[1])/2, 9, color= c, alpha = 1)
    ax[2].set_ylim([0, 10])
    ax[2].set_ylabel('running std')
    #make axis prettier
    for axis in ax:
        axis.grid(False)
        axis.set_xticks([])
        axis.spines[['top', 'right']].set_visible(False)
    ax[0].axhline(y = 120, xmax = .95, xmin = .95 - (10/eye_frame_s)/(endframe - startframe), linewidth = 3, color = 'grey')#120
    ax[0].text(endframe - (15/eye_frame_s), 122, '10s', c = 'grey')
    return fig

def plot_saccade_distribution(fish,
                              stimorder =['stationary', 'forward', 'backward', 'left', 'right', 'dot_l', 'dot_r',
                 str(['dot_l', 'forward']), str(['dot_l', 'backward']), str(['dot_l', 'left']), str(['dot_l', 'right']),
                 str(['dot_r', 'forward']), str(['dot_r', 'backward']), str(['dot_r', 'left']), str(['dot_r', 'right']),]):
    """
    For each stimulus, what does the saccade looks like
    :param fish:
    :return:
    """
    fig, ax = plt.subplots(1, 3, figsize = (10, 10), dpi = 240)

    #stimorder = #[f'dot_{side}_3',  f'dot_{side}_5', f'dot_{side}_10',f'dot_{side}_20', f'dot_{side}_40']
    temp_saccade_df = fish.saccade_df.melt(id_vars = 'eye_realstimuli', value_vars = ['e0_change', 'e1_change'], var_name= 'eye', value_name = 'change')
    sns.stripplot(temp_saccade_df, y = 'eye_realstimuli', x = 'change', hue = 'eye', hue_order = ['e0_change', 'e1_change'], palette = ['blue', 'red'], ax = ax[0], order = stimorder, legend = True, dodge = False, jitter = .2, orient = 'h', alpha = 0.6)
    sns.stripplot(fish.saccade_df, y = 'eye_realstimuli', x = 'e_conv_change',  color = 'grey', ax = ax[1], order= stimorder,  orient = 'h')
    sns.stripplot(fish.saccade_df, y = 'eye_realstimuli', x = 'e_conj_change',  color = 'grey', ax = ax[2], order= stimorder,  orient = 'h')

    for axis in ax:
        axis.axvline(0, linewidth = 2, color = 'lightgrey', linestyle = '--')
        axis.grid(False)
        axis.set_ylabel('')
        axis.set_xlim([-20, 20])
        axis.spines[['top', 'right']].set_visible(False)
        for i in range(len(stimorder)):
            axis.axhline(i, linewidth = 1, alpha = 0.8, zorder = -3, color = 'lightgrey')
    ax[0].invert_xaxis()
    ax[2].invert_xaxis()
    ax[2].set_yticks([])
    ax[1].set_yticks([])
    ax[0].set_xlabel('single eye change')
    ax[1].set_xlabel('convergence')
    ax[2].set_xlabel('conjugated')
    ax[0].set_yticklabels(stimorder, rotation =0)
    #make sense of the axis
    ax[0].legend(ax[0].get_legend_handles_labels()[0], ['L', 'R'], loc='upper left', fontsize = 'xx-small',
                 markerscale = .8,  fancybox = False)
    ax[0].set_xticks([30, 0, -30])
    ax[0].set_xticklabels(['30\nL', '0', '-30\nR'])
    ax[1].set_xticks([-30, 0, 30])
    ax[1].set_xticklabels(['-30\ndiv.', '0', '30\nconv.'])
    ax[2].set_xticks([30, 0, -30])
    ax[2].set_xticklabels(['30\nL', '0', '-30\nR'])
    return fig

def plot_all_saccades(fish, stim_matrix):
    fish.stimulus_df.loc[:, 'stim_name'] = [str(i) for i in fish.stimulus_df.stim_name]
    saccade_start_eyeindex = [i[0] for i in fish.saccade_df.cont_tuples_eyeindex]

    fig, ax = plt.subplots(stim_matrix.shape[0] * 20, stim_matrix.shape[1], dpi = 240, figsize = (8, 20))#assume maximum 10 trials per stim
    for row in range(stim_matrix.shape[0]):
        for col in range(stim_matrix.shape[1]):
            stim = stim_matrix[row][col]
            axes = [ax[row * 20 + i, col] for i in range(20)]
            if stim == 'skip':
                for axis in axes:
                    axis.set_axis_off()
            else:
                stim_df = fish.stimulus_df[fish.stimulus_df.stim_name == stim]
                n = 0
                for _, stimrow in stim_df.iterrows():
                    axis = axes[n]
                    stim_starts = stimrow.real_starttime_s
                    stim_stationarys = stimrow.stationary_time
                    if type(stim_stationarys) == list:
                        stim_stationarys = min(stim_stationarys)
                    stim_durations = stimrow.duration
                    stim_ends = stim_starts + stim_durations
                    e = fish.eye_df[(fish.eye_df.real_time_s >= stim_starts) & (fish.eye_df.real_time_s <= stim_ends)]
                    if np.mean(e.th_e0) <= np.mean(e.th_e1):
                        axis.plot(e.index, np.add(e.th_e0, 40), linewidth = 0.1, c = 'skyblue')#force the smaller e0 to be on the top
                        axis.plot(e.index, e.th_e1, linewidth = 0.1, c = 'rosybrown')
                    elif np.mean(e.th_e0) > np.mean(e.th_e1):
                        axis.plot(e.index, np.add(e.th_e1, 40), linewidth=0.1, c='skyblue')  # force the smaller e0 to be on the top
                        axis.plot(e.index, e.th_e0, linewidth=0.1, c='rosybrown')
                    axis.axvspan(e.index[np.argmin(np.abs(np.subtract(e.real_time_s, stim_starts + stim_stationarys)))], e.index[np.argmin(np.abs(np.subtract(e.real_time_s, stim_starts + stim_durations)))], color = 'lightgrey', alpha = 0.2, linewidth = 0)
                    axis.set_yticks([])
                    axis.set_xticks([])
                    axis.set_ylabel('')
                    axis.set_xlabel('')
                    axis.spines[['left', 'top', 'right']].set_visible(False)
                    #look for if there is any saccades being captured here
                    instim_saccade_index = [i for i,x in enumerate(saccade_start_eyeindex) if e.index[0] <= x <= e.index[-1]]
                    saccade_df = fish.saccade_df.loc[instim_saccade_index]
                    for _, saccade_row in saccade_df.iterrows():
                        conv_mapping = np.clip(saccade_row.e_conv_change, -10, 10)
                        c = cmap_conv((conv_mapping +10)/20)
                        axis.scatter((saccade_row.cont_tuples_eyeindex[0] + saccade_row.cont_tuples_eyeindex[1])/2 , -40, s = 3, color = c)#120
                    n += 1
                if n < 10:
                    for axis in axes[n:]:
                        axis.set_axis_off()
        #add lines between different stims
        sp = fig.subplotpars
        height =sp.top - sp.bottom
        h = height/stim_matrix.shape[0]
        for row in range(1, stim_matrix.shape[0]):
            line = Line2D([sp.left, sp.right], [sp.bottom + h * row, sp.bottom + h * row], transform=fig.transFigure, color = 'dimgrey', linewidth = 1)
            fig.lines.append(line)
        width =sp.right - sp.left
        w = width/stim_matrix.shape[1]
        for col in range(1, stim_matrix.shape[1]):
            line = Line2D([sp.left + w * col, sp.left + w * col],[sp.top, sp.bottom], transform=fig.transFigure, color = 'dimgrey', linewidth = 1)
            fig.lines.append(line)
    return fig


def tail_finder(fish, time_window_s=.1, min_bout_s=0.05, min_interval_s=0.1, startframe=0, endframe=100000,
                std_scale=3):
    """
    Find the tail bout
    :param fish:
    :param time_window_s:
    :param min_bout_s:
    :param min_interval_s:
    :return:
    """
    print('2p: neg/pos = L/R\n behavior: neg/pos = R/L')
    fig, ax = plt.subplots(2, 1, dpi=240, figsize=(12, 3))

    # quick process eye
    if fish.eye_df.th_e0.mean() <= fish.eye_df.th_e1.mean():
        fish.eye_df.loc[:, "e_vergence"] = fish.eye_df.th_e1 - fish.eye_df.th_e0
    else:
        fish.eye_df.loc[:, "e_vergence"] = fish.eye_df.th_e0 - fish.eye_df.th_e1

    # plot tail trace
    tail_segment = fish.tail_df.tail_sum[startframe:endframe]
    ax[0].plot(fish.tail_df.tail_sum[startframe:endframe], linewidth=1, color='dimgrey')

    # plot tailsum
    # get running std
    fish.tail_df.tail_sum = np.subtract(fish.tail_df.tail_sum, np.nanmean(fish.tail_df.tail_sum))
    pos = np.where(fish.tail_df.tail_sum > 0, fish.tail_df.tail_sum, np.nan)
    neg = np.where(fish.tail_df.tail_sum < 0, fish.tail_df.tail_sum, np.nan)
    tail_frame_s = np.diff(fish.tail_df['real_time_s']).mean()
    frame_window = int(time_window_s / tail_frame_s)
    tail_std = fish.tail_df.tail_sum.rolling(window=frame_window).std()
    ax[1].plot(tail_segment.rolling(window=frame_window).std(), linewidth=1, color='dimgrey')
    # get boundary
    boundary = np.mean(tail_std) + std_scale * np.std(tail_std)
    ax[1].axhline(y=boundary, linewidth=1, color='black', linestyle='--')
    # get tail bouts
    bout_on = tail_std > boundary
    bout_on = [int(x) for x in bout_on]
    on_index = np.where(np.diff(bout_on) == 1)[0]
    off_index = np.where(np.diff(bout_on) == -1)[0]
    on_tuples = []
    if len(on_index) != 0 and len(off_index) != 0:
        if on_index[0] > off_index[0]:
            on_index = np.concatenate([[0], on_index])
        if on_index[-1] > off_index[-1]:
            off_index = np.concatenate([off_index, [len(fish.tail_df) - 1]])
        on_tuples = [(on, off) for on, off in zip(on_index, off_index) if off - on < len(fish.tail_df) and off > on]
    cont_on_index = []
    cont_off_index = []
    if len(on_tuples) > 0:
        cont_on_index = [on_tuples[0][0]]
        if len(on_tuples) > 1:
            big_interval = np.array([on_tuples[i][0] - on_tuples[i - 1][1] for i in range(1, len(on_tuples))]) \
                           > (min_interval_s / tail_frame_s)
            for i in range(0, len(big_interval)):
                if big_interval[i]:
                    cont_on_index = cont_on_index + [on_tuples[i + 1][0]]
                    cont_off_index = cont_off_index + [on_tuples[i][1]]
        cont_off_index = cont_off_index + [on_tuples[-1][1]]
    cont_tuples = [(on, off) for on, off in zip(cont_on_index, cont_off_index) if
                   off - on > (min_bout_s / tail_frame_s)]
    cont_tuples_realtime_s = [(fish.tail_df.real_time_s.iloc[tu[0]], fish.tail_df.real_time_s.iloc[tu[1]]) for tu in
                              cont_tuples]
    #
    tail_strength = np.full(len(cont_tuples), np.nan)
    tail_angle = np.full(len(cont_tuples), np.nan)
    tail_fromstart_s = np.full(len(cont_tuples), np.nan)
    eye_vergence = np.full(len(cont_tuples), np.nan)
    tail_angle_pos = np.full(len(cont_tuples), np.nan)
    tail_angle_posmax = np.full(len(cont_tuples), np.nan)
    tail_angle_neg = np.full(len(cont_tuples), np.nan)
    tail_angle_negmin = np.full(len(cont_tuples), np.nan)
    tail_duration_s = np.full(len(cont_tuples), np.nan)
    tail_frequency_s = np.full(len(cont_tuples), np.nan)
    tail_stimuli = np.full(len(cont_tuples), np.nan, dtype=object)
    tail_realstimuli = np.full(len(cont_tuples), np.nan, dtype=object)
    for i in range(len(cont_tuples)):
        tail_strength[i] = np.nanmean(tail_std[cont_tuples[i][0]:cont_tuples[i][1]])
        tail_angle[i] = np.nanmean(fish.tail_df.tail_sum[cont_tuples[i][0]:cont_tuples[i][1]])
        preeye_df = fish.eye_df[(fish.eye_df['real_time_s'] <= cont_tuples_realtime_s[i][0]) & (
                    fish.eye_df['real_time_s'] >= cont_tuples_realtime_s[i][0] - 0.5)]
        posteye_df = fish.eye_df[(fish.eye_df['real_time_s'] <= cont_tuples_realtime_s[i][1]) & (
                    fish.eye_df['real_time_s'] >= cont_tuples_realtime_s[i][0])]
        eye_vergence[i] = np.nanmean(posteye_df.e_vergence) - np.nanmean(preeye_df.e_vergence)  # larger = more converged
        tail_angle_pos[i] = np.nanmean(pos[cont_tuples[i][0]:cont_tuples[i][1]])
        tail_angle_posmax[i] = np.nanmax(pos[cont_tuples[i][0]:cont_tuples[i][1]])
        tail_angle_neg[i] = np.nanmean(neg[cont_tuples[i][0]:cont_tuples[i][1]])
        tail_angle_negmin[i] = np.nanmin(neg[cont_tuples[i][0]:cont_tuples[i][1]])
        tail_duration_s[i] = np.multiply((cont_tuples[i][1] - cont_tuples[i][0]), tail_frame_s)
        tail_of_interest = list(fish.tail_df.iloc[cont_tuples[i][0]:cont_tuples[i][1]].tail_sum)
        mean_line = np.mean(tail_of_interest)
        crossing = np.subtract(tail_of_interest, mean_line)
        crossing = np.sign(crossing)
        crossing = np.count_nonzero(np.diff(crossing))
        tail_frequency_s[i] = np.divide(crossing / 2, tail_duration_s[i])
        stim_delay_s = np.subtract(cont_tuples_realtime_s[i][0], fish.stimulus_df.real_starttime_s)
        stim_delay_s = [t for t in stim_delay_s if t > 0]
        stim_nearest_row = np.argmin(stim_delay_s)  # all the stimulus happening before this tail
        stim_delay_s = stim_delay_s[stim_nearest_row]
        tail_fromstart_s[i] = stim_delay_s
        stim_name = fish.stimulus_df['stim_name'].iloc[stim_nearest_row]
        stim_duration = fish.stimulus_df['duration'].iloc[stim_nearest_row]
        stim_stationarytime = fish.stimulus_df['stationary_time'].iloc[stim_nearest_row]
        tail_stimuli[i] = str(stim_name)
        if type(stim_name) == list or '[' in stim_name:  # overlap stimulus
            if stim_delay_s < min(stim_stationarytime):
                stim_name = 'stationary'
            elif stim_delay_s < max(stim_stationarytime):
                if type(stim_name) == list:
                    stim_name = stim_name[1]
                else:
                    stim_name= stim_name.split("'")[3]
        else:  # one stimulus
            if stim_delay_s < stim_stationarytime:
                stim_name = 'stationary'
        tail_realstimuli[i] = str(stim_name)

    tail_bout_df = pd.DataFrame(
        {'cont_tuples_tailindex': cont_tuples,
         'cont_tuples_realtime_s': cont_tuples_realtime_s,
         'tail_strength': tail_strength, 'tail_angle': tail_angle,
         'eye_vergence': eye_vergence,
         'tail_angle_pos': tail_angle_pos,
         'tail_angle_posmax': tail_angle_posmax, 'tail_angle_neg': tail_angle_neg, 'tail_angle_negmin': tail_angle_negmin,
         'tail_duration_s': tail_duration_s,  'tail_frequency_s': tail_frequency_s,
         'tail_stimuli': tail_stimuli, 'tail_realstimuli': tail_realstimuli, 'tail_fromstart_s': tail_fromstart_s
         })
    # plot bouts
    for tu, s in zip(cont_tuples, tail_realstimuli):
        if tu[0] >= startframe and tu[1] <= endframe:
            ax[0].axvspan(tu[0], tu[1], color='pink', alpha=0.8)

    fish.bout_df = tail_bout_df

    # make axis prettier
    for axis in ax:
        axis.grid(False)
        axis.set_xticks([])
        axis.spines[['top', 'right']].set_visible(False)
    ax[1].set_ylim([0, 1])
    return fig


def plot_boutangle_distribution(fish, stim_lists, s = 50, group = False, tail_bout_df = None,
                                stimulus_df = None):
    def fix_ax(ax, title):
        """
        Make axis look prettier
            ax: the axes to plot on
        """
        ax.set_theta_offset(np.deg2rad(-90))
        ax.set_theta_direction('counterclockwise')
        ax.grid(linewidth=0.5, linestyle=":", c='grey')
        ax.set_yscale('log')
        ax.set_rlim([0.01, 1])
        ax.set_yticks([0.01, 0.1, 0.5, 1], labels=['0', '.1', '.5', '1'])
        ax.tick_params(axis='y', which='major', pad=-2)
        ax.set_xlim([np.deg2rad(180), np.deg2rad(-180)])
        ax.spines[['start', 'end']].set_color("black")
        ax.spines['polar'].set_linestyle((0, (2, 5)))
        ax.spines['polar'].set_linewidth(0.8)
        ax.spines['polar'].set_color("black")
        ax.set_xticks([-np.pi / 4, 0, np.pi / 4], labels=[])
        ax.set_xlabel('duration (s)', labelpad=-105)
        ax.set_title(title, weight='bold')

    fig, ax = plt.subplots(2, len(stim_lists), dpi=600, figsize=(8, 4), subplot_kw={'projection': 'polar'},
                           gridspec_kw={'wspace': 0})
    if not group:
        stimulus_df = fish.stimulus_df.copy()
        tail_bout_df = fish.bout_df.copy()
    stimulus_df.loc[:, 'stim_name'] = [str(i) for i in stimulus_df.stim_name]
    tail_bout_df.loc[:, 'tail_realstimuli'] = [str(i) for i in tail_bout_df.tail_realstimuli]
    tail_bout_df.loc[:, 'eye_vergence_norm'] = np.divide(
        np.subtract(tail_bout_df.nearest_saccade_conv, np.nanpercentile(tail_bout_df.nearest_saccade_conv, 50)),
        (np.nanpercentile(tail_bout_df.nearest_saccade_conv, 95) - np.nanpercentile(tail_bout_df.nearest_saccade_conv,                                                                             50)))
    tail_bout_df.loc[:, 'eye_vergence_norm'] = np.clip(
        [i if not np.isnan(i) else 0 for i in tail_bout_df.eye_vergence_norm], 0, 1)
    tail_bout_df.loc[:, 'eye_vergence_norm'] = [i if i != np.nan else 0 for i in tail_bout_df.eye_vergence_norm]
    for n, stimlist in enumerate(stim_lists):
        if '[' not in stimlist[0] and 'dot' not in stimlist[0]:#grating
            ax[1, n].axis('off')
            for stim in stimlist:
                stim_tail_bout_df = tail_bout_df[tail_bout_df['tail_realstimuli'] == stim]
                color = [i for i in utils.stim_colors[stim]]
                if len(stim_tail_bout_df) > 0:
                    ax[0, n].scatter(stim_tail_bout_df.tail_angle, stim_tail_bout_df.tail_duration_s,
                                     edgecolor=None,  facecolor=color, linewidth = 0, s=s, marker='.', alpha = 1, zorder = 0)#alpha=stim_tail_bout_df.eye_vergence_norm, zorder = 0)
                    # ax[0, n].scatter(np.median(stim_tail_bout_df.tail_angle), np.median(stim_tail_bout_df.tail_duration_s),
                    #               edgecolor=color, facecolor=color, s=s, zorder = 1)
            fix_ax(ax[0, n], 'gratings')
        elif '[' not in stimlist[0] and 'dot' in stimlist[0]:#dot
            for stim in stimlist:
                #color = [i for i in utils.stim_colors[stim]]
                if 'dot_l' in stim:
                    color = 'slateblue'
                elif 'dot_r' in stim:
                    color = 'mediumvioletred'
                stim_tail_bout_df = tail_bout_df[tail_bout_df['tail_realstimuli'] == stim]
                stim_df = stimulus_df[stimulus_df.stim_name == stim].iloc[0]
                _, _, _, center_time = utils.get_dot_pos(stim_df)
                first_half_bout_df = stim_tail_bout_df[stim_tail_bout_df.tail_fromstart_s <= center_time+0.5]
                second_half_bout_df = stim_tail_bout_df[stim_tail_bout_df.tail_fromstart_s > center_time+0.5]
                if len(first_half_bout_df) > 0:
                    alpha = first_half_bout_df.eye_vergence_norm#np.subtract(1, np.divide(np.subtract(center_time, first_half_bout_df.tail_fromstart_s), 3))
                    alpha = np.clip(alpha, 0, 1)
                    ax[0, n].scatter(first_half_bout_df.tail_angle, first_half_bout_df.tail_duration_s,
                                  edgecolor=None,  facecolor=color, linewidth = 0, s=s, marker='.', alpha = alpha, zorder = 0)
                    # ax[0, n].scatter(np.median(first_half_bout_df.tail_angle), np.median(first_half_bout_df.tail_duration_s),
                    #               edgecolor=color, facecolor=color, s=s, zorder = 1)
                if len(second_half_bout_df) > 0:
                    alpha = second_half_bout_df.eye_vergence_norm#np.subtract(1, np.divide(np.subtract(second_half_bout_df.tail_fromstart_s, center_time), 3))
                    alpha = np.clip(alpha, 0, 1)
                    ax[1, n].scatter(second_half_bout_df.tail_angle, second_half_bout_df.tail_duration_s,
                                  edgecolor=None,  facecolor=color,linewidth = 0, s=s, marker='.',  alpha=alpha, zorder = 0)
                    # ax[1, n].scatter(np.median(second_half_bout_df.tail_angle), np.median(second_half_bout_df.tail_duration_s),
                    #               edgecolor=color, facecolor=color, s=s, zorder = 1)
            fix_ax(ax[0, n], 'dot')
            fix_ax(ax[1, n], 'dot')
        else: #overlap
            for stim in stimlist:
                stim_tail_bout_df = tail_bout_df[tail_bout_df['tail_realstimuli'] == stim]
                color = [i for i in utils.stim_colors[stim.split("'")[1]]]
                stim_df = stimulus_df[stimulus_df.stim_name == stim].iloc[0]
                _, _, _, center_time = utils.get_dot_pos(stim_df)
                first_half_bout_df = stim_tail_bout_df[stim_tail_bout_df.tail_fromstart_s <= center_time]
                second_half_bout_df = stim_tail_bout_df[stim_tail_bout_df.tail_fromstart_s > center_time]
                if len(first_half_bout_df) > 0:
                    alpha = first_half_bout_df.eye_vergence_norm#np.subtract(1, np.divide(np.subtract(center_time, first_half_bout_df.tail_fromstart_s), 3))
                    alpha = np.clip(alpha, 0, 1)
                    ax[0, n].scatter(first_half_bout_df.tail_angle, first_half_bout_df.tail_duration_s,
                                     edgecolor=None, facecolor=color,linewidth = 0, s=s, marker='.', alpha=alpha, zorder=0)
                    # ax[0, n].scatter(np.median(first_half_bout_df.tail_angle),
                    #                   np.median(first_half_bout_df.tail_duration_s),
                    #                   edgecolor=color, facecolor=color, s=s, zorder=1)
                if len(second_half_bout_df) > 0:
                    alpha =  second_half_bout_df.eye_vergence_norm#np.subtract(1, np.divide(np.subtract(second_half_bout_df.tail_fromstart_s, center_time), 3))
                    alpha = np.clip(alpha, 0, 1)
                    ax[1, n].scatter(second_half_bout_df.tail_angle, second_half_bout_df.tail_duration_s,
                                     edgecolor=None,  facecolor=color,linewidth = 0, s=s, marker='.', alpha=alpha, zorder=0)
                    # ax[1, n].scatter(np.median(second_half_bout_df.tail_angle),
                    #                   np.median(second_half_bout_df.tail_duration_s),
                    #                   edgecolor=color, facecolor=color, s=s, zorder=1)
            fix_ax(ax[0, n], 'gratings + dot')
            fix_ax(ax[1, n], 'gratings + dot')
    return fig


def plot_bouttime_distribution(fish, stim_lists, s = 50, group = False, tail_bout_df = None,
                                stimulus_df = None, classified = False):
    color_dict = {'L': utils.green, 'R': utils.green, 'H': utils.pink}
    def fix_dotax(ax, title):
        """
        Make axis look prettier
            ax: the axes to plot on
        """
        ax.axhline(0, linewidth=1, color='grey', linestyle=':')
        ax.set_ylim([-180, 180])
        ax.set_yticks([-180, 0, 180])
        ax.set_ylabel('dot angle(°)')
        ax.set_yticklabels(['L', '0', 'R'])
        ax.set_xlim([0, 35])
        ax.grid(False)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlabel('time (s)')
        ax.set_title(title, weight='bold')

    def fix_ax(ax, title):
        """
        Make axis look prettier
            ax: the axes to plot on
        """
        ax.axhline(0, linewidth=1, color='grey', linestyle=':')
        ax.set_xlim([0, 35])
        ax.set_ylim([-1.5, 1.5])
        ax.set_yticks([-2, 0, 2])
        ax.set_yticklabels(['L', '0', 'R'])
        ax.grid(False)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('mean bout angle(°)')
        ax.set_title(title, weight='bold')

    fig, ax = plt.subplots(2, len(stim_lists), dpi=600, figsize=(10, 5), gridspec_kw={'wspace': 0.5})
    if not group:
        stimulus_df = fish.stimulus_df.copy()
        tail_bout_df = fish.bout_df.copy()
    stimulus_df.loc[:, 'stim_name'] = [str(i) for i in stimulus_df.stim_name]
    tail_bout_df.loc[:, 'tail_realstimuli'] = [str(i) for i in tail_bout_df.tail_realstimuli]
    tail_bout_df.loc[:, 'eye_vergence_norm'] = np.divide(np.subtract(tail_bout_df.nearest_saccade_conv, np.nanpercentile(tail_bout_df.nearest_saccade_conv, 50)),
                                                 (np.nanpercentile(tail_bout_df.nearest_saccade_conv, 95) - np.nanpercentile(tail_bout_df.nearest_saccade_conv, 50)))
    tail_bout_df.loc[:, 'eye_vergence_norm'] = np.clip([i if not np.isnan(i) else 0 for i in tail_bout_df.eye_vergence_norm], 0, 1)
    for n, stimlist in enumerate(stim_lists):
        ax_dot = ax[0, n].twinx()
        if '[' not in stimlist[0]:  # gratings or dot
            for stim in stimlist:
                stim_tail_bout_df = tail_bout_df[tail_bout_df['tail_stimuli'] == stim]
                if classified:
                    color = [color_dict[s] for s in stim_tail_bout_df.bout_type]
                    ax[0, n].scatter(stim_tail_bout_df.tail_fromstart_s, stim_tail_bout_df.tail_angle,
                                     color=color, s=s, marker='.', linewidth=0, alpha=stim_tail_bout_df.eye_vergence_norm, zorder=1)
                else:
                    color = ['slateblue' if 'dot_l' in s else 'mediumvioletred' if 'dot_r' in s else utils.stim_colors[s.split('_')[0]] for s in stim_tail_bout_df.tail_realstimuli]
                    if len(stim_tail_bout_df) > 0:
                        ax[0, n].scatter(stim_tail_bout_df.tail_fromstart_s, stim_tail_bout_df.tail_angle,
                                  color=color, s=s, marker='.', linewidth=0, alpha=stim_tail_bout_df.eye_vergence_norm, zorder=1)
                if 'dot' in stim:#put dot trajectory
                    _, ts, degs, center_time = utils.get_dot_pos(stimulus_df[stimulus_df.stim_name == stim].iloc[0])
                    if 'dot_l' in stim:
                        stim_color = 'slateblue'
                    elif 'dot_r' in stim:
                        stim_color = 'mediumvioletred'
                    ax_dot.scatter(ts, degs, s=.1, c=stim_color, alpha=0.8)
                    ax[0, n].axvline(center_time, linestyle='--', color='lightgrey', zorder = -2)
                    ax[1, n].axvline(center_time, linestyle='--', color='lightgrey', zorder = -2)
            stim_start_time = stimulus_df[stimulus_df.stim_name == stim].iloc[0]['stationary_time']
            stim_durations = stimulus_df[stimulus_df.stim_name == stim].iloc[0]['duration']
            #ax[n].axvspan(stim_start_time, stim_durations, linewidth=1, color='lightgrey', alpha=0.8, linestyle=':', zorder=-1)
            if 'dot' not in stimlist[0]:
                fix_ax(ax[0, n], 'gratings')
            else:
                fix_ax(ax[0, n], 'dot')
                fix_dotax(ax_dot, 'dot')
        else:#overlap dot
            for stim in stimlist:
                stim_tail_bout_df = tail_bout_df[tail_bout_df['tail_stimuli'] == stim]
                if len(stim_tail_bout_df) >= 1:
                    if classified:
                        color = [color_dict[s] for s in stim_tail_bout_df.bout_type]
                        ax[0, n].scatter(stim_tail_bout_df.tail_fromstart_s, stim_tail_bout_df.tail_angle,
                                         color=color, s=s, marker='.', linewidth=0, alpha=stim_tail_bout_df.eye_vergence_norm, zorder=2)
                    else:
                        color = [utils.stim_colors[s] if '[' not in s else utils.stim_colors[s.split("'")[3]] for s in
                                 stim_tail_bout_df.tail_realstimuli]
                        ax[0, n].scatter(stim_tail_bout_df.tail_fromstart_s, stim_tail_bout_df.tail_angle,
                                  color=color, s=s, marker='.', linewidth=0, alpha=stim_tail_bout_df.eye_vergence_norm, zorder=2)
                if 'dot' in stim:#put dot trajectory
                    if 'dot_l' in stim:
                        stim_color = 'slateblue'
                    elif 'dot_r' in stim:
                        stim_color = 'mediumvioletred'
                    _, ts, degs, center_time = utils.get_dot_pos(stimulus_df[stimulus_df.stim_name == stim].iloc[0])
                    ax_dot.scatter(ts, degs, s=.1, c=stim_color, alpha=0.8, zorder = 0)
                    ax[0, n].axvline(center_time, linestyle='--', color='lightgrey', zorder = 1)
                    ax[1, n].axvline(center_time, linestyle='--', color='lightgrey', zorder = 1)
            stim_start_time = stimulus_df[stimulus_df.stim_name == stim].iloc[0]['stationary_time']
            stim_durations = stimulus_df[stimulus_df.stim_name == stim].iloc[0]['duration']
            #ax[n].axvspan(stim_start_time[0], stim_durations, linewidth=1, color='lightgrey', alpha=0.8, linestyle=':', zorder=-1)
            #ax[n].axvspan(stim_start_time[1], stim_durations, linewidth=1, color='lightgrey', alpha=0.8, linestyle=':', zorder=-1)
            fix_ax(ax[0, n], 'gratings + dot')
            fix_dotax(ax_dot, 'gratings + dot')
        # plot frequency
        dotl_stims = [s for s in stimlist if 'dot_l' in s]
        dotr_stims = [s for s in stimlist if 'dot_r' in s]
        other_stims = [s for s in stimlist if 'dot' not in s]
        for stims in [dotl_stims, dotr_stims, other_stims]:
            if len(stims) > 0:
                bin_s = 1
                row_bout_df = tail_bout_df[tail_bout_df['tail_stimuli'].isin(stims)]
                if classified:
                    nh_bout_df = row_bout_df[row_bout_df.bout_type != 'H']
                    if len(nh_bout_df) > 0:
                        freqs = []
                        for fish in nh_bout_df.real_fish_id.unique():
                            count, bin_loc = np.histogram(nh_bout_df[nh_bout_df.real_fish_id == fish].tail_fromstart_s,
                                                          bins=int(35 // bin_s), range=[0, 35])
                            bin_center = (bin_loc[:-1] + bin_loc[1:]) / 2
                            freq = np.divide(count, bin_s)
                            freqs.append(freq)
                        ax[1, n].plot(bin_center, np.nanmean(np.array(freqs), axis=0), color=utils.green,  zorder=1)
                    h_bout_df = row_bout_df[row_bout_df.bout_type == 'H']
                    if len(h_bout_df) > 0:
                        freqs = []
                        for fish in h_bout_df.real_fish_id.unique():
                            count, bin_loc = np.histogram(h_bout_df[h_bout_df.real_fish_id == fish].tail_fromstart_s,
                                                          bins=int(35 // bin_s), range=[0, 35])
                            bin_center = (bin_loc[:-1] + bin_loc[1:]) / 2
                            freq = np.divide(count, bin_s)
                            freqs.append(freq)
                        ax[1, n].plot(bin_center, np.nanmean(np.array(freqs), axis=0), color=utils.pink, zorder=1)
                else:
                    l_bout_df = row_bout_df[row_bout_df.tail_angle < 0]
                    if len(l_bout_df) > 0:
                        count, bin_loc = np.histogram(l_bout_df.tail_fromstart_s, bins=int(35//bin_s), range=[0, 35])
                        bin_center = (bin_loc[:-1] + bin_loc[1:]) / 2
                        freq = np.divide(count, bin_s)
                        ax[1, n].plot(bin_center, freq, color = 'navy', zorder = 1)
                    r_bout_df = row_bout_df[row_bout_df.tail_angle > 0]
                    if len(r_bout_df) > 0:
                        count, bin_loc = np.histogram(r_bout_df.tail_fromstart_s, bins=int(35//bin_s), range=[0, 35])
                        freq = np.divide(count, bin_s)
                        bin_center = (bin_loc[:-1] + bin_loc[1:]) / 2
                        ax[1, n].plot(bin_center, freq, color= 'maroon', zorder=1)
            ax[1, n].set_ylim([0, 10])
            ax[1, n].set_xlim([0, 35])
            ax[1, n].set_ylabel('freq')
    return fig


def plot_all_tails(fish, stim_matrix):
    fish.stimulus_df.loc[:, 'stim_name'] = [str(i) for i in fish.stimulus_df.stim_name]
    bout_start_tailindex = [i[0] for i in fish.bout_df.cont_tuples_tailindex]

    fig, ax = plt.subplots(stim_matrix.shape[0] * 20, stim_matrix.shape[1], dpi = 240, figsize = (8, 20))#assume maximum 10 trials per stim
    for row in range(stim_matrix.shape[0]):
        for col in range(stim_matrix.shape[1]):
            stim = stim_matrix[row][col]
            axes = [ax[row * 20 + i, col] for i in range(20)]
            if stim == 'skip':
                for axis in axes:
                    axis.set_axis_off()
            else:
                stim_df = fish.stimulus_df[fish.stimulus_df.stim_name == stim]
                n = 0
                for _, stimrow in stim_df.iterrows():
                    axis = axes[n]
                    stim_starts = stimrow.real_starttime_s
                    stim_durations = stimrow.duration
                    stim_stationary_s = stimrow.stationary_time
                    stim_ends = stim_starts + stim_durations
                    t = fish.tail_df[(fish.tail_df.real_time_s >= stim_starts) & (fish.tail_df.real_time_s <= stim_ends)]
                    axis.plot(t.index, t.tail_sum, linewidth = 0.1, c = 'black', zorder = 0)
                    if '[' in stim and 'stationary' not in stim:
                        axis.axvspan(t.index[np.argmin(np.abs(np.subtract(t.real_time_s, stim_starts + stim_stationary_s[0])))], t.index[np.argmin(np.abs(np.subtract(t.real_time_s, stim_starts + stim_durations)))], color = 'lightgrey', alpha = 0.2, linewidth = 0)
                        axis.axvspan(t.index[np.argmin(np.abs(np.subtract(t.real_time_s, stim_starts + stim_stationary_s[1])))], t.index[np.argmin(np.abs(np.subtract(t.real_time_s, stim_starts + stim_durations)))], color = 'lightgrey', alpha = 0.2, linewidth = 0)
                    elif 'dot' in stim:
                        axis.axvspan(t.index[np.argmin(np.abs(np.subtract(t.real_time_s, stim_starts + stim_stationary_s)))], t.index[np.argmin(np.abs(np.subtract(t.real_time_s, stim_starts + stim_durations)))], color = 'lightgrey', alpha = 0.2, linewidth = 0)
                    else:
                        axis.axvspan(t.index[np.argmin(np.abs(np.subtract(t.real_time_s, stim_starts + stim_stationary_s)))], t.index[np.argmin(np.abs(np.subtract(t.real_time_s, stim_starts + stim_durations)))], color = 'lightgrey', alpha = 0.2, linewidth = 0)
                    axis.set_yticks([])
                    axis.set_xticks([])
                    axis.set_ylabel('')
                    axis.set_xlabel('')
                    axis.spines[['left', 'top', 'right']].set_visible(False)
                    #look for if there is any saccades being captured here
                    instim_bout_index = [i for i,x in enumerate(bout_start_tailindex) if t.index[0] <= x <= t.index[-1]]
                    bout_df = fish.bout_df.loc[instim_bout_index]
                    for _, bout_row in bout_df.iterrows():
                        axis.scatter((bout_row.cont_tuples_tailindex[0] + bout_row.cont_tuples_tailindex[1])/2 , 0, s = 3, c = bout_row.tail_angle, cmap = 'bwr', vmin = -1, vmax = 1, zorder = 1)#120
                    n += 1
                if n < 20:
                    for axis in axes[n:]:
                        axis.set_axis_off()
        #add lines between different stims
        sp = fig.subplotpars
        height =sp.top - sp.bottom
        h = height/stim_matrix.shape[0]
        for row in range(1, stim_matrix.shape[0]):
            line = Line2D([sp.left, sp.right], [sp.bottom + h * row, sp.bottom + h * row], transform=fig.transFigure, color = 'dimgrey', linewidth = 1)
            fig.lines.append(line)
        width =sp.right - sp.left
        w = width/stim_matrix.shape[1]
        for col in range(1, stim_matrix.shape[1]):
            line = Line2D([sp.left + w * col, sp.left + w * col],[sp.top, sp.bottom], transform=fig.transFigure, color = 'dimgrey', linewidth = 1)
            fig.lines.append(line)
    return fig


def plot_simple_nearestsaccade(tail_bout_df, stim_list, color, stimulus_df, classified = False):
    rng = np.random.default_rng()
    tail_bout_df.loc[:, 'eye_vergence_norm'] = np.divide(
            np.subtract(tail_bout_df.nearest_saccade_conv, np.nanpercentile(tail_bout_df.nearest_saccade_conv,50)),
            (np.nanpercentile(tail_bout_df.nearest_saccade_conv, 95) - np.nanpercentile(tail_bout_df.nearest_saccade_conv,50)))
    tail_bout_df.loc[:, 'eye_vergence_norm'] = np.clip([i if not np.isnan(i) else 0 for i in tail_bout_df.eye_vergence_norm], 0, 1)
    fig, ax = plt.subplots(len(stim_list), 1, figsize = (6, 1), dpi = 240)
    for n, stim in enumerate(stim_list):
        stim_df = stimulus_df[stimulus_df.stim_name == stim].iloc[0]
        if 'dot' in stim:
            _, _, _, center_time = utils.get_dot_pos(stim_df)
            ax[n].axvline(center_time, color = 'lightgrey', linestyle = '--', zorder = -1)

        stim_bout_df = tail_bout_df[tail_bout_df.tail_stimuli == stim]
        if classified:
            color = [utils.pink if s == 'H' else utils.green for s in stim_bout_df.bout_type]
            ax[n].scatter(x = stim_bout_df.tail_fromstart_s, y = rng.uniform(low=0, high=5, size=len(stim_bout_df)), color = color, alpha =stim_bout_df.eye_vergence_norm, s = 2, linewidth = 0, zorder =1)
        else:
            ax[n].scatter(x = stim_bout_df.tail_fromstart_s, y = rng.uniform(low=0, high=5, size=len(stim_bout_df)), color = color, alpha = stim_bout_df.eye_vergence_norm, s = 2, linewidth = 0, zorder =1)
        ax[n].set_ylabel(stim, rotation = 0)

    for axis in ax:
        axis.set_ylim(-0.5, 5.5)
        axis.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlim([0, 35])


def plot_simple_nearestsaccadecombine(tail_bout_df, stim_lists, stimulus_df, classified = False):
    rng = np.random.default_rng()
    tail_bout_df.loc[:, 'eye_vergence_norm'] = np.divide(
            np.subtract(tail_bout_df.nearest_saccade_conv, np.nanpercentile(tail_bout_df.nearest_saccade_conv,50)),
            (np.nanpercentile(tail_bout_df.nearest_saccade_conv, 95) - np.nanpercentile(tail_bout_df.nearest_saccade_conv,50)))
    tail_bout_df.loc[:, 'eye_vergence_norm'] = np.clip([i if not np.isnan(i) else 0 for i in tail_bout_df.eye_vergence_norm], 0, 1)
    fig, ax = plt.subplots(len(stim_lists), 1, figsize = (6, 1), dpi = 240)
    for n, stimrow in enumerate(stim_lists):
        for stim in stimrow:
            stim_df = stimulus_df[stimulus_df.stim_name == stim].iloc[0]
            ax[n].set_ylabel(stim, rotation = 0)
        stim_bout_df = tail_bout_df[tail_bout_df.tail_stimuli.isin(stimrow)]
        if classified:
            colors = [utils.pink if s == 'H' else utils.green for s in stim_bout_df.bout_type]
            ax[n].scatter(x=stim_bout_df.tail_fromstart_s, y=rng.uniform(low=0, high=5, size=len(stim_bout_df)),
                          color=colors, alpha= stim_bout_df.eye_vergence_norm, s=2, linewidth=0, zorder=3)#we can also adjust alpha here
        else:
            colors = [utils.stim_colors['dot_l'] if 'dot_l' in s else utils.stim_colors['dot_r'] if 'dot_r' in s else 'grey' for s in stim_bout_df.tail_stimuli]
            ax[n].scatter(x = stim_bout_df.tail_fromstart_s, y = rng.uniform(low=0, high=5, size=len(stim_bout_df)), color = colors, alpha = stim_bout_df.eye_vergence_norm, s = 3, linewidth = 0, zorder =3)

        if 'dot' in stim:
            _, _, _, center_time = utils.get_dot_pos(stim_df)
            ax[n].axvline(center_time, color = 'lightgrey', linestyle = '--', zorder = 0)

    for axis in ax:
        axis.set_ylim(-0.5, 5.5)
        axis.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlim([0, 35])


def plot_simplesaccade(eye_saccade_df, stim_list, color, stimulus_df, classified = False):
    rng = np.random.default_rng()
    eye_saccade_df.loc[:, 'eye_vergence_norm'] = np.divide(
            np.subtract(eye_saccade_df.e_conv_change, np.nanpercentile(eye_saccade_df.e_conv_change,50)),
            (np.nanpercentile(eye_saccade_df.e_conv_change, 95) - np.nanpercentile(eye_saccade_df.e_conv_change,50)))
    eye_saccade_df.loc[:, 'eye_vergence_norm'] = np.clip([i if not np.isnan(i) else 0 for i in eye_saccade_df.eye_vergence_norm], 0, 1)
    fig, ax = plt.subplots(len(stim_list), 1, figsize = (4, 1), dpi = 240)
    for n, stim in enumerate(stim_list):
        stim_df = stimulus_df[stimulus_df.stim_name == stim].iloc[0]
        if 'dot' in stim:
            _, _, _, center_time = utils.get_dot_pos(stim_df)
            ax[n].axvline(center_time, color = 'lightgrey', linestyle = '--', zorder = -1)

        stim_saccade_df = eye_saccade_df[eye_saccade_df.eye_stimuli == stim]
        if classified:
            colors = [utils.pink if s == 'H' else utils.green for s in stim_saccade_df.saccade_type]
            ax[n].scatter(x = stim_saccade_df.eye_fromstart_s, y = rng.uniform(low=1, high=4, size=len(stim_saccade_df)), color = colors, alpha = stim_saccade_df.eye_vergence_norm, s = 1, linewidth = 0, zorder =1)
        else:
            ax[n].scatter(x = stim_saccade_df.eye_fromstart_s, y = rng.uniform(low=1, high=4, size=len(stim_saccade_df)), color = color, alpha = stim_saccade_df.eye_vergence_norm, s = 1, linewidth = 0, zorder =1)
        ax[n].set_ylabel(stim, rotation = 0)

    for axis in ax:
        axis.set_ylim(-0.5, 5.5)
        axis.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlim([0, 35])


def plot_simpleeyecombine(eye_saccade_df, stim_lists, stimulus_df, classified = False):
    rng = np.random.default_rng()
    eye_saccade_df.loc[:, 'eye_vergence_norm'] = np.divide(
            np.subtract(eye_saccade_df.e_conv_change, np.nanpercentile(eye_saccade_df.e_conv_change,50)),
            (np.nanpercentile(eye_saccade_df.e_conv_change, 95) - np.nanpercentile(eye_saccade_df.e_conv_change,50)))
    eye_saccade_df.loc[:, 'eye_vergence_norm'] = np.clip([i if not np.isnan(i) else 0 for i in eye_saccade_df.eye_vergence_norm], 0, 1)
    fig, ax = plt.subplots(len(stim_lists), 1, figsize = (4, 1), dpi = 240)
    for n, stimrow in enumerate(stim_lists):
        for stim in stimrow:
            stim_df = stimulus_df[stimulus_df.stim_name == stim].iloc[0]
            ax[n].set_ylabel(stim, rotation = 0)
        stim_saccade_df = eye_saccade_df[eye_saccade_df.eye_stimuli.isin(stimrow)]
        if classified:
            colors = [utils.pink if s == 'H' else utils.green for s in stim_saccade_df.saccade_type]
            ax[n].scatter(x=stim_saccade_df.eye_fromstart_s, y=rng.uniform(low=1, high=4, size=len(stim_saccade_df)),
                          color=colors, alpha=stim_saccade_df.eye_vergence_norm, s=1, linewidth=0, zorder=3)
        else:
            colors = [utils.stim_colors['dot_l'] if 'dot_l' in s else utils.stim_colors['dot_r'] if 'dot_r' in s else 'grey' for s in stim_saccade_df.eye_stimuli]
            ax[n].scatter(x = stim_saccade_df.eye_fromstart_s, y = rng.uniform(low=1, high=4, size=len(stim_saccade_df)), color = colors, alpha = stim_saccade_df.eye_vergence_norm, s = 1, linewidth = 0, zorder =3)

        if 'dot' in stim:
            _, _, _, center_time = utils.get_dot_pos(stim_df)
            ax[n].axvline(center_time, color = 'lightgrey', linestyle = '--', zorder = 0)

    for axis in ax:
        axis.set_ylim(-0.5, 5.5)
        axis.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlim([0, 35])