# from nptdms import TdmsFile
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pandas as pd
import pathlib as Path

# local imports
import sys
sys.path.append(r'C:\Users\Kaitlyn\PyCharmProjects\imaging\caImageAnalysis')
from utilities import arrutils
import constants, stimuli

### creating the tail df ###
def bhvr_log_to_df(bhvr_log_path, metadata_log_path):
    '''
    turning stytra behavior log into a pandas dataframe with datetime stamps
    bhvr_log_path: path to the behavior log
    metadata_log_path: path to the metadata log
    '''
    
    bhvr_df = pd.read_csv(bhvr_log_path, sep=";", dtype=np.float32)
    bhvr_df.drop(bhvr_df.columns[0], axis=1, inplace = True)

    with open(metadata_log_path, 'r') as json_file:
        metadata = json.load(json_file)
        start_dt_str = metadata['general']['t_protocol_start'].split('T')[1]

    start_dt = datetime.strptime(start_dt_str, "%H:%M:%S.%f")
    tail_dts = []
    for seconds in bhvr_df.t.values:
        microseconds = int(seconds * 1e6)
        new_datetime = start_dt + timedelta(microseconds=microseconds)
        tail_dts.append(new_datetime.time())
    bhvr_df['t_dt'] = tail_dts
    
    return bhvr_df

def tail_df_creator(bhvr_data_folder, saving = True):
    '''
    Creating the tail data dataframe from either single or multiple experiments in the same folder
    bhvr_data_folder: path to the folder with the behavior data
    saving: whether to save the dataframe or not, will be saved in the parents folder path
    '''
    from utilities import pathutils

    behavior_log_paths = pathutils.pathcrawler(bhvr_data_folder, set(), [], mykey = 'behavior_log')
    metadata_paths = pathutils.pathcrawler(bhvr_data_folder, set(), [], mykey = 'metadata')
            
    bhvr_df_lst = []
    for r in range(len(behavior_log_paths)):
        bhvr_df = bhvr_log_to_df(behavior_log_paths[r], metadata_paths[r])
        bhvr_df_lst.append(bhvr_df)

    if len(bhvr_df_lst) > 1:
        all_tail_data = pd.concat(bhvr_df_lst)
        all_tail_data.reset_index(drop = True, inplace = True)

        # resetting the t values
        transition_idx = []
        for i in range(1, len(all_tail_data.t.values)):
            if all_tail_data.t.values[i - 1] > all_tail_data.t.values[i]: # compare current value to previous value
                transition_idx.append(i)

        new_t_arr = all_tail_data.t.values[:].copy()

        for e, idx in enumerate(transition_idx):
            previous_value = all_tail_data.t.values[idx - 1] # seconds
            pause_time = dateToMillisec(all_tail_data.t_dt.values[idx+1]) - dateToMillisec(all_tail_data.t_dt.values[idx])  # getting the pause between experiments (in milliseconds)
            if idx == transition_idx[-1]:
                next_idx = len(all_tail_data.t.values)
            else:
                next_idx = transition_idx[e+1]
            for i in range(idx, len(all_tail_data.t.values[idx:next_idx])):
                new_t_arr[i] += previous_value + (pause_time/1000) # pause_time in milliseconds, but adding to a seconds value

        all_tail_data.drop('t', axis = 1, inplace = True) 
        all_tail_data['t'] = new_t_arr
    else:
        all_tail_data = bhvr_df
    
    if saving:
        all_tail_data.to_hdf(bhvr_data_folder.joinpath('tail_df.h5'), key='tail')
        print('saved tail dataframe')

### tail data with pstim data ###
def tag_pstim_df_with_tail_frames(tail_df, pstim_df):
    '''
    Add in 'tail frame' column to the stimulus df (helpful when you want to pair WITHOUT imaging data)
    this is the same as the 'tag frames' function in fishy by specific for pstim df and a tail df
    :param tail_df: typical tail df, all of the data from the bhvr log, need to have a 't_dt' col with datetime
    :param pstim_df: typical pandastim stimulus df, need to have a 'time' col with datetime
    :return: new pstim_df with the matching 'tail_frame' in a separate column
    '''

    if tail_df.t_dt.values[0] > tail_df.t_dt.values[-1]:  # overnight
        # pre-midnight tail_df and pstim_df
        tail_df_premidnight = tail_df[tail_df['t_dt'] > tail_df['t_dt'].iloc[-1]]
        pstim_df_premidnight = pstim_df[pstim_df['time'] > pstim_df['time'].values[-1]]
        frame_matches_premidnight = [
         tail_df_premidnight[tail_df_premidnight['t_dt'] < pstim_df_premidnight['time'].values[-1]].index[i] for i
         in
         range(len(pstim_df_premidnight))]
        # post-midnight frametimes_df and df
        tail_df_postmidnight = tail_df[tail_df['t_dt'] < tail_df['t_dt'].values[0]]
        pstim_df_postmidnight = pstim_df[pstim_df['time'] < pstim_df['time'].values[0]]
        # use a for loop to deal with if the first frame of post-midnight df is actually earlier than first frame of
        # post-midnight frametimes_df, but still after midnight.
        frame_matches_postmidnight = []
        for i in range(len(pstim_df_postmidnight)):
         smaller_df = tail_df_postmidnight[tail_df_postmidnight['t_dt'] < pstim_df_postmidnight['time'].values[i]]
         if smaller_df.empty:
             frame_matches_postmidnight = frame_matches_postmidnight + [frame_matches_premidnight[-1]]
         else:
             frame_matches_postmidnight = frame_matches_postmidnight + [smaller_df.index[-1]]
        frame_matches = frame_matches_premidnight + frame_matches_postmidnight
    else:
        frame_matches = [tail_df[tail_df.t_dt < pstim_df.time.values[i]].index[-1] for i in range(len(pstim_df))]

    pstim_df.loc[:, "tail_frame"] = frame_matches
    pstim_df = pstim_df[pstim_df['tail_frame'] != 0]
    pstim_df.reset_index(inplace=True, drop=True)

    pstim_df = stimuli.add_repetitions_to_stimulus_df(pstim_df)

    return pstim_df

### helpful utils ###
def dateToMillisec(datetime):
    '''
    Changes datetime object into milliseconds
    datetime: datetime object
    returns milliseconds
    '''
    return (
        datetime.microsecond / 1000
        + datetime.second * 1000
        + datetime.minute * 60 * 1000
        + datetime.hour * 60 * 60 * 1000
    )

def find_tail_sum_std(tail_df):
    tail_hz = 1/np.mean(np.diff(tail_df.iloc[:100]['t']))
    smooth_tailframe = int(0.05 * tail_hz)
    std = [np.std(tail_df.tail_sum[i - smooth_tailframe:i + smooth_tailframe]) 
            for i in range(smooth_tailframe, len(tail_df.tail_sum) - smooth_tailframe)]
    tail_df['std'] =[0] * smooth_tailframe + std + [0] * smooth_tailframe
    return tail_df

def normalize_tail_sum(tail_sum_values):
    baseline = np.nanmean(tail_sum_values)
    norm_tail_sum_values = np.subtract(tail_sum_values, baseline)
    norm_tail_sum_values_filled = pd.Series(norm_tail_sum_values).ffill().to_numpy()

    return norm_tail_sum_values_filled


def analyze_tail(tail_df, stimulus_df, img_hz, stimulus_df_frame_col = 'frame',
                 stimulus_s = 5, strength_boundary = 0.25, min_on_s = 0.1, cont_cutoff_s = 0.05):
    """
    # from cleo - danionella paper
    works in conjunction with the stimulus df (with visual motion cues)
    capture tail events happened in the current inputs.
        stimulus_df: the dataframe contain all the stimulus and their onset frames
        tail_df: the dataframe of the tail movement and their corresponding frames
        img_hz: imaging speed (NOTE: will be different if using imaging or tail frames)
        stimuli_s: the seconds that the stimuli was on
        strength_boundary: the minimal std for a tail to be counted as on - in general can be the std of the tail sum, or can set arbirtatirly
        min_on_s: the minimal frame for a bout to be considered bouts (in tail frames)
        cont_cutoff_s: the minimal frame of bout interval (in imaging speed rather than behavior speed) for the bout
        to be counted as being continuous
    Return:
        tail_bout_df: a dataframe contains all the bout information, which includes
            cont_tuples: a list that contains the frames for [on_frame, off_frame] of each bout
            tail_strength: a list that contains the strength of the tail for each bout, calculated by standard deviation
            tail_angle_pos: a list that contains all the bouts and their corresponding max positive tail angle
            tail_angle_neg:a list that contains all the bouts and their corresponding max negative tail angle
            tail_duration_s: a list that contains all the bouts and their corresponding duration in seconds
            tail_stimuli: a list that contains all the stimuli for each bouts, if they are happened within the stimuli
             presentation. If not, the stimuli is labeled "spontaneous"
    """

    tail_df = tail_df.ffill()
    tail_hz = 1/np.mean(np.diff(tail_df.iloc[:100]['t']))

    #calibrate to mean
    baseline = np.nanmean(tail_df.tail_sum)
    tail_df.tail_sum = np.subtract(tail_df.tail_sum, baseline)
    tail_df.tail_sum = tail_df.tail_sum.ffill()

    #collect positive and negative tail movement
    pos = np.where(tail_df.tail_sum > 0,tail_df.tail_sum, 0)
    neg = np.where(tail_df.tail_sum < 0,tail_df.tail_sum, 0)

    # group/smooth by running window  of ~100ms
    smooth_tailframe = int(0.05 * tail_hz)
    std = [np.std(tail_df.tail_sum[i - smooth_tailframe:i + smooth_tailframe]) for i in range(smooth_tailframe, len(tail_df.tail_sum) - smooth_tailframe)]
    tail_df['std'] =[0] * smooth_tailframe + std + [0] * smooth_tailframe
    if strength_boundary is None:
        strength_boundary = np.nanmean(tail_df['std'].values) + (np.nanstd(tail_df['std'].values)*2.5)
    bout_on = tail_df['std']> strength_boundary
    bout_on = [int(x) for x in bout_on]
    on_index = np.where(np.diff(bout_on) == 1)[0]
    on_index = [i + smooth_tailframe for i in on_index]
    off_index = np.where(np.diff(bout_on) == -1)[0]
    off_index = [i + smooth_tailframe for i in off_index]
    if len(on_index) != 0 and len(off_index) != 0:
        if on_index[0] > off_index[0]:
            on_index = np.concatenate([[0], on_index])
        if on_index[-1] > off_index[-1]:
            off_index = np.concatenate([off_index, [len(tail_df) - 1]])
        on_tuples = [(on, off) for on, off in zip(on_index, off_index) if off - on < len(tail_df) and off > on]
    cont_on_index = []
    cont_off_index = []
    if len(on_tuples) > 0:
        cont_on_index = [on_tuples[0][0]]
        if len(on_tuples) > 1:
            big_interval = np.array([on_tuples[i][0] - on_tuples[i - 1][1] for i in range(1, len(on_tuples))]) \
                           > (cont_cutoff_s * tail_hz)
            for i in range(0, len(big_interval)):
                if big_interval[i]:
                    cont_on_index = cont_on_index + [on_tuples[i + 1][0]]
                    cont_off_index = cont_off_index + [on_tuples[i][1]]
        cont_off_index = cont_off_index + [on_tuples[-1][1]]
    cont_tuples = [(on, off) for on, off in zip(cont_on_index, cont_off_index) if off - on > (min_on_s * tail_hz)]

    #calculate the actual frame(approx.) and image onset index/frames
    if stimulus_df_frame_col == 'frame':
        cont_tuples_imageframe = [(tail_df.iloc[tu[0]].frame, tail_df.iloc[tu[1]].frame) for tu in cont_tuples]
    else: # else we don't care about this column
        cont_tuples_imageframe = np.full(len(cont_tuples), np.nan)

    tail_strength = np.full(len(cont_tuples), np.nan)
    tail_angle = np.full(len(cont_tuples), np.nan)
    tail_angle_pos = np.full(len(cont_tuples), np.nan)
    tail_angle_posmax = np.full(len(cont_tuples), np.nan)
    tail_angle_neg = np.full(len(cont_tuples), np.nan)
    tail_angle_negmin = np.full(len(cont_tuples), np.nan)
    tail_duration_s = np.full(len(cont_tuples), np.nan)
    tail_frequency_s = np.full(len(cont_tuples), np.nan)
    tail_stimuli = ['spontaneous'] *len(cont_tuples)
    tail_stimuli_rep = [-1] * len(cont_tuples) # default rep for spontaneous is -1
    for i in range(len(cont_tuples)):
        tail_strength[i] = np.nanmean(tail_df['std'][cont_tuples[i][0]:cont_tuples[i][1]])
        tail_angle[i] = np.nanmean(tail_df.tail_sum[cont_tuples[i][0]:cont_tuples[i][1]])
        tail_angle_pos[i] = np.nanmean(pos[cont_tuples[i][0]:cont_tuples[i][1]])
        tail_angle_posmax[i] = np.max(pos[cont_tuples[i][0]:cont_tuples[i][1]])
        tail_angle_neg[i] = np.nanmean(neg[cont_tuples[i][0]:cont_tuples[i][1]])
        tail_angle_negmin[i] = np.min(neg[cont_tuples[i][0]:cont_tuples[i][1]])
        tail_duration_s[i] = np.divide((cont_tuples[i][1] - cont_tuples[i][0]), tail_hz)
        tail_of_interest = list(tail_df.iloc[cont_tuples[i][0]:cont_tuples[i][1]].tail_sum)
        mean_line = np.mean(tail_of_interest)
        crossing = np.subtract(arrutils.pretty(tail_of_interest, 3), mean_line)
        crossing = np.sign(crossing)
        crossing = np.count_nonzero(np.diff(crossing))
        tail_frequency_s[i] = np.divide(crossing / 2, tail_duration_s[i])
        if not stimulus_df.empty:
            if stimulus_df_frame_col == 'frame': # comparing IMAGING frames with the stimulus
                if cont_tuples_imageframe[i][0] > stimulus_df.iloc[0][stimulus_df_frame_col] :
                    stimulus_responding = stimulus_df[stimulus_df[stimulus_df_frame_col] <= cont_tuples_imageframe[i][0]].iloc[-1]#find the nearest stimuli before and see if tail happens within the stimulus
                    if stimulus_responding[stimulus_df_frame_col] + stimulus_s * img_hz >= cont_tuples_imageframe[i][0]:#if tail starts before the stimulus ends
                        tail_stimuli[i] = stimulus_responding['stim_name']
                        tail_stimuli_rep[i] = stimulus_responding['rep']
            else: # comparing TAIL frames with the stimulus
                if cont_tuples[i][0] > stimulus_df.iloc[0][stimulus_df_frame_col] :
                    stimulus_responding = stimulus_df[stimulus_df[stimulus_df_frame_col] <= cont_tuples[i][0]].iloc[-1]#find the nearest stimuli before and see if tail happens within the stimulus
                    if stimulus_responding[stimulus_df_frame_col] + stimulus_s * img_hz >= cont_tuples[i][0]:#if tail starts before the stimulus ends
                        tail_stimuli[i] = stimulus_responding['stim_name']
                        tail_stimuli_rep[i] = stimulus_responding['rep']

    tail_bout_df = pd.DataFrame(
        {'cont_tuples_tailindex': cont_tuples, 'cont_tuples_imageframe': cont_tuples_imageframe,
         'tail_strength': tail_strength, 'tail_angle': tail_angle, 'tail_angle_pos': tail_angle_pos,
         'tail_angle_posmax': tail_angle_posmax, 'tail_angle_neg': tail_angle_neg, 'tail_angle_negmin': tail_angle_negmin,
         'tail_duration_s': tail_duration_s, 'tail_frequency_s': tail_frequency_s,'tail_stimuli': tail_stimuli, 'tail_stimuli_rep': tail_stimuli_rep})

    return tail_df, tail_bout_df


### PLOTTING TAIL DATA ###

def make_bout_plots(tail_analysis_dataframe, stimuli_groups=None, stimuli_color_dict=None,
                    length_of_stim_rep_s = 6, norm_to_spont = True, plot_both = True):
    if stimuli_color_dict == None:
        stimuli_color_dict = {**constants.monocular_dict, **constants.combined_binocular_dict,
                              **constants.shearing_stims_dict, **{'spontaneous': [0.5, 0.5, 0.5]}}
        alpha = 0.3
        stimuli_color_dict = {key: vals + [alpha] for key, vals in stimuli_color_dict.items()}
    if stimuli_groups == None:
        stimuli_groups = [['spontaneous', 'converging', 'diverging'],
                          ['left', 'forward', 'right', 'backward'],
                          ['lateral', 'medial', 'binocular'],
                          ['forward_backward', 'forward_x', 'x_backward'],
                          ['backward_forward', 'x_forward', 'backward_x']]
    stim_order = [i for lst in stimuli_groups for i in lst]
    sorted_palette = {k: stimuli_color_dict[k] for k in stim_order}
    # gathering some more data
    if 'binocular' in stim_order:
        renamed_stims_bout_df = stimuli.combine_binocular_stims_for_tail_df(tail_analysis_dataframe)  # combining the binocular stims
        final_df = pd.concat([renamed_stims_bout_df, tail_analysis_dataframe[tail_analysis_dataframe.tail_stimuli.isin(
            ['right', 'left'])]])  # putting the right and left back in for better plotting
    else:
        final_df = tail_analysis_dataframe

    if norm_to_spont:
        original_vals = final_df['tail_angle'].values
        spont_median = np.median(final_df[final_df.tail_stimuli == 'spontaneous']['tail_angle'].values)
        norm_vals = original_vals - spont_median
        final_df['tail_angle'] = norm_vals

    final_df = final_df[final_df['tail_stimuli'].isin(stim_order)]
    # # derive the bouts per stim - keeping in case i want to use this later
    # bouts_per_stim = final_df.groupby('tail_stimuli').size().reset_index(name='total_bouts')
    # bouts_per_stim = bouts_per_stim.set_index('tail_stimuli').reindex(stim_order, fill_value=0).reset_index()
    # reps_per_stim = pd.DataFrame(columns=['tail_stimuli', 'length_in_s'], index=np.arange(len(stim_order)))
    # for i, each_stim in enumerate(stim_order):
    #     subdf = final_df[final_df.tail_stimuli == each_stim]
    #     reps_per_stim.iloc[i]['tail_stimuli'] = each_stim
    #     reps_per_stim.iloc[i]['length_in_s'] = (subdf.tail_stimuli_rep.nunique()) * length_of_stim_rep_s
    #     if each_stim == 'spontaneous': # if spontaneous, just keep the number of reps to be the same as the number of bouts so it just goes to 1
    #         reps_per_stim.iloc[i]['length_in_s'] = bouts_per_stim[bouts_per_stim.tail_stimuli == 'spontaneous'].total_bouts.values[0]
    # bouts_per_stim['bouts_per_s'] = bouts_per_stim['total_bouts'] / reps_per_stim['length_in_s']

    # now make it so that bouts are counted as bouts per sec
    all_stim_df_lst = []
    for i, each_stim in enumerate(stim_order):
        stim_df = final_df[final_df.tail_stimuli == each_stim]
        grouped_stim_df = stim_df.groupby('tail_stimuli_rep').size().reset_index(name='total_bouts')
        grouped_stim_df['tail_stimuli'] = each_stim
        grouped_stim_df['bouts_per_s'] = grouped_stim_df['total_bouts'] / length_of_stim_rep_s
        if each_stim == 'spontaneous':
            grouped_stim_df['bouts_per_s'] = grouped_stim_df['total_bouts'] / grouped_stim_df['total_bouts']
        grouped_stim_df = grouped_stim_df[['tail_stimuli', 'bouts_per_s']]
        all_stim_df_lst.append(grouped_stim_df)
    bouts_per_s_df = pd.concat(all_stim_df_lst).reset_index(drop=True)

    # metrics to plot
    metrics = ['tail_angle', 'tail_frequency_s', 'tail_duration_s', 'tail_strength', 'bouts_per_s']

    # ORDERED BY STIMULI ORDER/GROUPS
    fig1, axes1 = plt.subplots(nrows=len(metrics), ncols=1, figsize=(12, 15), sharex=True)
    for ax, metric in zip(axes1, metrics):
        start = 0
        for group in stimuli_groups:
            width = len(group)
            ax.axvspan(start - 0.5 + 0.1, start + width - 0.5 - 0.1, color='lightgray', alpha=0.08)
            start += width
        plot_df = final_df
        if metric == 'bouts_per_s':
            plot_df = bouts_per_s_df
            bars = sns.barplot(x='tail_stimuli', y=metric, data=plot_df, width=0.6,alpha=0.4,
                               order=stim_order, palette=sorted_palette, ax=ax,
                               capsize = 0.2,errwidth = 1, errorbar = 'se')
        else:
            sns.boxplot(x='tail_stimuli', y=metric, data=plot_df, ax=ax, boxprops=dict(alpha=0.4), width=0.6,
                        order=stim_order, palette=sorted_palette, showfliers=False)
        sns.stripplot(x='tail_stimuli', y=metric, data=plot_df, ax=ax,
                      order=stim_order, hue='tail_stimuli', palette=sorted_palette, dodge=False, size=4, alpha=0.7)
        ax.get_legend().remove()
        if metric == 'tail_angle':
            ax.axhline(0, color='lightgray')
            if norm_to_spont:
                ax.set_ylabel('normalized tail angle')
        if metric == 'tail_duration_s':
            ax.set_ylim(0, 1)

        ax.axhline(np.nanmedian(plot_df[plot_df.tail_stimuli == 'spontaneous'][metric].values), color = 'k', linestyle='--')
        ax.set_ylabel(metric)
        ax.set_xlabel('')  # only bottom plot will have xlabel
        ax.tick_params(axis='x', rotation=45)  # rotate stimuli labels
    axes1[-1].set_xlabel('Stimulus')
    sns.despine()
    fig1.tight_layout()

    # ORDERED BY MAXIMUM VALUE TO MINIMUM, REGARDLESS OF STIM
    if plot_both:
        fig2, axes2 = plt.subplots(nrows=len(metrics), ncols=1, figsize=(12, 16))
        for ax, metric in zip(axes2, metrics):
            if metric != 'bout_count':
                order = final_df.groupby('tail_stimuli')[metric].median().sort_values(ascending=False).index
            else:
                order = bouts_per_stim.set_index('tail_stimuli')['total_bouts'].sort_values(ascending=False).index
            sorted_palette = {k: stimuli_color_dict[k] for k in order}

            if metric != 'bout_count':
                sns.boxplot(x='tail_stimuli', y=metric, data=final_df, ax=ax, boxprops=dict(alpha=0.4), width=0.6,
                            order=order, palette=sorted_palette, showfliers=False)
                sns.stripplot(x='tail_stimuli', y=metric, data=final_df, ax=ax, order=order,
                              palette=sorted_palette, dodge=False, size=4, alpha=0.7)
                if metric == 'tail_angle':
                    ax.axhline(0, linestyle='--', color='lightgray')
                if metric == 'tail_duration_s':
                    ax.set_ylim(0, 1)
                ax.axhline(np.nanmedian(final_df[final_df.tail_stimuli == 'spontaneous'][metric].values), color='k',
                           linestyle='--')
            else:
                sns.barplot(x='tail_stimuli', y='total_bouts', data=bouts_per_stim, width=0.6, order=order,
                            palette=sorted_palette, ax=ax)
                ax.set_ylim(0, bout_count_lim)
            ax.tick_params(axis='x', labelsize=8)
            ax.set_ylabel(metric)
            ax.set_xlabel('')
            ax.tick_params(axis='x', rotation=45)
        axes2[-1].set_xlabel('Stimulus')
        sns.despine()
        fig2.tight_layout()
    else:
        fig2 = 'None'

    return fig1, fig2

def make_cleo_polar_plot(ax, this_tail_bout_df, stim_directions, plotting_variable, scatter = True):
        """
        plotting subgroups of stimulus locked bout angle vs tail metric
            ax: the axes to plot on
            this_tail_bout_df: the tail analysis dataframe (all of it)
            stim_directions: list of stimulus directions to plot
            plotting_variable: metric of the tail analysis dataframe that you are plotting (on the x axis)
            scatter: whether or not to plot scatter plot (otherwise it will only plot the bars)
        """
        # get all bouts for requested stimuli
        response_index = [
            idx for idx in this_tail_bout_df.index
            if this_tail_bout_df.tail_stimuli[idx] in stim_directions]

        if len(response_index) == 0:
            # nothing to plot, just fix axis and return
            return

        # collect values
        tail_angle = np.array([this_tail_bout_df.tail_angle[idx] for idx in response_index])
        tail_plotting_variable = np.array([this_tail_bout_df[plotting_variable][idx] for idx in response_index])
        tail_stimuli_lst = [this_tail_bout_df.tail_stimuli[idx] for idx in response_index]

        # decide color dictionary
        if stim_directions == ['spontaneous']:
            clr_dict = {}
            tail_stimuli_color = ['grey'] * len(tail_stimuli_lst)
        else: # just use all the color dicts combined together
            clr_dict = {**constants.monocular_dict, **constants.binocular_dict, **constants.combined_binocular_dict,
                        **constants.shearing_stims_dict, **{'spontaneous': [0.5, 0.5, 0.5]}}
            tail_stimuli_color = [clr_dict.get(stim, (0.5, 0.5, 0.5))[:3] for stim in tail_stimuli_lst]

        # filter invalid values
        mask = np.isfinite(tail_angle) & np.isfinite(tail_plotting_variable)
        if not np.any(mask):
            return

        tail_angle = tail_angle[mask]
        tail_plotting_variable = tail_plotting_variable[mask]
        tail_stimuli_lst = [tail_stimuli_lst[i] for i in np.where(mask)[0]]
        tail_stimuli_color = [tail_stimuli_color[i] for i in np.where(mask)[0]]

        # scatter points
        if scatter:
            ax.scatter(np.deg2rad(tail_angle), tail_plotting_variable,
                   color=tail_stimuli_color, linewidth=0, s=3, alpha=0.5, zorder=3)

        for stim in set(tail_stimuli_lst):
            stim_angles_deg = np.array([a for a, s in zip(tail_angle, tail_stimuli_lst) if s == stim])
            stim_metrics = np.array([m for m, s in zip(tail_plotting_variable, tail_stimuli_lst) if s == stim])
            if stim_angles_deg.size == 0:
                continue

            stim_angles_rad = np.deg2rad(stim_angles_deg)
            # mean_angle = np.arctan2(np.sin(stim_angles_rad).mean(),
            #                         np.cos(stim_angles_rad).mean())
            mean_angle = np.arctan2(np.nanmedian(np.sin(stim_angles_rad)),
                                    np.nanmedian(np.cos(stim_angles_rad)))
            median_metric = np.nanmedian(stim_metrics) # <-- use metric median as bar length
            if median_metric <= 0:
                continue  # skip for log-scale axes

            clr = 'grey' if stim == 'spontaneous' else clr_dict.get(stim, (0.5, 0.5, 0.5))
            ax.bar(mean_angle, median_metric, width=0.1, color=clr, alpha=0.5, zorder=2)

def fix_ax_cleo_polar_plot(ax, variable, ylim=2, log_scale=False):
    """
    Make axis look prettier for the half polar plots
        ax: axes to plot
        variable: variable that you are plotting (for the y label)
        ylim: ylim limits (the radius of the polar plot), if None = automatic scaling
        log_scale: log scale (True or False), if you want to plot the polar plot data on a log scale
    """
    ax.set_theta_offset(np.deg2rad(-90))
    ax.set_theta_direction('counterclockwise')
    ax.grid(linewidth=0.5, linestyle=":", c='grey')

    if log_scale:
        ax.set_rscale('symlog')
    if ylim == None:
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    else:
        ax.set_ylim(0, ylim)
    ax.tick_params(axis='y', labelsize=6, labelrotation=30)
    ax.set_ylabel(variable)

    # angular axis: restrict to -90° .. +90°
    ax.set_thetalim(np.deg2rad(-90), np.deg2rad(90))
    ax.set_xticks(np.deg2rad([-90, -45, 0, 45, 90]))
    ax.tick_params(axis='x', labelsize=6, labelrotation=0)

    # style spines if supported
    try:
        ax.spines[['start', 'end']].set_color("black")
        ax.spines['polar'].set_linestyle((0, (2, 5)))
        ax.spines['polar'].set_linewidth(0.8)
        ax.spines['polar'].set_color("black")
    except Exception:
        pass


































# reads in the tail data into a df --> custom 2p way

# def tail_reader(tail_path):
#     
#     tail_data = TdmsFile(tail_path)
#     tail_df = tail_data.as_dataframe()
#     tail_df = tail_df[tail_df["/'TailLoc'/'Time'"].notna()]
#     tail_df.loc[:, "t"] = (
#         tail_df["/'TailLoc'/'Time'"].values - tail_df["/'TailLoc'/'Time'"].values[0]
#     )

#     t_arr = []
#     for t in range(len(tail_df.t.values)):
#         t_arr.append(np.timedelta64(tail_df.t.values[t], "ms").astype(int))
#     tail_df["t"] = t_arr
#     tail_df["/'TailLoc'/'Time'"] = tail_df["/'TailLoc'/'Time'"].dt.tz_localize(
#         "US/Eastern"
#     )

#     # add extra column at the end with the converted time
#     tail_ts = []
#     for i in range(len(tail_df)):
#         try:
#             val = dt.strptime(
#                 str(tail_df["/'TailLoc'/'Time'"].iloc[i]).split(" ")[1].split("-")[0],
#                 "%H:%M:%S.%f",
#             ).time()
#         except:
#             val = dt.strptime(
#                 str(tail_df["/'TailLoc'/'Time'"].iloc[i]).split(" ")[1].split("-")[0],
#                 "%H:%M:%S",
#             ).time()
#         tail_ts.append(val)
#     tail_df.loc[:, "conv_t"] = tail_ts

#     converted_tail_times = []
#     tail_times = tail_df["conv_t"].values

#     # converted time needs to be changed by this hour value given by lab view data
#     add_hour = (
#         str(tail_df["/'TailLoc'/'Time'"].iloc[0])
#         .split(" ")[1]
#         .split("-")[1]
#         .split(":")[0]
#     )

#     for i in range(len(tail_times)):
#         tail_times[i] = tail_times[i].replace(
#             hour=tail_times[i].hour - int(add_hour),
#             minute=tail_times[i].minute,
#             second=tail_times[i].second,
#             microsecond=tail_times[i].microsecond,
#         )
#     converted_tail_times.append(
#         dateToMillisec(tail_times[i])
#     )  # convert to milliseconds

#     new_tail_t = np.asarray(converted_tail_times)
#     tail_df = tail_df.iloc[1:]

#     return tail_df
