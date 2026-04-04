import numpy as np
import pandas as pd
import seaborn as sns
import sys, os, shutil
sys.path.append(r'C://Users//Zichen//anaconda3//envs//2ptank//Lib//site-packages//')
import utils
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec,colormaps

pink = "#FB008C"
green = "#72D100"
cmap_conv = LinearSegmentedColormap.from_list("conv",[green, (1, 1, 1), pink])
cmap_conj = colormaps.get_cmap('coolwarm_r')

plt.rcParams.update({'font.size': 13})

def get_pertrial_f(fish):
    """
    Get the dF/F for the fish organized by stimulus and trials, the f includes the entire duration of the stimulus (stationary + moving period)
    :param fish:
    :param stim_onset_type: if 'overlap', the stationary time for overlap stim counts the time before the overlap starts; if 'gratings', the stationary time for overlap stim counts when the grating starts
    :return:
    """
    stim_list = fish.stimulus_df.stim_name.unique()
    f_pertrial_dict = {stim: None for stim in stim_list}
    for stim in stim_list:
        #get important stimulus information
        stim_df = fish.stimulus_df[fish.stimulus_df['stim_name'] == stim].reset_index(drop=True)
        duration = stim_df['duration'].iloc[0] + 20
        print('getting pertrial f stim duration + 20s')
        #define plotting range (what are the frames of data to grab in general)
        image_range_frame = (0, int((duration) / fish.image_s))#general
        #for this stimulus, get the data shaped like trial, neuron, time
        #axis 0: trial; axis 1: neuron; axis 2: frame
        f_pertrial_perplane = {plane: np.full(shape=(len(stim_df), len(fish.f_dict[plane]), image_range_frame[1] - image_range_frame[0]), fill_value=np.nan) for plane in fish.planes}
        for trial, stim_row in stim_df.iterrows():
            for plane in fish.planes:
                start_frame = utils.find_nearest_frameindex(fish.frametimes_dict[plane], stim_row['real_starttime_s'])#this is frmae at stim start time
                try:
                    f_pertrial_perplane[plane][trial, :, :] = fish.f_dict[plane].iloc[:, start_frame + image_range_frame[0]:start_frame + image_range_frame[1]]
                except ValueError:
                    print('imaging ends before last visstim')
                    temp_f = fish.f_dict[plane].iloc[:,start_frame + image_range_frame[0]:start_frame + image_range_frame[1]]
                    f_pertrial_perplane[plane][trial, :, :temp_f.shape[1]] = temp_f
        f_pertrial_dict[stim] = np.concatenate([f for f in f_pertrial_perplane.values()], axis = 1)#OG index still gets maintained
    return f_pertrial_dict


def select_visbarcode(fish, f_pertrial_dict, baseline_s=10, response_s=10, perc_trial_threshold=0.8):
    """Select barcoded neurons with 1.8std above baseline and below 1.8std for the opposite stimulus"""
    # TODO: replace xpos with rois?
    # build a neuron accumulator
    stim_list = fish.stimulus_df.stim_name.unique()
    all_stim = [s for s in stim_list if ('pause' not in s)]
    all_n_index = []
    for plane in fish.planes:
        all_n_index = all_n_index + list(fish.f_dict[plane].index)
    barcode = pd.DataFrame(columns=all_stim, index=all_n_index)
    for stim in all_stim:
        stim_df = fish.stimulus_df[fish.stimulus_df['stim_name'] == stim].reset_index(drop=True)
        stationary_time = stim_df['stationary_time'].iloc[0]
        if '[' in stim:  # overlap
            stationary_time = min(stationary_time)
        # define plotting range (what are the frames of data to grab in general)

        stationary_frame = int(stationary_time / fish.image_s)
        baseline_frame = int((stationary_time - baseline_s) / fish.image_s)  # how many frame before stationary frame
        response_frame = int((stationary_time + response_s) / fish.image_s)  # how many frame after stationary frame
        # get all the baseline per trial
        f_pertrial = f_pertrial_dict[stim]
        f_pertrial_baseline = f_pertrial[:, :, baseline_frame:stationary_frame]
        f_pertrial_response = f_pertrial[:, :, stationary_frame:response_frame]
        f_pertrial_baselinemean = np.mean(f_pertrial_baseline, axis=2)
        f_pertrial_baselinestd = np.std(f_pertrial_baseline, axis=2)
        f_pertrial_responsemax = np.mean(f_pertrial_response, axis=2)
        # if response > baselien + 1.8*std
        boundary = .5
        print(f"selection criteria: max >= baseline mean * {boundary}")
        f_responding = (f_pertrial_responsemax - f_pertrial_baselinemean) / f_pertrial_baselinestd >= boundary
        f_responding_percentage = np.divide(np.sum(f_responding, axis=0), f_responding.shape[0])
        barcode.loc[:, stim] = f_responding_percentage
    # actually picking barcode neurons
    barred_neurons = {s: None for s in all_stim}
    for stim in all_stim:
        barred_neurons[stim] = list(barcode[(barcode[stim] >= perc_trial_threshold)].index)
    # get all elements
    vis_neurons = list(set().union(*barred_neurons.values()))
    return vis_neurons

def get_dot_dsi(fish, f_pertrial_dict, baseline_s = 7, response_s = 10, grating_stim = None):
    if grating_stim == None:
        dot_stim = [s for s in f_pertrial_dict.keys() if 'dot' in s and '[' not in s and 'pause' not in s]
    else:
        dot_stim = [s for s in f_pertrial_dict.keys() if grating_stim in s and '[' in s and 'pause' not in s]
    #Calculate the mean/max response of this neuron to each visual stimulus
    stim_responses = {}
    stim_responsetimes = {}
    for stim in dot_stim:
        #define plotting range (what are the frames of data to grab in general)
        stim_df = fish.stimulus_df[fish.stimulus_df['stim_name'] == stim].reset_index(drop=True)
        stationary_time = stim_df['stationary_time'].iloc[0]
        if grating_stim is not None:#overlapstimulus
            stationary_time = stationary_time[0]
            print('looking at overlap onset time, not grating')
        stationary_frame = int(stationary_time/fish.image_s)
        baseline_frame = int((stationary_time - baseline_s)/ fish.image_s)#how many frame before stationary frame
        response_frame = int((stationary_time + response_s)/ fish.image_s)#how many frame after stationary frame
        #get all the baseline per trial
        f_pertrial = f_pertrial_dict[stim]
        f_pertrial_baseline = f_pertrial[:, :, baseline_frame:stationary_frame]
        f_pertrial_response = f_pertrial[:, :, stationary_frame:response_frame]#[:, :, stationary_frame + int(4/ fish.image_s):response_frame]
        #get how big is the response
        f_pertrial_baselinemean = np.mean(f_pertrial_baseline, axis = 2)
        f_pertrial_responsemax = np.max(f_pertrial_response, axis = 2)
        f_pertrial_realresponse = np.subtract(f_pertrial_responsemax, f_pertrial_baselinemean)
        f_response = f_pertrial_realresponse.mean(axis = 0)
        stim_responses[stim] = f_response
        #get the timing of the response
        f_pertrial_responseargmax = np.argmax(f_pertrial_response, axis = 2)
        f_response_argmax = np.mean(f_pertrial_responseargmax, axis = 0)
        stim_responsetimes[stim] = f_response_argmax
    stim_responses = pd.DataFrame(stim_responses)
    stim_responsetimes = pd.DataFrame(stim_responsetimes)

    dot_r_stim = [i for i in dot_stim if 'dot_r' in i][0]
    dot_l_stim = [i for i in dot_stim if 'dot_l' in i][0]
    timing_index = (stim_responsetimes[dot_r_stim] - stim_responsetimes[dot_l_stim]) / (
                           stim_responsetimes[dot_r_stim] + stim_responsetimes[dot_l_stim] + 1e-9)#+1, earlier for l dot, -1, earlier for dot r

    mag_index = (stim_responses[dot_r_stim] - stim_responses[dot_l_stim]) / (
                        stim_responses[dot_r_stim] + stim_responses[dot_l_stim] + 1e-9)#+1, earlier for l dot, -1, earlier for dot r
    mag_index = (mag_index + 1) / 2
    stim_responses =pd.DataFrame({'timing': timing_index,'DSI': mag_index})
    return stim_responses

def select_dotbarcode(fish, f_pertrial_dict, baseline_s =10, response_s = 10, perc_trial_threshold = 0.8):
    """Select barcoded neurons with 1.8std above baseline and below 1.8std for the opposite stimulus"""
    #TODO: replace xpos with rois
    #build a neuron accumulator
    stim_list = fish.stimulus_df.stim_name.unique()
    dot_stim = [s for s in stim_list if ('dot' in s) and ('[' not in s)]
    all_n_index = []
    for plane in fish.planes:
        all_n_index = all_n_index + list(fish.f_dict[plane].index)
    barcode = pd.DataFrame(columns = dot_stim, index = all_n_index)
    for stim in dot_stim:
        stim_df = fish.stimulus_df[fish.stimulus_df['stim_name'] == stim].reset_index(drop=True)
        stationary_time = stim_df['stationary_time'].iloc[0]
        #define plotting range (what are the frames of data to grab in general)
        stationary_frame = int(stationary_time/fish.image_s)
        baseline_frame = int((stationary_time - baseline_s)/ fish.image_s)#how many frame before stationary frame
        response_frame = int((stationary_time + response_s)/ fish.image_s)#how many frame after stationary frame
        #get all the baseline per trial
        f_pertrial = f_pertrial_dict[stim]
        f_pertrial_baseline = f_pertrial[:, :, baseline_frame:stationary_frame]
        f_pertrial_response = f_pertrial[:, :, stationary_frame:response_frame]
        f_pertrial_baselinemean = np.mean(f_pertrial_baseline, axis = 2)
        f_pertrial_baselinestd = np.std(f_pertrial_baseline, axis = 2)
        f_pertrial_responsemax = np.max(f_pertrial_response, axis = 2)
        #if response > baselien + 1.8*std
        boundary = 2
        print(f"selection criteria: max >= baseline mean * {boundary}")
        f_responding = (f_pertrial_responsemax - f_pertrial_baselinemean)/f_pertrial_baselinestd >= boundary
        f_responding_percentage = np.divide(np.sum(f_responding, axis = 0), f_responding.shape[0])
        barcode.loc[:, stim] = f_responding_percentage
    #actually picking barcode neurons
    barred_neurons = {s: None for s in dot_stim}
    for stim in dot_stim:
        barred_neurons[stim] = list(barcode[(barcode[stim] >= perc_trial_threshold)].index)
    return barred_neurons

def sort_dotneurons(fish, f_pertrial_dict, barred_dotneurons):
    """
    Sort the dot responding neurons by how early they response to dot across trials
    """
    stim_list = fish.stimulus_df.stim_name.unique()
    dot_stim = [s for s in stim_list if ('dot' in s) and ('[' not in s)]
    for stim in dot_stim:
        stationary_s = fish.stimulus_df[fish.stimulus_df.stim_name == stim].iloc[0].stationary_time
        stationary_frame = int(stationary_s/fish.image_s)
        f_avg = f_pertrial_dict[stim][:, barred_dotneurons[stim], stationary_frame:].mean(axis = 0)#avg across trial, only looks at nonstationary period
        peak_frame = np.argmax(f_avg, axis = 1)#find max frame for each neuron
        sort_i = np.argsort(peak_frame)
        barred_dotneurons[stim] = [barred_dotneurons[stim][i] for i in sort_i]
    return barred_dotneurons


def plot_dotneurons(fish, f_pertrial_dict, barred_dotneurons):
    stim_list = fish.stimulus_df.stim_name.unique()
    dot_stim = [s for s in stim_list if ('dot' in s) and ('[' not in s)]
    fig, ax = plt.subplots(3, len(dot_stim), figsize = (10, 10))
    for n, stim in enumerate(dot_stim):
        ax[0, n].imshow(fish.img_dict[0], origin='lower', cmap='Greys_r')
        ax[0, n].scatter(fish.pos_all.loc[barred_dotneurons[stim], 'xpos'],
                         fish.pos_all.loc[barred_dotneurons[stim], 'ypos'], c=range(len(barred_dotneurons[stim])),
                         cmap='plasma', s=.3)
        ax[0, n].grid(False)
        ax[0, n].set_xticks([])
        ax[0, n].set_yticks([])

        for offset, plane in enumerate(fish.planes[::-1]):
            n_plane = np.where(fish.pos_all.loc[barred_dotneurons[stim], 'zpos'] == plane)
            counts, bins = np.histogram(fish.pos_all.loc[n_plane]['ypos'], bins=10, range = [0, fish.img_dict[0].shape[0]])
            perc = [i/len(barred_dotneurons[stim]) for i in counts]
            ax[1, n].bar(bins[:-1], perc, width=np.diff(bins), align='edge', alpha=0.7, bottom=offset * .05)

        ax[1, n].grid(False)
        ax[1, n].set_xticks([0, fish.img_dict[0].shape[0]])
        ax[1, n].set_xticklabels(['P', 'A'])
        ax[1, n].set_yticks([0, len(fish.planes) * .05])
        ax[1, n].set_yticklabels(['V', 'D'])

        sns.heatmap(f_pertrial_dict[stim][:, barred_dotneurons[stim], :].mean(axis=0), cmap='viridis', ax=ax[2, n],
                    xticklabels=[], yticklabels=[])

        ax[0, n].set_title(stim)
    return fig

def plot_timing(fish, stim_responses):
    fig = plt.figure(dpi=240, figsize=(12, 4))
    gs = gridspec.GridSpec(2, len(fish.planes) + 1, height_ratios=[1, 0.5], figure=fig)
    ax_imshow_all = fig.add_subplot(gs[0, len(fish.planes)])
    ax_timing_all = fig.add_subplot(gs[1, len(fish.planes)])
    ax_timing_all.set_ylim([0, 1])
    ax_timing_all.set_xlim([-1, 1])
    #only plot visually responsive neurons that have been somewhat barrcoded
    barred = set(stim_responses.index[
                     stim_responses[[c for c in stim_responses.columns if c.startswith("barred_")]].any(axis=1)])
    for plane in fish.planes:
        #plot rainbow plot
        ax_imshow = fig.add_subplot(gs[0, plane])
        ax_imshow.imshow(fish.img_dict[plane], cmap = 'gray', origin = 'lower')
        #get neurons in this plane
        plane_n = list(set(fish.pos_dict[plane].index) & barred)
        #color: tuned angles
        colors = stim_responses.loc[plane_n, 'timing']
        colors = plt.cm.coolwarm_r(plt.Normalize(vmin=-.2, vmax=.2)(colors))
        #alpha: dsi
        alphas = stim_responses.loc[plane_n, 'DSI']
        alphas = [i if not np.isnan(i) else 0 for i in alphas]
        ax_imshow.scatter(fish.pos_dict[plane].loc[plane_n, 'xpos'], fish.pos_dict[plane].loc[plane_n, 'ypos'], c = colors, s = .1, alpha = 1)
        ax_imshow_all.scatter(fish.pos_dict[plane].loc[plane_n, 'xpos'], fish.pos_dict[plane].loc[plane_n, 'ypos'], c = colors, s = .1, alpha = 1)
        ax_imshow.set_axis_off()

        #plot vector plot
        ax_timing = fig.add_subplot(gs[1, plane])
        times = stim_responses.loc[plane_n, 'timing'].values
        mags = stim_responses.loc[plane_n, 'DSI'].values
        ax_timing.scatter(times, mags, c=colors, alpha=0.7, s=1)
        ax_timing_all.scatter(times, mags, c=colors, alpha=0.7, s=.1)
        ax_timing.set_ylim([0, 1])
        ax_timing.set_xlim([-1, 1])
    #plot it across all neurons
    ax_imshow_all.imshow(fish.img_dict[plane], cmap = 'gray', origin = 'lower')
    ax_imshow_all.set_axis_off()
    return fig

def get_grating_dsi(fish, f_pertrial_dict, baseline_s = 7, response_s = 10, dot_stim = None):
    if dot_stim == None:
        omr_stim = [s for s in f_pertrial_dict.keys() if 'dot' not in s and '[' not in s and 'pause' not in s]
    else:
        omr_stim = [s for s in f_pertrial_dict.keys() if dot_stim in s and '[' in s and 'pause' not in s]
    #Calculate the mean/max response of this neuron to each visual stimulus
    stim_responses = {}
    for stim in omr_stim:
        #define plotting range (what are the frames of data to grab in general)
        stim_df = fish.stimulus_df[fish.stimulus_df['stim_name'] == stim].reset_index(drop=True)
        stationary_time = stim_df['stationary_time'].iloc[0]
        if dot_stim is not None:#overlapstimulus
            stationary_time = stationary_time[0]
            print('looking at overlap onset time, not grating')
        stationary_frame = int(stationary_time/fish.image_s)
        baseline_frame = int((stationary_time - baseline_s)/ fish.image_s)#how many frame before stationary frame
        response_frame = int((stationary_time + response_s)/ fish.image_s)#how many frame after stationary frame
        #get all the baseline per trial
        f_pertrial = f_pertrial_dict[stim]
        f_pertrial_baseline = f_pertrial[:, :, baseline_frame:stationary_frame]
        f_pertrial_response = f_pertrial[:, :, stationary_frame:response_frame]#[:, :, stationary_frame + int(4/ fish.image_s):response_frame]
        f_pertrial_baselinemean = np.mean(f_pertrial_baseline, axis = 2)
        f_pertrial_responsemax = np.mean(f_pertrial_response, axis = 2)
        f_pertrial_response = np.subtract(f_pertrial_responsemax, f_pertrial_baselinemean)
        f_response = f_pertrial_response.mean(axis = 0)
        stim_responses[stim] = f_response
    stim_responses = pd.DataFrame(stim_responses)
    #DSI = fpref - fopp / (fpref + fopp)
    dirs = stim_responses.columns
    if dot_stim is not None:#overlap
        angles = [utils.omr_angles[s.split("'")[3]] for s in dirs]
        pref = stim_responses.max(axis=1)
        pref_dir = stim_responses.idxmax(axis=1).tolist()
        pref_dir_angle = [utils.omr_angles[s.split("'")[3]] for s in pref_dir]
        opp_dir_angle = [(a + 180) % 360 for a in pref_dir_angle]
        opp_dir = opp_dir = [str([dot_stim, utils.angles_omr[a]]) for a in opp_dir_angle]
    else:
        angles = [utils.omr_angles[s] for s in dirs]
        pref = stim_responses.max(axis=1)
        pref_dir = stim_responses.idxmax(axis=1).tolist()
        pref_dir_angle = [utils.omr_angles[s] for s in pref_dir]
        opp_dir_angle = [(a + 180)%360 for a in pref_dir_angle]
        opp_dir = [utils.angles_omr[a] for a in opp_dir_angle]
    opp = [row[opp_dir[n]] for n, row in stim_responses.iterrows()]
    stim_responses['DSI1'] = (pref - opp) / (pref + opp)

    # DSI2 (vector sum)
    stim_responses['dir'] = stim_responses[dirs].apply(lambda r: utils.weighted_mean_angle(angles, r.values), axis=1)
    stim_responses['DSI2'] = stim_responses[dirs].apply(
        lambda r: np.abs(
            np.sum(np.array(r.values, dtype=float) * np.exp(1j * np.deg2rad(np.array(angles, dtype=float))))) / np.sum(
            r.values), axis=1)
    return stim_responses

def select_gratingbarcode(fish, f_pertrial_dict, baseline_s =10, response_s = 10, perc_trial_threshold = 0.8):
    """Select barcoded neurons with 1.8std above baseline and below 1.8std for the opposite stimulus"""
    #TODO: replace xpos with rois
    #build a neuron accumulator
    stim_list = fish.stimulus_df.stim_name.unique()
    omr_stim = [s for s in stim_list if ('dot' not in s) and ('[' not in s) and ('pause' not in s)]
    all_n_index = []
    for plane in fish.planes:
        all_n_index = all_n_index + list(fish.f_dict[plane].index)
    barcode = pd.DataFrame(columns = omr_stim, index = all_n_index)
    for stim in omr_stim:
        stim_df = fish.stimulus_df[fish.stimulus_df['stim_name'] == stim].reset_index(drop=True)
        stationary_time = stim_df['stationary_time'].iloc[0]
        #define plotting range (what are the frames of data to grab in general)
        stationary_frame = int(stationary_time/fish.image_s)
        baseline_frame = int((stationary_time - baseline_s)/ fish.image_s)#how many frame before stationary frame
        response_frame = int((stationary_time + response_s)/ fish.image_s)#how many frame after stationary frame
        #get all the baseline per trial
        f_pertrial = f_pertrial_dict[stim]
        f_pertrial_baseline = f_pertrial[:, :, baseline_frame:stationary_frame]
        f_pertrial_response = f_pertrial[:, :, stationary_frame:response_frame]
        f_pertrial_baselinemean = np.mean(f_pertrial_baseline, axis = 2)
        f_pertrial_baselinestd = np.std(f_pertrial_baseline, axis = 2)
        f_pertrial_responsemax = np.max(f_pertrial_response, axis = 2)
        #if response > baselien + 1.8*std
        boundary = 2
        print(f"selection criteria: max >= baseline mean * {boundary}")
        f_responding = (f_pertrial_responsemax  - f_pertrial_baselinemean)/f_pertrial_baselinestd >= boundary
        f_responding_percentage = np.divide(np.sum(f_responding, axis = 0), f_responding.shape[0])
        barcode.loc[:, stim] = f_responding_percentage
    #actually picking barcode neurons
    barred_neurons = {s: None for s in omr_stim}
    for stim in omr_stim:
        opp_stim = utils.angles_omr[(utils.omr_angles[stim]+ 180)%360]
        barred_neurons[stim] = list(barcode[(barcode[stim] >= perc_trial_threshold)].index)  # ].index
                                            #& (barcode[opp_stim] < perc_trial_threshold)].index)
    return barred_neurons

def sort_gratingneurons(fish, f_pertrial_dict, barred_gratingneurons):
    """
    Sort the grating responding neurons by how early they response to dot across trials
    """
    stim_list = fish.stimulus_df.stim_name.unique()
    dot_stim = [s for s in stim_list if ('dot' not in s) and ('[' not in s) and ('pause' not in s)]
    for stim in dot_stim:
        stationary_s = fish.stimulus_df[fish.stimulus_df.stim_name == stim].iloc[0].stationary_time
        stationary_frame = int(stationary_s/fish.image_s)
        f_avg = f_pertrial_dict[stim][:, barred_gratingneurons[stim], stationary_frame:].mean(axis = 0)#avg across trial, only looks at nonstationary period
        peak_frame = np.argmax(f_avg, axis = 1)#find max frame for each neuron
        sort_i = np.argsort(peak_frame)
        barred_gratingneurons[stim] = [barred_gratingneurons[stim][i] for i in sort_i]
    return barred_gratingneurons

def plot_gratingneurons(fish, f_pertrial_dict, barred_gratingneurons, stim_responses = None):
    stim_list = fish.stimulus_df.stim_name.unique()
    grating_stim = [s for s in stim_list if ('dot' not in s) and ('[' not in s) and ('pause' not in s)]
    grating_stim = np.sort(grating_stim)
    fig, ax = plt.subplots(3, len(grating_stim), figsize = (10, 8))
    for n, stim in enumerate(grating_stim):
        ax[0, n].imshow(fish.img_dict[2], origin='lower', cmap='Greys_r')
        if type(stim_responses) != pd.DataFrame:
            ax[0, n].scatter(fish.pos_all.loc[barred_gratingneurons[stim], 'xpos'],
                         fish.pos_all.loc[barred_gratingneurons[stim], 'ypos'], c=range(len(barred_gratingneurons[stim])),
                         cmap='plasma', s=.3)
        else:
            ax[0, n].scatter(fish.pos_all.loc[barred_gratingneurons[stim], 'xpos'],
                             fish.pos_all.loc[barred_gratingneurons[stim], 'ypos'],
                             c=stim_responses.loc[barred_gratingneurons[stim], 'DSI2'], vmax = 1, vmin = 0,
                             cmap='plasma', s=.3)
        #ax[0, n].grid(False)
        #ax[0, n].set_xticks([])
        #ax[0, n].set_yticks([])
        ax[0, n].set_title(stim)

        for offset, plane in enumerate(fish.planes[::-1]):
            n_plane = np.where(fish.pos_all.loc[barred_gratingneurons[stim], 'zpos'] == plane)
            counts, bins = np.histogram(fish.pos_all.loc[n_plane]['ypos'], bins=10, range = [0, fish.img_dict[0].shape[0]])
            perc = [i/len(barred_gratingneurons[stim]) for i in counts]
            ax[1, n].bar(bins[:-1], perc, width=np.diff(bins), align='edge', alpha=0.7, bottom=offset * .05)

        ax[1, n].grid(False)
        ax[1, n].set_xticks([0, fish.img_dict[0].shape[0]])
        ax[1, n].set_xticklabels(['P', 'A'])
        ax[1, n].set_yticks([0, len(fish.planes) * .05])
        ax[1, n].set_yticklabels(['V', 'D'])

        sns.heatmap(f_pertrial_dict[stim][:, barred_gratingneurons[stim], :].mean(axis=0), cmap='viridis', ax=ax[2, n],
                    xticklabels=[], yticklabels=[], vmin = 0, vmax = 0.75)


    return fig

def plot_rainbow(fish, stim_responses):
    fig = plt.figure(dpi=240, figsize=(12, 4))
    gs = gridspec.GridSpec(2, len(fish.planes) + 1, height_ratios=[1, 0.5], figure=fig)
    ax_imshow_all = fig.add_subplot(gs[0, len(fish.planes)])
    ax_polar_all = fig.add_subplot(gs[1, len(fish.planes)], polar = True)
    ax_polar_all.set_theta_offset(np.pi/2)
    ax_polar_all.set_ylim([0, 1])
    #only plot visually responsive neurons that have been somewhat barrcoded
    barred = set(stim_responses.index[
                     stim_responses[[c for c in stim_responses.columns if c.startswith("barred_")]].any(axis=1)])
    for plane in fish.planes:
        #plot rainbow plot
        ax_imshow = fig.add_subplot(gs[0, plane])
        ax_imshow.imshow(fish.img_dict[plane], cmap = 'gray', origin = 'lower')
        #get neurons in this plane
        plane_n = list(set(fish.pos_dict[plane].index) & barred)
        #color: tuned angles
        colors = stim_responses.loc[plane_n, 'dir']
        colors = [utils.angle_to_rgba(c, 1, 1) for c in colors]
        #alpha: dsi
        alphas = stim_responses.loc[plane_n, 'DSI2']
        alphas = [i if not np.isnan(i) else 0 for i in alphas]
        ax_imshow.scatter(fish.pos_dict[plane].loc[plane_n, 'xpos'], fish.pos_dict[plane].loc[plane_n, 'ypos'], c = colors, s = .1, alpha = np.clip(alphas, 0, 1))
        ax_imshow_all.scatter(fish.pos_dict[plane].loc[plane_n, 'xpos'], fish.pos_dict[plane].loc[plane_n, 'ypos'], c = colors, s = .1, alpha = np.clip(alphas, 0, 1))
        ax_imshow.set_axis_off()


        #plot vector plot
        ax_polar = fig.add_subplot(gs[1, plane], polar = True)
        ax_polar.set_theta_offset(np.pi/2)
        dirs_rad = np.deg2rad(stim_responses.loc[plane_n, 'dir'].values)
        radii = stim_responses.loc[plane_n, 'DSI2'].values
        ax_polar.scatter(dirs_rad, radii, c=colors, alpha=0.7, s=1)
        ax_polar_all.scatter(dirs_rad, radii, c=colors, alpha=0.7, s=.1)
        ax_polar.set_ylim([0, 1])
    #plot it across all neurons
    ax_imshow_all.imshow(fish.img_dict[plane], cmap = 'gray', origin = 'lower')
    ax_imshow_all.set_axis_off()
    return fig

def select_overlapbarcode(fish, f_pertrial_dict, baseline_s =10, response_s = 10, perc_trial_threshold = 0.8):
    """Select barcoded neurons with 1.8std above baseline and below 1.8std for the opposite stimulus"""
    #TODO: replace xpos with rois
    #build a neuron accumulator
    stim_list = fish.stimulus_df.stim_name.unique()
    overlap_stim = [s for s in stim_list if ('[' in s) and ('pause' not in s)]
    all_n_index = []
    for plane in fish.planes:
        all_n_index = all_n_index + list(fish.f_dict[plane].index)
    barcode = pd.DataFrame(columns =overlap_stim, index = all_n_index)
    for stim in overlap_stim:
        stim_df = fish.stimulus_df[fish.stimulus_df['stim_name'] == stim].reset_index(drop=True)
        stationary_time = stim_df['stationary_time'].iloc[0][1]
        print(f'baseline starts at t0 = {stationary_time}')
        #define plotting range (what are the frames of data to grab in general)
        stationary_frame = int(stationary_time/fish.image_s)
        baseline_frame = int((stationary_time - baseline_s)/ fish.image_s)#how many frame before stationary frame
        response_frame = int((stationary_time + response_s)/ fish.image_s)#how many frame after stationary frame
        #get all the baseline per trial
        f_pertrial = f_pertrial_dict[stim]
        f_pertrial_baseline = f_pertrial[:, :, baseline_frame:stationary_frame]
        f_pertrial_response = f_pertrial[:, :, stationary_frame:response_frame]
        f_pertrial_baselinemean = np.mean(f_pertrial_baseline, axis = 2)
        f_pertrial_baselinestd = np.std(f_pertrial_baseline, axis = 2)
        f_pertrial_responsemax = np.max(f_pertrial_response, axis = 2)
        #if response > baselien + 1.8*std
        boundary = 1.8
        print(f"selection criteria: max >= baseline max * {boundary}")
        f_responding = (f_pertrial_responsemax  - f_pertrial_baselinemean)/f_pertrial_baselinestd >= boundary
        f_responding_percentage = np.divide(np.sum(f_responding, axis = 0), f_responding.shape[0])
        barcode.loc[:, stim] = f_responding_percentage
    #actually picking barcode neurons
    barred_neurons = {s: None for s in overlap_stim}
    for stim in overlap_stim:
        barred_neurons[stim] = list(barcode[(barcode[stim] >= perc_trial_threshold)].index)
    return barred_neurons

def sort_overlapneurons(fish, f_pertrial_dict, barred_overlapneurons):
    """
    Sort the grating responding neurons by how early they response to dot across trials
    """
    stim_list = fish.stimulus_df.stim_name.unique()
    overlap_stim = [s for s in stim_list if ('[' in s) and ('pause' not in s)]
    for stim in overlap_stim:
        stationary_s = fish.stimulus_df[fish.stimulus_df.stim_name == stim].iloc[0].stationary_time[0]#grating stim: 20s
        print(f'sorted from t0 = {stationary_s}')
        stationary_frame = int(stationary_s/fish.image_s)
        f_avg = f_pertrial_dict[stim][:, barred_overlapneurons[stim], stationary_frame:].mean(axis = 0)#avg across trial, only looks at nonstationary period
        peak_frame = np.argmax(f_avg, axis = 1)#find max frame for each neuron
        sort_i = np.argsort(peak_frame)
        barred_overlapneurons[stim] = [barred_overlapneurons[stim][i] for i in sort_i]
    return barred_overlapneurons

def plot_overlapneurons(fish, f_pertrial_dict, barred_overlapneurons):
    stim_list = fish.stimulus_df.stim_name.unique()
    overlap_stim = [s for s in stim_list if ('[' in s) and ('pause' not in s)]
    overlap_stim = np.sort(overlap_stim)
    fig, ax = plt.subplots(3, len(overlap_stim), figsize = (20, 8))
    for n, stim in enumerate(overlap_stim):
        ax[0, n].imshow(fish.img_dict[3], origin='lower', cmap='Greys_r')
        ax[0, n].scatter(fish.pos_all.loc[barred_overlapneurons[stim], 'xpos'],
                         fish.pos_all.loc[barred_overlapneurons[stim], 'ypos'], c=range(len(barred_overlapneurons[stim])),
                         cmap='plasma', s=.3)
        ax[0, n].grid(False)
        ax[0, n].set_xticks([])
        ax[0, n].set_yticks([])
        ax[0, n].set_title(stim)

        for offset, plane in enumerate(fish.planes[::-1]):
            n_plane = np.where(fish.pos_all.loc[barred_overlapneurons[stim], 'zpos'] == plane)
            counts, bins = np.histogram(fish.pos_all.loc[n_plane]['ypos'], bins=10, range = [0, fish.img_dict[0].shape[0]])
            perc = [i/len(barred_overlapneurons[stim]) for i in counts]
            ax[1, n].bar(bins[:-1], perc, width=np.diff(bins), align='edge', alpha=0.7, bottom=offset * .05)

        ax[1, n].grid(False)
        ax[1, n].set_xticks([0, fish.img_dict[0].shape[0]])
        ax[1, n].set_xticklabels(['P', 'A'])
        ax[1, n].set_yticks([0, len(fish.planes) * .05])
        ax[1, n].set_yticklabels(['V', 'D'])

        sns.heatmap(f_pertrial_dict[stim][:, barred_overlapneurons[stim], :].mean(axis=0), cmap='viridis', ax=ax[2, n],
                    xticklabels=[], yticklabels=[], vmin = 0, vmax = 0.75)


    return fig