
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pingouin as pg
import numpy as np
import math
import scipy
import sys
sys.path.append(r'C:\Users\Kaitlyn\PyCharmProjects\imaging\caImageAnalysis')
import constants, angles
from utilities import arrutils

## PREPROCESSING ##
omr_stim = ['forward', 'right', 'backward', 'left']
response_s = 10

# gather data so that for each neuron, I can get the tuned direction and tuned strength
# then also gathering the location of these neurons, and the specific responses to each of these stimuli

def get_volume_tuning_and_locations(fishvol, omr_stim, response_s = 10):
    """
    Function to get the tuning of neurons in a volume fish to OMR stimuli.
    
    Parameters:
    - fishvol: VizStimVolume object containing the fish data. (entire volume)
    - omr_stim: List of OMR stimuli to analyze.
    - response_s: Duration of the response to analyze after the stimulus onset.
    
    Returns:
    - volume_tuned_df: DataFrame containing tuning values for each neuron.
    - volume_locations_df: DataFrame containing neuron locations and IDs.
    - volume_extended_array_df: DataFrame containing extended responses for plotting.
    """

    column_ids = omr_stim + ['weighted_angle', 'weights']
    # stim_angles = [constants.deg_dict[stim] for stim in omr_stim]

    volume_tuned_df_lst = []
    volume_locations_df_lst = []
    volume_extended_array_df_lst = []

    for plane, one_fish in enumerate(fishvol):
        x_midline = one_fish.return_x_midline()
        one_fish_tuned_df = pd.DataFrame(columns = column_ids, index = np.arange(0, len(one_fish.f_cells)))
        one_fish_locations_df = pd.DataFrame(columns = ['plane', 'neur_id', 'coords', 'pt', 'side'], index = np.arange(0, len(one_fish.f_cells))) # for getting coordinates for gpl file
        one_fish_extended_array_df = pd.DataFrame(columns = omr_stim, index = np.arange(0, len(one_fish.f_cells))) # for plotting traces later
        array_df = pd.DataFrame(one_fish.extended_responses_normf)
        rois = one_fish.return_cell_rois(np.arange(0, len(one_fish.f_cells)))
        pt_neurons = one_fish.return_cells_by_saved_roi('Pt')
        for n in range(len(one_fish.f_cells)):
            for stim in omr_stim:
                trials = array_df.iloc[n][stim]
                all_trial_responses = np.zeros(shape=(len(trials), 1))
                for t, v in enumerate(trials):
                    trial_baseline_mean = v[:-one_fish.offsets[0]].mean()
                    trial_baseline_std = v[:-one_fish.offsets[0]].std()
                    trial_response_mean = v[-one_fish.offsets[0]:-one_fish.offsets[0]+ int(response_s*one_fish.img_hz)].mean()
                    trial_response_max = v[-one_fish.offsets[0]:-one_fish.offsets[0]+ int(response_s*one_fish.img_hz)].max()
                    trial_response_tuning_value = (trial_response_mean - trial_baseline_mean) / trial_baseline_std
                    trial_response_tuning_value_max = (trial_response_max - trial_baseline_mean) / trial_baseline_std
                    all_trial_responses[t] = trial_response_tuning_value
                response_overall = all_trial_responses.mean(axis=0)[0]
                one_fish_tuned_df.iloc[n][stim] = response_overall
                one_fish_extended_array_df.iloc[n][stim] = {'mean': np.nanmean(trials, axis = 0),
                                                            'std': np.nanstd(trials, axis = 0)}
            one_fish_locations_df.iloc[n]['plane'] = plane
            one_fish_locations_df.iloc[n]['neur_id'] = n
            one_fish_locations_df.iloc[n]['coords'] = rois[n]
            one_fish_locations_df.iloc[n]['pt'] = True if n in pt_neurons else False
            one_fish_locations_df.iloc[n]['side'] = 'left' if rois[n][0] < x_midline else 'right'

            # weights = np.nanmax(list(one_fish_tuned_df.iloc[n])[:len(omr_stim)])
            # one_fish_tuned_df.iloc[n]['weighted_angle'] = angles.weighted_mean_angle(stim_angles, list(one_fish_tuned_df.iloc[n])[:len(omr_stim)])
            # one_fish_tuned_df.iloc[n]['weights'] = weights
        volume_tuned_df_lst.append(one_fish_tuned_df)
        volume_locations_df_lst.append(one_fish_locations_df)
        volume_extended_array_df_lst.append(one_fish_extended_array_df)

    volume_tuned_df = pd.concat(volume_tuned_df_lst, axis=0).reset_index(drop=True)
    volume_locations_df = pd.concat(volume_locations_df_lst, axis=0).reset_index(drop=True)
    volume_extended_array_df = pd.concat(volume_extended_array_df_lst, axis=0).reset_index(drop=True)

    return volume_tuned_df, volume_locations_df, volume_extended_array_df

# using cleo's method to get top responders
def cleo_top_responders(specific_stim, volume_tuned_df, volume_location_df, neuron_top_n = 100, neuron_target_n = 30, angle_range = 45):
    """
    Function to get top responders based on the weighted angle and tuning of the whole volume
    
    Parameters:
    - omr_stim: The specific OMR stimulus to analyze.
    - volume_tuned_df: DataFrame containing tuning values for each neuron.
    - volume_location_df: DataFrame containing neuron coodinates and IDs.
    - neuron_top_n: Number of top neurons to consider.
    - neuron_target_n: Number of target neurons to select from the top responders.
    - angle_range: Range of angles to consider for tuning (in degrees).
    
    Returns:
    - neurons_topresponders: DataFrame of top responder neurons.
    """
    # criteria 1: only located in the Pt & same side as the omr stimulus
    pt_df = volume_location_df[volume_location_df['pt'] == True]
    if specific_stim == 'left':
        pt_df = pt_df[pt_df['side'] == 'left']
    if specific_stim == 'right':
        pt_df = pt_df[pt_df['side'] == 'right']
    pt_neurons = pt_df.index
    pt_vol_tuned_df = volume_tuned_df.loc[pt_neurons, :]
    
    #criteria 2: only tuned towards the correct stimuli (within a certain angle range)
    anglestim = constants.deg_dict[specific_stim]
    entire_angle_range = (anglestim - angle_range, anglestim + angle_range)
    neurons_topresponders = pt_vol_tuned_df[angles.is_within_short_arc(pt_vol_tuned_df['weighted_angle'], 
                                                                       entire_angle_range[0], entire_angle_range[1])].index
    
    #criteria 3: gather specific number of neurons within the top responders
    neurons_topresponders = pt_vol_tuned_df.loc[neurons_topresponders, :].sort_values(by=specific_stim, ascending=False)[:neuron_top_n]
    neurons_topresponders = neurons_topresponders.loc[np.random.choice(neurons_topresponders.index, neuron_target_n, replace=False), :]
    
    return neurons_topresponders

# location and polar plot of the top responders
def plot_top_responders(top_responder_index, vol_tuned_df, vol_locations_df, image):
    """
    Function to plot the top responders in a volume fish.
    
    Parameters:
    - top_responders_index: List of indices for the top responder neurons.
    - vol_tuned_df: DataFrame containing tuning values for each neuron.
    - vol_locations_df: DataFrame containing neuron locations and IDs.
    - image: Image to overlay the neuron locations on.
    
    Returns:
    - None
    """
    # get all the data
    vol_tuned_df_top_responders = vol_tuned_df.loc[top_responder_index, :]
    top_responder_coords = vol_locations_df.loc[top_responder_index, 'coords'].values

    theta =  [math.radians(a) for a in vol_tuned_df_top_responders['weighted_angle'].values]
    weights = list(vol_tuned_df_top_responders['weights'].values)
    norm_weights = ([min(i, np.percentile(weights, 100)) for i in weights] - 
                    np.percentile(weights,1))/(np.percentile(weights, 100) - np.percentile(weights,1))
    norm_weights[norm_weights<0] = 0
    tuned_color_sum = [angles.angle_to_rgba(i, saturation = 1, alpha = 1) for i in vol_tuned_df_top_responders['weighted_angle'].values]

    fig = plt.figure(figsize=(12, 8))
    gs = matplotlib.gridspec.GridSpec(2, 4, height_ratios=[1, 1])  # 2 rows, 4 columns

    ax1 = fig.add_subplot(gs[0, 0:2])
    im = ax1.imshow(image, cmap='gray', vmax = np.percentile(image, 99))
    ax1.set_axis_off()
    for i, coords in enumerate(top_responder_coords):
        ax1.scatter(x=coords[0], y=coords[1], s=30, marker='o', linewidth=0, color=tuned_color_sum[i], alpha=0.9)

    ax2 = fig.add_subplot(gs[0, 2:], polar=True)
    ax2.vlines(x = theta, ymin = [0] * len(norm_weights), ymax = norm_weights, 
            alpha = norm_weights, color = tuned_color_sum, linewidth = 0.8)
    ax2.set_xticks(np.deg2rad([0, 90, 180, 270])) 
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction('clockwise')
    ax2.set_ylim([0, 1])
    ax2.set_yticks([0, 0.5, 1])
    plt.show()


def plot_top_responder_traces(top_responder_index, vol_responses_df, response_s = 10, stim_offset = 10, 
                              img_hz = 1.03):
    """
    Function to plot the response traces of top responder neurons.
    
    Parameters:
    - top_responder_index: List of indices for the top responder neurons.
    - vol_responses_df: DataFrame containing response (extended array) data for each neuron.
    - response_s: Duration of the response to analyze after the stimulus onset.
    
    Returns:
    - None
    """
    duration_stim_on = int(response_s / img_hz)

    for n in top_responder_index:
        fig, ax = plt.subplots(1, 4, figsize=(12, 2))
        for i in range(len(vol_responses_df.columns)):
            clr = constants.monocular_dict[vol_responses_df.iloc[0].index[i]]
            ax[i].plot(vol_responses_df.iloc[n][i]['mean'], color = 'k')
            ax[i].fill_between(np.arange(len(vol_responses_df.iloc[0][i]['mean'])), 
                            vol_responses_df.iloc[n][i]['mean'] - vol_responses_df.iloc[n][i]['std'], 
                            vol_responses_df.iloc[n][i]['mean'] + vol_responses_df.iloc[n][i]['std'], 
                            alpha=0.2, color = 'gray')
            ax[i].set_title(f'ind {n}, {vol_responses_df.columns[i]}')
            ax[i].axvspan(stim_offset, stim_offset + duration_stim_on, color=clr, alpha=0.2)
            ax[i].set_ylim([-0.01, 1])
            if i > 0:
                ax[i].set_yticklabels([])
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['top'].set_visible(False)
        plt.show()

## POST PROCESSING ##

# gathering data functions
def get_photostimulated_cell_response_arrays(full_f_trace, ps_event_lst, frame_window = (-10, 10)):
    '''
    get photostimulated cell response array for one cell

    full_f_trace: array of full fluorescence trace for one cell
    ps_event_lst: list of photostimulation events (frames)
    frame_window: tuple of frames before and after photostimulation event to include in the response array
    '''
    frame_subset = arrutils.subsection_arrays(ps_event_lst, frame_window)
    if frame_subset[-1][-1] > len(full_f_trace): # adjust frames in case this is out of range
        new_frame_subset = []
        for s in frame_subset:
            new_frame_subset.append([q for q in s if q < len(full_f_trace)])
        frame_subset = np.array(new_frame_subset)
    subset_f_trace = full_f_trace[frame_subset]

    return subset_f_trace

def make_stim_ensemble_df(fishvol, vol_stim_sites_df, framerange=(-10, 7)):
    #gather all the population responses of the stimulated cells for plotting and stats

    ensemble_response = [fishvol[plane].normcells[[int(i) for i in fishvol[plane].stimmed_cell_id_array]] 
                        for plane in vol_stim_sites_df.plane.unique() if len(fishvol[plane].stimmed_cell_id_array) > 0]
    ensemble_response = np.concatenate(ensemble_response, axis=0)
    ensemble_badframes = [fishvol[plane].ps_event_start for plane in vol_stim_sites_df.plane.values 
                        if len(fishvol[plane].stimmed_cell_id_array) > 0]

    framerange_lst = np.arange(framerange[0], framerange[1]) 
    ensemble_response_subsets = []
    for e, single_cell_trace in enumerate(ensemble_response):
        single_cell_trace_subset = get_photostimulated_cell_response_arrays(single_cell_trace,
                                            ensemble_badframes[e],
                                            frame_window=framerange)
        ensemble_response_subsets.append(single_cell_trace_subset)

    ensemble_aroundstim_df = pd.DataFrame(columns = ['ps_event', 'neuron', 'frame', 'norm_f'])
    for n_neuron, each_neuron in enumerate(ensemble_response_subsets):
        for each_event, each_trace in enumerate(each_neuron):
            for frame, f_val in enumerate(each_trace):
                ensemble_aroundstim_df = ensemble_aroundstim_df.append({'ps_event': int(each_event), # individual photostimulation event
                                                                        'neuron': int(n_neuron), # photostimulated neuron in ensemble
                                                                        'frame': int(framerange_lst[frame]), # relative frame to photostimulation event
                                                                        'norm_f': f_val}, # normalized fluorescence value of that neuron
                                                                    ignore_index=True)
    return ensemble_aroundstim_df

def make_responder_arrays(fishvol, framerange = (-5, 5), region = None):

    total_responders = np.sum([len(fish.f_cells) for fish in fishvol])
    if region != None:
        total_responders = np.sum([len(fish.return_cells_by_saved_roi(region)) for fish in fishvol])

    responders_aroundstim = np.full([total_responders, # every neuron
                                    len(fishvol[0].ps_event_start), # for each stimulation event
                                    -framerange[0] + framerange[1],  # frames around the stimulation event
                        ], 
                        np.nan)  #stim_index
    responder_rois = np.zeros((total_responders, 2))  # to store the locations of the responders
    responders_fulltraces = np.zeros((total_responders, len(fishvol[0].normcells[0])))  # to store the full traces of the responders

    neuron_n = 0
    for plane, fish in fishvol.volumes.items():
        cell_traces = fish.normcells
        cell_idx = range(len(cell_traces))
        if region != None:
            cell_idx = fish.return_cells_by_saved_roi(region)
            cell_traces = fish.normcells[cell_idx]
        for n, single_cell_trace in zip(cell_idx, cell_traces):
            responders_fulltraces[neuron_n] = single_cell_trace  # store the full trace of the responder
            for stim_index, stim_frame in enumerate(fish.ps_event_start):
                single_cell_trace_subset = get_photostimulated_cell_response_arrays(single_cell_trace,
                                                                                    [stim_frame],
                                                                                    frame_window= framerange)
                location = fish.return_singlecell_rois(n)
                responder_rois[neuron_n] = location
                responders_aroundstim[neuron_n, stim_index, :] = single_cell_trace_subset
            neuron_n += 1
    
    return responders_aroundstim, responder_rois, responders_fulltraces

# plotting functions
def plot_stimsites_raw(fishvol, vol_stim_sites_df, saving = True):

    fig, ax = plt.subplots(len(vol_stim_sites_df.plane.unique()), 2, figsize=(25, 15), gridspec_kw={
        "height_ratios": [len(vol_stim_sites_df[vol_stim_sites_df['plane'] == plane]) / 
                        np.sum([len(vol_stim_sites_df[vol_stim_sites_df['plane'] == plane]) for plane in vol_stim_sites_df.plane.unique()]) 
                        for plane in vol_stim_sites_df.plane.unique()], 
        "width_ratios": [1, 2]})
    #plot stim site
    for row,_ in enumerate(vol_stim_sites_df.plane.unique()):
        fig.delaxes(ax[row, 0])
    gs = fig.add_gridspec(len(vol_stim_sites_df.plane.unique()), 2, width_ratios=[1, 2])
    imshow_rows = len(vol_stim_sites_df.plane.unique())//2 + 1
    ax_imshow = fig.add_subplot(gs[0:imshow_rows, 0])
    ax_imshow.imshow(fishvol[2].ops['refImg'], cmap="gray")
    #scatter across plane
    for plane in vol_stim_sites_df.plane.unique():
        plane_rois = np.array(fishvol[plane].return_cell_rois(np.arange(len(fishvol[plane].normcells))))
        ax_imshow.scatter(plane_rois[:,0], plane_rois[:,1], color='grey', s=5)
        plane_stim_rois = np.array(fishvol[plane].stimmed_cell_coords)
        ax_imshow.scatter(plane_stim_rois[:,0], plane_stim_rois[:,1], color='red', s=10, label=plane)
    ax_imshow.axis(False)
    #plot heatmap
    for plane_index, plane in enumerate(vol_stim_sites_df.plane.unique()):
        ax_heatmap = ax[plane_index, 1]
        ax_heatmap.set_title(plane)
        plane_stim_cells = [int(i) for i in fishvol[plane].stimmed_cell_id_array]
        if len(plane_stim_cells) > 0:  # there are stimulated cells
            plane_stim_cells_traces = fishvol[plane].normcells[plane_stim_cells]
            sns.heatmap(plane_stim_cells_traces, ax=ax_heatmap, cmap="gray", xticklabels=False, yticklabels=False, cbar=False)
            [ax_heatmap.axvline(frame, color='red', linewidth=2) for frame in fishvol[plane].ps_event_start]
        ax_heatmap.axis(False)

    #plot populat response overall
    ensemble_response = [fishvol[plane].normcells[[int(i) for i in fishvol[plane].stimmed_cell_id_array]] 
                        for plane in vol_stim_sites_df.plane.unique() if len(fishvol[plane].stimmed_cell_id_array) > 0]
    ensemble_response = np.concatenate(ensemble_response, axis=0)
    ax_lineplot = fig.add_subplot(gs[imshow_rows:, 0])
    ax_lineplot.plot(ensemble_response.mean(axis = 0))
    mean = ensemble_response.mean(axis = 0)
    sem = scipy.stats.sem(ensemble_response)
    interval = sem * scipy.stats.t.ppf((1 + .95) / 2, ensemble_response.shape[0] - 1)
    ax_lineplot.fill_between(range(len(mean)), np.subtract(mean, interval), np.add(mean, interval), alpha=0.5)
    [ax_lineplot.axvline(frame, color='red', linewidth=2) for frame in fishvol[0].ps_event_start]
    ax_lineplot.set_ylabel('pop. F (mean±sem)')
    ax_lineplot.set_xlabel('frame')

    plt.rcParams.update({ "figure.facecolor": (1.0, 1.0, 1.0, 1),  "axes.facecolor": (0.0, 1.0, 0.0, 0)})

    if saving:
        save_path_directory = Path(fishvol[0].folder_path.parents[1]).joinpath('graphs')
        if not save_path_directory.exists():
            save_path_directory.mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(save_path_directory).joinpath('stimsites_raw.png'))
    plt.show()


def plot_and_get_stimsites_stats(stim_ensemble_df, framerange = (-10, 7), baseline_frame_min_show = -5,
                                 f_col_name = 'norm_f',
                                 saving = True, save_path_directory = None):
    cmap = plt.get_cmap('gnuplot')
    effectsizes_per_poststim_frame_dict = {} # keys are each ps event, values are the effect sizes for each of the post-stim frame
    framerange_lst = np.arange(framerange[0], framerange[1])
    fig, ax = plt.subplots(len(range(baseline_frame_min_show, framerange[1])), len(stim_ensemble_df['ps_event'].unique()), figsize = (30, 10),
                            gridspec_kw={'wspace': 0.1, 'hspace': -.4})
    # plot the distribution of fluorescence values for each frame, for each photostimulation event of the photostimulated ensemble
    for ps_event_index in stim_ensemble_df['ps_event'].unique():
        ps_event_index = int(ps_event_index)  # ensure ps_event_index is an integer
        one_event_df = stim_ensemble_df[stim_ensemble_df['ps_event'] == ps_event_index]  # example for one event
        for frame_index, frame in enumerate(range(baseline_frame_min_show, framerange[1])):
            color = cmap(frame_index / (-baseline_frame_min_show + framerange[1])) 
            sns.kdeplot(data=one_event_df[one_event_df['frame'] == frame][f_col_name].values,
                        color=color, ax = ax[frame_index, ps_event_index])
            sns.histplot(one_event_df[one_event_df['frame'] == frame][f_col_name].values, color=color, alpha=0.3,
                            bins=25, linewidth=0, binrange=(0, 1), stat='density', ax = ax[frame_index, ps_event_index])
            ax[frame_index, ps_event_index].axvline(np.median(one_event_df[one_event_df['frame'] == frame][f_col_name].values),
                                                    .5, .9, color='black',linewidth=3)
            ax[frame_index, ps_event_index].set_xlim([-0.1, 1])
            ax[frame_index, ps_event_index].set_ylim([-5, 5])
            ax[frame_index, ps_event_index].grid(False)
            ax[frame_index, ps_event_index].spines[['bottom', 'left', 'top', 'right']].set_visible(False)
            ax[frame_index, ps_event_index].set_axis_off()
            if frame < 0: # if a baseline frame, then this is labeled with a gray bar
                ax[frame_index, ps_event_index].axvline(-0.1, .5, .72, color='gainsboro', linewidth=10, zorder=-1, solid_capstyle='butt')
            if ps_event_index == 0:
                ax[frame_index, 0].text(0.02, 0.9, f'f:{frame}', transform=ax[frame_index, 0].transAxes,
                                            fontsize=10, va='top', ha='left',
                                            bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=0.6))
        ax[0, ps_event_index].set_title(f'event {ps_event_index}')

        #for each stimulation, test what is the post-stimulation significant frames compared to the baseline (pre-stimulation 10 frames)
        #compare everything to the one frame before stimuli
        baseline = one_event_df[one_event_df['frame'] < 0].groupby('neuron')[f_col_name].mean()  #baseline: anything below 0
        p_values = []
        effectsizes = []
        #@chatgpt
        poststim_frames = [f for f in framerange_lst if f > 0]
        for frame in poststim_frames:
            response = one_event_df[one_event_df['frame'] == frame].groupby('neuron')[f_col_name].mean()  #one neuron is repeating, so grouping everyone here
            result = pg.wilcoxon(response, baseline)
            p_val = result.loc['Wilcoxon', 'p-val']
            effectsize = result.loc['Wilcoxon', 'CLES']
            p_values.append(p_val)
            effectsizes.append(effectsize - 0.5)  #correct the CLES: since 50% probability = response ~= baseline
        # Apply Holm-Bonferroni correction with a variation (go from the temporal sequence of frames occuring)
        rejected = []
        for i, p_val in enumerate(p_values):
            if p_val < .05 / (len(p_values) - i):  # Holm-Bonferroni correction
                rejected.append(True)
            else:
                rejected.append(False)
        # Find where to stop (first non-significant test)
        first_non_significant = poststim_frames[next((i for i, reject in enumerate(rejected) if not reject), len(rejected) - 1)]
        #plot significant frame ==> the larger the spot size, the larger effect size
        for frame_index in poststim_frames[:first_non_significant]:
            color = cmap((frame_index - framerange[0] - 1) / (-framerange[0] + framerange[1]))
            ax[frame_index - baseline_frame_min_show - 1, ps_event_index].scatter([.8], [.5], s=5 ** (10 * effectsizes[frame_index - 1]),
                                                                    color=color, alpha=0.8)
        effectsizes_per_poststim_frame_dict[ps_event_index] = effectsizes # gives the effect size for each poststim frame in the dataframe (i.e. anything above relative frame 0)
    
    if saving:
        plt.savefig(Path(save_path_directory).joinpath('stimsites_stats.png'))
    plt.show()  
    
    return effectsizes_per_poststim_frame_dict


def calculate_weighted_response(effectsize_dictionary, reponder_traces_aroundstim, post_stim_frame =5, photostim_window=(-10, 7)):
    stim_intensity = [np.mean(effectsize_dictionary[ps_event_index]) for ps_event_index in effectsize_dictionary.keys()]
    stim_intensity = np.multiply(stim_intensity, 100)
    baseline = np.mean(reponder_traces_aroundstim[:, :, :-photostim_window[0]],
                       axis=2)  # first axis: neuron, second axis: stimulation event, 3rd: frames around the stimulation event
    response = np.mean(reponder_traces_aroundstim[:, :, -photostim_window[0]:-photostim_window[0] + post_stim_frame],
                       axis=2)
    response = np.subtract(response, baseline)  # response = response - baseline
    weighted_response = np.mean(np.multiply(response, stim_intensity), axis=1)

    return weighted_response

def plot_responders_raw(fishvol, effectsize_dictionary, stim_sites_vol_df, post_stim_frame = 5, framerange = (-5, 5), saving = True):
    
    stim_intensity =[np.mean(effectsize_dictionary[ps_event_index]) for ps_event_index in effectsize_dictionary.keys()]
    stim_intensity = np.multiply(stim_intensity, 100)

    responders_aroundstim, responder_rois, responders_fulltraces = make_responder_arrays(fishvol, framerange=framerange)

    baseline = np.mean(responders_aroundstim[:, :, :-framerange[0]],
                       axis=2)  # first axis: neuron, second axis: stimulation event, 3rd: frames around the stimulation event
    response = np.mean(responders_aroundstim[:, :, -framerange[0]:-framerange[0] + post_stim_frame], axis=2)
    response = np.subtract(response, baseline)  # response = response - baseline
    weighted_response = np.mean(np.multiply(response, stim_intensity), axis=1)

    fig, ax = plt.subplots(2, 2, figsize=(15, 15), gridspec_kw={"width_ratios": [1, 1]})
    #plot response site
    ax_imshow = ax[0, 0]
    ax_imshow.imshow(fishvol[2].ops['refImg'], cmap='gray')
    max_weighted_response = np.percentile(np.abs(weighted_response), 95)
    ax_imshow.scatter(responder_rois[:, 0], responder_rois[:, 1], c=weighted_response, cmap='bwr', s=3, vmin=-max_weighted_response, vmax=max_weighted_response)
    ax_imshow.scatter(stim_sites_vol_df.x_stim.values, stim_sites_vol_df.y_stim.values, color='yellow', s=10) 
    ax_imshow.axis(False)

    #sort weighted response
    weighted_response_sortindex = np.argsort(weighted_response)[::-1]
    responders_fulltraces_sorted = responders_fulltraces[weighted_response_sortindex]

    #plot heatmap
    ax_heatmap = ax[0, 1]
    sns.heatmap(responders_fulltraces_sorted, ax=ax_heatmap, cmap='gray',
                xticklabels=False, yticklabels=False, cbar=False)
    [ax_heatmap.axvline(badframe, color='red', linewidth=2)for badframe in fishvol[0].ps_event_start]
    ax_heatmap.axis(False)

    #plot lineplot
    ax_lineplot = ax[1, 1]
    for i, f in enumerate(responders_fulltraces_sorted[:20, :]):
        ax_lineplot.plot(np.add(f, 1 * i), color='grey', linewidth=1)
    [ax_lineplot.axvline(badframe, color='red', linewidth=2)for badframe in fishvol[0].ps_event_start]
    ax_lineplot.axis(False)

    #plot population response overall
    ax_lineplot = ax[1, 0]
    mean = responders_fulltraces_sorted.mean(axis= 0)
    sem = scipy.stats.sem(responders_fulltraces_sorted, axis= 0)
    interval = sem * scipy.stats.t.ppf((1 + .95) / 2, responders_fulltraces_sorted.shape[0] - 1)
    ax_lineplot.plot(mean, color='black', linewidth=2)
    ax_lineplot.fill_between(range(len(mean)), np.subtract(mean, interval), np.add(mean, interval), color = 'k', alpha=0.7)
    [ax_lineplot.axvline(badframe, color='red', linewidth=2)for badframe in fishvol[0].ps_event_start]
    ax_lineplot.set_ylabel('pop. F (mean±sem)')
    ax_lineplot.set_xlabel('frame')
    if saving:
        save_path_directory = Path(fishvol[0].folder_path.parents[1]).joinpath('graphs')
        if not save_path_directory.exists():
            save_path_directory.mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(save_path_directory).joinpath('responders_raw.png'))
    plt.show()

