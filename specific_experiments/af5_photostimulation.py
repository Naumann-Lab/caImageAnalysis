
from pathlib import Path
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pingouin as pg
import warnings
import numpy as np
import math
import scipy
from scipy.ndimage import gaussian_filter1d
import sys
sys.path.append(r'C:\Users\Kaitlyn\PyCharmProjects\imaging\caImageAnalysis')
import bruker_images, process, tailtracking, stimuli, photostim_data_pipeline
from fishy import BaseFish, VizStimVolume, WorkingFish, PhotostimFish
from utilities import arrutils, coordutils, plotutils

## some constant stimuli, barcode dict information ##
af5_stims = ['forward', 'backward', 'converging', 'diverging',
             'medial_left', 'lateral_left', 'medial_right', 'lateral_right']  # monocular cues

id_dict = {'direction_up_L': np.array([1, 0, 1, 0, 1, 0, 0, 0]),
           'direction_down_L': np.array([0, 1, 0, 1, 0, 0, 0, 1]),
           'direction_up_R': np.array([1, 0, 1, 0, 0, 0, 1, 0]),
           'direction_down_R': np.array([0, 1, 0, 1, 0, 1, 0, 0])}

def preprocess_omr_stack(omr_tseries_path, tail_data_path, suite2p_params = None):
    '''
    Preprocess the OMR stack
    :param omr_tseries_path: path to data folder
    :param tail_data_path: tail data folder path
    :param suite2p_params: the parameter dictionary if i want to change something
    :return:
    '''
    # organize images into an output folder
    if not omr_tseries_path.joinpath('output_folders').exists():
        with warnings.catch_warnings(record=True):
            print('organizing images')
            bruker_images.bruker_img_organization(omr_tseries_path, testkey='Cycle', single_plane=False, pstim_file=True)

    paths = {}
    with os.scandir(Path(omr_tseries_path).joinpath('output_folders')) as entries:
        for entry in entries:
            if os.path.isdir(entry.path):
                paths[entry.name] = entry.path

    print('processing tail data')
    if not Path(tail_data_path).joinpath('tail_df.h5').exists():
        tailtracking.tail_df_creator(Path(tail_data_path))
        master_tail_df_path = Path(tail_data_path).joinpath('tail_df.h5')
        for p in paths.keys():
            shutil.copy(master_tail_df_path, Path(paths[p]).joinpath('tail_df.h5'))

    # these suite2p params have been working well for this FOV
    if suite2p_params == None:
        suite2p_params = {'preclassify': 0.1, 'threshold_scaling': 1,
                      'max_overlap': 0.85}
    print(f'suite 2p params: {suite2p_params}')
    for p in paths.keys():
        print('processing {}'.format(p))
        baseFish = BaseFish(folder_path=paths[p], frametimes_key='frametimes')
        if 'rotated_image' not in baseFish.data_paths.keys():
            process.run_image_rotation(baseFish, angle=90, crop=0)
        if 'suite2p' not in baseFish.data_paths.keys():
            baseFish = BaseFish(folder_path=paths[p], frametimes_key='frametimes')
            process.run_suite2p(baseFish, custom_parameter_dict=suite2p_params, force=True)
        baseFish = BaseFish(folder_path=paths[p], frametimes_key='frametimes')
        baseFish.draw_roi2('brain', overwrite=True)

    return print('preprocess complete')

def create_fishvolume(omr_tseries_path):
    '''
    creating the fishvolume, parameters are hard coded here just since I know they will work for this specific dataset
    :param omr_tseries_path: path to data folder
    :return: a fishvolume
    '''

    paths = {}
    with os.scandir(Path(omr_tseries_path).joinpath('output_folders')) as entries:
        for entry in entries:
            if os.path.isdir(entry.path):
                paths[entry.name] = entry.path
    paths = dict(sorted(paths.items(),
            key=lambda x: int(x[0].replace("plane_", ""))))

    cardinal_dirs = ['forward', 'right', 'left', 'backward']
    fishvolume = VizStimVolume()
    for p in paths.keys():
        afish = WorkingFish(
            folder_path=paths[p],
            frametimes_key="frametimes",
            stim_key="pstim",
            tail_key="tail_df",
            stim_fxn=stimuli.pandastim_to_df,
            stim_order=cardinal_dirs,
            seconds_motion_is_on=7,
            stim_offset=5,
            used_offsets=(-6, 12),  # baseline = 4.6 sec vis stim started, evoked = ~9 sec
            bool_data_type='normf',
            bruker_invert=False,  # as of January 2025, no need to invert stims
            rep_mode='common',  # for any mismatching stimuli reps
            )
        fishvolume.add_volume(afish)  # add them to volume

    return fishvolume


def make_af5_rois_for_plane(plane_fishy):
    '''
    Make af5 right and left rois for one specific plane
    :param plane_fishy: a fishy object of one plane
    :return:
    '''
    plane_fishy.draw_roi2('af5_right', overwrite=True)
    plane_fishy.draw_roi2('af5_left', overwrite=True)

    return print('af regions made')


def identify_directional_neurons(fishvolume):
    '''
    identify the directional neurons in AF5 region
    :param fishvolume: the normal omr fishvolume
    :return: dataframe, with information about directional identity
    '''

    # look across the whole volume in the proposed regions (broad ROIs)
    df_lst = []
    for plane, fishy in fishvolume.volumes.items():
        if 'af5_right' in fishy.roi_dict.keys():
            af5_R_neurons = fishy.return_cells_by_saved_roi('af5_right')
            af5_L_neurons = fishy.return_cells_by_saved_roi('af5_left')
            af5_neurons = np.concatenate([af5_R_neurons, af5_L_neurons])
            af5_neurons_rois = fishy.return_cell_rois(af5_neurons)
            motion_resp_dict_lst = photostim_data_pipeline.gather_visual_motion_responses_for_df(fishy,
                                                                                                 cell_id_array=af5_neurons,
                                                                                                 motion_cues=af5_stims,
                                                                                                 get_df_f=True)
            for idx, a in enumerate(af5_neurons):
                if a in af5_R_neurons:
                    side = 'R'
                else:
                    side = 'L'
                side_dict = {k: id_dict[k] for k in list(id_dict.keys()) if side in k}

                # id the barcode, suppression barcode
                responses = motion_resp_dict_lst[idx]
                barcode = np.zeros(shape=(len(af5_stims)))
                supp_barcode = np.zeros(shape=(len(af5_stims)))
                for n, a_stim in enumerate(list(responses.keys())):
                    a_stim_resp = responses[a_stim]
                    evoked_mean = np.nanmean(a_stim_resp['mean'][-fishy.offsets[0]:])
                    baseline_mean = np.nanmean(a_stim_resp['mean'][:-fishy.offsets[0]])
                    baseline_std = np.nanmean(a_stim_resp['std'][:-fishy.offsets[0]])
                    if evoked_mean >= (baseline_mean + (1.8 * baseline_std)):
                        barcode[n] = 1
                    if evoked_mean < baseline_mean:
                        supp_barcode[n] = 1

                # find out the barcode
                if np.array_equal(barcode, side_dict[list(side_dict.keys())[0]]):
                    id = 'direction_up'
                    opp_barcode = side_dict[list(side_dict.keys())[1]]
                elif np.array_equal(barcode, side_dict[list(side_dict.keys())[1]]):
                    id = 'direction_down'
                    opp_barcode = side_dict[list(side_dict.keys())[0]]
                else:
                    id = 'none'
                    supp_keyword = False

                if id != 'none':
                    matching_idx = np.where((supp_barcode == 1) & (opp_barcode == 1))[0]
                    if len(matching_idx) > 1:
                        supp_keyword = True
                    else:
                        supp_keyword = False

                row = {'plane': plane, 'neur_id': a, 'side': side, 'coords': af5_neurons_rois[idx],
                       'id': id, 'responses': responses, 'supp': supp_keyword}
                df_lst.append(row)
        else:
            print(f'no af5 regions on {plane}')

    df = pd.DataFrame(df_lst)
    df.to_hdf(Path(fishvolume[0].folder_path.parents[1]).joinpath('af5_barcode_df.h5'), key = 'af5_barcode')

    return df

## plotting functions ##
matplotlib.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 14,
    'pdf.fonttype': 42,   # editable text in Illustrator
    'ps.fonttype': 42
})

def plot_all(fishvolume, identity_df):
    '''
    run all the plotting functions
    :param fishvolume:
    :param identity_df:
    :return:
    '''
    plot_locations_of_directional_neurons(fishvolume, identity_df)
    plot_tuning_traces(fishvolume, identity_df)

def plot_locations_of_directional_neurons(fishvolume, identity_df):
    '''
    plot the locations (rois) of all the directional af5 neurons, across entire volume
    :param fishvolume:
    :param identity_df:
    :return:
    '''
    directional_df = identity_df[identity_df.id.isin(['direction_down', 'direction_up'])]

    fig, ax = plt.subplots(1, len(fishvolume), figsize=(len(fishvolume)*4,8))

    for m, ex_fish in enumerate(fishvolume):
        sub_ax = ax[m]
        sub_ax.imshow(ex_fish.rescaled_ref, cmap='gray', vmax=np.percentile(ex_fish.rescaled_ref, 99))
        if 'af5_right' in ex_fish.roi_dict.keys():
            # plot the broad region of AF5 that I circled
            af5_right_coords = np.load(ex_fish.roi_dict['af5_right'])
            af5_left_coords = np.load(ex_fish.roi_dict['af5_left'])
            for clr, coords in zip(['tomato', 'royalblue'], [af5_right_coords, af5_left_coords]):
                sub_ax.fill(coords[:, 0], coords[:, 1], facecolor='none', alpha=1, edgecolor=clr, linewidth=1.5)

            # plot the directional cells
            plane_df = directional_df[directional_df.plane == f'plane_{m}'].reset_index(drop=True)
            for t in range(len(plane_df)):
                roi = plane_df.iloc[t].coords
                id = plane_df.iloc[t].id
                clr = 'tomato' if id == 'direction_up' else 'royalblue'
                sub_ax.scatter(roi[0], roi[1], color=clr, s=20, zorder=5)

            # plot all the other af5 neurons in light gray, just to see what was considered af5
            other_af5_neurons = np.concatenate(
                [ex_fish.return_cells_by_saved_roi('af5_right'), ex_fish.return_cells_by_saved_roi('af5_left')])
            other_af5_neurons = np.array([o for o in other_af5_neurons if o not in plane_df.neur_id.values])
            other_af5_neurons_rois = np.array(ex_fish.return_cell_rois(other_af5_neurons))
            sub_ax.scatter(other_af5_neurons_rois[:, 0], other_af5_neurons_rois[:, 1], color='lightgray', alpha=0.4,
                           s=20, zorder=3)

        sub_ax.axis('off')
        sub_ax.set_title(f'plane {m}')
    plt.tight_layout()
    plt.savefig(Path(fishvolume[0].folder_path.parents[1]).joinpath('locations_of_directional_neurons.png'))
    plt.show()

def plot_tuning_traces(fishvolume, identity_df):
    '''
    plot the tuning traces of each individual neuron in the af5 region to stimulate, save a png
    :param fishvolume:
    :param identity_df:
    :return:
    '''

    directional_df = identity_df[identity_df.id.isin(['direction_down', 'direction_up'])]
    # individual traces
    frames_motion_on = int(fishvolume[0].seconds_motion_is_on * fishvolume[0].img_hz)
    directional_df = directional_df.sort_values(by='id')
    new_saving_folder = Path(fishvolume[0].folder_path.parents[1]).joinpath('cell_tuning_traces')
    os.makedirs(new_saving_folder, exist_ok=True)

    for i in range(len(directional_df)):
        cell_id = directional_df.neur_id.iloc[i]
        side = directional_df.side.iloc[i]
        responses = directional_df.responses.iloc[i]
        suppression = directional_df.supp.iloc[i]
        panel_num = len(responses.keys())
        trace_clr = 'tomato' if directional_df.id.iloc[i] == 'direction_up' else 'royalblue'
        fig, ax = plt.subplots(1, panel_num, figsize=(panel_num * 2, 1), sharey=True)
        for a, a_stim in enumerate(list(responses.keys())):
            vizmotion_responses = responses[a_stim]
            ax[a].axhline(0, color='lightgray')
            ax[a].plot(vizmotion_responses['mean'], color=trace_clr, linewidth=2)
            ax[a].fill_between(np.arange(len(vizmotion_responses['mean'])),
                               vizmotion_responses['mean'] - vizmotion_responses['std'],
                               vizmotion_responses['mean'] + vizmotion_responses['std'],
                               color='gray', alpha=0.2)
            stimuli.flexible_stim_shader([-fishvolume[0].offsets[0]], [a_stim],
                                         frames_motion_on,
                                         label=False, subplot=ax[a], alpha=0.1)
            ax[a].set_title(a_stim)
            ax[a].set_xlim(0, len(vizmotion_responses['mean']))
            ticks = [0, -fishvolume[0].offsets[0], len(vizmotion_responses['mean'])]
            ax[a].set_xticks(ticks)
            ax[a].set_xticklabels([int((t + fishvolume[0].offsets[0]) * fishvolume[0].img_hz) for t in ticks])
            ax[a].set_xlabel('time (s)')

        ax[0].set_ylabel('dF/F')
        sns.despine()
        fig.suptitle(f'cell id: {cell_id}, supp: {suppression}', y=1.4)
        # plt.tight_layout()
        plt.savefig(Path(new_saving_folder).joinpath(f'af5_{side}_cell{cell_id}_tuning.png'), bbox_inches='tight')
        plt.show()


## photostim analysis functions ##
def create_photostim_fishvolume(folder_path):
    paths = {}
    with os.scandir(Path(folder_path).joinpath('output_folders')) as entries:
        for entry in entries:
            if os.path.isdir(entry.path):
                paths[entry.name] = entry.path

    stim_sites_volume_df = pd.read_hdf(Path(folder_path).joinpath('stim_sites_volume.h5'), key='volume_stim')
    stim_plane = int(stim_sites_volume_df.iloc[0].plane)

    stim_fishvolume = VizStimVolume()
    for k, p in enumerate(paths.keys()):
        stimmed_plane_keyword = False
        if k == stim_plane:
            stimmed_plane_keyword = True
            print(f'stimmmed plane is {k}')
        afish = PhotostimFish(folder_path=paths[p],
                              frametimes_key="frametimes",
                              stim_type_keyword='single_cell',
                              stimmed_plane=stimmed_plane_keyword,
                              tail_key='tail')
        stim_fishvolume.add_volume(afish)

    return stim_fishvolume

def get_barcode_df(omr_data_folder_path):
    '''
    Adding in a couple more columns so that it runs for the functional types dataframe
    :param omr_data_folder_path:
    :return:
    '''
    barcode_df = pd.read_hdf(omr_data_folder_path.joinpath('af5_barcode_df.h5'))
    # add these columns for the functional types df making
    barcode_df['neur_ids'] = barcode_df['neur_id']
    barcode_df['barcoding'] = barcode_df['id']
    barcode_df[barcode_df.id != 'none']

    return barcode_df

def make_functional_types_df(omr_fishvol, photostim_fishvol, barcode_df):
    # make the functional types df - have to redo after the bad frames change

    func_types_df_path = photostim_fishvol[0].folder_path.parents[1].joinpath('functional_types_df.h5')
    stim_sites_volume_df = pd.read_hdf(Path(photostim_fishvol[0].folder_path.parents[1]).joinpath('stim_sites_volume.h5'), key='volume_stim')
    stimmed_plane = int(stim_sites_volume_df.iloc[0].plane)

    if func_types_df_path.exists():
        functypesdf = pd.read_hdf(func_types_df_path)
    else:
        df = photostim_data_pipeline.build_functional_types_df(omr_fishvol,
                                                               photostim_fishvol,
                                                               barcode_df,
                                                               regions=['af5_left', 'af5_right'],
                                                               motor_correlation=False,
                                                               plot_stim_sites = False)

        midline_information = coordutils.return_midline_coords_per_plane_dict(photostim_fishvol[0].folder_path.parents[1])
        df2 = df.dropna(subset=['stim_neur_id'])
        df3 = photostim_data_pipeline.add_photostimulation_responses_to_functional_df(df2, photostim_fishvol,
                                                                                      response_window=photostim_fishvol[
                                                                                          0].photostim_frame_window,
                                                                                      trace_type='df_f')
        df4 = photostim_data_pipeline.add_photostimulation_responses_to_functional_df(df3, photostim_fishvol,
                                                                                      response_window=photostim_fishvol[
                                                                                          0].photostim_frame_window,
                                                                                      trace_type='zscore')
        df5 = photostim_data_pipeline.add_sideness_to_functional_types_df(photostim_fishvol[stimmed_plane].stim_sites_df,
                                                                          df4, midline_information)
        df5.reset_index(drop=True)
        df5.to_hdf(photostim_fishvol[0].folder_path.parents[1].joinpath('functional_types_df.h5'), key='functional_types')

        functypesdf = df5

    return functypesdf

def plot_self_success(photostim_fishvol, trace_type = 'df/f', heatmap_limits = [-1, 2],
                      save_path = None):
    '''
    plotting the self-sucess for the stimualted cell
    :param photostim_fishvol: the entire photostim fishvolume
    :param trace_type: type of data to plot (df/f or zscore)
    :param heatmap_limits: the limits for the heatmap of all the trials
    :param save_path: where to save the figure
    :return: the figure, saved
    '''

    stim_sites_volume_df = pd.read_hdf(
        Path(photostim_fishvol[0].folder_path.parents[1]).joinpath('stim_sites_volume.h5'), key='volume_stim')
    stimmed_plane = int(stim_sites_volume_df.iloc[0].plane)
    stim_fishy = photostim_fishvol[stimmed_plane]
    print(stim_fishy.folder_path)

    if trace_type == 'zscore':
        traces = stim_fishy.zdiff_cells
    else:
        traces = stim_fishy.normcells
    stim_cell_id = stim_fishy.stimmed_cell_id_array[0]
    stim_array_full_expt = traces[stim_cell_id]
    photstim_frame_subset = arrutils.subsection_arrays(stim_fishy.ps_event_start, stim_fishy.photostim_frame_window)
    stim_array_by_trial = stim_array_full_expt[photstim_frame_subset]

    if trace_type == 'df/f': # calculate the df/f with normcells
        dff_list = []
        base_means = []
        for trial in stim_array_by_trial:
            base_data = trial[:-stim_fishy.photostim_frame_window[0]]
            base_means.append(np.nanmean(base_data))
            dff_trial = (trial - np.nanmean(base_data)) / np.nanmean(base_data)
            dff_list.append(dff_trial)
        stim_array_by_trial = np.array(dff_list)
        stim_array_full_expt = (stim_array_full_expt - np.nanmean(base_means)) / np.nanmean(base_means)

    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    stim_cell_ax = ax[0, 0]
    tail_ax = ax[1, 0]
    heatmap_ax = ax[0, 1]
    avg_lineplot_ax = ax[1, 1]

    stim_cell_ax.plot(stim_array_full_expt, color='k')
    stim_cell_ax.set_ylabel(trace_type)
    for a in [stim_cell_ax, tail_ax]:
        [a.axvline(p, color='r') for p in stim_fishy.ps_event_start]
        a.axhline(0, color='lightgray', linestyle='--')
    stim_cell_ax.set_title('full experiment')

    tail_ax.plot(stim_fishy.tail_df.frame, np.degrees(stim_fishy.tail_df.tail_sum), color='k')
    tail_ax.set_ylabel('tail sum (deg)')
    tail_ax.set_title('evoked tail behavior')

    stim_array_by_trial = gaussian_filter1d(stim_array_by_trial, sigma=1, mode='nearest')
    heatmap = sns.heatmap(stim_array_by_trial, cmap='bwr', ax=heatmap_ax,
                          vmin=heatmap_limits[0], vmax=heatmap_limits[1], center=0,
                          cbar_kws={"ticks": [heatmap_limits[0], 0, heatmap_limits[1]]})
    heatmap.collections[0].colorbar.set_label(trace_type, rotation=270)
    heatmap_ax.set_ylabel('trials')
    heatmap_ax.set_title('self-success per trial')

    ## averaged lineplot with std filled
    [avg_lineplot_ax.plot(t, color='lightgray', linewidth=1) for t in stim_array_by_trial]
    avg = np.nanmean(stim_array_by_trial, axis=0)
    avg_lineplot_ax.plot(avg, color='k', linewidth=1.5, label='mean')
    avg_lineplot_ax.set_ylim(-2, 4)
    avg_lineplot_ax.set_ylabel(trace_type)
    avg_lineplot_ax.set_title('self-success average')
    avg_lineplot_ax.legend(loc='upper left')
    avg_lineplot_ax.axhline(0, color='dimgray', linestyle='--')

    avg_lineplot_ax.axvline(-stim_fishy.photostim_frame_window[0], color='r')
    heatmap_ax.axvline(-stim_fishy.photostim_frame_window[0], color='k', linewidth=2)

    sns.despine()
    # all fps in imaging time since the tail data is plotted that way
    for subplot in [avg_lineplot_ax, stim_cell_ax, tail_ax, heatmap_ax]:
        plotutils.add_time_scalebar(subplot, length_secs=5, fps=stim_fishy.img_hz, label='5 s', pad_frac=0.008)
        subplot.set_xticks([])
        subplot.set_xlabel('')
        subplot.spines['bottom'].set_visible(False)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def plot_correlated_responders_heatmap_locations(photostim_fishvol, df, trace_type = 'stim_responses_zscore_local',
                                                 extreme_perc = 2, ideal_trace = None, save_path = None ):

    stim_offsets = photostim_fishvol[0].photostim_frame_window
    stim_cell_coords = df[df.photostim == True].neur_coords.iloc[0]
    stim_cell_plane = df[df.photostim == True].plane.iloc[0]
    hist_bin_size = 50
    trace_label = trace_type.split('_')[2]

    all_cell_ids = df.resp_cell_id.values
    all_zscore_resps = np.array([df[trace_type].values[i]['stim_0'][0] for i in range(len(df))])
    concat_zscore_resps = np.array([np.concatenate(i) for i in all_zscore_resps])
    photostim_frames = stimuli.stimulus_start_frames_for_plots(baseline_offset=-stim_offsets[0],
                                                               length_of_total_frame_arr=np.diff(stim_offsets)[0],
                                                               number_of_stims_in_set=int(
                                                                   concat_zscore_resps.shape[1] / np.diff(stim_offsets)[
                                                                      0]))
    if ideal_trace == None:
        ideal_trace = np.zeros(concat_zscore_resps.shape[1])
        for sf in photostim_frames:
            ideal_trace[sf + 1:sf + 7] = 1

    corr_vals = (concat_zscore_resps @ ideal_trace) / (len(ideal_trace) - 1)
    sort_idx = np.argsort(corr_vals)[::-1]  # largest to smallest
    resp_sorted = concat_zscore_resps[sort_idx]
    corr_sorted_vals = corr_vals[sort_idx]
    cell_ids_sorted = all_cell_ids[sort_idx]

    top_thresh = np.nanpercentile(corr_sorted_vals, 100 - extreme_perc)
    bottom_thresh = np.nanpercentile(corr_sorted_vals, extreme_perc)
    top_mask = corr_sorted_vals >= top_thresh
    bottom_mask = corr_sorted_vals <= bottom_thresh

    ## find the top and bottom corr neurons = take out the rois and superneuron traces
    resp_cell_coords = {}
    resp_cell_planes = {}
    select_traces = {}
    for corr_type in (['top_corr', 'bottom_corr']):
        if corr_type == 'top_corr':
            cell_inds = cell_ids_sorted[top_mask]
            traces = resp_sorted[top_mask]
        if corr_type == 'bottom_corr':
            cell_inds = cell_ids_sorted[bottom_mask]
            traces = resp_sorted[bottom_mask]
        resp_cell_coords[corr_type] = [v for v in
                                       df[df.resp_cell_id.isin(cell_inds)].neur_coords.values]
        resp_cell_planes[corr_type] = df[df.resp_cell_id.isin(cell_inds)].plane.values
        select_traces[corr_type] = traces

    ## setting up the figure with custom ratios
    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    outer = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1, 1, 2])  # controls rows only
    top = outer[0:2].subgridspec(2, 4, width_ratios=[5, 0.5, 4, 5],
                                 wspace=0.15)  # custom width ratios for the first 2 rows
    ax_heatmap = fig.add_subplot(top[:, 0])
    ax_corr_val_hist = fig.add_subplot(top[:, 1])
    top_corr_plt = fig.add_subplot(top[0, 2])
    anti_corr_plt = fig.add_subplot(top[1, 2])
    best_neurons_plt = fig.add_subplot(top[:, 3])

    bottom = outer[2].subgridspec(1, 4)
    img0 = fig.add_subplot(bottom[0])
    img1 = fig.add_subplot(bottom[1])
    img2 = fig.add_subplot(bottom[2])
    img3 = fig.add_subplot(bottom[3])

    ## responder heatmap
    im = ax_heatmap.imshow(resp_sorted, aspect='auto', cmap='gray_r', vmin=0, vmax=1)
    ax_heatmap.set_ylabel('neurons (sorted)')
    for sf in photostim_frames:
        ax_heatmap.axvspan(sf, sf + 1, color='red', alpha=0.8)
    plotutils.highlight_region(ax_heatmap, top_mask, "red", alpha=0.15)
    plotutils.highlight_region(ax_heatmap, bottom_mask, "blue", alpha=0.15)
    cbar = fig.colorbar(im, ax=ax_heatmap)
    cbar.set_label(trace_label)

    ## correlation values histogram
    sns.histplot(y=corr_vals, bins=hist_bin_size, ax=ax_corr_val_hist, kde=True,
                 orientation="horizontal", color="lightgray", edgecolor=None)
    ax_corr_val_hist.set_ylabel("correlation with ideal trace")
    ax_corr_val_hist.set_xlabel("count")
    hist_ylim = ax_corr_val_hist.get_ylim()
    ax_corr_val_hist.axhspan(top_thresh, hist_ylim[1], color="red", alpha=0.15)
    ax_corr_val_hist.axhspan(hist_ylim[0], bottom_thresh, color="blue", alpha=0.15)
    sns.despine(ax=ax_corr_val_hist, top=True, right=True)

    ## plot top and bottom correlated cell traces & locations
    example_img = photostim_fishvol[2].rescaled_ref
    best_neurons_plt.imshow(example_img, cmap='gray', vmax=np.percentile(example_img, 98), aspect='auto')
    best_neurons_plt.scatter(stim_cell_coords[0], stim_cell_coords[1], s=40, alpha=0.8, color='limegreen')
    for corr_type, color in zip(['top_corr', 'bottom_corr'], ['red', 'blue']):
        rois = np.array(resp_cell_coords[corr_type])
        if len(rois) > 0:
            best_neurons_plt.scatter(rois[:, 0], rois[:, 1], s=40, color=color, alpha=0.6)
            subplot = top_corr_plt if corr_type == 'top_corr' else anti_corr_plt
            subplot.axhline(0, color='lightgray', linestyle='--', alpha=0.5)
            for trace in select_traces[corr_type]:
                subplot.plot(trace, color=color, alpha=0.2)
            subplot.plot(np.nanmean(select_traces[corr_type], axis=0), color='k')
            for sf in photostim_frames:
                subplot.axvspan(sf, sf + 1, color='red', alpha=0.8)
            subplot.set_ylabel(trace_label)
            sns.despine(ax=subplot, left=False, right=True, top=True, bottom=True)
            subplot.set_xticks([])
            subplot.set_xlabel('')
            plotutils.add_time_scalebar(subplot, length_secs=10, fps=photostim_fishvol[2].img_hz, label='10 s', pad_frac=0.01)
            subplot.set_title(f'traces for {extreme_perc}% {corr_type}, n = {len(rois)}')
    best_neurons_plt.axis('off')
    best_neurons_plt.set_aspect("equal", adjustable="box")

    ## plot the location and corr val of ALL neurons
    abs_val = max([abs(corr_sorted_vals.min()), abs(corr_sorted_vals.max())])
    cmap_norm = matplotlib.colors.TwoSlopeNorm(vmin=-abs_val, vcenter=0, vmax=abs_val)
    sorted_rois = np.array([df[df.resp_cell_id == c].neur_coords.values[0] for c in cell_ids_sorted])
    sorted_planes = np.array([df[df.resp_cell_id == c].plane.values[0] for c in cell_ids_sorted])
    for plane, image_plt in zip(list(photostim_fishvol.volumes.keys()), [img0, img1, img2, img3]):
        image = photostim_fishvol.volumes[plane].rescaled_ref
        image_plt.imshow(image, cmap='gray', vmax=np.percentile(image, 98), aspect='auto')
        if plane == stim_cell_plane:
            image_plt.scatter(stim_cell_coords[0], stim_cell_coords[1], s=40, alpha=0.8, color='limegreen', zorder=3)
        plane_inds = np.where(sorted_planes == plane)[0]
        plane_rois = sorted_rois[plane_inds]
        image_plt.scatter(plane_rois[:, 0], plane_rois[:, 1], s=30, c=corr_sorted_vals[plane_inds],
                          cmap='bwr', norm=cmap_norm, alpha=0.8, zorder=1)
        image_plt.axis('off')
        image_plt.set_title(plane)
        image_plt.set_aspect("equal")
    sm = matplotlib.cm.ScalarMappable(cmap='bwr', norm=cmap_norm)
    sm.set_array([])  # required for matplotlib
    cbar = fig.colorbar(sm, ax=[img0, img1, img2, img3], shrink=0.8)
    cbar.set_label("correlation with ideal trace")

    fig.suptitle(f'{extreme_perc}% best/least correlated across whole volume')
    if save_path != None:
        plt.savefig(save_path)
    plt.show()

def plot_bhvr_each_photostim_trial(fishy, tail_bout_df, tail_offsets = None, save_path = None):
    # plot all trials together on the same plot
    if tail_offsets == None:
        tail_offsets = [-int(3 * fishy.tail_hz), int(5 * fishy.tail_hz)]
    tail_stim_duration = int((fishy.ps_event_duration / 1000) * fishy.tail_hz) # duration in tail frames
    tail_stim_events = fishy.tail_df[fishy.tail_df.stim_events != 'None'] # each of the tail df stim events (col)
    stim_event_starts = tail_stim_events.index.values
    tail_frame_subset = arrutils.subsection_arrays(stim_event_starts, tail_offsets)

    # adding in direction of each bout to list on top of the trial plot
    tail_bout_df['direction'] = 'None'
    for a in range(len(tail_bout_df)):
        bout_dir = 'right' if tail_bout_df.iloc[a].tail_angle > 0 else 'left'
        tail_bout_df['direction'].iloc[a] = bout_dir

    fig, ax = plt.subplots(1, len(tail_frame_subset), figsize=(20, 3), sharey=True)
    for v, frames in enumerate(tail_frame_subset):
        vals = np.degrees(fishy.tail_df.tail_sum.values[frames])
        all_bout_inds = [np.arange(v[0], v[1]) for v in tail_bout_df.cont_tuples_tailindex.values]
        bouts_during_frames = []
        for i, arr in enumerate(all_bout_inds):
            if np.intersect1d(frames[-tail_offsets[0]:], arr).size > 0:
                bouts_during_frames.append(i)
        bout_dirs = [tail_bout_df.direction.iloc[j] for j in bouts_during_frames]
        ax[v].plot(vals, color='k', linewidth=0.8)
        ax[v].axvspan(-tail_offsets[0], -tail_offsets[0] + tail_stim_duration, color='r', alpha=0.3)
        ax[v].set_xticks([])
        ax[v].set_xticklabels('')
        ax[v].set_ylim(-150, 150)
        ax[v].set_title(f'trial{v}_{str(bout_dirs)}')
    sns.despine(bottom=True)
    plotutils.add_time_scalebar(ax[-1], 1, fishy.tail_hz, label='1 s')
    fig.tight_layout()
    if save_path != None:
        plt.savefig(save_path)
    plt.show()



