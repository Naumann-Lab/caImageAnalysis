'''
Multiple functions to help process photostimulation and OMR datasets together and plotting

'''

import os
from pathlib import Path
from symbol import continue_stmt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.path as mpltPath
from matplotlib.patches import Circle, Patch
import seaborn as sns
from bcdict import BCDict
import shutil
from tiffile import imread
import math
import scipy

# local imports
from utilities import arrutils, plotutils, statutils, coordutils, roiutils, pathutils
from bruker_images import get_micronstopixels_scale, plot_imgs_with_overlay
from utilities.roiutils import create_circular_mask, draw_roi, create_polygon_mask
from fishy import PhotostimFish
import constants, stimuli, angles

# --- CREATE SAVING FOLDERS FOR SPECIFIC FOLDER STRUCTURE ON ANALYSIS PLOT OUTPUTS --- #
def create_saving_folder(fish_id_folder, cell_id=None):
    os.makedirs(fish_id_folder, exist_ok=True) # make sure the fish id folder is created
    if cell_id is None:
        sub_saving_directory = fish_id_folder
    else:
        sub_saving_directory = Path(fish_id_folder).joinpath(cell_id)
        os.makedirs(sub_saving_directory, exist_ok=True)

    return sub_saving_directory

def save_file_on_box_for_jacob(fish_id, file_path, master_directory = None):
    if master_directory is None:
        master_directory = Path(r'C:\Users\Kaitlyn\Box\PROJECT 98□ Data_for_Jacob')
    fish_id_folder = Path(master_directory).joinpath(fish_id)
    os.makedirs(fish_id_folder, exist_ok=True)  # make sure the fish id folder is created
    destination_path = Path(fish_id_folder).joinpath(file_path.name)
    shutil.copyfile(file_path, destination_path)


# --- MAKING MASTER DATAFRAME --- #
def build_functional_types_df(omr_fishvolume,
                              stim_fishvolume,
                              omr_fishvolume_barcoding_df,
                              match_cells_within_radius_um = 10,
                              regions = ['Pt', 'Hb', 'nMLF'],
                              motor_correlation = False,
                              plot_stim_sites = True):
    '''
    Build the master dataframe that would contain all the functional types of neurons in the OMR stack and compared to all the stim datasets
    omr_fishvolume = VolumeFish object, OMR dataset, needs to have tail data too
    stim_fishvolume = VolumeFish object, concatenated photostimulation dataset (one big dataset)
    volume_barcoding_df = dataframe, barcoding data for each neuron in the OMR stack
    regions = list, brain regions to look for in the OMR stack, default is Pt, Hb, nMLF
    motor_correlation = bool, if you want to include the motor correlation in the dataframe, default is False (only works with tail tracking data)
    plot_stim_sites = bool, if you want to plot the photostimulation sites, default is True (to check out if the stim dataset and the OMR cells match)

    returns a dataframe with the following columns:
    plane - int, plane number
    omr_neur_id - int, neuron id in the OMR stack
    neur_coords - list, coordinates of the OMR cell
    visual_barcode - str, visual motion barcoding of the neuron
    region - str, brain region of the neuron in the OMR stack (Pt, Hb, nMLF or nan if not in these)
    motor_corr - float, motor correlation of the neuron in the OMR stack
    photostim - bool, if the neuron was photostimulated
    if photostim...
        stim_events - list, the event order (number) of when that neuron was photostimulated
        stim_frames - list, frames in the stim dataset of when that neuron was photostimulated
    stim_neur_id - int, neuron id photostimulation dataset
    '''

    # process the functional information for all neurons into a giant dataframe, now just working with the OMR fish volume and the stim fish volume
    df_lst = [] 
    for plane in np.unique(omr_fishvolume_barcoding_df.plane.values):
        sub_functional_types_df = pd.DataFrame(columns = ['resp_cell_id', 'omr_neur_id', 'stim_neur_id', 'neur_coords', 'plane', 'region',  
                                        'visual_barcode', 'motion_responses', 'photostim', 'stim_frames', 'stim_events'])
        omr_plane_barcoding_df = omr_fishvolume_barcoding_df[omr_fishvolume_barcoding_df.plane == plane]
        omr_Fish = omr_fishvolume.volumes[plane]
        omr_Fish.load_saved_rois()
        print(f'adding regions {omr_Fish.roi_dict.keys()}')
        omr_data_rois = omr_Fish.return_cell_rois(range(len(omr_Fish.f_cells)))

        sub_functional_types_df['omr_neur_id'] = range(len(omr_Fish.f_cells))
        print('loading omr data')
        all_motion_responses_lst = gather_visual_motion_responses_for_df(omr_Fish, cell_id_array = None,
                                                                         motion_cues = constants.photostim_motion_cues)
        sub_functional_types_df['motion_responses'] = all_motion_responses_lst
        sub_functional_types_df['neur_coords'] = omr_data_rois
        sub_functional_types_df['plane'] = [plane] * len(sub_functional_types_df)
        if motor_correlation:
            sub_functional_types_df['motor_corr'] = omr_Fish.motor_pearson_corrs

        # default is none/false for these functional identities
        sub_functional_types_df['region'] = [np.nan] * len(sub_functional_types_df)
        sub_functional_types_df['visual_barcode'] = ['None'] * len(sub_functional_types_df)
        sub_functional_types_df['photostim'] = [False] * len(sub_functional_types_df)
        sub_functional_types_df['stim_frames'] = [None] * len(sub_functional_types_df)
        sub_functional_types_df['stim_events'] = [None] * len(sub_functional_types_df)
        
        for l in sub_functional_types_df.omr_neur_id.values:
            if l in omr_plane_barcoding_df.neur_ids.values:
                sub_functional_types_df['visual_barcode'].iloc[l] = omr_plane_barcoding_df[omr_plane_barcoding_df.neur_ids == l].barcoding.values[0]
            for k in regions:
                if k in omr_Fish.roi_dict.keys():
                    if l in omr_Fish.return_cells_by_saved_roi(k):
                        sub_functional_types_df['region'].iloc[l] = k

        stim_photostimFish = stim_fishvolume.volumes[plane]
        # gather xy_offsets if there are any
        if os.path.exists(stim_photostimFish.folder_path.parents[1].joinpath('omr_to_stim_data_offsets.npy')):
            xy_offset_dictionary = np.load(stim_photostimFish.folder_path.parents[1].joinpath('omr_to_stim_data_offsets.npy'), allow_pickle='TRUE').item()
            # only if the drift is more than half a cell body, should we worry about offsets
            drift_offset = 4
            if (abs(float(xy_offset_dictionary['x_drift_px'])) <= drift_offset) & (abs(float(xy_offset_dictionary['y_drift_px'])) <= drift_offset):
                xy_offset = (0, 0)
            elif (abs(float(xy_offset_dictionary['x_drift_px'])) <= drift_offset) & (abs(float(xy_offset_dictionary['y_drift_px'])) > drift_offset):
                xy_offset = (0, float(xy_offset_dictionary['y_drift_px']))
            elif (abs(float(xy_offset_dictionary['x_drift_px'])) > drift_offset) & (abs(float(xy_offset_dictionary['y_drift_px'])) <= drift_offset):
                xy_offset = (float(xy_offset_dictionary['x_drift_px']), 0)
            else:
                xy_offset = (float(xy_offset_dictionary['x_drift_px']), float(xy_offset_dictionary['y_drift_px']))
            print(f'xy offset between datasets = {xy_offset}')
        else:
            xy_offset = (0, 0)
            print('no xy offset saved between datasets')

        matched_cell_ids = coordutils.match_cell_ids(sub_functional_types_df.omr_neur_id.values, omr_Fish.stats,
                                                    range(len(stim_photostimFish.f_cells)), stim_photostimFish.stats,
                                                    um_to_px = stim_photostimFish.um_per_px,
                                                    xy_offset = xy_offset)

        sub_functional_types_df['stim_neur_id'] = matched_cell_ids.values()
            
        if 'stim_sites' in stim_photostimFish.data_paths.keys():
            # need to identify which neurons were photostimulated by matching spatially between OMR and stim datasets
            print('matching stimulation sites with OMR cells')
            if not hasattr(stim_photostimFish, 'stimmed_cell_coords'):
                print('identifying stim cells in the fish class')
                stim_photostimFish.stimmed_cell_coords, stim_photostimFish.stimmed_cell_id_array, stim_photostimFish.stimmed_cells_matched_stim_ids_dict = PhotostimFish.identify_stim_cells(stim_photostimFish,
                                                                                                                                                                                        within_radius_um= match_cells_within_radius_um,
                                                                                                                                                                                        frame_window=[stim_photostimFish.photostim_frame_window[0],
                                                                                                                                                                                                        stim_photostimFish.evoked_num_frames])
            
            # sort the stimmed cell ids by the cell id order of the stim sites df
            # important for the next step of matching the stim events and frames
            key_order = stim_photostimFish.stim_sites_df.cell_ids.values
            print(f'df order: {key_order}')
            sorted_stimmed_closest_cell_id_dict = {key: stim_photostimFish.stimmed_cells_matched_stim_ids_dict[key] for key in key_order if key in stim_photostimFish.stimmed_cells_matched_stim_ids_dict}
            print(f'sorted dict: {stim_photostimFish.stimmed_cells_matched_stim_ids_dict}')
            stimmed_cell_id_array = np.array(list(sorted_stimmed_closest_cell_id_dict.values()))
            print(f'sorted array: {stimmed_cell_id_array}')

            omr_photostim_cell_id_lst = []
            bad_indices = []
            for idx, actual_s_cell in enumerate(stimmed_cell_id_array):
                for o_cell_id, s_cell_id in matched_cell_ids.items():
                    if actual_s_cell == s_cell_id:
                        omr_photostim_cell_id_lst.append(int(o_cell_id))
                if actual_s_cell not in matched_cell_ids.values():
                    omr_photostim_cell_id_lst.append(np.nan)
                    bad_indices.append(idx)
                    pass # if you don't find a match, then pass and we won't worry about that stimulation site
            print(f'omr_photostim_cell_id_lst: {omr_photostim_cell_id_lst}')
            # now get make sure to add the correct stim frame and events for each cell
            stim_frames = stim_photostimFish.stim_sites_df.stim_frames.values
            stim_events = stim_photostimFish.stim_sites_df.stim_events.values
            for n in range(len(stimmed_cell_id_array)):
                if n not in bad_indices:
                    sub_functional_types_df.loc[[omr_photostim_cell_id_lst[n]], 'photostim'] = True
                    sub_functional_types_df.loc[[omr_photostim_cell_id_lst[n]], 'stim_frames'] = pd.Series([stim_frames[n]], 
                                                                                                           index = sub_functional_types_df.index[[omr_photostim_cell_id_lst[n]]])
                    sub_functional_types_df.loc[[omr_photostim_cell_id_lst[n]], 'stim_events'] = pd.Series([stim_events[n]], 
                                                                                                           index = sub_functional_types_df.index[[omr_photostim_cell_id_lst[n]]])
            if plot_stim_sites:
                try:
                    plt.figure(figsize = (6, 6))
                    plt.imshow(stim_photostimFish.rescaled_ref, cmap = 'gray', vmax = np.percentile(stim_photostimFish.rescaled_ref, 99))
                    plt.title(f'{plane}')
                    for n in range(len(stimmed_cell_id_array)):
                        if n not in bad_indices:
                            print('white is OMR cell, red is stim id cell')
                            plt.scatter(sub_functional_types_df.neur_coords.loc[omr_photostim_cell_id_lst[n]][0], 
                                        sub_functional_types_df.neur_coords.loc[omr_photostim_cell_id_lst[n]][1], color = 'white', s = 10)
                            plt.annotate(omr_photostim_cell_id_lst[n], (sub_functional_types_df.neur_coords.loc[omr_photostim_cell_id_lst[n]][0], 
                                            sub_functional_types_df.neur_coords.loc[omr_photostim_cell_id_lst[n]][1]), color = 'white')
                            plt.scatter(stim_photostimFish.stimmed_cell_coords[n][0], stim_photostimFish.stimmed_cell_coords[n][1], color = 'red', s = 10)
                            plt.annotate(stimmed_cell_id_array[n], (stim_photostimFish.stimmed_cell_coords[n][0], 
                                                                                        stim_photostimFish.stimmed_cell_coords[n][1]), color = 'red')
                    plt.show() 
                except:
                    print('cannot plot for some reason')
        else:
            pass  
        df_lst.append(sub_functional_types_df)   

    functional_types_df = pd.concat(df_lst).reset_index(drop = True)
    final_functional_types_df = add_resp_cell_ids_to_df(functional_types_df)
    
    return final_functional_types_df    

def add_resp_cell_ids_to_df(functional_types_df, specific_stim_key_order = []):
    '''
    Add the responding cell id to the dataframe
    functional_types_df = dataframe, the master functional types dataframe
    specific_stim_key_order = if there is a specific order of stim key values (important for ensemble identification)

    returns the dataframe with the responding cell id
    '''
    responders = functional_types_df[functional_types_df.photostim == False]
    stimulators = functional_types_df[functional_types_df.photostim == True]

    resp_cell_id_lst = []
    for n in range(len(responders)):
        resp_cell_id_lst.append(f'resp_{n}')
    responders['resp_cell_id'] = resp_cell_id_lst

    if len(specific_stim_key_order) < 1:
        stim_resp_cell_id_lst = []
        for t in range(len(stimulators)):
            stim_resp_cell_id_lst.append(f'stim_{t}')
    else:
        stim_resp_cell_id_lst = [f'stim_{k}' for k in specific_stim_key_order]
    stimulators['resp_cell_id'] = stim_resp_cell_id_lst

    updated_functional_types_df = pd.concat([responders, stimulators]).reset_index(drop = True)

    return updated_functional_types_df

def gather_visual_motion_responses_for_df(vizstimfishy, cell_id_array = None, motion_cues = constants.photostim_motion_cues):
    '''
    Getting the complete visual motion response from the OMR dataset for each cell in the OMR dataset
    vizstimfishy = VizStimFish object, the OMR dataset (one fish, one plane)
    cell_id_array = list, the cell ids to get the responses from, default is all cells
    motion_cues = list, the motion cues to get responses from, default is the photostim motion cues

    Use offsets from the fishy to decide the windows to take the data

    returns a list of dictionaries, each dictionary contains the motion responses for each cell (mean and std)
    '''
    if cell_id_array is None: # default is all cells in the OMR fishy
        cell_id_array = range(len(vizstimfishy.f_cells))
    extended_responses_df = pd.DataFrame(vizstimfishy.extended_responses_normf) # using the normed data here, but actually does not matter
    motion_resp_dict_lst = []
    for k in cell_id_array:
        motion_responsive_dict = BCDict()
        omr_cell_extended_raw_resp = extended_responses_df.iloc[k]
        for stim in motion_cues:
            if stim not in motion_responsive_dict.keys():
                motion_responsive_dict[stim] = {}
            motion_resp = omr_cell_extended_raw_resp[stim]
            df_f_motion_resp = np.zeros(shape = np.array(motion_resp).shape)
            for a, arr in enumerate(motion_resp):
                base_e = arr[:-vizstimfishy.offsets[0]]
                plot_e = (arr - np.nanmean(base_e)) / np.nanmean(base_e)
                df_f_motion_resp[a] = plot_e 
            # gathering mean and std response around the visual motion cue
            motion_responsive_dict[stim]['mean'] = np.nanmean(df_f_motion_resp, axis = 0) 
            motion_responsive_dict[stim]['std'] = np.nanstd(df_f_motion_resp, axis = 0)
        motion_resp_dict_lst.append(motion_responsive_dict)

    return motion_resp_dict_lst

def gather_photostimulation_responses_for_df(responder_f_traces, stimulated_cell_ids, stimulated_frames_array,
                                             photostim_response_frame_windows = [-15, 30], type = 'df_f'):
    '''
    Gather the df/f responses of the responding cells to each photostimulated cell
    responder_f_traces = list, the traces of the responding cells (n responder cells x len of imaging), this should include ALL cells including the stim ones
    stimulated_cell_ids = list, the cell ids of the stimulated cells (n photostimulated cells)
    stimulated_frames_array = list, the frames of which each photostimulated cell was stimulated (n photostimulated cells x m trials of photostimulation)
    photostim_response_frame_windows = list, the window of frames around the photostimulation event to consider, default is [-4, 7]
    type = str, the type of responses, default is 'df_f' - note 'local' (can also be 'raw' aka no changes to the traces), this needs to match the traces that come in
    '''
    
    photostimulation_responses_lst = []
    avg_evoked_activity_master_lst = []

    # for each responding cell, get the responses of that cell to each stimulation event
    for resp_cell_id, responding_trace in enumerate(responder_f_traces):
        resp_to_each_stimulation_dict = BCDict()
        avg_evoked_activity_lst = []
        for stim_cell_ind, s_cell_id in enumerate(stimulated_cell_ids):
            if s_cell_id not in resp_to_each_stimulation_dict.keys():
                resp_to_each_stimulation_dict[s_cell_id] = {}
            stimmed_frames = stimulated_frames_array[stim_cell_ind] # should be the correct frames
            frame_subset = arrutils.subsection_arrays(stimmed_frames, photostim_response_frame_windows)

            if frame_subset[-1][-1] > len(responding_trace): # adjust frames in case this is out of range
                new_frame_subset = []
                for s in frame_subset:
                    new_frame_subset.append([q for q in s if q < len(responding_trace)])
                frame_subset = np.array(new_frame_subset)

            resp_raw_trial = np.array([responding_trace[g] for g in frame_subset if len(responding_trace[g] ) > 0])
            resp_df_f_trial = np.zeros(shape = (len(resp_raw_trial), len(resp_raw_trial[0])))
            resp_raw_trial2 = np.zeros(shape = (len(resp_raw_trial), len(resp_raw_trial[0])))

            stim_evoked_df_f_trial = np.zeros(shape = (len(resp_raw_trial), 1))
            stim_evoked_raw_trial = np.zeros(shape = (len(resp_raw_trial), 1))

            for d, f in enumerate(resp_raw_trial):
                base_e = f[:-photostim_response_frame_windows[0]] # i.e. frames 0:4
                plot_e = (f - np.nanmean(base_e)) / np.nanmean(base_e)
                evoked_e = np.nanmedian(plot_e[-photostim_response_frame_windows[0]:-photostim_response_frame_windows[0] + photostim_response_frame_windows[1]]) # evoked df f for each trial, i.e. frames 4:end
                stim_evoked_df_f_trial[d] = evoked_e
                resp_df_f_trial[d] = plot_e # full trace for each trial

                # calculate raw traces of your input array
                evoked_raw = np.nanmedian(f[-photostim_response_frame_windows[0]:-photostim_response_frame_windows[0] + photostim_response_frame_windows[1]]) # evoked raw trace for each trial, i.e. frames 4:end
                stim_evoked_raw_trial[d] = evoked_raw
                resp_raw_trial2[d] = f # full trace for each trial

            if type == 'df_f':
                resp_to_each_stimulation_dict[s_cell_id] = [resp_df_f_trial]
                avg_evoked_activity_lst.append(np.nanmean(stim_evoked_df_f_trial))
            elif type == 'zscore': # only want raw traces
                resp_to_each_stimulation_dict[s_cell_id] = [resp_raw_trial2]
                avg_evoked_activity_lst.append(np.nanmean(stim_evoked_raw_trial))
            elif type == 'global_df_f': # only want raw traces
                resp_to_each_stimulation_dict[s_cell_id] = [resp_raw_trial2]
                avg_evoked_activity_lst.append(np.nanmean(stim_evoked_raw_trial))

        # put all of this information into lists to add to the final dataframe
        photostimulation_responses_lst.append(resp_to_each_stimulation_dict)
        avg_evoked_activity_master_lst.append(avg_evoked_activity_lst)
    
    return photostimulation_responses_lst, avg_evoked_activity_master_lst

def add_photostimulation_responses_to_functional_df(functional_info_df,
                                                    stim_fishvolume,
                                                    response_window = [-15, 30],
                                                     trace_type = 'df_f'):

    # default is to have the entired photostim == True dataset for the 'stimulated info'
    # keeping this flexible in case I need to change what I want to use as my 'stimulated cells' (i.e. changing trials or actual cells)
    stimulated_functional_types_df = functional_info_df[functional_info_df.photostim == True]
    # gathering responses from the photostimulation dataset
    resp_f_trace_array, resp_norm_trace_array, resp_zscore_trace_array = prepare_data_for_plotting(functional_info_df, stim_fishvolume)
    stimulated_cell_ids = stimulated_functional_types_df.resp_cell_id.unique()

    if trace_type == 'df_f':
        photostimulation_responses_lst, avg_evoked_df_f_lst = gather_photostimulation_responses_for_df(resp_f_trace_array, stimulated_cell_ids,
                                                                                                   stimulated_functional_types_df.stim_frames.values,
                                                                                                   photostim_response_frame_windows = response_window)
        functional_info_df['stim_responses'] = photostimulation_responses_lst
        functional_info_df['avg_evoked_df_f'] = avg_evoked_df_f_lst

    elif trace_type == 'zscore':
        photostimulation_responses_lst_zscore, avg_evoked_zscore_lst = gather_photostimulation_responses_for_df(resp_zscore_trace_array, stimulated_cell_ids,
                                                                                                   stimulated_functional_types_df.stim_frames.values,
                                                                                                   photostim_response_frame_windows = response_window,
                                                                                                   type = 'zscore')

        functional_info_df['stim_responses_zscore'] = photostimulation_responses_lst_zscore
        functional_info_df['avg_evoked_zscore'] = avg_evoked_zscore_lst
    
    elif (trace_type == 'global_df_f') & ('baseline_f' in functional_info_df.columns):
        global_df_f_traces = np.zeros(shape = resp_f_trace_array.shape)
        for n, raw_trace in enumerate(resp_f_trace_array):
            baseline_f = functional_info_df['baseline_f'].iloc[n]['mean']
            global_df_f_trace = (raw_trace - baseline_f) / baseline_f
            global_df_f_traces[n] = global_df_f_trace

        photostimulation_responses_lst_global_df_f, avg_evoked_global_df_f_lst = gather_photostimulation_responses_for_df(global_df_f_traces, stimulated_cell_ids,
                                                                                                   stimulated_functional_types_df.stim_frames.values,
                                                                                                   photostim_response_frame_windows = response_window,
                                                                                                   type = 'global_df_f')
        functional_info_df['stim_responses_global_df_f'] = photostimulation_responses_lst_global_df_f
        functional_info_df['avg_evoked_global_df_f'] = avg_evoked_global_df_f_lst
    else:
        print('did not add responses')

    return functional_info_df

def add_regions_to_functional_df(functional_info_df, omr_fish_volume):

    '''
    Add the regions to the functional dataframe in case this needs to be edited outside of the master processing function above
    functional_info_df = dataframe, the functional dataframe with the id's and planes of the neurons
    omr_fish_volume = VolumeFish object, the OMR dataset (one fish, one plane)

    returns the functional dataframe with the regions added/updated
    '''

    region_lst_per_cell = []
    for r in range(len(functional_info_df)):
        xy_coord = functional_info_df.iloc[r].neur_coords
        plane = functional_info_df.iloc[r].plane
        fishy = omr_fish_volume.volumes[plane]
        fishy.load_saved_rois()
        region = np.nan # default will be nan's
        for roi_name, roi_path in fishy.roi_dict.items():
            if roi_name != 'midline': # don't want to include the midline
                mplt_path = mpltPath.Path(np.load(Path(roi_path)))
                if mplt_path.contains_points([(xy_coord[0], xy_coord[1])]):
                    region = roi_name
        region_lst_per_cell.append(region)
    
    functional_info_df['region'] = region_lst_per_cell

    return functional_info_df


def add_forward_and_backward_response_to_df(df,
                                            omr_fishy):
    '''
    Add the forward and backward motion responses to the functional dataframe
    :param df: functional types dataframe to add info to
    :param omr_fishy: example omr fish to grab some parameters from
    :return: df with the new columns
    '''
    baseline_frames = -omr_fishy.offsets[0]
    motion_on_frames = int((omr_fishy.seconds_motion_is_on) * (omr_fishy.img_hz))

    forward_resp_list = []
    backward_resp_list = []
    for row in range(len(df)):
        forward_resp = False
        backward_resp = False
        for specific_stim in ['forward', 'backward']:
            vizmotion_responses_avg = df.iloc[row].motion_responses[specific_stim]['mean']
            vizmotion_responses_std = df.iloc[row].motion_responses[specific_stim]['std']
            baseline_avg = np.nanmean(vizmotion_responses_avg[:baseline_frames])
            baseline_std = np.nanmean(vizmotion_responses_std[:baseline_frames])
            motion_on_avg = np.nanmean(vizmotion_responses_avg[baseline_frames:baseline_frames + motion_on_frames])
            if motion_on_avg >= (baseline_avg + 1.8 * baseline_std):
                if specific_stim == 'forward':
                    forward_resp = True
                if specific_stim == 'backward':
                    backward_resp = True
        forward_resp_list.append(forward_resp)
        backward_resp_list.append(backward_resp)

    df['forw_resp'] = forward_resp_list
    df['back_resp'] = backward_resp_list

    return df

def find_baseline_values(functional_df, stim_fishvolume, number_of_baseline_frames = 30, buffer = 5):
    '''
    Find the baseline values for each neuron trace in the functional dataframe
    functional_df = dataframe, the functional dataframe with the id's and planes of the neurons
    stim_fishvolume = VolumeFish object, the photostimulation dataset (one fish, one plane)
    number_of_baseline_frames = int, the number of frames to use for the baseline, default is 30
    buffer = int, the number of frames to use as a buffer for the baseline, default is 5
    '''
    f_arr, _, zscore_arr = prepare_data_for_plotting(functional_df, stim_fishvolume)

    baseline_zscore = []
    baseline_f = []
    baseline_global_df_f = []
    for b in range(len(f_arr)):
        # get zscore baseline
        mean_z_b = np.nanmean(zscore_arr[b][buffer+2:number_of_baseline_frames-buffer], axis = 0)
        std_z_b = np.nanstd(zscore_arr[b][buffer+2:number_of_baseline_frames-buffer], axis = 0)
        baseline_zscore.append({'mean': mean_z_b, 'std': std_z_b})

        # gather raw f baseline
        mean_f_b = np.nanmean(f_arr[b][buffer+2:number_of_baseline_frames-buffer], axis = 0)
        std_f_b = np.nanstd(f_arr[b][buffer+2:number_of_baseline_frames-buffer], axis = 0)
        baseline_f.append({'mean': mean_f_b, 'std': std_f_b})

        # find the global df/f baseline
        global_df_f = (f_arr[b] - mean_f_b) / mean_f_b
        mean_b = np.nanmean(global_df_f[buffer+2:number_of_baseline_frames-buffer], axis = 0)
        std_b = np.nanstd(global_df_f[buffer+2:number_of_baseline_frames-buffer], axis = 0)
        baseline_global_df_f.append({'mean': mean_b, 'std': std_b})
    
    return baseline_f, baseline_zscore, baseline_global_df_f

# --- VISUAL MOTION FUNCTIONAL IDENTITY CHARACTERISTICS  for single neurons --- #

def get_tuning(cell_vizmotion_resp_dict, motion_stim_list = ['forward', 'right', 'backward', 'left'],
               vizmotion_on_frame = 15, frames_motion_on = 17, plotting = False):
    '''
    get tuning for a single neuron
    :param cell_vizmotion_resp_dict: 'motion_responses' in the df from one cell/row
    :param motion_stim_list: types of motion stims to use for find the tuning - note this has to be of the cardinal/intercardinal directions
    :param vizmotion_on_frame: the frame that the motion cue started (in the dict arrays)
    :param frames_motion_on: number of frames that motion was presented for
    :param plotting: if you want to plot the polar plots for the cell
    :return:
        tuning_angle: the tuning direction angle
        tuning_weight:  the tuning weight (magnitude)
        tuning_color: combined color showing the tuning of this type of neuron
        dsi: direction selectivity of this neuron (within these stim types)
    '''

    degree_ids = [constants.deg_dict[i] for i in motion_stim_list]
    stimuli_colors = [constants.monocular_dict[i] for i in motion_stim_list]

    specific_direction_resp = np.array([cell_vizmotion_resp_dict[stim]['mean'] for stim in motion_stim_list])
    # don't need to normalize to any baseline since this is df/f
    val_resp_per_stim = np.nanmean(specific_direction_resp[:, vizmotion_on_frame: vizmotion_on_frame + frames_motion_on], axis = 1)

    degs_and_resps_dict = dict(zip(motion_stim_list, val_resp_per_stim))
    degree_responses = [np.clip(degs_and_resps_dict[i], a_min=0, a_max=999) for i in motion_stim_list]
    neuron_peak_deg = angles.weighted_mean_angle(degree_ids, degree_responses)
    neuron_peak_resp = np.nanmax(degree_responses)
    neuron_peak_stim = motion_stim_list[degree_responses.index(neuron_peak_resp)]
    try:
        neuron_color = np.average(stimuli_colors, axis=0, weights=degree_responses)  # scale each channel by the value
    except:
        neuron_color = [0.5, 0.5, 0.5]

    if plotting:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot([math.radians(d) for d in degree_ids], degree_responses, linestyle = '-', color = neuron_color)
        ax.fill([math.radians(d) for d in degree_ids], degree_responses, alpha = 0.4, color = neuron_color)
        ax.plot([0,math.radians(neuron_peak_deg)], [0,neuron_peak_resp], linestyle = '-', color = 'k')
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks([math.radians(d) for d in degree_ids])
        ax.set_xticklabels(degree_ids)
        plt.show()

    tuning_angle = neuron_peak_deg
    tuning_weight = neuron_peak_resp
    tuning_color = neuron_color

    # also get the dsi value for these motion stimuli
    inverse_stim = constants.nulldict[neuron_peak_stim]
    inverse_val = degree_responses[motion_stim_list.index(inverse_stim)]
    dsi = np.clip((neuron_peak_resp - inverse_val) / neuron_peak_resp, a_min=0, a_max=1)

    return tuning_angle, tuning_weight, tuning_color, dsi

def get_bi(cell_vizmotion_resp_dict, vizmotion_on_frame = 15, frames_motion_on = 17):
    '''
    binocularity index function
    :param cell_vizmotion_resp_dict: 'motion_responses' in the df from one cell/row
    :param vizmotion_on_frame: the frame that the motion cue started (in the dict arrays)
    :param frames_motion_on: number of frames that motion was presented for
    :return: the binocularity index value for the neuron
    '''
    binoc_stims = constants.binocular_dict.keys()
    binoc_stims_resp = np.array([cell_vizmotion_resp_dict[stim]['mean'] for stim in binoc_stims])
    val_resp_per_stim = np.nanmax(binoc_stims_resp[:, vizmotion_on_frame: vizmotion_on_frame + frames_motion_on], axis = 1)

    max_resps_dict = dict(zip(binoc_stims, val_resp_per_stim))

    eyeR = np.clip(max(max_resps_dict["medial_left"], max_resps_dict['lateral_right']), a_min=0, a_max=2)
    eyeL = np.clip(max(max_resps_dict['lateral_left'], max_resps_dict['medial_right']), a_min=0, a_max=2)
    bi = (eyeR - eyeL) / (eyeR + eyeL)
    return bi

def get_motion_weights(cell_motion_resp_dict, vizmotion_on_frame = 15, frames_motion_on = 17):
    '''
    gather the motion 'weights' - one value for each motion stimuli (derived from taking the mean response during motion on)
    :param cell_motion_resp_dict: 'motion_responses' in the df from one cell/row
    :param vizmotion_on_frame: the frame that the motion cue started (in the dict arrays)
    :param frames_motion_on: number of frames that motion was presented for
    :return: motion weights dictionary, keys are motion stimuli, values are the motion weight
    '''
    motion_weights = {key: [] for key in list(cell_motion_resp_dict.keys())}
    for each_motion in cell_motion_resp_dict.keys():
        one_motion_mean_arr = cell_motion_resp_dict[each_motion]['mean']
        one_motion_weights = np.nanmean(one_motion_mean_arr[vizmotion_on_frame: vizmotion_on_frame + frames_motion_on])
        motion_weights[each_motion] = one_motion_weights
    return motion_weights

def get_suppression_value(cell_motion_weights):
    '''
    get the suppression value for a single neuron using the motion weights
    :param cell_motion_weights: one value per stimuli, depicting how responsive a neuron is to those motion stimuli
    :return:
        lowest_val: the suppression value (mean response to the suppressed stimuli)
        least_activated_stim: the motion stimuli that is least activated (suppressed)
    '''
    # determine suppression as the avg value of motion responses to the least responsive stimuli
    lowest_val_idx = np.argmin(list(cell_motion_weights.values()))
    lowest_val = list(cell_motion_weights.values())[lowest_val_idx]
    least_activated_stim = list(cell_motion_weights.keys())[lowest_val_idx]

    return lowest_val, least_activated_stim

def add_vizmotion_functional_identity_info_to_df(df, vizmotion_on_frame = 15, frames_motion_on = 17,
                                                 tuning_motion_stim = ['forward', 'right', 'backward', 'left']):
    '''
    Add the visual motion functional identity information to the functional types df
    :param df: a functional types dataframe (at least with 'motion responses' as a column
    :param vizmotion_on_frame: the frame that the motion cue started (in the dict arrays)
    :param frames_motion_on: number of frames that motion was presented for
    :return: a functional types dataframe with the new functional identity info in the new columns
    '''
    ang_lst = []
    weight_lst = []
    color_lst = []
    bi_lst = []
    dsi_lst = []
    motion_weights_lst = []
    suppression_values_lst = []

    for each_row in range(len(df)):
        cell_motion_resp_info = df.iloc[each_row].motion_responses
        if cell_motion_resp_info != 'None':
            motion_weights = get_motion_weights(cell_motion_resp_info, vizmotion_on_frame= vizmotion_on_frame, frames_motion_on=frames_motion_on)
            tuning_angle, tuning_weight, tuning_color, dsi_val = get_tuning(cell_motion_resp_info,
                                                                            motion_stim_list = tuning_motion_stim,
                                                                            vizmotion_on_frame=vizmotion_on_frame,
                                                                            frames_motion_on=frames_motion_on, plotting=False)
            bi_val = get_bi(cell_motion_resp_info, vizmotion_on_frame=vizmotion_on_frame, frames_motion_on=frames_motion_on)
            supp_val, supp_stim = get_suppression_value(motion_weights)
        else:
            tuning_angle = 'None'
            tuning_weight = 'None'
            tuning_color = 'None'
            bi_val = 'None'
            dsi_val = 'None'
            motion_weights = 'None'
            supp_val = 'None'

        ang_lst.append(tuning_angle)
        weight_lst.append(tuning_weight)
        color_lst.append(tuning_color)
        bi_lst.append(bi_val)
        dsi_lst.append(dsi_val)
        motion_weights_lst.append(motion_weights)
        suppression_values_lst.append(supp_val)

    df['motion_weights'] = motion_weights_lst
    df['tuning_angle'] = ang_lst
    df['tuning_weight'] = weight_lst
    df['tuning_color'] = color_lst
    df['dsi'] = dsi_lst
    df['bi'] = bi_lst
    df['supp_score'] = suppression_values_lst

    return df

# --- GATHERING ACTIVITY METRICS, SELF-SUCCESS TRIALS --- #

def average_traces_across_stimulated_cells(ind_dict, trace_dict, stim_keys):
    """
    Important for making the 'mock figure 3/4'
    For neurons common to all stim_keys, average their trace arrays across stim keys.
    For neurons unique to only one stim_key, include their trace as-is.
    
    Parameters:
    - ind_dict: dict with stim keys mapping to list of neuron indices
    - trace_dict: dict with stim keys mapping to list of 1D np arrays (traces), aligned with ind_dict order
    - stim_keys: list of stim keys to consider (e.g. ['stim_6', 'stim_1'])
    
    Returns:
    - dict mapping neuron index -> averaged trace (common neurons) or original trace (unique neurons)
    """
    # Get sets of neurons for each stim key
    neuron_sets = [set(ind_dict[k]) for k in stim_keys]
    print(neuron_sets)
    
    # Neurons common to all stim keys
    common_neurons = set.intersection(*neuron_sets)
    # Neurons unique to any one stim key (union minus common)
    all_neurons = set.union(*neuron_sets)
    unique_neurons = all_neurons - common_neurons
    
    result_traces = {}
    
    # Average traces for common neurons
    for neuron in common_neurons:
        traces_to_average = []
        for k in stim_keys:
            inds = ind_dict[k]
            traces = trace_dict[k]
            pos = inds.index(neuron)
            traces_to_average.append(traces[pos])
        avg_trace = np.mean(np.vstack(traces_to_average), axis=0)
        result_traces[neuron] = avg_trace
    
    # Include unique neurons with their original trace
    for neuron in unique_neurons:
        # Find which stim key contains this neuron (only one, by definition)
        for k in stim_keys:
            if neuron in ind_dict[k]:
                pos = ind_dict[k].index(neuron)
                result_traces[neuron] = trace_dict[k][pos]
                break
    
    return result_traces


def prepare_data_for_plotting(data_df, fishvolume, dataset_type='stim'):
    '''
    Prepare the data for plotting
    '''
    f_trace_array = np.zeros(shape=(len(data_df), len(fishvolume[0].f_cells[0])))
    normf_trace_array = np.zeros(shape=(len(data_df), len(fishvolume[0].f_cells[0])))
    zscored_trace_array = np.zeros(shape=(len(data_df), len(fishvolume[0].f_cells[0])))
    if dataset_type == 'stim':
        cell_ids = data_df.stim_neur_id.values
    if dataset_type == 'omr':
        cell_ids = data_df.omr_neur_id.values
    for r_ind, r_cell in enumerate(cell_ids):  # all cell ids in the whole dataframe
        r_cell = int(r_cell)
        dataFish = fishvolume.volumes[data_df.plane.values[r_ind]]
        f_trace_array[r_ind] = dataFish.f_cells[r_cell]
        arr = dataFish.normcells[r_cell]
        z = (arr - np.nanmean(arr)) / np.nanstd(arr) # manually zscoring
        normf_trace_array[r_ind] = arr
        # zscored_trace_array[r_ind] = dataFish.zdiff_cells[r_cell] # i realized that this is causing a smoothing issue...
        zscored_trace_array[r_ind] = z

    return f_trace_array, normf_trace_array, zscored_trace_array

def determine_self_success_trials(only_stimulated_cells_df,
                                    response_type = 'df/f',
                                    std_threshold = 1.2,
                                    immediate_resp_window = 5,
                                    baseline_frames = 4,
                                    photostim_window = [-15, 30],
                                    save_plots_path = None,
                                    plotting = True,
                                    saving_dict = None):
    '''
    using df/f or zscore responses, threshold is baseline + std dev above baseline

    Gather the self-success trials for each photostimulated cell (can save plots and dictionary)

    only_stimulated_cells_df = dataframe, the functional data dataframe of only the photostimulated cells (needs to at least have columns resp_cell_id and stim_responses)
    std_threshold = float, the threshold for the standard deviation of the baseline activity to determine if a trial is significant, default is 1.2
    immediate_resp_window = int, the number of frames after the photostimulation event to consider as the immediate response, default is 4
    baseline_frames = int, the number of frames to use for the baseline, default is 4 (right before the stimulation)
    photostim_window = list, the window of frames around the photostimulation event to consider, default is [-4, 7]
    save_plots_path = if you want to save the plots, this is the folder they will go
    plotting = True or False if you want to see the plots or not
    saving_dict = if you want to save the significnat trials dictionary, provide a path

    return
    self_success_trials_dict = BCDict, a dictionary of the significantly photostimulated trials for each photostimulated cell
    '''
    # create a legend for the plot
    legend_elements = [matplotlib.lines.Line2D([0], [0], color='red', linewidth=3, label='photostimulation event'),  # Red line
                   matplotlib.lines.Line2D([0], [0], color='black', linewidth=3, label='self-success'),# black line = self-success trial
                   matplotlib.lines.Line2D([0], [0], color='lightgrey', linewidth=3, label='not self-success')] # gray line = not self-success trial

    self_success_trials_dict = BCDict()
    for idx, stim_cell in enumerate(only_stimulated_cells_df.resp_cell_id.values):
        if stim_cell not in self_success_trials_dict.keys():
            self_success_trials_dict[stim_cell] = {}
        one_stim_cell_info = only_stimulated_cells_df[only_stimulated_cells_df.resp_cell_id == stim_cell]
        string_cell_id = 'stim_' + str(idx)
        if response_type == 'df/f':
            all_responses_to_itself = one_stim_cell_info.stim_responses.values[0][string_cell_id][0]
        if response_type == 'zscore':
            all_responses_to_itself = one_stim_cell_info.stim_responses_zscore.values[0][string_cell_id][0]

        line_colors = [] # gather line colors for plotting
        significant_trials_count = [] # just to count the number of self-success trials easily, not really necessary
        for a, b in enumerate(all_responses_to_itself):
            if a not in self_success_trials_dict[stim_cell].keys():
                self_success_trials_dict[stim_cell][a] = np.nan
            b = arrutils.pretty(b, 2) # need to smooth this data since it looks so noisy (bc of imaging faster)
            mean_baseline_activity = np.nanmean(b[-photostim_window[0] - baseline_frames:-photostim_window[0]])
            std_baseline_activity = np.nanstd(b[:-photostim_window[0]])
            mean_evoked_activity = np.nanmean(b[-photostim_window[0]:-photostim_window[0] + immediate_resp_window])

            if mean_evoked_activity > mean_baseline_activity + (std_threshold * std_baseline_activity):
                self_success_trials_dict[stim_cell][a] = mean_evoked_activity
                significant_trials_count.append(a)
                line_colors.append('black')
            else:
                self_success_trials_dict[stim_cell][a] = np.nan
                significant_trials_count.append(np.nan)
                line_colors.append('lightgrey')
        
        if plotting:
            fig, ax = plt.subplots(1, all_responses_to_itself.shape[0], figsize = (18, 2))
            ymin = np.nanmin(all_responses_to_itself)
            ymax = np.nanmax(all_responses_to_itself)
            for a, b in enumerate(all_responses_to_itself):
                ax[a].plot(arrutils.pretty(b, 2), color = line_colors[a], alpha = 1)
                ax[a].set_ylim(ymin, ymax)
                ax[a].axhline(0, color = 'grey', linestyle = '--') # 0 df/f baseline
                ax[a].axvline(-photostim_window[0], color = 'red', linestyle = '-') # the photostimulation event
                ax[a].spines['top'].set_visible(False)
                ax[a].spines['right'].set_visible(False)
                ax[a].set_title(f'trial {a}')
                if a > 0:
                    ax[a].tick_params(axis='y', left=False, labelleft=False)
                else:
                    ax[a].set_ylabel('dF/F')
            fig.suptitle(stim_cell, y = 1.1, fontsize = 14)
            ax[-1].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5))
            if save_plots_path is not None:
                plt.savefig(Path(save_plots_path).joinpath(f'{stim_cell}_trace_per_trial_with_sign.png'), format="png", dpi = 300, bbox_inches = 'tight')
            plt.show()

        if saving_dict is not None:
            saving_path = Path(saving_dict).joinpath('self_success_trials.npy')
            np.save(saving_path, self_success_trials_dict)
    
    return self_success_trials_dict

def determine_self_success_trials_v2(only_stimulated_cells_df,
                                         response_window = [-15, 30],
                                        evoked_response_window = 5,
                                         baseline_response_window = 10,
                                         saving_dict_path = None):
    """
    DONT USE - based on other people's papers, not as strict criteria

    Function to find the self-success trials for each neuron based on the zscore of the responses.
    Peak evoked z score activity > than 0.25 + mean baseline is threshold for self-success
    Args:
        only_stimulated_cells_df (pd.DataFrame): DataFrame containing all the stimulated cells and their responses.
        response_window (list): The window of frames before and after the photostimulation event.
        evoked_response_window (int): The number of frames immediately after the photostim event to use 
                                    (as these are the neurons that should be stimulated, look within a narrow window).
        baseline_response_window (int): The number of frames to consider for the baseline response. 
                                        (the longer the window the better to account for noise)
        saving_dict_path (str): Path to save the self_success_trials_dict.
    Returns:
        self_success_dict (dict): Dictionary containing the self_success trials for each neuron.
    """

    self_success_trials_dict = {}
    for s in range(len(only_stimulated_cells_df)):
        stim_cell_id = only_stimulated_cells_df.iloc[s].resp_cell_id
        zscore_responses = only_stimulated_cells_df.iloc[s].stim_responses_zscore
        self_zscore_responses = zscore_responses[stim_cell_id][0]
        
        sig_trials = []
        for n, trial_responses in enumerate(self_zscore_responses):
            baseline_zscore = np.nanmean(trial_responses[-response_window[0] - baseline_response_window:-response_window[0]])
            peak_evoked_zscore = np.nanmax(trial_responses[-response_window[0]:-response_window[0]+ evoked_response_window])
            if peak_evoked_zscore > 0.25 + baseline_zscore:
                sig_trials.append(n)
        self_success_trials_dict[stim_cell_id] = sig_trials
    
    if saving_dict_path is not None:
        saving_path = Path(saving_dict_path).joinpath('self_success_trials.npy')
        np.save(saving_path, self_success_trials_dict)
    
    return self_success_trials_dict

def determine_trial_weights(stimmed_func_df,
                            trace_type = 'zscore',
                            self_success = True,
                            photostim_response_window = [-15, 30],
                            evoked_response_window = 5,
                            baseline_frames = 10,
                            saving_dict_path = None):
    '''
    Determine the trial weights for each self-success trial of each photostimulated cell
    weight = peak evoked activity - baseline activity

    stimmed_func_df = dataframe, the dataframe with the functional types of the neurons, only the photostimulated cells
    trace_type = 'zscore' or 'df/f', the type of traces to use for determining weights
    self_success = True, if you want to use just the self-success trials or not
    photostim_response_window = list, the window of frames around the photostimulation event to use, default is [-15, 30]
    evoked_response_window = int, the number of frames after the photostimulation event to consider as the evoked time window, default is 5
    baseline_frames = int, the number of frames before the photostimulation event to consider as the baseline time window, default is 10
    saving_dict_path = str, the path to save the dictionary, default is None (not saved)
    
    returns
    weighted_vals_dict = dict, the dictionary with the weighted values for each self-success trial of each photostimulated cell
    '''
    weighted_vals_dict = {}
    for i, one_stim_cell in enumerate(stimmed_func_df.resp_cell_id.unique()):
        if trace_type == 'df/f':
            all_trace_responses = stimmed_func_df.stim_responses.iloc[i]
        if trace_type == 'zscore':
            all_trace_responses = stimmed_func_df.stim_responses_zscore.iloc[i]

        if self_success:
            self_success_trials = stimmed_func_df.self_success_trials.iloc[i] # use only self success trials
        else:
            self_success_trials = range(len(all_trace_responses[one_stim_cell][0])) # use all trials

        select_trace_responses = all_trace_responses[one_stim_cell][0][self_success_trials]
        baseline_activity_per_trial = select_trace_responses[:,-photostim_response_window[0] - baseline_frames:
                                                                -photostim_response_window[0]]
        baseline_activity_per_trial = np.nanmean(baseline_activity_per_trial, axis=1)

        peak_evoked_activity_per_trial = select_trace_responses[:,-photostim_response_window[0]:
                                                                    -photostim_response_window[0] + evoked_response_window]
        peak_evoked_activity_per_trial = np.nanmax(peak_evoked_activity_per_trial, axis=1)

        weighted_values = peak_evoked_activity_per_trial - baseline_activity_per_trial

        min_norm = 0.01 # Set the minimum desired normalized value, this way nothing is 0 value for weighted mean
        arr_min = weighted_values.min()
        arr_max = weighted_values.max()
        if arr_max == arr_min:
            norm_weighted_values = np.full_like(weighted_values, fill_value=min_norm)
        else:
            norm_weighted_values = (weighted_values - arr_min) / (arr_max - arr_min)
            norm_weighted_values = norm_weighted_values * (1 - min_norm) + min_norm # Now scale to [min_norm, 1]

        weighted_vals_dict[one_stim_cell] = norm_weighted_values

    if saving_dict_path != None:
        np.save(saving_dict_path.joinpath('trial_weights.npy'), weighted_vals_dict)

    return weighted_vals_dict

def add_self_success_trials_to_df(functional_info_df, self_success_trials_dict):
    '''
    Add self success trials to the dataframe, else nan
    '''
    functional_info_df['self_success_trials'] = [np.nan] * len(functional_info_df)
    for d in self_success_trials_dict.keys():
        self_success_trials_list = []
        try: # if this is a dictionary in each value
            for k, v in self_success_trials_dict[d].items():
                if v is not np.nan:
                    self_success_trials_list.append(k)
        except: # or if it's just a list, not a dictionary
            self_success_trials_list = self_success_trials_dict[d]
        if len(self_success_trials_list) > 0:
            index = functional_info_df[functional_info_df['resp_cell_id'] == d].index.tolist()
            functional_info_df.loc[index, 'self_success_trials'] = pd.Series([self_success_trials_list], index = index)

    return functional_info_df

def add_trial_weights_to_df(functional_info_df, weighted_vals_dict):
    '''
    Add weights for each trial to the dataframe, else nan
    '''
    functional_info_df['trial_weights'] = [np.nan] * len(functional_info_df)
    for d in weighted_vals_dict.keys():
        weighted_vals_list = []
        try: # if this is a dictionary in each value
            for k, v in weighted_vals_dict[d].items():
                if v is not np.nan:
                    weighted_vals_list.append(k)
        except: # or if it's just a list, not a dictionary
            weighted_vals_list = weighted_vals_dict[d]
        if len(weighted_vals_list) > 0:
            index = functional_info_df[functional_info_df['resp_cell_id'] == d].index.tolist()
            functional_info_df.loc[index, 'trial_weights'] = pd.Series([weighted_vals_list], index = index)

    return functional_info_df

def gather_evoked_activity_for_select_trials(photostim_responses_per_trial, good_trial_numbers = None, weights = None,
                                              immediate_response_window = 5, photostim_response_window = [-15, 30],
                                              r_type = 'peak'):
    '''
    Get the evoked activity for each trial, using the photostimulation responses, just for one cell

    photostim_responses_per_trial = list, the photostimulation responses for each trial (n trials x len of imaging)
    good_trial_numbers = list, the trial numbers to consider, default is None (all trials)
    immediate_response_window = int, the number of frames after the photostimulation event to consider as the immediate response, default is 5
    photostim_response_window = list, the window of frames around the photostimulation event to consider, default is [-15, 30]
    r_type = str, the type of response to consider, default is 'peak' (can also be 'mean' or 'min')

    returns
    avg_evoked_activity_per_trial = list, the evoked activity for each trial
    avg_evoked_activity = float, the average evoked activity for all trials
    '''
    if good_trial_numbers == None:
        good_trial_numbers = list(range(len(photostim_responses_per_trial)))
    
    avg_evoked_activity_per_trial = []
    for trial_num, ps_resp in enumerate(photostim_responses_per_trial):
        if trial_num in good_trial_numbers:
            evoked_activity_window = ps_resp[-photostim_response_window[0]:-photostim_response_window[0] + immediate_response_window]
            if r_type == 'peak':
                evoked_activity = np.nanmax(evoked_activity_window)
            elif r_type == 'mean':
                evoked_activity = np.nanmean(evoked_activity_window)
            elif r_type == 'min':
                evoked_activity = np.nanmin(evoked_activity_window)
            else:
                evoked_activity = np.nanmean(evoked_activity_window)
            avg_evoked_activity_per_trial.append(evoked_activity)
    
    if weights is None:
        avg_evoked_activity = np.nanmean(avg_evoked_activity_per_trial)
    else:
        avg_evoked_activity = np.average(avg_evoked_activity_per_trial, weights = weights)
    
    return avg_evoked_activity_per_trial, avg_evoked_activity

def gather_averaged_latency_to_peak(photostim_responses_per_trial, good_trial_numbers = None, weights = None,
                                              immediate_response_window = 5, photostim_response_window = [-15, 30],
                                              img_hz = 1):
    '''
    Get the average latency to peak for the zscored activity of the photostim responses.
    
    photostim_responses_per_trial = list, the photostimulation responses for each trial (n trials x len of imaging)
    good_trial_numbers = list, the trial numbers to consider, default is None (all trials)
    weights = list, the weights for each trial, default is None (no weights)
    immediate_response_window = int, the number of frames after the photostimulation event to consider as the evoked time window, default is 5
    photostim_response_window = list, important to know baseline, the window of frames around the photostimulation event to consider, default is [-15, 30]
    img_hz = int, the imaging speed for gather time, default is 1

    returns
    latency_to_peak_frame = int, the frame number of the peak response (post the photostim event, not including baseline frames)
    latency_to_peak_time = float, the time of the peak response in seconds
    peak_value = float, the value of the peak response
    
    '''
    if good_trial_numbers == None:
        good_trial_numbers = list(range(len(photostim_responses_per_trial)))
    
    all_select_traces = photostim_responses_per_trial[good_trial_numbers]
    
    if weights is not None: # grab the weighted average of the trials before calculating the latency
        avg_trace = np.average(all_select_traces, axis=0, weights=weights)
    else:
        avg_trace = np.nanmean(all_select_traces, axis = 0)

    # only look at the evoked activity window
    avg_evoked_trace = avg_trace[-photostim_response_window[0]:-photostim_response_window[0] + immediate_response_window]
    smoothed_avg_evoked_trace = arrutils.pretty(avg_evoked_trace, 2)
    peak_value = np.nanmax(smoothed_avg_evoked_trace)
    latency_to_peak_frame = np.nanargmax(smoothed_avg_evoked_trace)
    latency_to_peak_time = latency_to_peak_frame / img_hz

    return latency_to_peak_frame, latency_to_peak_time, peak_value

# --- RSCHRMINE EXPRESSION & SPATIAL LOCATION ANALYSIS --- #
def check_rsChrmine_expression(red_channel_img, select_rois, save_path, offset = 1.0):
    '''
    - Work in progress -

    Identify if the cell has enough rsChrmine expression to be a true choice for photostimulation
    Use the red channel stacks (reference) stacks

    returns
    rsChrmine_dict -- a dictionary with the roi number as the key and a boolean as the value: True if has rsChrmine expression, False if not
    '''
    # gather baseline section
    if not os.path.exists(Path(save_path).joinpath('rois/baseline_red_channel.npy')):
        draw_roi(red_channel_img, save_path, 'baseline_red_channel')
    baseline_points = np.load(Path(save_path).joinpath('rois/baseline_red_channel.npy'))
    baseline_mask = create_polygon_mask(red_channel_img.shape, baseline_points)
    baseline_roi = red_channel_img[baseline_mask]

    baseline = np.mean(baseline_roi)  # Mean brightness of the full image
    std_dev = np.std(baseline_roi)    # Standard deviation of brightness
    threshold = baseline + (offset * std_dev) # Compute dynamic threshold

    rsChrmine_dict = {}
    for n, each_roi in enumerate(select_rois):
        if n not in rsChrmine_dict.keys():
            rsChrmine_dict[n] = {}

        mask = create_circular_mask(red_channel_img.shape, each_roi[0], each_roi[1], int(0.52 * 3))
        red_roi = red_channel_img[mask]
        mean_brightness = np.nanmean(red_roi)

        if mean_brightness > threshold:
            rsChrmine_dict[n] = True
        else:
            rsChrmine_dict[n] = False
    
    return rsChrmine_dict

def use_rschrmine_image_for_rois(rschrmine_exp_folder,
                                 gcamp_fishy_vol,
                                 rois=['nMLF', 'Hb'],
                                 crop_images = False):
    '''
    Use the rsChrmine images to determine ROIs for future analysis
    Make sure to collect a series of volumes of rsChrmine expression (matching the gcamp volume)
    :param rschrmine_exp_folder: folder path that the rsChrmine stack is saved
    :param gcamp_fishy_vol: the gcamp fish volume (normal omr fish vol)
    :param rois: the specific rois that you want to draw using the rschrmine channel, typically the nmlf and hindbrain ones
    :return:
    '''
    # 1 - gather all the tif paths to make an averaged stack image
    stack_img_lst = []
    for file in os.listdir(rschrmine_exp_folder):
        if file.endswith('.ome.tif') and 'Ch1' in file:  # will always be the red channel
            stack_img_lst.append(imread(rschrmine_exp_folder / file))
    full_stack_img_array = np.array(stack_img_lst)
    avg_stack_arr = np.nanmean(full_stack_img_array, axis=0)
    avg_stack_arr = np.array([np.rot90(img) for img in avg_stack_arr])

    # 2 - check out the overlay between the two images, and it is cropped just fyi, for just nMLF region
    for e, fish in enumerate(gcamp_fishy_vol):
        if crop_images:
            crop_gcamp = fish.rescaled_ref[100:300, 150:350]
            crop_rschrmine = avg_stack_arr[e][100:300, 150:350]
        else:
            crop_gcamp = fish.rescaled_ref
            crop_rschrmine = avg_stack_arr[e]
        plot_imgs_with_overlay(crop_gcamp, crop_rschrmine, 'gcamp', 'rschrmine')

    # 3 - draw the actual new rois, only using the rschrmine channel & plot them on the gcamp image
    for e, fish in enumerate(gcamp_fishy_vol):
        for r in rois:
            roiutils.draw_roi(avg_stack_arr[e], fish.folder_path, r)
        fish.load_saved_rois()
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(fish.rescaled_ref, vmax=np.percentile(fish.rescaled_ref, 99))
        for roi in rois:
            roiPath = Path(fish.folder_path).joinpath('rois')
            roi_pts = np.load(roiPath.joinpath(f'{roi}.npy'))
            path = mpltPath.Path(roi_pts)
            coords = path.to_polygons()
            ax.fill([i[0] for i in coords[0]], [i[1] for i in coords[0]], color='white', alpha=0.5)
        plt.title(f'plane {e}')
        plt.show()

    return print('new rois are saved')

def plotting_location_of_stim_site_per_trial(stimulated_cell_df, full_volume_stim_sites_df, photostim_fish_volume, save_location = None):
    '''
    Plot the location of the stimulation site (from Bruker) and the closest ROI to the stimulation site for each trial
    stimulated_cell_df = dataframe, the dataframe of the stimulated cells
    full_volume_stim_sites_df = dataframe, the full dataframe of the photostimulation stim sites (aka master stim sites df)
    photostim_fish_volume =  the photostimulation data in fish volume format
    save_location = str, the path to save the images to, default is None (a subfolder will be created for the exact stim cell)

    '''

    legend_elements = [matplotlib.lines.Line2D([0], [0], color='red', linewidth=2, label='stimulation point'),  # Red line, stimulation site from bruker
                    matplotlib.patches.Patch(facecolor='green', edgecolor='black', label='stimulated cell')]  # Green box, matched ROI to the stimulation site

    # to make this run faster, first index into the plane df to just load in the specific photostim fish once
    unique_planes = sorted(np.unique(stimulated_cell_df.plane.values)) # make sure its in order
    for p, plane in enumerate(unique_planes):
        one_plane_stimulated_cell_df = stimulated_cell_df[stimulated_cell_df.plane == plane]
        specific_plane_fish = photostim_fish_volume.volumes[plane]
        if not hasattr(specific_plane_fish, 'stimmed_cells_matched_stim_ids_dict'):
            print('need to identify stim cell ids in fishvolume')
            break
        um_per_pxs = get_micronstopixels_scale(specific_plane_fish.data_paths['info_xml'])
        sp_size_pxs = np.nanmean(specific_plane_fish.stim_sites_df.sp_size.values) / um_per_pxs  # average pixels diameter

        # collect the full image over the course of the experiment
        full_img = specific_plane_fish.load_image()
        baseline_img = np.nanmean(full_img[:20, :, :], axis=0)  # baseline image (collecting the first 20 frames now, hard coded)
        if len(one_plane_stimulated_cell_df) == 0:
            print('no stimulation sites on this plane')
            break

        for stim_cell_num in range(len(one_plane_stimulated_cell_df)):
            one_stim_row = one_plane_stimulated_cell_df.iloc[stim_cell_num]
            stim_events_frames = one_stim_row.stim_frames
            # bc sometimes there are not matching cells btween dataframes depending on OMR dataset matching for the func types df
            if [int(one_stim_row.stim_neur_id)] in list(specific_plane_fish.stimmed_cells_matched_stim_ids_dict.values()):
                _stim_cell_id =  [k for k, v in specific_plane_fish.stimmed_cells_matched_stim_ids_dict.items() if v == [int(one_stim_row.stim_neur_id)]][0]
                # need to index into the correct row
                full_volume_stim_sites_df_specific_stim_site = full_volume_stim_sites_df[full_volume_stim_sites_df.cell_ids == _stim_cell_id]
                original_stim_roi = [full_volume_stim_sites_df_specific_stim_site.x_stim.values[0], full_volume_stim_sites_df_specific_stim_site.y_stim.values[0]]

                # closest ROI to the programmed stim site (green circle, full ROI is filled in)
                stim_full_roi = specific_plane_fish.stats[int(one_stim_row.stim_neur_id)]

                fig, ax = plt.subplots(1, len(stim_events_frames) + 1, figsize = (20, 5))
                fig.suptitle(one_stim_row.resp_cell_id, y = 0.8)
                ax[0].imshow(baseline_img, cmap = 'gray', vmax = np.percentile(baseline_img, 99))
                ax[0].scatter(stim_full_roi['xpix'], stim_full_roi['ypix'], color = 'limegreen', linewidth=1, s = 1)
                ax[0].set_title(f'baseline')

                for n, p in enumerate(stim_events_frames):
                    stim_img = full_img[p:p+10, :,:]
                    stim_img_avg = np.nanmean(stim_img, axis = 0)
                    ax[n+1].imshow(stim_img_avg, cmap = 'gray', vmax = np.percentile(stim_img_avg, 95), vmin = np.percentile(stim_img_avg, 10))
                    ax[n+1].scatter(stim_full_roi['xpix'], stim_full_roi['ypix'], color = 'limegreen', linewidth=1, s = 1)
                    ax[n+1].set_title(f'trial {n}')

                # Create a white cross of the stimulation site from the og coordinates on all plots (from the Bruker, what i programmed in to stimulate)
                cross_x = original_stim_roi[0]
                cross_y = original_stim_roi[1]
                cross_size = int(sp_size_pxs)/2
                for a in ax.flatten():
                    a.plot([cross_x - cross_size, cross_x + cross_size], [cross_y, cross_y], 'r', linewidth=2)  # Horizontal
                    a.plot([cross_x, cross_x], [cross_y - cross_size, cross_y + cross_size], 'r', linewidth=2)  # Vertical
                    a.set_xlim(int(original_stim_roi[0] - 30), int(original_stim_roi[0] + 30))
                    a.set_ylim(int(original_stim_roi[1] + 30), int(original_stim_roi[1] - 30))

                ax[0].legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, -0.05))
                plt.tight_layout()
                if save_location is not None:
                    save_path = create_saving_folder(save_location, one_stim_row.resp_cell_id)
                    plt.savefig(Path(save_path).joinpath(f'locations_per_trial.png'), format="png", dpi = 300)
                plt.show()
            else:
                print('no stimulation site match')

    return None        

def compute_overlap_metrics(photostim_info_df, master_stim_sites_df, stim_fish_volume, stim_cell_id = 'stim_0'):
    '''
    FOR ONE STIM SITE
    Computing the overlap metrics between the cell source and the stimulation site. 
    The function computes the overlap ratio, coverage of the cell source, and coverage of the stimulation site, with ordered list of the stim cells
    :param photostim_info_df: DataFrame containing the information of the photostimulated cells, the functional data df of only photostimulated cells
    :param master_stim_sites_df: DataFrame containing the information of the stimulation sites, find in the data folder
    :param stim_fish_volume: VolumeFish object containing the fish volume data
    :param stim_cell_id: str, the id of the stimulated cell in the functional data df, default is 'stim_0'

    :return: overlap_ratios, coverage_of_cell_lst, coverage_of_stim_site_lst, stim_cell_source_id_lst
    '''

    um_to_px = get_micronstopixels_scale(stim_fish_volume[0].data_paths['info_xml'])

    cell_course_info = photostim_info_df[photostim_info_df.resp_cell_id == stim_cell_id]
    cell_source_plane = cell_course_info.plane
    cell_source_stim_neur_id = int(cell_course_info.stim_neur_id)
    cell_source_stim_events = cell_course_info.stim_events

    # grab the matching row/stimmed cell in the master stim sites df based on a different column
    for i in range(len(master_stim_sites_df)): 
        if cell_source_stim_events[0] == master_stim_sites_df.iloc[i].stim_events[0]:
            matching_ind = i

    # get the x & y pixels of the stimulation site
    bruker_stimulation_coord_x = master_stim_sites_df.iloc[matching_ind].x_stim
    bruker_stimulation_coord_y = master_stim_sites_df.iloc[matching_ind].y_stim
    spiral_diameter_um = master_stim_sites_df.iloc[matching_ind].sp_size
    spiral_diameter_px = spiral_diameter_um / um_to_px
    spiral_diameter_px = int(spiral_diameter_px) # convert spiral size to pixels
    bruker_stimulation_site_xpix, bruker_stimulation_site_ypix = roiutils.points_within_circle(bruker_stimulation_coord_x, 
                                                                                    bruker_stimulation_coord_y,
                                                                                    radius = spiral_diameter_px/2)
    bruker_circle = [(i, bruker_stimulation_site_ypix[n]) for n, i in enumerate(bruker_stimulation_site_xpix)] # convert into coordinates

    # identify the pixels in the cell source polygon
    plane_fish = stim_fish_volume.volumes[cell_source_plane]
    cell_source_xpix = plane_fish.stats[cell_source_stim_neur_id]['xpix']
    cell_source_ypix = plane_fish.stats[cell_source_stim_neur_id]['ypix']
    cell_source_polygon = [(i, cell_source_ypix[n]) for n, i in enumerate(cell_source_xpix)] 

    # find the overlap ratio between the two polygons, based on pixels
    overlap_size, overlap_ratio = coordutils.get_overlap_between_neurons(cell_source_xpix, cell_source_ypix, 
                                                                        bruker_stimulation_site_xpix, bruker_stimulation_site_ypix)

    # compute the coverage of the cell source and the stimulation site
    percent_of_cell, percent_of_stim_site = roiutils.compute_coverage(bruker_circle, cell_source_polygon)

    return overlap_ratio, percent_of_cell, percent_of_stim_site

def plotting_nearby_cells_xy(stim_cell_name,
                             same_plane_df,
                             plotting_fishy,
                             bruker_coordinate,
                             radius_um=10,
                             cmap_name="gnuplot2",
                             photostim_window=[-8, 12],
                             photostim_stim_offset=0,
                             photostim_ylim=[-0.5, 1.2],
                             save_path = None):
    '''
    Plotting the nearby cells from a target cell in XY (traces of all the cells and locations, with bruker cross too)
    Using info from the big functional types df

    stim_cell_name = resp cell id name of the stimulated cell (i.e. 'stim_0')
    same_plane_df = subset of the functional types df that is the same plane at the stimmed cell
    plotting_fishy = fish from the stim fishvolume to use for plotting locations, getting source shapes
    bruker_coordinate = the og coordinate from the bruker that should be where was stimulated
    radius_um = radius of the nearby cells to collect, in ums
    cmap_name = the color map to use for sources/traces colors
    photostim_window = frames before and after the photostimulation event that were selected for the responses
    photostim_stim_offset = if there needs to be an offset in the photostim line (default is 0)
    photostim_ylim = y limits for plotting the heatmap, traces
    save_path = provide the parent directory folder

    '''
    ums_per_px = get_micronstopixels_scale(plotting_fishy.data_paths['info_xml'])

    one_stim_row = same_plane_df.loc[same_plane_df[same_plane_df.resp_cell_id == stim_cell_name].index[0]]
    one_stim_cell_coords = one_stim_row.neur_coords
    one_stim_cell_name = one_stim_row.resp_cell_id
    one_stim_idx = one_stim_row.name

    nearby_coords_idx, nearby_coords_lst = coordutils.determine_nearby_cells_xy(one_stim_cell_coords,
                                                                     same_plane_df.neur_coords.values, ums_per_px,
                                                                     radius_um)
    nearby_cells_df_index = same_plane_df.index[nearby_coords_idx]
    nearby_responding_cells_df = same_plane_df.loc[nearby_cells_df_index]
    nearby_responding_cells_df = nearby_responding_cells_df.drop(
        one_stim_idx)  # don't want the same stimmed cell in this
    nearby_responding_cells_shapes = [plotting_fishy.stats[int(n)] for n in
                                      nearby_responding_cells_df.stim_neur_id.values]
    stim_cell_shape = plotting_fishy.stats[int(one_stim_row.stim_neur_id)]

    num_traces = len(nearby_responding_cells_df)
    cmap = plt.get_cmap(cmap_name)  # or "viridis", "plasma", "tab20", etc.
    colors = [cmap(i / num_traces) for i in range(num_traces)]

    # Setting up the figure
    # Always ensure at least 3 columns
    ncols = max(num_traces, 3)
    image_end = int(ncols * 0.3)
    heatmap_start = image_end + 1
    heatmap_end = heatmap_start + int(ncols * 0.4)
    stimtrace_start = heatmap_end + 1
    heatmap_end = min(heatmap_end, ncols - 1) # Make sure indices don't overlap or exceed
    stimtrace_start = min(stimtrace_start, ncols - 1)

    fig = plt.figure(figsize=(20, 10), constrained_layout=True)
    gs = matplotlib.gridspec.GridSpec(2, ncols,
                           height_ratios=[2, 1],  # Top row taller
                           hspace=0.4,  # Space between rows
                           wspace=0.2  # Space between columns
                           )
    ax_image = fig.add_subplot(gs[0, :image_end])
    ax_heatmap = fig.add_subplot(gs[0, heatmap_start:heatmap_end])
    ax_stimtrace = fig.add_subplot(gs[0, stimtrace_start:])
    ax_traces = [fig.add_subplot(gs[1, i]) for i in range(num_traces)]

    # --- IMAGE subplot (top left)
    ax_image.imshow(plotting_fishy.rescaled_ref, cmap="gray", vmax=np.percentile(plotting_fishy.rescaled_ref, 99))
    [ax_image.scatter(a['xpix'], a['ypix'], s=3, color=colors[i]) for i, a in enumerate(nearby_responding_cells_shapes)]
    ax_image.scatter(stim_cell_shape['xpix'], stim_cell_shape['ypix'], color="limegreen", s=3)
    cross_size = int(6 / ums_per_px) / 2
    ax_image.plot([bruker_coordinate[0] - cross_size, bruker_coordinate[0] + cross_size],
                  [bruker_coordinate[1], bruker_coordinate[1]], 'r', linewidth=3)  # Horizontal
    ax_image.plot([bruker_coordinate[0], bruker_coordinate[0]],
                  [bruker_coordinate[1] - cross_size, bruker_coordinate[1] + cross_size], 'r', linewidth=3)  # Vertical
    # center image around bruker coordinate
    ax_image.set_xlim(bruker_coordinate[0] - int(radius_um * 2 / ums_per_px),
                      bruker_coordinate[0] + int(radius_um * 2 / ums_per_px))
    ax_image.set_ylim(bruker_coordinate[1] + int(radius_um * 2 / ums_per_px),
                      bruker_coordinate[1] - int(radius_um * 2 / ums_per_px))
    ax_image.set_yticks([])
    ax_image.set_xticks([])
    ax_image.set_title(stim_cell_name)

    # --- HEATMAP subplot (top center)
    stim_response = one_stim_row.stim_responses[stim_cell_name]
    print(stim_response[0].shape)
    heatmap_data = stim_response[0][:, :np.diff(photostim_window)[0]]  # shape: (trials, timepoints) - adjusted for the photostim window
    im = ax_heatmap.imshow(heatmap_data, vmin=-photostim_ylim[1], vmax=photostim_ylim[1], aspect="auto", cmap="coolwarm")
    ax_heatmap.axvline(x=-photostim_window[0] + photostim_stim_offset, color="red")
    ax_heatmap.set_title("trial responses of photostimmed cell", fontsize=12)
    ax_heatmap.set_xlabel("frames")
    ax_heatmap.set_ylabel("trial")
    plt.colorbar(im, ax=ax_heatmap, fraction=0.02, pad=0.04)

    # --- STIM TRACE AVG (top right)
    avg_stim_response = np.nanmean(heatmap_data, axis=0)
    max_evoked_val = np.nanmax(avg_stim_response[-photostim_window[0]:])
    # response metric is the peak value - avg baseline for the 1 sec before the stimulation event
    resp_metric = max_evoked_val - np.nanmean(avg_stim_response[int(-photostim_window[0] - plotting_fishy.img_hz):-photostim_window[0]])
    ci_lower, ci_upper = statutils.calculate_ci(np.array(heatmap_data))
    [ax_stimtrace.plot(np.arange(len(m)), arrutils.pretty(m, 2), color="grey", alpha=0.3) for m in heatmap_data]
    ax_stimtrace.plot(np.arange(len(heatmap_data[0])), arrutils.pretty(avg_stim_response, 2), color="limegreen")
    ax_stimtrace.fill_between(np.arange(len(heatmap_data[0])), ci_lower, ci_upper, color="skyblue", alpha=0.4,
                              label="95% CI")
    ax_stimtrace.axvline(x=-photostim_window[0] + photostim_stim_offset, color="red")
    ax_stimtrace.axhline(0, color="grey", linestyle="--")
    ax_stimtrace.set_ylim(photostim_ylim[0], photostim_ylim[1])
    ax_stimtrace.set_xlim(0, len(heatmap_data[0]) - 1)
    ax_stimtrace.set_title('{:.2f}'.format(resp_metric), rotation=30, fontsize=10)

    # --- TRACE subplots (bottom row)
    for n in range(num_traces):
        responding_cell_name = nearby_responding_cells_df.iloc[n].resp_cell_id
        response_to_one_stim = nearby_responding_cells_df.iloc[n].stim_responses[stim_cell_name]
        shape_x = nearby_responding_cells_shapes[n]['xpix']
        shape_y = nearby_responding_cells_shapes[n]['ypix']
        # annotate the image
        annotate_offset = 2
        ax_image.text(max(shape_x), np.mean(shape_y) - annotate_offset, str(responding_cell_name),
                      color='r', fontsize=10, ha='center', va='center',
                      bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

        if isinstance(response_to_one_stim, (list, np.ndarray)) and not isinstance(response_to_one_stim[0],(float, int)):
            if np.array(response_to_one_stim).shape[1] == np.diff(photostim_window)[0]:
                response_to_one_stim = response_to_one_stim
            else:
                response_to_one_stim = response_to_one_stim[0]
            response_to_one_stim = response_to_one_stim[:, :np.diff(photostim_window)[0]]
            avg_response = np.nanmean(response_to_one_stim, axis=0)
            max_evoked_val = np.nanmax(avg_response[-photostim_window[0]:])
            # response metric is the peak value - avg baseline for the 1 sec before the stimulation event
            resp_metric = max_evoked_val - np.nanmean(
                avg_stim_response[int(-photostim_window[0] - plotting_fishy.img_hz):-photostim_window[0]])
            ci_lower, ci_upper = statutils.calculate_ci(np.array(response_to_one_stim))
            if not isinstance(response_to_one_stim, (float, int)):
                [ax_traces[n].plot(
                    np.arange(len(m)),
                    arrutils.pretty(m, 2),
                    color="grey",
                    alpha=0.3
                ) for m in response_to_one_stim]

                ax_traces[n].plot(
                    np.arange(len(response_to_one_stim[0])),
                    arrutils.pretty(avg_response, 2),
                    color=colors[n]
                )
                ax_traces[n].fill_between(
                    np.arange(len(response_to_one_stim[0])),
                    ci_lower, ci_upper,
                    color="skyblue", alpha=0.3, label="95% CI"
                )
                ax_traces[n].axvline(
                    x=-photostim_window[0] + photostim_stim_offset,
                    color="red"
                )
                ax_traces[n].axhline(0, color="grey", linestyle="--")
                ax_traces[n].set_title('{:.2f}'.format(resp_metric), rotation=30, fontsize=10)
                ax_traces[n].set_ylim(photostim_ylim[0], photostim_ylim[1])
                ax_traces[n].set_xlim(0, len(response_to_one_stim[0]) - 1)

    # Optionally remove ticks from all trace subplots
    for a in ax_traces[1:]:
        a.set_yticks([])
        a.set_xticks([])

    if save_path is not None:
        final_saving_path = create_saving_folder(save_path, one_stim_cell_name)
        plt.savefig(Path(final_saving_path).joinpath('off_target_xy.png'), format="png", dpi=300)
    plt.show()


def plotting_z_off_target_single_cells(df, stim_fishy, trace_type = 'df/f', save_location = None):
    '''
    plotting z off target effects for the closest cell in z to the stimulated cell coordinate (not the bruker cross)
    only single cells above and below the stimulated cell

    :param df: master functional types dataframe
    :param stim_fishy: and example stim fish just to get the photostim frame window value
    :param trace_type: 'df/f' (default) or 'zscore' for type of traces to plot
    :param save_location: parent folder to save these pngs in
    :return: figure
    '''

    stim_df = df[df.photostim == True]
    for n in range(len(stim_df)):
        example_stimmed_cell_info = stim_df.iloc[n]
        stimmed_cell_id = example_stimmed_cell_info.resp_cell_id
        stimmed_cell_coord = example_stimmed_cell_info.neur_coords
        stimmed_cell_plane = example_stimmed_cell_info.plane

        # Build the figure
        n_planes = len(np.unique(df.plane.values))
        fig = plt.figure(figsize=(8, 1.5*n_planes + 2))
        gs = matplotlib.gridspec.GridSpec(n_planes, 2, width_ratios=[1.5, 1], height_ratios=[1]*n_planes, hspace=0.4, wspace=0.6)
        ax3d = fig.add_subplot(gs[0:2, 0], projection='3d')
        ax3d.view_init(20, -45)  # Set elevation and azimuth
        ax2d = fig.add_subplot(gs[2, 0])
        ax = np.empty((n_planes, 2), dtype=object)

        # first gather all the data that I will need
        plot_coords = []
        plot_colors = []
        for p, each_plane in enumerate(np.unique(df.plane.values)):
            one_plane_data_df = df[df.plane == each_plane]
            closest_coordinate, closest_ind = coordutils.closest_coordinates(stimmed_cell_coord[0], stimmed_cell_coord[1],
                                                                             one_plane_data_df.neur_coords.values)
            closest_cell_info = one_plane_data_df.iloc[closest_ind]
            closest_cell_id = closest_cell_info.resp_cell_id
            plot_coords.append((closest_coordinate[0], closest_coordinate[1], p)) # Rebuild full 3D coord for plot
            if stimmed_cell_plane == each_plane:
                plot_colors.append('red')
            elif 'blue' not in plot_colors:
                plot_colors.append('blue')
            else:
                plot_colors.append('green')

            ax[p, 1] = fig.add_subplot(gs[p, 1])
            if trace_type == 'df/f':
                closest_cell_stim_traces = closest_cell_info.stim_responses[stimmed_cell_id][0]
                ax[p, 1].set_ylim(-.5, 1.5)
            if trace_type == 'zscore':
                closest_cell_stim_traces = closest_cell_info.stim_responses_zscore[stimmed_cell_id][0]
                ax[p, 1].set_ylim(-2, 2)
            avg_closest_cell_stim_traces = arrutils.pretty(np.nanmean(closest_cell_stim_traces, axis = 0), 2)
            std_closest_cell_stim_traces = arrutils.pretty(np.nanstd(closest_cell_stim_traces, axis = 0), 2)
            title = each_plane
            if 'stim' in closest_cell_id:
                title = str(each_plane) +', stimulated'
            ax[p, 1].plot(avg_closest_cell_stim_traces, label = closest_cell_id, color = plot_colors[p])
            ax[p, 1].fill_between(range(len(avg_closest_cell_stim_traces)),
                                       avg_closest_cell_stim_traces - std_closest_cell_stim_traces,
                                        avg_closest_cell_stim_traces + std_closest_cell_stim_traces, alpha=0.1, color = plot_colors[p])
            ax[p, 1].set_title(title)
            ax[p, 1].set_ylabel(trace_type)
            ax[p, 1].axvline(x=-stim_fishy.photostim_frame_window[0], color='red', linestyle='-', linewidth=2)
            ax[p, 1].spines['top'].set_visible(False)
            ax[p, 1].spines['right'].set_visible(False)
            ax[p, 1].axhline(0, color = 'grey', linestyle = '--')

        # add coordinates to the 3d volume plot and 2d plot & adjust plots
        plot_coords = np.array(plot_coords)
        ax3d.scatter(plot_coords[:,0], plot_coords[:,1], plot_coords[:,2], c=plot_colors, s=50, depthshade=False)
        ax3d.set_xlim3d(min(plot_coords[:,0]) - 30, max(plot_coords[:,0]) + 30)
        ax3d.set_xlabel('x')
        ax3d.set_ylim3d(max(plot_coords[:,1]) + 30, min(plot_coords[:,1]) - 30)
        ax3d.set_ylabel('y')
        ax3d.set_zlim3d(n_planes-1, 0)
        ax3d.set_zticks(np.linspace(n_planes-1, 0,n_planes))
        ax3d.set_zticklabels(np.unique(df.plane.values)[::-1], rotation=15,)
        ax3d.set_title('XYZ positions')
        ax2d.scatter(plot_coords[:,0], plot_coords[:,1], c=plot_colors, s=50)
        ax2d.set_xlim(min(plot_coords[:,0]) - 30, max(plot_coords[:,0]) + 30)
        ax2d.set_ylim(max(plot_coords[:,1]) + 30, min(plot_coords[:,1]) - 30)
        ax2d.set_title('XY positions')
        ax2d.set_xlabel('x')
        ax2d.set_ylabel('y')

        if save_location is not None:
            final_saving_path = create_saving_folder(save_location, stimmed_cell_id)
            plt.savefig(Path(final_saving_path).joinpath('off_target_z.png'), format="png", dpi=300)
        plt.show()


def plotting_z_off_target_group(df,
                                stim_fishy,
                                trace_type='df/f',
                                radius_um=10,
                                max_cells_per_plane=5,
                                save_location=None):
    '''
    Plotting off target in z effects, but now groups for each plane (not just single cells)

    :param df: functional types df
    :param stim_fishy: example fish from the stimulated fish volume for some params to use in this
    :param trace_type: either 'df/f' or 'zscore' for trace type
    :param radius_um: the radius of the size around the photostimulated neuron for looking at groups
    :param max_cells_per_plane: the number of cells to plot per plane (note: will choose from the closest neurons)
    :param save_location: where to save the figure
    :return: figure
    '''
    stim_df = df[df.photostim == True]
    unique_planes = np.sort(df.plane.unique())
    n_planes = len(unique_planes)

    # Plane-specific colormaps
    plane_colormaps = {0: matplotlib.cm.get_cmap('Blues'),
        1: matplotlib.cm.get_cmap('Greens'),
        2: matplotlib.cm.get_cmap('Oranges')}
    start, stop = 0.3, 0.9  # narrower, more distinct slice, for better color contrast

    for n in range(len(stim_df)):  # for each stimulated cell
        example_stimmed_cell_info = stim_df.iloc[n]
        stimmed_cell_id = example_stimmed_cell_info.resp_cell_id
        stimmed_cell_coord = example_stimmed_cell_info.neur_coords
        stimmed_cell_plane = example_stimmed_cell_info.plane

        all_cells_by_plane = {}
        plot_coords = []
        plot_colors = []

        for plane_idx, p in enumerate(unique_planes):  # go through each plane
            plane_df = df[df.plane == p]
            coords = plane_df.neur_coords.values

            # Find nearby cells and distances
            nearby_inds, nearby_coords = coordutils.determine_nearby_cells_xy(
                stimmed_cell_coord, coords, stim_fishy.um_per_px, radius_um=radius_um
            )

            # Compute distances and sort
            distances = [np.linalg.norm(np.array(stimmed_cell_coord[:2]) - np.array(c)) for c in nearby_coords]
            sorted_data = sorted(zip(distances, nearby_inds, nearby_coords), key=lambda x: x[0])
            sorted_inds = [i for _, i, _ in sorted_data][:max_cells_per_plane]

            all_cells_by_plane[plane_idx] = []

            for i, idx in enumerate(sorted_inds):
                cell_info = plane_df.iloc[idx]
                cell_id = cell_info.resp_cell_id
                coord = cell_info.neur_coords
                plot_coords.append((coord[0], coord[1], plane_idx))
                is_stimmed = p == stimmed_cell_plane and np.allclose(coord, stimmed_cell_coord)
                frac = start + (stop - start) * (i / (max_cells_per_plane - 1))
                color = 'red' if is_stimmed else plane_colormaps[plane_idx](frac)
                plot_colors.append(color)

                # Extract trace
                if trace_type == 'df/f':
                    trace = cell_info.stim_responses[stimmed_cell_id][0]
                else:
                    trace = cell_info.stim_responses_zscore[stimmed_cell_id][0]

                all_cells_by_plane[plane_idx].append({
                    'trace': arrutils.pretty(np.nanmean(trace, axis=0), 2),
                    'std': arrutils.pretty(np.nanstd(trace, axis=0), 2),
                    'color': color,
                    'cell_id': cell_id,
                    'is_stimmed': is_stimmed})

        # Plotting
        n_cols = max_cells_per_plane + 2  # +2 for 3D and 2D plots
        fig = plt.figure(figsize=(3 * n_cols, 2 * n_planes))
        gs = matplotlib.gridspec.GridSpec(n_planes, n_cols, width_ratios=[1.5] + [1] * (n_cols - 1), hspace=0.6,
                                          wspace=0.6)

        for row, p in enumerate(unique_planes):
            # 3D and 2D plots (only once per plane)
            if row == 0:
                # 3D on top, 2D below — same column (col=0), different rows (row=0 and row=1)
                ax3d = fig.add_subplot(gs[0, 0], projection='3d')
                ax3d.view_init(20, -45)

                ax2d = fig.add_subplot(gs[1, 0])
                plot_coords_arr = np.array(plot_coords)
                ax3d.scatter(plot_coords_arr[:, 0], plot_coords_arr[:, 1], plot_coords_arr[:, 2],
                             c=plot_colors, s=50, depthshade=False)
                ax3d.set_xlim3d(min(plot_coords_arr[:, 0]) - 30, max(plot_coords_arr[:, 0]) + 30)
                ax3d.set_ylim3d(max(plot_coords_arr[:, 1]) + 30, min(plot_coords_arr[:, 1]) - 30)
                ax3d.set_zlim3d(n_planes - 1, 0)
                ax3d.set_xlabel('x')
                ax3d.set_ylabel('y')
                ax3d.set_title('XYZ positions')
                ax3d.set_zticks(np.arange(n_planes))
                ax3d.set_zticklabels(unique_planes)

                ax2d.scatter(plot_coords_arr[:, 0], plot_coords_arr[:, 1], c=plot_colors, s=50)
                ax2d.set_xlim(min(plot_coords_arr[:, 0]) - 30, max(plot_coords_arr[:, 0]) + 30)
                ax2d.set_ylim(max(plot_coords_arr[:, 1]) + 30, min(plot_coords_arr[:, 1]) - 30)
                ax2d.set_title('XY positions')
                ax2d.set_xlabel('x')
                ax2d.set_ylabel('y')

            # Plot each trace for this plane
            for col, trace_info in enumerate(all_cells_by_plane[row]):
                ax = fig.add_subplot(gs[row, col + 1])
                trace = trace_info['trace']
                std = trace_info['std']
                color = trace_info['color']
                cell_id = trace_info['cell_id']
                is_stimmed = trace_info['is_stimmed']

                ax.plot(trace, color=color, label=f'{cell_id}')
                ax.fill_between(range(len(trace)), trace - std, trace + std, alpha=0.2, color=color)
                ax.axvline(x=-stim_fishy.photostim_frame_window[0], color='red', linestyle='-', linewidth=2)
                ax.axhline(0, color='grey', linestyle='--')

                if trace_type == 'df/f':
                    ax.set_ylim(-0.5, 1.5)
                else:
                    ax.set_ylim(-2, 2)

                ax.set_title('stimulated' if is_stimmed else cell_id, fontsize=8)
                ax.tick_params(labelsize=6)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

        if save_location is not None:
            final_saving_path = create_saving_folder(save_location, stimmed_cell_id)
            plt.savefig(Path(final_saving_path).joinpath('off_target_z_group.png'), format="png", dpi=300)

        plt.show()


# --- IDENTIFY GROUPS OF NEURONS FOR ANALYSIS --- #
def get_most_and_least_activated_neurons(subset_functional_types_df, stim_cell_id = 'stim_0', num_of_neurons = 25, 
                                         response_window = [-15, 30], evoked_response_window = 5, weights = None):
    """
    Function to find the most activated neurons based on the zscore of the responses.
    Note - these will always be unique neurons from each other in the two output lists
    Args:
        subset_functional_types_df (pd.DataFrame): DataFrame containing all the stimulated cells and their responses.
        stim_cell_id (str): The id of the stimulated cell to get responders from.
        num_of_neurons (int): The number of neurons to get.
        response_window (list): The window of frames before and after the photostimulation event.
        evoked_response_window (int): The number of frames immediately after the photostim event to use 
                                    (as these are the neurons that should be stimulated, look within a narrow window).
    
    Returns:
        top_activated_neurons (list): List of the most activated neurons. (resp cell ids)
        least_activated_neurons (list): List of the least activated neurons.(resp cell ids)
    
    """
    select_trials = subset_functional_types_df[subset_functional_types_df.resp_cell_id == stim_cell_id].self_success_trials.values[0]
    peak_evoked_response_dict = {}
    for cell_num in range(len(subset_functional_types_df)):
        cell_id = subset_functional_types_df.iloc[cell_num].resp_cell_id
        responses_to_all_stim_cells = subset_functional_types_df.iloc[cell_num].stim_responses_zscore
        response_to_stim_cell = responses_to_all_stim_cells[stim_cell_id][0]
        responses_to_self_success_trials = response_to_stim_cell[select_trials]
        evoked_responses = responses_to_self_success_trials[:, -response_window[0]:-response_window[0] + evoked_response_window]
        peak_evoked_responses = np.nanmax(evoked_responses, axis = 1) # max evoked response for each trial
        if weights is None:
            peak_evoked_response_dict[cell_id] = np.nanmean(peak_evoked_responses) # avg peak evoked response across all trials
        else:
            peak_evoked_response_dict[cell_id] = np.average(peak_evoked_responses, weights = weights)

    # sort the peak evoked response dictionary (average peak) to find top responders
    sorted_peak_evoked_response_dict = dict(sorted(peak_evoked_response_dict.items(), key=lambda item: item[1], reverse=True))
    top_activated_neurons = list(sorted_peak_evoked_response_dict.keys())[:num_of_neurons]
    least_activated_neurons = list(sorted_peak_evoked_response_dict.keys())[-num_of_neurons:]

    return top_activated_neurons, least_activated_neurons

def gather_thresholded_responders_and_vals(df,
                                           stim_fishy_vol,
                                           std_thresh=2,
                                           evoked_response_window=8,
                                           response_type='df/f',
                                           color_vals_response_type = 'df/f', # in case this should be different than how you find the neurons
                                           stim_cell_id_lst = None,
                                           use_weights=False):
    '''
    Gather the responder indices in the dataframe, plus the values you can use to plot the activity of each cell on a scatter image
    Use this function with plotting_thresholded_responders_locations_and_traces()

    :param df: entire functional types df
    :param stim_fishy_vol: full stim fish volume
    :param std_thresh: the std dev threshold to set to call the neuron 'responsive to photostim'
    :param evoked_response_window: the window from the photostimulation event that will be used for determining responsiveness
    :param response_type: can be 'df/f' or 'zscore' depending on what is wanted for responsitivity metrics
    :param stim_cell_id_lst: list of cell ids to use for the stimulated cells, default will be all of them
    :param use_weights: whether to use trial_weights or not in determining responsiveness, default will be False

    :return: downstream_responders_indices_dict = dictionary, each stim cell (keys) with a list of the indicies of the
    functional types df that are the thresholded responders & color_values_dict = dictionary, each stim cell (keys) with a list of the average
    evoked response (single val per cell) to use for plotting colors later
    '''
    fish_of_stim_cell = stim_fishy_vol[0] # default fish just to get the windows params
    stim_offset = fish_of_stim_cell.photostim_frame_window[0]
    evoked_end = -stim_offset + evoked_response_window

    if stim_cell_id_lst is None:
        stim_cell_id_lst = df[df.photostim == True].resp_cell_id.values

    if use_weights:
        trial_weights_lst = np.array(df[df.resp_cell_id.isin(stim_cell_id_lst)].trial_weights.values)
    else:
        num_trials = list(df.iloc[0].stim_responses.values())[0][0].shape[0]
        trial_weights_lst = np.array([np.ones(num_trials) for n in range(len(stim_cell_id_lst))])

    downstream_responders_indices_dict = {}
    color_values_dict = {}
    for idx, stim_cell in enumerate(stim_cell_id_lst):
        trial_weights = trial_weights_lst[idx]
        downstream_responder_lst = []
        color_value_lst = []
        for i in range(len(df)):
            if response_type == 'df/f':
                ps_response_per_trial = df.iloc[i].stim_responses[stim_cell][0]
            elif response_type == 'zscore':
                ps_response_per_trial = df.iloc[i].stim_responses_zscore[stim_cell][0]

            if color_vals_response_type == 'df/f':
                colors_vals_ps_response_per_trial = df.iloc[i].stim_responses[stim_cell][0]
            elif color_vals_response_type == 'zscore':
                colors_vals_ps_response_per_trial = df.iloc[i].stim_responses_zscore[stim_cell][0]

            # Compute weighted stdev across trials, weighted by their respective trial weights, for each time point
            baseline_values = ps_response_per_trial[:, :-stim_offset]  # trials x time
            std_per_trial = np.std(baseline_values, axis=1)  # calculate std per trial first
            avg_std_baseline = np.average(std_per_trial, weights=trial_weights)  # find the weighted average std overall
            avg_ps_response = np.average(ps_response_per_trial, axis=0, weights=trial_weights)
            avg_ps_response = arrutils.pretty(avg_ps_response, 2)
            avg_baseline = np.nanmean(avg_ps_response[: -stim_offset])
            avg_evoked = np.nanmean(avg_ps_response[-stim_offset:evoked_end])

            # used for coloring the neurons, basically a normalized response value that takes into account baseline
            color_avg_ps_response = np.average(colors_vals_ps_response_per_trial, axis=0, weights=trial_weights)
            color_value = np.nanmean(color_avg_ps_response[-stim_offset:evoked_end]) - np.nanmean(avg_ps_response[:-stim_offset])

            if avg_evoked >= ((std_thresh * avg_std_baseline) + avg_baseline):
                downstream_responder_lst.append(i)
                color_value_lst.append(color_value)
        downstream_responders_indices_dict[stim_cell] = downstream_responder_lst
        color_values_dict[stim_cell] = color_value_lst

    return downstream_responders_indices_dict, color_values_dict


# --- PLOTTING FUNCTIONS FOR VISUALIZING ACTIVITY OF DATASET --- #
def plotting_pairs_heatmap_and_traces(stimulated_traces, responder_traces, window_frames, img_hz, 
                                      optional_figsuptitle = None, ylims = None, vmin = -3, vmax = 3,
                                      savepath = None, sorted_trials = True):
    '''
    Plotting the trace and heatmap of every trial for pairs of stimulated and responder cells
    stimulated_traces - the traces of the stimulated cell for each trial, should be in dF/F
    responder_traces - the traces of the responder cell for each trial, should be in dF/F
    window_frames - the window of frames to look at before and after the photostimulation event
    img_hz - the hz of the imaging
    optional_figsuptitle - optional title for the figure
    vmin - the minimum value for the heatmap
    vmax - the maximum value for the heatmap

    returns a figure
    '''
    
    fig, ax = plt.subplots(2, 2, figsize = (8, 7), sharey='row', gridspec_kw={'height_ratios': [1, 2]})

    stimulated_event_frame = -window_frames[0]
    
    if sorted_trials:
        sorted_trial_numbers = [i for i in np.argsort(np.nanmedian(stimulated_traces[:, stimulated_event_frame:], axis = 1))][::-1]
        stimulated_traces = np.array([stimulated_traces[i] for i in sorted_trial_numbers])
        responder_traces = np.array([responder_traces[i] for i in sorted_trial_numbers])
        ylabel = 'sorted stimulation trials'
    else:
        ylabel = 'stimulation trials'
    if ylims is None:
        ymax = np.nanmax([np.nanmax(stimulated_traces), np.nanmax(responder_traces)])
        ymin = np.nanmin([np.nanmin(stimulated_traces), np.nanmin(responder_traces)])
    else:
        ymax = ylims[1]
        ymin = ylims[0]

    for n, df_f_traces in enumerate([stimulated_traces, responder_traces]):
        if n == 0:
            name = 'stimulated'
        else:
            name = 'responder'
        [ax[0,n].plot(np.arange(len(m)), arrutils.pretty(m, 2), color = 'grey', alpha = 0.3) for m in df_f_traces]
        ax[0,n].plot(np.arange(len(df_f_traces[0])), arrutils.pretty(np.nanmean(df_f_traces, axis = 0), 2), color = 'k')
        ci_lower, ci_upper = statutils.calculate_ci(df_f_traces)
        ax[0,n].fill_between(np.arange(len(df_f_traces[0])), ci_lower, ci_upper, color='skyblue', alpha=0.4, label='95% CI') 
        ax[0,n].set_ylabel('df/f')
        ax[0,n].axvline(x = stimulated_event_frame, color = 'red')
        ax[0,n].axhline(0, color = 'grey', linestyle = '--')
        ax[0,n].set_title(name)
        ax[0,n].set_ylim(ymin, ymax)
        ax[0,n].spines['top'].set_visible(False)
        ax[0,n].spines['right'].set_visible(False)

        sns.heatmap(df_f_traces, ax = ax[1,n], yticklabels=sorted_trial_numbers, cmap = 'coolwarm', cbar_kws={'label': 'df/f'}, vmin=vmin, vmax=vmax)
        ax[1,n].axvline(x = window_frames[0], color = 'black')
        ax[1,n].set_ylabel(ylabel)
        ax[1,n].axvline(x = stimulated_event_frame, color = 'red')
        
        [a.set_xticks([0, -window_frames[0] , np.diff(window_frames)[0]-1], (np.array([window_frames[0], 0, np.diff(window_frames)[0]-1]) 
                                                  * img_hz).astype(int)) for a in ax.flatten()]
        [a.set_xlabel('time (sec)') for a in ax.flatten()]
    
    if optional_figsuptitle:
        fig.suptitle(optional_figsuptitle)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi = 300, bbox_inches = 'tight')
    
    return ax

def plotting_pairs_location_of_cells(fish_img, stimulated_roi, responder_roi, responder_alphas = None, optional_title = None, show_text = False, show_legend = True):
    '''
    Plotting location of the stimulated cells and either one or multiple responding cells
    fish_img - the image of the fish, background of this image
    stimulated_roi - the location of the stimulated cell
    responder_roi - the location of the responding cell(s)
    responder_alphas - the alpha of the responding cell(s), likely motor correlation values
    optional_title - optional title for the figure
    show_text - show the text of the responding cell index
    show_legend - show the legend

    returns a figure
    '''

    plt.figure(figsize = (9, 9))
    plt.imshow(fish_img, cmap = 'gray', vmax = np.percentile(fish_img, 99))
    plt.scatter(stimulated_roi[0], stimulated_roi[1], edgecolor = 'tab:red', s = 50, linewidth=2, label = 'stimulated', facecolors='none')
    
    alphas = [1] * len(responder_roi)
    if responder_alphas is not None:
        alphas = responder_alphas
    if len(responder_roi) > 1:
        for i, r in enumerate(responder_roi):
            plt.scatter(r[0], r[1], edgecolor = 'tab:blue', s = 50, linewidth=2, label = f'responder_{i}', 
            facecolors='tab:green', alpha = alphas[i])
            if show_text:
                plt.annotate(f'{i}', (r[0], r[1]), color = 'white')
    else:
        plt.scatter(responder_roi[0], responder_roi[1], edgecolor = 'tab:blue', s = 50,
        linewidth=2, label = 'responder', facecolors='tab:green', alpha = alphas[0])   
    
    if optional_title:
        plt.title(optional_title)

    if show_legend:
        plt.legend(bbox_to_anchor=(1.05, 1))
    else:
        plt.legend('',frameon=False)
    plt.tight_layout()
    plt.axis('off')
    
    return plt.show()

def plotting_time_series_of_responders(raw_fluor_responding_traces, array_stimulated_frames, labels_per_neuron = None, img_hz = 1, length_of_plot = None):
    '''
    Plotting the large time series of the responding neurons with the stimulated frames marked
    raw_fluor_r - the raw traces of the responding neurons, shape = (neurons, entire trace)
    array_stimulated_frames - the array of stimulated frames, shape = (stimulated neurons, frames)
    labels_per_neuron - the labels of the neurons
    img_hz - the hz of the recording
    length_of_plot - the length of the plot in frames (x -axis)

    returns a figure
    '''
    if length_of_plot is None:
        length_of_plot = len(raw_fluor_responding_traces[0])

    fig, ax = plt.subplots(figsize = (15, 8))

    random_colors = plotutils.make_color_list_from_cmap(array_stimulated_frames.shape[0], colormap = plt.cm.rainbow)

    for n, r in enumerate(raw_fluor_responding_traces):
        r = arrutils.norm_0to1(r)
        plt.plot(np.arange(len(r)), arrutils.pretty(r + n, 2), linewidth = 1, color = 'k')

    for s, s_frames in enumerate(array_stimulated_frames):
        stimmed_neuron = int(s)
        color = random_colors[s]
        [plt.axvline(w, color = color, alpha = 0.5) for w in s_frames]
        [plt.text(w, 1.01, stimmed_neuron, color=color, ha='center', va='center', rotation=90, fontsize=8,
                transform=ax.get_xaxis_transform()) for w in s_frames if w < length_of_plot]
    ax.spines[['top', 'right']].set_visible(False)
    if labels_per_neuron is None:
        y_labels_per_neuron = range(n+1)
    else:
        y_labels_per_neuron = labels_per_neuron
    plt.yticks(ticks=[0.5 + x for x in range(n+1)], labels=y_labels_per_neuron, fontsize=8,)
    plt.xticks(ticks =np.arange(0, length_of_plot + 500, 500), labels = [int(i) for i in np.arange(0, length_of_plot + 500, 500) * img_hz])
    plt.xlabel('time (sec)')
    plt.ylabel('responding neuron')
    plt.xlim(0, length_of_plot-5)

    return ax    

def plotting_vizmotion_and_photostimulation_responses(vizmotion_photostim_info_df,
                                                      lst_motion_cues = None,
                                                      lst_motion_cues_colors = None,
                                                      response_type = 'df/f',
                                                      img_hz = 1,
                                                      sec_motion_on = 5,
                                                      weights = None,
                                                      vizmotion_stim_offset = 9,
                                                      photostim_window_frames = [-15, 30],
                                                      number_stim_response_panels = None,
                                                      photostim_ylim = [-0.5, 1.2],
                                                      photostim_stim_offset = 0,
                                                      filtered_stim_trials = None,
                                                      save_location = None):
    '''
    Plot responses to all the visual motion cues and photostimulated neurons for neurons in the dataframe
    Make the vizmotion_photostim_info_df from the above function 'build_vizmotion_photostim_info_df' 
    Currently makes a figure with 6 motion responsive panels and 'n' photostimulation response panels (depends on how many neurons were stimulated)

    vizmotion_photostim_info_df - the dataframe including the motion, photostim responses, average evoked activity
    lst_motion_cues - list, the visual motion stimuli to plot
    lst_motion_cues_colors - list, the colors associated with each visual motion stimulus
    img_hz - the hz of the imaging speed of the OMR fishvolume
    response_type - the type of response trace to plot ('df/f' or 'zscore')
    vizmotion_stim_offset - the offset of the photostimulation event in the visual motion responses dataset (fish.stim_offset)
    photostim_window_frames - the window of frames to look at before and after the photostimulation event (chosen arbitrarily, how many frames to look at)
    photostim_ylim - the y-axis limits of the photostimulation responses plots
    photostim_stim_offset - the offset of the photostimulation event in the plot (in frames)
    filtered_stim_trials - the filtered trials to plot, if None, plot all trials; 
                            needs to be a dictionary with the stim cell id as the keys and the good trials as numbers in a list in the values
    save - bool, if you want to save the figure
    save_location - Path, where to save the figure

    returns a figure
    
    '''

    if lst_motion_cues is None:
        lst_motion_cues = ['medial_left', 'lateral_left', 'left', 'medial_right', 'lateral_right', 'right', 'converging', 'diverging',
               'forward_backward', 'forward_x', 'x_backward', 'backward_forward', 'x_forward', 'backward_x',  
               'forward', 'backward']
    if lst_motion_cues_colors is None:
        lst_motion_cues_colors = ['tab:green', 'tab:purple', 'black', 'tab:green', 'tab:purple', 'black', 'c', 'm', 
                      'black', 'teal', 'tab:orange', 'black', 'teal', 'tab:orange',
                       'teal', 'tab:orange']

    frames_motion_on = sec_motion_on * img_hz
    if number_stim_response_panels is None:
        panel_num = 6 + len(vizmotion_photostim_info_df.avg_evoked_df_f.values[0])
    else:
        panel_num = 6 + number_stim_response_panels

    for i in range(len(vizmotion_photostim_info_df)):
        vizmotion_responses = vizmotion_photostim_info_df.motion_responses.iloc[i]
        if response_type == 'df/f':
            photostim_responses = vizmotion_photostim_info_df.stim_responses.iloc[i]
        elif response_type == 'zscore':
            photostim_responses = vizmotion_photostim_info_df.stim_responses_zscore.iloc[i]
        else: # default is df/f
            photostim_responses = vizmotion_photostim_info_df.stim_responses.iloc[i]
        omr_cell_id = vizmotion_photostim_info_df.omr_neur_id.iloc[i]
        plane_id = vizmotion_photostim_info_df.plane.iloc[i]
        viz_barcode = vizmotion_photostim_info_df.visual_barcode.iloc[i]

        resp_cell_id_name = vizmotion_photostim_info_df.resp_cell_id.iloc[i]
        if 'resp' in resp_cell_id_name:
            val = resp_cell_id_name.split('_')[1]
            title = 'Responder #' + val
        else:
            val = resp_cell_id_name.split('_')[1]
            title = 'Stimulated #' + val
        
        fig, ax = plt.subplots(1, panel_num, figsize = (20, 1))

        [a.axvspan(vizmotion_stim_offset, frames_motion_on + vizmotion_stim_offset, color = 'grey', alpha = 0.05) for a in ax[:6].flatten()]

        vizstim_y_max = []
        vizstim_y_min = []
        for m, vizstim_cue in enumerate(lst_motion_cues):
            if 'left' in lst_motion_cues[m]:
                select_ax = 0
                select_title = 'Lefts'
            if 'right' in lst_motion_cues[m]:
                select_ax = 1
                select_title = 'Rights'
            if 'ing' in lst_motion_cues[m]:
                select_ax = 2
                select_title = 'Conv./Div.'
            if ('forward' == lst_motion_cues[m]) or ('backward' == lst_motion_cues[m]):
                select_ax = 3
                select_title = 'For./Back.'
            if lst_motion_cues[m] in ['forward_backward', 'forward_x', 'x_backward']:
                select_ax = 4
                select_title = 'CW Shear'
            if lst_motion_cues[m] in ['backward_forward', 'x_forward', 'backward_x',]:
                select_ax = 5
                select_title = 'CCW Shear'
            
            ax[select_ax].plot(vizmotion_responses['mean'][vizstim_cue], color = lst_motion_cues_colors[m])
            ax[select_ax].fill_between(np.arange(len(vizmotion_responses['mean'][vizstim_cue])), 
                                    vizmotion_responses['mean'][vizstim_cue] - vizmotion_responses['std'][vizstim_cue], 
                                    vizmotion_responses['mean'][vizstim_cue] + vizmotion_responses['std'][vizstim_cue], 
                                    color = lst_motion_cues_colors[m], alpha = 0.1)
            ax[select_ax].set_title(select_title, rotation = 30, fontsize = 10)
            vizstim_y_max.append(np.nanmax(vizmotion_responses['mean'][vizstim_cue]))
            vizstim_y_min.append(np.nanmin(vizmotion_responses['mean'][vizstim_cue]))

        plot_ind = 6
        for s, stim_cell_id in enumerate(photostim_responses.keys()):
            response_to_stim = photostim_responses[stim_cell_id]
            if isinstance(response_to_stim, (list, np.ndarray)) and not isinstance(response_to_stim[0], (float, int)):
                # some random filtering for best use due to data types and shapes 
                if np.array(response_to_stim).shape[1] == np.diff(photostim_window_frames)[0]:
                    response_to_stim = response_to_stim
                else:      
                    response_to_stim = response_to_stim[0]

                if filtered_stim_trials is not None: # filter the tirals for more accurate response profiles
                    try:
                        good_trial_for_stim_cell = filtered_stim_trials[stim_cell_id]
                        if not isinstance(good_trial_for_stim_cell, (float, int)):
                            response_to_stim = [m for n, m in enumerate(response_to_stim) if n in good_trial_for_stim_cell]
                        else:
                            response_to_stim = np.nan
                    except:
                        response_to_stim = np.nan
                if weights is not None:
                    avg_response, ci_lower, ci_upper = statutils.weighted_avg_and_ci(response_to_stim, weights = weights[stim_cell_id])
                else:
                    avg_response = np.nanmean(response_to_stim, axis = 0)
                    ci_lower, ci_upper = statutils.calculate_ci(np.array(response_to_stim))

                if not isinstance(response_to_stim, (float, int)): # make sure that this is not nan's
                    [ax[plot_ind].plot(np.arange(len(m)), arrutils.pretty(m, 2), color = 'grey', alpha = 0.3) for m in response_to_stim]
                    ax[plot_ind].plot(np.arange(len(response_to_stim[0])), arrutils.pretty(avg_response, 2), color = 'k')
                    ax[plot_ind].fill_between(np.arange(len(response_to_stim[0])), ci_lower, ci_upper, color='skyblue', alpha=0.4, label='95% CI') 
                    ax[plot_ind].axvline(x = -photostim_window_frames[0] + photostim_stim_offset, color = 'red')
                    ax[plot_ind].axhline(0, color = 'grey', linestyle = '--')
                    ax[plot_ind].set_title(stim_cell_id, rotation = 30, fontsize = 10)
                    ax[plot_ind].set_ylim(photostim_ylim[0], photostim_ylim[1])
                    ax[plot_ind].set_xlim(0, len(response_to_stim[0])-1)
                    plot_ind += 1

        ax[0].set_ylabel('dF/F')
        if type is None:
            ax[6].set_ylabel('dF/F')
        elif type == 'zscore':
            ax[6].set_ylabel('zscore')
        [a.set_xlabel('time (s)') for a in ax.flatten()]
        [a.set_xticks(ticks = [int(i) for i in np.linspace(0, len(vizmotion_responses['mean'][vizstim_cue])-1, 2)], 
                        labels = [int(i) for i in np.linspace(0, len(vizmotion_responses['mean'][vizstim_cue])-1, 2) / img_hz]) for a in ax[:6].flatten()]
        [a.set_xticks(ticks = [int(i) for i in np.linspace(0, np.diff(photostim_window_frames)-1, 2)], 
                        labels = [int(i) for i in np.linspace(0, np.diff(photostim_window_frames)-1, 2) / img_hz]) for a in ax[6:].flatten()]
        
        # to normalize my vizstim responses between 0 and 1
        [a.set_ylim(min(vizstim_y_min), max(vizstim_y_max)) for a in ax[:6].flatten()]
        ax[0].set_yticks([min(vizstim_y_min), max(vizstim_y_max)])
        ax[0].set_yticklabels(['0','1'])

        [a.spines['top'].set_visible(False) for a in ax.flatten()]
        [a.spines['right'].set_visible(False) for a in ax.flatten()]
        [a.tick_params(axis='y', left=False, labelleft=False) for a in ax[1:6].flatten()]
        [a.tick_params(axis='y', left=False, labelleft=False) for a in ax[7:].flatten()]
        
        if 'ranked_resp' in vizmotion_photostim_info_df.columns:
            title = 'Ranked #' + vizmotion_photostim_info_df.ranked_resp.iloc[i]
            resp_cell_id_name = 'ranked_' + vizmotion_photostim_info_df.ranked_resp.iloc[i] + ' ' + resp_cell_id_name

        figure_title = f'{title} omr #{omr_cell_id}, {plane_id}, {viz_barcode}'
        fig.suptitle(figure_title, fontsize = 16, y = 1.8)
        fig.subplots_adjust(wspace=0.5)
    
        if save_location is not None:
            save_dir = create_saving_folder(save_location, resp_cell_id_name)
            if response_type == 'df/f': # the backslash causes issues for the save path
                response_type = 'df_f'
            save_name = f'vizmotion_photostim_responses_{response_type}.png'
            plt.savefig(Path(save_dir).joinpath(save_name), dpi = 300, bbox_inches = 'tight')
            save_name = f'vizmotion_photostim_responses_{response_type}.svg'
            plt.savefig(Path(save_dir).joinpath(save_name), dpi = 300, bbox_inches = 'tight')
        
        plt.show()
    
    return print('done')

def plotting_correlation_maps_avg_photostim_evoked_resp(vizmotion_photostim_info_df, omr_fish_volume, color_map = None, limits = [-0.3, 0.3], 
                                                        stimulation_cell_color = 'limegreen', title = None, save = False, save_location = None):
    '''
    Plotting the average evoked activity to all photostimulated cells in the volume
    Hard coded right not to plot the average evoked activity that is in the dataframe, probably not useful and need to change for future renditions

    vizmotion_photostim_info_df - the dataframe with the photostimulation responses (average evoked activity), responder/stimulated cell ids, neuron coordinates
    omr_fish_volume - VolumeFish object, the OMR dataset (all the planes together)
    color_map - the colormap to use for the correlation maps (default is the custom color map from red to blue)
    limits - the limits of the color map (vmin and vmax values)
    stimulation_cell_color - the color of the stimulated cells on the map (not evoked activity)
    save - bool, if you want to save the figure
    save_location - Path, where to save the figure
    
    '''
    if color_map is None:
        color_map = plotutils.build_cmap_blue_to_red()

    fig, ax = plt.subplots(1, len(omr_fish_volume.volumes.keys()), figsize = (20, 20))
    fig2, ax2 = plt.subplots(1, figsize = (12, 12))
    for p, plane in enumerate(omr_fish_volume.volumes.keys()):
        one_plane_responder_df = vizmotion_photostim_info_df[(vizmotion_photostim_info_df.plane == plane) & 
                                                             (vizmotion_photostim_info_df.resp_cell_id.str.contains('resp'))]
        one_plane_responder_df = one_plane_responder_df.iloc[one_plane_responder_df['avg_evoked_df_f'].apply(lambda x: abs(np.nanmean(x))).argsort()]
        one_plane_stimulated_df = vizmotion_photostim_info_df[(vizmotion_photostim_info_df.plane == plane) 
                                                              & (vizmotion_photostim_info_df.resp_cell_id.str.contains('stim'))]
        one_fish = omr_fish_volume[p]
        ax[p].imshow(one_fish.rescaled_ref, cmap = 'gray', vmax = np.percentile(one_fish.rescaled_ref, 99))

        if p == int(len(omr_fish_volume)/2): # choose a middle plane for plotting the volume background image
            ax2.imshow(one_fish.rescaled_ref, cmap = 'gray', vmax = np.percentile(one_fish.rescaled_ref, 99))

        resp_vals = [np.nanmean(v) for v in one_plane_responder_df.avg_evoked_df_f.values]
        resp_colors = plotutils.clip_and_map_colors(resp_vals, vmin= limits[0], vmax= limits[1], cmap_name=color_map)
        for k in range(len(one_plane_responder_df)):
            ax[p].scatter(one_plane_responder_df.neur_coords.values[k][0], one_plane_responder_df.neur_coords.values[k][1], s = 10, 
                        color = resp_colors[k], zorder=2)
        if len(one_plane_stimulated_df) > 0:
            for l in range(len(one_plane_stimulated_df)):
                ax[p].scatter(one_plane_stimulated_df.neur_coords.values[l][0], one_plane_stimulated_df.neur_coords.values[l][1], s = 10, 
                            color = stimulation_cell_color, zorder=3)
        ax[p].set_title(f'{plane}')

    volume_responder_df = vizmotion_photostim_info_df[(vizmotion_photostim_info_df.resp_cell_id.str.contains('resp'))]
    volume_responder_df = volume_responder_df.iloc[volume_responder_df['avg_evoked_df_f'].apply(lambda x: abs(np.nanmean(x))).argsort()] 
    volume_resp_vals = [np.nanmean(v) for v in volume_responder_df.avg_evoked_df_f.values]
    volume_resp_colors = plotutils.clip_and_map_colors(volume_resp_vals, vmin=limits[0], vmax=limits[1], cmap_name=color_map)
    for c, coord in enumerate(volume_responder_df.neur_coords.values):
        ax2.scatter(coord[0], coord[1], s = 90, color = volume_resp_colors[c], zorder=2)
            
    volume_stimulated_df = vizmotion_photostim_info_df[(vizmotion_photostim_info_df.resp_cell_id.str.contains('stim'))]
    for coord in volume_stimulated_df.neur_coords.values:
        ax2.scatter(coord[0], coord[1], s = 90, color = stimulation_cell_color, zorder=3)
        
    [a.axis('off') for a in ax.flatten()]
    ax2.axis('off')
    if title is None:
        title = 'volume'
    else:
        title = title
    ax2.set_title(title)
    
    if save:
        fig.savefig(save_location.joinpath('correlation_maps_avg_photostim_evoked_resp_individual_planes.png'), dpi = 300, bbox_inches = 'tight')
        fig2.savefig(save_location.joinpath('correlation_maps_avg_photostim_evoked_resp_volume.png'), dpi = 300, bbox_inches = 'tight')

    fig.show()
    fig2.show()

    return print('done')

def plotting_group_location_vizmotion_photostim_responses(functional_types_df, photostim_cell_id, cell_ids_of_group, fish_brain_image, group_name, 
                                                          evoked_response_window = 5, r_type = 'peak', photostim_start_frame = 15, 
                                                          colors_range = [-1, 2], vizmotion_limits = [0,7], weights = None, save_path = None):
    """
    Function to plot the location of the neurons and their responses to photostimulation and visual motion.
    Args:
        functional_types_df (pd.DataFrame): DataFrame containing all the stimulated cells and their responses.
        photostim_cell_id (str): The id of the stimulated cell to get responders from.
        cell_ids_of_group (list): List of the cell ids of the group to plot.
        fish_brain_image (ndarray): The image of the fish to plot on.
        group_name (str): The name of the group to plot.
        r_type (str): The type of response to plot ('peak' or 'mean') for the colors of the neurons.
        colors_range (str): The range of zscore limits for the color map on the heatmap and colors for the location of the neurons
        evoked_response_window (int): The time window to look at for the evoked response.
        photostim_start_frame (int): The frame at which the photostimulation event occurs. (determined from how the functional types df was made)
        weights (list): The weights to use for the peak evoked response calculation. (specific to the photostimmed cell)
        save_path (str): Path to save the figure.
    """
        
    plotting_specific_cells_df = functional_types_df[functional_types_df.resp_cell_id.isin([photostim_cell_id] + cell_ids_of_group)]
    stimmed_cell_vis_barcode = functional_types_df[functional_types_df.resp_cell_id == photostim_cell_id].visual_barcode.values[0]
    self_success_trials = functional_types_df[functional_types_df.resp_cell_id == photostim_cell_id].self_success_trials.values[0]
    raster_array = np.zeros((len(plotting_specific_cells_df)-1, len(plotting_specific_cells_df.iloc[0].stim_responses_zscore[photostim_cell_id][0][0])))
    stim_array = np.zeros((1, len(plotting_specific_cells_df.iloc[0].stim_responses_zscore[photostim_cell_id][0][0])))
    responder_values_for_plotting = []
    # gather and organize traces from photostimulation events
    for i in range(len(plotting_specific_cells_df)):
        cell_id = plotting_specific_cells_df.iloc[i].resp_cell_id
        responses_to_all_stim_cells = plotting_specific_cells_df.iloc[i].stim_responses_zscore
        response_to_stim_cell = responses_to_all_stim_cells[photostim_cell_id][0]
        responses_to_self_success_trials = response_to_stim_cell[self_success_trials]
        _, avg_evoked_activity = gather_evoked_activity_for_select_trials(responses_to_self_success_trials, r_type = r_type, weights = weights, 
                                                                          immediate_response_window= evoked_response_window) # over a certain time window
        responder_values_for_plotting.append(avg_evoked_activity)
        if 'stim' not in cell_id:
            raster_array[i] = arrutils.pretty(np.nanmean(responses_to_self_success_trials, axis = 0), 2)
        else:
            stim_array[0] = arrutils.pretty(np.nanmean(responses_to_self_success_trials, axis = 0), 2)
    sorted_raster_array, sorted_raster_array_inds = arrutils.sort_array_by_max(raster_array) # sort the responses by their max response post ps event
    sorted_raster_array = np.concatenate([sorted_raster_array, stim_array], axis = 0)
    sorted_inds_with_stim = np.concatenate([sorted_raster_array_inds, [len(plotting_specific_cells_df)-1]], axis = 0)
    
    fig, ax = plt.subplots(1, 3, figsize = (15, 5)) # make figure

    # location of the neurons
    ax[0].imshow(fish_brain_image, cmap = 'gray', vmax = np.percentile(fish_brain_image, 99))
    coords = plotting_specific_cells_df.neur_coords.values[sorted_inds_with_stim]
    color_map = plotutils.build_cmap_blue_to_red()
    responder_colors_for_plotting = plotutils.clip_and_map_colors(responder_values_for_plotting, vmin=colors_range[0], vmax=colors_range[1], cmap_name=color_map)
    responder_colors_for_plotting_sorted = [responder_colors_for_plotting[i] for i in sorted_inds_with_stim]
    for n, c in enumerate(coords):
        marker = 'o'
        faceclr = responder_colors_for_plotting_sorted[n]
        if plotting_specific_cells_df.iloc[n].resp_cell_id == photostim_cell_id: # if the stimmed cell, then mark with a red x
            marker = 'x'
            faceclr = 'red'
        edgeclr = None
        ax[0].scatter(c[0], c[1], facecolor = faceclr, edgecolor = edgeclr, marker = marker, s = 30)
        ax[0].annotate(str(n), xy = (c[0], c[1]),xytext=(c[0], c[1]), color = 'white', fontsize = 8)
    ax[0].axis('off')
    ax[0].set_title(f'{group_name} responders to {photostim_cell_id} ({stimmed_cell_vis_barcode})')

    # raster plot of all the photostimulation trials, last one is the stimmed cell
    sns.heatmap(sorted_raster_array, cmap = 'bwr', cbar = True, center=0, vmin = colors_range[0], vmax = colors_range[1], ax = ax[1])
    ax[1].axvline(photostim_start_frame, color = 'black')
    ax[1].set_title(f'averaged photostim responses (zscore)')
    ax[1].set_xlabel('frames')
    ax[1].set_ylabel('neurons')

    # motion responses of all the neurons, last one is the stimmed cell
    len_one_motion_response_array = len(plotting_specific_cells_df.iloc[0].motion_responses['mean'][constants.photostim_motion_cues[0]])
    all_motion_responses_array = np.zeros((len(plotting_specific_cells_df), len(constants.photostim_motion_cues)*len_one_motion_response_array))
    for i in sorted_inds_with_stim:
        array_lst = []
        for m in constants.photostim_motion_cues:
            array = plotting_specific_cells_df.iloc[i].motion_responses['mean'][m]
            array_lst.append(array)
        concat_array = np.concatenate(array_lst, axis = 0)
        all_motion_responses_array[i] = concat_array
    start_motion_frames = stimuli.stimulus_start_frames_for_plots(frames_motion_on= 7,  # used offsets in the photostim fishy
                                    length_of_total_frame_arr = len_one_motion_response_array,
                                    number_of_stims_in_set = len(constants.photostim_motion_cues))
    sns.heatmap(all_motion_responses_array, cmap = 'gray_r', cbar = True, vmin = vizmotion_limits[0], vmax = vizmotion_limits[1], ax = ax[2])
    stimuli.flexible_stim_shader(start_motion_frames, constants.photostim_motion_cues, 7, subplot = ax[2], ylim = 0.9)
    ax[2].set_title(f'averaged visual motion responses (df/f)')
    ax[2].xaxis.set_major_locator(plt.NullLocator())
    ax[2].set_xticks([])
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(Path(save_path).joinpath(f'{photostim_cell_id}_{group_name}.png'), dpi = 300)
    else:
        plt.show()
    plt.show()

    return print('done')

def plotting_thresholded_responders_locations_and_traces(df,
                                                        stim_fishy_vol,
                                                        responder_inds_dict = None,
                                                        vals_for_colors_dict = None,
                                                        std_thresh=2,
                                                        evoked_response_window=8,
                                                        use_weights = False,
                                                        response_type='df/f',
                                                        color_map = 'coolwarm',
                                                        vmin = -1,
                                                        vmax = 3,
                                                        save_location = None):
    '''
    Plotting location of neurons per plane, with heatmap responses as well
    '''

    # gather the thresholded responders and vals
    if responder_inds_dict is None:
        responder_inds_dict, vals_for_colors_dict = gather_thresholded_responders_and_vals(df,
                                                                                           stim_fishy_vol,
                                                                                           std_thresh=std_thresh,
                                                                                           evoked_response_window=evoked_response_window,
                                                                                           response_type=response_type,
                                                                                           stim_cell_id_lst=None,
                                                                                           use_weights=use_weights)

    # develop norm/map for colors for scatter plot
    norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # center it around 0
    cmap = plt.get_cmap(color_map)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    for stim_cell, resp_idx in responder_inds_dict.items():
        subset_df = df.iloc[resp_idx]
        if len(subset_df) > 3:  # making sure there is enough to plot
            coord_of_stim_cell = df[df.resp_cell_id == stim_cell].neur_coords.values[0]
            plane_of_stim_cell = df[df.resp_cell_id == stim_cell].plane.values[0]
            cell_colors = mappable.to_rgba(vals_for_colors_dict[stim_cell])

            # prep the figure
            fig = plt.figure(figsize=(20, 15))
            gs = matplotlib.gridspec.GridSpec(2, 3, height_ratios=[1, 1.5])
            aximages = [fig.add_subplot(gs[0, i]) for i in range(3)]
            axheatmapavg = fig.add_subplot(gs[1, 0])
            axheatmaptrials = fig.add_subplot(gs[1, 1:])
            [aximages[n].imshow(fish.rescaled_ref, cmap='gray', vmax=np.percentile(fish.rescaled_ref, 99.9), alpha=0.8)
             for n, fish in enumerate(stim_fishy_vol)]
            aximages[1].set_title(f'{stim_cell}')
            aximages[int(plane_of_stim_cell.split('_')[1])].scatter(coord_of_stim_cell[0], coord_of_stim_cell[1],
                                                                    s=25,
                                                                    color='limegreen')  # plot the location of the stimulated cell
            for a in aximages:
                a.set_xticks([])
                a.set_yticks([])
                a.set_ylim(512, 0)
                a.set_xlim(0, 512)

            all_downstream_responses = []
            for cell in range(len(subset_df)):
                resp_cell_coordinates = subset_df.iloc[cell].neur_coords
                resp_cell_plane = subset_df.iloc[cell].plane
                ps_response_per_trial = subset_df.iloc[cell].stim_responses[stim_cell][0]
                all_downstream_responses.append(ps_response_per_trial)

                subplot = int(resp_cell_plane.split('_')[1])
                aximages[subplot].scatter(resp_cell_coordinates[0], resp_cell_coordinates[1], s=15,
                                          color=cell_colors[cell])
                aximages[subplot].text(resp_cell_coordinates[0], resp_cell_coordinates[1] - 10, str(cell),
                                       color='white', fontsize=10, ha='center', va='center')

            avg_heatmap_data = [arrutils.pretty(np.nanmean(traces, axis=0), 2) for traces in all_downstream_responses]
            avg_heatmap = sns.heatmap(avg_heatmap_data, ax=axheatmapavg, vmin=vmin, vmax=vmax, cmap=color_map, center=0,
                                      cbar_kws={"aspect": 50})
            avg_heatmap.collections[0].colorbar.set_label(response_type, fontsize=12, rotation=270, )
            axheatmapavg.axvline(-stim_fishy_vol[0].photostim_frame_window[0], color='r')
            axheatmapavg.set_title('average response')

            trial_heatmap_data = [np.concatenate(data) for data in all_downstream_responses]
            trial_heatmap = sns.heatmap(trial_heatmap_data, ax=axheatmaptrials, vmin=vmin, vmax=vmax, cmap=color_map,
                                        center=0, cbar_kws={"aspect": 50})
            trial_heatmap.collections[0].colorbar.set_label(response_type, fontsize=12, rotation=270, )
            # gather the number of ps events from the first example downstream responder traces
            ps_event_lines = np.linspace(-stim_fishy_vol[0].photostim_frame_window[0],
                                         len(trial_heatmap_data[0]) - stim_fishy_vol[0].photostim_frame_window[1],
                                         all_downstream_responses[0].shape[0])
            [axheatmaptrials.axvline(line, color='r') for line in ps_event_lines]
            axheatmaptrials.set_title('concatenated trial responses')

            # Build legend entries mapping 'cell' index to resp_cell_id
            legend_labels = [f"{cell}: {subset_df.iloc[cell].resp_cell_id}"for cell in range(len(subset_df))]
            handles = [matplotlib.lines.Line2D([0], [0], marker='o', color='w', label=label,
                                               markerfacecolor=cell_colors[cell], markersize=8)
                       for cell, label in enumerate(legend_labels)]
            fig.legend(handles=handles,
                       loc='lower center',  # or 'center right'
                       bbox_to_anchor=(0.5, -0.05),  # move below the figure
                       ncol=8,  # number of columns (adjust to fit)
                       fontsize=10,
                       title="Responder Cell IDs",
                       frameon=True)

            if save_location is not None:
                final_saving_path = create_saving_folder(save_location, stim_cell)
                plt.savefig(Path(final_saving_path).joinpath('threshold_responders_locations_and_traces.png'), format="png", dpi=300)
            plt.show()


def plotting_thresholded_responders_per_region_and_photostim_barcode(df,
                                                                     stim_fishy_vol,
                                                                     matching_barcode='None',
                                                                     evoked_response_window=8,  # about 2 secs
                                                                     std_thresh=1.8,
                                                                     response_type='df/f',
                                                                     use_weights = False,
                                                                     regions=['Pt', 'nMLF', 'Hb'],
                                                                     color_map='coolwarm',
                                                                     vmin=-1,
                                                                     vmax=3,
                                                                     save_location=None):
    # set up the saving path
    data_folder_path = stim_fishy_vol[0].folder_path.parents[1].joinpath('analysis_of_responders')
    # make an analysis_of_responders folder in this path to keep all that data together
    os.makedirs(data_folder_path, exist_ok=True)

    # develop norm/map for colors for scatter plot
    norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # center it around 0
    cmap = plt.get_cmap(color_map)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    # gathering the midlines & add it to the df
    midline_paths = pathutils.pathcrawler(stim_fishy_vol[0].folder_path.parents[3], inset=set(), inlist=[],
                                          mykey='midline')
    midlines_per_plane = []
    for each_path in midline_paths:
        x_midline = int(np.nanmean([x for x, y in np.load(Path(each_path))]))
        midlines_per_plane.append(x_midline)
    df['side'] = None # need to add in the 'side' column
    for i in range(len(df)):
        side = coordutils.determine_sideness_of_cell(df.iloc[i].neur_coords,
                                          midlines_per_plane[int(df.iloc[i].plane.split('_')[1])])
        df['side'].iloc[i] = side

    # prepping the photostim cell info
    specific_photostim_barcode_df = df[(df.photostim == True) & (df.visual_barcode == matching_barcode)]
    specific_stim_cell_lst = specific_photostim_barcode_df.resp_cell_id.values
    specific_stim_cell_sides = {key: val for key, val
                                in zip(specific_photostim_barcode_df.resp_cell_id.values, specific_photostim_barcode_df.side.values)}
    photostim_coords = specific_photostim_barcode_df.neur_coords.values

    fig, ax = plt.subplots(1, len(regions) + 1, figsize=(4 * len(regions) + 3, 2 * (len(regions) + 1)),
                           gridspec_kw={'width_ratios': [2] * len(regions) + [0.8]})
    responsive_regional_proportions = []
    for r, region in enumerate(regions):
        # making the regional df
        regional_df = df[(df.region == region) & (df.photostim == False)].reset_index(drop=True)
        combined_df = pd.concat([regional_df, specific_photostim_barcode_df]).reset_index(drop=True)
        combined_df.drop(
            columns=['omr_neur_id', 'stim_neur_id', 'photostim', 'stim_frames', 'stim_events', 'avg_evoked_df_f',
                     'self_success_trials', 'avg_evoked_zscore'], inplace=True)
        # getting the info, metrics
        downstream_responders_indices_dict, color_values_dict = gather_thresholded_responders_and_vals(combined_df,
                                                                                                        stim_fishy_vol,
                                                                                                        std_thresh = std_thresh,
                                                                                                        evoked_response_window = evoked_response_window,
                                                                                                        response_type = response_type,
                                                                                                        stim_cell_id_lst = specific_stim_cell_lst,
                                                                                                        use_weights = use_weights)

        ax[r].imshow(stim_fishy_vol[1].rescaled_ref, cmap='gray',
                     vmax=np.percentile(stim_fishy_vol[1].rescaled_ref, 99))
        [ax[r].scatter(a[0], a[1], s=20, color='red') for a in photostim_coords]
        ax[r].axis('off')
        ax[r].set_title(region)

        # for each stim cell, need to draw arrows to the responsive cell
        sideness_dict = {'ipsi': [], 'contra': []}
        for ind, stim_cell_name in enumerate(downstream_responders_indices_dict.keys()):
            # per stimmed cell, how many responsive neurons were ipsilateral vs contralateral
            ipsi_count = 0
            contra_count = 0

            responsive_cell_ids = downstream_responders_indices_dict[stim_cell_name]
            connectivity_colors = mappable.to_rgba(color_values_dict[stim_cell_name])
            regional_responsive_coords = combined_df.neur_coords[responsive_cell_ids]
            regional_responsive_barcodes = [r.split('_')[0] for r in combined_df.visual_barcode[responsive_cell_ids].values]
            regional_responsive_sides = combined_df.side[responsive_cell_ids].values

            for c, coord in enumerate(regional_responsive_coords):
                clr = 'white'
                if regional_responsive_barcodes[c] in constants.eva_types_colors.keys():
                    clr = constants.eva_types_colors[regional_responsive_barcodes[c]]
                ax[r].scatter(coord[0], coord[1], s=20, color=clr)
                ax[r].annotate('', xy=(coord[0], coord[1]),
                               xytext=(photostim_coords[ind][0], photostim_coords[ind][1]),
                               arrowprops=dict(facecolor='red', edgecolor=connectivity_colors[c], arrowstyle='->', linewidth=1))
                # if the photostimmed side is the same as the responsive side
                if specific_stim_cell_sides[stim_cell_name] == regional_responsive_sides[c]:
                    ipsi_count += 1
                else:
                    contra_count += 1
            sideness_dict['ipsi'].append(ipsi_count)
            sideness_dict['contra'].append(contra_count)
            np.save(data_folder_path.joinpath(f'{matching_barcode}_to_{region}_responders_sideness.npy'), sideness_dict)

        # total number of responders
        unique_responders = np.unique(
            [item for sublist in list(downstream_responders_indices_dict.values()) for item in sublist])
        try:
            proportion = float("{:.4f}".format(len(unique_responders) / len(combined_df)))
        except:
            proportion = 0
        responsive_regional_proportions.append(proportion)

    # make a bar graph on the side here showing regional differences
    ax[-1].set_box_aspect(ax[0].get_position().height / ax[0].get_position().width)
    bar_plot = ax[-1].bar(regions, responsive_regional_proportions, color='dimgray')
    for bar in bar_plot:  # adding in some text for the height of the bars
        height = bar.get_height()
        x_pos = bar.get_x() + bar.get_width() / 2
        ax[-1].text(x_pos, height + 0.02, f'{height}', ha='center')
    ax[-1].set_xticklabels(regions, rotation=45)
    ax[-1].set_ylabel('proportion of responsive cells/region ')
    ax[-1].set_ylim(0, 0.5)
    ax[-1].spines[['top', 'right']].set_visible(False)
    ax[-1].set_title(f'barcode: {matching_barcode}')

    # save the proportion per fish, per region back into the data folders

    save_dict = {key: val for key, val in zip(regions, responsive_regional_proportions)}
    np.save(data_folder_path.joinpath(f'responsive_cell_proportions_per_region_{matching_barcode}.npy'), save_dict)


    # build the colormap bar for the connectivity lines
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.25, top=0.92, wspace=0.3)
    cbar_ax = fig.add_axes([0.2, 0.12, 0.4, 0.05])  # [left, bottom, width, height] in figure coordinates
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(response_type)

    # build the legend for the types of cells
    barcode_legend = [matplotlib.patches.Patch(facecolor=color, edgecolor='black', label=key)
                      for key, color in constants.eva_types_colors.items()]
    none_barcode = [matplotlib.patches.Patch(facecolor='white', edgecolor='black', label='no barcode')]
    photostim_legend = [matplotlib.lines.Line2D([0], [0], marker='o', color='red', label='photostimulated',
                                                markersize=6, linestyle='None')]
    legend_handles = barcode_legend + none_barcode + photostim_legend
    fig.legend(handles=legend_handles,
               loc='lower center',
               bbox_to_anchor=(0.85, 0.1),  # adjust as needed [x, y]
               ncol=2,
               title='key',
               frameon=False,
               fontsize='small',
               handleheight=1.2)

    if save_location is not None:
        plt.savefig(Path(save_location).joinpath(f'stimmed_{matching_barcode}_regional_threshold_responders.png'),
                    format="png", dpi=300)
    plt.tight_layout()
    plt.show()


def plotting_thresholded_responders_to_stimmed_iMmMm_and_oB(df,
                                                            stimy_fish_vol,
                                                            evoked_response_window=6,
                                                            response_type='zscore',  # 'df/f'
                                                            plotting_trace_type = 'df/f', # since sometimes plotting response types is different than scoring types
                                                            std_threshold=1.8,
                                                            color_map = 'coolwarm',
                                                            vmin=-1,
                                                            vmax=2,
                                                            saving_dir=None
                                                            ):
    '''
    Making the 'mock figure', iMm and oB neurons and thresholded responders across the entire FOV
    :param df: functional data types df
    :param stimy_fish_vol: stim fish volume
    :param evoked_response_window: number of frames for the evoked (analysis) window post ps event
    :param response_type: 'df/f' or 'zscore'
    :param std_threshold: stdev threshold above baseline
    :param vmin: min for colormap
    :param vmax: max for colormap
    :param saving_dir: location to save this figure
    :return:
    '''

    # STEP 1: get resp cell ids for the iMm/Mm and oB neurons that were stimmed
    Mm_stim_cell_ids = df[(df.photostim == True) & (df.visual_barcode.str.contains('Mm'))].resp_cell_id.values
    oB_stim_cell_ids = df[(df.photostim == True) & (df.visual_barcode.str.contains('oB')) & ~(
        df.visual_barcode.str.contains('ioB'))].resp_cell_id.values

    # STEP 2: need to gather the responder indices for the iMm and oB neurons
    # for now its default to use a weighted mean
    Mm_responder_inds, Mm_responder_color_vals = gather_thresholded_responders_and_vals(df,
                                                                                        stimy_fish_vol,
                                                                                        evoked_response_window=evoked_response_window,
                                                                                        response_type=response_type,
                                                                                        color_vals_response_type=plotting_trace_type,
                                                                                        std_thresh=std_threshold,
                                                                                        stim_cell_id_lst=Mm_stim_cell_ids,
                                                                                        use_weights=True)
    oB_responder_inds, oB_responder_color_vals = gather_thresholded_responders_and_vals(df,
                                                                                        stimy_fish_vol,
                                                                                        evoked_response_window=evoked_response_window,
                                                                                        response_type=response_type,
                                                                                        color_vals_response_type=plotting_trace_type,
                                                                                        std_thresh=std_threshold,
                                                                                        stim_cell_id_lst=oB_stim_cell_ids,
                                                                                        use_weights=True)

    # STEP 3: find traces and make plot (defualt coolwarm colors)
    img_hz = stimy_fish_vol[0].img_hz
    norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # center it around 0
    cmap = plt.get_cmap(color_map)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig = plt.figure(figsize=(12, 12))
    gs = matplotlib.gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05], wspace=0.3, hspace=0.2)
    axes = np.empty((2, 2), dtype=object)
    axes[0, 0] = fig.add_subplot(gs[0, 0])  # top-left image
    axes[0, 1] = fig.add_subplot(gs[0, 1])  # top-right image
    axes[1, 0] = fig.add_subplot(gs[1, 0])  # bottom-left heatmap
    axes[1, 1] = fig.add_subplot(gs[1, 1])  # bottom-right heatmap

    cbar_ax = fig.add_subplot(gs[0, 2])  # Dedicated axis for colorbar (only aligns with top-right plot)

    for b, barcode in enumerate(['Mm/iMm', 'oB']):
        if barcode == 'Mm/iMm':
            ind_dict = Mm_responder_inds
            vals_for_colors = Mm_responder_color_vals
        else:
            ind_dict = oB_responder_inds
            vals_for_colors = oB_responder_color_vals

        # 1) set subplots
        ax_img = axes[0, b]
        ax_heat = axes[1, b]

        # 2) plot images
        ax_img.imshow(stimy_fish_vol[0].rescaled_ref, cmap='gray',
                      vmax=np.percentile(stimy_fish_vol[0].rescaled_ref, 99))
        ax_img.axis('off')
        ax_img.set_title(barcode)

        # 3) gather data for heatmap and plot scatter plots
        traces_dict = {s: [] for s in ind_dict.keys()}
        for each_stimmed_cell, ind_lst in ind_dict.items():
            coord_of_stim_cell = df[df.resp_cell_id == each_stimmed_cell].neur_coords.values[0]
            trial_weights = np.array(df[df.resp_cell_id.isin([each_stimmed_cell])].trial_weights.values)[0]
            connectivity_colors = mappable.to_rgba(vals_for_colors[each_stimmed_cell])
            responder_df = df.loc[ind_lst]
            ax_img.scatter(coord_of_stim_cell[0], coord_of_stim_cell[1], color='red', s=20)
            all_downstream_responses = []
            for a, v in enumerate(responder_df.neur_coords.values):
                responder_barcode = responder_df.visual_barcode.iloc[a]
                if responder_barcode == 'None':
                    barcode_clr = 'white'
                else:
                    gen_responder_barcode = responder_barcode.split('_')[0]
                    barcode_clr = constants.eva_types_colors[gen_responder_barcode]
                ax_img.scatter(v[0], v[1], color=barcode_clr, s=20)
                ax_img.annotate('', xy=(v[0], v[1],), xytext=(coord_of_stim_cell[0], coord_of_stim_cell[1]),
                                arrowprops=dict(facecolor='red', edgecolor=connectivity_colors[a], arrowstyle='->',
                                                linewidth=1))
                if plotting_trace_type == 'df/f':  # df/f (local)
                    ps_response_per_trial = responder_df.iloc[a].stim_responses[each_stimmed_cell][0]
                else:  # zscore data
                    ps_response_per_trial = responder_df.iloc[a].stim_responses_zscore[each_stimmed_cell][0]
                weighted_response = np.average(ps_response_per_trial, axis=0, weights=trial_weights)
                all_downstream_responses.append(weighted_response)
            traces_dict[each_stimmed_cell] = all_downstream_responses

        # 4) plot heatmap - each individual stimulated cell has its responders
        traces_all = []
        stim_labels = []
        group_bounds = []
        row_start = 0
        for stim, traces in traces_dict.items():
            if len(traces) == 0:
                continue
            traces_matrix = np.vstack(traces)  # (n_resps, timepoints)
            traces_all.append(traces_matrix)
            row_end = row_start + traces_matrix.shape[0]
            group_bounds.append(row_end)
            stim_labels.append((stim, row_start, row_end))
            row_start = row_end
        data_matrix = np.vstack(traces_all)  # big combined matrix of all the data in order
        sns.heatmap(data_matrix, ax=ax_heat, cmap=color_map, vmin=vmin, vmax=vmax, cbar=False)

        for bound in group_bounds[:-1]:
            ax_heat.axhline(bound, color="white", linestyle="--", linewidth=1)

        # set y-axis ticks at group centers
        yticks = []
        ylabels = []
        for stim, start, end in stim_labels:
            yticks.append((start + end) / 2)
            ylabels.append(f"{stim} (n={end - start})")
        ax_heat.set_yticks(yticks)
        ax_heat.set_yticklabels(ylabels, rotation=0, fontsize=6)

        x_start = -stimy_fish_vol[0].photostim_frame_window[0]
        x_end_evoked = x_start + evoked_response_window
        x_end_ps_duration = x_start + ((stimy_fish_vol[0].ps_event_duration / 1000) * img_hz)  # duration in frames
        ax_heat.axvspan(x_start, x_end_ps_duration, color='red', alpha=0.3)  # add stim span
        ax_heat.set_xticks(np.arange(data_matrix.shape[1]))
        ax_heat.set_xticklabels(['{:.2f}'.format(i / img_hz) for i in np.arange(data_matrix.shape[1])], rotation=30,
                                fontsize=6)
        ax_heat.set_xlabel('time(s)')

        # add in evoked window and photostim duration on top
        bar_height = 0.03
        y_pos_evoked = 1.06
        y_pos_ps_duration = 1.01
        ax_heat.add_patch(matplotlib.patches.Rectangle((x_start, y_pos_evoked),
                                                       x_end_evoked - x_start,  # width in data coords
                                                       bar_height,
                                                       transform=matplotlib.transforms.blended_transform_factory(
                                                           ax_heat.transData, ax_heat.transAxes),
                                                       color='green', alpha=0.5, clip_on=False))
        ax_heat.add_patch(matplotlib.patches.Rectangle((x_start, y_pos_ps_duration),
                                                       x_end_ps_duration - x_start,  # width in data coords
                                                       bar_height,
                                                       transform=matplotlib.transforms.blended_transform_factory(
                                                           ax_heat.transData, ax_heat.transAxes),
                                                       color='red', alpha=0.5, clip_on=False))

    # add legend about the heat map lines & color code of the barcoded neurons
    heatmap_legend_elements = [
        matplotlib.lines.Line2D([0], [0], color='red', linestyle='--', lw=1.5, label='photostimulation event'),
        matplotlib.patches.Patch(facecolor='red', alpha=0.5, label='photostimulation duration'),
        matplotlib.patches.Patch(facecolor='green', alpha=0.5, label='evoked window')]
    barcode_legend = [matplotlib.patches.Patch(facecolor=color, edgecolor='black', label=key)
                      for key, color in constants.eva_types_colors.items()]
    none_barcode = [matplotlib.patches.Patch(facecolor='white', edgecolor='black', label='no barcode')]
    photostim_cell_legend = [matplotlib.lines.Line2D([0], [0], marker='o', color='red', label='photostimulated cell',
                                                     markersize=6, linestyle='None')]
    legend_handles = photostim_cell_legend + barcode_legend + none_barcode + heatmap_legend_elements
    ax_heat.legend(handles=legend_handles, loc='upper left',
                   bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=8)
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label(plotting_trace_type)

    if saving_dir is not None:
        plt.savefig(Path(saving_dir).joinpath(f'threshold_responders_to_iMmMm_oB_stim.png'),
                    format="png", dpi=300)
    plt.show()

# --- FULL PROCESSING TO RUN THROUGH MULTIPLE FXNS -- #
def full_process_creating_df(omr_fishy_vol,
                             stim_fishy_vol,
                             vol_barcode_df,
                             match_cells_within_radius_um = 10,
                             self_success_keywords={'trace_type': 'zscore',
                                                    'std_threshold': 1.5,
                                                    'save_plots_path': None,
                                                    'plotting': True},
                             weighted_trial_keywords={'self_success': True}):
    '''
    Full preprocessing step to make the functional types dataframe (a combo of all the above functions)
    window ranges are default from the fish volume info

    :param omr_fishy_vol: omr dataset fishvolume
    :param stim_fishy_vol: stim dataset fishvolume
    :param vol_barcode_df: barcoding information
    :param self_success_keywords: args for determine_self_success_trials()
    :param weighted_trial_keywords: args for determine_weighted_trials()
    :return: complete functional types dataframe and saved
    '''

    # before starting, check that the nMLF and Hb rois are made in the omr fishvolume
    print('check that all rois are made')
    omr_fishy_vol[0].load_saved_rois()
    with os.scandir(omr_fishy_vol[0].folder_path.joinpath('rois')) as entries:
        saved_regions = [entry.name.split('.')[0] for entry in entries]
    result = all(elem in saved_regions for elem in ['Hb', 'nMLF', 'Pt'])
    if result == False: # ends function before starting
        return print('need to make regions before creating dataframe')

    # 1 - build the functional dataframe
    print('building the df')
    functional_types_df = build_functional_types_df(omr_fishvolume=omr_fishy_vol,
                                                    stim_fishvolume=stim_fishy_vol,
                                                    omr_fishvolume_barcoding_df=vol_barcode_df,
                                                    motor_correlation=False,  # no tail movement
                                                    regions=['Pt', 'nMLF', 'Hb'],
                                                    match_cells_within_radius_um=match_cells_within_radius_um)
    functional_types_df.dropna(subset=['stim_neur_id'], inplace=True)  # drop nan's
    functional_types_df.reset_index(drop=True, inplace=True)

    # 2 - add photostimulation responses to the functional dataframe, both 'df/f' and 'zscore'
    print('adding responses')
    functional_types_df = add_photostimulation_responses_to_functional_df(functional_types_df,
                                                                          stim_fishy_vol,
                                                                          response_window=
                                                                          stim_fishy_vol[
                                                                              0].photostim_frame_window)
    functional_types_df = add_photostimulation_responses_to_functional_df(functional_types_df,
                                                                          stim_fishy_vol,
                                                                          response_window=
                                                                          stim_fishy_vol[
                                                                              0].photostim_frame_window,
                                                                          trace_type='zscore')

    # 3 - gather self-success trials & add to dataframe
    print('adding self-success trials')
    self_success_trials_dict = determine_self_success_trials(functional_types_df[functional_types_df.photostim == True],
                                                                response_type=self_success_keywords['trace_type'],
                                                                baseline_frames=int(np.ceil(stim_fishy_vol[0].img_hz)),
                                                                photostim_window=stim_fishy_vol[0].photostim_frame_window,
                                                                std_threshold=self_success_keywords['std_threshold'],
                                                                immediate_resp_window=stim_fishy_vol[0].evoked_num_frames,
                                                                save_plots_path=self_success_keywords['save_plots_path'],
                                                                plotting=self_success_keywords['plotting'],
                                                                saving_dict=stim_fishy_vol[0].folder_path.parents[1])
    functional_types_df = add_self_success_trials_to_df(functional_types_df, self_success_trials_dict)

    # 4 - add weights for self-success trials
    print('adding weights')
    weights_dict = determine_trial_weights(functional_types_df[functional_types_df.photostim == True],
                                           trace_type = self_success_keywords['trace_type'],
                                            self_success= weighted_trial_keywords['self_success'],
                                            photostim_response_window=stim_fishy_vol[0].photostim_frame_window,
                                            evoked_response_window=stim_fishy_vol[0].evoked_num_frames,
                                            baseline_frames=int(np.ceil(stim_fishy_vol[0].img_hz)),  # 1 second of baseline
                                            saving_dict_path=stim_fishy_vol[0].folder_path.parents[1])
    functional_types_df = add_trial_weights_to_df(functional_types_df, weights_dict)

    # 5 - save the functional dataframe
    functional_types_df.to_hdf(stim_fishy_vol[0].folder_path.parents[1].joinpath('functional_types_df.h5'),
                               key='functional')

    return functional_types_df

def full_process_to_check_quality(omr_fishy_vol,
                                  stim_fishy_vol,
                                  df,
                                  fish_id,
                                  trace_type = 'zscore',
                                  evoked_response_window = 8,
                                  std_threshold_for_responders = 2,
                                  color_map = 'coolwarm',
                                  color_bar_limits = [-1, 3],
                                  use_weights = False):
    '''
    Full process to spit out analysis images for each stimulated cell into Box folder 'analysis_outputs'
    These plots all have to do with data quality (location, off target effects, basic barcoding/ps responses, etc)

    :param omr_fishy_vol: fish volume of the omr dataset
    :param stim_fishy_vol: fish volume of the stim dataset
    :param df: functional types dataframe
    :param fish_id: the name of the fish for this data (will be used to make a folder in the master Box folder)
    :param evoked_response_window: the number of frames immediately after the ps event to use to determine the most responsive neurons
    :param std_threshold_for_responders: the number of standard deviations that the cell needs to be greater than for responders
    :param color_map: color map to use for plotting connectivity lines
    :param color_bar_limits: colorbar limits to use for plotting connectivity lines
    :param use_weights: whether to use weights when finding thresholded responders
    :return: all plots into a specific fish folder in the Box folder
    '''

    master_saving_directory = Path(r'C:\Users\Kaitlyn\Box\PROJECT 22□ OMR Photostim Project\analysis_outputs')

    vol_stim_sites_info_df = pd.read_hdf(stim_fishy_vol[0].folder_path.parents[1].joinpath('master_stim_sites.h5'), key="stim", allow_pickle = True)
    fish_id_folder = Path(master_saving_directory).joinpath(fish_id)
    os.makedirs(fish_id_folder, exist_ok=True)
    print(fish_id_folder)
    vmin, vmax = color_bar_limits

    # 1 - vizmotion and ps responses, multi plots, for both df/f and zscore
    just_stim_cells_df = df[(df.photostim == True)]
    for _type in ['df/f', 'zscore']:
        print(f'plotting vismotion and ps resps for {_type}')
        if _type == 'df/f':
            ylims = [-0.5, 1.5]
        if _type == 'zscore':
            ylims = [-1.5, 1.5]
        plotting_vizmotion_and_photostimulation_responses(just_stim_cells_df,
                                                            lst_motion_cues = constants.photostim_motion_cues,
                                                            lst_motion_cues_colors = constants.photostim_motion_cues_colors,
                                                            img_hz = omr_fishy_vol[0].img_hz,
                                                            response_type = _type,
                                                            vizmotion_stim_offset = omr_fishy_vol[0].stim_offset,
                                                            photostim_window_frames = stim_fishy_vol[0].photostim_frame_window,
                                                            photostim_ylim = ylims,
                                                            photostim_stim_offset = 0,
                                                            save_location = fish_id_folder,
                                                            filtered_stim_trials=None)
    # 2 - check location of cell at each ps event
    print(f'plotting locations of stimmed cell and stim site per trial')
    plotting_location_of_stim_site_per_trial(just_stim_cells_df,
                                             vol_stim_sites_info_df,
                                             stim_fishy_vol,
                                             save_location = fish_id_folder)

    # 3 - plot nearby in xy
    print(f'plotting nearby xy cells')
    for i, s in enumerate(just_stim_cells_df.resp_cell_id.values):
        try:
            plane_no = just_stim_cells_df[just_stim_cells_df.resp_cell_id == s].plane.values[0]
            _stim_cell_id =  [k for k, v in stim_fishy_vol.volumes[plane_no].stimmed_cells_matched_stim_ids_dict.items()
                              if v[0] == int(df[df.resp_cell_id == s].stim_neur_id.values[0])][0] # what was programmed into the Bruker (red cross)
            full_volume_stim_sites_df_specific_stim_site = vol_stim_sites_info_df[vol_stim_sites_info_df.cell_ids == _stim_cell_id] # need to index into the correct row
            bruker_coord = [full_volume_stim_sites_df_specific_stim_site.x_stim.values[0], full_volume_stim_sites_df_specific_stim_site.y_stim.values[0]]

            plotting_nearby_cells_xy(s,
                                     df[df.plane == plane_no],
                                     stim_fishy_vol.volumes[plane_no],
                                     bruker_coord,
                                     # this is the same window that I used to find matching cell ids in the photostim fishy, and will make a difference
                                     photostim_window = [-8,6],
                                     radius_um = 10,
                                     save_path = fish_id_folder)
        except:
            print(f'cannot plot nearby xy cells for {s}')
            pass

    # 4 - plot nearby in z
    print(f'plotting z off target')
    plotting_z_off_target_single_cells(df, stim_fishy_vol[0], save_location = fish_id_folder)
    try: # not sure if the off target group thing will work for good
        plotting_z_off_target_group(df,
                                stim_fishy_vol[0],
                                trace_type='df/f',
                                radius_um=10,
                                max_cells_per_plane=6,
                                save_location=fish_id_folder)
    except:
        pass

    # 5 - plot the most responsive cells across the whole volume
    print(f'plotting most responsive neurons across the entire volume (thresholded) with traces')
    plotting_thresholded_responders_locations_and_traces(df,
                                                        stim_fishy_vol,
                                                        std_thresh= std_threshold_for_responders,
                                                        evoked_response_window=evoked_response_window,
                                                        response_type=trace_type,
                                                        color_map = color_map,
                                                        use_weights= use_weights,
                                                        vmin = vmin,
                                                        vmax = vmax,
                                                        save_location = fish_id_folder)

    # 6 - plot the most responsive cells across the whole volume, per region and barcoded neurons
    print(f'plotting most responsive neurons across the whole volume, per region and barcoded photostimmed neurons')
    all_barcodes_of_stimmed_neurons = df[df.photostim == True].visual_barcode.unique()
    for barcode in all_barcodes_of_stimmed_neurons:
        plotting_thresholded_responders_per_region_and_photostim_barcode(df,
                                                                         stim_fishy_vol,
                                                                         matching_barcode=barcode,
                                                                         std_thresh=std_threshold_for_responders,
                                                                         evoked_response_window=evoked_response_window,
                                                                         response_type=trace_type,
                                                                         regions=['Pt', 'nMLF', 'Hb'],
                                                                         color_map=color_map,
                                                                         use_weights=use_weights,
                                                                         vmin=vmin,
                                                                         vmax=vmax,
                                                                         save_location=fish_id_folder)
    try:
        print(f'plotting responders to iMm/Mm and oB stimulations')
        plotting_thresholded_responders_to_stimmed_iMmMm_and_oB(df,
                                                                stim_fishy_vol,
                                                                evoked_response_window=evoked_response_window,
                                                                response_type=trace_type,
                                                                std_threshold=std_threshold_for_responders,
                                                                vmin=vmin,
                                                                vmax=vmax,
                                                                saving_dir=fish_id_folder)
    except:
        print('no iMm/Mm or oB responders')

    return print('completed full process data quality check')
