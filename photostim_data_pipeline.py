'''
Multiple functions to help process photostimulation and OMR datasets together and plotting

'''

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Circle, Patch
import seaborn as sns
from bcdict import BCDict

# local imports
from utilities import arrutils, plotutils, statutils, coordutils, roiutils
from bruker_images import get_micronstopixels_scale
from utilities.roiutils import create_circular_mask, draw_roi, create_polygon_mask
from fishy import PhotostimFish
import constants

# putting photostim and omr data together into a master functional types dataframe #
     
def build_functional_types_df(omr_fishvolume, stim_fishvolume, omr_fishvolume_barcoding_df, regions = ['Pt', 'Hb', 'nMLF'], 
                                  motor_correlation = False, plot_stim_sites = True):
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
        omr_tailFish = omr_fishvolume.volumes[plane]
        omr_tailFish.load_saved_rois()
        omr_data_rois = omr_tailFish.return_cell_rois(range(len(omr_tailFish.f_cells)))

        sub_functional_types_df['omr_neur_id'] = range(len(omr_tailFish.f_cells))
        print('loading omr data')
        all_motion_responses_lst = gather_visual_motion_responses_for_df(omr_tailFish, cell_id_array = None, motion_cues = constants.photostim_motion_cues)
        sub_functional_types_df['motion_responses'] = all_motion_responses_lst
        sub_functional_types_df['neur_coords'] = omr_data_rois
        sub_functional_types_df['plane'] = [plane] * len(sub_functional_types_df)
        if motor_correlation:
            sub_functional_types_df['motor_corr'] = omr_tailFish.motor_pearson_corrs

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
                if k in omr_tailFish.roi_dict.keys():
                    if l in omr_tailFish.return_cells_by_saved_roi(k):
                        sub_functional_types_df['region'].iloc[l] = k

        stim_photostimFish = stim_fishvolume.volumes[plane]
        matched_cell_ids = coordutils.match_cell_ids(sub_functional_types_df.omr_neur_id.values, omr_tailFish.stats,
                                                        range(len(stim_photostimFish.f_cells)), stim_photostimFish.stats,
                                                        um_to_px = get_micronstopixels_scale(stim_photostimFish.data_paths['info_xml']))
        sub_functional_types_df['stim_neur_id'] = matched_cell_ids.values()
            
        if 'stim_sites' in stim_photostimFish.data_paths.keys():
            print('matching stimuluation sites with OMR cells')
            # need to identify which neurons were photostimulated by matching spatially
            stim_photostimFish.stimmed_cell_coords, _, stimmed_closest_cell_id_dict = PhotostimFish.identify_stim_cells(stim_photostimFish, overlap = True)
            
            # sort the stimmed cell ids by the cell id order of the stim sites df
            # important for the next step of matching the stim events and frames
            key_order = stim_photostimFish.stim_sites_df.cell_ids.values
            print(f'df order: {key_order}')
            sorted_stimmed_closest_cell_id_dict = {key: stimmed_closest_cell_id_dict[key] for key in key_order if key in stimmed_closest_cell_id_dict}
            print(f'sorted dict: {sorted_stimmed_closest_cell_id_dict}')
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
                            # red is OMR-id'ed cell, white is photostim cell
                            plt.scatter(sub_functional_types_df.neur_coords.loc[omr_photostim_cell_id_lst[n]][0], 
                                        sub_functional_types_df.neur_coords.loc[omr_photostim_cell_id_lst[n]][1], color = 'red', s = 10)
                            plt.annotate(omr_photostim_cell_id_lst[n], (sub_functional_types_df.neur_coords.loc[omr_photostim_cell_id_lst[n]][0], 
                                            sub_functional_types_df.neur_coords.loc[omr_photostim_cell_id_lst[n]][1]), color = 'red')
                            plt.scatter(stim_photostimFish.stimmed_cell_coords[n][0], stim_photostimFish.stimmed_cell_coords[n][1], color = 'white', s = 10)
                            plt.annotate(stimmed_cell_id_array[n], (stim_photostimFish.stimmed_cell_coords[n][0], 
                                                                                        stim_photostimFish.stimmed_cell_coords[n][1]), color = 'white') 
                    plt.show() 
                except:
                    print('cannot plot for some reason')
        else:
            pass  
        df_lst.append(sub_functional_types_df)   

    functional_types_df = pd.concat(df_lst).reset_index(drop = True)
    final_functional_types_df = add_resp_cell_ids_to_df(functional_types_df)
    
    return final_functional_types_df    

def add_resp_cell_ids_to_df(functional_types_df):
    '''
    Add the responding cell id to the dataframe
    functional_types_df = dataframe, the master functional types dataframe

    returns the dataframe with the responding cell id
    '''
    responders = functional_types_df[functional_types_df.photostim == False]
    stimulators = functional_types_df[functional_types_df.photostim == True]

    resp_cell_id_lst = []
    for n in range(len(responders)):
        resp_cell_id_lst.append(f'resp_{n}')
    responders['resp_cell_id'] = resp_cell_id_lst

    stim_resp_cell_id_lst = []
    for t in range(len(stimulators)):
        stim_resp_cell_id_lst.append(f'stim_{t}')
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

def gather_photostimulation_responses_for_df(responder_f_traces, stimulated_f_traces, stimulated_frames_array, 
                                             photostim_response_frame_windows = [-4, 7], type = 'df_f'):
    '''
    Gather the df/f responses of the responding cells to each photostimulated cell
    responder_f_traces = list, the traces of the responding cells (n responder cells x len of imaging), make sure not to include the stimulated cells
    stimulated_f_traces = list, the traces of the stimulated cells (n photostimulated cells x len of imaging)
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
        for stim_cell_ind, stimulated_trace in enumerate(stimulated_f_traces):
            s_cell_id = f'stim_{stim_cell_ind}'
            if s_cell_id not in resp_to_each_stimulation_dict.keys():
                resp_to_each_stimulation_dict[s_cell_id] = {}
            stimmed_frames = stimulated_frames_array[stim_cell_ind]
            frame_subset = arrutils.subsection_arrays(stimmed_frames, photostim_response_frame_windows)

            if frame_subset[-1][-1] > len(stimulated_trace): # adjust frames in case this is out of range
                new_frame_subset = []
                for s in frame_subset:
                    new_frame_subset.append([q for q in s if q < len(stimulated_trace)])
                frame_subset = np.array(new_frame_subset)

            resp_raw_trial = np.array([responding_trace[g] for g in frame_subset if len(responding_trace[g] ) > 0])
            resp_df_f_trial = np.zeros(shape = (len(resp_raw_trial), len(resp_raw_trial[0])))
            resp_raw_trial2 = np.zeros(shape = (len(resp_raw_trial), len(resp_raw_trial[0])))

            stim_evoked_df_f_trial = np.zeros(shape = (len(resp_raw_trial), 1))
            stim_evoked_raw_trial = np.zeros(shape = (len(resp_raw_trial), 1))

            for d, f in enumerate(resp_raw_trial):
                base_e = f[:-photostim_response_frame_windows[0]] # i.e. frames 0:4
                plot_e = (f - np.nanmean(base_e)) / np.nanmean(base_e)
                evoked_e = np.nanmedian(plot_e[-photostim_response_frame_windows[0]:]) # evoked df f for each trial, i.e. frames 4:end
                stim_evoked_df_f_trial[d] = evoked_e
                resp_df_f_trial[d] = plot_e # full trace for each trial

                # calculate raw traces of your input array
                evoked_raw = np.nanmedian(f[-photostim_response_frame_windows[0]:]) # evoked raw trace for each trial, i.e. frames 4:end
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

def add_photostimulation_responses_to_functional_df(functional_info_df, stim_fishvolume, response_window = [-4, 7], 
                                                    stimulated_functional_types_df = None, type = 'df_f'):

    # default is to have the entired photostim == True dataset for the 'stimulated info'
    # keeping this flexible in case I need to change what I want to use as my 'stimulated cells' (i.e. changing trials or actual cells)
    if stimulated_functional_types_df is None: 
        stimulated_functional_types_df = functional_info_df[functional_info_df.photostim == True]
    
    # gathering responses from the photostimulation dataset
    resp_f_trace_array, _, resp_zscore_trace_array = prepare_data_for_plotting(functional_info_df, stim_fishvolume)
    stim_f_trace_array, _, stim_zscore_trace_array = prepare_data_for_plotting(stimulated_functional_types_df, stim_fishvolume)

    if type == 'df_f':
        photostimulation_responses_lst, avg_evoked_df_f_lst = gather_photostimulation_responses_for_df(resp_f_trace_array, stim_f_trace_array, 
                                                                                                   stimulated_functional_types_df.stim_frames.values,
                                                                                                   photostim_response_frame_windows = response_window)
        functional_info_df['stim_responses'] = photostimulation_responses_lst
        functional_info_df['avg_evoked_df_f'] = avg_evoked_df_f_lst

    elif type == 'zscore':
        photostimulation_responses_lst_zscore, avg_evoked_zscore_lst = gather_photostimulation_responses_for_df(resp_zscore_trace_array, stim_zscore_trace_array, 
                                                                                                   stimulated_functional_types_df.stim_frames.values,
                                                                                                   photostim_response_frame_windows = response_window,
                                                                                                   type = 'zscore')

        functional_info_df['stim_responses_zscore'] = photostimulation_responses_lst_zscore
        functional_info_df['avg_evoked_zscore'] = avg_evoked_zscore_lst
    
    elif (type == 'global_df_f') & ('baseline_f' in functional_info_df.columns):
        global_df_f_traces = np.zeros(shape = resp_f_trace_array.shape)
        for n, raw_trace in enumerate(resp_f_trace_array):
            baseline_f = functional_info_df['baseline_f'].iloc[n]['mean']
            global_df_f_trace = (raw_trace - baseline_f) / baseline_f
            global_df_f_traces[n] = global_df_f_trace

        photostimulation_responses_lst_global_df_f, avg_evoked_global_df_f_lst = gather_photostimulation_responses_for_df(global_df_f_traces, stim_f_trace_array, 
                                                                                                   stimulated_functional_types_df.stim_frames.values,
                                                                                                   photostim_response_frame_windows = response_window,
                                                                                                   type = 'global_df_f')
        functional_info_df['stim_responses_global_df_f'] = photostimulation_responses_lst_global_df_f
        functional_info_df['avg_evoked_global_df_f'] = avg_evoked_global_df_f_lst

    return functional_info_df

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



# exclusion and analysis functions for the photostimulation data #

def check_rsChrmine_expression(red_channel_img, select_rois, save_path, offset = 1.0):
    '''
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

def gather_significant_photostim_trials(only_stimulated_cells_df, std_threshold = 1.2, immediate_resp_window = 4, 
                                        photostim_window = [-4, 7], save_plots_path = None, plotting = True, saving_dict = None):
    '''
    Gather the significantly photostimulated trials for each photostimulated cell (can save plots and dictionary)

    only_stimulated_cells_df = dataframe, the functional data dataframe of only the photostimulated cells (needs to at least have columns resp_cell_id and stim_responses)
    std_threshold = float, the threshold for the standard deviation of the baseline activity to determine if a trial is significant, default is 1.2
    immediate_resp_window = int, the number of frames after the photostimulation event to consider as the immediate response, default is 4
    photostim_window = list, the window of frames around the photostimulation event to consider, default is [-4, 7]

    return
    significant_trials_dict = BCDict, a dictionary of the significantly photostimulated trials for each photostimulated cell
    '''
    # create a legend for the plot
    legend_elements = [matplotlib.lines.Line2D([0], [0], color='red', linewidth=3, label='photostimulation event'),  # Red line
                   matplotlib.lines.Line2D([0], [0], color='black', linewidth=3, label='significant trial'),# black line = significant trial
                   matplotlib.lines.Line2D([0], [0], color='lightgrey', linewidth=3, label='not significant trial')] # gray line = not significant trial

    significant_trials_dict = BCDict()
    for idx, stim_cell in enumerate(only_stimulated_cells_df.resp_cell_id.values):
        if stim_cell not in significant_trials_dict.keys():
            significant_trials_dict[stim_cell] = {}
        one_stim_cell_info = only_stimulated_cells_df[only_stimulated_cells_df.resp_cell_id == stim_cell]
        try:
            string_cell_id = 'stim' + str(idx)
            all_responses_to_itself = (one_stim_cell_info.stim_responses.values[0][string_cell_id][0])
        except:
            string_cell_id = 'stim_' + str(idx)
            all_responses_to_itself = (one_stim_cell_info.stim_responses.values[0][string_cell_id][0])

        line_colors = [] # gather line colors for plotting
        significant_trials_count = [] # just to count the number of significant trials easily, not really necessary
        for a, b in enumerate(all_responses_to_itself):
            if a not in significant_trials_dict[stim_cell].keys():
                significant_trials_dict[stim_cell][a] = np.nan
            mean_baseline_activity = np.nanmean(b[:-photostim_window[0]])
            std_baseline_activity = np.nanstd(b[:-photostim_window[0]])
            mean_evoked_activity = np.nanmean(b[-photostim_window[0]:-photostim_window[0] + immediate_resp_window])

            if mean_evoked_activity > mean_baseline_activity + (std_threshold * std_baseline_activity):
                significant_trials_dict[stim_cell][a] = mean_evoked_activity
                significant_trials_count.append(a)
                line_colors.append('black')
            else:
                significant_trials_dict[stim_cell][a] = np.nan
                significant_trials_count.append(np.nan)
                line_colors.append('lightgrey')
        
        if plotting:
            fig, ax = plt.subplots(1, all_responses_to_itself.shape[0], figsize = (18, 2))
            
            ymin = np.nanmin(all_responses_to_itself)
            ymax = np.nanmax(all_responses_to_itself)
            for a, b in enumerate(all_responses_to_itself):
                ax[a].plot(b, color = line_colors[a], alpha = 1)
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
                plt.savefig(Path(save_plots_path).joinpath(f'{stim_cell}_trace_per_trial_with_sign.png'), dpi = 300, bbox_inches = 'tight')
            plt.show()

        if saving_dict is not None:
            saving_path = Path(saving_dict).joinpath('significant_trials.npy')
            np.save(saving_path, significant_trials_dict)
    
    return significant_trials_dict

def add_significant_trials_to_df(functional_info_df, significant_trials_dict):
    '''
    Add significant trials to the dataframe, else nan
    '''
    functional_info_df['significant_trials'] = [np.nan] * len(functional_info_df)
    for d in significant_trials_dict.keys():
        significant_trials_list = []
        for k, v in significant_trials_dict[d].items():
            if v is not np.nan:
                significant_trials_list.append(k)
        if len(significant_trials_list) > 0:
            index = functional_info_df[functional_info_df['resp_cell_id'] == d].index.tolist()
            functional_info_df.loc[index, 'significant_trials'] = pd.Series([significant_trials_list], index = index)

    return functional_info_df
        
def gather_evoked_activity_for_select_trials(photostim_responses_per_trial, good_trial_numbers = None, 
                                              immediate_response_window = 4, photostim_response_window = [-4, 7]):
    if good_trial_numbers == None:
        good_trial_numbers = list(range(len(photostim_responses_per_trial)))
        # print('using all trials')
    avg_evoked_activity_lst = []
    for trial_num, ps_resp in enumerate(photostim_responses_per_trial):
        if trial_num in good_trial_numbers:
            evoked_activity = ps_resp[-photostim_response_window[0]:-photostim_response_window[0] + immediate_response_window]
            avg_evoked_activity_lst.append(np.nanmean(evoked_activity))
        else:
            avg_evoked_activity_lst.append(np.nan)
    return avg_evoked_activity_lst

def plotting_location_of_stim_site_per_trial(stimulated_cell_df, full_volume_stim_sites_df, photostim_fish_volume, save_path = None):
    '''
    Plot the location of the stimulation site (from Bruker) and the closest ROI to the stimulation site for each trial
    stimulated_cell_df = dataframe, the dataframe of the stimulated cells
    full_volume_stim_sites_df = dataframe, the full dataframe of the photostimulation stim sites (aka master stim sites df)
    photostim_fish_volume =  the photostimulation data in fish volume format
    save_path = str, the path to save the images to, default is None

    '''

    legend_elements = [matplotlib.lines.Line2D([0], [0], color='red', linewidth=2, label='stimulation point'),  # Red line, stimulation site from bruker
                    matplotlib.patches.Patch(facecolor='green', edgecolor='black', label='targeted cell')]  # Green box, closest ROI to the stimulation site

    for stim_cell_num in range(len(stimulated_cell_df)):
        try:
            one_stim_row = stimulated_cell_df.iloc[stim_cell_num]
            specific_plane_fish = photostim_fish_volume[int(one_stim_row['plane'].split('_')[1])]
            um_per_pxs = get_micronstopixels_scale(specific_plane_fish.data_paths['info_xml'])
            sp_size_pxs = np.nanmean(specific_plane_fish.stim_sites_df.sp_size.values) / um_per_pxs # average pixels diameter
            stim_events_frames = one_stim_row.stim_frames

            # what was programmed into the Bruker (red cross)
            _stim_cell_id =  [k for k, v in specific_plane_fish.stimmed_cells_matched_stim_ids_dict.items() if v == one_stim_row.stim_neur_id][0]
            # need to index into the correct row
            full_volume_stim_sites_df_specific_stim_site = full_volume_stim_sites_df[full_volume_stim_sites_df.cell_ids == _stim_cell_id]
            original_stim_roi = [full_volume_stim_sites_df_specific_stim_site.x_stim.values[0], full_volume_stim_sites_df_specific_stim_site.y_stim.values[0]]

            # closest ROI to the programmed stim site (green circle, full ROI is filled in)
            stim_full_roi = specific_plane_fish.stats[int(one_stim_row.stim_neur_id)]

            # collect the full image over the course of the experiment
            full_img = specific_plane_fish.load_image()
            baseline_img = np.nanmean(full_img[:20, :, :], axis = 0) # baseline image (collecting the first 20 frames now, hard coded)

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
            if save_path is not None:
                plt.savefig(Path(save_path).joinpath(f'{one_stim_row.resp_cell_id}_locations_per_trial.png'), dpi = 300)   
            plt.show()         
        except:
            print(f'could not plot {one_stim_row.resp_cell_id}') # sometimes have a 'nan' matching cell id

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

# def compute_off_target_activation_in_xy()
# FOR ONLY ONE STIM SITE

# def compute_off_target_activation_in_z()

# def process_self_success_info()
# make the plots for all the information in the self success info
# save a dataframe with the info as well
# make a folder for the stimulated cell in a specific location that contains all these plots and dataframe?


# plotting functions for visualizing the photostimulation data (developed Winter 2025) #

def prepare_data_for_plotting(data_df, fishvolume, dataset_type = 'stim'):
    '''
    Prepare the data for plotting
    '''
    f_trace_array = np.zeros(shape = (len(data_df), len(fishvolume[0].f_cells[0])))
    normf_trace_array = np.zeros(shape = (len(data_df), len(fishvolume[0].f_cells[0])))
    zscored_trace_array = np.zeros(shape = (len(data_df), len(fishvolume[0].zdiff_cells[0])))
    if dataset_type == 'stim':
        cell_ids = data_df.stim_neur_id.values
    if dataset_type == 'omr':
        cell_ids = data_df.omr_neur_id.values
    for r_ind, r_cell in enumerate(cell_ids): # all cell ids in the whole dataframe
        r_cell = int(r_cell)
        dataFish = fishvolume.volumes[data_df.plane.values[r_ind]]
        f_trace_array[r_ind] = dataFish.f_cells[r_cell]
        normf_trace_array[r_ind] = dataFish.normcells[r_cell]
        zscored_trace_array[r_ind] = dataFish.zdiff_cells[r_cell]
    
    return f_trace_array, normf_trace_array, zscored_trace_array

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
        [ax[0,n].plot(np.arange(len(m)), arrutils.pretty(m), color = 'grey', alpha = 0.3) for m in df_f_traces]
        ax[0,n].plot(np.arange(len(df_f_traces[0])), arrutils.pretty(np.nanmean(df_f_traces, axis = 0)), color = 'k')
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
        plt.plot(np.arange(len(r)), arrutils.pretty(r + n), linewidth = 1, color = 'k')

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

def plotting_vizmotion_and_photostimulation_responses(vizmotion_photostim_info_df, lst_motion_cues = None, lst_motion_cues_colors = None,
                                                      type = None, img_hz = 1,
                                                      vizmotion_stim_offset = 9, photostim_window_frames = [-4, 7], number_stim_response_panels = None,
                                                      photostim_ylim = [-0.5, 1.2], photostim_stim_offset = 0, 
                                                      filtered_stim_trials = None, save = False, save_location = None):
    '''
    Plot responses to all the visual motion cues and photostimulated neurons for neurons in the dataframe
    Make the vizmotion_photostim_info_df from the above function 'build_vizmotion_photostim_info_df' 
    Currently makes a figure with 6 motion responsive panels and 'n' photostimulation response panels (depends on how many neurons were stimulated)

    vizmotion_photostim_info_df - the dataframe including the motion, photostim responses, average evoked activity
    lst_motion_cues - list, the visual motion stimuli to plot
    lst_motion_cues_colors - list, the colors associated with each visual motion stimulus
    img_hz - the hz of the imaging speed of the OMR fishvolume
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

    frames_motion_on = 10 * img_hz
    if number_stim_response_panels is None:
        panel_num = 6 + len(vizmotion_photostim_info_df.avg_evoked_df_f.values[0])
    else:
        panel_num = 6 + number_stim_response_panels

    for i in range(len(vizmotion_photostim_info_df)):
        vizmotion_responses = vizmotion_photostim_info_df.motion_responses.iloc[i]
        if type is None:
            photostim_responses = vizmotion_photostim_info_df.stim_responses.iloc[i]
        elif type == 'global_df_f':
            photostim_responses = vizmotion_photostim_info_df.stim_responses_global_df_f.iloc[i]
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
        for stim_cell_id, response_to_stim in photostim_responses.items():
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

                if not isinstance(response_to_stim, (float, int)): # make sure that this is not nan's
                    [ax[plot_ind].plot(np.arange(len(m)), arrutils.pretty(m), color = 'grey', alpha = 0.3) for m in response_to_stim]
                    ax[plot_ind].plot(np.arange(len(response_to_stim[0])), arrutils.pretty(np.nanmedian(response_to_stim, axis = 0)), color = 'k')
                    ci_lower, ci_upper = statutils.calculate_ci(np.array(response_to_stim))
                    ax[plot_ind].fill_between(np.arange(len(response_to_stim[0])), ci_lower, ci_upper, color='skyblue', alpha=0.4, label='95% CI') 
                    ax[plot_ind].axvline(x = -photostim_window_frames[0] + photostim_stim_offset, color = 'red')
                    ax[plot_ind].axhline(0, color = 'grey', linestyle = '--')
                    ax[plot_ind].set_title(stim_cell_id, rotation = 30, fontsize = 10)
                    ax[plot_ind].set_ylim(photostim_ylim[0], photostim_ylim[1])
                    ax[plot_ind].set_xlim(0, len(response_to_stim[0])-1)
                    plot_ind += 1

        ax[0].set_ylabel('dF/F')
        [a.set_xlabel('time (s)') for a in ax.flatten()]
        [a.set_xticks(ticks = [int(i) for i in np.linspace(0, len(vizmotion_responses['mean'][vizstim_cue])-1, 2)], 
                        labels = [int(i) for i in np.linspace(0, len(vizmotion_responses['mean'][vizstim_cue])-1, 2) * img_hz]) for a in ax[:6].flatten()]
        [a.set_xticks(ticks = [int(i) for i in np.linspace(0, np.diff(photostim_window_frames)-1, 2)], 
                        labels = [int(i) for i in np.linspace(0, np.diff(photostim_window_frames)-1, 2) * img_hz]) for a in ax[6:].flatten()]
        
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
    
        if save:
            save_name = resp_cell_id_name +'_vizmotion_photostim_responses.png'
            plt.savefig(save_location.joinpath(save_name), dpi = 300, bbox_inches = 'tight')
            save_name = resp_cell_id_name +'_vizmotion_photostim_responses.svg'
            plt.savefig(save_location.joinpath(save_name), dpi = 300, bbox_inches = 'tight')
        
        plt.show()
    
    return print('done')

def plotting_correlation_maps_avg_photostim_evoked_resp(vizmotion_photostim_info_df, omr_fish_volume, color_map = None, limits = [-0.3, 0.3], 
                                                        stimulation_cell_color = 'limegreen', title = None, save = False, save_location = None):
    '''
    Plotting the average evoked activity to all photostimulated cells in the volume
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
        one_plane_stimulated_df = vizmotion_photostim_info_df[(vizmotion_photostim_info_df.plane == plane) & (vizmotion_photostim_info_df.resp_cell_id.str.contains('stim'))]
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



























# older functions, keeping here in case I need it later #

# def build_functional_types_df(omr_fishvolume, stim_fishvolume, omr_fishvolume_barcoding_df):
#     '''
#     Build the master dataframe that would contain all the functional types of neurons in the OMR stack and compared to all the stim datasets
#     omr_fishvolume = VolumeFish object, OMR dataset, needs to have tail data too
#     stim_fishvolume = VolumeFish object, concatenated photostimulation dataset (one big dataset)
#     volume_barcoding_df = dataframe, barcoding data for each neuron in the OMR stack

#     will plot the plane and location of photostimulated cells

#     returns a dataframe with the following columns:
#     plane - int, plane number
#     omr_neur_id - int, neuron id in the OMR stack
#     neur_coords - list, coordinates of the OMR cell
#     visual_barcode - str, visual motion barcoding of the neuron
#     region - str, brain region of the neuron in the OMR stack (Pt, Hb, nMLF or nan if not in these)
#     motor_corr - float, motor correlation of the neuron in the OMR stack
#     photostim - bool, if the neuron was photostimulated
#     if photostim...
#         stim_events - list, the event order (number) of when that neuron was photostimulated
#         stim_frames - list, frames in the stim dataset of when that neuron was photostimulated
#     stim_neur_id - int, neuron id photostimulation dataset
#     '''

#     # process the functional information for all neurons into a giant dataframe, now just working with the OMR fish volume and the stim fish volume

#     df_lst = [] 
#     for plane in np.unique(omr_fishvolume_barcoding_df.plane.values):
#         sub_functional_types_df = pd.DataFrame(columns = ['plane', 'omr_neur_id', 'neur_coords', 'visual_barcode', 'region', 'motor_corr', 'photostim'])
#         omr_plane_barcoding_df = omr_fishvolume_barcoding_df[omr_fishvolume_barcoding_df.plane == plane]
#         omr_tailFish = omr_fishvolume.volumes[plane]
#         omr_data_rois = omr_tailFish.return_cell_rois(range(len(omr_tailFish.f_cells)))

#         sub_functional_types_df['omr_neur_id'] = range(len(omr_tailFish.f_cells))
#         sub_functional_types_df['neur_coords'] = omr_data_rois
#         sub_functional_types_df['plane'] = [plane] * len(sub_functional_types_df)
#         sub_functional_types_df['motor_corr'] = omr_tailFish.motor_pearson_corrs

#         # default is none/false for these functional identities
#         sub_functional_types_df['region'] = [np.nan] * len(sub_functional_types_df)
#         sub_functional_types_df['visual_barcode'] = ['None'] * len(sub_functional_types_df)
#         sub_functional_types_df['photostim'] = [False] * len(sub_functional_types_df)
#         sub_functional_types_df['stim_frames'] = [None] * len(sub_functional_types_df)
#         sub_functional_types_df['stim_events'] = [None] * len(sub_functional_types_df)
        
#         for l in sub_functional_types_df.omr_neur_id.values:
#             if l in omr_plane_barcoding_df.neur_ids.values:
#                 sub_functional_types_df['visual_barcode'].iloc[l] = omr_plane_barcoding_df[omr_plane_barcoding_df.neur_ids == l].barcoding.values[0]
#             if l in omr_tailFish.return_cells_by_saved_roi('Hb'):
#                 sub_functional_types_df['region'].iloc[l] = 'Hb'
#             if l in omr_tailFish.return_cells_by_saved_roi('Pt'):
#                 sub_functional_types_df['region'].iloc[l] = 'Pt'
#             if 'nMLF' in omr_tailFish.roi_dict.keys():  # sometimes there is not a nMLF region with cells, so would run an error
#                 if l in omr_tailFish.return_cells_by_saved_roi('nMLF'):
#                     sub_functional_types_df['region'].iloc[l] = 'nMLF'

#         stim_photostimFish = stim_fishvolume.volumes[plane]
#         matched_cell_ids = coordutils.match_cell_ids(sub_functional_types_df.omr_neur_id.values, omr_tailFish.stats,
#                                                         range(len(stim_photostimFish.f_cells)), stim_photostimFish.stats)
#         sub_functional_types_df['stim_neur_id'] = matched_cell_ids
            
#         # need to identify which neurons were photostimulated by matching spatially, add in the stim events and frames for each cell
#         plt.figure(figsize = (6, 6))
#         plt.imshow(stim_photostimFish.rescaled_ref, cmap = 'gray', vmax = np.percentile(stim_photostimFish.rescaled_ref, 99))
#         plt.title(f'{plane}')
#         if 'stim_sites' in stim_photostimFish.data_paths.keys():
#             # better to use closest center coordinates to match ROIs, rather than overlap
#             stim_photostimFish.stimmed_cell_coords, stim_photostimFish.stimmed_cell_id_array = PhotostimFish.identify_stim_cells(stim_photostimFish, overlap = False)
#             omr_photostim_cell_id_lst = [np.where(matched_cell_ids == int(s))[0][0] for s in stim_photostimFish.stimmed_cell_id_array]
#             stim_frames = stim_photostimFish.stim_sites_df.stim_frames.values
#             stim_events = stim_photostimFish.stim_sites_df.stim_events.values
#             for n in range(len(stim_photostimFish.stimmed_cell_id_array)):
#                 sub_functional_types_df.loc[[omr_photostim_cell_id_lst[n]], 'photostim'] = True
#                 sub_functional_types_df.loc[[omr_photostim_cell_id_lst[n]], 'stim_frames'] = pd.Series([stim_frames[n]], index = sub_functional_types_df.index[[omr_photostim_cell_id_lst[n]]])
#                 sub_functional_types_df.loc[[omr_photostim_cell_id_lst[n]], 'stim_events'] = pd.Series([stim_events[n]], index = sub_functional_types_df.index[[omr_photostim_cell_id_lst[n]]])
                
#                 # red is OMR-id'ed cell, white is photostim cell
#                 plt.scatter(sub_functional_types_df.neur_coords.loc[omr_photostim_cell_id_lst[n]][0], 
#                             sub_functional_types_df.neur_coords.loc[omr_photostim_cell_id_lst[n]][1], color = 'red', s = 10)
#                 plt.annotate(omr_photostim_cell_id_lst[n], (sub_functional_types_df.neur_coords.loc[omr_photostim_cell_id_lst[n]][0], 
#                                 sub_functional_types_df.neur_coords.loc[omr_photostim_cell_id_lst[n]][1]), color = 'red')
#                 plt.scatter(stim_photostimFish.stimmed_cell_coords[n][0], stim_photostimFish.stimmed_cell_coords[n][1], color = 'white', s = 10)
#                 plt.annotate(stim_photostimFish.stimmed_cell_id_array[n], (stim_photostimFish.stimmed_cell_coords[n][0], 
#                                                                             stim_photostimFish.stimmed_cell_coords[n][1]), color = 'white') 
#             plt.show() 
#         else:
#             pass  

#         df_lst.append(sub_functional_types_df)   

#     functional_types_df = pd.concat(df_lst).reset_index(drop = True)
    
#     return functional_types_df    

# def build_vizmotion_photostim_info_df(responder_functional_types_df, stimulated_functional_types_df, motion_cues, 
#                                       omr_fish_volume, stim_fish_volume, save = False, save_location = None):
#     '''
#     Build a dataframe that contains all of the motion responses and photostimulation responses together
#     This uses the 'build_functional_types_df' function to get the starting df of the responders and the stimulated cells
#     responder_functional_types_df - dataframe, the functional types df of the responding cells
#     stimulated_functional_types_df - dataframe, the functional types df of the stimulated cells
#     motion_cues - list, the motion cues to get responses from
#     omr_fish_volume - VolumeFish object, the OMR dataset (all the planes together)
#     stim_fish_volume - VolumeFish object, the photostimulation dataset (all the planes together)
#     save - bool, if you want to save the dataframe
#     save_location - Path, where to save the dataframe

#     returns a dataframe with the following columns:
#     resp_cell_id - str, the id of the responding cell
#     omr_neur_id - int, the neuron id in the OMR stack
#     stim_neur_id - int, the neuron id in the photostimulation dataset
#     neur_coords - list, the coordinates of the neuron
#     plane - int, the plane number
#     visual_barcode - str, the visual motion barcoding of the neuron (from before, may not be accurate)
#     photostim - bool, if the neuron was photostimulated
#     motion_responses - dict, the motion responses of the neuron to the motion cues
#     stim_responses - dict, the responses of the neuron to each photostimulated neuron
#     avg_evoked_df_f - list, the average evoked df f for each photostimulated neuron in the same order


#     '''

#     all_neurons_df = pd.concat([responder_functional_types_df, stimulated_functional_types_df]).reset_index(drop = True)

#     resp_f_trace_array, _, _ = prepare_data_for_plotting(all_neurons_df, stim_fish_volume)
#     stim_f_trace_array, _, _ = prepare_data_for_plotting(stimulated_functional_types_df, stim_fish_volume)

#     # photostimulation time window
#     window = [-4, 7]

#     # creating a new dataframe to store all of the information
#     new_info_df = pd.DataFrame(columns = ['resp_cell_id', 'omr_neur_id', 'stim_neur_id', 'neur_coords', 'plane', 'region', 'visual_barcode', 'photostim',
#                                         'motion_responses', 'stim_responses', 'avg_evoked_df_f'])
#     new_info_df['resp_cell_id'] = [f'resp_{i}' for i in range(len(responder_functional_types_df))] + [f'stim_{i}' for i in range(len(stimulated_functional_types_df))]
#     new_info_df['omr_neur_id'] = all_neurons_df.omr_neur_id
#     new_info_df['stim_neur_id'] = all_neurons_df.stim_neur_id
#     new_info_df['neur_coords'] = all_neurons_df.neur_coords
#     new_info_df['plane'] = all_neurons_df.plane
#     new_info_df['visual_barcode'] = all_neurons_df.visual_barcode
#     new_info_df['photostim'] = all_neurons_df.photostim
#     new_info_df['region'] = all_neurons_df.region

#     lst_motion_responses = []
#     lst_stim_responses = []
#     lst_avg_evoked = []

#     # for each responding neuron in the whole volume:
#     for i in range(len(all_neurons_df)):

#         # information about this cell
#         omr_cell_id = all_neurons_df.omr_neur_id.iloc[i]
#         plane_id = int(str(all_neurons_df.plane.iloc[i]).split('_')[1])

#         # get the visual responses of this cell, put it into a dictionary
#         motion_responsive_dict = BCDict()
#         omr_fish = omr_fish_volume[plane_id]
#         omr_cell_extended_raw_resp = pd.DataFrame(omr_fish.extended_responses_normf).iloc[omr_cell_id]
#         for s, stim in enumerate(motion_cues):
#             if stim not in motion_responsive_dict.keys():
#                 motion_responsive_dict[stim] = {}
#             motion_resp = omr_cell_extended_raw_resp[stim]
#             df_f_motion_resp = np.zeros(shape = np.array(motion_resp).shape)
#             for a, arr in enumerate(motion_resp):
#                 base_e = arr[:-omr_fish.offsets[0]]
#                 plot_e = (arr - np.nanmean(base_e)) / np.nanmean(base_e)
#                 df_f_motion_resp[a] = plot_e
#             motion_responsive_dict[stim]['mean'] = np.nanmean(df_f_motion_resp, axis = 0) 
#             motion_responsive_dict[stim]['std'] = np.nanstd(df_f_motion_resp, axis = 0)
        
#         # for each stimulated cell, gather the responses of the responding cell 
#         responding_trace = resp_f_trace_array[i]
#         resp_to_each_stimulation_dict = BCDict()
#         avg_evoked_df_f_lst = []
#         for s_ind, s_trace in enumerate(stim_f_trace_array):
#             s_cell_id = f'stim{s_ind}'
#             if s_cell_id not in resp_to_each_stimulation_dict.keys():
#                 resp_to_each_stimulation_dict[s_cell_id] = {}
#             stimmed_frames = stimulated_functional_types_df.stim_frames.values[s_ind]
#             frame_subset = arrutils.subsection_arrays(stimmed_frames, window)
#             if frame_subset[-1][-1] > len(s_trace): # adjust frames in case this is out of range
#                 new_frame_subset = []
#                 for s in frame_subset:
#                     new_frame_subset.append([q for q in s if q < len(s_trace)])
#                 frame_subset = np.array(new_frame_subset)

#             resp_raw_trial = np.array([responding_trace[g] for g in frame_subset if len(responding_trace[g] ) > 0])
#             resp_df_f_trial = np.zeros(shape = (len(resp_raw_trial), len(resp_raw_trial[0])))
#             stim_evoked_df_f_trial = np.zeros(shape = (len(resp_raw_trial), 1))
#             for d, f in enumerate(resp_raw_trial):
#                 base_e = f[:-window[0]]
#                 plot_e = (f - np.nanmean(base_e)) / np.nanmean(base_e)
#                 stim_evoked_df_f_trial[d] = np.nanmedian(plot_e[:-window[0]]) # evoked df f for each trial
#                 resp_df_f_trial[d] = plot_e # full trace for each trial
            
#             resp_to_each_stimulation_dict[s_cell_id] = [resp_df_f_trial]
#             avg_evoked_df_f_lst.append(np.nanmean(stim_evoked_df_f_trial))

#         # put all of this information into the new dataframe
#         lst_motion_responses.append(motion_responsive_dict)
#         lst_stim_responses.append(resp_to_each_stimulation_dict)
#         lst_avg_evoked.append(avg_evoked_df_f_lst)

#     new_info_df['motion_responses'] = lst_motion_responses
#     new_info_df['stim_responses'] = lst_stim_responses
#     new_info_df['avg_evoked_df_f'] = lst_avg_evoked

#     if save:
#         new_info_df.to_hdf(save_location.joinpath('vizmotion_photostim_info.h5'), key = 'df', mode = 'w')

#     return new_info_df










# this is to check the quality of the photostimulation dataset - wihtout automated gui
class SinglePhotostimPipeline:
    def __init__(self, 
                 PhotostimFishVolume, 
                 frame_offsets, 
                 saving_directory, 
                 VizStimFishVolume = None, 
                 min_max_vals = (-1, 1), 
                 pre_frames = 30):
        '''
        PhotostimFishVolume - VolumeFish object, photostimulation data
        VizStimFishVolume - VizVolumeFish object, visual motion data
        frame_offsets - list of int, the frame offsets to use for the photostimulation analysis
        '''
        self.PhotostimFishVolume = PhotostimFishVolume
        self.VizStimFishVolume = VizStimFishVolume
        self.frame_offsets = frame_offsets
        self.saving_directory = saving_directory

        volume_stim_sites_df = pd.read_hdf(self.PhotostimFishVolume[0].folder_path.parents[1].joinpath('stim_sites_volume.h5'))
        self.stimmed_plane_num = volume_stim_sites_df.plane.values[0]

        if hasattr(self.PhotostimFishVolume, "volumes"):
            stimmed_plane_fish = self.PhotostimFishVolume[self.stimmed_plane_num]
        else:
            stimmed_plane_fish = self.PhotostimFishVolume
        
        um_per_pxs = get_micronstopixels_scale(stimmed_plane_fish.data_paths['info_xml'])
        self.sp_size_pxs = stimmed_plane_fish.stim_sites_df.sp_size / um_per_pxs # diameter
        
        shift_shading_keyword = False
        # if 'caiman' in stimmed_plane_fish.data_paths.keys():
        #     shift_shading_keyword = True
        
        if not os.path.exists(self.saving_directory):
            os.makedirs(self.saving_directory)

        self.stimulated_cell_response(stimmed_plane_fish, shift_shading =shift_shading_keyword)
        plt.savefig(self.saving_directory.joinpath(f'photostim_cell_responses_plane{self.stimmed_plane_num}_{stimmed_plane_fish.folder_path.parents[1].name}.svg'), 
                    dpi = 300, bbox_inches = 'tight', transparent = True, format = 'svg')

        # for each plane in the volumetric stack:
        if self.VizStimFishVolume is not None:
            fig0, fig1 = self.zscored_fov_vizstim_reference(stimmed_plane_fish, vmin = min_max_vals[0], vmax = min_max_vals[1])
        else:
            fig0, fig1 = self.zscored_fov(stimmed_plane_fish)
        
        fig2 = self.pre_every_ps_event_imgs(stimmed_plane_fish, pre_frames)

        fig0.savefig(self.saving_directory.joinpath(f'zscored_bigfov_plane{self.stimmed_plane_num}_{stimmed_plane_fish.folder_path.parents[1].name}.png'), 
                     dpi = 300, bbox_inches = 'tight', format = 'png')
        fig1.savefig(self.saving_directory.joinpath(f'zscored_smallfov_plane{self.stimmed_plane_num}_{stimmed_plane_fish.folder_path.parents[1].name}.png'), 
                     dpi = 300, bbox_inches = 'tight', format = 'png')
        fig2.savefig(self.saving_directory.joinpath(f'pre_photostimevent_imgs_plane{self.stimmed_plane_num}_{stimmed_plane_fish.folder_path.parents[1].name}.png'),
                        dpi = 300, bbox_inches = 'tight', format = 'png')

    def stimulated_cell_response(self, stimmed_plane_fish, shift_shading = False):
        '''
        looking at the stimuluated cell's responses to each photostimulation event
        '''
        frame_subset = arrutils.subsection_arrays(stimmed_plane_fish.ps_event_start, self.frame_offsets) # frame numbers for each event
        single_stim  = stimmed_plane_fish.raw_traces[0]
        xmin_shading = -self.frame_offsets[0] - stimmed_plane_fish.ps_event_duration_frames
        xmax_shading = -self.frame_offsets[0]
        if shift_shading: # need to shift shading for caiman datasets
            xmin_shading = -self.frame_offsets[0] 
            xmax_shading = -self.frame_offsets[0] + stimmed_plane_fish.ps_event_duration_frames

        # iterate through each photostimulation event, then grab the frames before and after each event
        each_trial = np.array([single_stim[s] for s in frame_subset])

        fig, (ax1, ax0, ax2) = plt.subplots(nrows = 1, ncols = 3, figsize = (15,4))

        # plot heatmap
        sns.heatmap(each_trial, ax=ax0, xticklabels=10, cmap = 'viridis', 
                    )
        ax0.axvspan(xmin = xmin_shading, xmax = xmax_shading, color = 'red', alpha = 0.4)
        ax0.set_xlabel('Time (sec)')
        ax0.set_ylabel('Stimulation Events')
        ax0.set_xticks([0, -self.frame_offsets[0], -self.frame_offsets[0]*2,-self.frame_offsets[0]*3 ], 
                    (np.array([0, -self.frame_offsets[0], -self.frame_offsets[0]*2, -self.frame_offsets[0]*3]) *
                        stimmed_plane_fish.img_hz).astype(int))
        ax0.set_title(f'Heatmap showing response to each photostimulation event')

        # plot trace 
        every_event_in_sec = plotutils.convert_frame_to_sec(stimmed_plane_fish.ps_event_start, stimmed_plane_fish.img_hz)
        every_frame_in_sec = plotutils.convert_frame_to_sec(np.arange(len(single_stim)), stimmed_plane_fish.img_hz) 
        ax1.plot(every_frame_in_sec, arrutils.pretty(single_stim), color = 'k', linewidth = 1.5)
        for e in every_event_in_sec:
            if shift_shading:
                ax1.axvspan(xmin = e, xmax = e + int(stimmed_plane_fish.ps_event_duration/1000), color = 'red', alpha = 0.8)
            else:
                ax1.axvspan(xmin = e - int(stimmed_plane_fish.ps_event_duration/1000), xmax = e, color = 'red', alpha = 0.8)
        ax1.set_xlim(every_frame_in_sec[0], every_frame_in_sec[-1])
        ax1.set_xlabel('Time (sec)')
        ax1.set_ylabel('Raw Pixel Intensity')
        ax1.set_title(f'Raw trace')

        # plot normalized average evoked activity 
        norm_trial = np.zeros(shape = (len(each_trial), len(each_trial[0])))
        for d, f in enumerate(each_trial):
            base_e = f[:-self.frame_offsets[0]]
            plot_e = (f - np.nanmean(base_e)) / np.nanmean(base_e)
            norm_trial[d] = plot_e
        ax2.plot(np.arange(len(norm_trial[0])), arrutils.pretty(np.nanmean(norm_trial, axis = 0)), color = 'k')
        ci_lower, ci_upper = statutils.calculate_ci(norm_trial)
        ax2.fill_between(np.arange(len(norm_trial[0])), ci_lower, ci_upper, color='skyblue', 
                        alpha=0.4, label='95% CI')  # Shade confidence interval
        ax2.axvspan(xmin = xmin_shading, xmax = xmax_shading, color = 'red', alpha = 0.4)
        ax2.set_xticks([0, -self.frame_offsets[0], -self.frame_offsets[0]*2,-self.frame_offsets[0]*3 ], 
                    (np.array([0, -self.frame_offsets[0], -self.frame_offsets[0]*2, -self.frame_offsets[0]*3]) * stimmed_plane_fish.img_hz).astype(int))
        ax2.set_xlabel('Time (sec)')
        ax2.set_ylabel('Normalized Pixel Intensity')
        ax2.set_ylim(-1.0, 1.0)
        ax2.set_title(f'Mean evoked dF/F for photostimmed site')
        ax2.legend()

        fig.suptitle('Activity of photostimmed site activity')
        plt.tight_layout()

    def zscored_fov(self, plane_fish, vmin = -1, vmax = 2):
        '''
        zscored activity of the field of view
        '''
        cmap = plotutils.build_cmap_blue_to_red()

        img_stack = plane_fish.load_image()
        avg_image = np.nanmean(img_stack, axis = 0)
        std_image = np.std(img_stack, axis=0)
        z_scored_stack = (img_stack - avg_image) / std_image

        #find the averaged z-scored stack for each photostimulation event
        avg_over_trials_z_scored_stack = np.zeros(shape = (len(plane_fish.ps_event_start), 
                                                   z_scored_stack.shape[1], z_scored_stack.shape[2]))
        
        instant_photostim_response_frames = round(plane_fish.ps_event_duration / 1000 * plane_fish.img_hz)
        
        for k, l in enumerate(plane_fish.ps_event_start):
            trial_z_scored_stack = z_scored_stack[l:l + instant_photostim_response_frames, :, :]
            if 'caiman' in plane_fish.data_paths.keys():
                trial_z_scored_stack = z_scored_stack[l + instant_photostim_response_frames:l + 2*instant_photostim_response_frames, :, :]
            trial_z_scored_image = np.nanmean(trial_z_scored_stack, axis=0)
            avg_over_trials_z_scored_stack[k] = trial_z_scored_image
        avg_over_trials_z_scored_image = np.nanmean(avg_over_trials_z_scored_stack, axis = 0)
        norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        
        fig0, (ax0, ax1) = plt.subplots(1, 2, figsize = (8,10))
        # anatomical image
        im0 = ax0.imshow(avg_image, cmap = 'gray', vmax = np.nanpercentile(avg_image, 99))
        ax0.axis('off')
        fig0.colorbar(im0, ax=ax0, shrink=0.4) 
        # ax0.scatter(plane_fish.stim_sites_df.x_stim, plane_fish.stim_sites_df.y_stim, 
        #             edgecolor = 'red', facecolor = 'none', s = 30)
        stim_site_circle = Circle((plane_fish.stim_sites_df.x_stim, plane_fish.stim_sites_df.y_stim), self.sp_size_pxs/2, color='red', fill=False, linewidth=2)
        ax0.add_patch(stim_site_circle)
        ax0.set_title('Anatomical image')

        # zscored image
        im1 = ax1.imshow(avg_over_trials_z_scored_image, cmap = cmap, norm=norm,)
        # ax1.scatter(plane_fish.stim_sites_df.x_stim, plane_fish.stim_sites_df.y_stim, 
        #             edgecolor = 'red', facecolor = 'none', s = 30)
        stim_site_circle = Circle((plane_fish.stim_sites_df.x_stim, plane_fish.stim_sites_df.y_stim), self.sp_size_pxs/2, color='red', fill=False, linewidth=2)
        ax1.add_patch(stim_site_circle)
        ax1.axis('off')
        ax1.set_title('Z-scored image')
        fig0.colorbar(im1, ax=ax1, shrink=0.4) 
        fig0.tight_layout()

        # smaller FOV
        # anatomical image
        fig1, (ax0, ax1) = plt.subplots(1, 2, figsize = (8,10), sharex = True, sharey = True)    
        im0 = ax0.imshow(avg_image, cmap = 'gray', vmax = np.nanpercentile(avg_image, 99))
        ax0.axis('off')
        fig1.colorbar(im0, ax=ax0, shrink=0.3) 
        xlim = ax0.set_xlim(int(plane_fish.stim_sites_df.x_stim - 50), int(plane_fish.stim_sites_df.x_stim + 50))
        ylim = ax0.set_ylim(int(plane_fish.stim_sites_df.y_stim + 50), int(plane_fish.stim_sites_df.y_stim - 50))
        center_x = (xlim[0] + xlim[1]) / 2
        center_y = (ylim[0] + ylim[1]) / 2
        # ax0.scatter(center_x, center_y, edgecolor = 'red', facecolor = 'none', s = 250)
        smaller_FOV_stim_site_circle = Circle((center_x, center_y), self.sp_size_pxs/2, color='red', fill=False, linewidth=2)
        ax0.add_patch(smaller_FOV_stim_site_circle)
        ax0.set_title('Anatomical image')

        # zscored image
        im1 = ax1.imshow(avg_over_trials_z_scored_image, cmap = cmap, norm=norm)
        fig1.colorbar(im1, ax=ax1, shrink=0.3) 
        # ax1.scatter(center_x, center_y,edgecolor = 'red', facecolor = 'none', s = 250)
        ax1.add_patch(smaller_FOV_stim_site_circle)
        ax1.axis('off')
        ax1.set_title('Z-scored image')
        plt.tight_layout()

        return fig0, fig1

    def zscored_fov_vizstim_reference(self, photostim_plane_fish, vmin = -1, vmax = 2):
        '''
        Adding in a Reference image from the visstim volume if that is present
        
        '''
        cmap = plotutils.build_cmap_blue_to_red()

        # reference OMR image
        omr_img_stack = self.VizStimFishVolume[self.stimmed_plane_num].load_image()
        omr_img_avg_image = np.nanmean(omr_img_stack, axis = 0)

        # photostimulation dataset image
        img_stack = photostim_plane_fish.load_image()
        avg_image = np.nanmean(img_stack, axis = 0)
        std_image = np.std(img_stack, axis=0)
        z_scored_stack = (img_stack - avg_image) / std_image

        #find the averaged z-scored stack for each photostimulation event
        avg_over_trials_z_scored_stack = np.zeros(shape = (len(photostim_plane_fish.ps_event_start), 
                                                   z_scored_stack.shape[1], z_scored_stack.shape[2]))
        instant_photostim_response_frames = round(photostim_plane_fish.ps_event_duration / 1000 * photostim_plane_fish.img_hz)
        for k, l in enumerate(photostim_plane_fish.ps_event_start):
            trial_z_scored_stack = z_scored_stack[l:l + instant_photostim_response_frames, :, :]
            trial_z_scored_image = np.nanmean(trial_z_scored_stack, axis=0)
            avg_over_trials_z_scored_stack[k] = trial_z_scored_image
        avg_over_trials_z_scored_image = np.nanmean(avg_over_trials_z_scored_stack, axis = 0)
        norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        # full FOV
        fig0, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize = (10,12))
        # reference OMR anatomical image
        im0 = ax0.imshow(omr_img_avg_image, cmap = 'gray', vmax = np.nanpercentile(omr_img_avg_image, 99))
        ax0.axis('off')
        fig0.colorbar(im0, ax=ax0, shrink=0.25) 
        stim_site_circle = Circle((photostim_plane_fish.stim_sites_df.x_stim, photostim_plane_fish.stim_sites_df.y_stim), 
                                  self.sp_size_pxs/2, color='red', fill=False, linewidth=2, )
        ax0.add_patch(stim_site_circle)
        ax0.set_title('OMR reference image')

        # anatomical photostim dataset image
        im1 = ax1.imshow(avg_image, cmap = 'gray', vmax = np.nanpercentile(avg_image, 99))
        ax1.axis('off')
        fig0.colorbar(im1, ax=ax1, shrink=0.25) 
        stim_site_circle = Circle((photostim_plane_fish.stim_sites_df.x_stim, photostim_plane_fish.stim_sites_df.y_stim), 
                                  self.sp_size_pxs/2, color='red', fill=False, linewidth=2, )
        ax1.add_patch(stim_site_circle)
        ax1.set_title('Anatomical of stimulation data stack')

        # zscored image
        im2 = ax2.imshow(avg_over_trials_z_scored_image, cmap = cmap, norm=norm,)
        fig0.colorbar(im2, ax=ax2, shrink=0.25) 
        stim_site_circle = Circle((photostim_plane_fish.stim_sites_df.x_stim, photostim_plane_fish.stim_sites_df.y_stim), 
                                  self.sp_size_pxs/2, color='red', fill=False, linewidth=2,)
        ax2.add_patch(stim_site_circle)
        ax2.axis('off')
        ax2.set_title('Z-scored image')
        fig0.tight_layout()

        # SMALLER FOV
        fig1, (ax0, ax1, ax3, ax2) = plt.subplots(1, 4, figsize = (10,10), sharex = True, sharey = True)

        # omr image    
        im0 = ax0.imshow(omr_img_avg_image, cmap = 'gray', vmax = np.nanpercentile(omr_img_avg_image, 99))
        ax0.axis('off')
        # fig1.colorbar(im0, ax=ax0, shrink=0.25) 
        xlim = ax0.set_xlim(int(photostim_plane_fish.stim_sites_df.x_stim - 50), int(photostim_plane_fish.stim_sites_df.x_stim + 50))
        ylim = ax0.set_ylim(int(photostim_plane_fish.stim_sites_df.y_stim + 50), int(photostim_plane_fish.stim_sites_df.y_stim - 50))
        center_x = (xlim[0] + xlim[1]) / 2
        center_y = (ylim[0] + ylim[1]) / 2
        smaller_FOV_stim_site_circle = Circle((center_x, center_y), self.sp_size_pxs/2, color='red', fill=False, linewidth=2, linestyle='--')
        ax0.add_patch(smaller_FOV_stim_site_circle)
        ax0.set_title('OMR reference image')

        # anatomical photostim dataset image
        im1 = ax1.imshow(avg_image, cmap = 'gray', vmax = np.nanpercentile(avg_image, 99))
        ax1.axis('off')
        # fig1.colorbar(im1, ax=ax1, shrink=0.25) 
        smaller_FOV_stim_site_circle = Circle((center_x, center_y), self.sp_size_pxs/2, color='red', fill=False, linewidth=2, linestyle='--')
        ax1.add_patch(smaller_FOV_stim_site_circle)
        ax1.set_title('Stimulation data image')

        # overlay of omr and stim dataset images
        ax3.imshow(omr_img_avg_image, cmap = 'Blues', vmax = np.nanpercentile(omr_img_avg_image, 99), label = 'OMR')
        ax3.imshow(avg_image, cmap = 'Reds', vmax = np.nanpercentile(avg_image, 99), alpha = 0.5, label = 'Stimulation')
        ax3.axis('off')
        smaller_FOV_stim_site_circle = Circle((center_x, center_y), self.sp_size_pxs/2, color='red', fill=False, linewidth=2, linestyle='--')
        ax3.add_patch(smaller_FOV_stim_site_circle)
        red_patch = Patch(color='red', label='OMR')
        blue_patch = Patch(color='blue', label='Stimulation')
        ax3.legend(handles=[red_patch, blue_patch], loc="lower left")  
        ax3.set_title('Overlay')

        # zscored image
        im2 = ax2.imshow(avg_over_trials_z_scored_image, cmap = cmap, norm=norm,)
        fig1.colorbar(im2, ax=ax2, shrink=0.25) 
        smaller_FOV_stim_site_circle = Circle((center_x, center_y), self.sp_size_pxs/2, color='red', fill=False, linewidth=2, linestyle='--')
        ax2.add_patch(smaller_FOV_stim_site_circle)
        ax2.axis('off')
        ax2.set_title('Z-scored image')
        fig1.tight_layout()

        return fig0, fig1
    
    def pre_every_ps_event_imgs(self, photostim_plane_fish, pre_frames = 30, post_frames = 5, vmin = -0.05, vmax = 0.05):
        '''
        Look over the stimulation spot before every photostim event in the stim dataset to make sure I am always hitting the right spot
        And finding out the z-scored activity of the FOV
        pre_frames = number of frames to average over before the photostimulation event
        '''
        full_img = photostim_plane_fish.load_image()
        fig, ax = plt.subplots(2, len(photostim_plane_fish.ps_event_start), figsize = (20, 5))
        stim_x = photostim_plane_fish.stim_sites_df.x_stim
        stim_y = photostim_plane_fish.stim_sites_df.y_stim
        cmap = plotutils.build_cmap_blue_to_red()
        norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        for n, p in enumerate(photostim_plane_fish.ps_event_start):
            pre_stim_img = np.nanmean(full_img[p-pre_frames:p, :,:], axis = 0)
            post_stim_img = full_img[p:p+post_frames, :,:]
            post_stim_avg_img = np.nanmean(post_stim_img, axis = 0)
            post_stim_std_img = np.std(post_stim_img, axis=0)
            post_stim_z_img = (post_stim_img - post_stim_avg_img) / post_stim_std_img

            img = ax[0, n].imshow(pre_stim_img, cmap = 'gray', vmax = np.percentile(pre_stim_img, 99.9)*0.5)
            xlim = ax[0, n].set_xlim(int(stim_x - 50), int(stim_x + 50))
            ylim = ax[0, n].set_ylim(int(stim_y + 50), int(stim_y - 50), )
            center_x = (xlim[0] + xlim[1]) / 2
            center_y = (ylim[0] + ylim[1]) / 2
            smaller_FOV_stim_site_circle = Circle((center_x, center_y), self.sp_size_pxs/2, color='red', fill=False, linewidth=2,)
            ax[0, n].add_patch(smaller_FOV_stim_site_circle)
            ax[0, n].axis('off')   
            ax[0, n].set_title(f'Event {n}')

            ax[1, n].imshow(np.nanmean(post_stim_z_img, axis = 0), cmap = cmap, norm = norm)
            print(f'min pixel value = {np.min(np.nanmean(post_stim_z_img, axis = 0))}, max pixel value = {np.max(np.nanmean(post_stim_z_img, axis = 0))}')
            xlim = ax[1, n].set_xlim(int(stim_x - 50), int(stim_x + 50))
            ylim = ax[1, n].set_ylim(int(stim_y + 50), int(stim_y - 50), )
            center_x = (xlim[0] + xlim[1]) / 2
            center_y = (ylim[0] + ylim[1]) / 2
            smaller_FOV_stim_site_circle = Circle((center_x, center_y), self.sp_size_pxs/2, color='red', fill=False, linewidth=2,)
            ax[1, n].add_patch(smaller_FOV_stim_site_circle)
            ax[1, n].axis('off') 

        fig.tight_layout()
        
        return fig