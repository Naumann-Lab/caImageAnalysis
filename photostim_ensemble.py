
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from itertools import combinations
import scipy
import random
from bcdict import BCDict
import math

# local imports
import constants, tailtracking, photostim_data_pipeline
from utilities import coordutils, arrutils
from fishy import PhotostimFish


# --- PREPROCESSING TO CREATE ENSEMBLES FOR PHOTOSTIM --- #

# create nested ensembles of specific sizes
def build_nested_ensembles(df, # the dataframe with just the photostimulateable cells
                           group_sizes = [1, 4, 7, 12],
                           um_per_pixel = 0.52
                           ):
    '''
    Building grades (nested) ensembles with options for large group of ensembles
    :param df: dataframe that contains the location, plane, side (or barcode), and photostim cell number (arbirtrary cell number used for unique cell ids)
    :param um_per_pixel: scaling factor to convert pixels to ums
    :param skip_sizes: list of ensemble sizes to skip (helpful for doing just a couple of ensemble sized groups
    :return: ensemble dataframe with the groups in each row, with location, type, etc
    '''
    ensembles = []
    for size in group_sizes:
        for combo in combinations(df.index, size):
            sub = df.loc[list(combo)]
            ensemble_cells = sub["photostim_cell_num"].tolist()
            ensemble_coords = sub.neur_coords.tolist()

            # compute pairwise distances within this ensemble
            dists = scipy.spatial.distance.pdist(ensemble_coords, 'euclidean')
            min_pairwise_dist = dists.min() if len(dists) > 0 else np.inf
            min_pairwise_dist_um = np.inf if min_pairwise_dist == np.inf else min_pairwise_dist*um_per_pixel

            # coords include plane: (x,y,plane)
            ensemble_coords_with_plane = [
                (row.neur_coords[0], row.neur_coords[1], row.plane)
                for row in sub.itertuples()]

            ensembles.append({
                    "barcoding": sub["barcoding"].iloc[0],
                    "ensemble_size": size,
                    "ensemble_cells": ensemble_cells,
                    "ensemble_coords": ensemble_coords_with_plane,
                    "min_dist_px": min_pairwise_dist,
                    "min_dist_um": min_pairwise_dist_um
            })

    e_df = pd.DataFrame(ensembles)

    return e_df

def stratified_sample_equal_change(df, n_per_size=5, seed=None):
    '''
    Grab equally random groups from each ensemble size that is in the dataframe
    :param df: nested ensemble dataframe
    :param n_per_size: number of ensembles to randomly choose from each unique ensemble size
    :param seed: can change the seed number if you want to try different seeds
    :return: list of chosen indices
    '''
    rng = random.Random(seed)
    chosen = []

    for size, group in df.groupby("ensemble_size"):
        # sample up to n_per_size from each ensemble_size
        k = min(n_per_size, len(group))
        chosen.extend(rng.sample(list(group.index), k))

    return chosen


def stratified_sample(df, n_per_size=5, seed=None, mm_weight=3):
    '''
    Grab equally random groups from each ensemble size in the dataframe,
    with preference for barcodes containing "Mm".

    Works in Python <3.9 (no counts= for random.sample).
    '''
    rng = random.Random(seed)
    chosen = []

    for size, group in df.groupby("ensemble_size"):
        k = min(n_per_size, len(group))

        # higher weight if barcoding has "Mm"
        weights = [mm_weight if "Mm" in str(bc) else 1 for bc in group['barcoding']]
        indices = group.index.tolist()

        # sample without replacement but weighted
        picked = set()
        while len(picked) < k:
            # draw one index, weighted
            choice = rng.choices(indices, weights=weights, k=1)[0]
            picked.add(choice)

        chosen.extend(picked)

    return list(chosen)

# add a control ensemble way above the top of the fish, same coordinates as the largest ensemble
def add_control_ensemble_above(e_df):
    # take the largest ensemble in the df
    max_e_size = np.max(e_df.ensemble_size.values)
    largest_ensemble_row = e_df[e_df.ensemble_size == max_e_size].iloc[0] # just grab the first of these
    control_row = {}
    for each_col in largest_ensemble_row.index:
        if each_col not in control_row.keys():
            control_row[each_col] = []
        if each_col == 'barcoding':
            control_row[each_col] = 'control_above'
        elif each_col == 'ensemble_coords':
            control_row[each_col] = [(i[0], i[1], -1) for i in largest_ensemble_row['ensemble_coords']]
        elif each_col == 'ensemble_cells':
            control_row[each_col] = [-1]* len(largest_ensemble_row['ensemble_cells'])
        else:
            control_row[each_col] = largest_ensemble_row[each_col]

    new_e_df = e_df.append(control_row, ignore_index=True)

    return new_e_df

def add_both_side_ensemble(ensemble_df, right_Pt_row, left_Pt_row, ensemble_name = 'all'):
    '''
    Add combined hemisphere ensemble (right and left Pt)
    right_Pt_row: the full row of the ensemble on the right side (from the 'ensemble_df')
    left_Pt_row: the full row of the ensemble on the left side (from the 'ensemble_df')
    ensemble_df: the dataframe with all the ensembles together (each row)
    ensemble_name: name of the new combined across sides ensemble, default = 'all'
    By changing the name, you can add in several ensembles stimulated across hemispheres ('all')

    returns: the new ensemble df, with the added control ensemble
    '''
    both_side_stim_row = {'barcoding': ensemble_name,
                            'ensemble_size': right_Pt_row.ensemble_size + left_Pt_row.ensemble_size,
                            'ensemble_cells': right_Pt_row.ensemble_cells + left_Pt_row.ensemble_cells ,
                            'ensemble_coords': right_Pt_row.ensemble_coords + left_Pt_row.ensemble_coords ,
                            'min_dist_px': min([right_Pt_row.min_dist_px, left_Pt_row.min_dist_px]),
                            'min_dist_um': min([right_Pt_row.min_dist_um, left_Pt_row.min_dist_um])}
    new_ensemble_df = ensemble_df.append(both_side_stim_row, ignore_index=True)

    return new_ensemble_df

def add_control_shiftx(ensemble_df, e_coords, shift_x, control_e_name = 'control_shiftx'):
    '''
    Add a control ensemble, shifted in x direction (off the fish brain)
    ensemble_df: the dataframe with all the ensembles together (each row)
    e_coords: coordinates of the ensemble that you just want to shift to the right/left in x
    shift_x: the distance to shift in x direction (always add in this value)
    control_e_name: the name of the new control ensemble, default: 'control_shiftx'

    returns: the new ensemble df, with the added control ensemble
    '''

    shiftx_coords = [(e[0]+shift_x, e[1], e[2]) for e in e_coords] #(x, y, plane)
    new_row_for_df = {'barcoding': control_e_name,
                        'ensemble_size': len(shiftx_coords),
                        'ensemble_cells': [-1]*len(shiftx_coords),
                        'ensemble_coords': shiftx_coords,
                        'min_dist_px': 0,
                        'min_dist_um': 0}
    new_ensemble_df = ensemble_df.append(new_row_for_df, ignore_index=True)

    return new_ensemble_df


def collect_unique_ensemble_coords(n_cells, list_of_images):
    '''
    Collect unique ensemble coordinates from clicking on FOV
    n_cells: number of cells to select for this unique ensemble
    list_of_images: list of all the images to select cells from (typically the mean image per plane)
    '''
    selector = roiutils.MultiPlaneCellSelector(list_of_images)
    selected_cells = selector.select_cells(n_cells=n_cells)
    return selected_cells


def add_unique_ensemble(ensemble_df, unique_coords, list_of_images, e_name='unique', um_per_pixel=0.6, plot=True):
    '''
    Add in a unique ensemble just by clicking points across the FOV
    Will need to add in new cell names for the unique cell numbers (if on the brain, otherwise -1 is good)
    ensemble_df: the dataframe with all the ensembles together (each row)
    unique_coords: coordinate list, each coord is in shape (x, y, z)
    list_of_images: list of all the images to select cells from (typically the mean image per plane)
    e_name: name of the new ensemble, default = 'unique'
    um_per_pixel: scale of um/px, default = 0.6
    plot: True/False, plot the location of the new ensemble chosen

    returns: the new ensemble df, with the added unique ensemble in the last row
    '''

    dists = scipy.spatial.distance.pdist(unique_coords, 'euclidean')
    min_pairwise_dist = dists.min() if len(dists) > 0 else np.inf
    min_pairwise_dist_um = np.inf if min_pairwise_dist == np.inf else min_pairwise_dist * um_per_pixel

    if plot:
        fig, ax = plt.subplots(1, len(list_of_images), figsize=(10, 10))
        for i, img in enumerate(list_of_images):
            ax[i].imshow(img, cmap='gray', vmax=np.percentile(img, 99))
            ax[i].axis('off')
            ax[i].set_title(f'plane {i}')
        for x, y, z in unique_coords:
            ax[z].scatter(x, y)
        plt.tight_layout()
        plt.show()

    new_row_for_df = {'barcoding': e_name,
                      'ensemble_size': len(unique_coords),
                      'ensemble_cells': [-1] * len(unique_coords),
                      'ensemble_coords': unique_coords,
                      'min_dist_px': min_pairwise_dist,
                      'min_dist_um': min_pairwise_dist_um}
    new_ensemble_df = ensemble_df.append(new_row_for_df, ignore_index=True)

    return new_ensemble_df



# --- ANALYZING DATA FROM ENSEMBLE STIMULATION --- #
def build_functional_types_df_for_ensembles(omr_fishvolume,
                                            stim_fishvolume,
                                            omr_fishvolume_barcoding_df,
                                            # match_cells_within_radius_um=10, # no longer need this with new way to match Bruker to cell
                                            regions=['Pt', 'Hb', 'nMLF'],
                                            motor_correlation=False,
                                            plot_stim_sites=True):
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
    final_stim_key_order = []
    for plane in np.unique(omr_fishvolume_barcoding_df.plane.values):
        sub_functional_types_df = pd.DataFrame(
            columns=['resp_cell_id', 'omr_neur_id', 'stim_neur_id', 'neur_coords', 'plane', 'region',
                     'visual_barcode', 'motion_responses', 'photostim', 'stim_frames', 'stim_events'])
        omr_plane_barcoding_df = omr_fishvolume_barcoding_df[omr_fishvolume_barcoding_df.plane == plane]
        omr_Fish = omr_fishvolume.volumes[plane]
        omr_Fish.load_saved_rois()
        print(f'adding regions {omr_Fish.roi_dict.keys()}')
        omr_data_rois = omr_Fish.return_cell_rois(range(len(omr_Fish.f_cells)))

        sub_functional_types_df['omr_neur_id'] = range(len(omr_Fish.f_cells))
        print('loading omr data')
        all_motion_responses_lst = photostim_data_pipeline.gather_visual_motion_responses_for_df(omr_Fish, cell_id_array=None,
                                                                         motion_cues=constants.photostim_motion_cues)
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
                sub_functional_types_df['visual_barcode'].iloc[l] = \
                omr_plane_barcoding_df[omr_plane_barcoding_df.neur_ids == l].barcoding.values[0]
            for k in regions:
                if k in omr_Fish.roi_dict.keys():
                    if l in omr_Fish.return_cells_by_saved_roi(k):
                        sub_functional_types_df['region'].iloc[l] = k

        stim_photostimFish = stim_fishvolume.volumes[plane]
        # gather xy_offsets if there are any
        if os.path.exists(stim_photostimFish.folder_path.parents[1].joinpath('omr_to_stim_data_offsets.npy')):
            xy_offset_dictionary = np.load(
                stim_photostimFish.folder_path.parents[1].joinpath('omr_to_stim_data_offsets.npy'),
                allow_pickle='TRUE').item()
            # only if the drift is more than half a cell body, should we worry about offsets
            drift_offset = 4
            if (abs(float(xy_offset_dictionary['x_drift_px'])) <= drift_offset) & (
                    abs(float(xy_offset_dictionary['y_drift_px'])) <= drift_offset):
                xy_offset = (0, 0)
            elif (abs(float(xy_offset_dictionary['x_drift_px'])) <= drift_offset) & (
                    abs(float(xy_offset_dictionary['y_drift_px'])) > drift_offset):
                xy_offset = (0, float(xy_offset_dictionary['y_drift_px']))
            elif (abs(float(xy_offset_dictionary['x_drift_px'])) > drift_offset) & (
                    abs(float(xy_offset_dictionary['y_drift_px'])) <= drift_offset):
                xy_offset = (float(xy_offset_dictionary['x_drift_px']), 0)
            else:
                xy_offset = (float(xy_offset_dictionary['x_drift_px']), float(xy_offset_dictionary['y_drift_px']))
            print(f'xy offset between datasets = {xy_offset}')
        else:
            xy_offset = (0, 0)
            print('no xy offset saved between datasets')

        # in case there are no stimmed cells on that plane
        if 'stim_sites' in stim_photostimFish.data_paths.keys():
            photostimmed_cells = stim_photostimFish.stimmed_cell_id_array
        else:
            photostimmed_cells = []

        matches_omr_to_stim_dict, aligned_stim_coords = coordutils.match_omr_to_stim_coords(
                    omr_indices=np.arange(len(omr_Fish.f_cells)),
                    omr_coords=omr_Fish.return_cell_rois(range(len(omr_Fish.f_cells))),
                    stim_indices=np.arange(len(stim_photostimFish.f_cells)),
                    stim_coords=stim_photostimFish.return_cell_rois(range(len(stim_photostimFish.f_cells))),
                    stimmed_indices=photostimmed_cells,
                    xy_offset=xy_offset,
                    max_distance=7,
                    tile_size=30)

        # Fill DataFrame column
        print('matching OMR and Stim cells')
        ordered_omr_to_stim_cell_matches = []
        for omr_idx in sub_functional_types_df.omr_neur_id.values:
            try:
                stim_cell_id = matches_omr_to_stim_dict[omr_idx]
                ordered_omr_to_stim_cell_matches.append(stim_cell_id)
            except:
                ordered_omr_to_stim_cell_matches.append('None')
        sub_functional_types_df['stim_neur_id'] = ordered_omr_to_stim_cell_matches

        if 'stim_sites' in stim_photostimFish.data_paths.keys():
            # need to identify which neurons were photostimulated by matching spatially between OMR and stim datasets
            print('matching stimulation sites with OMR cells')

            # sort the stimmed cell ids by the cell id order of the stim sites df
            # important for the next step of matching the stim events and frames
            key_order = stim_photostimFish.stim_sites_df.cell_ids.values
            print(f'unsorted df order: {key_order}')
            sorted_stimmed_closest_cell_id_dict = {key: stim_photostimFish.stimmed_cells_matched_stim_ids_dict[key] for
                                                   key in key_order if
                                                   key in stim_photostimFish.stimmed_cells_matched_stim_ids_dict.keys()}
            sorted_key_order = list(sorted_stimmed_closest_cell_id_dict.keys())
            final_stim_key_order.append(sorted_key_order)
            print(f'sorted df order: {sorted_key_order}')
            print(f'sorted dict: {stim_photostimFish.stimmed_cells_matched_stim_ids_dict}')
            stimmed_cell_id_array = np.array([a for a in sorted_stimmed_closest_cell_id_dict.values()])
            print(f'sorted stim cell array: {stimmed_cell_id_array}')

            # gather the matching OMR cells to each stimulated cell
            omr_photostim_cell_id_lst = [] # list of matching omr to stim cells (for plotting)
            bad_indices = [] # list of all the stim cells that don't match an OMR cell
            # match the OMR cell id to the photostimmed cell number & add to the dataframe
            for n, stim_site_idx in enumerate(sorted_key_order):
                stim_cell = int(sorted_stimmed_closest_cell_id_dict[stim_site_idx])
                stim_frames = stim_photostimFish.stim_sites_df[
                    stim_photostimFish.stim_sites_df.cell_ids == stim_site_idx].stim_frames.values[0]
                stim_events = stim_photostimFish.stim_sites_df[
                    stim_photostimFish.stim_sites_df.cell_ids == stim_site_idx].stim_events.values[0]
                if stim_cell not in matches_omr_to_stim_dict.values():
                    omr_photostim_cell_id_lst.append('None')
                    bad_indices.append(n)

                    stim_cell_roi = stim_photostimFish.return_singlecell_rois(stim_cell)
                    plane = stim_photostimFish.stim_sites_df[
                        stim_photostimFish.stim_sites_df.cell_ids == stim_site_idx].plane.values[0]
                    new_row = {'resp_cell_id': f'stim_{stim_site_idx}', 'omr_neur_id': 'None', 'stim_neur_id': stim_cell,
                               'neur_coords': stim_cell_roi, 'plane': plane,'region': 'Pt',
                               'visual_barcode': 'None', 'motion_responses': 'None',
                               'photostim': True, 'stim_frames': stim_frames, 'stim_events': stim_events}
                    sub_functional_types_df.loc[len(sub_functional_types_df)] = new_row
                else:
                    omr_cell_id = int([o_cell_id for o_cell_id, s_cell_id in matches_omr_to_stim_dict.items() if stim_cell == s_cell_id][0])
                    omr_photostim_cell_id_lst.append(omr_cell_id)

                    sub_functional_types_df.loc[[omr_cell_id], 'stim_neur_id'] = stim_cell
                    sub_functional_types_df.loc[[omr_cell_id], 'photostim'] = True
                    sub_functional_types_df.loc[[omr_cell_id], 'resp_cell_id'] = f'stim_{stim_site_idx}'
                    sub_functional_types_df.loc[[omr_cell_id], 'stim_frames'] = pd.Series([stim_frames],
                        index=sub_functional_types_df.index[[omr_cell_id]])
                    sub_functional_types_df.loc[[omr_cell_id], 'stim_events'] = pd.Series([stim_events],
                        index=sub_functional_types_df.index[[omr_cell_id]])

            print(f'matching omr_photostim_cell_id_lst: {omr_photostim_cell_id_lst}')
            print(f'bad indices: {bad_indices}')

            if plot_stim_sites: # plot matching OMR and Stim cells from each dataset
                try:
                    plt.figure(figsize=(6, 6))
                    plt.imshow(stim_photostimFish.rescaled_ref, cmap='gray',
                               vmax=np.percentile(stim_photostimFish.rescaled_ref, 99))
                    plt.title(f'{plane}')
                    for n in range(len(stimmed_cell_id_array)):
                        if n not in bad_indices:
                            print('white is OMR cell, red is stim id cell')
                            plt.scatter(sub_functional_types_df.neur_coords.loc[omr_photostim_cell_id_lst[n]][0],
                                        sub_functional_types_df.neur_coords.loc[omr_photostim_cell_id_lst[n]][1],
                                        color='white', s=10)
                            plt.annotate(omr_photostim_cell_id_lst[n],
                                         (sub_functional_types_df.neur_coords.loc[omr_photostim_cell_id_lst[n]][0],
                                          sub_functional_types_df.neur_coords.loc[omr_photostim_cell_id_lst[n]][1]),
                                         color='white')
                            plt.scatter(stim_photostimFish.stimmed_cell_coords[n][0],
                                        stim_photostimFish.stimmed_cell_coords[n][1], color='red', s=10)
                            plt.annotate(stimmed_cell_id_array[n], (stim_photostimFish.stimmed_cell_coords[n][0],
                                                                    stim_photostimFish.stimmed_cell_coords[n][1]),
                                         color='red')
                    plt.show()
                except:
                    print('cannot plot for some reason')
        else:
            pass
        df_lst.append(sub_functional_types_df)

    functional_types_df = pd.concat(df_lst).reset_index(drop=True)
    try:
        _final_functional_types_df = photostim_data_pipeline.reorder_and_assign_resp_ids(functional_types_df)
    except Exception as e:
        print(e)
        _final_functional_types_df = functional_types_df
    # clear anywhere that there is not any stim neur id matches (can't get photostim response data)
    final_functional_types_df = _final_functional_types_df[_final_functional_types_df['stim_neur_id'] != "None"]
    final_functional_types_df = final_functional_types_df.reset_index(drop=True)

    return final_functional_types_df


def update_regions_in_df_post_alignment(df, master_folder_path):
    """
    Updates the 'region' column of df based on polygon membership.
    Run this post alignment, rois that you will gather are Pt, nMLF, aHB
    master_folder_path: directory containing subfolders for each plane (plane_0, plane_1, ...) (ends with output_folders)
    """
    from shapely.geometry import Point, Polygon

    updated_regions = []

    # Cache polygons by plane
    polygon_dict = {}
    for plane_folder in os.listdir(master_folder_path):
        if not plane_folder.startswith("plane_"):
            continue
        plane_roi_path = os.path.join(master_folder_path, plane_folder, 'rois')
        polygons = {}
        for f in os.listdir(plane_roi_path):
            if f.endswith(".npy"):
                region_name = os.path.splitext(f)[0]
                print(region_name)
                polygon_coords = np.load(os.path.join(plane_roi_path, f))
                print(len(polygon_coords))
                polygons[region_name] = Polygon(polygon_coords)
        polygon_dict[plane_folder] = polygons

    # Iterate over rows and assign regions
    for _, row in df.iterrows():
        plane = row["plane"]
        coord = row["neur_coords"]
        if not isinstance(coord, (list, tuple, np.ndarray)) or len(coord) != 2:
            updated_regions.append(np.nan)
            continue

        point = Point(coord[0], coord[1])
        polygons = polygon_dict.get(plane, {})
        assigned_region = np.nan

        for region_name, poly in polygons.items():
            if poly.contains(point):
                assigned_region = region_name
                break  # if you want one region per cell

        updated_regions.append(assigned_region)

    df["region"] = updated_regions
    return df

def gather_ensemble_activity_subsets(stim_fishvolume, master_stim_sites_df, master_stim_ensemble_df, evoked_window=6):
    # 1 - gather all the stimmed traces that is in the master stim sites
    stimmed_cell_info_dict = BCDict()
    for i, cell_id in enumerate(master_stim_sites_df.cell_ids.values):
        fishy = stim_fishvolume.volumes[master_stim_sites_df.iloc[i].plane]
        stimmed_cell_id = fishy.stimmed_cells_matched_stim_ids_dict[cell_id][0]
        if not isinstance(stimmed_cell_id, float): #if there is a match in cell ids (not nan vals)
            # add to the dictionary
            if cell_id not in stimmed_cell_info_dict.keys():
                stimmed_cell_info_dict[cell_id] = {}

            # collect traces
            norm_trace = fishy.normcells[stimmed_cell_id]
            avg_baseline = np.nanmean(fishy.normcells[stimmed_cell_id][fishy.baseline_frames - 20: fishy.baseline_frames])
            global_df_f_trace = (norm_trace - avg_baseline) / avg_baseline
            zscore_trace = fishy.zdiff_cells[stimmed_cell_id]

            # get all frames, ROI for the cell
            cell_specific_ps_events = np.array(fishy.ps_event_start)
            location = fishy.return_singlecell_rois(stimmed_cell_id)
            stimmed_cell_info_dict[cell_id] = {'norm_trace': norm_trace,
                                               'zscore_trace': zscore_trace,
                                               'global_df_f_trace': global_df_f_trace,
                                               'roi': location,
                                               'ps_frames': cell_specific_ps_events}

    # 2 - use this information to gather the ensemble traces, weights
    baseline_frames = np.arange(0, -fishy.photostim_frame_window[0])
    evoked_frames = np.arange(-fishy.photostim_frame_window[0], -fishy.photostim_frame_window[0] + evoked_window)

    ensemble_traces_dict = BCDict()
    for key in ['stim_responses', 'stim_responses_zscore', 'trial_weights']:
        if key not in ensemble_traces_dict.keys():
            ensemble_traces_dict[key] = {}

    for i, each_ensemble in enumerate(master_stim_ensemble_df.ensemble_id.values):
        if master_stim_ensemble_df[master_stim_ensemble_df.ensemble_id == each_ensemble].barcoding.values[0] != 'control_above':
            if each_ensemble not in ensemble_traces_dict['stim_responses'].keys():
                ensemble_traces_dict['stim_responses'][each_ensemble] = {}
                ensemble_traces_dict['stim_responses_zscore'][each_ensemble] = {}
            ensemble_cell_ids = master_stim_ensemble_df.iloc[i].ensemble_cells
            stim_events = master_stim_ensemble_df.iloc[i].stim_events

            # {cell: trials x frames} for both df/f and zscore activity
            for trace_type in ['norm_trace', 'zscore_trace']:
                dict_key = 'stim_responses' if trace_type == 'norm_trace' else 'stim_responses_zscore'
                sub_single_ensemble_trace_dict = {}
                for n, each_cell in enumerate(ensemble_cell_ids):
                    stim_cell_key = 'stim_' + str(each_cell)
                    if stim_cell_key not in sub_single_ensemble_trace_dict:
                        sub_single_ensemble_trace_dict[stim_cell_key] = []
                    # calculate cell trace
                    if each_cell in stimmed_cell_info_dict.keys():
                        cell_trace = stimmed_cell_info_dict[each_cell][trace_type]
                        matching_ps_frames = stimmed_cell_info_dict[each_cell]['ps_frames'][stim_events]
                        frame_subset = arrutils.subsection_arrays(matching_ps_frames, offsets=fishy.photostim_frame_window)
                        cell_trace_subsets = np.array([cell_trace[a] for a in frame_subset])
                        baseline_per_trial = np.nanmean(cell_trace_subsets[:, baseline_frames], axis=1)
                        evoked_per_trial = np.nanmean(cell_trace_subsets[:, evoked_frames], axis=1)
                        if trace_type == 'norm_trace':  # use this to find the 'local df/f'
                            cell_trace_subsets = np.array([(arr - baseline_per_trial[l]) / baseline_per_trial[l] for l, arr in
                                                       enumerate(cell_trace_subsets)])
                    sub_single_ensemble_trace_dict[stim_cell_key] = cell_trace_subsets

                ensemble_traces_dict[dict_key][each_ensemble] = sub_single_ensemble_trace_dict

    return ensemble_traces_dict

def gather_ensemble_evoked_tail_data(stim_fishvol, master_ensemble_df):
    # 1 - normalize the tail sum just to double check
    fishy = stim_fishvol[0]
    fishy.tail_df['tail_sum'] = tailtracking.normalize_tail_sum(fishy.tail_df.tail_sum.values)

    # subset out the tail data for each ensemble
    ensemble_tail_data_dict = {}
    for i, ensemble_id in enumerate(master_ensemble_df.ensemble_id.values):
        stim_events = master_ensemble_df.iloc[i].stim_events
        matching_ps_frames = np.array(fishy.ps_event_start)[stim_events]
        frame_subset = arrutils.subsection_arrays(matching_ps_frames, fishy.photostim_frame_window)
        tail_data_subsets = np.array([fishy.tail_df[fishy.tail_df.frame.isin(a)].tail_sum.values for a in frame_subset])
        tail_data_deg = np.array([np.degrees(a) for a in tail_data_subsets])
        ensemble_tail_data_dict[ensemble_id] = tail_data_deg

    return ensemble_tail_data_dict

def add_ensemble_responses_to_df(functional_types_df, master_stim_ensemble_df, data_fishvolume, photostim_frame_window = None):
    '''
    Adding the photostimulation responses for all the responding cells, actually their responses to ensembles not stimulated cells individually
    Now better to include local df/f (baseline per stim event), global df/f (baseline avg before every stim event),
    local zscore (zscoring only around the stim event), global zscore (entire trace zscored, then take the subset)
    :param functional_types_df: the master functional types dataframe that you are adding these responses to
    :param master_stim_ensemble_df: the df that has all the ensemble information (called: master_stim_ensembles.h5)
    :param data_fishvolume: the stimulated data fishvolume
    :return: the functional types df now with the photostimulation responses
    '''

    _, all_norm_trace_array, all_zscore_trace_array = photostim_data_pipeline.prepare_data_for_plotting(
        functional_types_df, data_fishvolume)
    stimulated_ensembles = master_stim_ensemble_df.ensemble_id.values
    stimulated_ensembles_stim_events = master_stim_ensemble_df.stim_events.values
    resp_cell_planes = functional_types_df.plane.values

    # 1 - raw traces in subsets for ease in analysis
    raw_traces_dict = BCDict()
    raw_norm_list = []
    zscore_global_list = []
    for resp_cell_id, responding_trace in enumerate(all_norm_trace_array):
        if resp_cell_id not in raw_traces_dict.keys():
            raw_traces_dict[resp_cell_id] = {}
        plane_fishy = data_fishvolume.volumes[resp_cell_planes[resp_cell_id]]
        resp_cell_ps_frames = plane_fishy.ps_event_start
        resp_cell_zscored_array = all_zscore_trace_array[resp_cell_id]
        resp_cell_zscored_dict = {}  # for dataframe
        resp_cell_raw_dict = {}  # for dataframe
        for e_index, e_id in enumerate(stimulated_ensembles):
            if e_id not in raw_traces_dict[resp_cell_id].keys():
                raw_traces_dict[resp_cell_id][e_id] = []
                resp_cell_zscored_dict[e_id] = []
            stim_events = stimulated_ensembles_stim_events[e_index]
            stimmed_frames = [resp_cell_ps_frames[s] for s in stim_events]  # should be the correct frames
            if photostim_frame_window is None:
                photostim_frame_window = plane_fishy.photostim_frame_window
            frame_subset = arrutils.subsection_arrays(stimmed_frames, photostim_frame_window)
            if frame_subset[-1][-1] > len(responding_trace):  # adjust frames in case this is out of range
                new_frame_subset = []
                for s in frame_subset:
                    new_frame_subset.append([q for q in s if q < len(responding_trace)])
                frame_subset = np.array(new_frame_subset)
            resp_raw_trial = np.array([responding_trace[g] for g in frame_subset if len(responding_trace[g]) > 0])
            raw_traces_dict[resp_cell_id][e_id] = resp_raw_trial  # shape = trials x frames
            resp_cell_raw_dict[e_id] = [resp_raw_trial]
            resp_raw_zscore = np.array(
                [resp_cell_zscored_array[g] for g in frame_subset if len(resp_cell_zscored_array[g]) > 0])
            resp_cell_zscored_dict[e_id] = [resp_raw_zscore]
        raw_norm_list.append(resp_cell_raw_dict)
        zscore_global_list.append(resp_cell_zscored_dict)  # entire trace zscored before looking at subset

    # 2 - calculate local df/f, global df/f, zscore
    dff_local_list = []
    dff_global_list = []
    zscore_local_list = []
    for each_cell, trace_subset_dict in raw_traces_dict.items():
        dff_local_resp_to_each_ensemble = {}
        dff_global_resp_to_each_ensemble = {}
        zscore_local_resp_to_each_ensemble = {}
        zscore_global_resp_to_each_ensemble = {}
        for ensemble_name, array in trace_subset_dict.items():
            local_dff_resp_traces = np.zeros(shape=(len(array), len(array[0])))
            local_zscore_resp_traces = np.zeros(shape=(len(array), len(array[0])))
            avg_baseline_per_trial = []
            for a, arr in enumerate(array):
                base_e = arr[:-photostim_frame_window[0]]  # anything before the photostim event
                avg_baseline_per_trial.append(np.nanmean(base_e))
                trace_e = (arr - np.nanmean(base_e)) / np.nanmean(base_e)
                local_dff_resp_traces[a] = trace_e  # full trace for each trial
                local_zscore_resp_traces[a] = scipy.stats.zscore(trace_e)
            dff_local_resp_to_each_ensemble[ensemble_name] = [local_dff_resp_traces]
            zscore_local_resp_to_each_ensemble[ensemble_name] = [local_zscore_resp_traces]

            global_dff_resp_traces = np.zeros(shape=(len(array), len(array[0])))
            global_avg_baseline = np.nanmean(avg_baseline_per_trial)
            for a, arr in enumerate(array):
                global_dff = (arr - global_avg_baseline) / global_avg_baseline
                global_dff_resp_traces[a] = global_dff
            dff_global_resp_to_each_ensemble[ensemble_name] = [global_dff_resp_traces]

        dff_local_list.append(dff_local_resp_to_each_ensemble)
        dff_global_list.append(dff_global_resp_to_each_ensemble)
        zscore_local_list.append(zscore_local_resp_to_each_ensemble)

    # 3 - add these new columns to the dataframe
    functional_types_df['stim_responses_norm_raw'] = raw_norm_list
    functional_types_df['stim_responses_dff_local'] = dff_local_list
    functional_types_df['stim_responses_dff_global'] = dff_global_list
    functional_types_df['stim_responses_zscore_local'] = zscore_local_list
    functional_types_df['stim_responses_zscore_global'] = zscore_global_list
    if 'stim_responses' in functional_types_df.columns:
        functional_types_df = functional_types_df.drop(['stim_responses', 'avg_evoked_df_f', 'stim_responses_zscore', 'avg_evoked_zscore'], axis=1)

    return functional_types_df


def min_distance_from_ensemble_um(stim_spots_px, responder_px, xy_um_per_px, plane_spacing_um=7.0, use_z=True):
    """
    Find the minimum distance of a cell from an ensemble
    stim_spots_px: list of (x_px, y_px, plane_index)
    responder_px:   (x_px, y_px, plane_index)
    xy_um_per_px:   float, µm per pixel in XY
    plane_spacing_um: float, µm between planes (default 7)
    use_z: whether to include z-distance
    returns: closest_val: the minimum distance from the ensemble
    """
    distances = []
    rx, ry, rp = responder_px
    for s in stim_spots_px:
        sx, sy, sp = s
        dx_um = (sx - rx) * xy_um_per_px
        dy_um = (sy - ry) * xy_um_per_px
        dz_um = (sp - rp) * plane_spacing_um if use_z else 0.0
        d = math.sqrt(dx_um*dx_um + dy_um*dy_um + dz_um*dz_um)
        distances.append((s, d))
    closest, closest_val = min(distances, key=lambda x: x[1])

    return closest_val

# gather activated and suppressed neurons for an ensemble
def get_ps_responders_dataframe(ensemble_id, huge_df,
                                photostim_event_frame=8, evoked_response_window=8,
                                activated_std_threshold=3, suppressed_std_threshold=2,
                                amplitude_threshold = 0,
                                type_of_traces = 'stim_responses_dff_local',
                                no_stimmmed_cells = True,
                                ensemble_weights = [] ):
    '''
    Gather responders that are activated and suppressed to an ensemble
    Returns a subset of the big functional types dataframe to gather other features of the responder neurons
    :param ensemble_id: ensemble id (i.e. 'A')
    :param huge_df: the large functional types dataframe
    :param photostim_event_frame: at what frame does the stimuluation start
    :param evoked_response_window: the response window to check out post photostimulation for looking at evoked activity
    :param activated_std_threshold: standard deviation threshold for activated neurons
    :param suppressed_std_threshold:standard deviation threshold for suppressed neurons (negative)
    :param amplitude_threshold: amplitude threshold for stimulation responses (must have an evoked activity larger or smaller than this number for associated responders, default is 0 (no amplitude threshold)
    :param type_of_traces: the column of traces for finding responders
    :param no_stimmmed_cells: if True, no photostimmed cells are included
    :return: responders dataframe and the evoked activity per cell (in order of the responders dataframe rows)
    '''

    inds_of_df = []
    evoked_dff = []

    if no_stimmmed_cells == True:
        subset_df = huge_df[huge_df.resp_cell_id.str.contains('resp')]
    else:
        subset_df = huge_df

    for each_neur, each_neuron_resp in enumerate(subset_df[type_of_traces].values):
        responses_to_ensemble = each_neuron_resp[ensemble_id][0]
        baseline_frames = responses_to_ensemble[:, :photostim_event_frame - 1]  # trials x time
        mean_baseline_activity_per_trial = np.nanmean(baseline_frames, axis=1)  # avg baseline activity per trial
        evoked_frames = responses_to_ensemble[:,
                        photostim_event_frame: photostim_event_frame + evoked_response_window]
        mean_evoked_activity_per_trial = np.nanmean(evoked_frames, axis=1)  # avg evoked dff activity per trial

        if len(ensemble_weights) > 1:
            avg_evoked_activity = np.average(mean_evoked_activity_per_trial, weights=ensemble_weights)
            avg_baseline_activity = np.average(mean_baseline_activity_per_trial, weights=ensemble_weights)
            # Compute the weighted variance of baseline means (not of within-trial stds)
            weighted_var_baseline = np.average((mean_baseline_activity_per_trial - avg_baseline_activity) ** 2, weights=ensemble_weights)
            std_across_trials = np.sqrt(weighted_var_baseline)
        else:
            avg_evoked_activity = np.nanmean(mean_evoked_activity_per_trial)
            avg_baseline_activity = np.nanmean(mean_baseline_activity_per_trial)
            std_across_trials = np.nanstd(mean_baseline_activity_per_trial)

        if (avg_evoked_activity >= (avg_baseline_activity + (activated_std_threshold * std_across_trials))) & (avg_evoked_activity >= amplitude_threshold):
            inds_of_df.append(each_neur)
            evoked_dff.append(avg_evoked_activity)
        if (avg_evoked_activity <= (avg_baseline_activity - (suppressed_std_threshold * std_across_trials))) & (avg_evoked_activity <= -amplitude_threshold) :
            inds_of_df.append(each_neur)
            evoked_dff.append(avg_evoked_activity)

    responder_df = subset_df.iloc[inds_of_df]
    evoked_dff = np.array(evoked_dff)

    return responder_df, evoked_dff


def plot_ps_connectivity_map(ax, stim_locations, responder_locations, responder_vals, responder_tuning_clrs,
                             limits=[-3, 6]):
    '''
    plotting the connectivity with arrows between stim cells and responders
    :param ax:
    :param stim_locations:
    :param responder_locations:
    :param responder_vals:
    :param responder_tuning_clrs:
    :param limits:
    :return:
    '''
    ax.set_ylim(512 - 20, 20)
    ax.set_xlim(20, 512 - 20)
    for s in stim_locations:
        ax.scatter(s[0], s[1], color='limegreen', s=25, zorder=4)

    responder_vals = np.array(responder_vals)

    x_coords = [r[0] for r in responder_locations]
    y_coords = [r[1] for r in responder_locations]
    ax.scatter(x_coords, y_coords, s=25, color=responder_tuning_clrs, zorder=4)

    norm = matplotlib.colors.TwoSlopeNorm(vmin=limits[0], vcenter=0, vmax=limits[1])
    cmap = plt.get_cmap('bwr')
    connectivity_colors = cmap(norm(responder_vals))

    for each_resp in range(len(responder_locations)):
        for each_stim in stim_locations:
            ax.annotate('', xy=(x_coords[each_resp] - 1, y_coords[each_resp] - 1), xytext=(each_stim[0], each_stim[1]),
                        arrowprops=dict(facecolor='red', edgecolor=connectivity_colors[each_resp], arrowstyle='->',
                                        linewidth=1), zorder=2)


def plot_directional_tuning_with_ps_activity(ax, ensemble_tuning_angle, ensemble_tuning_strength,
                                             responding_tuning_angle, responding_tuning_strength,
                                             responding_tuning_colors, responding_evoked_dff, ):
    '''
    plotting the directional tuning and strength of responder neurons, transparency indicates how well the cell was responding
    :param ax:
    :param ensemble_tuning_angle:
    :param ensemble_tuning_strength:
    :param responding_tuning_angle:
    :param responding_tuning_strength:
    :param responding_tuning_colors:
    :param responding_evoked_dff:
    :return:
    '''
    ensemble_tuning_angle = [math.radians(a) for a in ensemble_tuning_angle]
    responding_tuning_angle = [math.radians(a) for a in responding_tuning_angle]
    alpha_val = [(v - min(responding_evoked_dff)) / (max(responding_evoked_dff) - min(responding_evoked_dff)) for v in
                 responding_evoked_dff]
    alpha_val = np.clip(alpha_val, 0, 1)
    if np.isnan(alpha_val).any():
        alpha_val = np.clip(responding_evoked_dff, 0, 1)

    for ang, strg in zip(ensemble_tuning_angle, ensemble_tuning_strength):
        ax.scatter(ang, strg, color='red', alpha=0.7, s=40, zorder=10)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    for r in range(len(responding_tuning_angle)):
        ax.scatter(responding_tuning_angle[r], responding_tuning_strength[r],
                   color=responding_tuning_colors[r], alpha=alpha_val[r], s=40, zorder=5)

    # mark circular mean
    mean_angle = scipy.stats.circmean(responding_tuning_angle, high=2 * np.pi, low=0)
    mean_strength = np.mean(responding_tuning_strength)
    ax.plot([mean_angle, mean_angle], [0, mean_strength], color='black', lw=2, linestyle='-', zorder=15)

    ax.set_rmin(0.0)

















# 10/20/25 - OLD BUT KEEPING IN CASE

# def add_ensemble_responses_to_df(functional_types_df, master_stim_ensemble_df, data_fishvolume, evoked_frame_window=8):
#     '''
#     Adding the photostimulation responses for all the responding cells but actually their responses to ensembles not stimulated cells individually
#     :param functional_types_df: the master functional types dataframe that you are adding these responses to
#     :param master_stim_ensemble_df: the df that has all the ensemble information (called: master_stim_ensembles.h5)
#     :param data_fishvolume: the stimulated data fishvolume
#     :param evoked_frame_window: the number of frames post photostimulation event that you will take the avg activity over
#     :return: the functional types df now with the photostimulation responses
#     '''
#     resp_f_trace_array, resp_norm_trace_array, resp_zscore_trace_array = photostim_data_pipeline.prepare_data_for_plotting(
#         functional_types_df, data_fishvolume)
#     stimulated_ensembles = master_stim_ensemble_df.ensemble_id.values
#     stimulated_ensembles_stim_events = master_stim_ensemble_df.stim_events.values
#     resp_cell_planes = functional_types_df.plane.values
#
#     for t, traces in enumerate([resp_norm_trace_array, resp_zscore_trace_array]):
#         trace_type = 'df/f' if t == 0 else 'zscore'
#         photostimulation_responses_lst = []
#         avg_evoked_activity_master_lst = []
#
#         # for each responding cell, get the responses of that cell to each stimulation event
#         for resp_cell_id, responding_trace in enumerate(traces):
#
#             # need to index into the correct plane for that cell to gather the right ps event frames
#             plane_fishy = data_fishvolume.volumes[resp_cell_planes[resp_cell_id]]
#             resp_cell_ps_frames = plane_fishy.ps_event_start
#
#             resp_to_each_ensemble_dict = BCDict()
#             avg_evoked_activity_lst = []
#             for e_index, e_id in enumerate(stimulated_ensembles):
#                 if e_id not in resp_to_each_ensemble_dict.keys():
#                     resp_to_each_ensemble_dict[e_id] = {}
#                 stim_events = stimulated_ensembles_stim_events[e_index]
#                 stimmed_frames = [resp_cell_ps_frames[s] for s in stim_events]  # should be the correct frames
#                 frame_subset = arrutils.subsection_arrays(stimmed_frames, plane_fishy.photostim_frame_window)
#
#                 if frame_subset[-1][-1] > len(responding_trace):  # adjust frames in case this is out of range
#                     new_frame_subset = []
#                     for s in frame_subset:
#                         new_frame_subset.append([q for q in s if q < len(responding_trace)])
#                     frame_subset = np.array(new_frame_subset)
#
#                 resp_raw_trial = np.array([responding_trace[g] for g in frame_subset if len(responding_trace[g]) > 0])
#                 resp_df_f_trial = np.zeros(shape=(len(resp_raw_trial), len(resp_raw_trial[0])))
#                 resp_raw_trial2 = np.zeros(shape=(len(resp_raw_trial), len(resp_raw_trial[0])))
#                 stim_evoked_df_f_trial = np.zeros(shape=(len(resp_raw_trial), 1))
#                 stim_evoked_raw_trial = np.zeros(shape=(len(resp_raw_trial), 1))
#
#                 for d, f in enumerate(resp_raw_trial):
#                     if trace_type == 'df/f':
#                         base_e = f[:-plane_fishy.photostim_frame_window[0]]  # anything before the photostim event
#                         plot_e = (f - np.nanmean(base_e)) / np.nanmean(base_e)
#                         evoked_e = np.nanmean(plot_e[-plane_fishy.photostim_frame_window[0]:-
#                                                                                             plane_fishy.photostim_frame_window[
#                                                                                                 0] + evoked_frame_window])  # evoked df f for each trial
#                         stim_evoked_df_f_trial[d] = evoked_e
#                         resp_df_f_trial[d] = plot_e  # full trace for each trial
#                     if trace_type == 'zscore':
#                         # calculate raw traces of your input array
#                         evoked_raw = np.nanmean(f[-plane_fishy.photostim_frame_window[0]:-
#                                                                                          plane_fishy.photostim_frame_window[
#                                                                                              0] + evoked_frame_window])  # evoked raw trace for each trial
#                         stim_evoked_raw_trial[d] = evoked_raw
#                         resp_raw_trial2[d] = f  # full trace for each trial
#
#                 if trace_type == 'df/f':
#                     resp_to_each_ensemble_dict[e_id] = [resp_df_f_trial]
#                     avg_evoked_activity_lst.append(np.nanmean(stim_evoked_df_f_trial))
#                 if trace_type == 'zscore':  # only want raw zscore traces, no need to normalize to baseline
#                     resp_to_each_ensemble_dict[e_id] = [resp_raw_trial2]
#                     avg_evoked_activity_lst.append(np.nanmean(stim_evoked_raw_trial))
#
#             # put all of this information into lists to add to the final dataframe
#             photostimulation_responses_lst.append(resp_to_each_ensemble_dict)
#             avg_evoked_activity_master_lst.append(avg_evoked_activity_lst)
#
#             if trace_type == 'df/f':
#                 df_f_photostim_responses_lst = photostimulation_responses_lst
#                 df_f_avg_evoked_activity_lst = avg_evoked_activity_master_lst
#             else:
#                 zscore_photostim_responses_lst = photostimulation_responses_lst
#                 zscore_avg_evoked_activity_lst = avg_evoked_activity_master_lst
#
#     functional_types_df['stim_responses'] = df_f_photostim_responses_lst
#     functional_types_df['avg_evoked_df_f'] = df_f_avg_evoked_activity_lst
#     functional_types_df['stim_responses_zscore'] = zscore_photostim_responses_lst
#     functional_types_df['avg_evoked_zscore'] = zscore_avg_evoked_activity_lst
#
#     return functional_types_df