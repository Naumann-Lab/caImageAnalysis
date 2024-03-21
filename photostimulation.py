# functions to preprocess photostimulation data 

import os
from pathlib import Path
import pandas as pd
import numpy as np
from tifffile import imread, imwrite
from datetime import datetime as dt, timedelta
import xml.etree.ElementTree as ET
import caiman as cm
from PIL import Image
import math
from scipy.signal import find_peaks 

from bruker_images import read_xml_to_str, read_xml_to_root
from utilities import arrutils
from utilities.roiutils import create_circular_mask
from utilities.coordutils import rotate_transform_coors, closest_coordinates

def find_no_baseline_frames(somefishclass, no_planes = 0):
    '''
    folder_path = path that contains all the data (xml file and original bruker images)
    volume = set to True if this is a volume stack since the baseline frame number is calculated differently
    '''
    if no_planes < 2:
        original_imgs = Path(somefishclass.folder_path).parents[1].joinpath("bruker_images")
        with os.scandir(original_imgs) as entries:
            for entry in entries:
                if 'Cycle' in entry.name and 'tif' in entry.name:
                    baseline_img = imread(entry.path)
                    break #we want the first tif file here in single image

        somefishclass.baseline_frames = baseline_img.shape[0]

    elif no_planes >= 2:
        ps_xml_name = Path(somefishclass.data_paths['ps_xml']).name
        somefishclass.baseline_frames = int(ps_xml_name.split('Cycle')[1].split('_')[0]) # baseline frames number is given in the cycle name for the volume
    
    return somefishclass.baseline_frames

def collect_stimulation_times(somefishclass):
    '''
    Calculating the stimulation times from either the voltage recording output (channel input 2)
    if not voltage recording, then can finid this based on the mark point xml file (not as exact)
    Returns the specific times in ms for each stimulation based on the start of the T-series
    '''

     # collecting stimulation timing based on ms in the xml file
    data = read_xml_to_str(somefishclass.data_paths['ps_xml'])
    for i in data.split("\n"):
        if "InitialDelay" in i:
            try:
                initial_delay_ms = int([i][0].split("InitialDelay=")[1].split('"')[1].split('.')[1]) # weird format in the xml file for this value, should not be a decimal
            except:
                initial_delay_ms = int([i][0].split("InitialDelay=")[1].split('"')[1])*10 # should be times 10 for ms
            interpointdelay_ms = int([i][0].split("InterPointDelay=")[1].split('"')[1])
            duration_ms = float([i][0].split("Duration=")[1].split('"')[1])
        elif "Repetitions" in i:
            no_repetitions = int([i][0].split("Repetitions=")[1].split('"')[1])
        elif "Iterations" in i:
            no_iterations = int([i][0].split("Iterations=")[1].split('"')[1])
            iteration_delay_ms = int(float([i][0].split("IterationDelay=")[1].split('"')[1]))

    full_duration_per_stim = initial_delay_ms + (no_repetitions * duration_ms) + ((no_repetitions-1) * interpointdelay_ms)

    # if there is a voltage recording, can gather start signals from there
    if "voltage_signal" in somefishclass.data_paths.keys():
        volt_csv = pd.read_csv(somefishclass.data_paths["voltage_signal"])
        monaco_signal = np.array(volt_csv[' monaco'])
        time = np.array(volt_csv['Time(ms)'])

        peaks, _ = find_peaks(monaco_signal, height = 0.10) # find peaks in voltage trace that are above 0.10 volts
        peak_starts = [peaks[i] for i in range(len(peaks)) if i == 0 or peaks[i] - peaks[i-1] > 100] # find only the start of each peak, each rep
        
        # grabbing the start of each TRIAL, so have to take into account the repetition number
        trial_starts = peak_starts[::no_repetitions]

        stim_times = [time[i] for i in trial_starts] # convert the peak start indices to the time in ms
    
    else: # if no voltage recording, then calculate from mark points xml file
        stim_times = [(full_duration_per_stim/1000)*m + (iteration_delay_ms/1000)*m for m in range(no_iterations)]


    return full_duration_per_stim, stim_times

def save_badframes_arr(somefishclass, no_planes = 1):

    somefishclass.baseline_frames = find_no_baseline_frames(somefishclass, no_planes)

    full_duration_per_stim, stim_times = collect_stimulation_times(somefishclass)
    stim_times_secs = [x/1000 for x in stim_times] # needs to be in seconds for comparing with the relative times in the xml file

    # using the information xml file to calculate the frames and times for each stimulation
    if no_planes > 1: # if volume stimulation
        plane_num = int(somefishclass.data_paths['move_corrected_image'].parents[0].name.split('_')[1])

        frametimes = []
        info_data = read_xml_to_str(somefishclass.data_paths["info_xml"])

        for i in info_data.split("\n"):
            if "relativeTime" in i:
                relative_time = [i.split("relativeTime=")[1].split('"')[1]][0]
                frametimes.append(float(relative_time))
        
        # find the second occurence where the stimulation starts in the list of frametimes
        # presuming that you are doing 2 t-series cycles:
        # the first occurence is the baseline recording, second is the markpoints start, then third value is the frame
        index = 1
        if "voltage_signal" in somefishclass.data_paths.keys():
            index = 2
        stim_ind = [index for index, value in enumerate(frametimes) if value < 0.01][index] 

        # only get the relative frametimes that happen during the stimulation
        stimulation_frametimes = frametimes[stim_ind:]

        for p in range(no_planes):
            if p == plane_num:
                plane_frametimes = stimulation_frametimes[p::no_planes]
                time_matches = [min(plane_frametimes, key=lambda y: abs(x - y)) for x in stim_times_secs] # list of frametimes values that match with the stim_times
                frames = [plane_frametimes.index(x) for x in time_matches] # list of frames that match with the stim_times

    elif no_planes == 1: # if single plane stimulation
        ##TODO: make it simpler code, combine this to just do what I am doing for volumes
        # only looking at stimulation cycle here for the right relative times
        root = read_xml_to_root(somefishclass.data_paths["info_xml"])
        frametimes = []
        for child in root:
            if child.tag == 'Sequence' and child.attrib['cycle'] == '2':
                for subchild in child:
                    if subchild.tag == 'Frame':
                        frametimes.append(float(subchild.attrib['relativeTime']))

        # calculate ps events in frame numbers
        time_matches = [min(frametimes, key=lambda y: abs(x - y)) for x in stim_times] # list of frametimes values that match with the stim_times
        frames = [frametimes.index(x) for x in time_matches] # list of frames that match with the stim_times
        
        # calculate the duration of one ps event in frame numbers
        frame_dur = min(frametimes, key=lambda y: abs(full_duration_per_stim/1000 - y))
        duration_in_frames = frametimes.index(frame_dur)
        
    ps_events = [somefishclass.baseline_frames + f for f in frames]
    somefishclass.badframes_arr = np.array(ps_events)
    # saving the bad frames array
    save_path = Path(somefishclass.data_paths['move_corrected_image']).parents[0].joinpath('bad_frames.npy')
    if Path(save_path).exists():
        somefishclass.badframes_arr = np.array(np.load(save_path)) # load in badframes
        print('load in bad frames array')
    else:
        np.save(save_path, somefishclass.badframes_arr)  # save badframes
        print('saved bad frames array')
   
    return somefishclass.badframes_arr

def identify_stim_sites(somebasefish, rotate = True, planes_stimed = [1,2,3,4]):
    '''
    planes_stimed is hard coded, not sure how to gather the z plane info with not a clear output file 
    use a base fish, saves a stimulated site dataframe for each unique plane
    '''
    somebasefish.stim_sites_df = pd.DataFrame(columns = ['plane', 'x_stim', 'y_stim', 'sp_size'])

    # use the info xml file to get the pixel data
    pixel_info = read_xml_to_str(somebasefish.data_paths['info_xml'])
    for i in (pixel_info.split("\n")):
        if "pixelsPerLine" in i:
            pixels_per_line = int(i.split('value=')[1].split('"')[1])
        if "linesPerFrame" in i:
            lines_per_frame = int(i.split('value=')[1].split('"')[1])
    
    # use the ps xml file to get the stim site data
    ps_xml = read_xml_to_str(somebasefish.data_paths['ps_xml'])
    X_stim_sites = []
    Y_stim_sites = []
    spiral_size_lst = []
    for r in range(ps_xml.count("Point Index=") + 1):
        for i in ps_xml.split("\n"):
            if f'Point Index="{r}"' in i:
                X_stim = float(i.split('X')[1].split('"')[1])*pixels_per_line
                X_stim_sites.append(round(X_stim))

                Y_stim = float(i.split('Y')[1].split('"')[1])*lines_per_frame
                Y_stim_sites.append(round(Y_stim))

                spiral_size = float(i.split('SpiralSizeInMicrons')[1].split('"')[1])
                spiral_size_lst.append(round(spiral_size))

    # need to rotate and transform the coordinates if the image is rotated from off the Bruker
    if rotate:
        coord_stim_sites = list(zip(X_stim_sites, Y_stim_sites))

        ##TODO: make this cleaner to find the correct y and x coords, i should not have to do this separately
        correct_y_coords = rotate_transform_coors(coord_stim_sites, 90, translation=(pixels_per_line, 0))
        correct_x_coords = rotate_transform_coors(coord_stim_sites, -90, translation=(0, pixels_per_line))

        X_stim_sites = [x[0] for x in correct_x_coords]
        Y_stim_sites = [-y[1] + pixels_per_line for y in correct_y_coords]

    if len(planes_stimed) > 1:
        # get values for the z steps in the info env file
        with open(somebasefish.data_paths['info_env'], "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "PVMarkPoints" in line and "active" in line:
                    ind_start = i+2
                if "PVGalvoPointGroup" in line:
                    ind_end = i
            ind_lines = np.arange(ind_start, ind_end, step=1)
            z_vals = [float(lines[i].split('Z')[1].split('"')[1]) for i in ind_lines]
        
        # convert z values into planes
        unique_z = np.unique(z_vals)
        map_z = {}
        for _p, p in enumerate(unique_z):
            map_z[p] = planes_stimed[_p]

        z_to_plane = [map_z[z] for z in z_vals]

        this_plane = int(somebasefish.folder_path.name.split('_')[1])
    else:
        z_to_plane = 0 # just the plane that you recorded from
        this_plane = 0

    somebasefish.stim_sites_df['x_stim'] = X_stim_sites
    somebasefish.stim_sites_df['y_stim'] = Y_stim_sites
    somebasefish.stim_sites_df['sp_size'] = spiral_size_lst
    somebasefish.stim_sites_df['plane'] = z_to_plane

    # save a big stim sites dataframe to the stim folder
    master_save_path = Path(somebasefish.folder_path.parents[1]).joinpath('stim_sites_volume.h5')
    if not os.path.exists(master_save_path):
        somebasefish.stim_sites_df.to_hdf(master_save_path, key="volume_stim")

    # trim dataframe to only include stim sites for that precise z plane to keep into that folder
    somebasefish.stim_sites_df = somebasefish.stim_sites_df[somebasefish.stim_sites_df['plane'] == this_plane]
    somebasefish.stim_sites_df.reset_index(inplace = True, drop = True)

    save_path = Path(somebasefish.folder_path).joinpath("stim_sites.hdf")
    somebasefish.stim_sites_df.to_hdf(save_path, key="stim")

    return somebasefish.stim_sites_df

def run_suite2p_PS(somebasefish, input_tau = 1.5, move_corr = False):
    '''
    somebasefish = the data you want to have suite2p run on
    input_tau = decay value for gcamp indicator (6s = 1.5, m = 1.0, f = 0.7)
    move_corr = binary, if you want the motion corrected image to be run as the main image or not
    '''
    from suite2p import run_s2p, default_ops
    from fishy import BaseFish

    if move_corr == True:
        imagepath = somebasefish.data_paths["move_corrected_image"]
    elif move_corr == False:
        imagepath = somebasefish.data_paths["rotated_image"]
    elif KeyError:
        imagepath = somebasefish.data_paths["image"]

    # make sure bad frames exists first
    bad_frames_path = imagepath.parents[0].joinpath('bad_frames.npy')
    if os.path.isfile(bad_frames_path) == False:
        save_badframes_arr(somebasefish)

    ps_s2p_ops = default_ops()
    ps_s2p_ops['data_path'] = [imagepath.parents[0].as_posix()]
    ps_s2p_ops['save_path'] = imagepath.parents[0].as_posix()
    ps_s2p_ops['tau'] = input_tau #gcamp6s = 1.5
    ps_s2p_ops['fs'] = BaseFish.hzReturner(somebasefish.frametimes_df)
    ps_s2p_ops['preclassify'] = 0.15
    ps_s2p_ops['block_size'] = [32, 32]
    ps_s2p_ops['allow_overlap'] = True
    ps_s2p_ops['tiff_list'] = [imagepath.name]
    ps_s2p_ops['two_step_registration'] = True
    ps_s2p_ops['keep_movie_raw'] = True

    db = {}
    run_s2p(ops=ps_s2p_ops, db=db)

def return_raw_coord_trace(cell_coord, img, s=5):
    '''
    returns the ROI location of a specified coordinate in the target fish
    '''
    msk = create_circular_mask(img.shape[1:], cell_coord[1], cell_coord[0], s)[:, ::-1]

    return np.nanmean(img[:, msk], axis=1)

def collect_raw_traces(somebasefish):
    
    stim_sites_df_path = Path(somebasefish.folder_path).joinpath("stim_sites.hdf")
    if not stim_sites_df_path.exists():
        identify_stim_sites(somebasefish)
    stim_sites_df = pd.read_hdf(stim_sites_df_path)
    img = imread(somebasefish.data_paths["move_corrected_image"])

    raw_traces = np.zeros((len(stim_sites_df), img.shape[0]))
    points = np.zeros((len(stim_sites_df), 2))
    for point in range(len(stim_sites_df)):
        pt = stim_sites_df.iloc[point]
        msk = create_circular_mask(img.shape[1:], pt.x_stim, pt.y_stim, pt.sp_size*3)
        msk_trace = np.nanmean(img[:, msk], axis=1)
        raw_traces[point] = msk_trace
        points[point] = [pt.x_stim, pt.y_stim]

    # save the raw traces   
    np.save(Path(somebasefish.folder_path).joinpath('raw_traces.npy'), raw_traces)

    return raw_traces, points

def all_stimmed_traces_array(stimulated_fishvolume):
    '''
    make an array of all the stimulated traces in the whole volume
    '''
    stim_traces_lst = []
    for v in stimulated_fishvolume:
        saved_raw_traces = Path(v.folder_path).joinpath('raw_traces.npy') 
        if not saved_raw_traces.exists():
            collect_raw_traces(v)

        loaded_raw_traces = np.load(saved_raw_traces)
        if loaded_raw_traces.shape[0] != 0: #don't include any planes that have no stim traces
            stim_traces_lst.append(np.load(saved_raw_traces))

    stim_traces_array = np.concatenate(stim_traces_lst, axis=0)

    return stim_traces_array

def identify_stimmed_planes(omr_tseries_folder_path, clst_label):
    '''
    folder_path = where the cluster df is located, will be the omr tseries folder path
    clst_label = the label of the stimulated cluster in this dataset (could be a cluster number or barcode label)
    returns the unique planes that were stimulated in the experiment
    '''
    if Path(omr_tseries_folder_path).joinpath("clusters.h5").exists():
        df = pd.read_hdf(Path(omr_tseries_folder_path).joinpath("clusters.h5"))
        one_category_df = df[df['cluster'] == clst_label]
    elif Path(omr_tseries_folder_path).joinpath("volume_barcoding_df.h5").exists():
        df = pd.read_hdf(Path(omr_tseries_folder_path).joinpath("volume_barcoding_df.h5"))
        one_category_df = df[df['barcoding'] == clst_label]

    all_stimmed_planes = []
    for v in one_category_df.plane.unique():
       all_stimmed_planes.append(int(v.split('_')[1]))

    # stimmed_planes = all_stimmed_planes.unique()
    stimmed_planes = sorted(all_stimmed_planes)

    return stimmed_planes

def correlations_with_stim_sites(somebasefish, traces_array = None, corr_threshold = 0.5, normalizing = 1, saving = True):
    '''
    for each cell, find the corrleation coefficients for each stim site, withput including the baseline period here
    traces_array = the array of traces that you are using to calculate the correlation coefficients, if you want this to be for a volume, input the array
    '''
    somebasefish.baseline_frames = find_no_baseline_frames(somebasefish, no_planes = 6)
    somebasefish.load_suite2p()

    somebasefish.stim_sites_df = pd.read_hdf(Path(somebasefish.folder_path).joinpath("stim_sites.hdf"))

    if traces_array is None:
        # load in raw pixel traces or run the function again/save the npy file if not
        if Path(somebasefish.folder_path).joinpath('raw_traces.npy').exists():
            traces_array = np.load(Path(somebasefish.folder_path).joinpath('raw_traces.npy'))
        else:
            traces_array, points = collect_raw_traces(somebasefish)
            np.save(Path(somebasefish.folder_path).joinpath('raw_traces.npy'), traces_array)

    somebasefish.normcells = arrutils.norm_0to1(somebasefish.f_cells)

    corr_dictionary = {}
    # for each cell, find the corrleation coefficients for each stim site, not including the baseline period here
    for cell_id, cell_trace in enumerate(somebasefish.normcells):
        if cell_id not in corr_dictionary.keys():
            corr_dictionary[cell_id] = {}  
        
        corrs = []
        cell_trace = cell_trace[somebasefish.baseline_frames:] # trim the cell trace to not include baseline
        for ind in traces_array:
            ind = ind[somebasefish.baseline_frames:] # trim the raw pixel trace to not include baseline
            corrs.append(round(np.corrcoef(ind, cell_trace)[0][1], 3))
        corr_dictionary[cell_id] = corrs
    
    corr_df = pd.DataFrame.from_dict(corr_dictionary, orient = 'index')

    avg_corr_lst = []
    for i in range(len(corr_df)):
        value = (corr_df.iloc[i].mean())/normalizing # normalizing to the positive control of the stimulated group
        avg_corr_lst.append(value)

    corr_df['avg_corr'] = avg_corr_lst

    if saving:
        corr_df.to_hdf(Path(somebasefish.folder_path).joinpath('correlation_df.hdf'), key="corr")
        print('saved correlation_df')

    corr_neurons = corr_df[corr_df.avg_corr > corr_threshold].index.values

    return corr_df, corr_neurons


