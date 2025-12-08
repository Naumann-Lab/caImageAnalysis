# functions to preprocess and help process photostimulation data 

import os
from pathlib import Path
import pandas as pd
import numpy as np
from tifffile import imread, imwrite
from scipy.signal import find_peaks 
from datetime import datetime as dt

# local imports
import sys
sys.path.append(r'C:\Users\Kaitlyn\PyCharmProjects\imaging\caImageAnalysis')
from bruker_images import get_micronstopixels_scale2, get_zstep_vals, get_pixelsperline
from bruker_images import read_xml_to_str
import fishy 
import process
from utilities import arrutils, pathutils
from utilities.roiutils import create_circular_mask, draw_roi, create_polygon_mask
from utilities.coordutils import rotate_transform_coors, closest_coordinates

# getting experiment information for later functions
def find_no_baseline_frames(somefishclass):
    '''
    folder_path = path that contains all the data (xml file and original bruker images)
    volume = set to True if this is a volume stack since the baseline frame number is calculated differently
    '''
    no_planes = identify_number_planes_in_expt(somefishclass)
    if no_planes < 2: # if this is a single plane imaging file
        original_imgs = Path(somefishclass.folder_path).parents[1].joinpath("bruker_images")
        with os.scandir(original_imgs) as entries:
            for entry in entries:
                if 'Cycle' in entry.name and 'tif' in entry.name:
                    baseline_img = imread(entry.path)
                    break #we want the first tif file here in single image

        somefishclass.baseline_frames = baseline_img.shape[0]

    elif no_planes >= 2 and 'ps_xml' in somefishclass.data_paths.keys(): # if this is a volume, but there is the MarkPoints xml file
        ps_xml_name = Path(somefishclass.data_paths['ps_xml']).name
        somefishclass.baseline_frames = int(ps_xml_name.split('Cycle')[1].split('_')[0]) # baseline frames number is given in the cycle name for the volume
    
    elif no_planes >= 2 and 'ps_log' in somefishclass.data_paths.keys(): # if this is a volume, with the automated gui
        somefishclass.baseline_frames, _, _ = utils_for_save_badframes_arr(somefishclass)

    else:
        somefishclass.baseline_frames = 0 

    return somefishclass.baseline_frames

def collect_stimulation_times(somefishclass):
    '''
    Calculating the stimulation times from either the voltage recording output (channel input 2)
    if not voltage recording, then can finid this based on the mark point xml file (not as exact)
    Returns the specific times in ms for each stimulation based on the start of the T-series

    '''

    # collecting stimulation timing based on ms in the xml file
    if 'ps_xml' in somefishclass.data_paths.keys():
        data = read_xml_to_str(somefishclass.data_paths['ps_xml'])
        for i in data.split("\n"):
            if "InitialDelay" in i:
                initial_delay_ms = 0 # initial delay is not a part of the photostimulation
                # try:
                #     initial_delay_ms = int([i][0].split("InitialDelay=")[1].split('"')[1].split('.')[1]) # weird format in the xml file for this value, should not be a decimal
                # except:
                #     initial_delay_ms = int([i][0].split("InitialDelay=")[1].split('"')[1])*10 # should be times 10 for ms
                interpointdelay_ms = int(float([i][0].split("InterPointDelay=")[1].split('"')[1]))
                duration_ms = float([i][0].split("Duration=")[1].split('"')[1])
            elif "Repetitions" in i:
                no_repetitions = int([i][0].split("Repetitions=")[1].split('"')[1])
            elif "Iterations" in i:
                no_iterations = int([i][0].split("Iterations=")[1].split('"')[1])
                iteration_delay_ms = int(float([i][0].split("IterationDelay=")[1].split('"')[1]))
        full_duration_per_stim = initial_delay_ms + (no_repetitions * duration_ms) + ((no_repetitions-1) * interpointdelay_ms)
    
    if 'ps_log' in somefishclass.data_paths.keys():
        with open(somefishclass.data_paths['ps_log']) as file:
            contents = file.read()
        lines = contents.split("\n")
        stim_lines = [l for l in lines if 'Stim event' in l]

        # full duration of a stimulation event from the output file, assuming all parameters are the same for each site
        cmd = stim_lines[0].split('-MarkAllPoints')[1]
        duration_ms = int(cmd.split('Monaco 1035')[0].split(' ')[-2])
        no_repetitions = cmd.count('Monaco 1035')
        try:
            interpointdelay_ms = int(cmd.split('True')[2].split(' ')[3])  # if there are mulitple reps
        except:
            interpointdelay_ms = 0  # since there is no repetitions
        spiral_size = float(cmd.split('True')[2].split(' ')[1])
        full_duration_per_stim = (no_repetitions * duration_ms) + ((no_repetitions-1) * interpointdelay_ms)

    # if there is a voltage recording, can gather start signals from there
    if "voltage_signal" in somefishclass.data_paths.keys():
        volt_csv = pd.read_csv(somefishclass.data_paths["voltage_signal"])
        monaco_signal = np.array(volt_csv[' monaco'])
        time = np.array(volt_csv['Time(ms)'])

        peaks, _ = find_peaks(monaco_signal, height = 0.10) # find peaks in voltage trace that are above 0.10 volts
        peak_starts = [peaks[i] for i in range(len(peaks)) if i == 0 or peaks[i] - peaks[i-1] > int(full_duration_per_stim)] # find only the start of each peak, each rep
        
        # grabbing the start of each TRIAL, so have to take into account the repetition number
        # trial_starts = peak_starts[::no_repetitions]
        trial_starts = peak_starts

        stim_times = [time[i] for i in trial_starts] # convert the peak start indices to the time in ms
    
    else: # if no voltage recording, then calculate from mark points xml file
        try:
            stim_times = [(full_duration_per_stim/1000)*m + (iteration_delay_ms/1000)*m for m in range(no_iterations)]
        except:
            stim_times = []

    return full_duration_per_stim, stim_times

def identify_number_planes_in_expt(somefishclass):
    '''
    identify the number of planes in the experiment based on the data structure
    '''
    no_planes = 0
    with os.scandir(Path(somefishclass.folder_path.parents[0])) as entries:
        for entry in entries:
            if os.path.isdir(entry.path):
                no_planes += 1

    return no_planes

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
    # elif omr_tseries_folder_path == None: # no OMR data to pull this from, then find out the stimmed planes from the xml files


    all_stimmed_planes = []
    for v in one_category_df.plane.unique():
       all_stimmed_planes.append(int(v.split('_')[1]))

    # stimmed_planes = all_stimmed_planes.unique()
    stimmed_planes = sorted(all_stimmed_planes)

    return stimmed_planes

# identifying the ps events, making bad frames arrays
def save_badframes_arr(somefishclass, automated_gui = False):
    '''
    Calculate the bad frames array based on the photostimulation events
    somefishclass = the fishy class that you are working with
    automated_gui = if you are using the automated gui to process the data (this changes some of the functions to use)

    Returns the bad frames array (and is saved)
    '''
    somefishclass.process_filestructure(somefishclass.midnight_noon_keyword) # update file structure
    no_planes = identify_number_planes_in_expt(somefishclass)

    if automated_gui == False:
        somefishclass.baseline_frames = find_no_baseline_frames(somefishclass)
        full_duration_per_stim, stim_times = collect_stimulation_times(somefishclass)
    else:
        somefishclass.baseline_frames, stim_times, full_duration_per_stim = utils_for_save_badframes_arr(somefishclass)

    somefishclass.ps_event_duration = full_duration_per_stim
    print(f'full duration per stimulation is {full_duration_per_stim}')
    stim_times_secs = [x/1000 for x in stim_times] # needs to be in seconds for comparing with the relative times in the xml file

    # using the information xml file to calculate the frames and times for each stimulation
    if no_planes > 1: # if volume stimulation
        plane_num = int(somefishclass.folder_path.name.split('_')[1])
    else:
        plane_num = 0
    frametimes = []
    info_data = read_xml_to_str(somefishclass.data_paths["info_xml"])
    for i in info_data.split("\n"):
        if "relativeTime" in i:
            relative_time = [i.split("relativeTime=")[1].split('"')[1]][0]
            frametimes.append(float(relative_time))
    
    # find where photostimulation events first start in the list of frametimes (relative times to the start of the T-series)
    if somefishclass.baseline_frames == 0:
        index = 1
        stim_ind = [index for index, value in enumerate(frametimes) if value < 0.01][index]
    elif ("voltage_signal" not in somefishclass.data_paths.keys()) & (automated_gui == True):
        # use the output file to find the time that corresponds to a frame
        with open(somefishclass.data_paths['ps_log']) as file:
            contents = file.read()
        lines = contents.split("\n")
        stim_lines = [l for l in lines if 'Stim event' in l]
        # find the specific first photostimulation event on that plane, of that sequence
        output_log_frametimes = []
        for log_entry in stim_lines:
            timestamp_str = log_entry.split()[0] + " " + log_entry.split()[1]
            timestamp_dt = dt.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
            time_only = timestamp_dt.time()
            output_log_frametimes.append(time_only)
        starting_ind_in_output_log_fts = [n for n, o in enumerate(output_log_frametimes) if o > somefishclass.frametimes_df.time[0]][0]
        first_stimulation_time = output_log_frametimes[starting_ind_in_output_log_fts:][0]
        stim_ind = [n for n, o in enumerate(somefishclass.frametimes_df.time.values) if o < first_stimulation_time][-1]
    else:
        stim_ind = 0

    # only get the relative frametimes that happen during the stimulation
    stimulation_frametimes = frametimes[stim_ind:]
    for p in range(no_planes):
        if p == plane_num:
            plane_frametimes = stimulation_frametimes[p::no_planes]
            time_matches = [min(plane_frametimes, key=lambda y: abs(x - y)) for x in stim_times_secs] # list of frametimes values that match with the stim_times
            print(f'time matches are {time_matches}')
            frames = [plane_frametimes.index(x) for x in time_matches] # list of frames that match with the stim_times
            frames = [f - 1 if f > 0 else f for f in frames] # subtract 1 from the frame number to account for a slight mismatch in frames?? need to check this
            print(f'matching frames are {frames}')

    ps_events = [somefishclass.baseline_frames + f for f in frames]
    ps_events = np.unique(ps_events)
    somefishclass.badframes_arr = np.array(ps_events)

    # saving the bad frames array
    save_path = Path(somefishclass.folder_path).joinpath('bad_frames.npy')
    np.save(save_path, somefishclass.badframes_arr)  # save badframes
    print('saved bad frames array')
   
    return somefishclass.badframes_arr


def concatenate_bad_frames_arr(dict_of_sequence_paths,
                               concatenated_dataset_path,
                               duration_of_stimulation=200):
    '''
    When having the final concatenated dataset, its better to match up stimulation times based on the voltage recording rather than the output log
    This is crucial when imaging faster than ~1 hz
    :param dict_of_sequence_paths: the dictionary of keys: paths for each sequence that was collected individually
    :param concatenated_dataset_path: the final location of the concatenated dataset, so that you can save teh new bad frames and compare the concatenated frametimes df
    :param duration_of_stimulation: needs to be in ms, default is 200 for now

    '''
    from datetime import datetime, timedelta
    import scipy

    all_stimulation_times = []
    for each_sequence_key, each_sequence_path in dict_of_sequence_paths.items():
        # load in the voltage recording file
        voltage_csv_path = pathutils.pathcrawler(each_sequence_path, inset=set(), inlist=[], mykey='csv')[0]
        volt_csv_df = pd.read_csv(voltage_csv_path)
        monaco_signal = np.array(volt_csv_df[' monaco'])
        time = np.array(volt_csv_df['Time(ms)'])

        # find peaks of the photostimulation signal
        peaks, _ = scipy.signal.find_peaks(monaco_signal,
                                           height=0.10)  # find peaks in voltage trace that are above 0.10 volts
        peak_starts = [peaks[i] for i in range(len(peaks)) if
                       i == 0 or peaks[i] - peaks[i - 1] > int(duration_of_stimulation)]
        add_seconds = [time[i] / 1000 for i in peak_starts]  # convert into seconds

        # load in the master frametimes to get the starting time of this whole stack
        master_frametimes_df = pd.read_hdf(
            Path(pathutils.pathcrawler(each_sequence_path, inset=set(), inlist=[], mykey='master_frametimes')[0]))
        start_img_time = str(master_frametimes_df.time.iloc[0])
        start_img_dt = datetime.strptime(start_img_time, "%H:%M:%S.%f")
        end_img_time = str(master_frametimes_df.time.iloc[-1])
        end_img_dt = datetime.strptime(end_img_time, "%H:%M:%S.%f")

        # gather the stimulation times by adding the seconds to the starting time
        stimulation_times = np.array([start_img_dt + timedelta(seconds=s) for s in add_seconds])
        stimulation_times = np.array([a for a in stimulation_times if a <= end_img_dt])

        # save the photostimulation datetimes in the respective folders
        np.save(Path(each_sequence_path).joinpath('phtostimulation_datetimes.npy'), stimulation_times)

        all_stimulation_times.append(stimulation_times)
    all_stimulation_times = np.concatenate(all_stimulation_times)

    with os.scandir(Path(concatenated_dataset_path).joinpath('output_folders')) as planes:
        for each_plane in planes:
            plane_frametimes_df = pd.read_hdf(Path(each_plane.path).joinpath('frametimes.h5'))
            plane_frametimes_df['time_dt'] = [datetime(1900, 1, 1, t.hour, t.minute, t.second, t.microsecond)
                                              for t in plane_frametimes_df['time']]
            matches = plane_frametimes_df.loc[
                [(plane_frametimes_df['time_dt'] - t).abs().idxmin() for t in all_stimulation_times]]
            time_matches = matches['time_dt'].tolist()
            frames = np.array(matches.index.tolist())

            save_path = Path(each_plane.path).joinpath('bad_frames.npy')
            np.save(save_path, frames)

    return print('saved concatenated bad frames array')

def manually_remove_bad_frames(base_fish, save = True):
    if base_fish.badframes_arr is None:
        save_badframes_arr(base_fish)
    
    img = base_fish.load_image()
    img_trimmed = np.array([frame for i, frame in enumerate(img) if i not in base_fish.badframes_arr])

    if save:
        original_img_save_path = Path(base_fish.folder_path).joinpath('original_image/original_img_stack.tif')
        imwrite(original_img_save_path, img)

        trimmed_img_save_path = Path(base_fish.folder_path).joinpath('img_stack.tif')
        imwrite(trimmed_img_save_path, img_trimmed)

    # move the other tiff files into the original_image folder, so now only working with the trimmed image
    tiff_files = list(Path(base_fish.folder_path).glob('*.tif'))
    for tiff_file in tiff_files:
        if 'img_stack' not in tiff_file.name:
            tiff_file.rename(Path(base_fish.folder_path).joinpath('original_image').joinpath(tiff_file.name))   

    return print('trimmed image saved')

def create_new_ps_events_array(base_fish):
    '''
    New array of the start frame for each photostimulation event
    with the trimmed image, this is different from original bad_frames_arr 
    '''
    
    base_fish.badframes_arr = np.load(Path(base_fish.folder_path).joinpath('bad_frames.npy'))

    og_end_of_ps_event = np.where(np.abs(np.diff(base_fish.badframes_arr)) > 2)[0]
    og_inds_of_ps_event = og_end_of_ps_event + 1
    og_inds_of_ps_event = np.array([0] + list(og_inds_of_ps_event))
    og_start_of_ps_event = base_fish.badframes_arr[og_inds_of_ps_event]

    start_ind = og_start_of_ps_event[0]
    new_ps_frames = [start_ind]
    for i, n in enumerate(og_start_of_ps_event):
        if i != 0:
            new_ps_frames.append(n - og_inds_of_ps_event[i])

    # save new ps_frames array
    save_path = Path(base_fish.folder_path).joinpath('ps_frames.npy')
    np.save(save_path, new_ps_frames)

    return new_ps_frames

# running source extraction

# troubleshooting on 10/10/24 - found these to be important params to change for photostim datasets, makes more cells than are actually cells
suite2p_params = { 'preclassify' : 0.01, 'threshold_scaling': 0.7, 'max_overlap':1}
# as of 12/09/24 - found this found more realistic cells
suite2p_params2 = { 'preclassify' : 0.05, 'threshold_scaling': 0.8, 'max_overlap':1}
# as of 3/4/25 - found this to be the best params for photostim datasets
suite2p_params3 = { 'preclassify' : 0.1, 'threshold_scaling': 0.8, 'max_overlap':1}

def run_suite2p_PS(somebasefish, input_tau = 1.5, custom_parameter_dict = None, move_corr = False, force = False):
    '''
    somebasefish = the data you want to have suite2p run on
    input_tau = decay value for gcamp indicator (6s = 1.5, m = 1.0, f = 0.7)
    spatial_scale = the predicted pixel size of a ROI (2 - 12 pixels, 1 - 6 pixels)
    move_corr = binary, if you want the motion corrected image to be run as the main image or not
    '''
    from suite2p import run_s2p, default_ops
    from fishy import BaseFish
    import shutil

    if force == True:
        shutil.rmtree(Path(somebasefish.folder_path).joinpath('suite2p'))

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

    # basic changes to suite2p ops to fit the fishy format
    basic_s2p_ops = {
            "data_path": [imagepath.parents[0].as_posix()],
            "save_path0": imagepath.parents[0].as_posix(),
            "tau": input_tau, #gcamp6s = 1.5, gcamp6m = 1.0, gcamp6f = 0.7
            "preclassify": 0.15,
            "allow_overlap": True,
            "block_size": [32, 32],
            "fs": fishy.BaseFish.hzReturner(somebasefish.frametimes_df),
            "tiff_list": [imagepath.name],
            "two_step_registration" : True,
            "keep_movie_raw":True,
        }

    ps_s2p_ops = default_ops()
    db = {}

    for item in basic_s2p_ops:
        ps_s2p_ops[item] = basic_s2p_ops[item]

    # for additional changes:
    if custom_parameter_dict is not None: # can edit parameters as you want
        for key in custom_parameter_dict:
            if key in ps_s2p_ops:
                ps_s2p_ops[key] = custom_parameter_dict[key]

    db = {}
    run_s2p(ops=ps_s2p_ops, db=db)

def run_caiman_cnmf_PS(base_fish, custom_parameter_dict = None, match_suite2p = True, keep_mmaps = False):
    manually_remove_bad_frames(base_fish)
    base_fish.process_filestructure(base_fish.midnight_noon_keyword) # update file structure
    process.caiman_cnmf(base_fish, custom_parameter_dict, match_suite2p, keep_mmaps)

# identifying stimed sites and collecting its data
def identify_stim_sites(somebasefish, rotate = True, stimulation_type = 'single_cell'):
    '''
    use a base fish, saves a stimulated site dataframe for each unique plane
    '''
    somebasefish.stim_sites_df = pd.DataFrame(columns = ['plane', 'x_stim', 'y_stim', 'sp_size'])

    # 1 - use the info xml file to get the pixel data
    pixel_info = read_xml_to_str(somebasefish.data_paths['info_xml'])
    for i in (pixel_info.split("\n")):
        if "pixelsPerLine" in i:
            pixels_per_line = int(i.split('value=')[1].split('"')[1])
        if "linesPerFrame" in i:
            lines_per_frame = int(i.split('value=')[1].split('"')[1])

    # 2 - gathering z planes from the xml files, but have to have special formatting
    z_planes_data = get_zstep_vals(somebasefish.data_paths['info_xml'])
    z_planes_data = np.unique(["{:.5f}".format(z) for z in z_planes_data])
    z_planes_data = np.array([f"{0.0:.5f}" if float(x) == 0.0 else x for x in z_planes_data]) # ensures no negative 0 values
    z_planes_data = sorted(z_planes_data, key=lambda x: float(x))

    ps_xml = read_xml_to_str(somebasefish.data_paths['ps_xml'])
    X_stim_sites = []
    Y_stim_sites = []
    spiral_size_lst = []

    # 3 - gather the list of strings for each photostimulation, and specific Point Index value for later indexing
    list_of_stimulations = [i for i in ps_xml.split("\n") if f'Point Index=' in i]
    idx_of_stimulations = [int(n.split('<Point Index="')[1].split('" X')[0]) - 1 for n in list_of_stimulations]
    somebasefish.stim_sites_df = somebasefish.stim_sites_df.reindex(np.arange(len(idx_of_stimulations)))
    for i in list_of_stimulations:
        X_stim = float(i.split('X')[1].split('"')[1])*pixels_per_line
        X_stim_sites.append(round(X_stim))

        Y_stim = float(i.split('Y')[1].split('"')[1])*lines_per_frame
        Y_stim_sites.append(round(Y_stim))

        spiral_size = float(i.split('SpiralSizeInMicrons')[1].split('"')[1])
        spiral_size_lst.append(round(spiral_size))

    # 4 - need to rotate and transform the coordinates if the image is rotated from off the Bruker
    if rotate:
        coord_stim_sites = list(zip(X_stim_sites, Y_stim_sites))

        ##TODO: make this cleaner to find the correct y and x coords, i should not have to do this separately
        correct_y_coords = rotate_transform_coors(coord_stim_sites, 90, translation=(pixels_per_line, 0))
        correct_x_coords = rotate_transform_coors(coord_stim_sites, -90, translation=(0, pixels_per_line))

        X_stim_sites = [x[0] for x in correct_x_coords]
        Y_stim_sites = [-y[1] + pixels_per_line for y in correct_y_coords]

    # 5 - getting the correct z plane values for the different types of stimulations
    if stimulation_type == 'single_cell': # single cell stimulation
        with open(somebasefish.data_paths['info_env'], "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "PVMarkPoints" in line and "active" in line:
                    z_step_stimulation_site = int(float(lines[i+2].split(" ")[-2].split("=")[1].split('"')[1]))
        z_step_stimulation_site = "{:.5f}".format(z_step_stimulation_site)  # format to match z_planes_data formatting
        if z_step_stimulation_site in z_planes_data: # sometimes the z vals do not line up exactly 
            z_to_plane = [a for a, b in enumerate(z_planes_data) if b == z_step_stimulation_site][0]
        # TO DO be wary of this...
        else: # NOTE THIS IS NOT EXACT PLANES - this is the closest z value to the plane number 
            z_to_plane = min(range(len(z_planes_data)), key=lambda i: abs(float(z_planes_data[i])- float(z_step_stimulation_site)))
        this_plane = z_to_plane

        if len(list_of_stimulations) > 1:  # if multiple lines in the list of stimulations, but this is not ensemble activity
            # need to learn the stim events
            print('mulitple stimulation sites in MP output for single cell')
            bf = np.load(Path(somebasefish.folder_path).joinpath('bad_frames.npy'))
            somebasefish.stim_sites_df["stim_frames"] = [[] for _ in range(len(somebasefish.stim_sites_df))]
            somebasefish.stim_sites_df["stim_events"] = [[] for _ in range(len(somebasefish.stim_sites_df))]
            for idx, frame in enumerate(bf):
                cell = idx_of_stimulations[idx % len(idx_of_stimulations)]
                somebasefish.stim_sites_df.at[cell, "stim_frames"].append(frame)
                somebasefish.stim_sites_df.at[cell, "stim_events"].append(idx)

    else: # ensemble stimulation
        with open(somebasefish.data_paths['info_env'], "r") as f:
            lines = f.readlines()
            # sometimes have multiple PVGalvoPointGroups, so need to check how many there are for the accurate z planes
            multiple_groups = sum('PVGalvoPointGroup' in line for line in lines)
            if multiple_groups < 2:
                ind_start = [i+2 for i, line in enumerate(lines) if "PVMarkPoints" in line and "active" in line][0]
                ind_end = [i for i, line in enumerate(lines) if "PVGalvoPointGroup" in line][0]
            else: 
                group_inds = [line.split('Indices="')[1].split('"')[0].split(',') for line in lines if 'PVGalvoPointGroup' in line and 'Indices' in line]
                one_index_stimulated = list_of_stimulations[1].split('Point Index=')[1].split('"')[1]
                #check if one of the stimulation inds is in the group inds from these lines
                group_num = [g for g, group_ind_lst in enumerate(group_inds) if one_index_stimulated in group_ind_lst][0]
                if group_num == 0:
                    ind_start = [i+2 for i, line in enumerate(lines)  if "PVMarkPoints" in line and "active" in line][0]
                    ind_end = [i for i, line in enumerate(lines) if "PVGalvoPointGroup" in line][1]
                else:
                    ind_start = [i+1 for i, line in enumerate(lines) if "PVGalvoPointGroup" in line][group_num - 1]
                    ind_end = [i for i, line in enumerate(lines) if "PVGalvoPointGroup" in line][group_num]
            ind_lines = np.arange(ind_start, ind_end, step=1)
            z_vals = [float(lines[i].split('Z')[1].split('"')[1]) for i in ind_lines]
            z_vals = ["{:.5f}".format(z) for z in z_vals]
            z_vals = [z_vals[i] for i in idx_of_stimulations]

        map_z_to_plane_num = {z: i for i, z in enumerate(z_planes_data)}  
        print(z_planes_data)      
        # convert z values into planes
        if all(key in z_vals for key in map_z_to_plane_num):
            map_z_to_plane_num = map_z_to_plane_num
        # # TO DO be wary of this...
        # else: # NOTE THIS IS NOT EXACT PLANES - this is the closest z value to the plane number
        #     map_z_to_plane_num = {min(z_vals, key=lambda z: abs(float(z) - float(old_key))): value for old_key, value in map_z_to_plane_num.items()}

        # make sure i get the closest correct plane
        lookup_float = {float(k): v for k, v in map_z_to_plane_num.items()}
        z_to_plane = []
        for val in z_vals:
            fval = float(val)
            closest_key = min(lookup_float.keys(), key=lambda k: abs(k - fval)) # Find the key in lookup_float with the smallest absolute difference
            z_to_plane.append(lookup_float[closest_key])
        this_plane = int(somebasefish.folder_path.name.split('_')[1])

    # 6 - adding stimmed frames for each unique cell/group that was stimulated

    somebasefish.stim_sites_df['x_stim'] = X_stim_sites
    somebasefish.stim_sites_df['y_stim'] = Y_stim_sites
    somebasefish.stim_sites_df['sp_size'] = spiral_size_lst
    somebasefish.stim_sites_df['plane'] = z_to_plane

    # save a big stim sites dataframe to the stim folder
    master_save_path = Path(somebasefish.folder_path.parents[1]).joinpath('stim_sites_volume.h5')
    somebasefish.stim_sites_df.to_hdf(master_save_path, key="volume_stim")

    # trim dataframe to only include stim sites for that precise z plane to keep into that folder
    somebasefish.stim_sites_df = somebasefish.stim_sites_df[somebasefish.stim_sites_df['plane'] == this_plane]
    somebasefish.stim_sites_df.reset_index(inplace = True, drop = True)

    save_path = Path(somebasefish.folder_path).joinpath("stim_sites.hdf")
    somebasefish.stim_sites_df.to_hdf(save_path, key="stim")

    return somebasefish.stim_sites_df

def identify_stim_sites_from_markpoints(info_xml_path, markpoints_xml_path, info_env_path, rotate = True, 
                                   stimmed_plane_num = 0, planes_stimed = [1,2,3,4]):
    '''
    old function
    planes_stimed is hard coded, not sure how to gather the z plane info with not a clear output file 
    does not use a BaseFish to collect paths and such
    '''
    stim_sites_df = pd.DataFrame(columns = ['plane', 'x_stim', 'y_stim', 'sp_size'])

    # use the info xml file to get the pixel data
    pixel_info = read_xml_to_str(info_xml_path)
    for i in (pixel_info.split("\n")):
        if "pixelsPerLine" in i:
            pixels_per_line = int(i.split('value=')[1].split('"')[1])
        if "linesPerFrame" in i:
            lines_per_frame = int(i.split('value=')[1].split('"')[1])
    
    # use the ps xml file to get the stim site data
    ps_xml = read_xml_to_str(markpoints_xml_path)
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
        with open(info_env_path, "r") as f:
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

        this_plane = stimmed_plane_num
    else:
        z_to_plane = 0 # just the plane that you recorded from
        this_plane = 0

    stim_sites_df['x_stim'] = X_stim_sites
    stim_sites_df['y_stim'] = Y_stim_sites
    stim_sites_df['sp_size'] = spiral_size_lst
    stim_sites_df['plane'] = z_to_plane

    stim_sites_df = stim_sites_df[stim_sites_df['plane'] == this_plane]
    stim_sites_df.reset_index(inplace = True, drop = True)

    return stim_sites_df

def return_raw_coord_trace(cell_coord, img, s=5):
    """
    cell_coord: (x, y) in pixel coordinates (column, row)
    img: shape (time, height, width)
    """
    mask = create_circular_mask(img.shape[1:], cell_coord[0], cell_coord[1], s)
    return np.nanmean(img[:, mask], axis=1)

def collect_raw_traces(somephotostimfish):
    
    try:
        img = imread(somephotostimfish.data_paths["move_corrected_image"])
    except:
        img = imread(somephotostimfish.data_paths["rotated_image"])

    raw_traces = np.zeros((len(somephotostimfish.stim_sites_df), img.shape[0]))
    points = np.zeros((len(somephotostimfish.stim_sites_df), 2))
    um_per_pxs = get_micronstopixels_scale2(somephotostimfish.data_paths['info_xml'])
    for point in range(len(somephotostimfish.stim_sites_df)):
        pt = somephotostimfish.stim_sites_df.iloc[point]
        pt_sp_size_pixels = pt.sp_size / um_per_pxs
        msk = create_circular_mask(img.shape[1:], pt.x_stim, pt.y_stim, pt_sp_size_pixels/2)
        msk_trace = np.nanmean(img[:, msk], axis=1)
        raw_traces[point] = msk_trace
        points[point] = [pt.x_stim, pt.y_stim]

    # save the raw traces   
    np.save(Path(somephotostimfish.folder_path).joinpath('raw_traces.npy'), raw_traces)

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

# more functions to help analysis on the photostimulation data
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

def calculate_evoked_response(arr_cell_traces, arr_subset, ps_offset = 0, frame_window = [-4, 7], r_type = 'mean'):
    '''
    ps_offset = the offset in frames from the photostimulation event to collect data, in case the event is longer than 1 frame
    '''
    evoked_response = np.zeros(shape = (arr_cell_traces.shape[0], 1))
    for a, arr in enumerate(arr_cell_traces):
        # evoked responses across trials for each ROI cell 
        each_trial = np.array([arr[s] for s in arr_subset])
        each_trial_evoked_resp = np.zeros(shape = (len(each_trial), len(each_trial[0])))
        # base_e = matching_single_cell_trace[matching_fish.badframes_arr[0] + offsets[0]:matching_fish.badframes_arr[0] ] 
        for d, f in enumerate(each_trial):
            base_e = f[:-frame_window[0]]
            plot_e = (f - np.nanmean(base_e)) / np.nanmean(base_e)
            if r_type == 'mean':
                each_trial_evoked_resp[d] = np.nanmean(plot_e[(-frame_window[0] + ps_offset):]) # frames necessary for looking at evoked window
            elif r_type == 'median':
                each_trial_evoked_resp[d] = np.nanmedian(plot_e[(-frame_window[0] + ps_offset):])
        evoked_response[a] = np.nanmean(each_trial_evoked_resp)

    return evoked_response

# helpful pre processing functions for the automated gui, keeping datasets all in separate folders
def process_output_files(folder):
    '''
    Process the output log text file to get the stimulation times and the z plane for each stimulation
    Information comes from the stimulation commands themselves
    folder - the folder path that contains the output log file abd bruker_coordinate_list txt files, for sorting through the data

    Returns a dataframe with the stimulation events and their corresponding z plane, x and y coordinates
    '''

    output_log_path = Path(folder).joinpath('output.txt')
    with open(output_log_path) as file:
        contents = file.read()
    lines = contents.split("\n")
    stim_lines = [l for l in lines if 'Stim event' in l]
    x_coord = [int(s.split('[')[1].split(',')[0]) for s in stim_lines]
    y_coord = [int(s.split('[')[1].split(',')[1]) for s in stim_lines]
    z_coord = [int(s.split('[')[1].split(',')[2].split(']')[0]) for s in stim_lines]

    # to find the correct z plane, look at the original coordinates txt file, in python orientation
    coordinate_txt_file = Path(folder).joinpath('bruker_coordinate_list.txt')
    with open(coordinate_txt_file) as file:
        coords_txt = file.read()
    coords = coords_txt.split("\n")
    og_coords_lst = [[int(d) for d in c.split(',')] for c in coords if ',' in c]
    unique_stimmed_z_planes = np.unique(sorted([i[2] for i in og_coords_lst]))

    # gathering x, y, z plane for each unique cell id
    z_planes = []
    cell_ids = []
    for each_sequence in range(int(len(x_coord)/len(og_coords_lst))):
        each_sequence_inds = [each_sequence*len(og_coords_lst), (each_sequence+1)*len(og_coords_lst)] # start and stop seq inds
        specific_z_coords = z_coord[each_sequence_inds[0]:each_sequence_inds[1]]
        unique_specific_z_coords = np.unique(sorted(specific_z_coords))
        z_coords_to_planes = {zc: zp for zc, zp in zip(unique_specific_z_coords, unique_stimmed_z_planes)}
        for e in range(len(x_coord[each_sequence_inds[0]:each_sequence_inds[1]])):
            x = x_coord[each_sequence_inds[0]:each_sequence_inds[1]][e]
            y = y_coord[each_sequence_inds[0]:each_sequence_inds[1]][e]
            z = z_coords_to_planes[z_coord[each_sequence_inds[0]:each_sequence_inds[1]][e]]
            for k, j in enumerate(og_coords_lst):
                if x == j[0] and y == j[1] and z == j[2]:
                    z_planes.append(f'plane_{j[2]}')
                    cell_ids.append(k)
    
    # gathering unique stim events for each cell id
    ps_events_per_cell = {}
    for c in cell_ids[:len(og_coords_lst)]:
        if c not in ps_events_per_cell.keys():
            ps_events_per_cell[c] = []
        ps_events_per_cell[c] = np.array([i for i, item in enumerate(cell_ids) if item == c])

    # gather photostimulation parameters from the output file and xml information file, assuming all parameters are the same for each site    
    try:
        with os.scandir(folder) as entries:  # find a xml file in this folder, but if not there...
            for entry in entries: 
                if 'xml' in entry.name: 
                    xml_info_path = entry.path
        px_per_line = get_pixelsperline(xml_info_path)
        um_per_px = get_micronstopixels_scale2(xml_info_path)
    except: # default values in case there is no xml file in the complete dataset folder
        px_per_line = 512
        um_per_px = 0.602463686424765
    
    cmd = stim_lines[0].split('-MarkAllPoints')[1]
    duration_ms = int(cmd.split('Monaco 1035')[0].split(' ')[-2])
    no_repetitions = cmd.count('Monaco 1035')
    try:
        interpointdelay_ms = int(cmd.split('True')[2].split(' ')[3]) # if there are mulitple reps
    except:
        interpointdelay_ms = 0 # since there is no repetitions
    spiral_size_perc = float(cmd.split('True')[2].split(' ')[1]) # reads the spiral size as a percent of X pixels
    spiral_size = spiral_size_perc * px_per_line * um_per_px # now spiral size in microns
    full_duration_per_stim = (no_repetitions * duration_ms) + ((no_repetitions-1) * interpointdelay_ms)

    output_df = pd.DataFrame({'plane': z_planes, 'x_stim': x_coord, 'y_stim': y_coord, 'cell_ids': cell_ids})
    output_df['sp_size'] = spiral_size
    output_df['stim_duration_ms'] = full_duration_per_stim
    output_df['stim_events'] = np.nan
    for c_id, unique_events in ps_events_per_cell.items():
        i = output_df.loc[output_df['cell_ids'] == c_id].index[0] # index of the first cell id 
        output_df.loc[[i], 'stim_events'] = pd.Series([unique_events], index=output_df.index[[i]]) # put in the unique events
    # trim the dataframe to only include the unique cell ids
    output_df = output_df.drop_duplicates(subset = 'cell_ids').reset_index(drop = True)

    output_df.to_hdf(Path(folder).joinpath('master_stim_sites.h5'), key="stim")
    
    return output_df

def process_output_files_for_ensembles(folder):

    try:
        ensemble_info_df = pd.read_hdf(Path(folder).joinpath('ensembles_df.h5'))
    except:
        return print('need ensemble info dataframe in folder')

    # read in the output text file
    output_log_path = Path(folder).joinpath('output.txt')
    with open(output_log_path) as file:
        contents = file.read()
    lines = contents.split("\n")
    stim_lines = [l for l in lines if 'Stim event' in l]

    try: # find a xml file in this folder, but if not there...
        with os.scandir(folder) as entries:
            for entry in entries:
                if 'xml' in entry.name:
                    xml_info_path = entry.path
        px_per_line = get_pixelsperline(xml_info_path)
        um_per_px = get_micronstopixels_scale2(xml_info_path)
    except:  # default values in case there is no xml file in the complete dataset folder
        print('using default values for pixel per line and um/px scale')
        px_per_line = 512
        um_per_px = 0.602463686424765

    # first save all the ensemble info for later
    stim_info = []
    for i, line in enumerate(stim_lines):
        if "-MarkAllPoints" not in line:
            continue

        # parse stim info
        cmd = line.split('-MarkAllPoints')[1]
        duration_ms = int(cmd.split('Monaco 1035')[0].split(' ')[-2])
        no_repetitions = cmd.count('Monaco 1035')
        try:
            interpointdelay_ms = int(cmd.split('True')[2].split(' ')[3])
        except:
            interpointdelay_ms = 0
        spiral_size_perc = float(cmd.split('True')[2].split(' ')[1])
        spiral_size = spiral_size_perc * px_per_line * um_per_px
        full_duration_per_stim = (no_repetitions * duration_ms) + ((no_repetitions - 1) * interpointdelay_ms)
        ensemble_id = line.split('ensemble ')[1].split(':')[0]

        stim_info.append({
            "stim_events": i,
            "stim_duration_ms": full_duration_per_stim,
            "sp_size": spiral_size,
            "ensemble_id": ensemble_id
        })

    stim_df = pd.DataFrame(stim_info)

    ensemble_output_df = ensemble_info_df.merge(stim_df.groupby("ensemble_id").agg({
                                            "stim_events": list,
                                            "stim_duration_ms": "first",  # all 9 should match
                                            "sp_size": "first"  # all 9 should match
                                        }).reset_index(),on="ensemble_id", how="left")
    cols = ["ensemble_id"] + [c for c in ensemble_output_df.columns if c != "ensemble_id"]
    ensemble_output_df = ensemble_output_df[cols]
    ensemble_output_df.to_hdf(Path(folder).joinpath('master_stim_ensembles.h5'), key="stim")

    # then use this to also save a typical master_stim_sites.h5 for use later
    # starting from your ensemble info df
    # explode both ensemble_cells and ensemble_coords so each cell in an ensemble gets its own row
    ensemble_output_df_copy = ensemble_output_df.copy()
    long_df = ensemble_output_df_copy.explode(["ensemble_cells", "ensemble_coords"]).reset_index(drop=True)
    long_df[["x_stim", "y_stim", "plane_num"]] = pd.DataFrame(long_df["ensemble_coords"].tolist(), index=long_df.index)
    long_df["plane"] = long_df["plane_num"].apply(lambda p: f"plane_{p}")
    single_sites_df = long_df.rename(columns={"ensemble_cells": "cell_ids"})
    single_sites_df = single_sites_df[["plane", "x_stim", "y_stim", "cell_ids","sp_size", "stim_duration_ms", "stim_events"]]
    single_sites_df = single_sites_df.drop_duplicates(subset=["plane", "x_stim", "y_stim", "cell_ids"]).reset_index(drop=True)
    single_sites_df.to_hdf(Path(folder).joinpath('master_stim_sites.h5'), key="stim")

    return ensemble_output_df, single_sites_df

def organize_bad_frames_in_individual_folders(folder, subfolder_keyword = 'sequence'):
    '''
    Organize the output dataframe to have stim sites df into each respective folder/plane
    folder - the folder path that contains the output log file abd bruker_coordinate_list txt files, for sorting through the data
    Make sure there are no other directories than the important data folders in the main folder
    '''
    if Path(folder).joinpath('master_stim_sites.h5').exists(): # if already ran the output processing function
        output_df = pd.read_hdf(Path(folder).joinpath('master_stim_sites.h5'), key="stim")
    else: # or if not
        output_df = process_output_files(folder)
    
    # gathering a dictionary of all the data folders
    data_folder_dict = {}
    data_count = 0
    with os.scandir(folder) as entries:
        for entry in entries:
            if (entry.is_dir()) & (subfolder_keyword in entry.name):
                data_folder_dict[data_count] = Path(entry.path)
                data_count += 1

    # using dictionary to index into the correct data sets in the stim sites df (stim duration), then save bad frames for each folder
    stim_duration_ms = output_df.stim_duration_ms.iloc[0] # assuming they are all the same
    for d, p in data_folder_dict.items():
        if 'output_folders' != p.name: 
            p = Path(p).joinpath('output_folders')
        with os.scandir(p) as entries:
            for entry in entries:
                if 'plane' in entry.name:
                    plane_path = Path(entry.path)
                    quick_fish = fishy.BaseFish(folder_path = plane_path, frametimes_key= 'frametimes')
                    print(plane_path)
                    bad_frames_lst = save_badframes_arr(quick_fish, automated_gui = True)
                    print(f'baseline frames = {quick_fish.baseline_frames}')
                    print(f'bad frames = {bad_frames_lst}')
                    img_hz = fishy.BaseFish.hzReturner(pd.read_hdf(plane_path.joinpath('frametimes.h5')))
                    stim_duration_frames = np.ceil((stim_duration_ms/1000) * img_hz) # rounding up
                    print('stim duration frames = ', stim_duration_frames)
                    cleaned_bad_frames_lst = arrutils.filter_list(lst = np.unique(bad_frames_lst), interval = stim_duration_frames)
                    print(f'length of cleaned up bad frames list: {len(cleaned_bad_frames_lst)}')
                    if len(cleaned_bad_frames_lst) < len(output_df): # if there are less bad frames than stim sites, meaning there was no baseline
                        cleaned_bad_frames_lst.insert(0, 0)
                    print(bad_frames_lst[0])
                    np.save(Path(plane_path).joinpath('bad_frames.npy'), cleaned_bad_frames_lst)

def organize_output_df(folder, type_of_stim = 'single_cell'):
    '''
    Organize the output dataframe to have stim sites df if this is a complete volume dataset 
    All collected datasets need to be concatenated into one folder
    folder - the folder path that contains the output log file abd bruker_coordinate_list txt files, for sorting through the data
    Make sure there are no other directories than the important data folders in the main folder
    '''
    if Path(folder).joinpath('master_stim_sites.h5').exists(): # if already ran the output processing function
        output_df = pd.read_hdf(Path(folder).joinpath('master_stim_sites.h5'), key="stim")
    else: # or if not
        if type_of_stim == 'single_cell':
            output_df = process_output_files(folder)
        if type_of_stim == 'ensemble':
            _, output_df = process_output_files_for_ensembles(folder)
    
    # gathering a dictionary of all the data folders
    data_folder_dict = {}
    with os.scandir(Path(folder).joinpath('output_folders')) as entries:
        for entry in entries:
            if 'plane' in entry.name:
                data_folder_dict[entry.name] = Path(entry.path)

    # using dictionary to index into the correct data sets in the stim sites df, adding in the stimulated frames and saving into each folder
    new_stim_sites_df_lst = []
    for plane_key, plane_path in data_folder_dict.items():
        if plane_key in output_df.plane.values:
            sub_output_df = output_df[output_df.plane == plane_key].reset_index(drop = True) # each plane's data
            sub_output_df['stim_frames'] = [None] * len(sub_output_df) # add in a column for the stimulated frames
            bad_frames_lst = np.load(Path(plane_path).joinpath(f'bad_frames.npy'))
            img_hz = fishy.BaseFish.hzReturner(pd.read_hdf(Path(plane_path).joinpath(f'frametimes.h5')))
            stim_duration = np.ceil((sub_output_df.stim_duration_ms.iloc[0]/1000) * img_hz) # rounding up
            cleaned_bad_frames_lst = arrutils.filter_list(lst = np.unique(bad_frames_lst), interval = stim_duration)

            #index into the correct stim events
            for ind in range(len(sub_output_df)):
                stim_event_indices = sub_output_df.stim_events[ind]
                stim_event_indices = [e for e in stim_event_indices if e < len(cleaned_bad_frames_lst)]
                print(stim_event_indices)
                specific_stim_frames = [cleaned_bad_frames_lst[e] for e in stim_event_indices]
                sub_output_df.loc[[ind], 'stim_frames'] = pd.Series([specific_stim_frames], index=sub_output_df.index[[ind]])

            sub_output_df.to_hdf(Path(plane_path).joinpath('stim_sites.hdf'), key="stim")
            new_stim_sites_df_lst.append(sub_output_df)

    # save into the main folder with the stimulated frames in the dataframe for ease processing later
    new_stim_sites_df = pd.concat(new_stim_sites_df_lst).reset_index(drop = True)
    new_stim_sites_df.to_hdf(Path(folder).joinpath('master_stim_sites.h5'), key = 'stim') 

def utils_for_save_badframes_arr(base_fish):
    '''
    Helper function to save the bad frames array for each plane in the fishy class if using the automated gui
    base_fish = the fishy class that you are working with
    '''

    with open(base_fish.folder_path.parents[2].joinpath('output.txt')) as file:
        contents = file.read()
    lines = contents.split("\n")

    finish_stim_lines_inds = [k-1 for k, l in enumerate(lines) if ('finished' in l)]
    first_stim_lines_inds = [k+1 for k, l in enumerate(lines) if ('began' in l)]

    # finding the start of the stim events for that specific dataset, using matching frametimes hours and minutes
    first_stim_lines = [lines[i] for i in first_stim_lines_inds]
    first_datetimes = [f.split(' ')[1] for f in first_stim_lines]
    datetimes_dtformat = [pd.Timestamp(i).time() for i in pd.to_datetime(first_datetimes)]
    datetimes_hours_minutes = [[i.hour, i.minute] for i in datetimes_dtformat]
    frametimes_hours_minutes = [[i.hour, i.minute] for i in base_fish.frametimes_df.time.values]
    ps_sequence = [event for event, i in enumerate(datetimes_hours_minutes) if i in frametimes_hours_minutes][0]

    # gather the STIM TIMES from that specific sequence
    stim_times = []
    for i in lines[first_stim_lines_inds[ps_sequence]: finish_stim_lines_inds[ps_sequence]+1]: # only the lines of the specific sequence
        stim_times.append(pd.Timestamp(i.split(' ')[1]).time())
    total_seconds = [t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6 for t in stim_times]
    stim_times_ms = [(s - total_seconds[0]) * 1000 for s in total_seconds] # Calculate relative time in milliseconds
    print(f'length of stim times from output file {len(stim_times)}')

    # gather the FULL DURATION PER STIM event 
    output_df = pd.read_hdf(base_fish.folder_path.parents[2].joinpath('master_stim_sites.h5'))
    full_duration_per_stim = output_df['stim_duration_ms'].iloc[0]

    # gather the BASELINE FRAMES in this dataset
    count = 0
    for dt in base_fish.frametimes_df.time.values:
        if dt < stim_times[0]:
            count += 1
    baseline_frames = count - 1

    base_fish.baseline_frames = baseline_frames

    return base_fish.baseline_frames, stim_times_ms, full_duration_per_stim


