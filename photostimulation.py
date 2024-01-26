# functions to preprocess photostimulation data 

import os
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
from tifffile import imread, imwrite
import tifftools
from datetime import datetime as dt, timedelta
import xml.etree.ElementTree as ET
import glob
import caiman as cm


def find_no_baseline_frames(somefishclass, no_planes = 0):
    '''
    folder_path = path that contains all the data (xml file and original bruker images)
    volume = set to True if this is a volume stack since the baseline frame number is calculated differently
    '''
    if no_planes < 0:
        original_imgs = Path(somefishclass.folder_path).joinpath("bruker_images")
        with os.scandir(original_imgs) as entries:
            for entry in entries:
                if 'Cycle' in entry.name and 'tif' in entry.name:
                    baseline_img = imread(entry.path)
                    break #we want the first tif file here in single image

        somefishclass.baseline_frames = baseline_img.shape[0]

    elif no_planes > 0:
        ps_xml_name = Path(somefishclass.data_paths['ps_xml']).name
        somefishclass.baseline_frames = int(ps_xml_name.split('Cycle')[1].split('_')[0]) # baseline frames number is given in the cycle name for the volume
    
    return somefishclass.baseline_frames

def save_badframes_arr(somefishclass, no_planes = 0):

    from PIL import Image
    import math

    somefishclass.baseline_frames = find_no_baseline_frames(somefishclass, no_planes)

    # collecting stimulation timing based on ms in the file
    with open(Path(somefishclass.data_paths['ps_xml']), "r") as f:
        data = f.read()

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

    full_duration_ms = initial_delay_ms + (no_repetitions * duration_ms) + ((no_repetitions-1) * interpointdelay_ms)
    stim_times = [(full_duration_ms/1000)*m + (iteration_delay_ms/1000)*m for m in range(no_iterations)]

    # using the information xml file to calculate the frames and times for each stimulation
    

    if no_planes > 0: # if volume stimulation
        plane_num = int(somefishclass.data_paths['move_corrected_image'].parents[0].name.split('_')[1])

        frametimes = []
        with open(somefishclass.data_paths["info_xml"], "r") as f:
            info_data = f.read()
        for i in info_data.split("\n"):
            if "relativeTime" in i:
                relative_time = [i.split("relativeTime=")[1].split('"')[1]][0]
                frametimes.append(float(relative_time))
        
        # find the index where the stimulation starts in the list of frametimes
        stim_ind = [index for index, value in enumerate(frametimes) if value < 0.01][1] 

        # only get the relative frametimes that happen during the stimulation
        stimulation_frametimes = frametimes[stim_ind:]

        for p in range(no_planes):
            if p == plane_num:
                plane_frametimes = stimulation_frametimes[p::no_planes]
                time_matches = [min(plane_frametimes, key=lambda y: abs(x - y)) for x in stim_times] # list of frametimes values that match with the stim_times
                frames = [plane_frametimes.index(x)-1 for x in time_matches] # list of frames that match with the stim_times

    else: # if single plane stimulation
        ##TODO: make it simpler code, combine this to just do what I am doing for volumes
        # only looking at stimulation cycle here for the right relative times
        tree = ET.parse(somefishclass.data_paths["info_xml"])
        root = tree.getroot()
        frametimes = []
        for child in root:
            if child.tag == 'Sequence' and child.attrib['cycle'] == '2':
                for subchild in child:
                    if subchild.tag == 'Frame':
                        frametimes.append(float(subchild.attrib['relativeTime']))

        # calculate ps events in frame numbers
        time_matches = [min(frametimes, key=lambda y: abs(x - y)) for x in stim_times] # list of frametimes values that match with the stim_times
        frames = [frametimes.index(x)-1 for x in time_matches] # list of frames that match with the stim_times
        # subtract 1 since that worked with the graph, I think it's because the first frame is 0 and not 1
        
        # calculate the duration of one ps event in frame numbers
        frame_dur = min(frametimes, key=lambda y: abs(full_duration_ms/1000 - y))
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

def identify_stim_sites(somebasefish, planes_stimed = [1,2,3,4]):
    '''
    planes_stimed is hard coded, not sure how to gather the z plane info with not a clear output file 
    use a base fish, saves a stimulated site dataframe for each unique plane
    '''

    og_img = imread(somebasefish.data_paths['image'])

    stim_sites_df = pd.DataFrame(columns = ['plane', 'x_stim', 'y_stim', 'sp_size'])

    pixels_per_line = int(og_img.shape[1])
    lines_per_frame = int(og_img.shape[2])
            
    with open(somebasefish.data_paths['ps_xml'], "r") as f:
        ps_xml = f.read()

    # get the values for the stim sites in pixels
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

    # get values for the z here
    with open(somebasefish.data_paths['info_env'], "r") as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            if "PVMarkPoints" in line and "active" in line:
                ind_start = i+2
            if "PVGalvoPointGroup" in line:
                ind_end = i
        ind_lines = np.arange(ind_start, ind_end, step=1)
        z_vals = [float(lines[i].split('Z')[1].split('"')[1]) for i in ind_lines]

    unique_z = np.unique(z_vals)
    map_z = {}
    for _p, p in enumerate(unique_z):
        map_z[p] = planes_stimed[_p]

    z_to_plane = [map_z[z] for z in z_vals]

    stim_sites_df['x_stim'] = X_stim_sites
    stim_sites_df['y_stim'] = Y_stim_sites
    stim_sites_df['sp_size'] = spiral_size_lst
    stim_sites_df['plane'] = z_to_plane

    # trim dataframe to only include stim sites for that precise z plane
    this_plane = int(somebasefish.folder_path.name.split('_')[1])
    stim_sites_df = stim_sites_df[stim_sites_df['plane'] == this_plane]
    stim_sites_df.reset_index(inplace = True, drop = True)

    save_path = Path(somebasefish.folder_path).joinpath("stim_sites.hdf")
    stim_sites_df.to_hdf(save_path, key="stim")

    return stim_sites_df

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


    def rotate_transform_coors(coordinates, angle_degrees, translation=(0, 0)):
        """
        this is necessary for Bruker images that are rotated
        Rotate, transform, and flip 2D coordinates. 

        - coordinates: List of (x, y) coordinates.
        - angle_degrees: Rotation angle in degrees.
        - translation: Tuple (tx, ty) for translation (default is (0, 0)).

        Returns:
        - List of transformed (x', y') coordinates.
        """
        # Convert angle to radians
        angle_radians = np.radians(angle_degrees)

        # Rotation matrix
        rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                                    [np.sin(angle_radians), np.cos(angle_radians)]])

        # Apply rotation
        rotated_coordinates = np.dot(rotation_matrix, np.array(coordinates).T).T

        # Apply translation
        translated_coordinates = rotated_coordinates + np.array(translation)

        return translated_coordinates.tolist()

def rotate_transform_coors(coordinates, angle_degrees, translation=(0, 0)):
    """
    Rotate and transform 2D coordinates.

    Parameters:
    - coordinates: List of (x, y) coordinates.
    - angle_degrees: Rotation angle in degrees.
    - translation: Tuple (tx, ty) for translation (default is (0, 0)).

    Returns:
    - List of transformed (x', y') coordinates.
    """
    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)

    # Rotation matrix
    rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                                [np.sin(angle_radians), np.cos(angle_radians)]])

    # Apply rotation
    rotated_coordinates = np.dot(rotation_matrix, np.array(coordinates).T).T

    # Apply translation
    translated_coordinates = rotated_coordinates + np.array(translation)

    return translated_coordinates.tolist()

