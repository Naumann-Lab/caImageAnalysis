import warnings
import logging
# Suppress tifffile logging warnings
logging.getLogger('tifffile').setLevel(logging.ERROR)
# Optional: also suppress standard UserWarnings if needed
warnings.filterwarnings("ignore", category=UserWarning, module="tifffile")

import os
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
from tifffile import imread, imwrite
from datetime import datetime as dt, timedelta
import xml.etree.ElementTree as ET
import glob
import caiman as cm
from scipy.signal import find_peaks

import sys
sys.path.append(r'C:\Users\Kaitlyn\PyCharmProjects\imaging\caImageAnalysis')
from utilities import pathutils, arrutils
# import fishy

def get_frametimes(info_xml_path, voltage_path):
    '''
    Calculating frame times from either just the information xml file or the voltage recording if it is there
    '''

    root = read_xml_to_root(info_xml_path)
    times = []
    for start in root.iter('PVScan'):
        begin_time = start.attrib['date'].split(' ')[1]
        start_dt = dt.strptime(begin_time, "%H:%M:%S").time()
        hour = int(start.attrib['date'].split(' ')[1].split(':')[0])
        if (start.attrib['date'].split(' ')[2] == 'PM') & (hour != 12): # convert to 24hr time for stim files
            start_dt = addHours(start_dt, float(12))

    if voltage_path:
        # use the voltage recording to get the frametimes
        volt_csv = pd.read_csv(voltage_path)
        frame_signal = np.array(volt_csv[' frame_out'])
        time_signal = np.array(volt_csv['Time(ms)'])

        frames, _ = find_peaks(frame_signal, height=5) # find peaks in voltage trace
        frame_starts = [frames[i] for i in range(len(frames)) if i == 0 or frames[i] - frames[i-1] > 10] # find only the start of each peak
        frametimes_ms = time_signal[frame_starts]
        times = [addSecs(start_dt, float(added_ms/1000)) for added_ms in frametimes_ms]

    elif not voltage_path:
        # use the info xml file to get the frametimes
        for frame in root.iter('Frame'):
            added_secs = frame.attrib['absoluteTime']
            frame_dt = addSecs(start_dt, float(added_secs))
            times.append(frame_dt)

    frametimes_df = pd.DataFrame(times)
    frametimes_df.rename({0: "time"}, axis=1, inplace=True)
    save_path = Path(info_xml_path).parents[0].joinpath(
        "master_frametimes.h5"
    )
    frametimes_df.to_hdf(save_path, key="frames", mode="a")  # saving master frametimes file

    return frametimes_df
    
def bruker_img_organization(folder_path, testkey = 'Cycle', safe=False, single_plane=False, pstim_file = True):
    '''
    PV 5.8 software bruker organization function (ome tif files into regular tif files)

    folder_path = data folder path
    testkey = the key that is in each bruker ome tif file
    safe = safety key
    single_plane = mark True if your data collected is a single plane, otherwise its a volume
    pstim_file = if you have a stimulus txt file that needs to be moved into each plane folder
    '''
    keyset = set()

    voltage_path = None

    with os.scandir(folder_path) as entries:
        for entry in entries:
            if 'companion' in entry.name:
                pass
            elif testkey in entry.name and 'tif' in entry.name:
                keyset.add(
                    entry.name.split("Cycle")[1].split("_")[0]
                )  # make a key for each volume or if single plane, each plane in the t-series
            elif entry.name.endswith(".xml") and "MarkPoints" not in entry.name and "Voltage" not in entry.name:
                info_xml_path = Path(entry.path)
            elif 'txt' in entry.name and pstim_file:
                pstim_path = Path(entry.path)
            elif 'Voltage' in entry.name and entry.name.endswith(".csv"):
                voltage_path = Path(entry.path)

    # making new output folders
    new_output = Path(folder_path).joinpath(
        "output_folders"
    )  # new folder to save the output tiffs
    if not os.path.exists(new_output):
        os.mkdir(new_output)

    # collect frame times from files
    frametimes_df = get_frametimes(info_xml_path, voltage_path = None)

    if single_plane == True:

        # collect only the tif files that you need (not in References)
        fls = [Path(x) for x in pathutils.pathcrawler(folder_path, inset=set(), inlist=[], mykey = testkey)]
        fls = [x for x in fls if 'References' not in str(x)]

        # fls = glob.glob(os.path.join(folder_path,'*.tif'))  #  change tif to the extension you need
        fls.sort()  # make sure your files are sorted alphanumerically
        m = cm.load_movie_chain(fls)
        save_fld = Path(new_output).joinpath(f"single_plane")
        if not os.path.exists(save_fld):
            os.mkdir(save_fld)
        m.save(os.path.join(save_fld,'img_stack.tif'))

        save_path = Path(save_fld).joinpath(
            "frametimes.h5"
        )
        frametimes_df.to_hdf(save_path, key="frames", mode="a")  # saving frametimes file into single plane

        if pstim_file:  # if pstim output exists, save into each folder
            shutil.copy(pstim_path, Path(save_fld).joinpath(f"pstim_output.txt"))
        else:
            print('no pstim file')

    else:
        # do everything for volumes here
        volume_path_dict = {k: {} for k in sorted(keyset)}
        print(sorted(keyset))

        # image paths go in this dict for each volume
        for k in volume_path_dict.keys():
            with os.scandir(folder_path) as entries:
                for entry in entries:
                    if f'Cycle{k}' in entry.name and 'tif' in entry.name:
                        volume_path_dict[k] = entry.path

        # number of planes gotten from the first image
        plane_no = imread(volume_path_dict[k]).shape[0]
        planes_dict = {k: [] for k in range(plane_no)}  # dictionary for each plane

        print(sorted(volume_path_dict.keys()))
        for k in sorted(volume_path_dict.keys()):
            vol_img = volume_path_dict[k]
            for n in range(len(planes_dict.keys())):
                img = imread(vol_img)[n]
                planes_dict[n].append(img)  # each plane_dict key is a different plane, with every image in a list    
    
        # getting plane stacks into specific folders
        for k, v in planes_dict.items():
            fld = Path(new_output).joinpath(f"plane_{k}")
            if not os.path.exists(fld):
                os.mkdir(fld)
            for i, individual in enumerate(v):
                _i = str(("%05d" % i))
                imwrite(
                    fld.joinpath(f"individual_img_{k}_{_i}.tif"), individual
                )  # saving new tifs, each one is a time series for each plane
            fls = glob.glob(os.path.join(fld,'*.tif'))  #  change tif to the extension you need
            fls.sort()  # make sure your files are sorted alphanumerically
            m = cm.load_movie_chain(fls)
            m.save(os.path.join(fld,f'img_stack_{k}.tif'))
            with os.scandir(fld) as entries:
                for entry in entries:
                    if 'individual' in entry.name:
                        os.remove(entry)

            for i in range(plane_no):
                _frametimes_df = frametimes_df.iloc[i:]
                subdf = _frametimes_df.iloc[::plane_no, :]
                subdf.reset_index(drop=True, inplace=True)
                if i == int(k):
                    saving = Path(fld).joinpath(f"frametimes.h5")
                    subdf.to_hdf(
                        saving, key="frames", mode="a"
                    )  # saving frametimes into each specific folder
                    if pstim_file:  # if pstim output exists, save into each folder
                        shutil.copy(pstim_path, Path(fld).joinpath(f"pstim_output.txt"))

    # move over the original images into a new folder
    moveto_folder = Path(folder_path).joinpath("bruker_images")
    if not os.path.exists(moveto_folder):
        os.mkdir(moveto_folder)

    # moving xml and env files into output plane folders
    move_xml_files(folder_path) 

    # removing all the tif files that were made
    with os.scandir(folder_path) as entries:
        for entry in entries:
            if testkey in entry.name and 'tif' in entry.name:
                new_location = moveto_folder.joinpath(entry.name)
                if os.path.exists(new_location):
                    if safe:
                        print("file already found at this location")
                    else:
                        os.remove(new_location)
                shutil.move(entry, new_location)
                
    return print('done')

def addSecs(tm, secs):
    '''
    Add seconds to datetime values

    tm = datetime value that needs to be changed
    secs = number of seconds you want to add to the tm value
    '''
    fulldate = dt(100, 1, 1, tm.hour, tm.minute, tm.second, tm.microsecond)
    fulldate = fulldate + timedelta(seconds=secs)
    return fulldate.time()

def addHours(tm, hrs):
    '''
    Add hours to datetime values
    
    tm = datetime value that needs to be changed
    hrs = number of hours you want to add to the tm value
    '''
    fulldate = dt(100, 1, 1, tm.hour, tm.minute, tm.second, tm.microsecond)
    fulldate = fulldate + timedelta(hours=hrs)
    return fulldate.time()

def move_xml_files(folder_path):
    '''
    Moving xml and env files into output plane folders
    folder_path = the master data folder path that contains the xml files and output folders
    '''
    voltage_path = None
    ps_xml_path = None  

    with os.scandir(folder_path) as entries:
        for entry in entries:
            if 'companion' in entry.name:
                pass
            elif entry.name.endswith(".xml") and "MarkPoints" not in entry.name and "Voltage" not in entry.name:
                info_xml_path = Path(entry.path)
            elif 'txt' in entry.name:
                pstim_path = Path(entry.path)
            elif 'Voltage' in entry.name and entry.name.endswith(".csv"):
                voltage_path = Path(entry.path)
            elif entry.name.endswith("xml") and 'MarkPoints' in entry.name:
                ps_xml_path = Path(entry.path)
            elif entry.name.endswith("env"):
                info_env_path = Path(entry.path)

    with os.scandir(Path(folder_path).joinpath('output_folders')) as entries:
        for entry in entries:
            fld = Path(entry.path)
            shutil.copy(info_xml_path, Path(fld).joinpath(Path(info_xml_path).name))
            shutil.copy(info_env_path, Path(fld).joinpath(Path(info_env_path).name))
            if ps_xml_path:
                shutil.copy(ps_xml_path, Path(fld).joinpath(Path(ps_xml_path).name))
            if voltage_path:
                shutil.copy(voltage_path, Path(fld).joinpath(Path(voltage_path).name))      
    return print('done')

def get_micronstopixels_scale(info_xml_file_path):
    '''
    Getting the scale microns per pixel from the xml file
    info_xml_file_path = info xml file path
    Will return pixel_size = microns/pixel
    '''
    with open(info_xml_file_path, "r") as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            if "micronsPerPixel" in line:
                pixel_size = float(str(lines[i + 1]).split('"')[-2])         
    return pixel_size

def get_micronstopixels_scale2(info_xml_file_path):
    '''
    Getting the scale microns per pixel from the xml file
    info_xml_file_path = info xml file path
    Will return pixel_size = microns/pixel
    '''
    with open(info_xml_file_path, "r") as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            if "micronsPerPixel" in line:
                pixel_size = float(str(lines[i + 1]).split('"')[-2])        
    return pixel_size

def get_pixelsperline(info_xml_file_path):
    '''
    Getting the number of pixels per line from the xml file
    info_xml_file_path = info xml file path
    Will return pixels_per_line = number of pixels per line
    '''
    with open(info_xml_file_path, "r") as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            if "linesPerFrame" in line:
                pixels_per_line = int(str(lines[i]).split('"')[-2])       
    return pixels_per_line

def read_xml_to_root(xml_file_path):
    '''
    Read xml file into a root directory
    '''
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    return root

def read_xml_to_str(xml_file_path):
    '''
    Read a xml file into a data string
    '''
    with open(xml_file_path, "r") as f:
        data = f.read()
        
    return data

def get_zstep_vals(info_xml_file, etl = True):
    '''
    Getting the z-step values from the information xml file
    Important for getting the stimulated plane number
    '''
    data_str = read_xml_to_str(info_xml_file)

    all_z_steps = []
    all_etl_steps = []
    for j, i in enumerate(data_str.split("\n")):
        if "page" in i:
            page = int(i.split('page=')[1].split('"')[1])
            all_z_steps.append(page)
        if 'ETL' in i:
            etl_step = float(i.split('value=')[1].split('"')[1])
            all_etl_steps.append(etl_step)
        if "Z Focus" in i:
            Z_focus_start = float(i.split('value=')[1].split('"')[1])
        if "micronsPerPixel" in i:
            next_line = data_str.split("\n")[j+1]
            pixel_size = float(next_line.split('value=')[1].split('"')[1])
            z_step_size = float(data_str.split("\n")[j+3].split('value=')[1].split('"')[1])

    num_z_steps = max(all_z_steps)
    zstep_vals = [Z_focus_start + n*z_step_size for n in range(num_z_steps)]

    if etl:
        zstep_vals = np.unique(all_etl_steps)
        zstep_vals = arrutils.filter_list(zstep_vals, 2)
        zstep_vals, _ = arrutils.fix_equal_interval(zstep_vals)

    return zstep_vals

def concatenate_datasets(experiment_folders, new_directory, full_duration_per_stim = None):
    '''
    Concatenate datasets from multiple experiments into one folder, most helpful for automated gui experiments
    Moves over rotated images, movement corrected images, bad_frames (if present), and frametimes into the new directory
    experiment_folders = list of folder paths that contain the individual experiments
    new_directory = the new directory path that will contain the concatenated data
    full_duration_per_stim = the full duration of the photostimulation event in ms
    '''

    plane_count = sum(os.path.isdir(os.path.join(Path(experiment_folders[0]).joinpath('output_folders'), item)) 
                  for item in os.listdir(Path(experiment_folders[0]).joinpath('output_folders')))
        
    if not new_directory.exists():
        new_directory.mkdir()
    new_directory2 = Path(new_directory).joinpath("output_folders")
    if not new_directory2.exists():
        new_directory2.mkdir()
    if full_duration_per_stim == None:  
        full_duration_per_stim = 100 # hard coded for now, will need to change if length of photostimulation is longer than 100 ms

    for p in range(plane_count):
        save_fld = Path(new_directory2).joinpath(f'plane_{p}')
        if not save_fld.exists():
            save_fld.mkdir()
        plane = f'plane_{p}'
        rotated_imgs = []
        movement_corrected_imgs = []
        frametimes_lst = []
        bad_frames_lst = []
        for fld in experiment_folders:
            frametimes_lst.append(pd.read_hdf(Path(fld).joinpath(f'output_folders/{plane}/frametimes.h5')))
            bad_frames_lst.append(np.load(Path(fld).joinpath(f'output_folders/{plane}/bad_frames.npy')))
            for file_path in glob.glob(os.path.join(Path(fld).joinpath(f'output_folders/{plane}'), "*.tif")):
                if 'rotated' in os.path.basename(file_path):
                    rotated_imgs.append(file_path)
                if 'movement_corr' in os.path.basename(file_path):
                    movement_corrected_imgs.append(file_path)

    # images
    m1 = cm.load_movie_chain(rotated_imgs)
    m1.save(os.path.join(save_fld,'img_rotated.tif'))
    m2 = cm.load_movie_chain(movement_corrected_imgs)
    m2.save(os.path.join(save_fld,'movement_corr_img.tif'))

    # frametimes
    pd.concat(frametimes_lst).reset_index(drop = True).to_hdf(Path(save_fld).joinpath('frametimes.h5'), key = 'frametimes')

    # bad frames
    len_expt = len(frametimes_lst[0])
    new_bad_frames_lst = []
    for a, b in enumerate(bad_frames_lst):
        img_hz = fishy.BaseFish.hzReturner(frametimes_lst[a])
        interval = round(full_duration_per_stim/1000 * img_hz)
        if interval == 0:
            interval = 1
        filtered_arr = arrutils.filter_list(lst = b, interval = interval)
        if filtered_arr[0] != 0:
            filtered_arr.insert(0, 0)   
        arr = [l + a*len_expt for l in filtered_arr]
        new_bad_frames_lst.append(arr)
        
    complete_bad_frames_arr = np.concatenate(new_bad_frames_lst)
    np.save(Path(save_fld).joinpath('bad_frames.npy'), complete_bad_frames_arr)

def collect_img_array_from_individual_volumes(folder, n = 30, plane_idx = 0, channel = 'Ch2'):
    '''
    Collect all img files in a folder, that are individual tifs & volumes
    :param folder: folder path that contains the individual images
    :param n: number of images to collect (default = 30)
    :param plane_idx: index of the plane to collect (default = 0)
    :param channel: channel to collect (default = 'Ch2', or can be "Ch1")
    :return: list of all the images read in the folder
    '''
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="tifffile")

    stack_img_lst = []
    _n = 0
    for file in os.listdir(folder):
        if file.endswith('.ome.tif') and channel in file:
            stack_img_lst.append(imread(folder / file))
            _n = _n + 1
            if _n == n:
                break
    full_stack_img_array = np.array(stack_img_lst)
    if plane_idx == None:
        _stack_img_array = full_stack_img_array
    else:
        _stack_img_array = full_stack_img_array[:,plane_idx,:,:]
    stack_img_array = np.nanmean(_stack_img_array[:n,:,:], axis = 0)

    return stack_img_array

def find_drift_between_images(img1_path, img2_path, ref_stack_array_path, info_xml_path,
                              alignment_mode='enhanced', ref_stack_z_microns=1):
    '''
    Find drift between two images, in relation to a reference stack
    img1 and img2 should be tif files
    ref_stack_array_path is a numpy array (saved after running xcorrelation on the scope)
    alignment_mode - the correlation mode that you want to run for the alignment algorithm

    returns: results dictionary with the drift in um for x, y, z between the images (in relation to the reference stack)
    '''
    import matplotlib.pyplot as plt
    sys.path.append(r'C:\Users\Kaitlyn\PyCharmProjects\imaging\scopeslip')
    import crossCorrelation

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="tifffile")

    # 1 - read in images & array, make averages, find microns to pixel conversion
    try: # if you have tif files
        full_img1 = imread(img1_path)
        img1 = np.nanmean(full_img1[50:550, :, :], axis=0) # having at least 300 images should work best for xcorr
    except: # if you have numpy arrays
        img1 = collect_img_array_from_individual_volumes(img1_path, n = 400, plane_idx = 0)
        img1 = np.rot90(img1)
    try:   
        full_img2 = imread(img2_path)
        img2 = np.nanmean(full_img2[50:550, :, :], axis=0)
    except:    
        img2 = collect_img_array_from_individual_volumes(img2_path, n = 400, plane_idx = 0)
        img2 = np.rot90(img2)

    if 'npy' in ref_stack_array_path.name:
        ref_stack_arr = np.load(ref_stack_array_path)
        ref_stack_arr = np.array([np.rot90(plane) for plane in ref_stack_arr])
    else:
        ref_stack_arr = collect_img_array_from_individual_volumes(ref_stack_array_path, plane_idx = None)
        ref_stack_arr = np.array([np.rot90(plane) for plane in ref_stack_arr])

    um_to_px = get_micronstopixels_scale(info_xml_path)

    # 2 - run cross correlation against the reference stack for each image
    img1_xcorr = crossCorrelation.CrossCorrelationAlignment(target_volume=ref_stack_arr, image=img1,
                                                            correlation_mode=alignment_mode)
    img2_xcorr = crossCorrelation.CrossCorrelationAlignment(target_volume=ref_stack_arr, image=img2,
                                                            correlation_mode=alignment_mode)

    img1_shiftx, img1_shifty, img1_shiftz = img1_xcorr.align(img1_xcorr.image)
    print(img1_shiftx, img1_shifty, img1_shiftz)
    img2_shiftx, img2_shifty, img2_shiftz = img2_xcorr.align(img2_xcorr.image)
    print(img2_shiftx, img2_shifty, img2_shiftz)

    # 3- plot the overlays
    # plot the overlays with the reference stack
    # plot_imgs_with_overlay(img1, ref_stack_arr[img1_shiftz], 'img1', 'img1 ref match')
    # plot_imgs_with_overlay(img2, ref_stack_arr[img2_shiftz], 'img2', 'img2 ref match')

    fig, ax = plot_imgs_with_overlay(img1, img2, 'img1', 'img2', boost_red = False, grayscale = False)
    plt.show()

    # 4 - determine the drift (in pixels and microns)
    x_diff = img1_shiftx - img2_shiftx
    y_diff = img1_shifty - img2_shifty
    z_diff = img1_shiftz - img2_shiftz

    y_diff = -y_diff # flipped since origin of images is at top left (important for plotting/matching cell ids)

    # i know that the ref stack is 1 um, so the z_diff is the same in um
    z_diff_um = z_diff * ref_stack_z_microns
    x_diff_um = '{:.2f}'.format(x_diff * um_to_px)
    y_diff_um = '{:.2f}'.format(y_diff * um_to_px)

    results_um = {'x_drift_um': x_diff_um,
               'y_drift_um': y_diff_um,
               'z_drift_um': z_diff_um}
    results_px = {'x_drift_px': '{:.2f}'.format(x_diff),
                  'y_drift_px': '{:.2f}'.format(y_diff),
                  'z_drift_px': '{:.2f}'.format(z_diff)}

    return results_um, results_px


def plot_imgs_with_overlay(imgA, imgB, titleA='imageA', titleB='imageB', boost_red = True, grayscale = False):

    import matplotlib.pyplot as plt
    if grayscale == False: # plotting red and green channels
        imgA_norm = (imgA - imgA.min()) / (imgA.max() - imgA.min())  # Normalize
        imgB_norm = (imgB - imgB.min()) / (imgB.max() - imgB.min())
        imgA_norm = imgA_norm ** 0.85  # Slight gamma correction
        imgB_norm = imgB_norm ** 0.6

        rgb = np.zeros((imgA.shape[0], imgA.shape[1], 3), dtype=float)
        if boost_red:
            rgb[..., 0] = np.clip(imgB_norm * 1.6, 0, 1)  # Red channel boosted
        else:
            rgb[..., 0] = imgB_norm
        rgb[..., 1] = imgA_norm  # Green channel
        rgb = np.clip(rgb, 0, 1)
    else: # plotting red and grayscale channels
        rgb = make_grayscale_and_red_overlay(imgA, imgB)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(imgA, cmap='gray', vmax=np.percentile(imgA, 99))
    ax[0].set_title(titleA)
    ax[1].imshow(imgB, cmap='gray', vmax=np.percentile(imgB, 99))
    ax[1].set_title(titleB)
    ax[2].imshow(rgb)
    ax[2].set_title("Red = Image 2, Green = Image 1")
    [a.axis('off') for a in ax.flatten()]
    plt.tight_layout()

    return fig, ax  # return so caller can add to them

def make_grayscale_and_red_overlay(imgA, imgB):
    '''
    Making a grayscale and red overlay image
    :param imgA: image 1 (gray image)
    :param imgB: image 2 (red image)
    :return: the overlap
    '''

    imgA_norm = (imgA - imgA.min()) / (imgA.max() - imgA.min())  # Normalize
    imgB_norm = (imgB - imgB.min()) / (imgB.max() - imgB.min())
    imgA_norm = imgA_norm ** 0.8  # Slight gamma correction
    imgB_norm = imgB_norm ** 0.7

    rgb = np.stack([imgA_norm, imgA_norm, imgA_norm], axis=-1)
    red_overlay = np.clip(imgB_norm * 1.3, 0, 1)
    rgb[..., 0] = np.clip(rgb[..., 0] + red_overlay, 0, 1)  # add to red channel

    return rgb
