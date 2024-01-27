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
        frame_signal = np.array(volt_csv[' Input 1'])
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
            elif entry.name.endswith(".xml") and "MarkPoints" in entry.name:
                ps_xml_path = Path(entry.path)
            elif entry.name.endswith(".env"):
                env_path = Path(entry.path)
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
    frametimes_df = get_frametimes(info_xml_path, voltage_path)

    if single_plane == True:

        fls = glob.glob(os.path.join(folder_path,'*.tif'))  #  change tif to the extension you need
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

        # image paths go in this dict for each volume
        for k in volume_path_dict.keys():
            with os.scandir(folder_path) as entries:
                for entry in entries:
                    if f'Cycle{k}' in entry.name and 'tif' in entry.name:
                        volume_path_dict[k] = entry.path

        # number of planes gotten from the first image
        plane_no = imread(volume_path_dict[k]).shape[0]
        planes_dict = {k: [] for k in range(plane_no)}  # dictionary for each plane

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
                    
                    # moves over all the xml files into each output folder
                    shutil.copy(ps_xml_path, Path(fld).joinpath("MarkPoints.xml"))
                    shutil.copy(info_xml_path, Path(fld).joinpath("information.xml"))
                    shutil.copy(env_path, Path(fld).joinpath("env_file.env"))

    # move over the original images into a new folder
    moveto_folder = Path(folder_path).joinpath("bruker_images")
    if not os.path.exists(moveto_folder):
        os.mkdir(moveto_folder)

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
    A BaseFish will be called so that it will use the data_paths dictionary
    '''
    from fishy import BaseFish

    brukerfish = BaseFish(folder_path = folder_path, frametimes_key = 'frametimes')
    with os.scandir(Path(brukerfish.folder_path.joinpath('output_folders'))) as entries:
        for entry in entries:
            fld = Path(entry.path)
            shutil.copy(brukerfish.data_paths['info_xml'], Path(fld).joinpath(Path(brukerfish.data_paths['info_xml']).name))
            shutil.copy(brukerfish.data_paths['ps_xml'], Path(fld).joinpath(Path(brukerfish.data_paths['ps_xml']).name))
            shutil.copy(brukerfish.data_paths['info_env'], Path(fld).joinpath(Path(brukerfish.data_paths['info_env']).name))
            print('xml and env files are copied')

def get_micronstopixels_scale(somebaseFish):
    '''
    If you have a Bruker fish with all the xml files moved over, you can get the scale microns per pixel from the xml file
    somebasefish = a Bruker based BaseFish
    Will return pixel_size = microns/pixel
    '''
    with open(somebaseFish.data_paths['info_xml'], "r") as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            if "micronsPerPixel" in line:
                pixel_size = float(str(lines[i + 1]).split('"')[-2])
                
    return pixel_size

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
