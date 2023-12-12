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



def bruker_img_organization_PV7(folder_path, testkey, safe=False):
    # PV 5.7 software bruker organization function
    keyset = set()
    # get images all together
    with os.scandir(folder_path) as entries:
        for entry in entries:
            if testkey and 'tif' in entry.name:
                keyset.add(
                    entry.name.split(".ome")[0].split("_")[-1]
                )  # make a key for each plane

    image_path_dict = {k: {} for k in keyset}
    # image paths go in this dict for each plane
    for k in image_path_dict.keys():
        with os.scandir(folder_path) as entries:
            for entry in entries:
                if k in entry.name and testkey in entry.name and 'tif' in entry.name:
                    entry_set = int(str(entry.path).split("Cycle")[1].split("_")[0])
                    image_path_dict[k][entry_set] = entry.path

    images = {k: [] for k in image_path_dict.keys()}  # dictionary for each image
    for k in image_path_dict.keys():
        img_paths = image_path_dict[k]
        for n in range(len(img_paths.keys())):
            img = imread(img_paths[n + 1])
            if len(img.shape) > 2:
                img = img[int(k)]  # if we load a volume from the master unvolume it
            images[k].append(img)

    # making the frametimes file
    with os.scandir(folder_path) as entries:
        for entry in entries:
            if entry.name.endswith(".xml") and "MarkPoints" in entry.name:
                slm_xml_path = Path(entry.path)
            elif entry.name.endswith(".xml") and "MarkPoints" not in entry.name:
                xml_path = Path(entry.path)

    with open(xml_path, "r") as f:
        data = f.read()

    times = []
    for i in data.split("\n"):
        if "time" in i:
            start = [i][0].split("time=")[1].split('"')[1]
            start_dt = dt.strptime(start[:-1], "%H:%M:%S.%f").time()

        elif "absoluteTime" in i:
            added_secs = [i.split("absoluteTime=")[1].split('"')[1]][0]
            frame_dt = addSecs(start_dt, float(added_secs))
            times.append(frame_dt)
    frametimes_df = pd.DataFrame(times)
    frametimes_df.rename({0: "time"}, axis=1, inplace=True)

    save_path = Path(folder_path).joinpath(
        "master_frametimes.h5"
    )  # saving master frametimes file
    frametimes_df.to_hdf(save_path, key="frames", mode="a")

    # reorganizing images into different folders
    new_output = Path(folder_path).joinpath(
        "output_folders"
    )  # new folder to save the output tiffs
    if not os.path.exists(new_output):
        os.mkdir(new_output)

    keylist = list(keyset)
    keylist.sort()
    z_len = int(keylist[-1])  # getting number of planes

    for k, v in images.items():
        # fullstack = np.array(v)
        fld = Path(new_output).joinpath(f"plane_{k}")
        if not os.path.exists(fld):
            os.mkdir(fld)
        for i, individual in enumerate(v):
            imwrite(
                fld.joinpath(f"individual_img_{k}_{i}.tif"), individual
            )  # saving new tifs, each one is a time series for each plane
        fls = glob.glob(os.path.join(fld,'*.tif'))  #  change tif to the extension you need
        fls.sort()  # make sure your files are sorted alphanumerically
        m = cm.load_movie_chain(fls)
        m.save(os.path.join(fld,f'img_stack_{k}.tif'))
        with os.scandir(fld) as entries:
            for entry in entries:
                if 'individual' in entry.name:
                    os.remove(entry)

        for i in range(len(frametimes_df[0 : z_len - 1])):
            _frametimes_df = frametimes_df.iloc[i:]
            subdf = _frametimes_df.iloc[::z_len, :]
            subdf.reset_index(drop=True, inplace=True)
            if i == int(k):
                saving = Path(fld).joinpath(f"frametimes.h5")
                subdf.to_hdf(
                    saving, key="frames", mode="a"
                )  # saving frametimes into each specific folder
            else:
                pass

            # will also want to copy over pstim file into each folder too

    # move over the original images into a new folder
    moveto_folder = Path(folder_path).joinpath("bruker_images")
    if not os.path.exists(moveto_folder):
        os.mkdir(moveto_folder)

    with os.scandir(folder_path) as entries:
        for entry in entries:
            if testkey and 'tif' in entry.name:
                new_location = moveto_folder.joinpath(entry.name)
                if os.path.exists(new_location):
                    if safe:
                        print("file already found at this location")
                        return
                    else:
                        os.remove(new_location)
                shutil.move(entry, new_location)


def bruker_img_organization_PV8(folder_path, testkey = 'Cycle', safe=False, single_plane=False, pstim_file = True):

    # PV 5.8 software bruker organization function, volume stack, single plane if True

    keyset = set()
    tiff_files_li = []

    with os.scandir(folder_path) as entries:
        for entry in entries:
            if 'companion' in entry.name:
                pass
            elif testkey in entry.name and 'tif' in entry.name:
                keyset.add(
                    entry.name.split("Cycle")[1].split("_")[0]
                )  # make a key for each volume or if single plane, each plane in the t-series
                tiff_files_li.append(Path(entry.path))
            elif entry.name.endswith(".xml") and "MarkPoints" not in entry.name:
                xml_path = Path(entry.path)
            elif 'txt' in entry.name and pstim_file:
                pstim_path = Path(entry.path)

    # making new output folders
    new_output = Path(folder_path).joinpath(
        "output_folders"
    )  # new folder to save the output tiffs
    if not os.path.exists(new_output):
        os.mkdir(new_output)

    if single_plane == True:

        fls = glob.glob(os.path.join(folder_path,'*.tif'))  #  change tif to the extension you need
        fls.sort()  # make sure your files are sorted alphanumerically
        m = cm.load_movie_chain(fls)
        save_fld = Path(new_output).joinpath(f"single_plane")
        if not os.path.exists(save_fld):
            os.mkdir(save_fld)
        m.save(os.path.join(save_fld,'img_stack.tif'))
        # making the frametimes file
        with open(xml_path, "r") as f:
            data = f.read()

        times = []
        for i in data.split("\n"):
            if "time" in i:
                start = [i][0].split("time=")[1].split('"')[1]
                start_dt = dt.strptime(start[:-1], "%H:%M:%S.%f").time()

            elif "absoluteTime" in i:
                added_secs = [i.split("absoluteTime=")[1].split('"')[1]][0]
                frame_dt = addSecs(start_dt, float(added_secs))
                times.append(frame_dt)
        frametimes_df = pd.DataFrame(times)
        frametimes_df.rename({0: "time"}, axis=1, inplace=True)

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

    # reorganizing images, frametimes, pstim files into different folders
        tree = ET.parse(xml_path)
        root = tree.getroot()

        times = []
        for start in root.iter('PVScan'):
            begin_time = start.attrib['date'].split(' ')[1]
            start_dt = dt.strptime(begin_time, "%H:%M:%S").time()
            hour = int(start.attrib['date'].split(' ')[1].split(':')[0])
            if (start.attrib['date'].split(' ')[2] == 'PM') & (hour != 12): # convert to 24hr time for stim files
                start_dt = addHours(start_dt, float(12))

        for frame in root.iter('Frame'):
            added_secs = frame.attrib['absoluteTime']
            frame_dt = addSecs(start_dt, float(added_secs))
            times.append(frame_dt)
        frametimes_df = pd.DataFrame(times)
        frametimes_df.rename({0: "time"}, axis=1, inplace=True)
        save_path = Path(folder_path).joinpath(
            "master_frametimes.h5"
        )
        frametimes_df.to_hdf(save_path, key="frames", mode="a")  # saving master frametimes file
    
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
    fulldate = dt(100, 1, 1, tm.hour, tm.minute, tm.second, tm.microsecond)
    fulldate = fulldate + timedelta(seconds=secs)
    return fulldate.time()

def addHours(tm, hrs):
    fulldate = dt(100, 1, 1, tm.hour, tm.minute, tm.second, tm.microsecond)
    fulldate = fulldate + timedelta(hours=hrs)
    return fulldate.time()

def find_baseline_frames(folder_path, volume = False):
    original_imgs = Path(folder_path).joinpath("bruker_images")
    with os.scandir(original_imgs) as entries:
        for entry in entries:
            if 'Cycle' in entry.name and 'tif' in entry.name:
                baseline_img = imread(entry.path)
                break #want the first tif file here

    baseline_frames = baseline_img.shape[0]

    if volume == True:
        with os.scandir(folder_path) as entries:
            for entry in entries:
                if 'MarkPoints' in entry.name:
                    mp_xml_file_name = str(entry.name)
        baseline_frames = int(mp_xml_file_name.split('Cycle')[1].split('_')[0])
    
    return baseline_frames

# I have updated functions in the fishy py file

# def find_photostim_frames(some_baseFish, threshold = 0.8):

#     """
#     :param folderpath: path to folder that contains image and xml file with mark points
#             buffer: needed to have a few more frames on the end of this next for loop to accurately capture all frames that are bad
#     :return:
#     """
#     from PIL import Image
#     import math
#     from fishy import BaseFish

#     img_path = some_baseFish.data_paths['move_corrected_image']

#     with os.scandir(Path(img_path).parents[2]) as entries:
#         for entry in entries:
#             if 'MarkPoints' in entry.name:
#                 xml_file = Path(entry.path)

#     with open(xml_file, "r") as f:
#         data = f.read()

#     for i in data.split("\n"):
#         if "InitialDelay" in i:
#             initial_delay_ms = int([i][0].split("InitialDelay=")[1].split('"')[1])
#             interpointdelay_ms = int([i][0].split("InterPointDelay=")[1].split('"')[1])
#             duration_ms = int([i][0].split("Duration=")[1].split('"')[1])
#             indices = int([i][0].split("Indices=")[1].split('"')[1].split('-')[1])
#         elif "Repetitions" in i:
#             no_repetitions = int([i][0].split("Repetitions=")[1].split('"')[1])
#         elif "Iterations" in i:
#             no_iterations = int([i][0].split("Iterations=")[1].split('"')[1])
#             iteration_delay_ms = int(float([i][0].split("IterationDelay=")[1].split('"')[1]))

#     full_duration_ms = initial_delay_ms + ((no_repetitions * duration_ms + interpointdelay_ms) * indices)
#     img_hz = BaseFish.hzReturner(some_baseFish.frametimes_df)
#     full_duration_frames = math.ceil((full_duration_ms / 1000) * img_hz)
#     iteration_delay_frames = math.ceil((iteration_delay_ms / 1000) * img_hz)

#     # IMAGE FRAMES
#     img = Image.open(img_path)
#     myArray = np.zeros((np.shape(img)[0], (np.shape(img)[1]), img.n_frames))

#     # read each frame into the array
#     for i in range(img.n_frames):
#         img.seek(i)
#         myArray[:, :, i] = img

#     # calculate a mean brightness trace
#     brightnessArray = myArray.mean(axis=(0, 1))

#     # use large changes in brightness (due to PMT shutter closure for laser) to do timing
#     diffArray = np.diff(brightnessArray)

#     # identify frames brightness 2 std below the mean of the whole array
#     ids = list(np.squeeze(np.where(diffArray > threshold)))
#     print(ids)
#     n=0
#     while n <= 3: # have to rerun this a few times maybe
#         for x, y in enumerate(ids):
#             if (y != ids[-1]) and ((ids[x + 1] - ids[x]) < (iteration_delay_frames-10)):  # if the next value is less than when the next iteration would be, then we want to keep the larger frame number
#                 ids.remove(ids[x + 1])
#                 print(ids)
#             elif y > 1100: #after all the iterations
#                 ids.remove(y)
#             elif y < 400: #baseline value
#                 ids.remove(y)
#             elif x == -1:
#                 if (ids[x] - ids[x-1]) < 5:
#                     ids.remove(ids[x])
#         if len(ids) > no_iterations:
#             n =+ 1
#         else:
#             break
#     print(ids)

#     frames_lst = []
#     for i in ids:
#         frames = np.arange(i, i + full_duration_frames + 1)
#         frames_lst.append(frames)
#     print(frames_lst)

#     badframes = [val for sublist in frames_lst for val in sublist]
#     badframes_arr = np.array(badframes)

#     photostim_events = pd.DataFrame(index=range(no_iterations), columns = ['frames'])
#     photostim_events['frames'] = frames_lst

#     np.save(Path(img_path).parents[0].joinpath('bad_frames.npy'), badframes_arr)  # save badframes
#     print('saved bad_frames.npy')

#     return badframes_arr, photostim_events

# def find_photostimulated_cell(some_baseFish, angle = 90, rois = [6], proximity = 8):
#     from fishy import BaseFish
#     import utilities

#     img_path = some_baseFish.data_paths['move_corrected_image'].parents[0].joinpath("original_image/img_stack.tif") #original image
#     BaseFish.load_suite2p(some_baseFish) #make sure to load in suite2p

#     with os.scandir(Path(img_path).parents[3]) as entries:
#         for entry in entries:
#             if 'MarkPoints' in entry.name:
#                 xml_file = Path(entry.path)

#     img = imread(img_path)
#     # load the photostim xml file
#     with open(xml_file, "r") as f:
#         data = f.read()

#     coors_lst = []
#     rotated_coors_lst = []
#     # read the x and y percentages
#     for r in rois:
#         for i in data.split("\n"):
#             if f'Point Index="{r}"' in i:
#                 myX = i.split('X')[1].split('"')[1]
#                 myY = i.split('Y')[1].split('"')[1]

#                 xCoord = img.shape[1] * float(myX)
#                 yCoord = img.shape[2] * float(myY)
#                 coors = np.array([xCoord, yCoord])
#                 coors_lst.append(coors)
#     if angle == 90:
#         for c in coors_lst:
#             stim_x = c[1]
#             stim_y = img.shape[2] - c[0]
#             _coors = np.array([stim_x, stim_y])
#             rotated_coors_lst.append(_coors)

#     df_data = {'original_coors': coors_lst, 'rotated_coors': rotated_coors_lst}
#     df = pd.DataFrame(df_data)

#     #finding cells based on proximity (in um) to the stimulation site
#     cell_num_lst = []
#     for roi_no, coor in enumerate(df.rotated_coors):
#         x_val, y_val = coor
#         cell_num = some_baseFish.return_cells_by_location(ymin=round(x_val) - proximity, ymax=round(x_val) + proximity,
#                                                      xmin=round(y_val) - proximity, xmax=round(y_val) + proximity)
#         cell_num_lst.append(cell_num)

#     df['cell_proximity'] = cell_num_lst
#     df['stim_cell'] = np.nan

#     # choosing stimulated cell to be the most responsive cell overall
#     zdiffcells = [utilities.arrutils.zdiffcell(z) for z in some_baseFish.f_cells]
#     for ind, m in enumerate(df.cell_proximity.values):
#         if m.shape[0] == 1:
#             df['stim_cell'].iloc[ind] = m
#         elif m.shape[0] > 1:
#             med_zdiff_lst = []
#             for x in range(m.shape[0]):
#                 med_zdiff_lst.append(abs(np.median(zdiffcells[x])))  # collective absolute median activity
#             max_value = max(med_zdiff_lst)
#             max_index = med_zdiff_lst.index(max_value)
#             df['stim_cell'].iloc[ind] = m[max_index]
#         else:
#             df['stim_cell'].iloc[ind] = np.nan

#     return df


# suite2p bruker ops dict
def run_suite2p_PS(somebasefish, input_tau = 1.5):
    from suite2p import run_s2p, default_ops
    from fishy import BaseFish

    # make sure bad frames exists first
    bad_frames_path = somebasefish.data_paths["move_corrected_image"].parents[0].joinpath('bad_frames.npy')
    if os.path.isfile(bad_frames_path) == False:
        find_photostim_frames(somebasefish, threshold = 0.8)

    bruker_s2p_ops = default_ops()
    bruker_s2p_ops['data_path'] = [somebasefish.data_paths["move_corrected_image"].parents[0].as_posix()]
    bruker_s2p_ops['save_path'] = somebasefish.data_paths["move_corrected_image"].parents[0].as_posix()
    bruker_s2p_ops['tau'] = input_tau #gcamp6s
    bruker_s2p_ops['fs'] = BaseFish.hzReturner(somebasefish.frametimes_df)
    bruker_s2p_ops['preclassify'] = 0.15
    bruker_s2p_ops['block_size'] = [32, 32]
    bruker_s2p_ops['allow_overlap'] = True
    bruker_s2p_ops['tiff_list']: [somebasefish.data_paths["move_corrected_image"]]
    bruker_s2p_ops['two_step_registration'] = True
    bruker_s2p_ops['keep_movie_raw'] = True

    db = {}
    run_s2p(ops=bruker_s2p_ops, db=db)



