import os
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
from tiffile import imread, imwrite
from datetime import datetime as dt, timedelta
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


def bruker_img_organization_PV8(folder_path, testkey = 'Cycle', safe=False, single_plane=False):

    # PV 5.8 software bruker organization function, volume stack, single plane if True

    keyset = set()
    pstim_path = None

    # get images all together
    with os.scandir(folder_path) as entries:
        for entry in entries:
            if 'Cycle' and 'tif' in entry.name:
                keyset.add(
                    entry.name.split("Cycle")[1].split("_")[0]
                )  # make a key for each plane
            elif entry.name.endswith(".xml") and "MarkPoints" in entry.name:
                stim_xml_path = Path(entry.path)
            elif entry.name.endswith(".xml") and "MarkPoints" not in entry.name:
                xml_path = Path(entry.path)
            elif 'pstim' in entry.name:
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
    else:
        volume_path_dict = {k: {} for k in keyset}

        # image paths go in this dict for each volume
        for k in volume_path_dict.keys():
            with os.scandir(folder_path) as entries:
                for entry in entries:
                    if k in entry.name and testkey in entry.name and 'tif' in entry.name:
                        volume_path_dict[k] = entry.path

        # number of planes gotten from the first image
        plane_no = imread(volume_path_dict[k]).shape[0]
        planes_dict = {k: [] for k in range(plane_no)}  # dictionary for each plane

        for k in volume_path_dict.keys():
            vol_img = volume_path_dict[k]
            for n in range(len(planes_dict.keys())):
                img = imread(vol_img)[n]
                planes_dict[n].append(img)  # each plane_dict key is a different plane, with every image in a list

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

    save_path = Path(folder_path).joinpath(
        "master_frametimes.h5"
    )
    frametimes_df.to_hdf(save_path, key="frames", mode="a") # saving master frametimes file

    # reorganizing images, frametimes, pstim files into different folders

    # getting plane stacks into specific folders
    if single_plane == True:
        save_path = Path(save_fld).joinpath("frametimes.h5")
        frametimes_df.to_hdf(save_path, key="frames", mode="a") # saving master frametimes file into single plane
        if pstim_path in locals():  # if pstim output exists, save into each folder
            shutil.copy(pstim_path, Path(save_fld).joinpath(f"pstim_output.txt"))
        else:
            print('no pstim file')
    else:
        for k, v in planes_dict.items():
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

        for i in range(plane_no):
            _frametimes_df = frametimes_df.iloc[i:]
            subdf = _frametimes_df.iloc[::plane_no, :]
            subdf.reset_index(drop=True, inplace=True)
            if i == int(k):
                saving = Path(fld).joinpath(f"frametimes.h5")
                subdf.to_hdf(
                    saving, key="frames", mode="a"
                )  # saving frametimes into each specific folder
                if pstim_path in locals():  # if pstim output exists, save into each folder
                    shutil.copy(pstim_path, Path(fld).joinpath(f"pstim_output.txt"))
            else:
                pass

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

def find_photostim_frames(folderpath, buffer = 2, thresh_int = 20):

    """
    :param folderpath: path to folder that contains image and xml file with mark points
            buffer: needed to have a few more frames on the end of this next for loop to accurately capture all frames that are bad
    :return:
    """
    from PIL import Image

    with os.scandir(folderpath) as entries:
        for entry in entries:
            if 'tif' in entry.name and 'movement_corr' in entry.name:
                img_path = Path(entry.path) # open movement correct image

    # IMAGE FRAMES
    img = Image.open(img_path)
    myArray = np.zeros((np.shape(img)[0], (np.shape(img)[1]), img.n_frames))

    # read each frame into the array
    for i in range(img.n_frames):
        img.seek(i)
        myArray[:, :, i] = img

    # calculate a mean brightness trace
    brightnessArray = myArray.mean(axis=(0, 1))

    # use large changes in brightness (due to PMT shutter closure for laser) to do timing
    diffArray = np.diff(brightnessArray)

    # identify frames with large brightness changes and clean up these arrays
    ids = list(np.squeeze(np.where(diffArray > thresh_int)))
    for x, y in enumerate(ids):
        if (y != ids[-1]) and (ids[x+1] - ids[x] == 1): #if the next value is only 1 away, then we want to keep the larger frame number
            ids.remove(ids[x])

    beginIds = list(np.squeeze(np.where(diffArray < -thresh_int)))
    for m, n in enumerate(beginIds):
        if (n != beginIds[-1]) and (beginIds[m+1] - beginIds[m] == 1): #if the next value is only 1 away, then we want to keep the smaller frame number
            beginIds.remove(beginIds[m+1])

    if len(beginIds) < len(ids): # inserting the first frame to be the start of beginning ids in case the first stimulation event was the first frame
        beginIds.insert(0, 0)

    badframes = []
    photostim_events = [] #list of start and end frames of each photostim event, helpful for plotting
    for i, j in enumerate(beginIds):
        photostim_events.append([j, ids[i]])
        for number in range(beginIds[i], ids[i] + buffer):
            badframes.append(number)

    badframes_arr = np.array(badframes)  # remove duplicates and get array

    np.save(folderpath.joinpath('bad_frames.npy'), badframes_arr)  # save badframes
    print('saved bad_frames.npy')

    return badframes_arr, photostim_events

def find_photostimulated_cell(folderpath, roi = 6):
    import xml.etree.ElementTree as et
    from PIL import Image

    with os.scandir(folderpath) as entries:
        for entry in entries:
            if 'tif' in entry.name:
                img = imread(Path(entry.path)) # open original image
            elif entry.name.endswith(".xml") and "MarkPoints" in entry.name:
                mark_pts_xml = Path(entry.path) # get path for mark points file

    # load the photostim xml file
    with open(mark_pts_xml, "r") as f:
        data = f.read()
    # read the x and y percentages
    for i in data.split("\n"):
        if f'Point Index="{roi}"' in i:
            myX = i.split('X')[1].split('"')[1]
            myY = i.split('Y')[1].split('"')[1]

    xCoord = img.shape[1] * float(myX)
    yCoord = img.shape[2] * float(myY)
    coors = np.array([xCoord, yCoord])

    # need to edit when stimulating more than just one cell

    return coors
#%%
