import os
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
from tifffile import imread, imwrite
from datetime import datetime as dt, timedelta


def bruker_img_organization(folder_path, testkey, safe=False):
    keyset = set()
    # get images all together
    with os.scandir(folder_path) as entries:
        for entry in entries:
            if testkey in entry.name:
                keyset.add(
                    entry.name.split(".ome")[0].split("_")[-1]
                )  # make a key for each plane

    image_path_dict = {k: {} for k in keyset}
    # image paths go in this dict for each plane
    for k in image_path_dict.keys():
        with os.scandir(folder_path) as entries:
            for entry in entries:
                if k in entry.name and testkey in entry.name:
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
            if entry.name.endswith(".xml"):
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
        fullstack = np.array(v)
        new_folder = Path(new_output).joinpath(f"plane_{k}")
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)
        imwrite(
            new_folder.joinpath(f"img_stack_{k}.tif"), fullstack
        )  # saving new tifs, each one is a time series for each plane

        for i in range(len(frametimes_df[0 : z_len - 1])):
            _frametimes_df = frametimes_df.iloc[i:]
            subdf = _frametimes_df.iloc[::z_len, :]
            subdf.reset_index(drop=True, inplace=True)
            if i == int(k):
                saving = Path(new_folder).joinpath(f"frametimes.h5")
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
            if testkey in entry.name:
                new_location = moveto_folder.joinpath(entry.name)
                if os.path.exists(new_location):
                    if safe:
                        print("file already found at this location")
                        return
                    else:
                        os.remove(new_location)
                shutil.move(entry, new_location)


def addSecs(tm, secs):
    fulldate = dt(100, 1, 1, tm.hour, tm.minute, tm.second, tm.microsecond)
    fulldate = fulldate + timedelta(seconds=secs)
    return fulldate.time()


#%%
