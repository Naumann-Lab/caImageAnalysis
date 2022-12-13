"""
the new, latest & greatest 
home to a variety of fishys
"""
import os

import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime as dt
from tifffile import imread
# local imports
import constants
from utilities import pathutils, arrutils


class BaseFish:
    def __init__(self, folder_path, frametimes_key="frametimes"):
        self.folder_path = Path(folder_path)
        self.frametimes_key = frametimes_key

        self.process_filestructure()  # generates self.data_paths
        self.raw_text_frametimes_to_df()  # generates self.frametimes_df

    def process_filestructure(self):
        self.data_paths = {}
        with os.scandir(self.folder_path) as entries:
            for entry in entries:
                if entry.name.endswith(".tif"):
                    if "movement_corr" in entry.name:
                        self.data_paths["move_corrected_image"] = Path(entry.path)
                    else:
                        self.data_paths["image"] = Path(entry.path)
                elif entry.name.endswith(".txt") and "log" in entry.name:
                    self.data_paths["log"] = Path(entry.path)
                elif entry.name.endswith(".txt") and self.frametimes_key in entry.name:
                    self.data_paths["frametimes"] = Path(entry.path)

                if os.path.isdir(entry.path):
                    if entry.name == "suite2p":
                        self.data_paths["suite2p"] = Path(entry.path).joinpath("plane0")

                    if entry.name == "original_image":
                        with os.scandir(entry.path) as imgdiver:
                            for poss_img in imgdiver:
                                if poss_img.name.endswith(".tif"):
                                    self.data_paths["image"] = Path(poss_img.path)

        if "image" in self.data_paths and "move_corrected_image" in self.data_paths:
            if (
                self.data_paths["image"].parents[0]
                == self.data_paths["move_corrected_image"].parents[0]
            ):
                try:
                    pathutils.move_og_image(self.data_paths["image"])
                except:
                    print('failed to move original image out of folder')

    def raw_text_frametimes_to_df(self):
        with open(self.data_paths["frametimes"]) as file:
            contents = file.read()
        parsed = contents.split("\n")

        times = []
        for line in range(len(parsed) - 1):
            times.append(dt.strptime(parsed[line], "%H:%M:%S.%f").time())
        times_df = pd.DataFrame(times)
        times_df.rename({0: "time"}, axis=1, inplace=True)
        self.frametimes_df = times_df

    def load_suite2p(self):
        self.ops = np.load(self.data_paths["suite2p"].joinpath("ops.npy"),
                           allow_pickle=True).item()
        self.iscell = np.load(self.data_paths["suite2p"].joinpath("iscell.npy"),
                              allow_pickle=True)[:, 0].astype(bool)
        self.stats = np.load(self.data_paths["suite2p"].joinpath("stat.npy"),
                             allow_pickle=True)
        self.f_cells = np.load(self.data_paths["suite2p"].joinpath("F.npy"))

    def return_cell_rois(self, cells):
        if isinstance(cells, int):
            cells = [cells]

        rois = []
        for cell in cells:
            ypix = self.stats[cell]["ypix"]
            xpix = self.stats[cell]["xpix"]
            mean_y = int(np.mean(ypix))
            mean_x = int(np.mean(xpix))
            rois.append([mean_y, mean_x])
        return rois

    @staticmethod
    def hzReturner(frametimes):
        increment = 15
        test0 = 0
        test1 = increment
        while True:
            testerBool = (
                frametimes.loc[:, "time"].values[test0].minute
                == frametimes.loc[:, "time"].values[test1].minute
            )
            if testerBool:
                break
            else:
                test0 += increment
                test1 += increment

            if test0 >= len(frametimes):
                increment = increment // 2
                test0 = 0
                test1 = increment

        times = [
            float(str(f.second) + "." + str(f.microsecond))
            for f in frametimes.loc[:, "time"].values[test0:test1]
        ]
        return 1 / np.mean(np.diff(times))

    def __str__(self):
        return "fish"

class VizStimFish(BaseFish):
    def __init__(self, stim_key="stims", stim_fxn=None, stim_fxn_args=None, *args, **kwargs):
        """

        :param stim_key: filename key to find stims in folder
        :param stim_fxn: processes stimuli of interest: returns df with minimum "stim_name" and "time" columns
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        if stim_fxn_args is None:
            stim_fxn_args = []
        self.stim_fxn_args = stim_fxn_args
        self.add_stims(stim_key, stim_fxn)

    def add_stims(self, stim_key, stim_fxn):
        with os.scandir(self.folder_path) as entries:
            for entry in entries:
                if stim_key in entry.name:
                    self.data_paths["stimuli"] = Path(entry.path)

        try:
            _ = self.data_paths["stimuli"]
        except KeyError:
            print("failed to find stimuli")
            return

        if stim_fxn:
            if self.stim_fxn_args:
                self.stimulus_df = stim_fxn(self.data_paths["stimuli"], **self.stim_fxn_args)
            else:
                self.stimulus_df = stim_fxn(self.data_paths["stimuli"])

            self.tag_frames()

    def tag_frames(self):
        frame_matches = [
            self.frametimes_df[
                self.frametimes_df.time >= self.stimulus_df.time.values[i]
            ].index[0]
            for i in range(len(self.stimulus_df))
        ]
        self.stimulus_df.loc[:, "frame"] = frame_matches
        self.stimulus_df.drop(columns="time", inplace=True)


# class TailTrackedFish(VizStimFish):
#     def __init__(self, tail_key, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     def katlyn1(self):
#     def katlyn2(self):

class WorkingFish(VizStimFish):
    '''
    the classic: the every-man's briefcase wielding workhorse
    '''
    def __init__(
        self,
        lightweight=False,
        invert=True,
        stim_offset=5,
        used_offsets=(-10, 14),
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        if "move_corrected_image" not in self.data_paths:
            raise TankError
        self.lightweightmode = lightweight
        self.stim_offset = stim_offset
        self.offsets = used_offsets
        self.invert = invert

        if invert:
            self.stimulus_df.stim_name = self.stimulus_df.stim_name.map(
                constants.invStimDict
            )

        if not lightweight:
            self.image = imread(self.data_paths["move_corrected_image"])
            if invert:
                self.image = self.image[:, :, ::-1]

            self.diff_image = self.make_difference_image()

        self.load_suite2p()
        self.build_stimdicts()

    def make_difference_image(self, selectivityFactor=1.5, brightnessFactor=10):
        if not hasattr(self, 'image'):
            self.image = imread(self.data_paths["move_corrected_image"])
            if self.invert:
                self.image = self.image[:, :, ::-1]

        diff_imgs = {}
        for stimulus_name in constants.monocular_dict.keys():
            stim_occurences = self.stimulus_df[
                self.stimulus_df.stim_name == stimulus_name
            ].frame.values

            stim_diff_imgs = []
            for ind in stim_occurences:
                peak = np.nanmean(self.image[ind : ind + self.offsets[1]], axis=0)
                background = np.nanmean(self.image[ind + self.offsets[0] : ind], axis=0)
                stim_diff_imgs.append(peak - background)

            diff_imgs[stimulus_name] = np.nanmean(stim_diff_imgs, axis=0)

        max_value = np.max([np.max(i) for i in diff_imgs.values()])  # for scaling

        color_images = []
        for stimulus_name, diff_image in diff_imgs.items():
            diff_image[diff_image < 0] = 0

            red_val = diff_image * constants.monocular_dict[stimulus_name][0]
            green_val = diff_image * constants.monocular_dict[stimulus_name][1]
            blue_val = diff_image * constants.monocular_dict[stimulus_name][2]

            red_val /= max_value
            green_val /= max_value
            blue_val /= max_value

            red_val -= red_val.min()
            green_val -= green_val.min()
            blue_val -= blue_val.min()

            color_images.append(
                np.dstack(
                    (
                        red_val**selectivityFactor,
                        green_val**selectivityFactor,
                        blue_val**selectivityFactor,
                    )
                )
            )
        new_max_value = np.max(color_images)
        _all_img = []
        for img in color_images:
            _all_img.append(img / new_max_value)

        final_image = np.sum(_all_img, axis=0)
        final_image /= np.max(final_image)

        if self.lightweightmode:
            del self.image

        return final_image * brightnessFactor

    def build_stimdicts(self):
        self.stim_dict = {i: {} for i in self.stimulus_df.stim_name.unique()}
        self.err_dict = {i: {} for i in self.stimulus_df.stim_name.unique()}
        self.zdiff_cells = [arrutils.zdiffcell(i) for i in self.f_cells]

        for stim in self.stimulus_df.stim_name.unique():
            arrs = arrutils.subsection_arrays(self.stimulus_df[self.stimulus_df.stim_name==stim].frame.values)

            for n, nrn in enumerate(self.zdiff_cells):
                resp_arrs = []
                for arr in arrs:
                    resp_arrs.append(nrn[arr])
                self.stim_dict[stim][n] = np.nanmean(resp_arrs, axis=0)
                self.err_dict[stim][n] = np.nanstd(resp_arrs, axis=0) / np.sqrt(len(resp_arrs))


class VolumeFish:
    def __init__(self):
        self.volumes = {}
        self.volume_inds = {}
        self.last_ind = 0

    def add_volume(self, new_fish, ind=None):
        assert str(new_fish) == "fish", "must be a fish" #  isinstance sometimes failing??
        # assert isinstance(new_fish, BaseFish), "must be a fish" #  this is randomly buggin out

        newKey = new_fish.folder_path.name
        self.volumes[newKey] = new_fish
        if ind:
            self.volume_inds[ind] = newKey
        else:
            self.volume_inds[self.last_ind] = newKey
            self.last_ind += 1

    def add_diff_imgs(self, *args, **kwargs):
        for v in self.volumes.values():
            v.diff_image = v.make_difference_image(*args, **kwargs)

    def volume_diff(self):
        all_diffs = [v.diff_image for v in self.volumes.values()]
        ind1 = [i.shape[0] for i in all_diffs]
        ind2 = [i.shape[1] for i in all_diffs]
        min_ind1 = min(ind1)
        min_ind2 = min(ind2)
        trim_diffs = [i[:min_ind1, :min_ind2, :] for i in all_diffs]
        return np.sum(trim_diffs, axis=0)

    #custom getter to extract volume of interest
    def __getitem__(self, index):
        return self.volumes[self.volume_inds[index]]

class TankError(Exception):
    '''
    Fish doesn't belong in the tank.
    Give him some processing first
    '''
    pass
