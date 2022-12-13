"""
the new, latest & greatest 
home to a variety of fishys
"""
import os

import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime as dt

# local imports
import constants


class BaseFish:
    def __init__(self, basePath, frametimes_key="frametimes"):
        self.process_filestructure(
            basePath, frametimes_key
        )  # generates self.data_paths
        self.raw_text_frametimes_to_df()  # generates self.frametimes_df

    def process_filestructure(self, folderPath, frametimes_key):
        self.folder_path = Path(folderPath)
        self.data_paths = {}
        with os.scandir(self.folder_path) as entries:
            for entry in entries:
                if entry.name.endswith(".tif"):
                    self.data_paths["image"] = Path(entry.path)
                elif entry.name.endswith(".txt") and "log" in entry.name:
                    self.data_paths["log"] = Path(entry.path)
                elif entry.name.endswith(".txt") and frametimes_key in entry.name:
                    self.data_paths["frametimes"] = Path(entry.path)

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


class ProcessFish(BaseFish):
    def __init__(self, input_tau=1.5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tau = input_tau
        self.run_movement_correction()
        self.run_suite2p()

    def run_movement_correction(self, caiman_ops=None, keep_mmaps=False):
        import caiman as cm
        import shutil
        from tifffile import imsave

        original_image_path = self.data_paths["image"]

        if not caiman_ops:
            caiman_ops = {
                "max_shifts": (3, 3),
                "strides": (25, 25),
                "overlaps": (15, 15),
                "num_frames_split": 150,
                "max_deviation_rigid": 3,
                "pw_rigid": False,
                "shifts_opencv": True,
                "border_nan": "copy",
                "downsample_ratio": 0.2,
            }
        c, dview, n_processes = cm.cluster.setup_cluster(
            backend="local", n_processes=12, single_thread=False
        )
        mc = cm.motion_correction.MotionCorrect(
            [original_image_path.as_posix()],
            dview=dview,
            max_shifts=caiman_ops["max_shifts"],
            strides=caiman_ops["strides"],
            overlaps=caiman_ops["overlaps"],
            max_deviation_rigid=caiman_ops["max_deviation_rigid"],
            shifts_opencv=caiman_ops["shifts_opencv"],
            nonneg_movie=True,
            border_nan=caiman_ops["border_nan"],
            is3D=False,
        )
        mc.motion_correct(save_movie=True)
        bord_px_rig = np.ceil(np.max(mc.shifts_rig)).astype(np.int)
        mc.pw_rigid = True  # turn the flag to True for pw-rigid motion correction
        mc.template = (
            mc.mmap_file
        )  # use the template obtained before to save in computation (optional)
        mc.motion_correct(save_movie=True, template=mc.total_template_rig)
        m_els = cm.load(mc.fname_tot_els)
        output = m_els[
            :,
            2 * bord_px_rig : -2 * bord_px_rig,
            2 * bord_px_rig : -2 * bord_px_rig,
        ]
        if not keep_mmaps:
            with os.scandir(original_image_path.parents[0]) as entries:
                for entry in entries:
                    if entry.is_file():
                        if entry.name.endswith(".mmap"):
                            os.remove(entry)
        dview.terminate()
        cm.stop_server()

        newdir = original_image_path.parents[0].joinpath("original_image")
        if not os.path.exists(newdir):
            os.mkdir(newdir)

        new_path = newdir.parents[0].joinpath("movement_corr_img.tif")

        new_original_image_path = newdir.joinpath("image.tif")
        if os.path.exists(new_original_image_path):
            os.remove(new_original_image_path)

        shutil.move(original_image_path, new_original_image_path)
        imsave(new_path, output)

        if os.path.exists(original_image_path) and os.path.exists(new_path):
            os.remove(original_image_path)

        self.data_paths["move_corrected_image"] = new_path

    def run_suite2p(self, s2p_ops=None):
        from suite2p.run_s2p import run_s2p, default_ops

        imageHz = self.hzReturner(self.frametimes_df)
        try:
            imagepath = self.data_paths["move_corrected_image"]
        except KeyError:
            imagepath = self.data_paths["image"]

        if not s2p_ops:
            s2p_ops = {
                "data_path": [imagepath.parents[0].as_posix()],
                "save_path0": imagepath.parents[0].as_posix(),
                "tau": self.tau,
                "preclassify": 0.15,
                "allow_overlap": True,
                "block_size": [50, 50],
                "fs": imageHz,
            }
        ops = default_ops()
        db = {}
        for item in s2p_ops:
            ops[item] = s2p_ops[item]

        output_ops = run_s2p(ops=ops, db=db)

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


class VizStimFish(BaseFish):
    def __init__(self, stim_key="stims", stim_fxn=None, *args, **kwargs):
        """

        :param stim_key: filename key to find stims in folder
        :param stim_fxn: processes stimuli of interest: returns df with minimum "stim_name" and "time" columns
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
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


class WorkingFish(VizStimFish):
    def __init__(self, lightweight=False, invert=True, stim_offset=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stim_offset = stim_offset

        if not lightweight:
            from tifffile import imread

            try:
                self.image = imread(self.data_paths["move_corrected_image"])
            except KeyError:
                return

            if invert:
                self.image = self.image[:, :, ::-1]

            self.diff_image = self.make_difference_image(*args, **kwargs)

    def make_difference_image(self, selectivityFactor=1.5, brightnessFactor=10):
        diff_imgs = {}
        for stimulus_name in constants.monocular_dict.keys():
            stim_occurences = self.stimulus_df[
                self.stimulus_df.stim_name == stimulus_name
            ].frame.values

            stim_diff_imgs = []
            for ind in stim_occurences:
                peak = np.nanmean(self.image[ind : ind + self.stim_offset], axis=0)
                background = np.nanmean(
                    self.image[ind - 2 * self.stim_offset : ind - 1], axis=0
                )
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
        return final_image * brightnessFactor
