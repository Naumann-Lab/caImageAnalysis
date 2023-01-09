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
from tqdm.auto import tqdm

# local imports
import constants
from utilities import pathutils, arrutils


class BaseFish:
    def __init__(
        self,
        folder_path,
        frametimes_key="frametimes",
        invert=True,
    ):
        self.folder_path = Path(folder_path)
        self.frametimes_key = frametimes_key

        self.invert = invert

        self.process_filestructure()  # generates self.data_paths
        self.raw_text_frametimes_to_df()  # generates self.frametimes_df
        # self.load_suite2p() # loads in suite2p paths

    def process_filestructure(self):
        self.data_paths = {}
        with os.scandir(self.folder_path) as entries:
            for entry in entries:
                if entry.name.endswith(".tif"):
                    if "movement_corr" in entry.name:
                        self.data_paths["move_corrected_image"] = Path(entry.path)
                    elif "rotated" in entry.name:
                        self.data_paths["rotated_image"] = Path(entry.path)
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
                    print("failed to move original image out of folder")

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
        self.ops = np.load(
            self.data_paths["suite2p"].joinpath("ops.npy"), allow_pickle=True
        ).item()
        self.iscell = np.load(
            self.data_paths["suite2p"].joinpath("iscell.npy"), allow_pickle=True
        )[:, 0].astype(bool)
        self.stats = np.load(
            self.data_paths["suite2p"].joinpath("stat.npy"), allow_pickle=True
        )
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

    def return_cells_by_location(self, xmin=0, xmax=99999, ymin=0, ymax=99999):
        cell_df = pd.DataFrame(
            self.return_cell_rois(np.arange(0, len(self.f_cells))), columns=["y", "x"]
        )
        return cell_df[
            (cell_df.y >= ymin)
            & (cell_df.y <= ymax)
            & (cell_df.x >= xmin)
            & (cell_df.x <= xmax)
        ].index.values

    def draw_roi(self, title="blank", overwrite=False):
        import cv2

        img = self.ops["refImg"].copy()

        img_arr = np.zeros((max(img.shape), max(img.shape)))

        for x in np.arange(img.shape[0]):
            for y in np.arange(img.shape[1]):
                img_arr[x, y] = img[x, y]

        self.ptlist = []

        def roigrabber(event, x, y, flags, params):
            if event == 1:  # left click
                if len(self.ptlist) == 0:
                    cv2.line(img, pt1=(x, y), pt2=(x, y), color=(255, 255), thickness=3)
                else:
                    cv2.line(
                        img,
                        pt1=(x, y),
                        pt2=self.ptlist[-1],
                        color=(255, 255),
                        thickness=3,
                    )

                self.ptlist.append((y, x))
            if event == 2:  # right click
                cv2.destroyAllWindows()

        cv2.namedWindow(f"roiFinder_{title}")

        cv2.setMouseCallback(f"roiFinder_{title}", roigrabber)

        cv2.imshow(f"roiFinder_{title}", np.array(img, "uint8"))
        try:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            cv2.destroyAllWindows()

        self.save_roi(title, overwrite)

    def save_roi(self, save_name, overwrite):

        savePathFolder = self.folder_path.joinpath("rois")
        if not os.path.exists(savePathFolder):
            os.mkdir(savePathFolder)

        savePath = savePathFolder.joinpath(f"{save_name}.npy")
        if not overwrite and os.path.exists(savePath) and save_name != "blank":
            raise OSError  # not overwriting prior data
        else:
            np.save(savePath, self.ptlist)
            print(f'saved {save_name}')

    def load_saved_rois(self):
        self.roi_dict = {}
        with os.scandir(self.folder_path.joinpath("rois")) as entries:
            for entry in entries:
                self.roi_dict[Path(entry.path).stem] = entry.path

    def return_cells_by_saved_roi(self, roi_name):
        try:
            self.load_saved_rois()
        except FileNotFoundError:
            pass

        if roi_name not in self.roi_dict:
            print("roi not found, please select")
            self.draw_roi(title=roi_name)
            self.load_saved_rois()

        roi_points = np.load(self.roi_dict[roi_name])
        import matplotlib.path as mpltPath

        path = mpltPath.Path(roi_points)

        all_cells = self.return_cells_by_location()
        all_rois = self.return_cell_rois(all_cells)

        cell_in_roi = path.contains_points(all_rois)

        selected_cells = all_cells[cell_in_roi]
        return selected_cells

    def clear_saved_roi(self, roi_name):
        self.load_saved_rois()
        try:
            os.remove(self.roi_dict[roi_name])
        except:
            pass
    def load_image(self):
        if "move_corrected_image" in self.data_paths.keys():
            image = imread(self.data_paths["move_corrected_image"])
        else:
            image = imread(self.data_paths["image"])

        if self.invert:
            image = image[:, :, ::-1]

        return image

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
        return f"fish {self.folder_path.name}"


class PurgeFish(BaseFish):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.purge()

    def purge(self):
        import shutil

        try:
            os.remove(self.folder_path.joinpath("suite2p"))
        except:
            pass
        try:
            shutil.rmtree(self.folder_path.joinpath("suite2p"))
        except:
            pass
        self.process_filestructure()


class VizStimFish(BaseFish):
    def __init__(
        self,
        stim_key="stims",
        stim_fxn=None,
        stim_fxn_args=None,
        stim_offset=5,
        used_offsets=(-10, 14),
        *args,
        **kwargs,
    ):
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

        if self.invert:
            self.stimulus_df.loc[:, "stim_name"] = self.stimulus_df.stim_name.map(
                constants.invStimDict
            )
        self.stim_offset = stim_offset
        self.offsets = used_offsets

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
                self.stimulus_df = stim_fxn(
                    self.data_paths["stimuli"], **self.stim_fxn_args
                )
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
        # self.stimulus_df.drop(columns="time", inplace=True) #this needs to be included in the stimulus_df for TailTrackingFish

    def make_difference_image(self, selectivityFactor=1.5, brightnessFactor=10):
        image = self.load_image()

        diff_imgs = {}
        # for stimulus_name in constants.monocular_dict.keys():
        for stimulus_name in [
            i
            for i in self.stimulus_df.stim_name.values.unique()
            if i in constants.monocular_dict.keys()
        ]:  # KF edit, only have relevant stims
            stim_occurences = self.stimulus_df[
                self.stimulus_df.stim_name == stimulus_name
            ].frame.values

            stim_diff_imgs = []
            for ind in stim_occurences:
                peak = np.nanmean(image[ind : ind + self.offsets[1]], axis=0)
                background = np.nanmean(image[ind + self.offsets[0] : ind], axis=0)
                stim_diff_imgs.append(peak - background)

            diff_imgs[stimulus_name] = np.nanmean(
                stim_diff_imgs, axis=0, dtype=np.float64
            )

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


class TailTrackedFish(VizStimFish):
    def __init__(
        self,
        tail_key="tail",
        tail_fxn=None,
        tail_fxn_args=None,
        bout_sig=4,
        bout_interpeak_dst=50,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # need to have 'tail' in the tail output file
        # tail_fxn is a variable for a fxn that is in the tailtracking.py

        if tail_fxn_args is None:
            tail_fxn_args = []
        self.tail_fxn_args = tail_fxn_args
        self.add_tail(tail_key, tail_fxn)
        self.stim_tail_frame_alignment()

        self.bout_finder(sig=4, interpeak_dst=50, height=None, width=None, prominence=1)
        self.bout_responsive_neurons()

    def add_tail(self, tail_key, tail_fxn):
        with os.scandir(self.folder_path) as entries:
            for entry in entries:
                if tail_key in entry.name:
                    self.data_paths["tail"] = Path(entry.path)
        try:
            _ = self.data_paths["tail"]
        except KeyError:
            print("failed to find tail data")
            return
        if tail_fxn:
            if self.tail_fxn_args:
                self.tail_df = tail_fxn(self.data_paths["tail"], **self.tail_fxn_args)
            else:
                self.tail_df = tail_fxn(self.data_paths["tail"])

        # aligning tail conv times to image frame times

    def stim_tail_frame_alignment(self):
        # trimming tail df to match the size of the imaging times
        if (
            "frame" not in self.tail_df.columns
        ):  # do not have to run this again if already has it
            try:
                self.tail_df = self.tail_df[
                    (self.tail_df.conv_t >= self.frametimes_df.values[0][0])
                    & (self.tail_df.conv_t <= self.frametimes_df.values[-1][0])
                ]

                for frameN in tqdm(
                    range(len(self.frametimes_df.values)),
                    "aligning frametimes to tail data",
                ):
                    try:
                        indices = self.tail_df[
                            (
                                self.tail_df.conv_t
                                >= self.frametimes_df.values[frameN][0]
                            )
                            & (
                                self.tail_df.conv_t
                                <= self.frametimes_df.values[frameN + 1][0]
                            )
                        ].index
                    except IndexError:
                        pass
                    self.tail_df.loc[indices, "frame"] = frameN
            except:
                pass

        # making a stimulus df with tail index and image index values for each stim
        final_t = self.frametimes_df["time"].iloc[-1]  # last time in imaging
        image_infos = []
        tail_infos = []
        self.tail_stimulus_df = self.stimulus_df.copy()
        for i in range(len(self.tail_stimulus_df)):
            if i == len(self.tail_stimulus_df) - 1:
                tail_infos.append(
                    self.tail_df[
                        (self.tail_df.conv_t >= self.tail_stimulus_df.time.values[i])
                        & (self.tail_df.conv_t <= final_t)
                    ].index
                )
                break
            else:
                tail_infos.append(
                    self.tail_df[
                        (self.tail_df.conv_t >= self.tail_stimulus_df.time.values[i])
                        & (
                            self.tail_df.conv_t
                            <= self.tail_stimulus_df.time.values[i + 1]
                        )
                    ].index
                )
        self.tail_stimulus_df.loc[:, "tail_index"] = tail_infos

        for j in range(len(self.tail_stimulus_df)):
            if j == len(self.tail_stimulus_df) - 1:
                image_infos.append(
                    self.frametimes_df[
                        (
                            self.frametimes_df["time"]
                            >= self.tail_stimulus_df.time.values[j]
                        )
                        & (self.frametimes_df["time"] <= final_t)
                    ].index
                )
            else:
                image_infos.append(
                    self.frametimes_df[
                        (
                            self.frametimes_df["time"]
                            >= self.tail_stimulus_df.time.values[j]
                        )
                        & (
                            self.frametimes_df["time"]
                            <= self.tail_stimulus_df.time.values[j + 1]
                        )
                    ].index
                )

        self.tail_stimulus_df.loc[:, "img_stacks"] = image_infos

    def bout_finder(
        self, sig=4, interpeak_dst=50, height=None, width=None, prominence=1
    ):
        from scipy.signal import find_peaks
        import scipy.ndimage

        # sig = sigma for gaussian filter on the tail data
        # interpeak_dst = ms, distance between bouts

        # tail deflection sum from central axis of fish, filtered with gaussian fit
        if width is None:
            width = [0, 750]
        if height is None:
            height = [20, 120]

        filtered_deflections = scipy.ndimage.gaussian_filter(
            self.tail_df["/'TailLoc'/'TailDeflectSum'"].values, sigma=sig
        )

        peak_deflection, peaks = scipy.signal.find_peaks(
            abs(filtered_deflections),
            height=height,
            threshold=None,
            prominence=prominence,
            width=width,
        )
        # get bout peaks
        leftofPeak = peaks["left_ips"]
        rightofPeak = peaks["right_ips"]
        peak_pts = np.stack([leftofPeak, rightofPeak], axis=1)
        bout_start = []
        bout_end = []
        n = 0
        while n < len(peak_pts) - 2:
            # if current right + minimum is less than the next left its good
            if peak_pts[n][1] + interpeak_dst <= peak_pts[n + 1][0]:
                bout_end.append(int(peak_pts[n][1]))
                bout_start.append(int(peak_pts[n][0]))
                n += 1
            # otherwise increase the index until thats the case
            else:
                while n < len(peak_pts) - 2:
                    n += 1
                    if peak_pts[n][1] + interpeak_dst <= peak_pts[n + 1][0]:
                        bout_end.append(int(peak_pts[n][1]))
                        bout_start.append(int(peak_pts[n][0]))
                        n += 1
                        break

        # accounts for interbout distance, left and right of each peak in filtered tail deflection data ("/'TailLoc'/'TailDeflectSum'")
        new_peak_pts = np.stack(
            [bout_start, bout_end], axis=1
        )  # all peaks in tail data
        tail_ind_start = self.tail_stimulus_df.iloc[0].tail_index.values[0]
        tail_ind_stop = self.tail_stimulus_df.iloc[-2].tail_index.values[-1]

        ind_0 = np.where(new_peak_pts[:, 0] >= tail_ind_start)[0][0]
        ind_1 = np.where(new_peak_pts[:, 1] <= tail_ind_stop)[0][-1]
        relevant_pts = new_peak_pts[
            ind_0:ind_1
        ]  # peaks only within the stimuli presentation

        dict_info = {}
        for bout_ind in range(len(relevant_pts)):
            if bout_ind not in dict_info.keys():
                dict_info[bout_ind] = {}

            bout_angle = np.sum(
                self.tail_df.iloc[:, 4].values[
                    relevant_pts[bout_ind][0] : relevant_pts[bout_ind][1]
                ]
            )  # total bout angle
            dict_info[bout_ind]["bout_angle"] = bout_angle

            frame_start = self.tail_df.iloc[:, -1].values[relevant_pts[bout_ind][0]]
            frame_end = self.tail_df.iloc[:, -1].values[relevant_pts[bout_ind][1]]
            dict_info[bout_ind]["image_frames"] = frame_start, frame_end

        self.tail_bouts_df = pd.DataFrame.from_dict(dict_info, "index")
        self.tail_bouts_df.loc[:, "bout_dir"] = np.zeros(self.tail_bouts_df.shape[0])
        self.tail_bouts_df["bout_dir"][self.tail_bouts_df["bout_angle"] > 0] = "left"
        self.tail_bouts_df["bout_dir"][self.tail_bouts_df["bout_angle"] < 0] = "right"
        # tail_bouts_df has bout indices, frames from image frametimes, and bout direction
        return self.tail_bouts_df

    def bout_responsive_neurons(self, offset=2, thresh=0.2):
        nrns = []
        vals = []
        bouts = []

        self.norm_cells = arrutils.norm_fdff(self.f_cells)
        for q in range(self.norm_cells.shape[0]):
            for bout in range(len(self.tail_bouts_df)):
                s = self.tail_bouts_df.iloc[:, 1].values[bout][0] - offset
                if s <= 0:
                    s = 0
                e = self.tail_bouts_df.iloc[:, 1].values[bout][1] + offset
                if e >= self.norm_cells.shape[1]:
                    e = self.norm_cells.shape[1]
                nrns.append(q)  # all the neurons
                bouts.append(bout)  # all the bouts
                vals.append(
                    np.median(self.norm_cells[q][int(s) : int(e)])
                )  # median response to a bout

        neurbout_df = pd.DataFrame({"neur": nrns, "bout": bouts, "fluor": vals})

        self.neurbout_dict = (
            {}
        )  # to make a dict with each neuron as key and item as bouts
        for n in neurbout_df.neur.unique():
            if n not in self.neurbout_dict.keys():
                self.neurbout_dict[n] = {}
            oneneur = neurbout_df[neurbout_df["neur"] == n]
            responsive = []
            for b in oneneur.bout.unique():
                if np.median(oneneur[oneneur.bout == b]["fluor"].values) >= thresh:
                    responsive.append(b)
                self.neurbout_dict[n] = responsive

        _resp_cells = []  # list of only the responsive neuron id's
        for key, value in self.neurbout_dict.items():
            for n in value:
                _resp_cells.append(key)
        output = set(_resp_cells)
        self.resp_cells = list(output)
        self.resp_cells.sort()


class WorkingFish(VizStimFish):
    """
    the classic: the every-man's briefcase wielding workhorse
    """

    def __init__(self, corr_threshold=0.65, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if "move_corrected_image" not in self.data_paths:
            raise TankError
        self.corr_threshold = corr_threshold

        self.diff_image = self.make_difference_image()

        self.load_suite2p()
        self.build_stimdicts()

    def build_stimdicts(self):
        self.stim_dict = {i: {} for i in self.stimulus_df.stim_name.unique()}
        self.err_dict = {i: {} for i in self.stimulus_df.stim_name.unique()}
        self.zdiff_cells = [arrutils.zdiffcell(i) for i in self.f_cells]

        for stim in self.stimulus_df.stim_name.unique():
            arrs = arrutils.subsection_arrays(
                self.stimulus_df[self.stimulus_df.stim_name == stim].frame.values,
                self.offsets,
            )

            for n, nrn in enumerate(self.zdiff_cells):
                resp_arrs = []
                for arr in arrs:
                    resp_arrs.append(nrn[arr])
                self.stim_dict[stim][n] = np.nanmean(resp_arrs, axis=0)
                self.err_dict[stim][n] = np.nanstd(resp_arrs, axis=0) / np.sqrt(
                    len(resp_arrs)
                )

        self.neuron_dict = {}
        for neuron in self.stim_dict[
            "forward"
        ].keys():  # generic stim to grab all neurons
            if neuron not in self.neuron_dict.keys():
                self.neuron_dict[neuron] = {}

            for stim in self.stimulus_df.stim_name.unique():
                self.neuron_dict[neuron][stim] = np.nanmedian(
                    self.stim_dict[stim][neuron][
                        -self.offsets[0] : -self.offsets[0] + self.stim_offset
                    ]
                )

    def build_booldf(self, stim_arr=None, zero_arr=True, force=False):
        if hasattr(self, "booldf"):
            if not force:
                return

        if not stim_arr:
            provided = False
        else:
            provided = True

        corr_dict = {}
        bool_dict = {}
        for stim in self.stim_dict.keys():
            if stim not in bool_dict.keys():
                bool_dict[stim] = {}
                corr_dict[stim] = {}
            for nrn in self.stim_dict[stim].keys():
                cell_array = self.stim_dict[stim][nrn]
                if zero_arr:
                    cell_array = np.clip(cell_array, a_min=0, a_max=99)
                if not provided:
                    stim_arr = np.zeros(len(cell_array))
                    stim_arr[
                        -self.offsets[0] + 2 : -self.offsets[0] + self.stim_offset - 2
                    ] = 1.5
                    stim_arr = arrutils.pretty(stim_arr, 3)
                corrVal = round(np.corrcoef(stim_arr, cell_array)[0][1], 3)

                corr_dict[stim][nrn] = corrVal
                bool_dict[stim][nrn] = corrVal >= self.corr_threshold
        self.booldf = pd.DataFrame(bool_dict)
        self.corrdf = pd.DataFrame(corr_dict)

        self.booldf = self.booldf.loc[self.booldf.sum(axis=1) > 0]

    def make_computed_image_data(self, colorsumthresh=1):
        if not hasattr(self, "neuron_dict"):
            self.build_stimdicts()
        xpos = []
        ypos = []
        colors = []
        neurons = []

        for neuron in self.neuron_dict.keys():
            myneuron = self.neuron_dict[neuron]
            clr_longform = [
                stimval * np.clip(i, a_min=0, a_max=99)
                for stimname, stimval in zip(myneuron.keys(), myneuron.values())
                if stimname in constants.monocular_dict.keys()
                for i in constants.monocular_dict[stimname]
            ]
            reds = clr_longform[::3]
            greens = clr_longform[1::3]
            blues = clr_longform[2::3]

            fullcolor = np.sum([reds, greens, blues], axis=1)

            if max(fullcolor) > 1.0:
                fullcolor /= max(fullcolor)
            fullcolor = np.clip(fullcolor, a_min=0, a_max=1.0)
            if np.sum(fullcolor) > colorsumthresh:
                yloc, xloc = self.return_cell_rois(neuron)[0]

                xpos.append(xloc)
                ypos.append(yloc)
                colors.append(fullcolor)
                neurons.append(neuron)
        return xpos, ypos, colors, neurons

    def make_computed_image_data_by_loc(
        self, xmin=0, xmax=99999, ymin=0, ymax=9999, *args, **kwargs
    ):
        xpos, ypos, colors, neurons = self.make_computed_image_data(*args, **kwargs)
        loc_cells = self.return_cells_by_location(
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
        )

        valid_cells = [i for i in neurons if i in loc_cells]
        valid_inds = [neurons.index(i) for i in valid_cells]
        valid_x = [i for n, i in enumerate(xpos) if n in valid_inds]
        valid_y = [i for n, i in enumerate(ypos) if n in valid_inds]
        valid_colors = [i for n, i in enumerate(colors) if n in valid_inds]
        return valid_x, valid_y, valid_colors, valid_cells

    def make_computed_image_data_by_roi(self, roi_name, *args, **kwargs):
        xpos, ypos, colors, neurons = self.make_computed_image_data(*args, **kwargs)
        selected_cells = self.return_cells_by_saved_roi(roi_name)

        valid_cells = [i for i in neurons if i in selected_cells]
        valid_inds = [neurons.index(i) for i in valid_cells]
        valid_x = [i for n, i in enumerate(xpos) if n in valid_inds]
        valid_y = [i for n, i in enumerate(ypos) if n in valid_inds]
        valid_colors = [i for n, i in enumerate(colors) if n in valid_inds]
        return valid_x, valid_y, valid_colors, valid_cells

    def return_degree_vectors(self, neurons):
        import angles

        if not hasattr(self, "booldf"):
            self.build_booldf()

        bool_monoc = self.booldf[constants.monocular_dict.keys()]
        monoc_bool_neurons = bool_monoc.loc[bool_monoc.sum(axis=1) > 0].index.values
        valid_neurons = [i for i in monoc_bool_neurons if i in neurons]

        thetas = []
        thetavals = []
        for n in valid_neurons:
            neuron_response_dict = self.neuron_dict[n]
            monoc_neuron_response_dict = {
                k: v
                for k, v in neuron_response_dict.items()
                if k in constants.monocular_dict.keys()
            }

            degree_ids = [
                constants.deg_dict[i] for i in monoc_neuron_response_dict.keys()
            ]
            degree_responses = [
                np.clip(i, a_min=0, a_max=999)
                for i in monoc_neuron_response_dict.values()
            ]

            theta = angles.weighted_mean_angle(degree_ids, degree_responses)
            thetaval = np.nanmean(degree_responses)

            thetas.append(theta)
            thetavals.append(thetaval)
        return thetas, thetavals


class VolumeFish:
    def __init__(self):
        self.volumes = {}
        self.volume_inds = {}
        self.last_ind = 0
        self.iter_ind = -1

    def add_volume(self, new_fish, ind=None):
        assert "fish" in str(
            new_fish
        ), "must be a fish"  #  isinstance sometimes failing??
        # assert isinstance(new_fish, BaseFish), "must be a fish" #  this is randomly buggin out

        newKey = new_fish.folder_path.name
        self.volumes[newKey] = new_fish
        if ind:
            self.volume_inds[ind] = newKey
        else:
            self.volume_inds[self.last_ind] = newKey
            self.last_ind += 1

    # custom getter to extract volume of interest
    def __getitem__(self, index):
        try:
            return self.volumes[self.volume_inds[index]]
        except KeyError:
            raise StopIteration  # technically thrown if your try to get a vol thats not there, useful because lets us loops

    def __len__(self):
        return self.last_ind


class VizStimVolume(VolumeFish):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_diff_imgs(self, *args, **kwargs):
        for v in tqdm(self.volumes.values()):
            v.diff_image = v.make_difference_image(*args, **kwargs)

    def volume_diff(self):
        all_diffs = [v.diff_image for v in self.volumes.values()]
        ind1 = [i.shape[0] for i in all_diffs]
        ind2 = [i.shape[1] for i in all_diffs]
        min_ind1 = min(ind1)
        min_ind2 = min(ind2)
        trim_diffs = [i[:min_ind1, :min_ind2, :] for i in all_diffs]
        return np.sum(trim_diffs, axis=0)

    def volume_computed_image(self, *args, **kwargs):
        all_x = []
        all_y = []
        all_colors = []
        all_neurons = []
        for v in self:
            xpos, ypos, colors, neurons = v.make_computed_image_data(*args, **kwargs)

            all_x += xpos
            all_y += ypos
            all_colors += colors
            all_neurons += neurons
        return all_x, all_y, all_colors, all_neurons

    def volume_computed_image_loc(self, *args, **kwargs):
        all_x = []
        all_y = []
        all_colors = []
        all_neurons = []
        for v in self:
            xpos, ypos, colors, neurons = v.make_computed_image_data_by_loc(
                *args, **kwargs
            )

            all_x += xpos
            all_y += ypos
            all_colors += colors
            all_neurons += neurons
        return all_x, all_y, all_colors, all_neurons

    def volume_computed_image_from_roi(self, *args, **kwargs):
        all_x = []
        all_y = []
        all_colors = []
        all_neurons = []
        for v in self:
            xpos, ypos, colors, neurons = v.make_computed_image_data_by_roi(
                *args, **kwargs
            )

            all_x += xpos
            all_y += ypos
            all_colors += colors
            all_neurons += neurons
        return all_x, all_y, all_colors, all_neurons


class TankError(Exception):
    """
    Fish doesn't belong in the tank.
    Give him some processing first
    """

    pass
