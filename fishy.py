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
from shapely.geometry import Polygon, Point
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

import bruker_images
# local imports
import constants
from utilities import pathutils, arrutils, roiutils, coordutils
import stimuli
import photostimulation



class BaseFish:
    def __init__(
        self,
        folder_path,
        frametimes_key="frametimes",
        invert=False,
        bruker_invert = False,
        midnight_noon = "noon"
    ):
        self.folder_path = Path(folder_path)
        self.frametimes_key = frametimes_key

        self.invert = invert
        self.bruker_invert = bruker_invert # inverts the stim names since bruker projector was displayed differently

        self.process_filestructure(midnight_noon)  # generates self.data_paths
        try:
            self.raw_text_frametimes_to_df(midnight_noon)  # generates self.frametimes_df
        except:
            print("failed to process frametimes from text")

        if 'suite2p' in self.data_paths.keys():
            self.load_suite2p()  # loads in suite2p paths
            self.rescaled_img()
            self.is_cell()
            self.zdiff_cells = [arrutils.zdiffcell(i) for i in self.f_cells]
            self.normf_cells = arrutils.norm_0to1(self.f_cells)
        if 'caiman' in self.data_paths.keys():
            print('loading once..')
            self.load_caiman()  # load in caiman data
            self.rescaled_img()
            self.is_cell()
            self.zdiff_cells = [arrutils.zdiffcell(i) for i in self.f_cells]
            self.normf_cells = arrutils.norm_0to1(self.f_cells)

    def process_filestructure(self, midnight_noon):
        self.data_paths = {}
        with os.scandir(self.folder_path) as entries:
            for entry in entries:
                if entry.name.endswith(".tif"):
                    if "img" in entry.name:
                        self.data_paths["move_corrected_image"] = Path(entry.path)
                    elif "rotated" in entry.name:
                        self.data_paths["rotated_image"] = Path(entry.path)
                    else:
                        self.data_paths["image"] = Path(entry.path)

                elif entry.name == 'tail_df.h5':
                    self.data_paths['tail'] = Path(entry.path)

                elif entry.name.endswith(".txt") and self.frametimes_key in entry.name:
                    self.data_paths["frametimes"] = Path(entry.path)

                elif entry.name == "frametimes.h5":
                    self.frametimes_df = pd.read_hdf(entry.path)
                    print("found and loaded frametimes h5")
                    if (np.diff(self.frametimes_df.index) > 1).any():
                        self.frametimes_df.reset_index(inplace=True)
                    if midnight_noon == 'midnight':
                        list = self.frametimes_df.time
                        list = [time.strftime(format="%H:%M:%S.%f") for time in list]
                        list = ['00' + time[2:] for time in list if time[:2] == '12']
                        self.frametimes_df.time = [dt.strptime(time, "%H:%M:%S.%f").time() for time in list]

                elif os.path.isdir(entry.path):
                    if entry.name == "suite_2p":
                        self.data_paths["suite2p"] = Path(entry.path).joinpath("plane0")
                    if entry.name == 'caiman':
                        self.data_paths["caiman"] = Path(entry.path)
                    if entry.name == "original_image":
                        with os.scandir(entry.path) as imgdiver:
                            for poss_img in imgdiver:
                                if poss_img.name.endswith(".tif"):
                                    self.data_paths["image"] = Path(poss_img.path)
                    elif entry.name == "ROIs":
                        self.roi_dict = {}
                        with os.scandir(entry.path) as rois:
                            for roi in rois:
                                self.roi_dict[roi.name[:-4]] = pd.read_csv(roi.path)

                elif entry.name.endswith(".npy"): # these are mislabeled so just flip here
                    if "xpts" in entry.name:
                        with open(entry.path, "rb") as f:
                            self.y_pts = np.load(f)
                    elif "ypts" in entry.name:
                        with open(entry.path, "rb") as f:
                            self.x_pts = np.load(f)

                # bruker information files
                elif entry.name.endswith("xml"):
                    if 'MarkPoints' in entry.name:
                        self.data_paths["ps_xml"] = Path(entry.path)
                    elif "Voltage" not in entry.name and 'MarkPoints' not in entry.name:
                        self.data_paths["info_xml"] = Path(entry.path)
                elif entry.name.endswith("env"):
                    self.data_paths["info_env"] = Path(entry.path)
                elif entry.name.endswith("csv") and 'Voltage' in entry.name:
                    self.data_paths["voltage_signal"] = Path(entry.path)

        if "image" in self.data_paths and "move_corrected_image" in self.data_paths:
            if (
                self.data_paths["image"].parents[0]
                == self.data_paths["move_corrected_image"].parents[0]
            ):
                try:
                    pathutils.move_og_image(self.data_paths["image"])
                except:
                    print("failed to move original image out of folder")

    def raw_text_frametimes_to_df(self, midnight_noon):
        if hasattr(self, "frametimes_df"):
            return
        with open(self.data_paths["frametimes"]) as file:
            contents = file.read()
        parsed = contents.split("\n")
        times = []
        for line in range(len(parsed) - 1):
            text = parsed[line]
            if midnight_noon == 'midnight':
                if text[:2] == '12':
                    text = '00' + text[2:]
            times.append(dt.strptime(text, "%H:%M:%S.%f").time())
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

    def load_caiman(self):
        # make a ops['refImg'] to be used later, like with suite2p data
        mean_img = np.nanmean(self.load_image(), axis = 0)
        self.ops = {'refImg': mean_img}

        self.iscell = np.load(
            self.data_paths["caiman"].joinpath("iscell.npy"), allow_pickle=True).astype(bool)

        self.stats = np.load(
            self.data_paths["caiman"].joinpath("coordinates_dict.npy"), allow_pickle=True)
        self.f_cells = np.load(self.data_paths["caiman"].joinpath("C.npy"))
        self.df_f_cells = np.load(self.data_paths["caiman"].joinpath("F_dff.npy"))

    def is_cell(self):
        """
        Clean up the self.f_cells and self.df_f_cells according to self.iscell and clean up cells that didn't
        change fluorscence throughout the trial at all

        also clean up cells that are smalelr
        """
        iscell_index = np.where(self.iscell)
        ischanging_index = np.where(np.amax(self.f_cells, 1) != np.amin(self.f_cells, 1))
        iscell_index = np.intersect1d(iscell_index, ischanging_index)
        self.f_cells = self.f_cells[iscell_index]
        self.stats = self.stats[iscell_index]
        if 'caiman' in self.data_paths.keys():
            self.df_f_cells = self.df_f_cells[iscell_index]

    def return_cell_rois(self, cells):
        rois = []
        for cell in cells:
            ypix = self.stats[cell]["ypix"]
            xpix = self.stats[cell]["xpix"]
            try:
                mean_y = int(np.mean(ypix))
                mean_x = int(np.mean(xpix))
            except ValueError:
                mean_y = np.nan
                mean_x = np.nan
            rois.append([mean_x, mean_y])
        return rois
    
    def return_singlecell_rois(self, single_cell):
        single_cell = int(single_cell)
        ypix = self.stats[single_cell]["ypix"]
        xpix = self.stats[single_cell]["xpix"]
        mean_y = int(np.mean(ypix))
        mean_x = int(np.mean(xpix))
        roi = ([mean_x, mean_y])
        
        return roi

    def return_cells_by_location(self, xmin=0, xmax=99999, ymin=0, ymax=99999):
        cell_df = pd.DataFrame(
            self.return_cell_rois(np.arange(0, len(self.f_cells))), columns=["x", "y"]
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
            print(f"saved {save_name}")

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

    def return_cells_by_location_imagej(self, roi):
        cell_df = pd.DataFrame(
            self.return_cell_rois(np.arange(0, len(self.f_cells))), columns=["x", "y"]
        )
        roi = Polygon(roi)
        return_index = []
        for index, cell in cell_df.iterrows():
            if Point(cell).within(roi):
                return_index.append(index)
        return return_index

    def load_image(self):
        if "move_corrected_image" in self.data_paths.keys():
            image = imread(self.data_paths["move_corrected_image"])
        else:
            image = imread(self.data_paths["image"])

        # # if self.invert:
        #     image = image[:, :, ::-1]

        return image

    def rescaled_img(self):
        self.rescaled_ref = self.ops['refImg']
        self.rescaled_ref = self.rescaled_ref/ self.rescaled_ref.max()
        self.rescaled_ref *= 2**12
        return print('rescaled img made')
    
    def zscored_img(self):
        stack = self.load_image()
        flat_stack = np.reshape(stack, (-1, stack.shape[2])) # Reshape the stack to (num_pixels x num_images)

        # Calculate mean and standard deviation along the pixel axis
        mean_intensity = np.mean(flat_stack, axis=0)
        std_intensity = np.std(flat_stack, axis=0)

        z_scored_stack = (stack - mean_intensity) / std_intensity

        return z_scored_stack

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

            if test1 >= len(frametimes):
                increment = increment // 2
                test0 = 0
                test1 = increment

        times = [
            float(str(f.second) + "." + str(f.microsecond))
            for f in frametimes.loc[:, "time"].values[test0:test1]
        ]
        return 1 / np.mean(np.diff(times))

    @staticmethod
    def tag_frames_to_df(frametimes_df, df, datetime_col_name = 'time'):
        '''
        static method to tag frames with any dataframe
        frametimes_df: the dataframe with your frametimes
        df: the target dataframe that you want matching frametimes with, needs to have one column with a datetime object
        datetime_col_name: the name of the datetime object column
        '''
        frame_matches = [frametimes_df[frametimes_df.time < df[datetime_col_name].values[i]].index[-1] for i in range(len(df))]

        df.loc[:, "frame"] = frame_matches
        df = df[df['frame'] != 0]
        df.reset_index(inplace = True, drop = True)

        return df

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
        stim_key="stim",
        stim_fxn=stimuli.pandastim_to_df,
        stim_fxn_args=None,
        legacy=False,
        stim_offset=5,
        used_offsets=(-10, 14),
        baseline_offset=-4, # adding a baseline number of frames
        r_type="median",  # response type - can be median, mean, peak of the stimulus response, default is median
        *args,
        **kwargs,
    ):
        """
        :param stim_key: filename key to find stims in folder
        :param stim_fxn: processes stimuli of interest: returns df with minimum "stim_name" and "time" columns
        :param stim_fxn_args:
        :param legacy:
        :param stim_offset:
        :param used_offsets:
        :param r_type:
        :param args:
        :param kwargs:
        """

        super().__init__(*args, **kwargs)
        if stim_fxn_args is None:
            stim_fxn_args = {}
        if not hasattr(self, "f_cells"):
            if 'suite_2p' in self.data_paths.keys():
                self.load_suite2p()
            if 'caiman' in self.data_paths.keys():
                self.load_caiman()
            self.is_cell()
        self.stim_fxn_args = stim_fxn_args
        self.add_stims(stim_key, stim_fxn, legacy)

        self.r_type = r_type

        # set up inversions
        if self.invert:
            self.stimulus_df.loc[:, "stim_name"] = self.stimulus_df.stim_name.map(
                constants.invStimDict
            )
        if self.bruker_invert:
            self.stimulus_df.loc[:, "stim_name"] = self.stimulus_df.stim_name.map(
                constants.bruker_invStimDict
            )

        # set up offsets
        self.stim_offset = stim_offset
        self.offsets = used_offsets
        self.baseline_offset = baseline_offset
        #self.diff_image = self.make_difference_image()

    def add_stims(self, stim_key, stim_fxn, legacy):
        with os.scandir(self.folder_path) as entries:
            for entry in entries:
                if stim_key in entry.name:
                    if entry.name.endswith('.csv'):
                        self.data_paths["stimuli"] = Path(entry.path)
                        dateparse = lambda x: pd.to_datetime(x, format='%H:%M:%S.%f').time()
                        self.stimulus_df = pd.read_csv(self.data_paths['stimuli'],  index_col=0, parse_dates=['time'], date_parser=dateparse)
                        break
                    elif entry.name.endswith('.txt'):
                        self.data_paths["stimuli"] = Path(entry.path)
                        if stim_fxn:
                            if self.stim_fxn_args:
                                try:
                                    self.stimulus_df = stim_fxn(
                                        self.data_paths["stimuli"], **self.stim_fxn_args
                                    )
                                except:
                                    try:
                                        self.stimulus_df = stim_fxn(
                                            self.folder_path, **self.stim_fxn_args
                                        )
                                    except:
                                        print("failed to generate stimulus df")
                            else:
                                self.stimulus_df = stim_fxn(self.data_paths["stimuli"])
                    self.unchop_stimulus_df = self.stimulus_df  # chop stimulus that are outside of frametime
        # THIS IS ASSUMING THAT THE IMAGING DIDN'T LAST FOR 24 HRS+
        if self.frametimes_df.time.values[0] > self.frametimes_df.time.values[-1]:  # overnight
            self.stimulus_df = pd.concat(
                            [self.stimulus_df[(self.stimulus_df.time > self.frametimes_df.time.values[0])],
                             self.stimulus_df[(self.stimulus_df.time < self.frametimes_df.time.values[-1])]])
        else:
            self.stimulus_df = self.stimulus_df[
                            (self.stimulus_df.time > self.frametimes_df.time.values[0]) &
                            (self.stimulus_df.time < self.frametimes_df.time.values[-1])]
        self.stimulus_df = self.tag_frames_to_df(self.frametimes_df, self.stimulus_df, 'time')
        if not legacy:
            try:
                _ = self.data_paths["stimuli"]
            except KeyError:
                print("failed to find stimuli")
                return




    # made a new generic tag frames function, but keeping this in case
    # def tag_frames(self):

    #     frame_matches = [self.frametimes_df[self.frametimes_df.time < self.stimulus_df.time.values[i]].index[-1] for i in range(len(self.stimulus_df))]

    #     self.stimulus_df.loc[:, "frame"] = frame_matches
    #     # self.stimulus_df.drop(columns="time", inplace=True) #this needs to be included in the stimulus_df for TailTrackingFish
    #     self.stimulus_df = self.stimulus_df[self.stimulus_df['frame'] != 0]
    #     self.stimulus_df.reset_index(inplace = True, drop = True)

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
                ind = int(ind)
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

class PhotostimFish(BaseFish):
    def __init__(
        self,
        no_planes = 5,
        stimmed_planes = [1, 2, 3, 4, 5],
        rotate = True,
        *args,
        **kwargs,
    ):
        '''
        :param no_planes: number of planes in the volume/dataset
        :param stimmed_planes: which planes were actually stimulated
        :param rotate: is the image rotated 90 degrees
        '''
        super().__init__(*args, **kwargs)

        # 0 - prep the cell traces
        if not hasattr(self, "f_cells"):
            if 'suite_2p' in self.data_paths.keys():
                self.load_suite2p()
            if 'caiman' in self.data_paths.keys():
                self.load_caiman()
        self.norm_fcells = arrutils.norm_fdff(self.f_cells)
        self.zdiff_cells = [arrutils.zdiffcell(z) for z in self.f_cells]

        # 1 - find bad frames, make sure this exists first
        try:
            self.badframes_arr = np.array(np.load(Path(self.folder_path).joinpath('bad_frames.npy')))
        except: 
            print('find bad frames and re run suite2p')
            photostimulation.save_badframes_arr(self)
        photostimulation.find_no_baseline_frames(self, no_planes)

        # 2 - id the stim sites and save the raw traces
        self.stim_sites_df = photostimulation.identify_stim_sites(self, rotate, planes_stimed = stimmed_planes)
        self.raw_traces, self.points = photostimulation.collect_raw_traces(self)

        # 3 - id the stimulated cells based on distance
        self.identify_stim_cells()

    def identify_stim_cells(self):
        '''
        Identify the suite2p cells that are stimulated in the dataset from coordinates of stim sites
        Returns:
        closest_coord_list = coordinates of the closest cell to the stim site
        closest_cell_id_list = the cell id of the closest cell to the stim site
        '''

        points_stim = [[int(self.stim_sites_df.x_stim.iloc[i]), int(self.stim_sites_df.y_stim.iloc[i])]
                       for i in range(len(self.stim_sites_df))] # stimulated points

        closest_coord_list = []
        closest_cell_id_list = []
        all_points = self.return_cell_rois(range(len(self.f_cells)))
        for p in points_stim:
            closest_coord, closest_cell_id = coordutils.closest_coordinates(p[0], p[1], all_points)
            closest_coord_list.append(closest_coord)
            closest_cell_id_list.append(closest_cell_id)

        return closest_coord_list, closest_cell_id_list

        
### THESE FXNS DO NOT WORK WELL but keeping for future iterations ###
    def activity_distances(self, single_cell):
        cell_nums = [c for c in range(len(self.normf_cells))]
        cell_rois = self.return_cell_rois(c for c in cell_nums)
        
        self.activity_dist_df = pd.DataFrame({'cell_num': cell_nums, 'cell_rois': cell_rois})

        self.activity_dist_df["avg_corr"] = pd.Series(dtype='int')
        for c in range(len(self.activity_dist_df)):
            coorVal_lst = [self.pscorr_dict[num][c] for num in range(len(self.photostim_events))]
            self.activity_dist_df["avg_corr"][c] = np.nanmean(coorVal_lst)
        
        stimcell_lst = self.stimulated_cells_df.stim_cell.values
        stimcell_lst = stimcell_lst[~np.isnan(stimcell_lst)]
        stimulated_corr_lst = [cell_rois[int(r)] for r in stimcell_lst]

        dists_lst = []
        if single_cell == True:
            for _s, s in enumerate(stimulated_corr_lst):
                if int(s[1]) < 400:
                    coor_lst = s
                    self.single_stim_cellnum = self.stimulated_cells_df.stim_cell.iloc[_s]
                else:
                    pass
        else:
            coor_lst = stimulated_corr_lst

        for r in cell_rois:
            dists = [np.linalg.norm(np.array(coors) - np.array(r)) for coors in coor_lst]
            dists_lst.append(min(dists))
        self.activity_dist_df.loc[:, 'dist'] = dists_lst

        post_avgs = []
        pre_avgs = []
        for c in self.activity_dist_df.cell_num:
            post_lst = []
            pre_lst = []
            for num in range(len(self.photostim_events)):
                pre_lst.append(self.photostim_resp_dict[num][c][:-self.stim_offsets[0]])
                post_lst.append(self.photostim_resp_dict[num][c][-self.stim_offsets[0]:])
            pre_avgs.append(np.nanmean(pre_lst, axis = 0))
            post_avgs.append(np.nanmean(post_lst, axis = 0)) # average for one cell, added to master list

        self.activity_dist_df['pre_stim_avgs'] = np.nanmean(pre_avgs, axis = 1)
        self.activity_dist_df['post_stim_avgs'] = np.nanmean(post_avgs, axis = 1)

        save_path = Path(self.xml_file).parents[0].joinpath("activity_distances.hdf")
        self.activity_dist_df.to_hdf(save_path, key="act")
        print('saved activity dist df')

        return self.activity_dist_df

class TailTrackedFish(BaseFish):
    def __init__(
        self,
        tail_key="tail",  # key to find tail data
        tail_offset=2,
        thresh=0.7,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.add_tail_paths(tail_key)
        self.tail_df = pd.read_hdf(self.data_paths["tail"])

        if 'frame' not in self.tail_df.columns:
            if self.frametimes_df.time.values[0] > self.frametimes_df.time.values[-1]: #overnight
                self.tail_df = pd.concat([self.tail_df[(self.tail_df.t_dt > self.frametimes_df.time.values[0])],
                    self.tail_df[(self.tail_df.t_dt < self.frametimes_df.time.values[-1])]])
            else:
                self.tail_df = self.tail_df[(self.tail_df.t_dt > self.frametimes_df.time.values[0]) &
                                                (self.tail_df.t_dt < self.frametimes_df.time.values[-1])]
            self.tail_df = self.tag_frames_to_df(self.frametimes_df, self.tail_df, 't_dt')
            self.tail_df.to_hdf(self.data_paths['tail'], key='tail')

        #self.bout_finder(sig=5, width=None, prominence=7)

    def add_tail_paths(self, tail_key):
        try:
            with os.scandir(self.folder_path) as entries:
                for entry in entries:
                    if tail_key in entry.name:
                        self.data_paths["tail"] = Path(entry.path)
        except KeyError:
            print("failed to find tail data")

        return

    def bout_finder(
        self, sig=5, width=None, prominence=7
    ):
        from scipy.signal import find_peaks
        import scipy.ndimage
        # sig = sigma for gaussian filter on the tail data
        # interpeak_dst = frames of tail data info, first value is minimum between peaks, second valus is maximum length of whole bout
        # tail deflection sum from central axis of fish, filtered with gaussian fit
        if width is None:
            width = [0, 500]

        filtered_deflections = scipy.ndimage.gaussian_filter(self.tail_df["/'TailLoc'/'TailDeflectSum'"].values, sigma=sig)

        peak_deflection, peaks = scipy.signal.find_peaks(
            abs(filtered_deflections), # doing absolute to find highest peaks regardless of direction
            prominence=prominence,
            width=width,
        )
        # get bout peaks
        leftofPeak = peaks[
            "left_ips"
        ]  # Interpolated positions of a horizontal line’s left and right junction points at each evaluation height
        rightofPeak = peaks["right_ips"]
        peak_pts = np.stack([leftofPeak, rightofPeak], axis=1)
        bout_start = []
        bout_end = []
        for n in range(len(peak_pts)):
            while n < len(peak_pts) - 2: #getting number of oscillations, right now is 1
                current_right = peak_pts[n][1]
                next_left = peak_pts[n + 1][0]
                diff = next_left - current_right
                # if current right + minimum is less than the next left its good
                # if interpeak distance minimum and interpeak distance maximum are met, then add the peak
                if (current_right <= next_left) & (diff < 200):
                    bout_end.append(int(peak_pts[n + 1][1]))
                    bout_start.append(int(peak_pts[n][0]))
                    n += 1
                else:
                    n += 1
                    break

        # accounts for interbout distance, left and right of each peak in filtered tail deflection data ("/'TailLoc'/'TailDeflectSum'")
        self.new_peak_pts = np.stack(
            [bout_start, bout_end], axis=1
        )  # all peaks in tail data
        if hasattr(self, "tail_stimulus_df"):
            tail_ind_start = self.tail_stimulus_df.iloc[0].tail_ind_start
            tail_ind_stop = self.tail_stimulus_df.iloc[-2].tail_ind_end
        else:  # if you don't have stimulus file
            tail_ind_start = self.tail_df.iloc[0].frame
            tail_ind_stop = self.tail_df.iloc[-2].frame

        ind_0 = np.where(self.new_peak_pts[:, 0] >= tail_ind_start)[0][0]
        ind_1 = np.where(self.new_peak_pts[:, 1] <= tail_ind_stop)[0][-1]
        pts_during_tail = self.new_peak_pts[
                          ind_0:ind_1
                          ]  # peaks only within the stimuli presentation

        pts_uniq = [] #only gathering unique bouts
        for i in pts_during_tail.tolist():
            if pts_uniq.__contains__(i):
                pass
            else:
                pts_uniq.append(i)

        # # making sure that all relevant peaks don't overlap with others
        # # need to run this function a few times because sometimes the peaks have many overlapping left/rights
        # # in future build a function that can check how many overlapping peaks and then run fxn according to that...
        pts_uniq_2 = arrutils.remove_nearest_vals(pts_uniq)
        pts_uniq_3 = arrutils.remove_nearest_vals(pts_uniq_2)
        # # pts_uniq_4 = arrutils.remove_nearest_vals(pts_uniq_3)
        self.relevant_pts = arrutils.remove_nearest_vals(pts_uniq_3)
        # self.relevant_pts = pts_uniq

        dict_info = {}
        rnge = 400
        z_thresh = 2
        zscored_tail = arrutils.zscoring(self.tail_df["/'TailLoc'/'TailDeflectSum'"].values)
        # making sure that I am capturing the whole bout based on what is the most significant z scored tail change before or after a peak point
        for bout_ind, pts in enumerate(self.relevant_pts):
            if bout_ind not in dict_info.keys():
                dict_info[bout_ind] = {}

            window = zscored_tail[pts[0] - rnge: pts[1] + rnge]
            start_not_pts = window[0:rnge]
            end_not_pts = window[len(window)- rnge:len(window)]
            conditions = [start_not_pts, end_not_pts]
            for c, arr in enumerate(conditions):
                _, pval = scipy.stats.ttest_ind(arr, window, nan_policy = 'omit')
                if pval > 0.05:
                    frame_start = self.tail_df.iloc[:, -1].values[pts[0]]
                    frame_end = self.tail_df.iloc[:, -1].values[pts[1]]
                else:
                    for q, r in enumerate(arr):
                        if c == 0:
                            if abs(r) > z_thresh:
                                frame_start = self.tail_df.iloc[:, -1].values[pts[0] - (rnge - q)]
                                break
                        else:
                            if abs(r) > z_thresh:
                                frame_end = self.tail_df.iloc[:, -1].values[pts[1] + q] # don't break since I want the last occurance

            bout_angle = np.sum(self.tail_df["/'TailLoc'/'TailDeflectSum'"][(self.tail_df["frame"] >= frame_start) & (self.tail_df["frame"] <= frame_end)].values)  # total bout angle
            dict_info[bout_ind]["bout_angle"] = bout_angle
            dict_info[bout_ind]["image_frames"] = frame_start, frame_end

        self.tail_bouts_df = pd.DataFrame.from_dict(dict_info, "index")
        self.tail_bouts_df.loc[:, "bout_dir"] = np.zeros(self.tail_bouts_df.shape[0])
        self.tail_bouts_df.drop(self.tail_bouts_df[(self.tail_bouts_df.bout_angle == 0.0) | (abs(self.tail_bouts_df.bout_angle) > 100000)].index, inplace = True)
        self.tail_bouts_df["bout_dir"][self.tail_bouts_df["bout_angle"] > 0] = "left"
        self.tail_bouts_df["bout_dir"][self.tail_bouts_df["bout_angle"] < 0] = "right"
        self.tail_bouts_df.reset_index(drop=True, inplace = True)
        # tail_bouts_df has bout indices, frames from image frametimes, and bout direction
        return self.tail_bouts_df


class WorkingFish(VizStimFish):
    def __init__(self, corr_threshold=0.65, bool_data_type = 'normf', stim_order = None, ref_image=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if "move_corrected_image" not in self.data_paths:
            raise TankError
        self.corr_threshold = corr_threshold
        if not hasattr(self, "f_cells"):
            print('loading cells in working fish again?')
            if 'suite_2p' in self.data_paths.keys():
                self.load_suite2p()
            if 'caiman' in self.data_paths.keys():
                self.load_caiman()
            self.is_cell()

        self.zdiff_stim_dict, self.zdiff_err_dict, self.zdiff_neuron_dict = self.build_stimdicts(self.zdiff_cells)
        self.normf_stim_dict, self.normf_err_dict, self.normf_neuron_dict = self.build_stimdicts(self.normf_cells)
        self.f_stim_dict, self.f_err_dict, self.f_neuron_dict = self.build_stimdicts(self.f_cells)

        self.build_stimdicts_extended_zdiff()
        self.build_stimdicts_extended_normf()

        self.build_booldf_corr()
        self.build_booldf_baseline()
        self.build_booldf_cluster()

    def build_stimdicts_extended_zdiff(self):
        # makes an array of z-scored calcium responses for each stim (not median)
        self.extended_responses_zdiff = {i: {} for i in self.stimulus_df.stim_name.unique()}
        for stim in self.stimulus_df.stim_name.unique():
            arrs = arrutils.subsection_arrays(
                self.stimulus_df[self.stimulus_df.stim_name == stim].frame.values,
                self.offsets,
            )

            for n, nrn in enumerate(self.zdiff_cells):
                resp_arrs = []
                for arr in arrs:
                    try:
                        resp_arrs.append(arrutils.pretty(nrn[arr], 2))
                    except IndexError:
                        pass
                
                self.extended_responses_zdiff[stim][n] = resp_arrs

    def build_stimdicts_extended_normf(self):
        # makes an array of normalized calcium responses for each stim (not median)
        self.extended_responses_normf = {i: {} for i in self.stimulus_df.stim_name.unique()}
        for stim in self.stimulus_df.stim_name.unique():
            arrs = arrutils.subsection_arrays(
                self.stimulus_df[self.stimulus_df.stim_name == stim].frame.values,
                self.offsets,
            )
            #normcells = arrutils.norm_fdff(self.f_cells)
            for n, nrn in enumerate(self.normf_cells):
                resp_arrs = []
                try:
                    for arr in arrs:
                        # resp_arrs.append(arrutils.pretty(nrn[arr], 2))
                        resp_arrs.append(nrn[arr])
                except:
                    pass
                self.extended_responses_normf[stim][n] = resp_arrs

    def build_stimdicts(self, traces):
        # makes an median value (can change what response type) of z-scored calcium response for each neuron for each stim
        self.stimulus_df = stimuli.validate_stims(self.stimulus_df, self.f_cells)
        stim_dict = {i: {} for i in self.stimulus_df.stim_name.unique()}
        err_dict = {i: {} for i in self.stimulus_df.stim_name.unique()}

        for stim in self.stimulus_df.stim_name.unique():
            arrs = arrutils.subsection_arrays(
                self.stimulus_df[(self.stimulus_df.stim_name == stim) & (self.stimulus_df.frame > 0)].frame.values,
                self.offsets,
            )#isolate interest time period after stim onset

            for n, nrn in enumerate(traces):
                resp_arrs = []
                for arr in arrs:
                    if arr[-1] < len(traces[0]): # making sure the arr is not longer than the cell trace (frames)
                        resp_arrs.append(nrn[arr])

                stim_dict[stim][n] = np.nanmean(resp_arrs, axis=0)
                err_dict[stim][n] = np.nanstd(resp_arrs, axis=0) / np.sqrt(
                    len(resp_arrs)
                )

        neuron_dict = {}
        for neuron in stim_dict[
            "forward"
        ].keys():  # generic stim to grab all neurons
            if neuron not in neuron_dict.keys():
                neuron_dict[neuron] = {}

            for stim in self.stimulus_df.stim_name.unique():
                if self.r_type == "median":
                    neuron_dict[neuron][stim] = np.nanmedian(
                        stim_dict[stim][neuron][
                            -self.offsets[0] : -self.offsets[0] + self.stim_offset
                        ]
                    )
                elif self.r_type == "peak":
                    neuron_dict[neuron][stim] = np.nanmax(
                        stim_dict[stim][neuron][
                            -self.offsets[0] : -self.offsets[0] + self.stim_offset
                        ]
                    )
                elif self.r_type == "mean":
                    neuron_dict[neuron][stim] = np.nanmean(
                        stim_dict[stim][neuron][
                            -self.offsets[0] : -self.offsets[0] + self.stim_offset
                        ]
                    )
                else:
                    neuron_dict[neuron][stim] = np.nanmedian(
                        stim_dict[stim][neuron][
                            -self.offsets[0] : -self.offsets[0] + self.stim_offset
                        ]
                    )
        return stim_dict, err_dict, neuron_dict

    def build_booldf_corr(self, stim_arr=None, zero_arr=True, force=False):
        if hasattr(self, "booldf"):
            if not force:
                return

        if not stim_arr:
            provided = False
        else:
            provided = True

        corr_dict = {}
        bool_dict = {}
        for stim in self.zdiff_stim_dict.keys():
            if stim not in bool_dict.keys():
                bool_dict[stim] = {}
                corr_dict[stim] = {}
            for nrn in self.zdiff_stim_dict[stim].keys():
                cell_array = self.zdiff_stim_dict[stim][nrn]
                if zero_arr:
                    cell_array = np.clip(cell_array, a_min=0, a_max=99)
                if not provided:
                    stim_arr = np.zeros(len(cell_array))
                    stim_arr[
                        -self.offsets[0] + 1 : -self.offsets[0] + self.stim_offset - 2
                    ] = 3
                    stim_arr = arrutils.pretty(stim_arr, 3)
                corrVal = round(np.corrcoef(stim_arr, cell_array)[0][1], 3)

                corr_dict[stim][nrn] = corrVal
                bool_dict[stim][nrn] = corrVal >= self.corr_threshold
        self.zdiff_corr_booldf = pd.DataFrame(bool_dict)
        self.zdiff_corrdf = pd.DataFrame(corr_dict)

        #booldf_allneuron = self.booldf
        #self.booldf = self.booldf.loc[self.booldf.sum(axis=1) > 0]

    def build_booldf_baseline(self):
        baseline_frames = sorted(np.array([np.add(self.stimulus_df['frame'], subtract) for subtract in
                                           range(self.baseline_offset, 0)]).flatten().tolist())
        baseline_normf = pd.DataFrame(self.normf_cells).iloc[:, baseline_frames]
        baseline_boundary_normf = np.add(baseline_normf.mean(axis=1), np.multiply(baseline_normf.std(axis=1), 1.8))
        bool_dict = {}
        for stim in self.normf_stim_dict.keys():
            if stim not in bool_dict.keys():
                bool_dict[stim] = {}
            for nrn in self.normf_stim_dict[stim].keys():
                cell_array = self.normf_stim_dict[stim][nrn]
                bool_dict[stim][nrn] = np.mean(cell_array) > baseline_boundary_normf[nrn]
            # for nrn in self.extended_responses_normf[stim].keys():
            #     acc = np.zeros(len(self.extended_responses_normf[stim][nrn]))
            #     for occurance in range(0, len(self.extended_responses_normf[stim][nrn])):
            #         acc[occurance] = np.mean(self.extended_responses_normf[stim][nrn][occurance]) > baseline_boundary_normf[nrn]
            #     bool_dict[stim][nrn] = np.sum(acc) > len(self.extended_responses_normf[stim][nrn]) * 0.5
        self.normf_baseline_booldf = pd.DataFrame(bool_dict)

    def build_booldf_cluster(self):
        from sklearn import mixture

        boundary = np.zeros(self.normf_cells.shape[0])
        for nrn in range(0, self.normf_cells.shape[0]):
            neuron = np.nan_to_num(self.normf_cells[nrn])
            gm = mixture.GaussianMixture(n_components=2, random_state=0).fit(pd.DataFrame(neuron))
            if gm.means_[0] > gm.means_[1]:
                calm = 1
            else:
                calm = 0
            boundary[nrn] = (gm.means_[calm] + 3 * np.sqrt(gm.covariances_[calm]))[0][0]
        bool_dict = {}
        for stim in self.normf_stim_dict.keys():
            if stim not in bool_dict.keys():
                bool_dict[stim] = {}
            # for nrn in self.normf_stim_dict[stim].keys():
            #     cell_array = self.normf_stim_dict[stim][nrn]
            #     bool_dict[stim][nrn] = np.mean(cell_array) > boundary[nrn]
            for nrn in self.extended_responses_normf[stim].keys():
                acc = np.zeros(len(self.extended_responses_normf[stim][nrn]))
                for occurance in range(0, len(self.extended_responses_normf[stim][nrn])):
                    acc[occurance] = np.mean(self.extended_responses_normf[stim][nrn][occurance]) > boundary[nrn]
                bool_dict[stim][nrn] = np.sum(acc) == len(self.extended_responses_normf[stim][nrn])
        self.normf_cluster_booldf = pd.DataFrame(bool_dict)

    def make_computed_image_data(self, neuron_dict, booldf, colorsumthresh=1, booltrim=False):
        #if not hasattr(self, "neuron_dict"):
        #    self.build_stimdicts()
        xpos = []
        ypos = []
        colors = []
        neurons = []

        for neuron in neuron_dict.keys():
            if booltrim:
                if not hasattr(self, "booldf"):
                    self.build_booldf()
                if neuron not in self.booldf.index:
                    continue
            myneuron = neuron_dict[neuron]
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
                xloc, yloc = self.return_cell_rois(neuron)[0]

                xpos.append(xloc)
                ypos.append(yloc)
                colors.append(fullcolor)
                neurons.append(neuron)
        return xpos, ypos, colors, neurons

    def make_computed_image_data_ref(self, colorsumthresh=1, booltrim=False):
        if not hasattr(self, "neuron_dict"):
            self.build_stimdicts()
        if not hasattr(self, "x_pts"):
            raise (TankError, "need processed x_pts present")

        xpos = []
        ypos = []
        colors = []
        neurons = []

        for neuron in self.neuron_dict.keys():
            if booltrim:
                if not hasattr(self, "booldf"):
                    self.build_booldf()
                if neuron not in self.booldf.index:
                    continue
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
                yloc = self.y_pts[neuron]
                xloc = self.x_pts[neuron]
                # yloc, xloc = self.return_cell_rois(neuron)[0]

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

    def return_degree_vectors(self, neurons, type):
        import angles

        if type == 'normf_baseline':
            booldf = self.normf_baseline_booldf
            neuron_dict = self.normf_neuron_dict
        elif type == 'normf_cluster':
            booldf = self.normf_cluster_booldf
            neuron_dict = self.normf_neuron_dict
        elif type == 'zdiff_corr':
            booldf = self.zdiff_corr_booldf
            neuron_dict = self.zdiff_neuron_dict
        bool_monoc = booldf[constants.monocular_dict.keys()]
        monoc_bool_neurons = bool_monoc.loc[bool_monoc.sum(axis=1) > 0].index.values
        monoc_valid_neurons = [i for i in monoc_bool_neurons if i in neurons]
        bool_neurons = booldf.loc[booldf.sum(axis=1) > 0].index.values
        valid_neurons = [i for i in bool_neurons if i in neurons]

        thetas = []
        thetavals = []
        degree_ids_dict = {key: None for key in valid_neurons}
        degree_responses_dict = {key: None for key in valid_neurons}
        for n in valid_neurons:
            neuron_response_dict = neuron_dict[n]
            degree_ids = [constants.deg_dict[i] for i in neuron_response_dict.keys()]
            degree_responses = [
                np.clip(i, a_min=0, a_max=999)
                for i in neuron_response_dict.values()]
            degree_ids_dict[n] = degree_ids
            degree_responses_dict[n] = degree_responses
            if n in monoc_valid_neurons:
                mononeuron_response_dict = {
                k: v
                for k, v in neuron_response_dict.items()
                if k in constants.monocular_dict.keys()}
                monoc_degree_ids = [constants.deg_dict[i] for i in mononeuron_response_dict.keys()]
                monoc_degree_responses = [
                    np.clip(i, a_min=0, a_max=999)
                    for i in mononeuron_response_dict.values()]
                theta = angles.weighted_mean_angle(monoc_degree_ids, monoc_degree_responses)
                thetaval = np.nanmean(monoc_degree_responses)
            else:
                theta = np.nan
                thetaval = np.nan
            thetas.append(theta)
            thetavals.append(thetaval)

        return thetas, thetavals, degree_ids_dict, degree_responses_dict



class WorkingFish_Tail(WorkingFish, TailTrackedFish):
    """
    utilizes tail tracked fish data with visual stimuli
    """

    def __init__(
        self,
        corr_threshold=0.65,
        bout_window=(-10, 10),
        bout_offset=3,
        percent=0.4,
        num_resp_neurons=15,
        ref_image=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bout_window = bout_window  # complete frames to right and left you want to be able to visualize
        self.bout_offset = bout_offset  # how many frames to right and left you want to analyze as responses in relation to bout
        self.percent = percent  # the top percentage that you will be collecting bouts to be called "responsive", so 0.4 = 40%
        self.num_resp_neurons = num_resp_neurons

        if "move_corrected_image" not in self.data_paths:
            raise TankError
        self.corr_threshold = corr_threshold

        if ref_image is not None:
            self.reference_image = ref_image

        # self.diff_image = self.make_difference_image()

        # if ~hasattr(self, "f_cells"):
        #     if 'suite_2p' in self.data_paths.keys():
        #         self.load_suite2p()
        #     if 'caiman' in self.data_paths.keys():
        #         self.load_caiman()
        #     self.is_cell()
        # if hasattr(self, "tail_stimulus_df"):
        #     self.stimulus_df = stimuli.validate_stims(self.stimulus_df, self.normf_cells)
        #     self.build_stimdicts()
        # else:
        #     pass
        #self.bout_locked_dict()
        #self.single_bout_avg_neurresp()
        #self.avg_bout_avg_neurresp()
        #self.neur_responsive_trials()
        #self.build_timing_bout_dict()

    def make_heatmap_bout_count(
        self,
    ):  # to visualize bout counts per stimulus type if have different velocities
        import seaborn as sns
        import matplotlib.pyplot as plt

        df_list = []
        for stim in range(len(self.tail_stimulus_df)):
            a = self.tail_stimulus_df.iloc[stim]
            q = a.img_ind_start
            d = a.img_ind_end
            no_bouts = np.array(
                (list(zip(*self.tail_bouts_df.image_frames.values))[0] >= q)
                & (list(zip(*self.tail_bouts_df.image_frames.values))[1] <= d)
            )
            bout_count = np.where(no_bouts == True)[0].shape[0]
            df = pd.DataFrame({"stim_name": [a.stim_name], "bout_count": [bout_count]})
            if a.velocity:
                v = a.velocity
                df["velocity"] = v
                df_list.append(df)
        all_dfs = pd.concat(df_list).reset_index(drop=True)
        df1 = all_dfs.groupby(["stim_name", "velocity"], sort=False).agg(["mean"])
        df1.columns = df1.columns.droplevel(0)
        df1.reset_index(inplace=True)

        heatmap_data = pd.pivot_table(
            df1, values="mean", index=["stim_name"], columns="velocity"
        )
        sns.heatmap(heatmap_data, cmap=sns.color_palette("Blues", as_cmap=True))
        plt.xlabel("Velocity (m/s)", size=14)
        plt.ylabel("Motion Direction", size=14)
        plt.title(" Bout Count/Stim", size=14)
        plt.tight_layout()

    def bout_locked_dict(
        self,
    ):  # collecting means of some frames before and after bouting split into each bout
        # bout_window is the frames before and after the bout that you are collecting
        self.zdiff_cells = [arrutils.zdiffcell(i) for i in self.f_cells]
        self.bout_zdiff_dict = {i: {} for i in range(len(self.tail_bouts_df))}

        for bout in range(len(self.tail_bouts_df)):
            arrs = arrutils.subsection_arrays(
                np.array([self.tail_bouts_df.image_frames[bout][0]], dtype=int),
                offsets=(self.bout_window),
            )
            for n, nrn in enumerate(self.zdiff_cells):
                resp_arrs = []
                for arr in arrs:
                    resp_arrs.append(arrutils.pretty(nrn[arr], 2))
                self.bout_zdiff_dict[bout][
                    n
                ] = resp_arrs  # for each bout, this is the array of each neuron

        return self.bout_zdiff_dict

    def single_bout_avg_neurresp(self):
        # make df and adding average and peak responses to dictionary with arrays of each neuron response with bout
        self.bout_zdiff_df = pd.DataFrame(self.bout_zdiff_dict)
        all_means = []
        all_peak = []
        for i in range(len(self.bout_zdiff_df)):
            if i == range(len(self.bout_zdiff_df))[-1]:
                all_arrays_one_neur = [
                    item
                    for sublist in self.bout_zdiff_df.iloc[-1:].values
                    for item in sublist
                ]
                all_means.append((np.nanmean(all_arrays_one_neur)))
                all_peak.append((np.nanmax(all_arrays_one_neur)))

            else:
                all_arrays_one_neur = [
                    item
                    for sublist in self.bout_zdiff_df.iloc[i : i + 1].values
                    for item in sublist
                ]
                all_means.append((np.nanmean(all_arrays_one_neur)))
                all_peak.append((np.nanmax(all_arrays_one_neur)))

        self.bout_zdiff_df["all_avg_resp"] = all_means
        self.bout_zdiff_df["all_peak_resp"] = all_peak

        # self.most_resp_bout_zdiff_df = self.bout_zdiff_df[self.bout_zdiff_df.overall_peak_resp > thresh_resp] # taking top neurons based on threshold
        self.most_resp_bout_zdiff_df = self.bout_zdiff_df.sort_values(
            ["all_peak_resp"], ascending=False
        )[
            0:self.num_resp_neurons
        ]  # taking top resp neurons
        self.responsive_neuron_ids = self.most_resp_bout_zdiff_df.index.values.tolist()
        self.most_resp_bout_avg = {}
        if "all_avg_resp" in self.most_resp_bout_zdiff_df.columns:
            sub_bout_zdiff_df = self.most_resp_bout_zdiff_df.drop(
                columns=["all_avg_resp", "all_peak_resp"]
            )
            for b in sub_bout_zdiff_df:
                if b not in self.most_resp_bout_avg.keys():
                    self.most_resp_bout_avg[b] = {}
                    one_bout = sub_bout_zdiff_df[b]
                    one_bout_arrs = []
                    for x in one_bout:
                        one_bout_arrs.append(x[0])
                        one_bout_list = [l.tolist() for l in one_bout_arrs]
                        one_bout_avg = np.mean(np.array(one_bout_list), axis=0)
                self.most_resp_bout_avg[b] = one_bout_avg

        return self.most_resp_bout_zdiff_df, self.most_resp_bout_avg

    def avg_bout_avg_neurresp(self):
        from scipy import stats
        self.avgbout_avgneur_dict = {}
        all_bout_len_avgs = []
        for bout_no in self.most_resp_bout_avg.keys():
            bout_len = (
                self.tail_bouts_df.iloc[bout_no].image_frames[1]
                - self.tail_bouts_df.iloc[bout_no].image_frames[0]
            )
            all_bout_len_avgs.append(bout_len)
            if bout_no not in self.avgbout_avgneur_dict.keys():
                self.avgbout_avgneur_dict[bout_no] = {}
            total_bout_arr = self.most_resp_bout_avg[bout_no]
            self.avgbout_avgneur_dict[bout_no] = total_bout_arr
        self.avgbout_avgneur_df = pd.DataFrame(self.avgbout_avgneur_dict)

        self.avgbout_avgneur_df["mean"] = self.avgbout_avgneur_df.mean(axis=1)

        sem_lst = []
        std_lst = []
        for j in self.avgbout_avgneur_df.iloc[:,:-1].values:
            sem_lst.append(stats.sem(j))
            std_lst.append(np.std(j))
        self.avgbout_avgneur_df["sem"] = sem_lst
        self.avgbout_avgneur_df["std"] = std_lst

        self.one_bout_len_avg = np.mean(all_bout_len_avgs)

        return self.avgbout_avgneur_df, self.avgbout_avgneur_dict

    def neur_responsive_trials(self):
        # getting the top percentage of responsive neurons (calculated by taking the mean responses)
        self.responsive_trial_bouts = []
        rsp_before_lst = []
        rsp_after_lst = []
        for (
            event
        ) in (
            self.most_resp_bout_avg.keys()
        ):  # finding mean values before and after bout
            bout_no = int(event)
            bout_len = int(
                self.tail_bouts_df.iloc[bout_no].image_frames[1]
                - self.tail_bouts_df.iloc[bout_no].image_frames[0]
            )
            if self.r_type == "peak":  # takes the peak
                rsp_before = np.nanmax(
                    self.most_resp_bout_avg[bout_no][
                        -self.bout_window[0] - self.bout_offset : -self.bout_window[0]
                    ]
                )
                rsp_before_lst.append(rsp_before)
                try:
                    rsp_after = np.nanmax(self.most_resp_bout_avg[bout_no][int(-self.bout_window[0] + bout_len) :
                                                                           int(-self.bout_window[0] + bout_len + self.bout_offset)])
                except:
                    rsp_after = np.nan # indexing issue, so just skipping it
                rsp_after_lst.append(rsp_after)
            else:  # takes the average
                rsp_before = np.nanmean(self.most_resp_bout_avg[bout_no][-self.bout_window[0] - self.bout_offset : -self.bout_window[0]])
                rsp_before_lst.append(rsp_before)
                rsp_after = np.nanmean(self.most_resp_bout_avg[bout_no][int(-self.bout_window[0] + bout_len) :
                                                                        int(-self.bout_window[0] + bout_len + self.bout_offset)])
                rsp_after_lst.append(rsp_after)

        # max values before and after bout
        max_before = max(rsp_before_lst)
        max_after = max(rsp_after_lst)

        # grab trials that are in top % of max values
        for i, before_val in enumerate(rsp_before_lst):
            if before_val > ((1 - self.percent) * max_before):
                self.responsive_trial_bouts.append(i)
        for j, after_val in enumerate(rsp_after_lst):
            if after_val > ((1 - self.percent) * max_after):
                self.responsive_trial_bouts.append(j)
        self.responsive_trial_bouts = sorted(set(self.responsive_trial_bouts))

        return self.responsive_trial_bouts

    def build_timing_bout_dict(self):
        self.timing_bout_dict = {}

        for n, neuron in enumerate(
            self.most_resp_bout_zdiff_df[self.responsive_trial_bouts].index.values
        ):
            if neuron not in self.timing_bout_dict.keys():
                self.timing_bout_dict[neuron] = {}
            all_arrays_one_neur = [
                item
                for sublist in self.most_resp_bout_zdiff_df[self.responsive_trial_bouts]
                .iloc[n]
                .values
                for item in sublist
            ]
            for s, subset in enumerate(all_arrays_one_neur):
                bout_no = self.responsive_trial_bouts[s]
                bout_len = int(
                    self.tail_bouts_df.iloc[bout_no].image_frames[1]
                    - self.tail_bouts_df.iloc[bout_no].image_frames[0]
                )
                self.timing_bout_dict[neuron]["before"] = np.nanmean(
                    subset[
                        -self.bout_window[0] - self.bout_offset : -self.bout_window[0]
                    ]
                )
                self.timing_bout_dict[neuron]["after"] = np.nanmean(
                    subset[
                        -self.bout_window[0]
                        + bout_len : -self.bout_window[0]
                        + bout_len
                        + self.bout_offset
                    ]
                )
                if bout_len != 0:
                    self.timing_bout_dict[neuron]["during"] = np.nanmean(
                        subset[
                            -self.bout_window[0]
                            - bout_len : -self.bout_window[0]
                            + bout_len
                        ]
                    )
                else:
                    # for my slow imaging
                    self.timing_bout_dict[neuron]["during"] = np.nanmean(
                        subset[-self.bout_window[0] : -self.bout_window[0] + 1]
                    )

    def make_taildata_avgneur_plots(self):
        import matplotlib.pyplot as plt
        from mimic_alpha import mimic_alpha as ma

        self.avgbout_avgneur_df, self.avgbout_avgneur_dict = self.avg_bout_avg_neurresp()

        for bout_no in self.responsive_trial_bouts:
            total_bout = self.avgbout_avgneur_df.iloc[:,bout_no]
            bout_len = (
                self.tail_bouts_df.iloc[bout_no].image_frames[1]
                - self.tail_bouts_df.iloc[bout_no].image_frames[0]
            )

            fig, ax = plt.subplots(
                nrows=1, ncols=2, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 2]}
            )
            fig.suptitle(
                f"Bout {bout_no}, Most Responsive neurons (n = {len(self.responsive_neuron_ids)})"
            )

            # tail movement data
            start = self.tail_bouts_df.iloc[bout_no].image_frames[0]
            end = self.tail_bouts_df.iloc[bout_no].image_frames[1]

            # visual stimuli shading
            if hasattr(self, "tail_stimulus_df"):
                if self.tail_stimulus_df.stim_name.isin(
                    constants.baseBinocs
                ).any():  # if binocular stimuli
                    stimuli.stim_shader(self)
                elif 'velocity' in self.tail_stimulus_df.columns:  # if you want to plot velocity values with motion stim
                    self.tail_stimulus_df.loc[
                        :, "color"
                    ] = self.tail_stimulus_df.stim_name.astype(str).map(
                        constants.velocity_mono_dict
                    )  # adding color for plotting
                    for stim in range(len(self.tail_stimulus_df)):
                        a = self.tail_stimulus_df.iloc[stim]
                        q = a.img_ind_start
                        v = a.velocity
                        ax[1].axvspan(
                            q - 1,
                            q + self.stim_offset + 2,
                            color=ma.colorAlpha_to_rgb(a.color[v][0], a.color[v][1])[0],
                            label=f"{a.stim_name},{a.velocity}",
                        )
                else:
                    print("no visual stimulus shading")
            else:
                print("no visual stimulus in experiment")

            ax[1].plot(
                self.tail_df.iloc[:, -1].values,
                self.tail_df.iloc[:, 4].values,
                color="black",
            )  # plotting deflect sum
            ax[1].axvspan(start, end, ymin=0.9, ymax=1, color="red", alpha=1)
            ax[1].set_xlim(start + self.bout_window[0], end + self.bout_window[1])
            ax[1].set_xlabel("Frames (from imaging data)")
            ax[1].set_ylabel("Z score Tail Deflection Sum")
            ax[1].set_title("Tail behavior")

            # neural z score trace, with std
            x = np.arange(len(total_bout))
            std_error = np.std(total_bout.values)
            ax[0].plot(x, total_bout.values, 'k-')
            ax[0].fill_between(x, total_bout.values - std_error, total_bout.values + std_error, alpha = 0.3)
            ax[0].set_title("Z score neural activity with Std")
            ax[0].set_ylabel("Z score average")
            ax[0].set_xlabel("Frames (from imaging data)")
            ax[0].set_ylim(-1, 1)
            ax[0].axvspan(
                -self.bout_window[0],
                bout_len + -self.bout_window[0],
                color="red",
                alpha=0.5,
            )

    def make_oneneur_allbout_plots(self, neur_id, num_bouts = None, save = True):
        import matplotlib.pyplot as plt

        if num_bouts == None:
            num_bouts = len(self.responsive_trial_bouts)
        else:
            num_bouts = num_bouts

        for ind, n in enumerate(self.most_resp_bout_zdiff_df[self.responsive_trial_bouts].index):
            if n == neur_id:
                one_neur_responses = self.most_resp_bout_zdiff_df[self.responsive_trial_bouts].iloc[ind]

        fig, axs = plt.subplots(
            nrows=1,
            ncols=num_bouts + 1,
            sharex=True,
            # sharey=True,
            figsize=(7, 2),
        )
        fig.suptitle(f"Neuron #{neur_id} Response to bouts")
        axs = axs.flatten()
        bout_len_lst = []
        for n, neur in enumerate(one_neur_responses):
                bout_no = one_neur_responses.index[n]
                if bout_no in self.responsive_trial_bouts[:num_bouts]:
                    bout_len = (
                            self.tail_bouts_df.iloc[bout_no].image_frames[1]
                            - self.tail_bouts_df.iloc[bout_no].image_frames[0]
                    )
                    bout_len_lst.append(bout_len)
                    axs[n].axhline(y = 0, color = 'black', alpha=0.3, linestyle='--')
                    axs[n].plot(neur[0])
                    axs[n].set_title(f"Bout {bout_no}")
                    axs[n].set_ylim(-1.2, 1.2)
                    # marks the bout to be only one frame in time, might need to change with frame rate
                    axs[n].axvspan(
                        -self.bout_window[0],
                        -self.bout_window[0] + bout_len,
                        color="red",
                        alpha=0.5,
                        )
                    axs[n].axis("off")
                else:
                    pass
                    # print(f'not showing bout number {bout_no} here')

        averages = [
            item for sublist in one_neur_responses.values for item in sublist
        ]
        avg_arr = [l.tolist() for l in averages]
        one_neur_avg = np.mean(np.array(avg_arr), axis=0)
        one_neur_std = np.std(np.array(avg_arr), axis = 0)
        axs[-1].axhline(y = 0, color = 'black', alpha=0.3, linestyle='--')
        axs[-1].plot(one_neur_avg, "k-")
        axs[-1].fill_between(np.arange(one_neur_avg.shape[0]), one_neur_avg - one_neur_std, one_neur_avg + one_neur_std, alpha = 0.5)
        axs[-1].set_title("Mean")
        axs[-1].set_ylim(-1.2, 1.2)
        axs[-1].axvspan(
            -self.bout_window[0],
            -self.bout_window[0] + np.mean(bout_len_lst),
            color="red",
            alpha=0.5,
            )
        axs[-1].axis("off")
        fig.tight_layout()
        plt.show()
        new_path = Path(self.folder_path).joinpath(f'neur{neur_id}_{num_bouts}bouts.png')
        fig.savefig(new_path, dpi=600)
        print('saved')


    def make_indneur_indbout_plots(self):
        # plotting each individual neuron to a bout, then mean of the neuron to all bouts
        import matplotlib.pyplot as plt

        for v, vals in enumerate(
            self.most_resp_bout_zdiff_df[self.responsive_trial_bouts].index
        ):
            one_neur_responses = self.most_resp_bout_zdiff_df[
                self.responsive_trial_bouts
            ].iloc[v]
            fig, axs = plt.subplots(
                nrows=1,
                ncols=len(self.responsive_trial_bouts) + 1,
                sharex=True,
                sharey=True,
                figsize=(10, 2),
            )
            fig.suptitle(f"Neuron #{vals} Response to bouts")
            axs = axs.flatten()
            for n, neur in enumerate(one_neur_responses):
                bout_no = one_neur_responses.index[n]
                bout_len = (
                    self.tail_bouts_df.iloc[bout_no].image_frames[1]
                    - self.tail_bouts_df.iloc[bout_no].image_frames[0]
                )
                axs[n].plot(neur[0])
                axs[n].set_title(f"Bout {one_neur_responses.index[n]}")
                axs[n].set_ylim(-1, 1)
                # marks the bout to be only one frame in time, might need to change with framerate
                axs[n].axvspan(
                    -self.bout_window[0],
                    -self.bout_window[0] + bout_len,
                    color="red",
                    alpha=0.5,
                )
                axs[n].axis("off")

            averages = [
                item for sublist in one_neur_responses.values for item in sublist
            ]
            avg_arr = [l.tolist() for l in averages]
            one_neur_avg = np.mean(np.array(avg_arr), axis=0)
            one_neur_std = np.std(np.array(avg_arr), axis = 0)
            axs[-1].axhline(y = 0, color = 'black', alpha=0.3, linestyle='--')
            axs[-1].plot(one_neur_avg, "k-")
            axs[-1].fill_between(np.arange(one_neur_avg.shape[0]), one_neur_avg - one_neur_std, one_neur_avg + one_neur_std, alpha = 0.5)
            axs[-1].set_title("Mean")
            axs[-1].set_ylim(-1, 1)
            axs[-1].axvspan(
                -self.bout_window[0],
                -self.bout_window[0] + self.one_bout_len_avg,
                color="red",
                alpha=0.5,
            )
            axs[-1].axis("off")

            fig.tight_layout()
            plt.show()

    def make_avgneur_indbout_plots(self):
        # plotting neuron averages for each plot
        import matplotlib.pyplot as plt
        from scipy import stats

        if hasattr(self, "one_bout_len_avg"):
            pass
        else:
            self.avg_bout_avg_neurresp()

        self.responsive_trial_bout_df = self.avgbout_avgneur_df[
            self.responsive_trial_bouts
        ]

        self.responsive_trial_bout_df["mean"] = self.responsive_trial_bout_df.mean(
            axis=1
        )

        sem_lst = []
        std_lst = []
        for j in self.responsive_trial_bout_df.iloc[:,:-1].values:
            sem_lst.append(stats.sem(j))
            std_lst.append(np.std(j))
        self.responsive_trial_bout_df["sem"] = sem_lst
        self.responsive_trial_bout_df["std"] = std_lst

        fig1, axs1 = plt.subplots(
            nrows=1,
            ncols=len(self.responsive_trial_bouts) + 1,
            sharex=True,
            sharey=True,
            figsize=(10, 2),
        )
        fig1.suptitle(f'Mean Neural Response to each "responsive" bout')

        for m, bout_no in enumerate(self.responsive_trial_bouts):
            total_bout = self.responsive_trial_bout_df[bout_no]
            bout_len = (
                self.tail_bouts_df.iloc[bout_no].image_frames[1]
                - self.tail_bouts_df.iloc[bout_no].image_frames[0]
            )

            x = np.arange(len(total_bout))
            std = np.std(total_bout.values)
            axs1[m].plot(total_bout, 'k-')
            axs1[m].fill_between(x, total_bout.values - std, total_bout.values + std, alpha = 0.3)
            axs1[m].set_title(f"Bout #{bout_no}")
            axs1[m].set_ylim(-1, 1)
            # marks the bout to be only one frame in time, might need to change with framerate
            axs1[m].axvspan(
                -self.bout_window[0],
                -self.bout_window[0] + bout_len,
                color="red",
                alpha=0.5,
            )
            axs1[m].axis("off")

            fig1.tight_layout()

        axs1[-1].plot(self.responsive_trial_bout_df["mean"], 'k-')
        axs1[-1].fill_between(x, self.responsive_trial_bout_df['mean'].values - self.responsive_trial_bout_df['std'].values,
                         self.responsive_trial_bout_df['mean'].values + self.responsive_trial_bout_df['std'].values, alpha = 0.5)
        axs1[-1].axis("off")
        axs1[-1].axvspan(
            -self.bout_window[0],
            -self.bout_window[0] + self.one_bout_len_avg,
            color="red",
            alpha=0.5,
        )
        axs1[-1].set_ylim(-1, 1)
        axs1[-1].set_title("Mean")

        plt.show()

    def make_computed_image_bouttiming(self, colorsumthresh=0.4, size = 150, alpha = 0.9, annotate_ids = True):
        from matplotlib.lines import Line2D
        import matplotlib.pyplot as plt

        if hasattr(self, "timing_bout_dict"):
            pass
        else:
            self.build_timing_bout_dict()

        xpos = []
        ypos = []
        colors = []
        neurons = []

        for neuron in self.timing_bout_dict.keys():
            myneuron = self.timing_bout_dict[neuron]
            clr_longform = [
                val * np.clip(i, a_min=0, a_max=99)
                for timing, val in zip(myneuron.keys(), myneuron.values())
                if timing in constants.bout_timing_color_dict.keys()
                for i in constants.bout_timing_color_dict[timing]
            ]
            reds = clr_longform[::3]
            greens = clr_longform[1::3]
            blues = clr_longform[2::3]

            fullcolor = np.sum([reds, greens, blues], axis=1)

            if max(fullcolor) > 1.0:
                fullcolor /= max(fullcolor)
            fullcolor = np.clip(fullcolor, a_min=0, a_max=1.0)
            if np.sum(fullcolor) > colorsumthresh:
                yloc, xloc = self.return_cell_rois(int(neuron))[0]

                xpos.append(xloc)
                ypos.append(yloc)
                colors.append(fullcolor)
                neurons.append(neuron)

        fig, axs = plt.subplots(1, 1, figsize=(12, 12))

        axs.scatter(
            xpos, ypos, c=colors, alpha=alpha, s=size
        )  ## most responsive neurons active before or after

        if annotate_ids == True:
            for i, txt in enumerate(neurons):
                axs.annotate(txt, (xpos[i], ypos[i]), c='pink')

        axs.imshow(
            self.ops["refImg"],
            cmap="gray",
            alpha=1,
            vmax=np.percentile(self.ops["refImg"], 99.5),
        )
        axs.set_title(f"Top {len(neurons)} Responsive Neurons Before/During/After Bout")
        axs.axis("off")
        markers = [
            plt.Line2D([0, 0], [0, 0], color=color, marker="o", linestyle="")
            for color in constants.bout_timing_color_dict.values()
        ]
        plt.legend(markers, constants.bout_timing_color_dict.keys(), numpoints=1)

        return xpos, ypos, colors, neurons


class VolumeFish:
    def __init__(self):
        self.volumes = {}
        self.volume_inds = {}
        self.last_ind = 0
        self.iter_ind = -1

    def add_volume(self, new_fish, ind=None, fakevol=False):
        assert "fish" in str(
            new_fish
        ), "must be a fish"  #  isinstance sometimes failing??
        # assert isinstance(new_fish, BaseFish), "must be a fish" #  this is randomly buggin out
        
        newKey = new_fish.folder_path.name
        if fakevol:
            newKey = new_fish.folder_path.parents[1].name.split('-')[0].split('_')[1]

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


# %%
