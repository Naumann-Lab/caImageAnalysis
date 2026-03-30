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
import matplotlib.pyplot as plt
from bcdict import BCDict

import sys
sys.path.append(r'C:\Users\Kaitlyn\PyCharmProjects\imaging\caImageAnalysis')
# local imports
import constants
import angles
from utilities import pathutils, arrutils, roiutils, coordutils, plotutils
import stimuli
import process
import photostim_utils
import tailtracking
import bruker_images


class BaseFish:
    def __init__(
        self,
        folder_path,
        frametimes_key="frametimes",
        invert=False,
        bruker_invert = False,
        caiman_type = None,
        midnight_noon = "noon" # mark as midnight if the imaging was across 23 to 00 hrs
    ):
        self.folder_path = Path(folder_path)
        self.frametimes_key = frametimes_key
        self.midnight_noon_keyword = midnight_noon  

        self.invert = invert
        self.bruker_invert = bruker_invert

        self.process_filestructure(midnight_noon) # generates self.data_paths
        try:
            self.raw_text_frametimes_to_df(midnight_noon)  # generates self.frametimes_df
        except:
            print("failed to process frametimes from text")
        
        self.img_hz = self.hzReturner(self.frametimes_df) 
        
        if 'suite2p' in self.data_paths.keys():
            self.load_suite2p() # loads in suite2p paths
            self.is_cell()  # clean up cells (close to edge, not changing fluor, nan location, etc)
        if 'caiman' in self.data_paths.keys():
            self.load_caiman(caiman_type) # load in caiman data 
            self.is_cell() # clean up cells (close to edge, not changing fluor, nan location, etc)

        
    def process_filestructure(self, midnight_noon = "noon"):
        self.data_paths = {}
        with os.scandir(self.folder_path) as entries:
            for entry in entries:
                if entry.name.endswith(".tif"):
                    if "movement_corr" in entry.name:
                        self.data_paths["move_corrected_image"] = Path(entry.path)
                    elif "rotated" in entry.name:
                        self.data_paths["rotated_image"] = Path(entry.path)
                    elif "img_stack" in entry.name:
                        self.data_paths["image"] = Path(entry.path) 
                    else:
                        self.data_paths["original_image"] = Path(entry.path)

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
                        list = ['00' + time[2:] if time[:2] == '12' else time for time in list]
                        self.frametimes_df.time = [dt.strptime(time, "%H:%M:%S.%f").time() for time in list]

                elif os.path.isdir(entry.path):
                    if entry.name == "suite2p" or entry.name == "suite_2p":
                        self.data_paths["suite2p"] = Path(entry.path).joinpath("plane0")
                    if entry.name == 'caiman':
                        self.data_paths["caiman"] = Path(entry.path)
                    if entry.name == "original_image":
                        with os.scandir(entry.path) as imgdiver:
                            for poss_img in imgdiver:
                                if poss_img.name.endswith(".tif"):
                                    self.data_paths["original_image"] = Path(poss_img.path)

                elif entry.name.endswith(".npy"): # these are mislabeled so just flip here
                    if "xpts" in entry.name:
                        with open(entry.path, "rb") as f:
                            self.y_pts = np.load(f)
                    elif "ypts" in entry.name:
                        with open(entry.path, "rb") as f:
                            self.x_pts = np.load(f)
                    elif "mean_img" in entry.name:
                        self.mean_img = np.load(Path(entry.path))
                
                # bruker information/processing files
                elif entry.name.endswith("xml"):
                    if 'MarkPoints' in entry.name:
                        self.data_paths["ps_xml"] = Path(entry.path)
                    elif ("Voltage" in entry.name) and ('MarkPoints' not in entry.name):
                        self.data_paths["voltage_xml"] = Path(entry.path)
                    elif "Voltage" not in entry.name and 'MarkPoints' not in entry.name:
                        self.data_paths["info_xml"] = Path(entry.path)
                        self.um_per_px = bruker_images.get_micronstopixels_scale(self.data_paths['info_xml'])
                elif entry.name.endswith("env"):
                    self.data_paths["info_env"] = Path(entry.path)
                elif entry.name.endswith("csv") and 'Voltage' in entry.name:
                    self.data_paths["voltage_signal"] = Path(entry.path)
                elif ('output.txt' in entry.name) & ('pstim' not in entry.name): # from automated gui experiments
                    self.data_paths["ps_log"] = Path(entry.path)
                elif 'stim_sites' in entry.name:
                    self.data_paths["stim_sites"] = Path(entry.path)

        # moving over the original image to a separate folder
        if "image" in self.data_paths and "move_corrected_image" in self.data_paths:
            if (self.data_paths["image"].parents[0] == self.data_paths["move_corrected_image"].parents[0]):
                try:
                    pathutils.move_og_image(self.data_paths["image"])
                except:
                    print("failed to move original image out of folder")
        elif "image" in self.data_paths and "rotated_image" in self.data_paths:
            if (self.data_paths["image"].parents[0] == self.data_paths["rotated_image"].parents[0]):
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
        self.rescaled_img()

    def load_caiman(self, caiman_type):
        # make a ops['refImg'] to be used later, like with suite2p data
        if hasattr(self, "mean_img"):
            pass
        else:
            img = self.load_image()
            self.mean_img = np.nanmean(img[:1000], axis = 0)
            np.save(Path(self.folder_path).joinpath('mean_img.npy'), self.mean_img)
        self.ops = {'refImg': self.mean_img}

        self.iscell = np.load(
            self.data_paths["caiman"].joinpath("iscell.npy"), allow_pickle=True
        ).astype(bool)

        self.stats = np.load(
            self.data_paths["caiman"].joinpath("coordinates_dict.npy"), allow_pickle=True
        )

        if caiman_type == 'raw':
            if not self.data_paths["caiman"].joinpath("raw.npy").exists():
                process.gather_raw_traces_from_cnmf_output(self)
            self.f_cells = np.load(self.data_paths["caiman"].joinpath("raw.npy"))
        else:
            self.f_cells = np.load(self.data_paths["caiman"].joinpath("C.npy"))
        self.dff_cells = np.load(self.data_paths["caiman"].joinpath("F_dff.npy"), allow_pickle=True)
        self.rescaled_img()

    def is_cell(self, edge_margin = 20):
        """
        For all types of output:
        0 - Always remove cells that are within 20 pixels of the border (for standard 512 x 512 FOV)
        For caiman output:
        1 - Clean up the self.f_cells and self.dff_cells according to self.iscell
        2 - clean up cells that didn't change fluorscence throughout the trial at all
        3 - remove cells with any nan location values
        4 - if caiman data, then make sure the cells are within the brain region
        """
        # 0 - remove cells that are too close to the edge - for all sources
        height, width = self.ops['refImg'].shape
        edge_cell_indices = []
        for idx, location_info in enumerate(self.stats):
            x = location_info['xpix']
            y = location_info['ypix']
            # Compute min and max of the cell location
            xmin = np.min(x)
            xmax = np.max(x)
            ymin = np.min(y)
            ymax = np.max(y)
            # Check if any pixel touches the margin
            if (xmin < edge_margin or xmax > (width - edge_margin - 1) or ymin < edge_margin or ymax > (height - edge_margin - 1)):
                edge_cell_indices.append(idx)

            # Check if there is a nan location
            for value in location_info.values():
                if isinstance(value, float) and np.isnan(value):
                    edge_cell_indices.append(idx)
        iscell_index = [index for index in range(len(self.f_cells)) if index not in edge_cell_indices]

        try:
            self.load_saved_rois()
            if 'brain' in list(self.roi_dict.keys()): # if there is a large brain ROI
                print('brain ROI found')
                iscell_inbrain_index = np.array(self.return_cells_by_saved_roi('brain'))
                iscell_index_2 = np.intersect1d(np.array(iscell_index), iscell_inbrain_index)
                iscell_index = iscell_index_2
        except:
            print('no brain ROI found')

        self.f_cells = self.f_cells[iscell_index]
        self.stats = self.stats[iscell_index]
        # if hasattr(self, 'dff_cells'):
        #     self.dff_cells = self.dff_cells[iscell_index]

        if 'caiman' in self.data_paths.keys(): # this does not overwrite the original caiman output
            # 1 - is part of is cell index
            iscell_index = np.where(self.iscell)

            # 2 - cell is changing
            ischanging_index = np.where(np.amax(self.f_cells, 1) != np.amin(self.f_cells, 1))
            iscell_index = np.intersect1d(iscell_index, ischanging_index)
            # iscell_index = iscell_index[0]

            # 3 - remove cells with a nan location
            notcell_index = []
            for e, x in enumerate(self.stats):
                for value in x.values():
                    if isinstance(value, float) and np.isnan(value):
                        notcell_index.append(e)
            iscell_index = [index for index in iscell_index if index not in notcell_index]

            self.f_cells = self.f_cells[iscell_index]
            self.stats = self.stats[iscell_index]
            # if hasattr(self, 'dff_cells'):
            #     self.dff_cells = self.dff_cells[iscell_index]

            # 4 - with caiman data, make sure that these cells are within the brain region
            try:
                iscell_inbrain_index = np.array(self.return_cells_by_saved_roi('brain')) # if there is a good brain ROI
            except KeyError:
                self.draw_roi2('brain', overwrite=True) # in case you need to get the brain ROI again
                iscell_inbrain_index = np.array(self.return_cells_by_saved_roi('brain'))
            iscell_index_2 = np.intersect1d(np.array(iscell_index), iscell_inbrain_index)
            self.f_cells = self.f_cells[iscell_index_2]
            self.stats = self.stats[iscell_index_2]
            # self.dff_cells = self.dff_cells[iscell_index_2]
            print('completed iscell check')

    def return_cell_rois(self, cells):
        if isinstance(cells, int):
            cells = [cells]

        rois = []
        for cell in cells:
            ypix = self.stats[cell]["ypix"]
            xpix = self.stats[cell]["xpix"]
            mean_y = int(np.nanmean(ypix))
            mean_x = int(np.nanmean(xpix))
            rois.append([mean_x, mean_y])
        return rois
    
    def return_singlecell_rois(self, single_cell):
        single_cell = int(single_cell)
        ypix = self.stats[single_cell]["ypix"]
        xpix = self.stats[single_cell]["xpix"]
        mean_y = int(np.nanmean(ypix))
        mean_x = int(np.nanmean(xpix))
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

    def draw_roi(self, title="blank", overwrite=False, brightness = 50, contrast =30):
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

                self.ptlist.append((x, y))
            if event == 2:  # right click
                cv2.destroyAllWindows()

        cv2.namedWindow(f"roiFinder_{title}")

        cv2.setMouseCallback(f"roiFinder_{title}", roigrabber)

        # plot_img = np.array(img, "uint8")
        plot_img = np.array(img, "int16")
        plot_img = plot_img * (contrast/127+1) - contrast + brightness
        plot_img = np.clip(plot_img, 0, 255)
        plot_img = np.uint8(plot_img)

        cv2.imshow(f"roiFinder_{title}", plot_img)
        try:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            cv2.destroyAllWindows()

        self.save_roi(title, overwrite)
    
    def draw_roi2(self, title="blank", overwrite=False, brightness=50, contrast=30):
        
        # Make a copy of the reference image
        img = self.ops["refImg"].copy()
        self.ptlist = []
        
        # Brightness / contrast adjustment
        plot_img = np.int16(img)
        plot_img = plot_img * (contrast / 127 + 1) - contrast + brightness
        plot_img = np.clip(plot_img, 0, 255).astype(np.uint8)

        # ---------- Try OpenCV GUI ----------
        try:
            import cv2

            window_name = f"roiFinder_{title}"

            def roigrabber(event, x, y, flags, params):
                if event == cv2.EVENT_LBUTTONDOWN:  # left click
                    if self.ptlist:
                        cv2.line(plot_img, self.ptlist[-1], (x, y), (255, 255, 255), 2)
                    cv2.circle(plot_img, (x, y), 3, (0, 255, 0), -1)
                    self.ptlist.append((x, y))
                    cv2.imshow(window_name, plot_img)

                elif event == cv2.EVENT_RBUTTONDOWN:  # right click closes polygon
                    if len(self.ptlist) > 2:
                        cv2.line(plot_img, self.ptlist[-1], self.ptlist[0], (255, 255, 255), 2)
                        cv2.imshow(window_name, plot_img)
                    cv2.waitKey(500)
                    cv2.destroyAllWindows()

            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, roigrabber)

            cv2.imshow(window_name, plot_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"[INFO] OpenCV GUI not available ({e}). Falling back to Matplotlib ROI selection.")
            import matplotlib
            matplotlib.use("Qt5Agg")

            import matplotlib.pyplot as plt
            from matplotlib.widgets import PolygonSelector

            def onselect(verts):
                self.ptlist = [(int(x), int(y)) for x, y in verts]
                plt.close()

            fig, ax = plt.subplots()
            ax.imshow(plot_img, cmap='gray')
            ax.set_title("Left-click to add points, right-click to close ROI")
            selector = PolygonSelector(ax, onselect, useblit=True,
                                        lineprops=dict(color='r', linestyle='-', linewidth=2, alpha=0.5))
            plt.show()

        # Save ROI after selection
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
        if self.folder_path.joinpath("rois").exists():
            with os.scandir(self.folder_path.joinpath("rois")) as entries:
                for entry in entries:
                    self.roi_dict[Path(entry.path).stem] = entry.path

    def return_cells_by_saved_roi(self, roi_name, overwrite=False):
        try:
            self.load_saved_rois()
        except FileNotFoundError:
            pass
        
        if overwrite:
            if roi_name not in self.roi_dict:
                print("roi not found, please select")
                self.draw_roi2(title=roi_name)
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

    def return_x_midline(self):
        self.load_saved_rois()
        if 'midline' not in self.roi_dict.keys():
            self.draw_roi2(title="midline")
            self.load_saved_rois()
        x_midline_points = np.load(self.roi_dict["midline"])
        x_midline = int(np.nanmean([x for x, y in x_midline_points]))

        return x_midline

    def load_image(self):
        print('loading image')
        if "move_corrected_image" in self.data_paths.keys():
            image = imread(self.data_paths["move_corrected_image"])
        elif "rotated_image" in self.data_paths.keys():
            image = imread(self.data_paths["rotated_image"])
        elif "image" in self.data_paths.keys():
            image = imread(self.data_paths["image"])
        else:
            image = imread(self.data_paths["original_image"])

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
        if frametimes_df.time.values[0] > frametimes_df.time.values[-1]:  # overnight
            #pre-midnight frametimes_df and df
            frametimes_df_premidnight = frametimes_df[frametimes_df.time > frametimes_df.time.iloc[-1]]
            df_premidnight = df[df[datetime_col_name] > df[datetime_col_name].values[-1]]
            frame_matches_premidnight= [frametimes_df_premidnight[frametimes_df_premidnight.time < df_premidnight[datetime_col_name].values[-1]].index[i] for i in
                            range(len(df_premidnight))]
            #post-midnight frametimes_df and df
            frametimes_df_postmidnight = frametimes_df[frametimes_df.time < frametimes_df.time.values[0]]
            df_postmidnight = df[df[datetime_col_name] < df[datetime_col_name].values[0]]
            #use a for loop to deal with if the first frame of post-midnight df is actually earlier than first frame of
            #post-midnight frametimes_df, but still after midnight.
            frame_matches_postmidnight = []
            for i in range(len(df_postmidnight)):
                smaller_df = frametimes_df_postmidnight[frametimes_df_postmidnight.time < df_postmidnight[datetime_col_name].values[i]]
                if smaller_df.empty:
                    frame_matches_postmidnight = frame_matches_postmidnight + [frame_matches_premidnight[-1]]
                else:
                    frame_matches_postmidnight = frame_matches_postmidnight + [smaller_df.index[-1]]
            frame_matches = frame_matches_premidnight + frame_matches_postmidnight
        else:
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

class TailTrackedFish(BaseFish):
    def __init__(
        self,
        tail_key="tail",  # key to find tail data
        # peak_threshold = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.add_tail_paths(tail_key)
        self.tail_df = pd.read_hdf(self.data_paths["tail"])
        self.tail_hz = 1 / np.mean(np.diff((self.tail_df[:100].t.values)))
        self.add_bout_analysis()

        if 'frame' not in self.tail_df.columns:
            self.tail_df = self.tail_df[(self.tail_df.t_dt > self.frametimes_df.time.values[0]) &
                                                (self.tail_df.t_dt < self.frametimes_df.time.values[-1])]
            self.tail_df = self.tag_frames_to_df(self.frametimes_df, self.tail_df, 't_dt')
            self.tail_df.to_hdf(self.data_paths['tail'], key='tail')
        else:
            print('tail df already has frames')
        
        # self.tail_pearsonr_correlation(select_cells = None)

    def add_tail_paths(self, tail_key):
        try:
            with os.scandir(self.folder_path) as entries:
                for entry in entries:
                    if tail_key in entry.name:
                        self.data_paths["tail"] = Path(entry.path)
        except KeyError:
            print("failed to find tail data")
        
        return

    def add_bout_analysis(self):
        try:
            with os.scandir(self.folder_path) as entries:
                for entry in entries:
                    if ('tail_analysis' in entry.name) or ('tail_bout_df.h5' in entry.name):
                        self.bout_analysis_df = pd.read_hdf(Path(entry.path))
        except KeyError:
            print("failed to find bout analysis data")

        return

    def tail_pearsonr_correlation(self, select_cells = None):
        import scipy
        new_tail_df = tailtracking.find_tail_sum_std(self.tail_df)
        # using standard deviation of tail sum for correlation, fill nan with 0 to have this run
        tail_trace = new_tail_df.groupby(['frame']).mean()['std'].fillna(0)
        if select_cells is None:
            cell_arr = self.f_cells
        else:
            cell_arr = self.f_cells[select_cells]
        self.motor_pearson_corrs = [scipy.stats.pearsonr(trace[:len(tail_trace)], tail_trace)[0] for trace in cell_arr]
        self.motor_pearson_pvals = [scipy.stats.pearsonr(trace[:len(tail_trace)], tail_trace)[1] for trace in cell_arr]

class VizStimFish(TailTrackedFish):
    def __init__(
        self,
        stim_key="stims",
        stim_fxn=stimuli.pandastim_to_df,
        stim_fxn_args=None,
        legacy=False,
        stim_offset=5,
        seconds_motion_is_on = 5,
        used_offsets=(-10, 14),
        baseline_offset=-4, # adding a baseline number of frames 
        r_type="median",# response type - can be median, mean, peak of the stimulus response, default is median
        rep_mode = 'common',  # if you want all equal number of stim reps
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
        if 'rep' not in self.stimulus_df.columns:
            self.stimulus_df = stimuli.add_repetitions_to_stimulus_df(self.stimulus_df) # add unique rep numbers to the stimulus df
            print(self.stimulus_df.rep.unique())

        self.rep_mode = rep_mode
        if self.rep_mode == 'common':
            self.stimulus_df = self.stimulus_df[self.stimulus_df.rep <= (self.stimulus_df.groupby('stim_name').count().rep.min())].reset_index(drop=True)
            print(self.stimulus_df.rep.unique())
        else:
            print('keeping all stimulus reps')
            print(self.stimulus_df.rep.unique())

        self.r_type = r_type
        self.seconds_motion_is_on = seconds_motion_is_on

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
        if stim_offset == None:
            self.stim_offset = int(self.seconds_motion_is_on * self.img_hz)
        else:
            self.stim_offset = stim_offset
        self.offsets = used_offsets
        self.baseline_offset = baseline_offset
        # self.diff_image = self.make_difference_image()

    def add_stims(self, stim_key, stim_fxn, legacy):
        with os.scandir(self.folder_path) as entries:
            for entry in entries:
                if stim_key in entry.name:
                    self.data_paths["stimuli"] = Path(entry.path)
        if not legacy:
            try:
                _ = self.data_paths["stimuli"]
            except KeyError:
                print("failed to find stimuli")
                return

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
            
            # might not need this - depends on how the data was gathered! #
            self.unchop_stimulus_df  = self.stimulus_df # chop stimulus that are outside of frametime

            if 'frame' not in self.stimulus_df.columns:
                if self.frametimes_df.time.values[0] > self.frametimes_df.time.values[-1]:  # overnight
                    self.stimulus_df = pd.concat(
                                    [self.stimulus_df[(self.stimulus_df.time > self.frametimes_df.time.values[0])],
                                    self.stimulus_df[(self.stimulus_df.time < self.frametimes_df.time.values[-1])]])
                else:
                    self.stimulus_df = self.stimulus_df[
                                    (self.stimulus_df.time > self.frametimes_df.time.values[0]) &
                                    (self.stimulus_df.time < self.frametimes_df.time.values[-1])]
                self.stimulus_df = self.tag_frames_to_df(self.frametimes_df, self.stimulus_df, 'time')
            else:
                print('stimulus df already has frames')

    def make_difference_image(self, selectivityFactor=1.5, brightnessFactor=10):
        image = self.load_image()

        diff_imgs = {}
        for stimulus_name in [
            i
            for i in self.stimulus_df.stim_name.values.unique()
            if i in constants.monocular_dict.keys()
        ]: 
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
    
    def make_specific_difference_image(self, stim_list, color_dict, brightnessFactor = 7.5, selectivityFactor = 2):
        '''
        Make a specific pixel brightness difference image for a list of stimuli
        stim_list: list of stimuli to color in the image
        color_dict: dictionary of color values for each stimulus
        '''
        image = self.load_image()
        diff_imgs = {}

        for stimulus_name in stim_list: 
            stim_occurences = self.stimulus_df[self.stimulus_df.stim_name == stimulus_name].frame.values

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

            red_val = diff_image * color_dict[stimulus_name][0]
            green_val = diff_image * color_dict[stimulus_name][1]
            blue_val = diff_image * color_dict[stimulus_name][2]

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

class PhotostimFish(TailTrackedFish):
    def __init__(
        self,
        rotate = True, 
        stim_type_keyword = 'single_cell',
        stimmed_plane = True,
        photostim_frame_window = [-8, 12],
        match_with_overlap = False,
        # evoked_num_frames = 6,
        # identify_stim_cell_within_radius_um = 7,
        *args,
        **kwargs,
    ):
        '''
        :param stim_type_keyword: what kind of stimulation type for this dataset, 'single_cell' or 'ensemble'
        :param rotate: is the image rotated 90 degrees
        :param stimmed_plane: is this plane stimulated (true or false), helpful when running through multiple planes in a volume
        :param photostim_frame_window: how many frames to use for photostimulation period (before and after the event), default: 8 frames pre, 12 frames post
        :param evoked_num_frames: how many frames to use for measuring the 'evoked' period, default: 6
        '''
        super().__init__(*args, **kwargs)
        self.add_parameter_df() # adding the stim parameters df if it exists
        self.photostim_frame_window = photostim_frame_window
        # self.evoked_num_frames = evoked_num_frames

        # 0 - prep the cell traces, image
        self.normcells = arrutils.norm_fdff(self.f_cells)
        self.zdiff_cells = [arrutils.zdiffcell(z) for z in self.f_cells]
        self.rescaled_img()

        # 1 - find bad frames and baseline frames, make sure this exists first #
        try:
            self.badframes_arr = np.array(np.load(Path(self.folder_path).joinpath('bad_frames.npy')))
        except: 
            print('find bad frames and re run suite2p')
            photostim_utils.save_badframes_arr(self)

        if 'ps_xml' in self.data_paths.keys():
            photostim_utils.find_no_baseline_frames(self)
        elif 'ps_log' in self.data_paths.keys():
            photostim_utils.find_no_baseline_frames(self)
        else: # protecting the automated gui experiments to keep running with photostim fish
            self.baseline_frames = 0

        # 2 - get basic photostim experiment info for any plane
        self.ps_event_duration, self.ps_event_ms_from_start = photostim_utils.collect_stimulation_times(self)
        self.ps_event_duration_frames = int(np.ceil(self.ps_event_duration / 1000 * self.img_hz))
        self.ps_event_start = arrutils.filter_list(lst=np.unique(self.badframes_arr),
                                                       interval=self.ps_event_duration_frames)

        ## IF WE HAVE A STIMMED PLANE - get matching cell ids, stim sites df ##
        if stimmed_plane:
            # 3 - gather and make the stim sites dataframe for different outputs #
            if ('stim_sites' not in self.data_paths.keys()): # need to create a stim sites df from scratch (works with MP files, voltage recording)
                self.ps_event_duration, _ = photostim_utils.collect_stimulation_times(self)
                self.ps_event_duration_frames = int(np.ceil(self.ps_event_duration/1000 * self.img_hz))
                self.ps_event_start = arrutils.filter_list(lst = np.unique(self.badframes_arr), interval = self.ps_event_duration_frames)
                self.stim_sites_df = photostim_utils.identify_stim_sites(self, rotate, stimulation_type = stim_type_keyword)
            if ('stim_sites' in self.data_paths.keys()): # need to upload current stim sites df (works with new automated output, normal outputs)
                self.stim_sites_df = pd.read_hdf(self.data_paths['stim_sites'], key="stim")
                if 'stim_duration_ms' in self.stim_sites_df.columns: # if special output type
                    self.ps_event_duration = self.stim_sites_df.stim_duration_ms.iloc[0] # assuming all the same
                    self.ps_event_start = self.stim_sites_df.stim_frames.values # already put these into the dataframe
                    if 'x_stim' not in self.stim_sites_df.columns:
                        self.stim_sites_df = self.stim_sites_df.rename(columns={'x': 'x_stim', 'y': 'y_stim'})
                # only single cell photostimulation, add in the stim frames and stim events for later analysis, with a Markpoints file
                if (len(self.stim_sites_df) == 1) & ('ps_xml' in self.data_paths.keys()):
                    self.stim_sites_df['stim_frames'] = [self.ps_event_start]
                    self.stim_sites_df['stim_events'] = [np.arange(len(self.ps_event_start))]
                if ('ps_log' in self.data_paths.keys()):
                    self.ps_event_start = arrutils.filter_list(lst=np.unique(self.badframes_arr),
                                                               interval=self.ps_event_duration_frames)

            # 4 - id the stim sites and save the raw traces #
            self.raw_traces, self.points = photostim_utils.collect_raw_traces(self)
            self.check_if_stim_site_within_brain() # make sure the stim site is within the brain
            # OLD - id the stimulated cells based on distance & max response #
            # self.stimmed_cell_coords, self.stimmed_cell_id_array, self.stimmed_cells_matched_stim_ids_dict = self.identify_stim_cells(within_radius_um = identify_stim_cell_within_radius_um,
            #                                                                                                                           frame_window = [self.photostim_frame_window[0],
            #                                                                                                                                           self.evoked_num_frames])
            # id the stimulated cells based on only location  #
            self.stimmed_cell_coords, self.stimmed_cell_id_array, self.stimmed_cells_matched_stim_ids_dict = self.identify_stim_cells_purely_location(overlap=match_with_overlap)

        # # 5 - match photostim times with tail data
        # if 'tail' in self.data_paths.keys():
        #     self.add_stim_events_to_tail_df()

    # if 'caiman' in self.data_paths.keys(): # not working for the automated gui yet
    #     photostim_utils.create_new_ps_events_array(self) # making a new ps_events start array since trimmed frames from caiman processing
    #     self.ps_event_start = np.load(Path(self.folder_path).joinpath('ps_frames.npy'))
    #     self.ps_event_duration = 100 # arbitrary setting duration to 100 ms
    #     self.ps_event_duration_frames = 1 # thus frames is 1

    def add_parameter_df(self):
        stim_parameters_csv_path = self.folder_path.parents[1].joinpath('stim_parameters.csv')
        if stim_parameters_csv_path.exists():
            self.stim_params_df = pd.read_csv(stim_parameters_csv_path)
        else:
            pass

    def check_if_stim_site_within_brain(self):
        '''
        identifies if stim sites are within the brain region
        important for control stim sites outside of the brain
        :return: self.stim_inds_within_brain - the good inds of the stim sites df that are within the brain
        '''
        import matplotlib.path as mpltPath

        self.load_saved_rois()
        if 'brain' in self.roi_dict.keys():
            brain_roi = np.load(self.roi_dict['brain'])
            brain_path = mpltPath.Path(brain_roi)
            all_inds = np.arange(len(self.stim_sites_df))
            all_rois = [(x, y) for x, y in zip(self.stim_sites_df.x_stim.values, self.stim_sites_df.y_stim.values)]
            pts_in_brain = brain_path.contains_points(all_rois)
            inds_in_brain = all_inds[pts_in_brain]
            self.stim_inds_within_brain = inds_in_brain
        else:
            print('no brain roi to check stim sites')

    def identify_stim_cells(self, within_radius_um = 10, frame_window = [-8, 6]):
        '''
        Identify the suite2p cells that are stimulated in the dataset from the overlap of stim site locations & max evoked responses

        Returns:
        closest_coord_list = coordinates of the closest cell to the stim site
        closest_cell_id_array = the cell id of the closest cell to the stim site
        closest_cell_id_dict = dict, keys are cell ids in the stim sites df, values are the cell ids from the fishy
        '''
        closest_coord_list = []
        closest_cell_id_array = np.zeros(shape = (len(self.stim_sites_df)))
        try:
            cell_ids = self.stim_sites_df.cell_ids.values
        except:
            cell_ids = range(len(self.stim_sites_df))
        print(cell_ids)
        closest_cell_id_dict = {i: [] for i in cell_ids}
        all_rois = self.return_cell_rois(range(len(self.f_cells)))
        for i in range(len(self.stim_sites_df)):
            stim_cell_id = cell_ids[i]
            print(f'finding match with stim site {stim_cell_id}')
            # 1 - look at cells within very close radius from the stimulation cross
            bruker_coordinate = [int(self.stim_sites_df.x_stim.iloc[i]), int(self.stim_sites_df.y_stim.iloc[i])]
            if 'stim_frames' in self.stim_sites_df.columns: # if select stim frames for this stimulation
                stimmed_frames = self.stim_sites_df.iloc[i].stim_frames
                print(stimmed_frames)
            else: # default it's just all the ps events on this plane
                stimmed_frames = self.ps_event_start
                print(stimmed_frames)
            nearby_coords_cell_ids, nearby_coords_list = coordutils.determine_nearby_cells_xy(target_coord = bruker_coordinate,
                                                                                           coordinates = all_rois,
                                                                                           ums_per_px = self.um_per_px,
                                                                                           radius_um= within_radius_um)
            print(f'number of nearby cells {len(nearby_coords_cell_ids)}')
            if len(nearby_coords_cell_ids) > 0: # if there are nearby vals
                # 2 - get the max response in the evoked window range (df/f)
                frame_subset = arrutils.subsection_arrays(stimmed_frames, frame_window) # get the frame subset to find df/f
                if frame_subset[-1][-1] > len(self.f_cells[0]): # adjust frames in case this is out of range
                    new_frame_subset = []
                    for s in frame_subset:
                        new_frame_subset.append([q for q in s if q < len(self.f_cells[0])])
                    frame_subset = np.array(new_frame_subset)
                resp_metric_arr = np.zeros(shape = (len(nearby_coords_cell_ids)))
                for n, responding_trace in enumerate(self.f_cells[nearby_coords_cell_ids]): # has to be raw fluorescence to find df/f
                    resp_raw_trial = np.array([responding_trace[g] for g in frame_subset if len(responding_trace[g] ) > 0])
                    resp_df_f_trial = np.zeros(shape = (len(resp_raw_trial), len(resp_raw_trial[0])))
                    for d, f in enumerate(resp_raw_trial):
                        base_e = f[:-frame_window[0]] # i.e. frames 0:8
                        plot_e = (f - np.nanmean(base_e)) / np.nanmean(base_e)
                        resp_df_f_trial[d] = plot_e # full trace for each trial
                    avg_resp_df_f = arrutils.pretty(np.nanmean(resp_df_f_trial, axis = 0), 2) # this looks much better on a smoothed trace (otherwise so noisy), only smoothing over 2 frames (about .5 sec)
                    max_resp_df_f = np.nanmax(avg_resp_df_f[-frame_window[0]:-frame_window[0] + frame_window[1]])
                    resp_metric = max_resp_df_f - np.nanmean(avg_resp_df_f[int(-frame_window[0] - self.img_hz):-frame_window[0]]) # max response - avg baseline of just 1 sec before stim
                    resp_metric_arr[n] = resp_metric

                # 3 - choose the neuron that has the largest response from baseline to be the stimmed cell (but don't want to keep double in the list)
                sorted_idx = np.argsort(resp_metric_arr)[::-1]
                # Loop until we find the first one that's not already taken
                for idx in sorted_idx:
                    candidate_id = nearby_coords_cell_ids[idx]
                    if candidate_id not in closest_cell_id_array:
                        stimmed_cell_id = candidate_id
                        stimmed_coord = nearby_coords_list[idx]
                        break
                closest_coord_list.append(stimmed_coord)
                closest_cell_id_array[i] = stimmed_cell_id
                closest_cell_id_dict[stim_cell_id] = [stimmed_cell_id]
            else:
                closest_coord_list.append(np.nan)
                closest_cell_id_array[i] = np.nan
                closest_cell_id_dict[stim_cell_id] = [np.nan]

        return closest_coord_list, closest_cell_id_array, closest_cell_id_dict

    def identify_stim_cells_purely_location(self, overlap = False):
        '''
        Identify the suite2p cells that are stimulated in the dataset from the overlap of stim site locations
        overlap: keyword to indicate if you want to use the centers of stim sites or the whole spiral size to overlap with cells

        Returns:
        closest_coord_list = coordinates of the closest cell to the stim site
        closest_cell_id_array = the cell id of the closest cell to the stim site
        '''

        if overlap == False:
            stim_points = np.column_stack([self.stim_sites_df.x_stim.values.astype(float),
                                            self.stim_sites_df.y_stim.values.astype(float)])
            all_points = np.asarray(self.return_cell_rois(range(len(self.f_cells))), float)

            matches, distances, aligned_suite2p_coords = coordutils.closest_coordinates_1to1(
                source_coords=stim_points,  # stim sites
                target_coords=all_points,  # suite2p cell locations
                xy_offset=(0,0), # no offset between bruker stim sites and suite2p coords (can adjust if needed)
                max_distance=10) # maximum of 10 micron distance away from matches

            # --- Allocate outputs in stim row-order ---
            closest_cell_id_array = np.full(len(stim_points), fill_value=-1, dtype=int)
            closest_coord_list = [None] * len(stim_points)
            closest_cell_id_dict = {}

            # Loop through matched stim site indices
            for stim_row_idx, suite2p_cell_idx in matches.items():
                stim_cell_id = self.stim_sites_df.cell_ids.iloc[stim_row_idx] # Get the stim site cell ID from dataframe
                closest_cell_id_array[stim_row_idx] = suite2p_cell_idx
                closest_coord_list[stim_row_idx] = self.return_singlecell_rois(suite2p_cell_idx)
                closest_cell_id_dict[stim_cell_id] = suite2p_cell_idx # **Use stim_cell_id as key** — preserving your custom IDs

        elif overlap == True:
            # first collect the xpix and ypix of the stimulation site based on the spiral size
            stim_sites_stat_dict = {}
            stim_sites_ind_list = []
            for p in range(len(self.stim_sites_df)):
                if p in self.stim_inds_within_brain:
                    stim_sites_ind_list.append(self.stim_sites_df.cell_ids.iloc[p])
                    radius_um = self.stim_sites_df.iloc[p].sp_size/2
                    if radius_um < 1:
                        radius_um = 2.5 # default size of 5 umn diameter
                    radius_px = radius_um/self.um_per_px
                    x_pix, y_pix = roiutils.points_within_circle(self.stim_sites_df.iloc[p].x_stim,
                                                                self.stim_sites_df.iloc[p].y_stim, radius_px)
                    stim_cell_id = int(self.stim_sites_df.iloc[p].cell_ids)
                    stim_sites_stat_dict[stim_cell_id] = {'xpix': x_pix, 'ypix': y_pix}

            # matching cell ids based on spiral size overlap and doing 1 to 1 matching
            closest_cell_id_dict = coordutils.match_cell_ids(cell_arr1 = np.array(stim_sites_ind_list),
                                                                stats_dict1 = stim_sites_stat_dict,
                                                                cell_arr2 = np.arange(len(self.f_cells)),
                                                                stats_dict2 = self.stats,
                                                                um_to_px = self.um_per_px)
            # identifying what those coordinates are, removing nan's and converting all to integers
            closest_cell_id_array = np.array([v for k, v in closest_cell_id_dict.items()])
            closest_cell_id_array = closest_cell_id_array[~np.isnan(closest_cell_id_array)]
            closest_cell_id_array = np.array([int(i) for i in closest_cell_id_array])
            closest_coord_list = [self.return_singlecell_rois(m) for m in closest_cell_id_array]

        return closest_coord_list, closest_cell_id_array, closest_cell_id_dict

    def find_precise_photostim_times(self):
        '''
        getting precise photostimulation datetimes from the voltage output
        the 'else' part will not work for the automated gui, i need to make that times txt file separately for that output
        this will break if that is the case
        :return:datetime array of the precise photostim times
        '''
        from datetime import timedelta, date
        from datetime import datetime as dt

        txt_file_path = Path(self.folder_path.parents[1]).joinpath("precise_photostim_times.txt")
        if txt_file_path.exists():
            photostim_dt_array = pd.to_datetime(pd.read_csv(txt_file_path, header=None)[0]).values
        elif (not txt_file_path.exists()) & ('ps_log' in self.data_paths.keys()):
            return print('get precise photostim times for automated gui output')
        else:
            root = bruker_images.read_xml_to_root(self.data_paths['voltage_xml'])
            start_time = [start.text for start in root.iter('DateTime')][0]
            start_dt = pd.to_datetime(start_time)

            info_data = bruker_images.read_xml_to_str(self.data_paths["info_xml"])
            for i in info_data.split("\n"):
                if ("relativeTime" in i) and ('VoltageRecording' not in i):
                    relative_time = float([i.split("relativeTime=")[1].split('"')[1]][0])
                    if relative_time == 0:
                        absolute_time_from_start = float([i.split("absoluteTime=")[1].split('"')[1]][0])

            real_start_dt = (dt.combine(date.today(), start_dt.tz_localize(None).time())
                             + timedelta(seconds=absolute_time_from_start))

            event_dts = real_start_dt + pd.to_timedelta(self.ps_event_ms_from_start, unit="ms")
            event_dts_notimezone = event_dts.tz_localize(None)
            pd.Series(event_dts_notimezone).to_csv(txt_file_path, index=False, header=False)
            photostim_dt_array = event_dts_notimezone.values

        return photostim_dt_array

    def add_stim_events_to_tail_df(self, force = True):
        '''
        Add in precise photostimulation events to the tail df
        :return: prints when done
        '''

        if force:
            tail_df_ts = self.tail_df.t_dt.values
            arr_sec = np.array([t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6 for t in tail_df_ts])

            photostim_dts_arr = self.find_precise_photostim_times()
            photostim_dts = [pd.to_datetime(a).time() for a in photostim_dts_arr]
            photostim_sec = [t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6 for t in photostim_dts]

            matching_tail_df_inds = np.array([np.argmin(np.abs(arr_sec - t)) for t in photostim_sec])

            # add a new column to the tail_df & save it
            new_tail_df = self.tail_df.copy()
            new_tail_df['stim_events'] = 'None'
            for i, idx in enumerate(matching_tail_df_inds):
                new_tail_df['stim_events'].iloc[idx] = i
            new_tail_df.to_hdf(self.data_paths['tail'], key='tail')
            self.tail_df = new_tail_df
            return print('added stim events to tail data')
        elif ('stim_events' in self.tail_df.columns) & (force == False):
            return print('already added stim events to tail data')

    ### OLD FUNCTIONS ###
    # def build_ps_corrdf(self, photostimulated_cell_arr = None, len_decay_frames = 10, ps_offset = 0, select_cells = None, frames_pre_post = [-3, 8],
    #                     trace_type = 'raw', evoked_response_type = 'mean'):
    #     '''
    #     building a photostim correlation dataframe with perfect photostim responders and evoked response from stimulation event
    #     len_decay_frames:
    #     select_cells:
    #     ps_offset: the offset after the photostim event (aka bad frame) that the response should start
    #     frames_pre_post: the frames before and after each photostim event to grab for calculations
    #     '''
    #     if select_cells is None:
    #         cell_traces = self.f_cells
    #         normcell_traces = self.normcells
    #     else:
    #         cell_traces = self.f_cells[select_cells]
    #         normcell_traces = self.normcells[select_cells]
    #
    #     if trace_type == 'raw':
    #         traces_array = cell_traces
    #     elif trace_type == 'norm':
    #         traces_array = normcell_traces
    #     elif trace_type == 'df/f' :
    #         traces_array = self.dff_cells
    #     else:
    #         traces_array = cell_traces
    #
    #     perfect_photostim_response = np.zeros(traces_array[0].shape)
    #     decay_lst = np.linspace(1, 0, len_decay_frames)
    #
    #     for i in self.ps_event_start:
    #         i = i + ps_offset
    #         perfect_photostim_response[i:(i + len_decay_frames)] = decay_lst
    #
    #     if photostimulated_cell_arr is None:
    #         photostimulated_cell_arr = self.raw_traces[0] # hardcoded for the first cell
    #     # photostimulated_cell_arr = self.f_cells[self.stimmed_cell_ids[0]] # hardcoded for the first cell, source extraction
    #
    #     self.ps_corrdf = pd.DataFrame(columns = ['traces', 'correlation', 'z_corr'])
    #     # correlation with the stimulated cell raw traces
    #     for b, c in enumerate(traces_array):
    #         cell_arr = c[self.baseline_frames:]
    #         z_cell_arr = arrutils.zscoring(cell_arr)
    #         stim_arr = arrutils.pretty(photostimulated_cell_arr)[self.baseline_frames:]
    #         z_stim_arr = arrutils.zscoring(stim_arr)
    #         corr = np.corrcoef(cell_arr, stim_arr)[0, 1]
    #         z_corr = np.corrcoef(z_cell_arr, z_stim_arr)[0, 1]
    #         self.ps_corrdf.loc[b, 'correlation'] = corr
    #         self.ps_corrdf.loc[b, 'z_corr'] = z_corr
    #         self.ps_corrdf.loc[b, 'traces'] = c
    #
    #     ps_trial_subset = arrutils.subsection_arrays(self.ps_event_start, frames_pre_post)
    #     self.ps_corrdf['evoked_response'] = photostimulation.calculate_evoked_response(arr_cell_traces = traces_array, arr_subset = ps_trial_subset,
    #                                                                                    ps_offset = ps_offset,frame_window = frames_pre_post,
    #                                                                                    r_type = evoked_response_type)
    #
    #     return self.ps_corrdf
    #
    # def make_connectivity_map_evoked_activity(self, stimulation_site_coord_lst, responding_cell_lst, arrows = True, annotate_txt = True, cbar_limit = None,
    #                           frames_pre_post = [-3, 8], stimulation_site_clr = 'green', dot_size = 80, savepath = None):
    #     '''
    #     Make a connectivity map based on average evoked response with arrows from one stimulation site to list of cells
    #     '''
    #     import matplotlib.pyplot as plt
    #     import matplotlib
    #
    #     if not hasattr(self, 'ps_corrdf'):
    #         print('building ps_corrdf')
    #         self.build_ps_corrdf(frames_pre_post = frames_pre_post)
    #
    #     _rois = self.return_cell_rois(responding_cell_lst)
    #     _clr_list = [self.ps_corrdf.evoked_response[b] for b in responding_cell_lst]
    #     if cbar_limit == None:
    #         set_max = max(_clr_list)
    #     else:
    #         set_max = cbar_limit
    #     set_min = -set_max
    #
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(self.rescaled_ref*1.3, cmap="gray",alpha=1, vmax=np.percentile(self.rescaled_ref, 99.9),)
    #     xvals = [r[0] for r in _rois]
    #     yvals = [r[1] for r in _rois]
    #     clrs = plotutils.clip_and_map_colors(_clr_list, vmin = set_min, vmax = set_max, cmap_name='coolwarm')
    #     plt.scatter(xvals, yvals, s = dot_size, color = clrs, cmap = 'coolwarm', edgecolor = 'white', vmin = set_min, vmax = set_max, zorder = 2)
    #
    #     # annotate text
    #     if annotate_txt:
    #         for d in range(len(responding_cell_lst)):
    #             plt.annotate(str(d+1), (xvals[d], yvals[d]), textcoords="offset points", xytext=(10, -10), color = 'white', fontsize=18)
    #
    #     for s in stimulation_site_coord_lst:
    #         plt.scatter(s[0], s[1], s = dot_size, color = stimulation_site_clr, edgecolor = 'white', marker = 'o', zorder=3)
    #     # draw arrows
    #     if arrows:
    #         for k in range(len(responding_cell_lst)):
    #             end = (xvals[k], yvals[k])
    #             for d in stimulation_site_coord_lst:
    #                 plt.annotate('', xy=(end[0]-2, end[1] - 1 ), xytext=(d[0], d[1]),
    #                             arrowprops=dict(arrowstyle = '-|>', linewidth=2, facecolor=clrs[k], edgecolor = clrs[k]), zorder=1)
    #
    #     cbar = plt.colorbar(matplotlib.cm.ScalarMappable(cmap='coolwarm'))
    #     cbar.set_label(label= 'Average Evoked Response', rotation = 270, labelpad=15)
    #     cbar.mappable.set_clim(vmin = set_min, vmax = set_max)
    #     plt.axis('off')
    #
    #     if savepath != None:
    #         plt.savefig(savepath, dpi = 300)
    #
    #     # plt.show()
    #
    # def make_connectivity_map_z_corr(self, stimulation_site_coord_lst, responding_cell_lst, arrows = True, annotate_txt = True, cbar_limit = None,
    #                            stimulation_site_clr = 'green', dot_size = 80, savepath = None):
    #     '''
    #     Make a connectivity map based on average evoked response with arrows from one stimulation site to list of cells
    #     '''
    #     import matplotlib.pyplot as plt
    #     import matplotlib
    #
    #     if not hasattr(self, 'ps_corrdf'):
    #         print('building ps_corrdf')
    #         self.build_ps_corrdf()
    #
    #     _rois = self.return_cell_rois(responding_cell_lst)
    #     _clr_list = [self.ps_corrdf.z_corr[b] for b in responding_cell_lst]
    #     if cbar_limit == None:
    #         set_max = max(_clr_list)
    #     else:
    #         set_max = cbar_limit
    #     set_min = -set_max
    #
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(self.rescaled_ref, cmap="gray",alpha=1, vmax=np.percentile(self.rescaled_ref, 99.9),)
    #     xvals = [r[0] for r in _rois]
    #     yvals = [r[1] for r in _rois]
    #     clrs = plotutils.clip_and_map_colors(_clr_list, vmin = set_min, vmax = set_max, cmap_name='coolwarm')
    #     plt.scatter(xvals, yvals, s = dot_size, color = clrs, cmap = 'coolwarm', edgecolor = 'white', vmin = set_min, vmax = set_max, zorder = 2)
    #
    #     # annotate text
    #     if annotate_txt:
    #         for d in range(len(responding_cell_lst)):
    #             plt.annotate(str(d+1), (xvals[d], yvals[d]), textcoords="offset points", xytext=(10, -10), color = 'white', fontsize=18)
    #
    #     for s in stimulation_site_coord_lst:
    #         plt.scatter(s[0], s[1], s = dot_size, color = stimulation_site_clr, edgecolor = 'white', marker = 'o', zorder=3)
    #     # draw arrows
    #     if arrows:
    #         for k in range(len(responding_cell_lst)):
    #             end = (xvals[k], yvals[k])
    #             for d in stimulation_site_coord_lst:
    #                 plt.annotate('', xy=(end[0]-2, end[1] - 1 ), xytext=(d[0], d[1]),
    #                             arrowprops=dict(arrowstyle = '-|>', linewidth=2, facecolor=clrs[k], edgecolor = clrs[k]), zorder=1)
    #
    #     cbar = plt.colorbar(matplotlib.cm.ScalarMappable(cmap='coolwarm'))
    #     cbar.set_label(label= 'Z-scored Correlation', rotation = 270, labelpad=15)
    #     cbar.mappable.set_clim(vmin = set_min, vmax = set_max)
    #     plt.axis('off')
    #
    #     if savepath != None:
    #         plt.savefig(savepath, dpi = 300)
    #
    #     plt.show()
    #
    #

class WorkingFish(VizStimFish):
    def __init__(self, corr_threshold=0.65, 
                 bool_data_type = 'normf', 
                 stim_order = None, 
                 ref_image=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if "move_corrected_image" not in self.data_paths:
            print('no movement corrected image')
        self.corr_threshold = corr_threshold
        if not hasattr(self, "f_cells"):
            if 'suite_2p' in self.data_paths.keys():
                self.load_suite2p()
            if 'caiman' in self.data_paths.keys():
                self.load_caiman()
                self.is_cell()

        # create different fluorescence arrays
        self.zdiff_cells = [arrutils.zdiffcell(i) for i in self.f_cells]
        self.normcells = arrutils.norm_0to1(self.f_cells)

        if ref_image is not None:
            self.reference_image = ref_image
        
        self.stim_order = stim_order # order to stimuli for average trace plots
        if self.stim_order is None:
            self.stim_order = self.stimulus_df.stim_name.unique()

        # self.neuron_each_stim_rep_arrays(stim_order)
        self.stim_start_frames = stimuli.stimulus_start_frames_for_plots(baseline_offset = -self.offsets[0],
                                                                         length_of_total_frame_arr = np.diff(self.offsets)[0], 
                                                                         number_of_stims_in_set = len(self.stim_order))

        self.bool_data_type = bool_data_type # type of data to compute boolean df and correlation values on

        self.zdiff_stim_dict, self.zdiff_err_dict, self.zdiff_neuron_dict = self.build_stimdicts(self.zdiff_cells)
        self.normf_stim_dict, self.normf_err_dict, self.normf_neuron_dict = self.build_stimdicts(self.normcells)
        self.f_stim_dict, self.f_err_dict, self.f_neuron_dict = self.build_stimdicts(self.f_cells)
        # if hasattr(self, 'dff_cells'):
        #     self.dff_stim_dict, self.dff_err_dict, self.dff_neuron_dict = self.build_stimdicts(self.dff_cells)

        if self.bool_data_type == 'zdiff':
            self.analysis_stim_dict = self.zdiff_stim_dict
            self.analysis_neuron_dict = self.zdiff_neuron_dict
        elif self.bool_data_type == 'normf':
            self.analysis_stim_dict = self.normf_stim_dict
            self.analysis_neuron_dict = self.normf_neuron_dict 
        elif self.bool_data_type == 'f':
            self.analysis_stim_dict = self.f_stim_dict
            self.analysis_neuron_dict = self.f_neuron_dict
        elif self.bool_data_type == 'df/f':
            self.analysis_stim_dict = self.dff_stim_dict
            self.analysis_neuron_dict = self.dff_neuron_dict

        self.build_stimdicts_extended_zdiff()
        self.build_stimdicts_extended_normf()
        self.build_stimdicts_extended_raw()

        self.build_booldf_corr()
        self.build_booldf_baseline()
        # self.build_booldf_cluster()

    def neuron_each_stim_rep_arrays(self, stim_order, trace_type = None):
        '''
        output -- array of shape: # of neurons, each repetition, and each stim (in the order of the stim_order) 
                array of activity (length of offsets * num of stims) 
        '''
        if trace_type is None:
            trace_type = self.bool_data_type
        if trace_type == 'normf':
            traces = self.normcells
        elif trace_type == 'zdiff':
            traces = self.zdiff_cells
        elif trace_type == 'raw':
            traces = self.f_cells
        elif trace_type == 'df/f':
            traces = self.dff_cells

        if 'rep' not in self.stimulus_df.columns:
            self.stimulus_df = stimuli.add_repetitions_to_stimulus_df(self.stimulus_df)
            self.stimulus_df = self.stimulus_df[self.stimulus_df.rep != -1]

        # set up the array
        n_neurons = len(traces)
        n_reps = self.stimulus_df.rep.max() + 1  # keeps full rep index space
        n_frames = np.diff(self.offsets)[0] * len(stim_order)

        self.neur_resps_each_stim_rep = np.full((n_neurons, n_reps, n_frames), np.nan) # make a full Nan array

        for r in sorted(self.stimulus_df.rep.unique()):

            one_rep = self.stimulus_df[self.stimulus_df.rep == r]

            all_arrs = np.full((len(stim_order), np.diff(self.offsets)[0]), np.nan)

            for st, stim in enumerate(stim_order):
                df = one_rep[one_rep.stim_name == stim]

                if df.empty:
                    continue  # leave this stim as NaNs

                arrs = arrutils.subsection_arrays(df.frame.values, self.offsets)
                if len(arrs) > 0:
                    all_arrs[st] = arrs[0]

            # flatten
            _all_arrs = np.array(all_arrs).ravel()

            # if everything is nan, skip
            if np.all(np.isnan(_all_arrs)):
                continue

            # only keep valid indices
            valid = ~np.isnan(_all_arrs)
            idx = _all_arrs[valid].astype(int)

            for n, nrn in enumerate(traces):
                try:
                    resp_arr = np.full_like(_all_arrs, np.nan, dtype=float)
                    resp_arr[valid] = nrn[idx]
                except IndexError:
                    continue
                self.neur_resps_each_stim_rep[n, r] = resp_arr
        
        return self.neur_resps_each_stim_rep

    def single_neuron_df_f_all_stim_array(self, cell_id, stim_order, num_nan_spaces = 3):
        '''
        Make the df/f array for a single neuron, nan's between each stim responses (at the end of the trace)
        :param cell_id: single neuron that you want to calculate this for
        :param stim_order: stimuli order
        :param num_nan_spaces: number of nan spaces to put between responses
        :return: traces array, shape = n trials x frames + extra nan spaces (using offset values)
        '''
        extended_resp = pd.DataFrame(self.extended_responses_normf).iloc[cell_id][stim_order]

        if 'rep' not in self.stimulus_df.columns:
            self.stimulus_df['rep'] = 0
            # get the number of reps for each stim, choose number of reps based on the minimum value
            all_reps = []
            for each_stim in self.stimulus_df.stim_name.unique():
                all_reps.append(len(self.stimulus_df[self.stimulus_df.stim_name == each_stim]))
            no_repetitions = min(all_reps)

            # set the rep value into a new column in the stimulus df
            n_stims = self.stimulus_df.stim_name.nunique()
            for i in range(no_repetitions):
                self.stimulus_df.iloc[(n_stims * i):(n_stims * i + n_stims)]['rep'] = i
        num_trials = max(self.stimulus_df.rep) + 1

        concatenated_traces_list = []
        for n, each_stim_resp in enumerate(extended_resp.values):
            each_stim_resp = np.array(each_stim_resp)
            baseline_arr = each_stim_resp[:, :-self.offsets[0] - 1]
            baseline_mean = np.nanmean(baseline_arr, axis=1)
            df_f_arr = np.array([(arr - baseline_mean[i]) / baseline_mean[i] for i, arr in enumerate(each_stim_resp)])
            if num_nan_spaces > 0:
                df_f_arr_with_nan = np.hstack([df_f_arr, np.full((df_f_arr.shape[0], num_nan_spaces), np.nan)])
                if df_f_arr_with_nan.shape[0] < num_trials:
                    pad_rows = num_trials - df_f_arr_with_nan.shape[0]
                    df_f_arr_with_nan = np.vstack(
                        [df_f_arr_with_nan, np.full((pad_rows, df_f_arr_with_nan.shape[1]), np.nan)])
            else:
                df_f_arr_with_nan = df_f_arr

            concatenated_traces_list.append(df_f_arr_with_nan)
        print(len(concatenated_traces_list))
        df_f_cell_resp = np.concatenate(concatenated_traces_list, axis=1)

        return df_f_cell_resp

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
                    if arr[-1] < len(nrn): # making sure the arr is not longer than the cell trace (frames)
                        resp_arrs.append(nrn[arr])
                
                self.extended_responses_zdiff[stim][n] = resp_arrs
    
    def build_stimdicts_extended_raw(self):
        # makes an array of normalized calcium responses for each stim (not median)
        self.extended_responses_raw = {i: {} for i in self.stimulus_df.stim_name.unique()}
        for stim in self.stimulus_df.stim_name.unique():
            arrs = arrutils.subsection_arrays(
                self.stimulus_df[self.stimulus_df.stim_name == stim].frame.values,
                self.offsets,
            )
            for n, nrn in enumerate(self.f_cells):
                resp_arrs = []
                for arr in arrs:
                    if arr[-1] < len(nrn): # making sure the arr is not longer than the cell trace (frames)
                        resp_arrs.append(nrn[arr])
                self.extended_responses_raw[stim][n] = resp_arrs

    def build_stimdicts_extended_normf(self):
        # makes an array of normalized calcium responses for each stim (not median)
        self.extended_responses_normf = {i: {} for i in self.stimulus_df.stim_name.unique()}
        for stim in self.stimulus_df.stim_name.unique():
            arrs = arrutils.subsection_arrays(
                self.stimulus_df[self.stimulus_df.stim_name == stim].frame.values,
                self.offsets,
            )
            for n, nrn in enumerate(self.normcells):
                resp_arrs = []
                for arr in arrs:
                    if arr[-1] < len(nrn): # making sure the arr is not longer than the cell trace (frames)
                        resp_arrs.append(nrn[arr])
                self.extended_responses_normf[stim][n] = resp_arrs

    def build_stimdicts(self, traces):
        # makes an median value (can change what response type) of your choice of calcium response for each neuron for each stim
        self.stimulus_df = stimuli.validate_stims(self.stimulus_df, self.f_cells)
        stim_dict = {i: {} for i in self.stimulus_df.stim_name.unique()}
        err_dict = {i: {} for i in self.stimulus_df.stim_name.unique()}

        for stim in self.stimulus_df.stim_name.unique():
            arrs = arrutils.subsection_arrays(
                self.stimulus_df[(self.stimulus_df.stim_name == stim) & (self.stimulus_df.frame > 0)].frame.values,
                self.offsets)#isolate interest time period after stim onset

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
        for neuron in stim_dict["forward"].keys():  # generic stim to grab all neurons
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

    def build_booldf_corr(self, stim_arr=None, zero_arr=True):

        if not stim_arr:
            provided = False
        else:
            provided = True

        corr_dict = {}
        bool_dict = {}
        for stim in self.analysis_stim_dict.keys():
            if stim not in bool_dict.keys():
                bool_dict[stim] = {}
                corr_dict[stim] = {}
            for nrn in self.analysis_stim_dict[stim].keys():
                cell_array = self.analysis_stim_dict[stim][nrn]
                if zero_arr:
                    cell_array = np.clip(cell_array, a_min=0, a_max=99)
                if not provided:
                    stim_arr = np.zeros(len(cell_array))

                    # # this ideal stim array works well for the gcamp6s data??
                    stim_arr[-self.offsets[0] + 1 : -self.offsets[0] + self.stim_offset - 2] = 3
                    
                    # this ideal stim array works well for the gcamp7f data
                    # stim_arr[-self.offsets[0]: -self.offsets[0] + self.stim_offset ] = 1

                    stim_arr = arrutils.pretty(stim_arr, 3)
                corrVal = round(np.corrcoef(stim_arr, cell_array)[0][1], 3)

                corr_dict[stim][nrn] = corrVal
                bool_dict[stim][nrn] = corrVal >= self.corr_threshold

        self.booldf = pd.DataFrame(bool_dict)
        self.corrdf = pd.DataFrame(corr_dict)

        return self.corrdf, self.booldf

    def build_booldf_baseline(self):
            baseline_frames = sorted(np.array([np.add(self.stimulus_df['frame'], subtract) for subtract in
                                            range(self.baseline_offset, 0)]).flatten().tolist())
            baseline_normf = pd.DataFrame(self.normcells).iloc[:, baseline_frames]
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

        boundary = np.zeros(self.normcells.shape[0])
        for nrn in range(0, self.normcells.shape[0]):
            gm = mixture.GaussianMixture(n_components=2, random_state=0).fit(pd.DataFrame(self.normcells[nrn]))
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

    def build_dsi_analysis_df(self, roi_name = None, cutoff_val = 0.25, stim_list = None):
        '''
        building a dsi, color, peak motion response dataframe
        dsi is calculated from 4 cardinal directions
        responses to motion are from the analysis_neuron_dict which will be the mean/median/max of the neuron to each stimulus
        that type of response is set by the r_type keyword
        color is calculated from a weighted mean of the responses to the stim_list
        :param roi_name: if you want to get only the dsi from a specific region of interest
        :param cutoff_val: if the neuron does not pass this threshold in its response value for any of the stimuli in stimlist, becomes gray
        :param stim_list: a list of stimulus names that you want to use to get the color combinations from
        :return: dataframe
        '''

        if stim_list is None:
            monoc_stims = list(constants.monocular_dict.keys())
            degree_ids = [constants.deg_dict[i] for i in monoc_stims]
        else:
            monoc_stims = stim_list
            degree_ids = [constants.deg_dict[i] for i in stim_list]
        df = pd.DataFrame(self.analysis_neuron_dict)
        continuous_colors = angles.make_clr_array()

        if roi_name == None:
            select_neurs = range(len(self.f_cells)) 
        else:
            select_neurs = self.return_cells_by_saved_roi(roi_name)  

        self.dsi_df = pd.DataFrame(index = range(len(select_neurs)), columns = ['neuron_id','dsi', 'peak',
                                                                                'mean_response', 'max_response',
                                                                                'color',
                                                                                'location', 'degree_response'])

        dsi_per_neuron = angles.calc_dsi_cardinaldirs(self, base_sec = 4, motion_on_sec = self.seconds_motion_is_on,
                                                      dsi_threshold = cutoff_val, use_df_f = False)
        for r, neuron in enumerate(select_neurs):
            one_neuron_resps = df[neuron][monoc_stims]
            mean_resps_dict = dict(zip(monoc_stims, one_neuron_resps))
            degree_responses = [np.clip(mean_resps_dict[i], a_min=0, a_max=999) for i in monoc_stims]
            neuron_peak = angles.weighted_mean_angle(degree_ids, degree_responses)
            
            dsi = dsi_per_neuron[r]
            
            if dsi == 0: # make non selective neurons gray
                color = [0.5, 0.5, 0.5, 0.15]
            else:
                if np.nanmax(one_neuron_resps) <= cutoff_val: # make grey if not responsive enough (below cut off)
                    color = [0.5, 0.5, 0.5, 0.15]
                else:
                    color = angles.continuous_clr_array(dsi, neuron_peak, continuous_colors) # otherwise get the color

            # add info to the analysis df
            self.dsi_df.iloc[r]['neuron_id'] = neuron
            self.dsi_df.iloc[r]['dsi'] = dsi
            self.dsi_df.iloc[r]['peak'] = neuron_peak
            self.dsi_df.iloc[r]['mean_response'] = np.nanmean(one_neuron_resps)
            self.dsi_df.iloc[r]['max_response'] = np.nanmax(one_neuron_resps)
            self.dsi_df.iloc[r]['color'] = color
            self.dsi_df.iloc[r]['location'] = self.return_singlecell_rois(r)
            self.dsi_df.iloc[r]['degree_response'] = degree_responses

        return self.dsi_df

    def make_computed_image_data(self, colorsumthresh=1, booltrim=False):
        if not hasattr(self, "analysis_stim_dict"):
            self.analysis_stim_dict, _, _  = self.build_stimdicts(self.normcells) # default is normalized cell traces
        xpos = []
        ypos = []
        colors = []
        neurons = []

        for neuron in self.analysis_neuron_dict.keys():
            if booltrim:
                if not hasattr(self, "booldf"):
                    self.corrdf, self.booldf = self.build_booldf_corr()
                if neuron not in self.booldf.index:
                    continue
            myneuron = self.analysis_neuron_dict[neuron]
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
    
    def make_specific_stims_boolean_image_data(self,  thresh, n_percent, selected_neurons = None, stim_class = 'monocular'):
        import angles
        from math import radians

        continuous_colors = angles.make_clr_array()
        colors = []
        xpos = []
        ypos = []
        neur_ids_lst = []

        if not hasattr(self, "analysis_stim_dict"):
            self.analysis_stim_dict, _, _ = self.build_stimdicts(self.normcells) # default is normalized cell traces

        # if not hasattr(self, "corrdf"):
        self.build_booldf_corr() # create correlation (not interested in the booldf) dataframes

        data = self.corrdf

        if selected_neurons is not None: # choosing selected neurons
            data = data.loc[selected_neurons]

        wholefield_stims_boolean = pd.DataFrame(index = data.index, columns = data.columns) # looking at only wholefield stimuli
        for stim in data.columns:
            for nrn in data.index:
                if data[stim][nrn] >= thresh:
                    wholefield_stims_boolean[stim][nrn] = True
                else:
                    wholefield_stims_boolean[stim][nrn] = False

        if stim_class == 'monocular':
            stims_of_interest_dict = constants.monocular_dict
            stims_of_interest_deg_dict = constants.deg_dict
        elif stim_class == 'binocular':
            stims_of_interest_dict = constants.monocular_dict
            stims_of_interest_deg_dict = constants.deg_dict

        wholefield_stims_trimmed = data[stims_of_interest_dict.keys()].loc[wholefield_stims_boolean.index]
        
        sorted_degs_dict = dict.fromkeys(stims_of_interest_dict.keys())
        sorted_vals_dict = dict.fromkeys(stims_of_interest_dict.keys())
        for stim in stims_of_interest_dict.keys():
            stim_bool = wholefield_stims_boolean[wholefield_stims_boolean[stim]==True]
            used_booldf = wholefield_stims_trimmed.copy()
            used_booldf = used_booldf.loc[stim_bool.index]
            used_booldf.fillna(0, inplace = True) # fill nan's with 0's 
            used_booldf = used_booldf.sort_values(by=stim, ascending=False)
            used_booldf = used_booldf.iloc[:int(len(used_booldf)*(n_percent/100))]
            neur_ids_lst.append(list(used_booldf.index))

            # degrees and values for polar plots, using median values in bool df
            comp = used_booldf.median(axis=0) # peak here?
            degs = [radians(stims_of_interest_deg_dict[i]) for i in comp.keys()]
            vals = [i for i in comp]
            sort_index = np.argsort(degs)
            sorted_degs = [degs[i] for i in sort_index] # this is for the whole plane (not individual neurons) #
            sorted_vals = [vals[i] for i in sort_index]
            sorted_degs.append(sorted_degs[0])
            sorted_vals.append(sorted_vals[0])
            sorted_degs_dict[stim] = sorted_degs
            sorted_vals_dict[stim] = sorted_vals

            for nrn in range(len(used_booldf)):
                k = used_booldf.iloc[nrn].keys()
                keys = [stims_of_interest_deg_dict[_] for _ in k] # how to change this for the binocular stims?
                vals = used_booldf.iloc[nrn].values
                theta = angles.weighted_mean_angle(keys, vals)
                dsi = angles.calc_dsi(used_booldf.iloc[nrn])
                clr = angles.continuous_clr_array(dsi, theta, continuous_colors)
                
                full_info = data.loc[used_booldf.iloc[nrn].name]

                roi = self.return_singlecell_rois(full_info.name)
                
                xpos.append(roi[0])
                ypos.append(roi[1])
                colors.append(clr)
        
        neur_ids_lst = [item for sublist in neur_ids_lst for item in sublist]
        
        return xpos, ypos, colors, neur_ids_lst, sorted_degs_dict, sorted_vals_dict

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
        self, xmin=0, xmax=99999, ymin=0, ymax=9999, *args, **kwargs):
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
            booldf = self.normf_baseline_booldf # 1.8 * std dev
            neuron_dict = self.normf_neuron_dict
        elif type == 'normf_cluster':
            booldf = self.normf_cluster_booldf
            neuron_dict = self.normf_neuron_dict
        elif type == 'zdiff_corr':
            booldf = self.zdiff_corr_booldf
            neuron_dict = self.zdiff_neuron_dict

        bool_monoc = booldf[constants.monocular_dict.keys()]
        monoc_bool_neurons = bool_monoc.loc[bool_monoc.sum(axis=1) > 0].index.values
        valid_neurons = [i for i in monoc_bool_neurons if i in neurons]

        thetas = []
        thetavals = []
        degree_ids_dict = {key: None for key in valid_neurons}
        degree_responses_dict = {key: None for key in valid_neurons}
        for n in valid_neurons:
            neuron_response_dict = neuron_dict[n]
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
            degree_ids_dict[n] = degree_ids
            degree_responses_dict[n] = degree_responses
            
        return thetas, thetavals, degree_ids_dict, degree_responses_dict

    def make_various_arrays(self, base_start=4, len_extendedarr=21, len_pre=7, len_on=7, rep_mode="all"):
        '''
        o_t = original traces in shape of:
              [# neurons, # reps (can include missing), # stimuli * frames]
        '''

        o_t = self.neur_resps_each_stim_rep
        n_neurons = o_t.shape[0]
        n_reps = o_t.shape[1]
        n_stim = len(self.stim_order)

        if rep_mode == "common":
            rep_idx = stimuli.get_common_reps(self, len_on)
        else:
            rep_idx = np.arange(self.neur_resps_each_stim_rep.shape[1])

        # --- initialize with NaNs ---
        o_t_base = np.full((n_neurons, n_stim, n_reps), np.nan)
        o_t_base_std = np.full((n_neurons, n_stim, n_reps), np.nan)
        o_t_on_max = np.full((n_neurons, n_stim, n_reps), np.nan)
        o_t_on_min = np.full((n_neurons, n_stim, n_reps), np.nan)
        o_t_on_avg = np.full((n_neurons, n_stim, n_reps), np.nan)
        o_t_diff = np.full((n_neurons, n_stim, n_reps), np.nan)

        o_t_diff_mean = np.full((n_neurons, n_stim), np.nan)

        for i in range(n_neurons):
            for j in range(n_stim):
                for k in rep_idx:

                    trace = o_t[i, k]

                    # skip missing reps
                    if np.all(np.isnan(trace)):
                        continue

                    # frame windows
                    b0 = len_extendedarr * j + base_start
                    b1 = len_extendedarr * j + len_pre
                    on0 = b1
                    on1 = len_extendedarr * j + len_pre + len_on

                    base_win = trace[b0:b1]
                    on_win = trace[on0:on1]

                    # skip if this stim window is missing
                    if np.all(np.isnan(base_win)) or np.all(np.isnan(on_win)):
                        continue

                    # --- stats (NaN safe) ---
                    o_t_base[i, j, k] = np.nanmean(base_win)
                    o_t_base_std[i, j, k] = np.nanstd(base_win)

                    o_t_on_max[i, j, k] = np.nanmax(on_win)
                    o_t_on_min[i, j, k] = np.nanmin(on_win)
                    o_t_on_avg[i, j, k] = np.nanmean(on_win)

                    # diff logic
                    if o_t_on_avg[i, j, k] > o_t_base[i, j, k]:
                        o_t_diff[i, j, k] = o_t_on_max[i, j, k] - o_t_base[i, j, k]
                    else:
                        o_t_diff[i, j, k] = o_t_on_min[i, j, k] - o_t_base[i, j, k]

                # mean across reps (ignoring missing reps)
                o_t_diff_mean[i, j] = np.nanmean(o_t_diff[i, j])

        return o_t_base, o_t_base_std, o_t_on_avg, o_t_on_max, o_t_diff_mean

    def find_general_motion_resp_neurons(self,
                                        frames_motion_on=7, # length in imaging frames for motion on window
                                        base_frames=0, # number of baseline frames from self.offsets[0] to compute baseline window
                                        rep_mode=None, # number of stimulus reps to include for analysis
                                         trace_type = None):  # the type of trace that you want to find these neurons with

        '''
        Allows for multiple number of reps per stimulus
        Identifying motion responsive neurons based on:
        1. correlation during stim-on
        2. peak > baseline + 1.8 * std
        3. response in >= 60% of available trials (per stim)

        o_t shape:
        [# neurons, # reps (can include missing), # stim * frames]
        '''

        if not hasattr(self, "neur_resps_each_stim_rep"):
            self.neur_resps_each_stim_rep = self.neuron_each_stim_rep_arrays(self.stim_order, trace_type)

        o_t = self.neur_resps_each_stim_rep

        n_neurons = o_t.shape[0]
        n_stim = len(self.stim_order)

        if rep_mode is None:
            rep_mode = self.rep_mode
        if rep_mode == "common":
            rep_idx = stimuli.get_common_reps(self, frames_motion_on)
        else:
            rep_idx = np.arange(self.neur_resps_each_stim_rep.shape[1])

        length_subset = np.diff(self.offsets)[0]
        before_stim = -self.offsets[0]

        o_t_base, o_t_base_std, o_t_on_avg, o_t_on_max, o_t_diff_mean = \
            self.make_various_arrays(
                base_start=base_frames,
                len_extendedarr=length_subset,
                len_pre=before_stim,
                len_on=frames_motion_on,
                rep_mode=rep_mode
            )

        resp_dict = BCDict() # boolean, neuron responses to each stimuli
        corr_dict = BCDict() # correlation values with a linear regresssion over the stim on window
        self.motion_responsive_neurons = [] # list of neurons if anything is true

        # basic stim template where the activity increases over the duration of the motion on window
        stim_template = np.linspace(0, 1, frames_motion_on)

        for i in range(n_neurons):
            resp_dict[i] = BCDict()
            corr_dict[i] = BCDict()
            for j in range(n_stim):
                corr_lst = []
                resp_lst = []
                for k in rep_idx:
                    # --- grab this rep's stim-on window ---
                    trace = o_t[i, k]
                    if np.all(np.isnan(trace)):
                        continue  # missing rep entirely
                    win0 = length_subset * j + before_stim
                    win1 = win0 + frames_motion_on
                    cell_arr = trace[win0:win1]
                    if np.all(np.isnan(cell_arr)):
                        continue  # stim missing for this rep
                    # -------- 1. correlation ----------
                    if np.nanstd(cell_arr) == 0:
                        corr_val = np.nan
                    else:
                        corr_val = np.corrcoef(stim_template, cell_arr)[0, 1]
                    corr_lst.append(corr_val)
                    # -------- 2. peak vs baseline ----------
                    if np.isnan(o_t_base[i, j, k]) or np.isnan(o_t_base_std[i, j, k]):
                        continue
                    if o_t_on_max[i, j, k] >= (o_t_base[i, j, k] + 1.8 * o_t_base_std[i, j, k]):
                        resp_lst.append(True)
                    else:
                        resp_lst.append(False)
                # -------- summary stats ----------
                mean_corr = np.nanmean(corr_lst)
                corr_dict[i][j] = mean_corr # the correlations with the stim_template
                # >= 60% of *available* trials
                n_valid_reps = len(resp_lst) # tells us the number of reps of stimuli for this neuron, could change
                if sum(resp_lst) >= int(np.ceil(n_valid_reps * 0.60)):
                    resp_dict[i][j] = True
                else:
                    resp_dict[i][j] = False
            # neuron-level call: responsive to ANY stim
            if any(resp_dict[i].values()):
                self.motion_responsive_neurons.append(i)
        self.corrdf = pd.DataFrame(corr_dict)

        return self.corrdf, self.motion_responsive_neurons

    def run_barcoding(self, stim_order, choice_barcode_dict, n_reps = 4, sec_motion_on = 8, response_threshold = 1.8,
                      baseline_frames = 4, response_type = 'median', trace_type = 'norm'):
        '''
        Running barcoding functions on this same VizStimFish object, so not needed to run in notebook separately
        '''
        from utilities import barcoding
        from math import isnan

        self.stim_order = stim_order
        self.n_reps = n_reps
        self.neur_resps_each_stim_rep = self.neuron_each_stim_rep_arrays(stim_order)
        self.barcode_type_dict, barcode_corr_dict, binary_codes_dict = barcoding.barcode_with_ideal_trace(self, 
                                                                                            frames_motion_on = int(self.img_hz*sec_motion_on), 
                                                                                            barcode_dict = choice_barcode_dict,
                                                                                            n_reps = n_reps,
                                                                                            stim_order = stim_order,
                                                                                            length_of_total_frame_arr = np.diff(self.offsets)[0],
                                                                                            std_thresh = response_threshold,
                                                                                            baseline_len = baseline_frames,
                                                                                            response_type = response_type,
                                                                                            trace_type = trace_type)
        
        forward_resp_cell_lst, backward_resp_cell_lst = barcoding.find_forward_responders(self, frames_motion_on = int(self.img_hz*sec_motion_on), 
                                                                                      std_thresh = response_threshold, n_reps = n_reps, evoked_resp_type = response_type)
        
        #clear the dictionary of cell to barcodes of 'nan's
        self.barcode_type_dict = {key: value for key, value in self.barcode_type_dict.items() if not (isinstance(value, float) and isnan(value))}
        barcode_corr_dict = {key: value for key, value in barcode_corr_dict.items() if not (isinstance(value, float) and isnan(value))}
        binary_codes_dict = {key: value for key, value in binary_codes_dict.items() if key in self.barcode_type_dict.keys()}

        self.barcoding_df = pd.DataFrame(index=range(len(self.barcode_type_dict.keys())), 
                                    columns = ['plane','neur_ids', 'neur_coords', 'barcoding', 'barcode_corr', 'barcode_code', 'forw_resp', 'back_resp'])
        self.barcoding_df['plane'] = [self.folder_path.name] * len(self.barcode_type_dict.keys())
        self.barcoding_df['neur_ids'] = self.barcode_type_dict.keys()
        self.barcoding_df['neur_coords'] = [self.return_singlecell_rois(n) for n in self.barcode_type_dict.keys()]
        self.barcoding_df['barcoding'] = self.barcode_type_dict.values()
        self.barcoding_df['barcode_corr']= barcode_corr_dict.values()
        self.barcoding_df['barcode_code'] = [v for v in binary_codes_dict.values()]
        for i, l in enumerate(self.barcode_type_dict.keys()):
            self.barcoding_df['forw_resp'].iloc[i] = False
            self.barcoding_df['back_resp'].iloc[i] = False
            if l in forward_resp_cell_lst: # if the forward response is True, with using all stims
                self.barcoding_df['forw_resp'].iloc[i] = True
            if l in backward_resp_cell_lst:
                self.barcoding_df['back_resp'].iloc[i] = True

        # adding in some extra columns that I always need
        self.barcoding_df['Pt'] = False
        self.barcoding_df['supp_barcode'] = [None] * len(self.barcoding_df)
        self.barcoding_df['supp_opposite_dir'] = [False] * len(self.barcoding_df)
        self.barcoding_df['side'] = ['R'] * len(self.barcoding_df)

        try:
            pt_neurons = self.return_cells_by_saved_roi('Pt')
        except:
            pt_neurons = []
            print('no Pt on this plane')
        for m, n in enumerate(self.barcoding_df.neur_ids.values):
            # 1 - if pt neurons
            if n in pt_neurons:
                self.barcoding_df.at[m, 'Pt'] = True

            # 2 - is suppressed by opposite direction
            barcode = self.barcoding_df.iloc[m].barcode_code
            barcoding_type = self.barcoding_df.iloc[m].barcoding
            supp_barcode = barcoding.suppression_barcode_score(self, n, stims=self.stim_order, suppression_std_thresh=1)
            self.barcoding_df['supp_barcode'].iloc[m] = supp_barcode
            if 1 in supp_barcode:
                if 'R' in barcoding_type:
                    opposite_direction_inds = [i for i, val in enumerate(self.stim_order) if
                                               ('right' not in val) and ('ing' not in val)]
                if 'L' in barcoding_type:
                    opposite_direction_inds = [i for i, val in enumerate(self.stim_order) if
                                               ('left' not in val) and ('ing' not in val)]
                if 1 in np.array(supp_barcode)[opposite_direction_inds]:
                    self.barcoding_df['supp_opposite_dir'].iloc[m] = True

            # 3 - if on the same side as the barcode
            side_bool = barcoding.check_barcoded_neur_location(self.barcoding_df.neur_coords.iloc[m],
                                                               self.return_x_midline(), barcoding_type)
            side = barcoding_type.split('_')[1]
            if (side_bool == True):
                _side = side
            elif (side_bool == False) & (side == 'R'):
                _side = 'L'
            else:
                _side = 'R'
            self.barcoding_df['side'].iloc[m] = _side
        
        return self.barcoding_df

    def find_bout_reducing_neurons(self, stim_order= constants.bouting_stims,
                                   sec_motion_on=5, std_thresh=1.8,
                                   baseline_frames = None, n_reps=None,
                                   response_type = 'mean',):
        '''
        finding the bout reducing neurons for an opposite barcoded group to stimulate, only from visual motion tuning
        basically just the neurons only responsive to backward
        right now hardcoded to use local df/f for determining responsitivity

        :param stim_order: the stimulus order to follow for gathering responses
        :param sec_motion_on: how long the motion is on for in seconds to determine how many frames to use for barcoding
        :param std_thresh: the threshold to determine if responsive or not
        :param n_reps: number of stimulus reps that the neuron has to pass the threshold for
        :param response_type: 'mean' or 'median' or 'max' for determining if responsive or not
        :return: a list of the neuron indices that are bout reducing
        '''

        from utilities import barcoding

        frames_motion_on = int(self.img_hz * sec_motion_on)
        if n_reps == None:
            n_reps = self.stimulus_df.rep.nunique()
        if baseline_frames == None:
            baseline_frames  = -self.offsets[0]

        inducing_idx = [stim_order.index(s) for s in constants.bout_inducing_stims if s in stim_order]
        reducing_idx = [stim_order.index(s) for s in constants.bout_reducing_stims if s in stim_order]
        backward_idx = stim_order.index('backward') if 'backward' in stim_order else None

        new_stim_resp_each_cell_arr = WorkingFish.neuron_each_stim_rep_arrays(self, stim_order)
        new_stim_start_frames = stimuli.stimulus_start_frames_for_plots(baseline_offset=-self.offsets[0],
                                                                        length_of_total_frame_arr=np.diff(self.offsets)[
                                                                            0],
                                                                        number_of_stims_in_set=len(stim_order))
        bout_reducing_neurons = []
        for n, neuron_arr in enumerate(new_stim_resp_each_cell_arr):
            neuron_binary_code = barcoding.barcode_binary_score_df_f(self,
                                                                     neuron_arr,
                                                                     stims=stim_order,
                                                                     stim_start_frames=new_stim_start_frames,
                                                                     frames_motion_on=frames_motion_on,
                                                                     base_length=baseline_frames,
                                                                     std_thresh=std_thresh,
                                                                     num_responding_trials=int(n_reps * 0.8),
                                                                     evoked_resp=response_type)
            neuron_binary_code = np.array(neuron_binary_code)
            # 1 - not responsive to any bout inducing stimuli
            no_bout_inducing = not neuron_binary_code[inducing_idx].any()

            # 2 - responsive to wholefield backward motion
            has_backward = neuron_binary_code[backward_idx] if backward_idx is not None else False

            # 3 - responsive to any of the bout reducing stimuli?
            has_bout_reducing = neuron_binary_code[reducing_idx].any()

            keep_neuron = no_bout_inducing and has_backward
            if keep_neuron:
                bout_reducing_neurons.append(n)

        return bout_reducing_neurons

    def add_bout_reducing_barcodes_to_barcoding_df(self, bout_reducing_neurons):
        '''
        add the bout reducing neurons to the barcoding df, so that i can have all the information together for experiments
        :param bout_reducing_neurons: the neuron ids of the bout reducing neurons from the function before
        :return: the same barcoding df with these new 'barcoded' neurons added
        note - the barcode_corr column is now a magnitude of response for the backward responses
        '''

        add_df = pd.DataFrame(columns=self.barcoding_df.columns,
                              index=range(len(bout_reducing_neurons)))
        add_df['plane'] = self.folder_path.name
        add_df['neur_ids'] = bout_reducing_neurons
        add_df['neur_coords'] = self.return_cell_rois(bout_reducing_neurons)
        add_df['barcoding'] = 'bout_reducing'
        add_df['back_resp'] = True
        add_df['forw_resp'] = False
        pt_neurons = self.return_cells_by_saved_roi('Pt')
        for m, n in enumerate(add_df.neur_ids.values):
            # if a Pt neuron
            if n in pt_neurons:
                add_df.at[m, 'Pt'] = True

            # gathering intensity of backward response for later sorting, choosing the most responsive neurons
            # this is being put into the df at 'barcode_corr'
            backward_array = np.array(self.extended_responses_normf['backward'][n])
            avg_intensity = []
            for each_rep in backward_array:
                base_mean = np.nanmean(each_rep[:-self.offsets[0]])
                # right now hardcoded for just 5 sec, should be find to finding the magnitude of response
                evoked_mean = np.nanmean(
                    each_rep[-self.offsets[0]: -self.offsets[0] + int(self.img_hz * 5)])
                intensity = evoked_mean - base_mean
                avg_intensity.append(intensity)
            add_df.at[m, 'barcode_corr'] = np.nanmean(avg_intensity)

            # add in the sidedness

            if add_df.iloc[m]['neur_coords'][0] > self.return_x_midline():
                side = 'R'
            else:
                side = 'L'
            add_df.at[m, 'side'] = side

        self.barcoding_df = pd.concat([self.barcoding_df, add_df]).reset_index(drop=True)
        self.barcoding_df = self.barcoding_df.drop_duplicates(subset=['neur_ids'], keep='first')


## THIS DOES NOT WORK WELL BUT KEEPING FOR FUTURE ITERATIONS ##
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

        if not hasattr(self, "f_cells"):
            if 'suite_2p' in self.data_paths.keys():
                self.load_suite2p()
            if 'caiman' in self.data_paths.keys():
                self.load_caiman()
                self.is_cell()
        if hasattr(self, "tail_stimulus_df"):
            self.stimulus_df = stimuli.validate_stims(self.stimulus_df, self.f_cells)
            self.build_stimdicts()
        else:
            pass
        self.bout_locked_dict()
        self.single_bout_avg_neurresp()
        self.avg_bout_avg_neurresp()
        self.neur_responsive_trials()
        self.build_timing_bout_dict()

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
            try:
                newKey = new_fish.folder_path.parents[1].name.split('-')[0].split('_')[1]
            except IndexError:
                newKey = new_fish.folder_path.parents[1].name
            if 'gcamp6s' in str(newKey):
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


# %%
