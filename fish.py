import os
import shutil

import pandas as pd
import numpy as np

try:
    import caiman as cm
except:
    print("no caiman available")

import matplotlib.pyplot as plt
import matplotlib as mpl

from datetime import datetime as dt
from tifffile import imread, imsave
from tqdm import tqdm
from pathlib import Path
from scipy.stats import zscore

try:
    from suite2p.run_s2p import run_s2p, default_ops
except:
    print("no s2p available")

from old_stuff.ImageAnalysisCodesV2 import angles
from arrutils import pretty


class LegacyFish:
    """
    all sorts of extra nonsense and things :) will be cleaned into a new fish
    ...Coming Soon...TM
    """

    def __init__(self, folderPath, stimkey="stims"):
        self.basePath = folderPath
        self.stimKey = stimkey
        self.parsePaths()

        try:
            self.frametimes_df = self.raw_text_frametimes_to_df(
                self.dataPaths["frametimes"]
            )
        except:
            print("no frametimes")
        try:
            self.log_steps_df = self.raw_text_logfile_to_df(
                self.dataPaths["log"], self.frametimes_df
            )
        except:
            print("no log present")

        try:
            self.stimulus_df, self.stimulus_df_condensed = self.pandastim_to_df(
                self.dataPaths["stimuli"]
            )
            self.stimulus_df_condensed.loc[:, "original_frame"] = self.frame_starts()
        except:
            print("no stimuli")

        try:
            self.tag_volume_frames()
        except:
            pass

    def monoc_neuron_colored_vol(self, std_thresh=1.8, alpha=0.75):
        self.parsePaths()
        monocular_dict = {
            "right": [1, 0.25, 0, alpha],
            "left": [0, 0.25, 1, alpha],
            "forward": [0, 1, 0, alpha],
            "backward": [1, 0, 1, alpha],
            "forward_left": [0, 0.75, 1, alpha],
            "forward_right": [0.75, 1, 0, alpha],
            "backward_left": [0.25, 0, 1, alpha],
            "backward_right": [1, 0, 0.25, alpha],
        }
        responses, stds, bool_df = self.return_response_dfs(std_thresh)

        cell_images = []
        ref_images = []
        for vol in self.dataPaths["volumes"].keys():
            ops, iscell, stats, f_cells = self.load_suite2p(
                self.dataPaths["volumes"][vol]["suite2p"]
            )
            plane_df = bool_df[int(vol)][monocular_dict.keys()]

    # single plane
    def monoc_neuron_colored(
        self, vol, std_thresh=1.8, alpha=0.75, kind="full", *args, **kwargs
    ):
        self.parsePaths()
        if kind == "full":
            monocular_dict = {
                "right": [1, 0.25, 0, alpha],
                "left": [0, 0.25, 1, alpha],
                "forward": [0, 1, 0, alpha],
                "backward": [1, 0, 1, alpha],
                "forward_left": [0, 0.75, 1, alpha],
                "forward_right": [0.75, 1, 0, alpha],
                "backward_left": [0.25, 0, 1, alpha],
                "backward_right": [1, 0, 0.25, alpha],
            }
        else:
            monocular_dict = {
                "right": [1, 0.25, 0, alpha],
                "left": [0, 0.25, 1, alpha],
                "forward": [0, 1, 0, alpha],
            }
        responses, stds, bool_df = self.return_response_dfs(std_thresh, *args, **kwargs)
        ops, iscell, stats, f_cells = self.load_suite2p(
            self.dataPaths["volumes"][vol]["suite2p"]
        )
        plane_df = bool_df[int(vol)][monocular_dict.keys()]

        cell_img = np.zeros((ops["Ly"], ops["Lx"], 4), "float64")
        for row in range(len(plane_df)):
            cell = plane_df.iloc[row]

            nrn_color = [0, 0, 0, 0]
            for stim in monocular_dict.keys():
                if cell[stim]:
                    nrn_color = [
                        nrn_color[i] + monocular_dict[stim][i]
                        for i in range(len(nrn_color))
                    ]
                else:
                    pass
            nrn_color = np.clip(nrn_color, a_min=0, a_max=1)
            ypix = stats[cell.name]["ypix"]
            xpix = stats[cell.name]["xpix"]

            for n, c in enumerate(nrn_color):
                cell_img[ypix, xpix, n] = c
        return cell_img, ops["refImg"]

    def return_response_dfs(self, bool_df_thresh=None, stdmode=True, otherThresh=0.05):
        self.parsePaths()
        if stdmode:
            raw_dfs = [
                self.neuron_response_df(vol, r_type="bg_subtracted")
                for vol in self.dataPaths["volumes"].keys()
            ]
            responses = [i[0] for i in raw_dfs]
            stds = [i[1] for i in raw_dfs]
            if not bool_df_thresh:
                return responses, stds
            else:
                bool_dfs = []
                for resp, dev in zip(responses, stds):
                    std_bool_df = (
                        resp >= dev * bool_df_thresh
                    )  # checks if response beats standard dev
                    threshold_bool_df = (
                        resp >= otherThresh
                    )  # checks if response meets base threshold
                    sum_bool_df = (
                        std_bool_df * 1 + threshold_bool_df * 1
                    )  #  converts to 0,1,2
                    bool_df = sum_bool_df >= 2

                    bool_dfs.append(bool_df)
                good_dfs = [
                    bdf[bdf.sum(axis=1) > 0] for bdf in bool_dfs
                ]  # trims it to only neurons that have responses
                return responses, stds, good_dfs
        else:
            raw_dfs = [
                self.neuron_response_df(vol, r_type="median")
                for vol in self.dataPaths["volumes"].keys()
            ]

            responses = [i[0] for i in raw_dfs]
            stds = [i[1] for i in raw_dfs]
            if not bool_df_thresh:
                return responses, stds
            else:
                bool_dfs = []
                for resp, dev in zip(responses, stds):
                    bool_df = resp >= otherThresh
                    bool_dfs.append(bool_df)
                good_dfs = [
                    bdf[bdf.sum(axis=1) > 0] for bdf in bool_dfs
                ]  # trims it to only neurons that have responses
                return responses, stds, good_dfs

    def volume_barcode_class_counts(self):
        self.parsePaths()
        volumes = self.dataPaths["volumes"].keys()
        responses = [
            self.neuron_response_df(vol, r_type="bg_subtracted") for vol in volumes
        ]
        barcodes = [self.generate_barcodes_fromstd(r) for r in responses]

        master = {}
        for barcode in barcodes:
            counted_df = (
                barcode.groupby("fullcomb")
                .count()
                .sort_values(ascending=False, by="neuron")
            )
            groupings = counted_df.index
            counts = counted_df.neuron.values
            for n, val in enumerate(groupings):

                count = counts[n]

                if val in master.keys():
                    master[val].append(count)
                else:
                    master[val] = [count]

        def fill_to_n(_list_, n):
            while len(_list_) < n:
                _list_.append(0)
            return

        max_n = len(master[0])
        _ = [fill_to_n(l, max_n) for k, l in master.items() if len(l) < max_n]
        master = pd.DataFrame(master)
        newAxis = (
            pd.DataFrame(master.astype(bool).sum(axis=0))
            .sort_values(ascending=False, by=0)
            .index.values
        )
        master = master[newAxis]

        melted = pd.melt(master, ignore_index=False)
        melted = melted.reset_index().rename(
            columns={"index": "volume", "variable": "master_combo"}
        )
        return melted

    def stimblast_cell(self, cell, vol, start_offset=10, end_offset=20):
        plot_dictionary = {
            (0, 0): "right",
            (0, 1): "medial_right",
            (0, 2): "lateral_right",
            (0, 3): "converging",
            (1, 0): "left",
            (1, 1): "medial_left",
            (1, 2): "lateral_left",
            (1, 3): "diverging",
            (2, 0): "forward",
            (2, 1): "x_forward",
            (2, 2): "forward_x",
            (2, 3): "forward_backward",
            (3, 0): "backward",
            (3, 1): "x_backward",
            (3, 2): "backward_x",
            (3, 3): "backward_forward",
            (4, 0): "forward_left",
            (4, 1): "forward_right",
            (4, 2): "backward_left",
            (4, 3): "backward_right",
        }

        col = str(vol) + "_frame"
        if col not in self.stimulus_df_condensed.columns:
            self.tag_volume_frames()

        ops, iscell, stats, f_cells = self.load_suite2p(
            self.dataPaths["volumes"][str(vol)]["suite2p"]
        )
        f_cells = self.norm_fdff(f_cells)
        neuron = f_cells[cell]

        fig, ax = plt.subplots(5, 4, figsize=(10, 10))

        for a in ax:
            for b in a:
                b.set_xticks([])
                # b.set_yticks([])
                b.set_ylim(-0.25, 1.0)

        for k, v in plot_dictionary.items():
            ax[k].set_title(v)

            stimmy = self.stimulus_df_condensed[
                self.stimulus_df_condensed.stim_name == v
            ]

            chunks = []
            for s in stimmy[col]:
                chunk = neuron[s - start_offset : s + end_offset]
                ax[k].plot(chunk)
                chunks.append(chunk)

            ax[k].plot(np.mean(chunks, axis=0), color="black", linewidth=2.5)
            ax[k].axvspan(start_offset, start_offset + 5, color="red", alpha=0.3)

        fig.tight_layout()
        plt.show()

    def stimblast_cell_limited(
        self, cell, vol, start_offset=10, end_offset=20, save=None
    ):
        plot_dictionary = {
            (0, 0): "right",
            (0, 1): "medial_right",
            (0, 2): "lateral_right",
            (0, 3): "converging",
            (0, 4): "left",
            (0, 5): "medial_left",
            (0, 6): "lateral_left",
            (1, 0): "diverging",
            (1, 1): "forward",
            (1, 2): "backward",
            (1, 3): "forward_left",
            (1, 4): "forward_right",
            (1, 5): "backward_left",
            (1, 6): "backward_right",
        }

        col = str(vol) + "_frame"
        if col not in self.stimulus_df_condensed.columns:
            self.tag_volume_frames()

        ops, iscell, stats, f_cells = self.load_suite2p(
            self.dataPaths["volumes"][str(vol)]["suite2p"]
        )
        f_cells = self.norm_fdff(f_cells)
        if isinstance(cell, int):
            neuron = f_cells[cell]
        else:
            neuron = np.nanmean(f_cells[cell], axis=0)

        fig, ax = plt.subplots(2, 7, figsize=(14, 6))

        for a in ax:
            for b in a:
                b.set_xticks([])
                # b.set_yticks([])
                b.set_ylim(-0.25, 1.0)

        for k, v in plot_dictionary.items():
            ax[k].set_title(v)

            stimmy = self.stimulus_df_condensed[
                self.stimulus_df_condensed.stim_name == v
            ]

            chunks = []
            for s in stimmy[col]:
                chunk = neuron[s - start_offset : s + end_offset]
                ax[k].plot(chunk)
                chunks.append(chunk)

            try:
                ax[k].plot(np.mean(chunks, axis=0), color="black", linewidth=2.5)
            except:
                ax[k].plot(np.mean(chunks[:-1], axis=0), color="black", linewidth=2.5)
            ax[k].axvspan(start_offset, start_offset + 5, color="red", alpha=0.3)

        fig.tight_layout()
        if save is not None:
            plt.savefig(save, format="svg")
        plt.show()

    def plot_cell(self, cells, vol, pretty=False, save=None):
        ops, iscell, stats, f_cells = self.load_suite2p(
            self.dataPaths["volumes"][str(vol)]["suite2p"]
        )
        cell_img = np.zeros((ops["Ly"], ops["Lx"]))

        if pretty:
            import cv2

        if isinstance(cells, int):
            cells = [cells]
        z = 1
        for cell in cells:
            ypix = stats[cell]["ypix"]
            xpix = stats[cell]["xpix"]
            if not pretty:
                cell_img[ypix, xpix] = 1
            else:
                mean_y = int(np.mean(ypix))
                mean_x = int(np.mean(xpix))
                cv2.circle(cell_img, (mean_x, mean_y), 3, z, -1)
                z += 1

        masked = np.ma.masked_where(cell_img == 0, cell_img)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(ops["refImg"], cmap=mpl.cm.gray, vmax=300)
        ax.imshow(
            masked,
            cmap=mpl.cm.gist_rainbow,
            interpolation=None,
            alpha=1,
            vmax=np.max(masked),
            vmin=0,
        )
        plt.axis("off")
        if save is not None:
            plt.savefig(save, format="svg")
        plt.show()

    def neuron_response_df(self, vol=0, offset=5, r_type="mean"):
        col = str(vol) + "_frame"
        if col not in self.stimulus_df_condensed.columns:
            self.tag_volume_frames()

        ops, iscell, stats, f_cells = self.load_suite2p(
            self.dataPaths["volumes"][str(vol)]["suite2p"]
        )
        f_cells = self.norm_fdff(f_cells)

        # single plane
        neuron_responses = {}
        neuron_stds = {}
        for stimulus in self.stimulus_df_condensed.stim_name.unique():
            stimmy_df = self.stimulus_df_condensed[
                self.stimulus_df_condensed.stim_name == stimulus
            ]
            starts = stimmy_df[col].values
            if r_type == "mean":
                stim_arr = np.concatenate([np.arange(a, a + offset) for a in starts])
                meanVals = np.nanmean(f_cells[:, stim_arr], axis=1)
                stdVals = [None]
            elif r_type == "median":
                stim_arr = np.concatenate([np.arange(a, a + offset) for a in starts])
                meanVals = np.nanmedian(f_cells[:, stim_arr], axis=1)
                stdVals = [None]
            elif r_type == "peak":
                stim_arr = np.concatenate([np.arange(a, a + offset) for a in starts])
                meanVals = np.nanmax(f_cells[:, stim_arr], axis=1)
            elif r_type == "bg_subtracted":
                allArrs = []
                for start_val in starts:
                    stimArr = f_cells[:, start_val + 2 : start_val + 2 + offset]
                    bgArr = f_cells[:, start_val - offset : start_val - 1]
                    diffArr = np.nanmean(stimArr, axis=1) - np.nanmean(bgArr, axis=1)
                    allArrs.append(diffArr)
                meanVals = np.nanmean(allArrs, axis=0)
                stdVals = np.nanstd(allArrs, axis=0)

            else:
                stim_arr = np.concatenate([np.arange(a, a + offset) for a in starts])
                meanVals = np.nanmean(f_cells[:, stim_arr], axis=1)

            neuron_responses[stimulus] = meanVals
            neuron_stds[stimulus] = stdVals

        return pd.DataFrame(neuron_responses), pd.DataFrame(neuron_stds)

    def singlePixelWise(self, plot=False, *args, **kwargs):
        self.parsePaths()
        img = imread(self.dataPaths["image"])

        diff = self.cardinal_pixelwise(
            img, self.frametimes_df, vols=False, *args, **kwargs
        )
        if plot:
            plt.figure(figsize=(8, 8))
            plt.imshow(diff)
            plt.axis("off")
            plt.show()
        else:
            return diff

    def volumePixelwise(self, _return=False, *args, **kwargs):
        self.parsePaths()
        if _return:
            diffs = []
        for v in self.dataPaths["volumes"].keys():
            img = imread(self.dataPaths["volumes"][v]["image"])

            frametimes = pd.read_hdf(self.dataPaths["volumes"][v]["frametimes"])
            frametimes.reset_index(inplace=True)
            frametimes.rename({"index": "raw_index"}, axis=1, inplace=True)

            diff = self.cardinal_pixelwise(img, frametimes, *args, **kwargs)
            if not _return:
                plt.figure(figsize=(8, 8))
                plt.imshow(diff)
                plt.title(v)
                plt.show()
            else:
                diffs.append(diff)
        if not _return:
            return
        return diffs

    def cardinal_pixelwise(
        self,
        pic,
        frametimes,
        offset=5,
        brighterFactor=1.5,
        brighter=10,
        vols=True,
        invert=False,
    ):
        if not invert:
            cardinals = {
                "forward": [0, 1, 0],
                "forward_left": [0, 0.75, 1],
                "left": [0, 0.25, 1],
                "backward_left": [0.25, 0, 1],
                "backward": [1, 0, 1],
                "backward_right": [1, 0, 0.25],
                "right": [1, 0.25, 0],
                "forward_right": [0.75, 1, 0],
            }
        else:
            pic = pic[:, :, ::-1]
            # this one handles fish inversion
            cardinals = {
                "backward": [0, 1, 0],
                "backward_right": [0, 0.75, 1],
                "right": [0, 0.25, 1],
                "forward_right": [0.25, 0, 1],
                "forward": [1, 0, 1],
                "forward_left": [1, 0, 0.25],
                "left": [1, 0.25, 0],
                "backward_left": [0.75, 1, 0],
            }
        diff_imgs = {}
        for stimulus_name in cardinals.keys():
            _stims = self.stimulus_df_condensed[
                self.stimulus_df_condensed.stim_name == stimulus_name
            ]
            _img = []
            for ind in _stims.original_frame.values:
                if vols:
                    s = frametimes.loc[frametimes.raw_index >= ind].index[0]
                else:
                    s = ind
                # img = np.nanmean(pic[s : s + offset], axis=0)
                img = np.nanmean(pic[s : s + 5], axis=0)
                # bg = np.nanmean(pic[s - offset : s], axis=0)
                bg = np.nanmean(pic[s - 10 : s - 3], axis=0)
                _img.append(img - bg)
            diff_img = np.mean(_img, axis=0)

            diff_imgs[stimulus_name] = diff_img

        maxVal = np.max([np.max(i) for i in diff_imgs.values()])

        imgs = []
        for name, image in diff_imgs.items():

            image[image < 0] = 0

            r = image * cardinals[name][0]
            g = image * cardinals[name][1]
            b = image * cardinals[name][2]

            r /= maxVal
            g /= maxVal
            b /= maxVal

            r -= r.min()
            g -= g.min()
            b -= b.min()
            imgs.append(
                np.dstack(
                    (r**brighterFactor, g**brighterFactor, b**brighterFactor)
                )
            )

        somenewmaxval = np.max(imgs)

        _all_img = []
        for img in imgs:
            _all_img.append(img / somenewmaxval)

        fin_img = np.sum(_all_img, axis=0)
        fin_img /= np.max(fin_img)
        return fin_img * brighter

    def raw_text_logfile_to_df(self, log_path, frametimes=None):
        with open(log_path) as file:
            contents = file.read()
        split = contents.split("\n")

        movesteps = []
        times = []
        for line in range(len(split)):
            if (
                "piezo" in split[line]
                and "connected" not in split[line]
                and "stopped" not in split[line]
            ):
                t = split[line].split(" ")[0][:-1]
                z = split[line].split(" ")[6]
                try:
                    if isinstance(eval(z), float):
                        times.append(dt.strptime(t, "%H:%M:%S.%f").time())
                        movesteps.append(z)
                except NameError:
                    continue
        else:
            # last line is blank and likes to error out
            pass
        log_steps = pd.DataFrame({"time": times, "steps": movesteps})

        if frametimes is not None:
            log_steps = self.log_aligner(log_steps, frametimes)
        else:
            pass
        return log_steps

    def img_splitter(self, clip=True, force=False):
        if not force:
            if "volumes" in self.dataPaths:
                print("skipped image split")
                return
        # imgOffs = [5, 4, 3, 2, 1]
        self.log_steps_df.steps = np.array(self.log_steps_df.steps, dtype=np.float32)
        diffArr = np.diff(self.log_steps_df.steps)
        _align_starts = np.where(diffArr == 0.5)
        align_starts = list(np.where(np.diff(_align_starts) > 200)[1])
        align_starts = [_align_starts[0][i + 1] for i in align_starts]
        align_starts = [_align_starts[0][0]] + align_starts
        # # stepResetVal = np.min(diffArr)
        # stepResetVal = 5 * np.median(diffArr)
        # imgSets = [j - i for i in imgOffs for j in np.where(diffArr == stepResetVal)]
        # imgSetDfs = [self.log_steps_df.iloc[imgset] for imgset in imgSets]
        stepSize = np.median(diffArr)
        sets = []
        for n, i in enumerate(align_starts):
            if n == 0:
                sets.append(self.log_steps_df.iloc[:i])
            elif 0 < n <= len(align_starts) - 1:
                sets.append(self.log_steps_df.iloc[align_starts[n - 1] : i])
        sets.append(self.log_steps_df.iloc[i:])

        def find_n_sep(vals, n):
            steppers = np.arange(6)[1:] * n
            for val in vals:
                new_arr = np.subtract(vals, val)
                if all(q in new_arr for q in steppers):
                    return val

        plane_dictionary = {0: [], 1: [], 2: [], 3: [], 4: []}

        for n, dset in enumerate(sets):
            magic_number = find_n_sep(dset.steps.unique(), stepSize)

            for v in plane_dictionary.keys():
                plane_dictionary[v].append(
                    dset[dset.steps == magic_number + (stepSize * v)]
                )

        imgSetDfs = [pd.concat(p) for p in plane_dictionary.values()]
        ourImgs = {}
        for n, imgsetdf in enumerate(tqdm(imgSetDfs, "ind calculation")):
            ourImgs[n] = []
            for row_i in range(len(imgsetdf)):
                row = imgsetdf.iloc[row_i]
                tval = row.time

                indVal = self.frametimes_df[self.frametimes_df["time"] >= tval].index[0]
                ourImgs[n].append(indVal)

        img = imread(self.dataPaths["image"])
        for key in tqdm(ourImgs, "img splitting"):
            imgInds = ourImgs[key]
            subStack = img[imgInds]
            if clip:
                subStack = subStack[:, :, 20:]

            subStackPath = Path(self.basePath).joinpath(f"img_stack_{key}")
            if not os.path.exists(subStackPath):
                os.mkdir(subStackPath)

            subStackImgPath = subStackPath.joinpath("image.tif")
            imsave(subStackImgPath, subStack)

            subStackFtPath = subStackPath.joinpath("frametimes.h5")
            if os.path.exists(subStackFtPath):
                os.remove(subStackFtPath)
            self.frametimes_df.loc[imgInds].to_hdf(subStackFtPath, key="frametimes")

    def singleMoveCorrection(self):

        imgpath = self.dataPaths["image"]
        movement_image = self.movement_correction(imgpath)

        newdir = imgpath.parents[0].joinpath("original_image")

        if not os.path.exists(newdir):
            os.mkdir(newdir)

        new_path = newdir.parents[0].joinpath("movement_corr_img.tif")

        og_img_path = newdir.joinpath("image.tif")
        if os.path.exists(og_img_path):
            os.remove(og_img_path)

        shutil.move(imgpath, og_img_path)
        imsave(new_path, movement_image)

        prev_path = Path(self.dataPaths["image"].parents[0]).joinpath("image.tif")
        if os.path.exists(prev_path) and os.path.exists(new_path):
            os.remove(prev_path)

    def singleSuite2p(self, input_tau=1.5, image_hz=None):
        imagepath = self.dataPaths["image"]

        if image_hz:
            imageHz = image_hz
        else:
            imageHz = self.hzReturner(self.frametimes_df)

        s2p_ops = {
            "data_path": [imagepath.parents[0].as_posix()],
            "save_path0": imagepath.parents[0].as_posix(),
            "tau": input_tau,
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
        return

    def volumeSuite2p(self, input_tau=1.5, force=False):
        self.parsePaths()
        assert "volumes" in self.dataPaths, "must contain image volumes"

        if not force:
            try:
                if os.path.exists(
                    self.dataPaths["volumes"]["0"]["image"]
                    .parents[0]
                    .joinpath("suite2p")
                ):
                    print("skipped suite2p")
                    return
            except:
                pass

        for key in tqdm(self.dataPaths["volumes"].keys(), "planes: "):
            imagepath = self.dataPaths["volumes"][key]["image"]

            frametimepath = self.dataPaths["volumes"][key]["frametimes"]
            frametimes = pd.read_hdf(frametimepath)
            imageHz = self.hzReturner(frametimes)

            s2p_ops = {
                "data_path": [imagepath.parents[0].as_posix()],
                "save_path0": imagepath.parents[0].as_posix(),
                "tau": input_tau,
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

        return

    def volumeMoveCorrection(self, force=False):
        self.parsePaths()

        if not force:
            try:
                if os.path.exists(
                    (
                        self.dataPaths["volumes"]["0"]["image"]
                        .parents[0]
                        .joinpath("original_image")
                    )
                ):
                    print("skipped move correct")
                    return
            except:
                pass

        assert "volumes" in self.dataPaths, "must contain image volumes"

        for key in tqdm(self.dataPaths["volumes"].keys(), "planes: "):
            imgpath = self.dataPaths["volumes"][key]["image"]

            if not os.path.exists(imgpath):
                print(f"skipping {imgpath}")
                continue
            movement_image = self.movement_correction(imgpath)

            newdir = imgpath.parents[0].joinpath("original_image")

            if not os.path.exists(newdir):
                os.mkdir(newdir)

            new_path = newdir.parents[0].joinpath("movement_corr_img.tif")

            og_img_path = newdir.joinpath("image.tif")
            if os.path.exists(og_img_path):
                os.remove(og_img_path)
            shutil.move(imgpath, og_img_path)
            imsave(new_path, movement_image)

            prev_path = Path(
                self.dataPaths["volumes"][key]["image"].parents[0]
            ).joinpath("image.tif")
            if os.path.exists(prev_path) and os.path.exists(new_path):
                os.remove(prev_path)

    def frame_starts(self):
        return [
            self.frametimes_df[
                self.frametimes_df.time >= self.stimulus_df_condensed.time.values[i]
            ].index[0]
            for i in range(len(self.stimulus_df_condensed))
        ]

    def tag_volume_frames(self):

        for v in self.dataPaths["volumes"].keys():

            frametimes = pd.read_hdf(self.dataPaths["volumes"][v]["frametimes"])
            frametimes.reset_index(inplace=True)
            frametimes.rename({"index": "raw_index"}, axis=1, inplace=True)

            f_starts = [
                frametimes.loc[frametimes.raw_index >= ind].index[0]
                for ind in self.stimulus_df_condensed.original_frame.values
            ]

            self.stimulus_df_condensed.loc[:, v + "_frame"] = f_starts

    def parsePaths(self):
        self.dataPaths = {"volumes": {}}
        with os.scandir(self.basePath) as entries:
            for entry in entries:
                if entry.name.endswith(".tif"):
                    self.dataPaths["image"] = Path(entry.path)
                elif entry.name.endswith(".txt") and "log" in entry.name:
                    self.dataPaths["log"] = Path(entry.path)
                elif entry.name.endswith(".txt") and self.stimKey in entry.name:
                    self.dataPaths["stimuli"] = Path(entry.path)
                elif entry.name.endswith(".txt"):
                    self.dataPaths["frametimes"] = Path(entry.path)

                # this one explores img stack folders
                if os.path.isdir(entry.path):
                    if "img_stack" in entry.name:
                        key = entry.name.split("_")[-1]
                        self.dataPaths["volumes"][key] = {
                            "frametimes": Path(entry.path).joinpath("frametimes.h5"),
                        }
                        with os.scandir(entry.path) as subentries:
                            for subentry in subentries:
                                if subentry.name.endswith(".tif"):
                                    self.dataPaths["volumes"][key]["image"] = Path(
                                        subentry.path
                                    )
                                if "suite2p" in subentry.name:
                                    self.dataPaths["volumes"][key]["suite2p"] = {
                                        "iscell": Path(subentry.path).joinpath(
                                            "plane0/iscell.npy"
                                        ),
                                        "stats": Path(subentry.path).joinpath(
                                            "plane0/stat.npy"
                                        ),
                                        "ops": Path(subentry.path).joinpath(
                                            "plane0/ops.npy"
                                        ),
                                        "f_cells": Path(subentry.path).joinpath(
                                            "plane0/F.npy"
                                        ),
                                        "f_neuropil": Path(subentry.path).joinpath(
                                            "plane0/Fneu.npy"
                                        ),
                                        "spikes": Path(subentry.path).joinpath(
                                            "plane0/spks.npy"
                                        ),
                                        "data": Path(subentry.path).joinpath(
                                            "plane0/data.bin"
                                        ),
                                    }

    def enact_purge(self):
        self.parsePaths()
        import shutil

        keys = self.dataPaths["volumes"].keys()
        for key in keys:
            p = list(self.dataPaths["volumes"][key].values())[0].parents[0]
            try:
                os.remove(p)
            except:
                pass
            try:
                shutil.rmtree(p)
            except:
                pass

    def legacy_volumesplit(self, len_thresh=150, crop=False, force=False):
        if len(self.dataPaths["volumes"].keys()) > 1:
            if not force:
                print("skipping volume split")
            else:
                image = imread(self.dataPaths["image"])
                frametimes = self.legacy_raw_text_frametimes_to_df(
                    self.dataPaths["frametimes"]
                )
                log_steps = self.legacy_raw_text_logfile_to_df(self.dataPaths["log"])
                frametimes = self.legacy_alignmentFramesSteps(
                    frametimes, log_steps, time_offset=0.009
                )
                if crop:
                    image = image[:, :, image.shape[1] // 10 :]

                for n, s in enumerate(tqdm(frametimes.step.unique())):
                    imgInds = frametimes[frametimes.step == s].index
                    # new_fts = frametimes[frametimes.step == s].drop(columns="step")

                    sub_img = image[imgInds]
                    if len(sub_img) >= len_thresh:

                        subStackPath = Path(self.basePath).joinpath(f"img_stack_{n}")
                        if not os.path.exists(subStackPath):
                            os.mkdir(subStackPath)

                        subStackFtPath = subStackPath.joinpath("frametimes.h5")
                        if os.path.exists(subStackFtPath):
                            os.remove(subStackFtPath)
                        self.frametimes_df.loc[imgInds].to_hdf(
                            subStackFtPath, key="frametimes"
                        )

                        subStackImgPath = subStackPath.joinpath("image.tif")
                        imsave(subStackImgPath, sub_img)
                del image
        else:
            image = imread(self.dataPaths["image"])
            frametimes = self.legacy_raw_text_frametimes_to_df(
                self.dataPaths["frametimes"]
            )
            log_steps = self.legacy_raw_text_logfile_to_df(self.dataPaths["log"])
            frametimes = self.legacy_alignmentFramesSteps(
                frametimes, log_steps, time_offset=0.009
            )
            if crop:
                image = image[:, :, image.shape[1] // 10 :]

            for n, s in enumerate(tqdm(frametimes.step.unique())):
                imgInds = frametimes[frametimes.step == s].index
                # new_fts = frametimes[frametimes.step == s].drop(columns="step")

                sub_img = image[imgInds]
                if len(sub_img) >= len_thresh:

                    subStackPath = Path(self.basePath).joinpath(f"img_stack_{n}")
                    if not os.path.exists(subStackPath):
                        os.mkdir(subStackPath)

                    subStackFtPath = subStackPath.joinpath("frametimes.h5")
                    if os.path.exists(subStackFtPath):
                        os.remove(subStackFtPath)
                    self.frametimes_df.loc[imgInds].to_hdf(
                        subStackFtPath, key="frametimes"
                    )

                    subStackImgPath = subStackPath.joinpath("image.tif")
                    imsave(subStackImgPath, sub_img)
            del image

    def zdiff_stimdicts(self, used_offsets=(-10, 14), invertstims=True):
        invStimDict = {
            "medial_right": "medial_left",
            "medial_left": "medial_right",
            "right": "left",
            "left": "right",
            "converging": "converging",
            "diverging": "diverging",
            "lateral_left": "lateral_right",
            "lateral_right": "lateral_left",
            "forward": "backward",
            "backward": "forward",
            "forward_left": "backward_right",
            "backward_left": "forward_right",
            "backward_right": "forward_left",
            "forward_right": "backward_left",
            "x_forward": "backward_x",
            "forward_x": "x_backward",
            "backward_x": "x_forward",
            "x_backward": "forward_x",
            "forward_backward": "forward_backward",
            "backward_forward": "backward_forward",
        }
        if invertstims:
            self.stimulus_df_condensed.loc[
                :, "stim_nameINV"
            ] = self.stimulus_df_condensed.stim_name.map(invStimDict)

            if len(self.dataPaths["volumes"].keys()) > 1:

                self.stimdicts = {
                    v: {"meanArr": None, "errArr": None}
                    for v in self.dataPaths["volumes"].keys()
                }
                for vol in tqdm(self.dataPaths["volumes"].keys()):
                    col = vol + "_frame"

                    ops, iscell, stats, f_cells = self.load_suite2p(
                        self.dataPaths["volumes"][str(vol)]["suite2p"]
                    )
                    pretty_cells = [self.normcell(i) for i in f_cells]
                    stimDict = {
                        i: {} for i in self.stimulus_df_condensed.stim_nameINV.unique()
                    }
                    errDict = {
                        i: {} for i in self.stimulus_df_condensed.stim_nameINV.unique()
                    }
                    for stim in self.stimulus_df_condensed.stim_nameINV.unique():
                        arrs = self.arrangedArrays(
                            self.stimulus_df_condensed[
                                self.stimulus_df_condensed.stim_nameINV == stim
                            ][col],
                            used_offsets,
                        )
                        for n, nrn in enumerate(pretty_cells):
                            resp_arrs = []
                            for arr in arrs:
                                resp_arrs.append(nrn[arr])
                            stimDict[stim][n] = np.nanmean(resp_arrs, axis=0)
                            errDict[stim][n] = np.nanstd(resp_arrs, axis=0) / np.sqrt(
                                len(resp_arrs)
                            )
                    self.stimdicts[vol]["meanArr"] = stimDict
                    self.stimdicts[vol]["errArr"] = errDict
        else:
            pass

    def zdiff_booldf(
        self, threshold=0.65, used_offsets=(-10, 13), stim_offset=5, zero_arr=True
    ):
        if len(self.dataPaths["volumes"].keys()) > 1:
            self.bool_dfs = []
            for vol in tqdm(self.dataPaths["volumes"].keys()):
                planeDict = {}
                for stim in self.stimdicts[vol]["meanArr"].keys():
                    if stim not in planeDict.keys():
                        planeDict[stim] = {}
                    for nrn in self.stimdicts[vol]["meanArr"][stim].keys():

                        cellArr = self.stimdicts[vol]["meanArr"][stim][nrn]
                        if zero_arr:
                            cellArr = np.clip(cellArr, a_min=0, a_max=99)
                        stimArr = np.zeros(len(cellArr))
                        stimArr[
                            -used_offsets[0] : -used_offsets[0] + stim_offset - 2
                        ] = 1.5
                        stimArr = pretty(stimArr)
                        corrVal = round(np.corrcoef(stimArr, cellArr)[0][1], 3)
                        if corrVal >= threshold:
                            planeDict[stim][nrn] = True
                        else:
                            planeDict[stim][nrn] = False
                planedf = pd.DataFrame(planeDict)
                planedf = planedf.loc[planedf.sum(axis=1) > 0]
                self.bool_dfs.append(planedf)

        else:
            pass

    def doublePlot(
        self, alpha=0.65, used_offsets=(-10, 13), threshold=0.7, brighter=10
    ):

        monocular_dict = {
            "right": [1, 0.25, 0, alpha],
            "left": [0, 0.25, 1, alpha],
            "forward": [0, 1, 0, alpha],
            "backward": [1, 0, 1, alpha],
            "forward_left": [0, 0.75, 1, alpha],
            "forward_right": [0.75, 1, 0, alpha],
            "backward_left": [0.25, 0, 1, alpha],
            "backward_right": [1, 0, 0.25, alpha],
        }
        fig, ax = plt.subplots(2, 5, figsize=(24, 10))

        pixelwiseDiff = self.volumePixelwise(
            _return=True, invert=True, brighter=brighter
        )
        for n, img in enumerate(pixelwiseDiff):
            ax[0][n].imshow(img)

        if not hasattr(self, "bool_dfs"):
            self.zdiff_stimdicts(used_offsets=used_offsets)
            self.zdiff_booldf(used_offsets=used_offsets, threshold=threshold)

        for m, vol in enumerate(self.dataPaths["volumes"].keys()):
            plane_df = self.bool_dfs[int(vol)][monocular_dict.keys()]
            ops, iscell, stats, f_cells = self.load_suite2p(
                self.dataPaths["volumes"][vol]["suite2p"]
            )
            cell_img = np.zeros((ops["Ly"], ops["Lx"], 4), "float64")
            for row in range(len(plane_df)):
                cell = plane_df.iloc[row]

                nrn_color = [0, 0, 0, 0]
                for stim in monocular_dict.keys():
                    if cell[stim]:
                        nrn_color = [
                            nrn_color[i] + monocular_dict[stim][i]
                            for i in range(len(nrn_color))
                        ]
                    else:
                        pass
                nrn_color = np.clip(nrn_color, a_min=0, a_max=1)
                ypix = stats[cell.name]["ypix"]
                xpix = stats[cell.name]["xpix"]

                for n, c in enumerate(nrn_color):
                    cell_img[ypix, xpix, n] = c

            ax[1][m].imshow(
                ops["refImg"][:, ::-1],
                cmap="gray",
                alpha=0.75,
                vmax=np.percentile(ops["refImg"], 99.5),
            )
            ax[1][m].imshow(cell_img[:, ::-1])

        plt.show()

    def triplePlot(
        self,
        alpha=0.65,
        used_offsets=(-10, 13),
        threshold=0.7,
        brighter=10,
        stim_offset=5,
    ):

        monocular_dict = {
            "right": [1, 0.25, 0, alpha],
            "left": [0, 0.25, 1, alpha],
            "forward": [0, 1, 0, alpha],
            "backward": [1, 0, 1, alpha],
            "forward_left": [0, 0.75, 1, alpha],
            "forward_right": [0.75, 1, 0, alpha],
            "backward_left": [0.25, 0, 1, alpha],
            "backward_right": [1, 0, 0.25, alpha],
        }
        fig, ax = plt.subplots(3, 5, figsize=(45, 15))

        pixelwiseDiff = self.volumePixelwise(
            _return=True, invert=True, brighter=brighter
        )
        for n, img in enumerate(pixelwiseDiff):
            ax[0][n].imshow(img)
            ax[0][n].axis("off")

        if not hasattr(self, "bool_dfs"):
            self.zdiff_stimdicts(used_offsets=used_offsets)
            self.zdiff_booldf(used_offsets=used_offsets, threshold=threshold)

        for m, vol in enumerate(self.dataPaths["volumes"].keys()):
            plane_df = self.bool_dfs[int(vol)][monocular_dict.keys()]
            ops, iscell, stats, f_cells = self.load_suite2p(
                self.dataPaths["volumes"][vol]["suite2p"]
            )
            cell_img = np.zeros((ops["Ly"], ops["Lx"], 4), "float64")
            for row in range(len(plane_df)):
                cell = plane_df.iloc[row]

                nrn_color = [0, 0, 0, 0]
                for stim in monocular_dict.keys():
                    if cell[stim]:
                        nrn_color = [
                            nrn_color[i] + monocular_dict[stim][i]
                            for i in range(len(nrn_color))
                        ]
                    else:
                        pass
                nrn_color = np.clip(nrn_color, a_min=0, a_max=1)
                ypix = stats[cell.name]["ypix"]
                xpix = stats[cell.name]["xpix"]

                for n, c in enumerate(nrn_color):
                    cell_img[ypix, xpix, n] = c

            ax[1][m].imshow(
                ops["refImg"][:, ::-1],
                cmap="gray",
                alpha=0.75,
                vmax=np.percentile(ops["refImg"], 99.5),
            )
            ax[1][m].imshow(cell_img[:, ::-1])
            ax[1][m].axis("off")

        monocStims = [
            "right",
            "left",
            "forward",
            "backward",
            "forward_left",
            "forward_right",
            "backward_left",
            "backward_right",
        ]
        pltdict = {0: 11, 1: 12, 2: 13, 3: 14, 4: 15}
        for g, vol in enumerate(self.dataPaths["volumes"].keys()):
            nrnDict = {}

            for nrn in self.bool_dfs[int(vol)].index.values:
                if nrn not in nrnDict.keys():
                    nrnDict[nrn] = {}

                for stim in monocStims:
                    val = np.nanmedian(
                        self.stimdicts[vol]["meanArr"][stim][nrn][
                            -used_offsets[0] : -used_offsets[0] + stim_offset
                        ]
                    )
                    nrnDict[nrn][stim] = val

            thetas = []
            thetavals = []
            for n in nrnDict.keys():
                degKeys = [angles.deg_dict[i] for i in nrnDict[n].keys()]
                degs = np.clip(list(nrnDict[n].values()), a_min=0, a_max=56)

                theta = angles.weighted_mean_angle(degKeys, degs)
                thetaval = np.nanmean(degs)

                thetas.append(theta)
                thetavals.append(thetaval)

            plotty = plt.subplot(3, 5, pltdict[g], polar=True)
            [
                plotty.plot(
                    [0, angles.radians(t)],
                    [0, tval],
                    linestyle="-",
                    c=angles.color_returner(tval, t, 0.15),
                    alpha=0.7,
                )
                for t, tval in zip(thetas, thetavals)
            ]
            plotty.set_theta_zero_location("N")
            plotty.set_theta_direction(-1)
            plotty.set_ylim(0, 1.15)

        fig.tight_layout()
        plt.show()

    @staticmethod
    def arrangedArrays(series, offsets=(-10, 10)):
        series = series.values
        a = []
        for repeat in range(len(series)):
            s = series[repeat] + offsets[0]
            e = series[repeat] + offsets[1]
            a.append(np.arange(s, e))
        return np.array(a)

    @staticmethod
    def normcell(arr):
        diffs = np.diff(arr)
        zscores = zscore(diffs)
        prettyz = pretty(zscores)
        return prettyz

    @staticmethod
    def legacy_raw_text_frametimes_to_df(time_path):
        """
        Parameters
        ----------
        time_path : TYPE path
            DESCRIPTION. path to the frame times (txt) collected by the imaging software

        Returns
        -------
        TYPE dataframe
            DESCRIPTION. raw data frame times will be listed in datetime format
        """
        with open(time_path) as file:
            contents = file.read()
        parsed = contents.split("\n")

        times = []
        for line in range(len(parsed) - 1):
            times.append(dt.strptime(parsed[line], "%H:%M:%S.%f").time())
        return pd.DataFrame(times)

    @staticmethod
    def legacy_raw_text_logfile_to_df(log_path):
        """
        Parameters
        ----------
        log_path : TYPE path
            DESCRIPTION. path to the log txt file from imaging software, contains steps

        Returns
        -------
        log_steps : TYPE dataframe
            DESCRIPTION. raw data log txt is filtered, only have the times and steps when the piezo moved
        """
        with open(log_path) as file:
            contents = file.read()
        split = contents.split("\n")

        movesteps = []
        times = []
        for line in range(len(split)):
            if (
                "piezo" in split[line]
                and "connected" not in split[line]
                and "stopped" not in split[line]
            ):
                t = split[line].split(" ")[0][:-1]
                z = split[line].split(" ")[6]
                try:
                    if isinstance(eval(z), float):
                        times.append(dt.strptime(t, "%H:%M:%S.%f").time())
                        movesteps.append(z)
                except NameError:
                    continue
        else:
            # last line is blank and likes to error out
            pass
        log_steps = pd.DataFrame({"times": times, "steps": movesteps})
        return log_steps

    @staticmethod
    def legacy_alignmentFramesSteps(
        frametimes, logtimes, intermediate_return=False, time_offset=0.1
    ):
        """
        Parameters
        ----------
        frametimes : TYPE dataframe
            DESCRIPTION. times that frames were taken, converted from the raw_text_frametimes_to_df function
        logtimes : TYPE dataframe
            DESCRIPTION. times and steps (um) when the piezo moved through the image collection, converted in the raw_text_logfile_to_df function

        Returns
        -------
        frametimes : TYPE dataframe, modified from the raw_text_logfile_to_df frametimes
            DESCRIPTION. contains the aligned steps from the log file to the times that the frames were collected
        """

        ## milliseconds off between the log/step information and frametimes time stamp
        logtimes_mod = []  ## modified logtimes list
        missed_steps = []

        for t in range(len(frametimes)):
            listed_time = str(frametimes.values[t][0]).split(":")
            time_val = float(listed_time[-1])

            seconds_min = time_val - time_offset
            seconds_max = time_val + time_offset
            # clip function to make sure the min is 0, no negative times
            seconds_min = np.clip(seconds_min, a_min=0, a_max=999)

            min_listed_time = listed_time.copy()
            min_listed_time[-1] = str(np.float32(seconds_min))

            max_listed_time = listed_time.copy()
            max_listed_time[-1] = str(np.float32(seconds_max))

            if seconds_max > 60:
                seconds_max -= 60
                max_listed_time[-1] = str(np.float16(seconds_max))
                new_seconds = int(max_listed_time[1]) + 1
                max_listed_time[1] = str(int(new_seconds))
            else:
                pass

            if seconds_max >= 60:
                seconds_max -= 60
                max_listed_time[-1] = str(np.float16(seconds_max))
                new_seconds = int(max_listed_time[1]) + 1
                max_listed_time[1] = str(int(new_seconds))
            else:
                pass

            mintime = dt.strptime(":".join(min_listed_time), "%H:%M:%S.%f").time()

            maxtime = dt.strptime(":".join(max_listed_time), "%H:%M:%S.%f").time()

            temp = logtimes[(logtimes.times >= mintime) & (logtimes.times <= maxtime)]

            ## sometimes there are missed steps (no frame with the next step in the stack) so we need to take that out
            if len(temp) != 0:
                logtimes_mod.append(temp)
            else:
                missed_steps.append(t)
        ## this is a check here, so if intermediate_return is true, then it will stop here and return the frametimes and logtimes_mod dataframes
        if intermediate_return:
            return frametimes, logtimes_mod

        frametimes_with_steps = []
        for df_row in logtimes_mod:
            frametimes_with_steps.append(df_row.steps.values[0])

        frametimes.drop(missed_steps, inplace=True)
        frametimes.loc[:, "step"] = frametimes_with_steps
        frametimes.loc[:, "step"] = frametimes.step.astype(np.float32)
        return frametimes

    @staticmethod
    def norm_fdff(f_cells):
        minVals = np.percentile(f_cells, 10, axis=1)
        zerod_arr = np.array(
            [np.subtract(f_cells[n], i) for n, i in enumerate(minVals)]
        )
        normed_arr = np.array([np.divide(arr, arr.max()) for arr in zerod_arr])
        return normed_arr

    @staticmethod
    def load_suite2p(suite2p_paths_dict):
        ops = np.load(suite2p_paths_dict["ops"], allow_pickle=True).item()
        iscell = np.load(suite2p_paths_dict["iscell"], allow_pickle=True)[:, 0].astype(
            bool
        )
        stats = np.load(suite2p_paths_dict["stats"], allow_pickle=True)
        f_cells = np.load(suite2p_paths_dict["f_cells"])
        return ops, iscell, stats, f_cells

    @staticmethod
    def log_aligner(logsteps, frametimes):
        trimmed_logsteps = logsteps[
            (logsteps.time >= frametimes.iloc[0].values[0])
            & (logsteps.time <= frametimes.iloc[-1].values[0])
        ]
        return trimmed_logsteps

    @staticmethod
    def raw_text_frametimes_to_df(time_path):
        with open(time_path) as file:
            contents = file.read()
        parsed = contents.split("\n")

        times = []
        for line in range(len(parsed) - 1):
            times.append(dt.strptime(parsed[line], "%H:%M:%S.%f").time())
        times_df = pd.DataFrame(times)
        times_df.rename({0: "time"}, axis=1, inplace=True)
        return times_df

    @staticmethod
    def pandastim_to_df(pstimpath):
        with open(pstimpath) as file:
            contents = file.read()

        lines = contents.split("\n")

        motionOns = [i for i in lines if "motionOn" in i.split("_&_")[-1]]
        times = [i.split("_&_")[0] for i in motionOns]
        stims = [eval(i[i.find("{") :]) for i in motionOns]
        stimulus_only = [i["stimulus"] for i in stims]

        stimulus_df = pd.DataFrame(stimulus_only)
        stimulus_df.loc[:, "datetime"] = times
        stimulus_df.datetime = pd.to_datetime(stimulus_df.datetime)
        stimulus_df.loc[:, "time"] = [
            pd.Timestamp(i).time() for i in stimulus_df.datetime.values
        ]

        mini_stim = stimulus_df[["stim_name", "time"]]
        mini_stim.stim_name = pd.Series(mini_stim.stim_name, dtype="category")
        return stimulus_df, mini_stim

    @staticmethod
    def movement_correction(img_path, keep_mmaps=False, inputParams=None):
        defaultParams = {
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
        if inputParams:
            for key, val in inputParams.items():
                defaultParams[key] = val
        try:
            c, dview, n_processes = cm.cluster.setup_cluster(
                backend="local", n_processes=12, single_thread=False
            )
            mc = cm.motion_correction.MotionCorrect(
                [img_path.as_posix()],
                dview=dview,
                max_shifts=defaultParams["max_shifts"],
                strides=defaultParams["strides"],
                overlaps=defaultParams["overlaps"],
                max_deviation_rigid=defaultParams["max_deviation_rigid"],
                shifts_opencv=defaultParams["shifts_opencv"],
                nonneg_movie=True,
                border_nan=defaultParams["border_nan"],
                is3D=False,
            )

            mc.motion_correct(save_movie=True)
            # m_rig = cm.load(mc.mmap_file)
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

            # imagePathFolder = Path(imagePath).parents[0]
            if not keep_mmaps:
                with os.scandir(img_path.parents[0]) as entries:
                    for entry in entries:
                        if entry.is_file():
                            if entry.name.endswith(".mmap"):
                                os.remove(entry)
            dview.terminate()
            cm.stop_server()
            return output

        except Exception as e:
            print(e)
            try:
                dview.terminate()
            except:
                pass
            cm.stop_server()

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

    @staticmethod
    def generate_barcodes(response_df=None, responseThreshold=0.25, bool_df=None):
        from itertools import combinations, chain

        assert (
            response_df is not None or bool_df is not None
        ), "must include bool or response df"

        # makes dataframe of neurons with their responses
        if bool_df is None:
            bool_df = pd.DataFrame(response_df >= responseThreshold)
        cols = bool_df.columns.values
        raw_groupings = [
            cols[np.where(bool_df.iloc[i] == 1)] for i in range(len(bool_df))
        ]
        groupings_df = pd.DataFrame(raw_groupings).T
        groupings_df.columns = bool_df.index

        nrows = 2 ** len(bool_df.columns)
        ncols = len(cols)
        # print(f'{nrows} possible combinations')

        # generates list of each neuron into its class
        all_combinations = list(
            chain(*[list(combinations(cols, i)) for i in range(ncols + 1)])
        )
        temp = list(groupings_df.T.values)
        new_list = [tuple(filter(None, temp[i])) for i in range(len(temp))]

        # puts neurons into total class framework
        setNeuronMappings = list(set(new_list))
        indexNeuronMappings = [setNeuronMappings.index(i) for i in new_list]

        # setmap -- neuron into class
        setmapNeuronMappings = [setNeuronMappings[i] for i in indexNeuronMappings]

        # allmap -- neurons into number of class
        allmapNeuronMappings = [all_combinations.index(i) for i in setmapNeuronMappings]

        # combine all info back into a dataframe
        a = pd.DataFrame(indexNeuronMappings)
        a.rename(columns={0: "neuron_grouping"}, inplace=True)
        a.loc[:, "set"] = setmapNeuronMappings
        a.loc[:, "fullcomb"] = allmapNeuronMappings
        a.loc[:, "neuron"] = groupings_df.columns.values

        barcode_df = a.sort_values(by="neuron")

        return barcode_df

    @staticmethod
    def generate_barcodes_fromstd(input_dfs, std_threshold=1.5):
        from itertools import combinations, chain

        response_df = input_dfs[0]
        std_df = input_dfs[1]

        # makes dataframe of neurons with their responses
        bool_df = response_df >= std_df * std_threshold
        cols = bool_df.columns.values
        raw_groupings = [
            cols[np.where(bool_df.iloc[i] == 1)] for i in range(len(bool_df))
        ]
        groupings_df = pd.DataFrame(raw_groupings).T
        groupings_df.columns = bool_df.index

        nrows = 2 ** len(response_df.columns)
        ncols = len(cols)
        # print(f'{nrows} possible combinations')

        # generates list of each neuron into its class
        all_combinations = list(
            chain(*[list(combinations(cols, i)) for i in range(ncols + 1)])
        )
        temp = list(groupings_df.T.values)
        new_list = [tuple(filter(None, temp[i])) for i in range(len(temp))]

        # puts neurons into total class framework
        setNeuronMappings = list(set(new_list))
        indexNeuronMappings = [setNeuronMappings.index(i) for i in new_list]

        # setmap -- neuron into class
        setmapNeuronMappings = [setNeuronMappings[i] for i in indexNeuronMappings]

        # allmap -- neurons into number of class
        allmapNeuronMappings = [all_combinations.index(i) for i in setmapNeuronMappings]

        # combine all info back into a dataframe
        a = pd.DataFrame(indexNeuronMappings)
        a.rename(columns={0: "neuron_grouping"}, inplace=True)
        a.loc[:, "set"] = setmapNeuronMappings
        a.loc[:, "fullcomb"] = allmapNeuronMappings
        a.loc[:, "neuron"] = groupings_df.columns.values

        barcode_df = a.sort_values(by="neuron")

        return barcode_df

    @staticmethod
    def return_all_combinations(response_df):
        from itertools import combinations, chain

        cols = response_df.columns.values
        ncols = len(cols)

        all_combinations = list(
            chain(*[list(combinations(cols, i)) for i in range(ncols + 1)])
        )

        return all_combinations
