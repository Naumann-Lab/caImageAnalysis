import pandas as pd
import numpy as np
from scipy.stats import zscore
from scipy.signal import find_peaks
import os
from datetime import datetime as dt, timedelta, date


class Gafftopsail:
    """
    However preprocessing have to be done. This includes
    - imaging data: run suite2p
    - tail_preprocessing
    - pstim_preprocessing
    - eye_preprocessing?

    Treat this merely as a data STORAGE mahcine, instaed of something that process any function since it's a lil slow
    """

    def __init__(self, path, sequence = 1, filelist = ['stimulus', 'tail', 'eye', 'imaging', 'processed_tail', 'processed_eye']):
        self.path = path
        self.filelist = filelist
        self.sequence = sequence
        if 'stimulus' in filelist: self.stimulus_df = self.load_stimulus()
        else: print('no stimulus files, zeroing with imaging frametimes')
        if 'tail' in filelist: self.tail_df = self.load_tail()
        if 'eye' in filelist: self.eye_df = self.load_eye()
        if 'imaging' in filelist: self.planes, self.f_dict, self.pos_dict, self.pos_all, self.img_dict, self.frametimes, self.frametimes_dict = self.load_imaging_data()
        if 'photostim_ensemble' in filelist: self.photostim_sites, self.photostim_time, self.photostim_duration_s = self.load_photostim_ensemble()
        if 'roi' in filelist: self.roi_dict = self.load_rois()
        if 'xml' in filelist: self.xml_info_files = self.load_xml_path()

        self.zero_timeline()
        if 'imaging' in filelist: self.image_s = self.get_framerate()
        if 'photostim_ensemble' in filelist: self.photostim_frame_dict = self.load_photostim_frames()

        try:
            os.mkdir(path + 'Graphs//')
        except FileExistsError:
            pass

        if 'processed_vis' in filelist: self.vis_df = self.load_vis_df()
        if 'processed_weightedf' in filelist: self.f_weighted = self.load_weighted_f()
        if 'processed_tail' in filelist: self.bout_df = self.load_bout_df()
        if 'processed_eye' in filelist: self.saccade_df = self.load_saccade_df()
        if 'alignment' in filelist: self.apos_dict, self.apos_all = self.load_alignment()

    def load_stimulus(self):
        # gather the stimulus file
        stimulus_df = pd.read_hdf(self.path + 'stimulus_df.hdf', dtype = object).reset_index(drop=True)
        # make stimulus file stationary friendly
        for n, row in stimulus_df.iterrows():
            if type(row['stim_name']) == list and 'stationary' in row['stim_name'][1]:  # actual dot only
                stimulus_df.loc[n, 'stim_name'] = row['stim_name'][0]
                stimulus_df.loc[n, 'angle'] = row['angle'][0]
                stimulus_df.loc[n, 'stationary_time'] = row['stationary_time'][0]
                stimulus_df.loc[n, 'velocity'] = row['velocity'][0]
                stimulus_df.at[n, 'circle_center'] = [row['circle_center'][0]]
            if type(row['stim_name']) == list and 'stationary' in row['stim_name'][0]:  # actual grating only
                stimulus_df.loc[n, 'stim_name'] = row['stim_name'][1]
                stimulus_df.loc[n, 'angle'] = row['angle'][1]
                stimulus_df.loc[n, 'stationary_time'] = row['stationary_time'][1]
                stimulus_df.loc[n, 'velocity'] = row['velocity'][1]
                stimulus_df.at[n, 'circle_center'] = [row['circle_center'][1]]
        return stimulus_df


    def load_tail(self):
        # gather all the tail data
        tailpath = self.path + 'Tail//'
        tail_df = pd.read_hdf(tailpath + 'tail_df.hdf').ffill()
        return tail_df

    def load_eye(self):
        # gather all the tail data
        eyepath = self.path + 'Eye//'
        eye_df = pd.read_hdf(eyepath + 'eye_df.hdf').ffill()
        return eye_df

    def load_imaging_data(self):
        # gather all imaging data
        imagingpath = self.path + 'Imaging/'
        planes_dir = [imagingpath + '//' + dir.name + '//'
                      for dir in os.scandir(imagingpath) if '.' not in dir.name]
        planes = [i for i in range(len(planes_dir))]

        def pretty(x, n=3):
            """
            runs a little smoothing fxn over the array from Matt^TM
            :param x: arr
            :param n: width of smooth
            :return: smoothed arr
            """
            return np.convolve(x, np.ones(n) / n, mode="same")

        #normalize f from 0 to 1
        def norm_rows(arr):
            # Z-score each row
            row_mean = arr.mean(axis=1, keepdims=True)
            row_std = arr.std(axis=1, keepdims=True)
            zscored = (arr - row_mean) / row_std
            zscored = np.nan_to_num(zscored)  # handle std=0
            #run box filter to smooth
            zscored = np.apply_along_axis(lambda row: pretty(row), 1, zscored)
            print('smoothed')
            # Normalize the z-scored rows
            row_min = zscored.min(axis=1, keepdims=True)
            row_max = zscored.max(axis=1, keepdims=True)
            normalized_arr = (zscored - row_min) / (row_max - row_min)
            normalized_arr = np.nan_to_num(normalized_arr)  # handle div by zero

            return normalized_arr


        kernel = np.ones(5) / 5
        f_dict = {plane: None for plane in range(len(planes_dir))}
        neuron_count = 0
        for plane, plane_dir in zip(range(len(planes_dir)), planes_dir):
            f = np.load(plane_dir + 'F.npy')
            #SMOOTHING
            #f = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode='same'), axis=1, arr=f)
            f_norm = norm_rows(f)
            f_dict[plane] = pd.DataFrame(data = f_norm, index = np.add(range(len(f_norm)), neuron_count))
            neuron_count += len(f_norm)
        #f_all = np.concatenate([f_dict[plane] for plane in range(len(planes_dir))], axis=0)
        #f_all = pd.DataFrame(f_all)

        # gather the reference image
        img_dict = {plane: None for plane in range(len(planes_dir))}
        for plane, plane_dir in zip(range(len(planes_dir)), planes_dir):
            ops = np.load(plane_dir + 'ops.npy', allow_pickle=True)
            img = ops.item()['meanImg']
            img_dict[plane] = np.rot90(np.flipud(img), k = -1)# make graph align to anteiror/posterior axis

        # gather cell location data @ chatgpt
        pos_dict = {plane: None for plane in range(len(planes_dir))}
        neuron_count = 0
        for plane, plane_dir in zip(range(len(planes_dir)), planes_dir):
            stat = np.load(plane_dir + 'stat.npy', allow_pickle=True)
            avg_xpos = np.array([np.mean(cell['ypix']) for cell in stat])
            avg_ypos = np.array([np.mean(cell['xpix']) for cell in stat])
            avg_zpos = np.array([plane] * len(avg_xpos))
            pos_dict[plane] = pd.DataFrame(data = np.array([avg_xpos, avg_ypos, avg_zpos]).T,
                                           columns = ['xpos', 'ypos', 'zpos'],
                                           index = np.add(range(len(avg_xpos)), neuron_count))
            neuron_count += len(avg_xpos)
        pos_all = pd.concat([pos_dict[plane] for plane in planes], ignore_index=True).reset_index(drop=True)

        # gather all frametimes data
        files = self.load_xml_path()
        frametimes = []
        for seq in range(self.sequence):
            file = open(files[seq])
            environment_data = file.read()
            # from Matt's Fishy
            first_start_time = True
            for i in environment_data.split("\n"):
                if "date" in i:#get date
                    date = i.split("date=\"")[1].split(" ")[0]
                if "time" in i and first_start_time:
                    start = [i][0].split("time=")[1].split('"')[1]
                    start_dt = dt.strptime(date + start[:-1], "%m/%d/%Y%H:%M:%S.%f")
                    first_start_time = False
                elif "absoluteTime" in i:
                    added_secs = [i.split("absoluteTime=")[1].split('"')[1]][0]
                    frame_dt = start_dt + timedelta(seconds=float(added_secs))
                    frametimes.append(frame_dt)
            file.close()
        frametimes_dict = {plane: None for plane in range(len(planes_dir))}
        for plane in range(len(planes_dir)):
           frametimes_dict[plane] = [frametimes[i] for i in range(plane, len(frametimes), len(planes))]

        return planes, f_dict, pos_dict, pos_all, img_dict, frametimes, frametimes_dict

    def load_rois(self):
        try:
            rois = [dir.name.split('.')[0] for dir in os.scandir(self.path + '//ROIs//plane_' + str(self.planes[0]) + '//')]
            roi_dict = {roi: {plane: pd.DataFrame() for plane in self.planes} for roi in rois}
            for roi in rois:
                for plane in self.planes:
                    roi_loc = pd.read_csv(self.path + 'ROIs//plane_' + str(plane) + '//' + roi + '.csv')
                    roi_dict[roi][plane] = roi_loc.iloc[:, ::-1]
        except FileNotFoundError:
            rois = [dir.name.split('.')[0] for dir in os.scandir(self.path + '//ROIs//')]
            roi_dict = {roi: {plane: pd.DataFrame() for plane in self.planes} for roi in rois}
            for roi in rois:
                for plane in self.planes:
                    roi_loc = pd.read_csv(self.path + 'ROIs//' + roi + '.csv')
                    roi_dict[roi][plane] = roi_loc.iloc[:, ::-1]
        return roi_dict

    def load_xml_path(self, type = 'info'):
        filelists = []
        for file in os.listdir(self.path + '//Imaging//'):
            if type == 'info':
                if file.endswith('.xml') and 'MarkPoints' not in file and 'Voltage' not in file:
                    filelists = filelists + [self.path + '//Imaging//' + file]
            elif type == 'markpoint':
                if file.endswith('.xml') and 'MarkPoints' in file:
                    filelists = filelists + [self.path + '//Imaging//' + file]
            elif type == 'env':
                if file.endswith('.env'):
                    filelists = filelists + [self.path + '//Imaging//' + file]
            elif type == 'voltagerecording_env':
                if file.endswith('.xml') and 'Voltage' in file:
                    filelists = filelists + [self.path + '//Imaging//' + file]
            elif type == 'voltagerecording':
                if file.endswith('.csv') and 'Voltage' in file:
                    filelists = filelists + [self.path + '//Imaging//' + file]
        return filelists

    def get_zpos(self):
        """@Kaitlyn bruker"""
        files = self.load_xml_path()
        file = open(files[0])#assuming all the z steps are the same!
        environment_data = file.read()
        file.close()
        all_z_steps = []
        all_etl_steps = []
        for j, i in enumerate(environment_data.split("\n")):
            if "page" in i:
                page = int(i.split('page=')[1].split('"')[1])
                all_z_steps.append(page)
            if 'ETL' in i:
                etl_step = float(i.split('value=')[1].split('"')[1])
                all_etl_steps.append(etl_step)
            if "Z Focus" in i:
                Z_focus_start = float(i.split('value=')[1].split('"')[1])
            if "micronsPerPixel" in i:
                next_line = environment_data.split("\n")[j + 1]
                pixel_size = float(next_line.split('value=')[1].split('"')[1])
                z_step_size = float(environment_data.split("\n")[j + 3].split('value=')[1].split('"')[1])
        num_z_steps = max(all_z_steps)
        zstep_vals = [Z_focus_start + n * z_step_size for n in range(num_z_steps)]
        return zstep_vals

    def find_closest_spatial_neuron(self, stim_x, stim_y, candidate_rois):
        x_diff = np.subtract(candidate_rois['xpos'], stim_x)
        y_diff = np.subtract(candidate_rois['ypos'], stim_y)
        dist = np.sqrt(x_diff ** 2 + y_diff ** 2)
        return candidate_rois.index[np.argmin(dist)]

    def get_photostim_time(self):
        """
        @Kaitlyn fishy
        :return:  in seconds
        """
        print('only looking at seq 1 photostim for now...')
        files = self.load_xml_path(type = 'markpoint')
        file = open(files[0])  # assuming all the stimulus is the same
        markpoint_data = file.read()
        file.close()
        #get how long each stimulation is
        info = 0
        for i in markpoint_data.split("\n"):
            if "InitialDelay" in i:
                initial_delay_ms = float([i][0].split("InitialDelay=")[1].split('"')[1])  # weird format in the xml file for this value, should not be a decimal
                interpointdelay_ms = int(float([i][0].split("InterPointDelay=")[1].split('"')[1]))
                duration_ms = float([i][0].split("Duration=")[1].split('"')[1])
                info += 1
            elif "Repetitions" in i:
                no_repetitions = int([i][0].split("Repetitions=")[1].split('"')[1])
                info += 1
            elif "Iterations" in i:
                no_iterations = int([i][0].split("Iterations=")[1].split('"')[1])
                iteration_delay_ms = int(float([i][0].split("IterationDelay=")[1].split('"')[1]))
                info += 1
            elif info >= 3:
                break#already got everything we need, no need to read further
        full_duration_per_stim = (no_repetitions * duration_ms) + ((no_repetitions - 1) * interpointdelay_ms)
        #get when did each stimulation happen from voltage signal
        self.rawframetiems = self.frametimes
        voltage_recording_starttime = self.frametimes[np.argmax(np.diff(self.frametimes)) + 2]#self.frametimes hasn't been zeroed yet
        # file = open(self.load_xml_path(type='voltagerecording_env'))
        # voltage_data = file.read()
        # file.close()
        # for i in voltage_data.split("\n"):#get the start timepoint
        #     if 'DateTime' in i:
        #         voltage_recording_starttime =i.split(">")[1].split("<")[0]#convert this into actual datetime
        #         voltage_recording_starttime = dt.fromisoformat(voltage_recording_starttime[:voltage_recording_starttime.rfind('-')])
        #         break
        volt_csv = pd.read_csv(self.load_xml_path(type = 'voltagerecording')[0], usecols=[' monaco', 'Time(ms)'])
        times = list(volt_csv['Time(ms)'])
        peaks, _ = find_peaks(volt_csv[' monaco'], height=0.10)  # find peaks in voltage trace that are above 0.10 volts
        trial_starts = [times[peaks[i]] for i in range(len(peaks)) if i == 0 or times[peaks[i]] - times[peaks[i - 1]] >
            full_duration_per_stim]# find only the start of each peak, each rep
        stim_times = [voltage_recording_starttime + timedelta(milliseconds = i) for i in trial_starts]  # convert the peak start indices to the time in ms
        return stim_times, 0.001 * full_duration_per_stim

    def load_photostim_ensemble(self):
        """@Kaitlyn fishy
             from the Markpoints xml file"""
        print('only looking at all photostim point and not differentiate by ensembles...')
        #load info xml
        files = self.load_xml_path()
        file = open(files[0])  # assuming all the z steps are the same!
        info_data = file.read()
        file.close()
        files = self.load_xml_path(type = 'markpoint')
        file = open(files[0])  # assuming all the z steps are the same!
        markpoint_data = file.read()
        file.close()
        files = self.load_xml_path(type = 'env')
        file = open(files[0])  # assuming all the z steps are the same!
        galvo_data = file.read()
        file.close()
        #get basic scaling data
        for i in info_data.split("\n"):
            if "pixelsPerLine" in i:
                pixels_per_line = int(i.split('value=')[1].split('"')[1])
            if "linesPerFrame" in i:
                lines_per_frame = int(i.split('value=')[1].split('"')[1])
        # find the x and y pos of each point
        xs = []
        ys = []
        for i in markpoint_data.split("\n"):
            if "Point Index" in i:
                ys = ys + [float(i.split("X=\"")[1].split("\"")[0]) * lines_per_frame]#each point needs to be calibrated
                xs = xs + [float(i.split("Y=\"")[1].split("\"")[0]) * lines_per_frame]
        #find the z pos of each group
        zstep_vals = self.get_zpos()
        zs = []
        for row, i in enumerate(galvo_data.split("\n")):
            if "PVCurrentMarkPointSeriesElements " in i:#get which group is actually being stimulated, note that only 1 group is applied
                stim_row = row + 2
        stim_group = galvo_data.split("\n")[stim_row].split('Points=')[1].split('"')[1]
        for i in galvo_data.split("\n"):
            if stim_group in i:
                stim_index = i.split("Indices=")[1].split('"')[1].split(",")
                break
        reading = False
        stim_index_index = 0
        for i in galvo_data.split("\n"):
            if stim_index_index == len(stim_index):
                break
            if reading and "Group" not in i and "List" not in i:
                if i.split("Index=")[1].split('"')[1] == stim_index[stim_index_index]:
                    stim_index_index += 1
                    zs = zs + [float(i.split('Z=')[1].split('"')[1])]
            if "PVMarkPoints" in i:
                reading = True
        zs = [self.planes[np.argmin(np.abs(np.subtract(i, zstep_vals)))] for i in zs]
        stim_sites_df = pd.DataFrame(data ={'xpos': xs, 'ypos': ys, 'zpos': zs})
        # get neuron n
        ns = []
        for _, stim_info in stim_sites_df.iterrows():
                neuron_index = self.find_closest_spatial_neuron(stim_info['xpos'], stim_info['ypos'], self.pos_dict[stim_info['zpos']])
                ns.append(neuron_index)
        stim_sites_df.loc[:, 'neuron_index'] = ns
        #load stim frames
        photostim_fromstart_time, photostim_duration_s = self.get_photostim_time()
        return stim_sites_df, photostim_fromstart_time, photostim_duration_s

    def get_framerate(self):
        image_s = np.array([i for i in np.diff(self.frametimes_dict[0]) if i < 5]).mean()
        return image_s

    def zero_timeline(self):
        """
        Get a consistent timeline in seconds for all the timeseries involved
        default Baseline: a column of REAL time to zero upon, default stimulus_df
        :return:
        """
        try:
            baseline = self.stimulus_df['real_starttime']
        except:
            baseline = self.frametimes
        t_start = min(baseline)
        if 'stimulus' in self.filelist:
            self.stimulus_df.loc[:, 'real_starttime_s'] = [(i.to_pydatetime() - t_start).total_seconds() for i in
                                                       self.stimulus_df['real_starttime']]  # stimulus
        if 'tail' in self.filelist:
            self.tail_df.loc[:, 'real_time_s'] = [(i.to_pydatetime() - t_start).total_seconds() for i in
                                              self.tail_df['real_time']]
            non_nan_stimrow = self.stimulus_df.index[self.stimulus_df['real_starttime_s'].notna()][-1]
            self.tail_df = self.tail_df[self.tail_df['real_time_s']<=self.stimulus_df['real_starttime_s'].iloc[non_nan_stimrow] + self.stimulus_df['duration'].iloc[non_nan_stimrow]]# chop it so only tail during stimulus is counted
        if 'eye' in self.filelist:
            self.eye_df.loc[:, 'real_time_s'] = [(i.to_pydatetime() - t_start).total_seconds() for i in
                                                  self.eye_df['real_time']]
            non_nan_stimrow = self.stimulus_df.index[self.stimulus_df['real_starttime_s'].notna()][-1]
            self.eye_df = self.eye_df[self.eye_df['real_time_s']<=self.stimulus_df['real_starttime_s'].iloc[non_nan_stimrow] + self.stimulus_df['duration'].iloc[non_nan_stimrow]]# chop it so only tail during stimulus is counted
        if 'imaging' in self.filelist:
            self.frametimes = [(i - t_start).total_seconds() for i in self.frametimes]  # imagin
            self.frametimes_dict = {plane: [(i - t_start).total_seconds() for i in self.frametimes_dict[plane]]
                                   for plane in self.frametimes_dict.keys()}
        if 'photostim_ensemble' in self.filelist:
            self.photostim_time = [(i - t_start).total_seconds() for i in self.photostim_time]

    def load_photostim_frames(self):
        """
        Return the index of in the frametime that is within the photostimulating timeframe on each plane. If no frame
        falls into this criteria, return the nearest frame
        """
        photostim_frame_dict = {plane: {stim_event: None for stim_event in range(len(self.photostim_time))} for plane in
                                self.planes}
        for plane in self.planes:
            photostim_frame_plane = []
            for i, t in enumerate(self.photostim_time):
                photostim_frame_dict[plane][i] = list(np.argwhere((np.array(self.frametimes_dict[plane]) >= t) & (
                            np.array(self.frametimes_dict[plane]) <= np.add(t, self.photostim_duration_s))).flatten())
                if len(photostim_frame_dict[plane][i]) == 0:  # not overlapping with any of the photostim frame, then we find the closest frame
                    photostim_frame_dict[plane][i] = list(np.argmin(np.abs(np.subtract(self.frametimes_dict[plane], t))).flatten())
        return photostim_frame_dict

    def load_vis_df(self):
        """Load the visual responsive properties of all cells (see 0_omr_functionalstack_pub) for ways to generate it,
        this is so far specific to Matt's photostim project"""
        vispath = self.path + 'Graphs//omr_pub_info.csv'
        vis_df = pd.read_csv(vispath, index_col = 0)
        return vis_df

    def load_weighted_f(self):
        """Load the response per cell according to the photostim weighted (see 1_photostim_weights) for ways to generate it,
        this is so far specific to Matt's photostim project'"""
        f_weighted_path = self.path + 'Graphs//f_weighted.csv'
        f_weighted = pd.read_csv(f_weighted_path, index_col=0)
        return f_weighted

    def load_bout_df(self):
        # gather all the tail data
        import re
        tailpath = self.path + 'Tail//'
        bout_df = pd.read_csv(tailpath + 'tail_bout_df.csv', index_col = 0, converters={
        "cont_tuples_tailindex": lambda x: (int(x.split('(')[2].split(')')[0]), int(x.split('(')[3].split(')')[0])),
        "cont_tuples_realtime_s":lambda x: (float(x.split('(')[2].split(')')[0]), float(x.split('(')[3].split(')')[0])),
        "tail_angle": float})
        return bout_df

    def load_saccade_df(self):
        # gather all the tail data
        eyepath = self.path + 'Eye//'
        saccade_df = pd.read_csv(eyepath + 'eye_saccade_df.csv', index_col = 0,
                                 converters={
                                     "cont_tuples_eyeindex": lambda x: (int(x.split('(')[2].split(')')[0]), int(x.split('(')[3].split(')')[0])),
                                     "cont_tuples_realtime_s": lambda x: (float(x.split('(')[2].split(')')[0]), float(x.split('(')[3].split(')')[0])),
                                 }
                                 )
        return saccade_df

    def load_alignment(self):
        """
        Load the aligned results where the neurons are mapped to the location of the atlas
        :return:
        """
        # gather all imaging data
        import ast
        imagingpath = self.path + 'Imaging/'
        planes_dir = [imagingpath + '//' + dir.name + '//Alignment//'
                      for dir in os.scandir(imagingpath) if '.' not in dir.name]
        planes = [i for i in range(len(planes_dir))]

        # gather cell location data @ chatgpt
        apos_dict = {plane: None for plane in range(len(planes_dir))}
        neuron_count = 0
        for plane, plane_dir in zip(range(len(planes_dir)), planes_dir):
            apos_dict[plane] = pd.read_csv(plane_dir + 'new_pos.csv', index_col = 0, converters={"regions": ast.literal_eval})
            apos_dict[plane].index = np.add(range(len(apos_dict[plane])), neuron_count)
            neuron_count += len(apos_dict[plane])
        apos_all = pd.concat([apos_dict[plane] for plane in planes], ignore_index=True).reset_index(drop=True)
        return apos_dict, apos_all

    def load_atlas(self):
        """
        Grab the atlas
        :return:
        """
        #read params to find the atlas
        #return atlas
        pass

class BuffaloBream:
    """
    This is the Matt's paper dataformat where Kaitly and I have preprocessed

    Treat this merely as a data STORAGE mahcine, instaed of something that process any function since it's a lil slow
    """

    def __init__(self, path, filelist = ['stimulus', 'imaging', 'photostim_ensemble'],
                 experimenter = 'zh'):
        self.path = path
        self.filelist = filelist
        self.experimenter = experimenter
        if 'imaging' in filelist: self.planes, self.f_dict, self.f_all, self.pos_dict, self.pos_all, self.img_dict, self.frametimes, self.frametimes_dict = self.load_imaging_data()
        if 'stimulus' in filelist: self.stimulus_df = self.load_stimulus()
        if 'photostim_ensemble' in filelist: self.photostim_sites, self.photostim_frame_dict = self.load_photostim_ensemble()
        if 'roi' in filelist: self.roi_dict = self.load_rois()
        self.zero_timeline()
        if 'imaging' in filelist: self.image_s = self.get_framerate()

        if 'processed_vis' in filelist: self.vis_df = self.load_vis_df()
        if 'processed_weightedf' in filelist: self.f_weighted = self.load_weighted_f()

        try:
            os.mkdir(path + 'Graphs//')
        except FileExistsError:
            pass

    def load_stimulus(self):
        imagingpath = self.path + 'output_folders/'
        firstplane_dir = [imagingpath + '//' + dir.name + '//'
                      for dir in os.scandir(imagingpath) if 'plane' in dir.name][0]
        # gather the stimulus file: it's the same in all plane directory
        stimulus_df = pd.read_csv(firstplane_dir + 'stimulus_df.csv', dtype = object).reset_index(drop=True)
        if self.experimenter == 'ZH':
            stimulus_df.loc[:, "time"] = [dt.combine(date.today(), dt.strptime(i, "%Y-%m-%d %H:%M:%S.%f").time()) for i in stimulus_df.time]#5/22/2025  3:43:27 PM
        elif self.experimenter == 'KF':
            stimulus_df.loc[:, "time"] = [dt.combine(date.today(), dt.strptime(i, "%H:%M:%S.%f").time()) for i in stimulus_df.time]
        # keep og stimulus df cuz no overlapping/stationary stimulus
        return stimulus_df

    def load_imaging_data(self):
        # gather all imaging data
        imagingpath = self.path + 'output_folders/'
        planes_dir = [imagingpath + '//' + dir.name + '//'
                      for dir in os.scandir(imagingpath) if 'plane' in dir.name]
        planes = [i for i in range(len(planes_dir))]

        def pretty(x, n=3):
            """
            runs a little smoothing fxn over the array from Matt^TM
            :param x: arr
            :param n: width of smooth
            :return: smoothed arr
            """
            return np.convolve(x, np.ones(n) / n, mode="same")

        # normalize f from 0 to 1
        def norm_rows(arr):
            # Z-score each row
            row_mean = arr.mean(axis=1, keepdims=True)
            row_std = arr.std(axis=1, keepdims=True)
            zscored = (arr - row_mean) / row_std
            zscored = np.nan_to_num(zscored)  # handle std=0
            # run box filter to smooth
            zscored = np.apply_along_axis(lambda row: pretty(row), 1, zscored)
            # Normalize the z-scored rows
            row_min = zscored.min(axis=1, keepdims=True)
            row_max = zscored.max(axis=1, keepdims=True)
            normalized_arr = (zscored - row_min) / (row_max - row_min)
            normalized_arr = np.nan_to_num(normalized_arr)  # handle div by zero

            return normalized_arr

        kernel = np.ones(3) / 3
        f_dict = {plane: None for plane in range(len(planes_dir))}
        neuron_count = 0
        for plane, plane_dir in zip(range(len(planes_dir)), planes_dir):
            f = np.load(plane_dir + 'suite2p//plane0//F.npy')
            f_z = zscore(f, axis=1)
            #SMOOTHING
            #f = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode='same'), axis=1, arr=f)
            f_norm = norm_rows(f_z)
            f_dict[plane] = pd.DataFrame(data = f_norm, index = np.add(range(len(f_norm)), neuron_count))
            neuron_count += len(f_norm)
        f_all = np.concatenate([f_dict[plane] for plane in range(len(planes_dir))], axis=0)
        f_all = pd.DataFrame(f_all)

        # gather the reference image
        img_dict = {plane: None for plane in range(len(planes_dir))}
        for plane, plane_dir in zip(range(len(planes_dir)), planes_dir):
            ops = np.load(plane_dir + 'suite2p//plane0//ops.npy', allow_pickle=True)
            img = ops.item()['meanImg']
            img_dict[plane] = np.rot90(np.flipud(img), k = 0)# make graph align to anteiror/posterior axis

        # gather cell location data @ chatgpt
        pos_dict = {plane: None for plane in range(len(planes_dir))}
        neuron_count = 0
        for plane, plane_dir in zip(range(len(planes_dir)), planes_dir):
            stat = np.load(plane_dir + 'suite2p//plane0//stat.npy', allow_pickle=True)
            avg_xpos = np.array([np.mean(cell['xpix']) for cell in stat])
            avg_ypos = np.array([img_dict[0].shape[1] - np.mean(cell['ypix']) for cell in stat])
            avg_zpos = np.array([plane] * len(avg_xpos))
            pos_dict[plane] = pd.DataFrame(data = np.array([avg_xpos, avg_ypos, avg_zpos]).T,
                                           columns = ['xpos', 'ypos', 'zpos'],
                                           index = np.add(range(len(avg_xpos)), neuron_count))
            neuron_count += len(avg_xpos)
        pos_all = pd.concat([pos_dict[plane] for plane in planes], ignore_index=True).reset_index(drop=True)

        # gather all frametimes data
        # from Matt's Fishy
        frametimes = []
        first_start_time = True
        if self.experimenter == 'ZH':
            for plane, plane_dir in zip(range(len(planes_dir)), planes_dir):
                frametimes = list(pd.read_hdf(plane_dir + 'frametimes.h5', allow_pickle=True).time)
        elif self.experimenter == 'KF':
            for plane, plane_dir in zip(range(len(planes_dir)), planes_dir):
                frametime = pd.read_hdf(plane_dir + 'frametimes.h5', allow_pickle=True).time
                frametimes = frametimes + list(frametime)
            frametimes = [dt.combine(date.today(), t) for t in frametimes]
            frametimes.sort()
        if frametimes[0] != 0.0 and self.experimenter == 'ZH':
            frametimes = [dt.combine(date.today(), dt.strptime(str(t), "%H:%M:%S.%f").time()) for t in frametimes]
        frametimes_dict = {plane: None for plane in range(len(planes_dir))}
        for plane in range(len(planes_dir)):
           frametimes_dict[plane] = [frametimes[i] for i in range(plane, len(frametimes), len(planes))]
        return planes, f_dict, f_all, pos_dict, pos_all, img_dict, frametimes, frametimes_dict

    def get_framerate(self):
        image_s = np.array([i for i in np.diff(self.frametimes_dict[0]) if i < 5]).mean()
        return image_s

    def load_rois(self):
        rois = [dir.name.split('.')[0] for dir in os.scandir(self.path + 'output_folders//plane_' + str(self.planes[0]) + '//rois//')]
        roi_dict = {roi: {plane: pd.DataFrame() for plane in self.planes} for roi in rois}
        for roi in rois:
            for plane in self.planes:
                roi_loc = np.load(self.path + 'output_folders//plane_' + str(self.planes[0]) + '//rois//' + roi + '.npy', allow_pickle = True)
                if self.experimenter == 'ZH':
                    roi_dict[roi][plane] = roi_loc#roi_loc.iloc[:, ::-1]
                else:
                    roi_dict[roi][plane] = np.array([[y, self.img_dict[0].shape[0] - x] for x, y in roi_loc])#[:, ::-1]
        return roi_dict

    def get_zpos(self):
        """@Kaitlyn bruker"""
        file = open(self.load_xml_path())
        environment_data = file.read()
        file.close()
        all_z_steps = []
        all_etl_steps = []
        for j, i in enumerate(environment_data.split("\n")):
            if "page" in i:
                page = int(i.split('page=')[1].split('"')[1])
                all_z_steps.append(page)
            if 'ETL' in i:
                etl_step = float(i.split('value=')[1].split('"')[1])
                all_etl_steps.append(etl_step)
            if "Z Focus" in i:
                Z_focus_start = float(i.split('value=')[1].split('"')[1])
            if "micronsPerPixel" in i:
                next_line = environment_data.split("\n")[j + 1]
                pixel_size = float(next_line.split('value=')[1].split('"')[1])
                z_step_size = float(environment_data.split("\n")[j + 3].split('value=')[1].split('"')[1])
        num_z_steps = max(all_z_steps)
        zstep_vals = [Z_focus_start + n * z_step_size for n in range(num_z_steps)]
        return zstep_vals

    def find_closest_spatial_neuron(self, stim_x, stim_y, candidate_rois):
        x_diff = np.subtract(candidate_rois['xpos'], stim_x)
        y_diff = np.subtract(candidate_rois['ypos'], stim_y)
        dist = np.sqrt(x_diff ** 2 + y_diff ** 2)
        return candidate_rois.index[np.argmin(dist)]

    def load_photostim_ensemble(self):
        """@Kaitlyn fishy
             from the Markpoints xml file"""
        print('only looking at all photostim point and not differentiate by ensembles...')
        #load info xml
        if self.experimenter == 'ZH':
            stim_sites_df = pd.read_hdf(self.path + 'output_folders//plane_' + str(self.planes[0]) + '//stim_sites.h5')
        elif self.experimenter == 'KF':
            stim_sites = []
            for plane in self.planes:
                stim_sites.append(pd.read_hdf(self.path + 'output_folders//plane_' + str(plane) + '//stim_sites.h5'))
            stim_sites_df = pd.concat(stim_sites, ignore_index= True).reset_index(drop = True)
            ns = []
            for i in range(len(stim_sites_df)):
                n = self.find_closest_spatial_neuron(stim_sites_df.iloc[i]['x_stim'],
                                                     self.img_dict[0].shape[1] - stim_sites_df.iloc[i]['y_stim'],
                                                     self.pos_dict[stim_sites_df.iloc[i]['plane']])
                ns.append(n)
            stim_sites_df.loc[:, 'neuron_index'] = ns
        #load stim frames
        def group_consecutive(lst):
            result = []
            temp = [lst[0]]
            for x in lst[1:]:
                if x == temp[-1] + 1:  # consecutive → keep grouping
                    temp.append(x)
                else:  # break → push temp to result
                    result.append(temp)
                    temp = [x]
            # don't forget the last group
            result.append(temp)
            return result
        bad_frames_dict = {plane: group_consecutive(list(np.load(self.path + "output_folders/plane_" + str(plane) + "/bad_frames.npy", allow_pickle=True))) for plane in self.planes}
        return stim_sites_df, bad_frames_dict

    def load_vis_df(self):
        """Load the visual responsive properties of all cells (see 0_omr_functionalstack_pub) for ways to generate it,
        this is so far specific to Matt's photostim project"""
        vispath = self.path + 'Graphs//omr_pub_info.csv'
        vis_df = pd.read_csv(vispath, index_col = 0)
        return vis_df

    def load_weighted_f(self):
        """Load the response per cell according to the photostim weighted (see 1_photostim_weights) for ways to generate it,
        this is so far specific to Matt's photostim project'"""
        f_weighted_path = self.path + 'Graphs//f_weighted.csv'
        f_weighted = pd.read_csv(f_weighted_path, index_col=0)
        return f_weighted

    def zero_timeline(self):
        """
        Get a consistent timeline in seconds for all the timeseries involved
        default Baseline: a column of REAL time to zero upon, default stimulus_df
        :return:
        """
        try:
            baseline = self.stimulus_df['time']
        except:
            baseline = self.frametimes
        t_start = min(baseline)
        if 'stimulus' in self.filelist:
            self.stimulus_df.loc[:, 'time'] = [(i - t_start).total_seconds() for i in
                                                       self.stimulus_df['time']]  # stimulus
        if 'imaging' in self.filelist:
            if type(self.frametimes[0]) == float:
                self.frametimes = [(i - t_start) for i in self.frametimes]  # imagin
                self.frametimes_dict = {plane: [(i - t_start) for i in self.frametimes_dict[plane]]
                                        for plane in self.frametimes_dict.keys()}
            else:
                self.frametimes = [(i - t_start).total_seconds() for i in self.frametimes]  # imagin
                self.frametimes_dict = {plane: [(i - t_start).total_seconds() for i in self.frametimes_dict[plane]]
                                       for plane in self.frametimes_dict.keys()}

class Coelacanth:
    """
    This is a class for fish that only has a behavior component to it, so they are kinda pre-historical
    However preprocessing have to be done. This includes
    - taileye_preprocessing
    - pstim_preprocessing
    - eye_preprocessing?

    Treat this merely as a data STORAGE mahcine, instaed of something that process any function since it's a lil slow
    """

    def __init__(self, path, sequence = 1, filelist = ['stimulus', 'tail', 'eye', 'imaging', 'processed_tail', 'processed_eye']):
        self.path = path
        self.filelist = filelist
        self.sequence = sequence
        if 'stimulus' in filelist: self.stimulus_df = self.load_stimulus()
        else: print('no stimulus files, zeroing with imaging frametimes')
        if 'taileye' in filelist: self.taileye_df = self.load_taileye()

        self.zero_timeline()

        try:
            os.mkdir(path + 'Graphs//')
        except FileExistsError:
            pass

        if 'processed_tail' in filelist: self.bout_df = self.load_bout_df()
        if 'processed_eye' in filelist: self.saccade_df = self.load_saccade_df()

    def load_stimulus(self):
        # gather the stimulus file
        stimulus_df = pd.read_hdf(self.path + 'stimulus_df.hdf', dtype=object).reset_index(drop=True)
        # make stimulus file stationary friendly
        for n, row in stimulus_df.iterrows():
            if type(row['stim_name']) == list and 'stationary' in row['stim_name'][1]:  # actual dot only
                stimulus_df.loc[n, 'stim_name'] = row['stim_name'][0]
                stimulus_df.loc[n, 'angle'] = row['angle'][0]
                stimulus_df.loc[n, 'stationary_time'] = row['stationary_time'][0]
                stimulus_df.loc[n, 'velocity'] = row['velocity'][0]
                stimulus_df.at[n, 'circle_center'] = [row['circle_center'][0]]
            if type(row['stim_name']) == list and 'stationary' in row['stim_name'][0]:  # actual grating only
                stimulus_df.loc[n, 'stim_name'] = row['stim_name'][1]
                stimulus_df.loc[n, 'angle'] = row['angle'][1]
                stimulus_df.loc[n, 'stationary_time'] = row['stationary_time'][1]
                stimulus_df.loc[n, 'velocity'] = row['velocity'][1]
                stimulus_df.at[n, 'circle_center'] = [row['circle_center'][1]]
        return stimulus_df


    def load_taileye(self):
        # gather all the tail and eye data
        taileye_df = pd.read_hdf(self.path + 'taileye_df.hdf').ffill()
        return taileye_df

    def zero_timeline(self):
        """
        Get a consistent timeline in seconds for all the timeseries involved
        default Baseline: a column of REAL time to zero upon, default stimulus_df
        :return:
        """
        baseline = self.stimulus_df['real_starttime']
        t_start = min(baseline)
        if 'stimulus' in self.filelist:
            self.stimulus_df.loc[:, 'real_starttime_s'] = [(i.to_pydatetime() - t_start).total_seconds() for i in
                                                       self.stimulus_df['real_starttime']]  # stimulus
        if 'taileye' in self.filelist:
            self.taileye_df.loc[:, 'real_time_s'] = [(i.to_pydatetime() - t_start).total_seconds() for i in
                                              self.taileye_df['real_time']]
            self.taileye_df = self.taileye_df[self.taileye_df['real_time_s']<=self.stimulus_df['real_starttime_s'].iloc[-1] + self.stimulus_df['duration'].iloc[-1]]# chop it so only tail during stimulus is counted



    def load_bout_df(self):
        # gather all the tail data
        tailpath = self.path + 'Tail//'
        bout_df = pd.read_csv(tailpath + 'tail_bout_df.hdf')
        return bout_df

    def load_saccade_df(self):
        # gather all the tail data
        saccade_df = pd.read_csv(self.path + 'eye_saccade_df.csv')
        return saccade_df