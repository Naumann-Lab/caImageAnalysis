# caImageAnalysis
<img align = "right" width = "240" src="images/mascot.png ">

### Introduction

Calcium imaging functional analysis

The default class is a BaseFish from fishy.py. We offer a process.py for various analyses. The standard implementation runs motion correction via caiman and then source extraction via suite2p, and most of the more advanced classes assume a suite2p folder with contained sources.

We leverage a folder structure that contains all experiment items


<br>

### Installation
This assumes you are using Anaconda and Python 3:

1. Install [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge)


2. Create a new environment and install [Caiman](https://caiman.readthedocs.io/en/master/Installation.html#installing-caiman). Note: helpful information on caiman can be found [here](https://github.com/EricThomson/CCN_caiman_mesmerize_workshop_2023)
    
    `mamba create -n caiman -c conda-forge caiman`


3. Activate the envs (note: python should be 3.7x)

   `mamba activate caiman`


4. Install [suite2p](https://github.com/MouseLand/suite2p)

   ` python -m pip install suite2p`


5. Edit suite2p documentation to fit with scipy and numpy versions compatible with caiman

    \Lib\site-packages\suite2p\detection\sparsedetect.py 

   Remove the 'keepdims' arg from line 256


6. Install more packages

   ` python -m pip install tables nptdms pyarrow`
   ` python -m pip install opencv-python`


7. Optional: Download [mimic_alpha](https://github.com/montefra/mimic_alpha) into your new caiman envs - this package converts a list of RGB color that mimic a RGBA on a given background. This is useful for plotting.

8. If using alignment code, download scopeslip from NaumannLab github and install SimpleITK
   go to a directory of your choice, then 'git clone https://github.com/Naumann-Lab/scopeslip'
   pip install SimpleITK
   pip install SimpleITK-SimpleElastix --user
   * if using mamba, I had problems with the correct directory

<br>

### Documentation of fishy.py – main file

- BaseFish = makes a class with folder structure, frametimes, image data, run suite2p 
   - self.folder_path = folder path object (Path()), indiciating where to look for the rest of the folders
   - self.data_path = dictionary with paths in the data folder
   -	self.frametimes_key = key word in the frametimes file (default: “frametimes”)
   -	self.invert = inversion of image and stimuli types (specifically for custom microscope, default: True)
   -	self.bruker_invert = inversion of stimuli types (specifically for bruker microscope, default: bruker_invert)
   -	self.frametimes_df = dataframe, containing frame (index column) and raw time (column 0)
   -	self.xpts = N/A
   -	self.ypts = N/A
   -	self.ops = output from suite2p, options and intermediate outputs (dictionary)
   -	self.iscell = output from suite2p, specifies whether an ROI is a cell, first column is 0/1, and second column is probability that the ROI is a cell based on the default classifier
   -	self.stats = output from suite2p, list of statistics computed for each cell
   -	self.f_cells  = output from suite2p, array of fluorescence traces (ROIs by timepoints)
   -	self.rescaled_ref = ndarray, containing the RGB values in their corresponding location (location in ndarray) for the rescaled refimg output from suite2p (for better visualization)
   -	self.process_filestructure() = load all files according to self.folder_path   
   -	self.raw_text_frametimes_to_df() = analyze the raw text frametimes_df files to dataframe format as self.frametimes_df
   -	self.load_suite2p() = load the data for the output from suite2p folder, including self.ops, self.iscell, self.stats, and self.f_cells 
   -	self.load_image() = load in the reference image of the stack
   -	self.return_cell_rois(cells) = return the x and y pos of a series of cells
      -	cells: series, containing the index of cells
   -	self.return_singlecell_rois(single_cell) = return the x and y pos of a cell
      -	single_cell: int, containing the index of one cell
   -	self.return_cells_by_location(xmin, xmax, ymin, ymax) = return the series of cells that are in the region of interest
      -	xmin, xmax, ymin, ymax: int, limiting the region of the box to look for (default: 0, 99999, 0, 99999)
   -	self.draw_roi(title, overwrite) = uses cv2 to draw ROIs onto the image for saving, will be made into a 'rois' folder
      -	title = name of the roi that you want to make
   -	self.save_roi(save_name, overwrite) = used in draw_roi to save the ROI file
   -	self.load_saved_rois() = load in the saved ROIs
   -	self.return_cells_by_saved_roi(roi_name) = returns the cells that have x, y positions within the ROI
      -	roi_name = ROI that you want cells from
   -	self.roi_dict = dictionary, containing all the ROI names (keys) and their correspoinding anchor points in dataframes (values), in which the x and y positions (column “xpos” and “ypos”) of anchor points are listed
   -	self.rescaled_img() = generate self.rescaled_ref
   -	hzReturner(frametimes) = returns imaging speed of the data
      -	frametimes = frametimes dataframe

-	VisStimFish(BaseFish) – makes a class with all of Basefish attributes, added stimulus set
   -	self.stim_fxn_args = dictionary for how the stimuli txt file should be processed (defauly empty)
   -	self.r_type = str, the statistical measure you are using to identify a ‘response’ ( “median”, “mean”, “peak”) (default median)
   -	self.stim_offset = int, the number of frames from stim ON (t = 0) to peak of a stim responsive trace
   -	self.used_offsets = tuple of ints, number of frames before and after stim ON (t = 0)
   -	self.baseline_offset = int, the number of frames from stim ON (t = 0) to trace back to gather baseline
   -	self.stimulus_df = dataframe, containing all relevant stimulus name (column 0) and their onset frames (column 1), noted that the stimulus happening outside of self.frametimes_df is excluded
   -	self.unchopped_df_stimulus = dataframe, similar to the structure of self.stimulus_df but the stimulus happened outside of the self.frametimes_df is not excluded
   -	self.diff_image = ndarray, RGB value based image array to color cells according to their directional selectivity across all stimuli present in all trials
   -	self.add_stims(stim_key, stim_fxn, legacy) = use the stimuli.txt file to extract stimuli type, stimuli information, and stimuli onset system time, call self.tag_frames() and creating self.df_stimulus
      -	stim_key: (default “stim”)
      -	stim_fxn: function to process txt file into a dataframe (default stimuli.pandastim_to_df)
      -	legacy: if this is from legacy Fishy script (default False)
   -	self.tag_frames() = use the system time alignment to identify the frame corresponding to stimulus onset, resulting frame stored in the column of “frames” in self.df_stimulus 
   -	self.make_difference_image(selectivityFactor, brightnessFactor) = calculate the average fluorescence change after stimuli presentation and combine all fluorescence change via color-coded stimuli type, result stored in self.diff_image
      -	selectivityFactor: float (default 1.5)
      -	brightnessFactor: float (default 10)

-	TailTrackedFish(VizStimFish) – makes a class with all of visfish, added tail data, aligned with stimuli, finding bouts and neurons responsive to those bouts, heatmap of number of bouts per stimulus presentation
   -	self.bout_finder() = filters tail deflection sum data, identifies peaks (see scipy.find_peaks() )

-	WorkingFish(VizStimFish) – makes an active working class, which can hold any of the previous classes, finding neurons responsive to certain stimuli and assembling all data
   -	self.zdiff_cells = ndarray, containing the identified cells and their fluorescence change across each frame as compared to last frame. Notice the difference in fluorescence is z-scored and smoothed within each cell.
   -	self.normf_cells = dataframe, the suite2p output containing the identified cells and their fluorescence value normalized to 0-1 within each cell. 
   -	self.zdiff_stim_dict, self.normf_stim_dict, self.f_stim_dict = dictionary, containing all stimuli names (1st key), and all cell index (2nd key) and their average zdiff/normalized/raw fluorescence change (value, list of floats) across multiple presentation of this stimuli 
   -	self.zdiff_err_dict, self.normf_err_dict, self.f_err_dict = dictionary, containing all stimuli names (1st key), and all cell index (2nd key) and their standard deviation zdiff/normalized/raw fluorescence change (value, list of floats) across multiple presentation of this stimuli 
   -	self.zdiff_neuron_dict, self.normf_neuron_dict, self.f_neuron_dict = dictionary, containing each neuron index (1st key), and their encounter to eah stimuli (2nd key). Within each containing the corresponding zdiff/normalized/raw fluorescence change across multiple presentation of the stimuli (value, list of floats).
   -	self.extended_responses = a dictionary containing each stimuli, and their corresponding fluorescence change response, data is smoothed
   -	self.extended_responses_zdiff, self.extended_responses_normf = dictionary, containing each stimuli (1st key), each cell index (2nd key), and values in dataframe, where it contains each presentation of the stimuli (value, index) and their corresponding zdiff/normalized fluorescence trace in each frame (value, columns), data is not smoothed
   -	self.zdiff_corr_booldf  = a dataframe containing all cells (index) and their correlation with an ideal cell trace responsing to each stimuli (column). The response is calculated via if the mean zdiff traces for each stimuli is correlated with an ideal trace.
   -	self.zdiff_corrdf = a dataframe containing all cells (row) and whether they are classified to be responsive to each stimuli (index). Details see self.zdiff_corr_booldf.
   -	self.normf_baseline_booldf = a dataframe containing all cells (row) and whether they are classified to be responsive to each stimuli (index). The response is determined if the mean normf cell trace is larger than the baseline traces of each cell.
   -	self.normf_cluster_booldf = a dataframe containing all cells (row) and whether they are classified to be responsive to each stimuli (index). The response is determined based on a bimodal clustering method and if during each encounter of the stimuli, the cell mean response across the offset windows are larger than the boundary of the “silent” cluster.
   -	self.build_stimdicts_extended_zdiff(),  self.build_stimdicts_extended_normf() = calculate the average fluorescence change as an array after each stimuli presentation (not a median/avg/peak single value), change in fluorescence is smoothed, results stored in self.extended_responses_zdiff or self.extended_responses_normf
   -	self.build_stimdicts(traces) = calculate the average fluorescence change after stimuli (default: median) presentation and combine across all presentations of each stimuli, results stored in self.neuron_dict
      -	traces: the type of trace that used to built the stim_dicts, can be in the format of self.f_cells, self_normf_cells, or self_zdiff_cells
      -	Finding a responsive neuron based on the expected responsive neuron peak relative to stim ON 
      -	take subset of neural traces of the self.used_offsets frame time window, 
      -	align each trace to individual stim on
      -	get all neurons to relative time to compare at only the time after stimulus presentation
   <img align = "center" width = "240" src="images/build_stimdicts_README.png ">
   -	self.build_booldf_corr() = determine if the average zdiff cell traces is correlated with an ideal trace in response to each stimuli, results stored in self.zdiff_corr_booldf and self.zdiff_corrdf. Note that the section that is commented out allowed for selection of neuron based on each presentation of the stimuli rather than the average of all presentations.
      -	corr_threshold: the correlation threshold between an ideal peak response and an unknown neuron that needs to be reached to affirm that the neuron is responsive (default: 0.65)
   -	self.build_booldf_baseline() = determine if the average normalized cell traces is above the average baseline trace for each cell. Baseline frames are determined by self.baseline_offset. Results stored in self.normf_baseline_booldf. Note that the section that is commented out allowed for selection of neuron based on each presentation of the stimuli rather than the average of all presentations.
   -	self.build_booldf_cluster() = A bimodal clustering is forced on the fluorscence magnitude of the entire cell traces. And only if the cell’s mean normalized in the responding window is larger than the boundary of the “silent” cluster, the cell is determined to be responsive. A cell to be overall responsive if it always respond to the stimuli during each presentation. Responding window is determined by self.used_offsets. Results stored in self.normf_cluster_booldf. Note that the section that is commented out allowed for selection of neuron based the average trace of all presentations to the stimuli rather than each presentation.
   -	self.make_computed_image_data(neuron_dict, booldf, colorsumthresh, booltrim) = output the neurons that are stimuli-tuned according to self.bool_df with their x, y, and stimuli-corresponding colors. 
   -	self.make_computed_image_data_ref(colorsumthresh, booltrim) = output the neurons that are stimuli-tuned according to self.bool_df with their x, y, and stimuli-corresponding colors. The x and y positions are computed for .npy images.
   -	self.make_computed_image_data_by_loc(xmin, xmax, ymin, ymax) = output the neurons that are stimuli-tuned according to self.bool_df with their x, y, and stimuli-corresponding colors that are fall into the given x and y spatial location area. 
   -	self.make_computed_image_data_by_roi(roi_name) = output the neurons that are stimuli-tuned according to self.bool_df with their x, y, and stimuli-corresponding colors that are fall into the given roi. 
   -	self.return_degree_vectors(neurons, type) = compute the average response tuning curve for the given neuron. The code right now only accounts for monocular stimuli.
      -	neurons: the series of neuron index to be analyzed
      -	type: str, the type of neuron trace to be used (“normf_baseline”, “normf_cluster”, “zdiff_corr”)

<br>

### Documentation of photostimulation.py 
Contains processing functions on photostimulation data from the Bruker, includes preprocessing and utilizes the BaseFish from fishy.py
- find_no_baseline_frames(somefishclass, no_planes) = finds baseline frames before photostimulation trials begin in the dataset
   -	somefishclass = Basefish instance of the data to use the folder paths
   -	no_planes = number of planes that are in the dataset, important for single or volume (default: 0)
-	collect_stimulation_times(somefishclass) = Calculating the stimulation times from either the voltage recording output (channel input 2), if not voltage recording, then can find this based on the mark point xml file (not as exact)
   -	somefishclass = Basefish instance of the data to use the folder paths
   -	Returns the duration of each stimulation event and specific times in ms for each event based on the start of the T-series
-	save_badframes_arr(somefishclass, no_planes) = saves the array of 'bad frames' when the stimulation event occurred, necessary for suite2p processing of photostimulation events
   -	somefishclass = Basefish instance of the data to use the folder paths
   -	no_planes = number of planes that are in the dataset, important for single or volume (default: 0)
   -	Returns somefishclass.badframes_arr = np array of the stimulation events in frame number 
-	identify_stim_sites(somebasefish, rotate, planes_stimed) = utilies for coordinates
(default: True)
   -	somefishclass = Basefish instance of the data to use the folder paths
   -	planes_stimed = list, the exact plane that were actually stimulated (as not every plane was stimmed) (default: [1,2,3,4])
   -	Default saves a 'stim_sites_volume.h5' dataframe, lists each plane, x, y, spiral size for all the stim sites (rows)
   -	Returns somebasefish.stim_sites_df = a stimulated site dataframe for each unique plane
-	run_suite2p_PS(somebasefish, input_tau, move_corr) = processes data in suite2p to calculate calicum sources of photostimulated data
   -	somebasefish = basefish, the data you want to have suite2p run on
   -	input_tau = decay value for gcamp indicator (6s = 1.5, m = 1.0, f = 0.7) (default: 1.5)
   -	move_corr = binary, if you want the motion corrected image to be run as the main data source or not (default: False)
-	return_raw_coord_trace(cell_coord, img, s) = returns the ROI location of a specified coordinate in the target fish
   -	cell_coord = list; x and y coordinate
   -	img = np array of time series image 
   -	s = size of mask on the np array img (default: 5)
   -	Returns the average trace of that mask on the img
-	collect_raw_traces(somebasefish) = find raw traces in pixels for each specific stim site
   -	somebasefish = basefish, the data you want to collect traces from
   -	Returns raw_traces = np.array of pixel values (columns) for each stim site (rows) in dataframe; points = coordinates for each stim site (rows)
-	all_stimmed_traces_array(stimulated_fishvolume) = make an array of all the stimulated traces in a whole volume
   -	stimulated_fishvolume = basefish in volume, this needs to be a volume for it to work
   -	Returns stim_traces_array
-	identify_stimmed_planes(omr_tseries_folder_path, clst_label) = returns the unique planes that were stimulated in the experiment
   -	omr_tseries_folder_path = folder path of the data where the cluster df (aka the stimulated clusters dataframe) is located, will be the omr tseries folder path
   -	clst_label = the label of the stimulated cluster in this dataset (could be a cluster number or barcode label) 
   -	Returns stimmed_planes in the dataset
-	correlations_with_stim_sites(somebasefish, traces_array, corr_threshold, normalizing, saving) = determining the correlations between stimulated sites (raw pixel traces) and all the suite2p cells, can normalize all data to a specific value (i.e. the positive control of all the stimulated sites to one another)
   -	somebasefish = basefish, the data you want to collect traces from
   -	traces_array = np array of traces that you want to compare the cell traces to, if this is for a volume of stim sites, then need to input yourself (default: None --> means that you will only be looking at correlations with traces on that plane only) 
   -	corr_threshold = selecting neurons based on this correlation threshold (default; 0.5)
   -	normalizing = value to normalize all data to (default: 1)
   -	saving = saving the correlation dataframe (default: True)

<br>

### Documentation of bruker_images.py
Contains pre-processing functions on any data from the Bruker
- get_frametimes(info_xml_path, voltage_path) = calculates frame times from either the information xml file or the voltage recording
   -	info_xml_path = path to info xml file
   -	voltage_path = path to voltage recording csv file
   -	Returns a frametimes_df (dataframe with the times and frames, saved as the 'master_frametimes.h5')
- bruker_img_organization(folder_path, testkey, safe, single_plane, pstim_file) = PV 5.8 software bruker organization function (ome tif files into regular tif files)
   -	testkey = str, the key that is in each bruker ome tif file (default:'Cycle')
   -	safe = True/False, safety key, if the end processed file exists already (default: False)
   -	single_plane = True/False, if there is a single plane in the recording or multiple planes (default: False)
   -	pstim_file = True/False, if there is a pstim txt file present (or not) that needs to be moved into each plane (default: True)
   -	Returns processed folders/files so that all the Bruker ome files are now divided into one folder per plane in output_folders, where also important files are copied and moved there from the master data folder
- addSecs(tm, secs) = Add seconds to datetime values
   -	tm = datetime value that needs to be changed
   -	secs = number of seconds you want to add to the tm value
   -	Returns new time with added secs (datetime object)
- addHours(tm, hrs) = Add hours to datetime values
   -	tm = datetime value that needs to be changed
   -	hrs = number of hours you want to add to the tm value
   -	Returns new time with added hrs (datetime object)
- move_xml_files(folder_path) = Moving xml, csv, env Bruker output files into output plane folders
   -	folder_path = the master data folder path that contains the xml files and output folders
- get_micronstopixels_scale(somebaseFish) = get the scale microns per pixel from the xml file
   -	info_xml_file_path = info xml file path
   -	Returns pixel_size (microns/pixel)
- read_xml_to_root(xml_file_path) = Read xml file into a root directory
   -	Returns root directory
- read_xml_to_str(xml_file_path) = Read a xml file into a data string
   -	Returns string


<br>

### Documentation of utilities folder
Generally contains processing functions on arrays, coordinates, paths, rois, stats, volumes, zmq, analysis methods
-	arrutils = utilies for arrays
-	barcoding = barcoding analysis functions (adapted from Whit)
-	clustering = hierarchical clustering analysis functions (adapted from Whit)
-	coordutils = utilies for coordinates
-	pathsutils = utilies for paths
-	roiutils = utilies for ROIs
-	statutils = utilies for statistics
-	volutils = utilies for volumes
-	zmqutils = utilies for zmq functions
