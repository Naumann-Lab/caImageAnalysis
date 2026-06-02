import os
import shutil
import numpy as np
from pathlib import Path

import caiman as cm
from caiman.source_extraction.cnmf import cnmf, params
from caiman.utils.visualization import get_contours
from scipy.ndimage import binary_fill_holes
import math
from caiman.source_extraction.cnmf import cnmf as cnmf_module

### PREPROCESS IMAGE TIFFS ###

def run_image_rotation(base_fish, angle=0, crop=0.075):
    """
    :param base_fish:
    :param angle:
    :param crop: percentage of image cropped on the fly back side (which is the left side with how it saves)
    :return:
    """

    from scipy.ndimage import rotate
    from tifffile import imread, imwrite

    image = imread(base_fish.data_paths["image"])

    image = image[:, :, int(image.shape[2] * crop) :]

    rotated_image = [rotate(img, reshape=False, angle=angle).astype(img.dtype) for img in image]
    imwrite(base_fish.folder_path.joinpath("img_rotated.tif"), rotated_image, bigtiff=True)

def run_image_rotation_90deg(base_fish, crop=0.0):
    """
    Rotate the imaging movie by +90 degrees (counter-clockwise) and save to disk.
    Uses np.rot90 → preserves dtype & avoids file size bloat seen with scipy.rotate.

    Parameters
    ----------
    base_fish : object
        Your fish object containing data_paths and folder_path
    crop : float
        Optional fraction to crop from the left side (same behavior as your original).
    """

    import numpy as np
    from tifffile import imread, imwrite

    # load full movie: shape (T, Y, X)
    image = imread(base_fish.data_paths["image"])

    # optional crop on X dimension *before* rotation
    if crop > 0:
        crop_px = int(image.shape[2] * crop)
        image = image[:, :, crop_px:]

    # rotate all frames by 90 degrees CCW
    # axes=(1,2) means rotate in the spatial dimensions only
    rotated = np.rot90(image, k=1, axes=(1, 2))

    # write to disk
    out_path = base_fish.folder_path.joinpath("img_rotated.tif")
    imwrite(out_path, rotated.astype(image.dtype), bigtiff=True)

def run_movement_correction(
    base_fish,
    caiman_ops=None,
    keep_mmaps=False,
    force=False,
    cropped = False
):
    import caiman as cm
    from tifffile import imsave

    base_fish.process_filestructure(midnight_noon = "noon")  # why not update :)

    if "move_corrected_image" in base_fish.data_paths.keys():
        if not force:
            print("movecorrect seems already done and not forced")
            return

    if "rotated_image" in base_fish.data_paths.keys():
        original_image_path = base_fish.data_paths["rotated_image"]
    else:
        original_image_path = base_fish.data_paths["image"]

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
        backend="local", n_processes=14, single_thread=False
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
    output = m_els
    
    if cropped:
        output = m_els[:,
            2 * bord_px_rig : -2 * bord_px_rig,
            2 * bord_px_rig : -2 * bord_px_rig,
        ] # this output is actually a cropped image

    if not keep_mmaps:
        with os.scandir(original_image_path.parents[0]) as entries:
            for entry in entries:
                if entry.is_file():
                    if entry.name.endswith(".mmap"):
                        os.remove(entry)
    dview.terminate()
    cm.stop_server()

    new_path = base_fish.folder_path.joinpath("movement_corr_img.tif")
    imsave(new_path, output) # saving the full motion corrected image here

### RUN SOURCE EXTRACTION FUNCTIONS ###

def run_suite2p(base_fish, input_tau=1.5, spatial_scale = 0, custom_parameter_dict=None, force=False):
    try:
        from suite2p import run_s2p, default_ops
    except:
        try:
            from suite2p.suite2p import run_s2p, default_ops
        except:
            print("failed to import suite2p")

    base_fish.process_filestructure(midnight_noon = base_fish.midnight_noon_keyword)  # why not update :)

    if "suite2p" in base_fish.data_paths.keys():
        if not force:
            print("suite2p seems already done and not forced")
            return

    imageHz = base_fish.hzReturner(base_fish.frametimes_df)
    try:
        imagepath = base_fish.data_paths[" "]
    except KeyError:
        imagepath = base_fish.data_paths["rotated_image"]

    # basic changes to suite2p ops to fit the fishy format
    basic_s2p_ops = {
            "data_path": [imagepath.parents[0].as_posix()],
            "save_path0": imagepath.parents[0].as_posix(),
            "tau": input_tau,
            "preclassify": 0.15,
            "allow_overlap": True,
            "block_size": [32, 32],
            "spatial_scale" : spatial_scale,
            "fs": imageHz,
            "tiff_list": [imagepath.name],
        }

    ops = default_ops()
    db = {}

    for item in basic_s2p_ops:
        ops[item] = basic_s2p_ops[item]

    # for additional changes:
    if custom_parameter_dict is not None: # can edit parameters as you want
        for key in custom_parameter_dict:
            if key in ops:
                ops[key] = custom_parameter_dict[key]

    output_ops = run_s2p(ops=ops, db=db)

def run_suite2p_normal(imagepath, imageHz, input_tau=1.5, custom_parameter_dict=None):
    try:
        from suite2p import run_s2p, default_ops
    except:
        try:
            from suite2p.suite2p import run_s2p, default_ops
        except:
            print("failed to import suite2p")

    basic_s2p_ops = {
            "data_path": [imagepath.parents[0].as_posix()],
            "save_path0": imagepath.parents[0].as_posix(),
            "tau": input_tau,
            "preclassify": 0.15,
            "allow_overlap": True,
            "block_size": [32, 32],
            "spatial_scale" : 0,
            "fs": imageHz,
            "tiff_list": [imagepath.name],
        }

    ops = default_ops()
    db = {}

    for item in basic_s2p_ops:
        ops[item] = basic_s2p_ops[item]

    # for additional changes:
    if custom_parameter_dict is not None: # can edit parameters as you want
        for key in custom_parameter_dict:
            if key in ops:
                ops[key] = custom_parameter_dict[key]

    output_ops = run_s2p(ops=ops, db=db)

def run_caiman_cnmf(base_fish, custom_parameter_dict = None, match_suite2p = True, keep_mmaps = False, force = True):
    '''
    base_fish: some BaseFish class that needs to be processed
    custom_parameter_dict: dictionary with custom parameters for caiman source extraction 
    match_suite2p: if you need data to match suite2p output
    keep_mmaps: if you want to keep the caiman memap file, typically don't need this
    '''
    from pathlib import Path
    import caiman as cm
    import math
    from caiman.source_extraction.cnmf import cnmf, params
    from caiman.utils.visualization import get_contours

    caiman_folder = Path(base_fish.folder_path).joinpath("caiman")
    if force and caiman_folder.exists():
        print('deleting old caiman output')
        shutil.rmtree(caiman_folder)

    if 'move_corrected_image' not in base_fish.data_paths.keys():
        movie_path = base_fish.data_paths['rotated_image']
    elif 'move_corrected_image' in base_fish.data_paths.keys():
        movie_path = base_fish.data_paths['move_corrected_image']
    else:
        movie_path = base_fish.data_paths['image']
        
    movie_orig = cm.load(movie_path)
    framerate = base_fish.hzReturner(base_fish.frametimes_df)

    correlation_image_orig = cm.local_correlations(movie_orig, swap_dim=False)
    correlation_image_orig[np.isnan(correlation_image_orig)] = 0 # get rid of NaNs, if they exist
    
    parameter_dict = {'fnames': [movie_path],
                      'fr': framerate, # framerate, very important!
                    'p': 1, # order of the autoregressive system
                    'nb': 2,
                    'merge_thr': 0.85, # merging threshold, max correlation allowed
                    'rf': 25, # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80, must be 3-4 times larger than one neuron size
                    'stride': 10, # amount of overlap between the patches in pixels
                    'K': 8, # Number of (expected) components per patch
                    'gSig': [4, 4], # expected half-width of neurons in pixels 
                    'ssub': 1,
                    'tsub': 1,
                    'method_init': 'greedy_roi',
                    'min_SNR': 1.5,
                    'rval_thr': 0.7,
                    'use_cnn': True,
                    'min_cnn_thr': 0.8,
                    'cnn_lowest': 0.1,
                    'decay_time': 0.4, # gcamp6f
                         }
    
    if custom_parameter_dict is not None: # can edit parameters as you want
        print('loading in custom parameters')
        for key in custom_parameter_dict:
            if key in parameter_dict:
                parameter_dict[key] = custom_parameter_dict[key]

    parameters = params.CNMFParams(params_dict=parameter_dict) # CNMFParams is the parameters class

    # stopping other servers, to make sure on the newest one
    _, cluster, n_processes = cm.cluster.setup_cluster(backend='local', 
                                                       n_processes=None, single_thread=False)
    
    # don't need to run motion correction here, straight to memmap
    mc_memmapped_fname = cm.save_memmap([movie_orig], base_name='memmap_',
                                         order='C', border_to_0=0, dview=cluster)

    #reshape frames in standard 3d format (T x X x Y)
    Yr, dims, num_frames = cm.load_memmap(mc_memmapped_fname)
    images = np.reshape(Yr.T, [num_frames] + list(dims), order='F') 
    
    cnmf_model = cnmf.CNMF(n_processes, params=parameters, dview=cluster)
    
    cnmf_fit = cnmf_model.fit(images)
    cnmf_refit = cnmf_fit.refit(images, dview=cluster)
    print('finished 2 iterations on cnmf model')
    
    # evaulating components
    cnmf_refit.estimates.evaluate_components(images, cnmf_refit.params, dview=cluster);
    # making df/f estimates
    cnmf_refit.estimates.detrend_df_f(quantileMin=8, frames_window=250,flag_auto=False,use_residuals=False);  
    
    if not keep_mmaps:
        with os.scandir(movie_path.parents[0]) as entries:
            for entry in entries:
                if entry.is_file():
                    if entry.name.endswith(".mmap"):
                        os.remove(entry)
    
    #saving cnmf model
    moveto_folder = Path(base_fish.folder_path).joinpath("caiman")
    if not os.path.exists(moveto_folder):
            os.mkdir(moveto_folder)
    save_path = str(moveto_folder) + '\\cnmf_results.hdf5'
    cnmf_refit.estimates.Cn = correlation_image_orig # squirrel away correlation image with cnmf object
    cnmf_refit.save(save_path)
    print('saved cnmf results')

    # saving calcium traces
    print(f'shape of raw traces {cnmf_refit.estimates.C.shape}')
    np.save(Path(moveto_folder).joinpath('raw.npy'), cnmf_refit.estimates.C + cnmf_refit.estimates.YrA) # raw traces
    np.save(Path(moveto_folder).joinpath('C.npy'), cnmf_refit.estimates.C) # denoised calcium
    df_f_traces = cnmf_refit.estimates.detrend_df_f(quantileMin=8, frames_window=250)
    np.save(Path(moveto_folder).joinpath('F_dff.npy'), df_f_traces) # df/f traces
    np.save(Path(moveto_folder).joinpath('baseline.npy'),cnmf_refit.estimates.bl) # baseline
    
    # grabbing coordinates and centers
    centers = cm.base.rois.com(cnmf_refit.estimates.A, *cnmf_refit.estimates.Cn.shape)
    correct_centers = centers[:, ::-1] #need to invert x and y positions in the CoM array
    coors = get_contours(cnmf_refit.estimates.A, correlation_image_orig.shape)
    og_coordinates_arr = np.array([coors[i]['coordinates'] for i in range(len(coors))])
    # remove any nan's in the coordinate list
    filtered_coordinates = []
    for coord_lst in og_coordinates_arr:
        new_coord_lst = []
        for coord in coord_lst:
            if not (math.isnan(coord[0]) or math.isnan(coord[1])):
                new_coord_lst.append([coord[0], coord[1]])
        filtered_coordinates.append(np.array(new_coord_lst))
    coordinates_arr = np.array(filtered_coordinates)

    #saving accepted cells
    accepted_cells_arr = np.zeros(shape = len(coordinates_arr))
    print(f'shape of accepted traces {accepted_cells_arr.shape}')
    for i in cnmf_refit.estimates.idx_components:
        accepted_cells_arr[i] = 1
    np.save(Path(moveto_folder).joinpath('iscell.npy'), accepted_cells_arr) # boolean, if a cell or not
    
    # saving coordinates and centers
    np.save(Path(moveto_folder).joinpath('center.npy'), correct_centers) # center of ROIs
    np.save(Path(moveto_folder).joinpath('coordinates.npy'),coordinates_arr) # spatial contours
    
    if match_suite2p:
        new_coordinates_arr = make_coordinates_into_dict(coordinates_arr)
        np.save(Path(moveto_folder).joinpath('coordinates_dict.npy'), new_coordinates_arr) # matching suite2p output
        
    cm.stop_server(dview=cluster)

def gather_raw_traces_from_cnmf_output(somebasefish):
    '''
    In case I did not save the raw traces from the process caiman function, this will gather that and put into the caiman folder
    Adds C and Yra --> Each row in YrA corresponds to the residual signal after denoising the corresponding component in estimates.C.
    '''
    import caiman as cm
    from pathlib import Path

    c_p = somebasefish.data_paths['caiman'].joinpath('cnmf_results.hdf5')
    cnmf_h5_file = cm.source_extraction.cnmf.cnmf.load_CNMF(c_p)
    raw_arr = cnmf_h5_file.estimates.YrA + cnmf_h5_file.estimates.C # raw is C + YrA (deconvolved + residual components)
    moveto_folder = somebasefish.data_paths['caiman'].joinpath('raw.npy')
    np.save( Path(moveto_folder) ,raw_arr)

    return print('saved raw traces from caiman output')

def gather_df_f_traces_from_cnmf_output(somebasefish):
    '''
    get the actual df/f traces need to calculate it with cnmf model
    :param somebasefish: basefish object with caiman processed
    :return: accurate df/f traces
    '''
    from caiman.source_extraction.cnmf.cnmf import load_CNMF

    cnmf_model_path = somebasefish.folder_path.joinpath('caiman/cnmf_results.hdf5')
    # load saved model
    cnmf_model = load_CNMF(cnmf_model_path)
    # compute df/f
    cnmf_model.estimates.detrend_df_f(quantileMin=8, frames_window=250)
    # get traces
    dff_traces = cnmf_model.estimates.F_dff   # shape: cells × time
    np.save(Path(cnmf_model_path.parents[0]).joinpath('F_dff.npy'), dff_traces)  # df/f traces

    return print('saved df/f traces')

def make_coordinates_into_dict(array_of_coors):
    
    #remove nan's
    nonan_coors_arr = []
    for x in array_of_coors:
        lst = []
        for y in x:
            if not np.isnan(y[0]):
                lst.append(y) 
        nonan_coors_arr.append(np.array(lst).astype(np.int32))
    
    # create coordinate list with xpix and ypix together (matches suite2p output)
    new_coordinates = []
    for idx in range(len(nonan_coors_arr)):
        try:
            idx_dict = {}
            idx_dict['xpix'] = nonan_coors_arr[idx][:,0]
            idx_dict['ypix'] = nonan_coors_arr[idx][:,1]
            new_coordinates.append(idx_dict)
        except:
            new_coordinates.append({'xpix': np.nan, 'ypix': np.nan})
        
    return new_coordinates

### RUN SOURCE EXTRACTION FUNCTIONS - CAIMAN, FOR SUPER LARGE DATASETS ###
# derived from claude ai - important functions

def make_subsampled_concat(movie_paths, subsample=3, out_path='subsampled_concat.tif', bad_frames_per_seq=None):
    """
    Load each sequence, subsample temporally, remove bad frames (±1 padding),
    concatenate, save as tif.

    Parameters
    ----------
    movie_paths        : list of str or Path
    subsample          : int — keep every Nth frame
    out_path           : str — where to save the concatenated subsampled tif
    bad_frames_per_seq : list of list of int or None
                         one list of bad frame indices per sequence,
                         in the original (pre-subsample) frame space.
                         e.g. [[10, 200], [45], [], [300, 301]]

    Returns
    -------
    out_path     : str
    """
    import tifffile

    frame_counts = []
    total_frames = 0

    for i, p in enumerate(movie_paths):
        movie = np.array(cm.load(str(p)))   # (T, H, W)
        subsampled = movie[::subsample]
        del movie  # free immediately

        T_sub = subsampled.shape[0]
        if bad_frames_per_seq is not None and len(bad_frames_per_seq[i]) > 0:
            bad_sub = set()
            for bf in bad_frames_per_seq[i]:
                bf_sub = bf // subsample
                for offset in [-1, 0, 1]:
                    idx = bf_sub + offset
                    if 0 <= idx < T_sub:
                        bad_sub.add(idx)

            keep_mask = np.ones(T_sub, dtype=bool)
            keep_mask[list(bad_sub)] = False
            n_removed = int((~keep_mask).sum())
            subsampled = subsampled[keep_mask]
            print(f"  {Path(p).name}: removed {n_removed} frames around "
                  f"{len(bad_frames_per_seq[i])} bad frame events")
        else:
            print(f"  {Path(p).name}: no bad frames")

        # cast to float32 to halve memory
        subsampled = subsampled.astype(np.float32)

        # append directly to tif on disk — never accumulate in RAM
        tifffile.imwrite(str(out_path), subsampled, append=True, bigtiff=True)

        frame_counts.append(subsampled.shape[0])
        total_frames += subsampled.shape[0]
        print(f"    {subsampled.shape[0]} frames written (running total: {total_frames})")
        del subsampled

    print(f"\nTotal subsampled concat: {total_frames} frames saved to {out_path}")
    return str(out_path)


def run_caiman_on_subsampled(subsampled_path, framerate_subsampled, caiman_folder, custom_params=None):
    """
    Run CaImAn CNMF on the subsampled concatenated movie.
    Only the spatial footprints (A matrix) from this are used downstream —
    the temporal traces C are discarded.

    Parameters
    ----------
    subsampled_path     : str — path to subsampled tif
    framerate_subsampled: float — effective framerate after subsampling
                          e.g. 6Hz / 3 = 2.0Hz
    caiman_folder       : Path — where to save the HDF5
    custom_params       : dict — override any default parameters

    Returns
    -------
    save_path : str — path to saved HDF5
    """
    caiman_folder = Path(caiman_folder)
    caiman_folder.mkdir(parents=True, exist_ok=True)

    movie_orig = cm.load(subsampled_path)
    correlation_image = cm.local_correlations(movie_orig, swap_dim=False)
    correlation_image[np.isnan(correlation_image)] = 0

    parameter_dict = {
        'fnames'      : [subsampled_path],
        'fr'          : framerate_subsampled,
        'p'           : 1,
        'nb'          : 2,
        'merge_thr'   : 0.75,
        'rf'          : 20,
        'stride'      : 7,
        'K'           : 10,
        'gSig'        : [4, 4],
        'ssub'        : 1,
        'tsub'        : 1,
        'method_init' : 'greedy_roi',
        'min_SNR'     : 2.0,
        'rval_thr'    : 0.85,
        'use_cnn'     : True,
        'min_cnn_thr' : 0.9,
        'cnn_lowest'  : 0.3,
        'decay_time'  : 1.5,   # GCaMP7f zebrafish RT
    }

    if custom_params is not None:
        for k, v in custom_params.items():
            parameter_dict[k] = v
        print(f"Custom params applied: {list(custom_params.keys())}")

    parameters = params.CNMFParams(params_dict=parameter_dict)

    _, cluster, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

    mc_memmapped_fname = cm.save_memmap(
        [movie_orig], base_name='memmap_', order='C',
        border_to_0=0, dview=cluster)

    Yr, dims, num_frames = cm.load_memmap(mc_memmapped_fname)
    images = np.reshape(Yr.T, [num_frames] + list(dims), order='F')

    cnmf_model = cnmf.CNMF(n_processes, params=parameters, dview=cluster)
    cnmf_fit   = cnmf_model.fit(images)
    cnmf_refit = cnmf_fit.refit(images, dview=cluster)
    print('Finished 2 CNMF iterations')

    cnmf_refit.estimates.evaluate_components(
        images, cnmf_refit.params, dview=cluster)

    n_acc = len(cnmf_refit.estimates.idx_components)
    n_rej = len(cnmf_refit.estimates.idx_components_bad)
    print(f"Accepted: {n_acc}  |  Rejected: {n_rej}")

    # clean up mmap
    for entry in os.scandir(Path(subsampled_path).parent):
        if entry.name.endswith('.mmap'):
            os.remove(entry)

    # save correlation image with object
    cnmf_refit.estimates.Cn = correlation_image

    save_path = str(caiman_folder / 'cnmf_subsampled_reference.hdf5')
    cnmf_refit.save(save_path)
    print(f"Saved reference CNMF to {save_path}")

    cm.stop_server(dview=cluster)
    return save_path


def save_roi_metadata(cnm, caiman_folder, match_suite2p=True):
    """
    Save ROI spatial metadata in CNMF style, with optional Suite2p style dictionary.
    """
    import math
    import numpy as np
    from pathlib import Path
    from caiman.base.rois import com as caiman_com
    from caiman.utils.visualization import get_contours

    caiman_folder = Path(caiman_folder)
    Cn = cnm.estimates.Cn
    A_all = cnm.estimates.A

    # ---- centers ----
    centers = caiman_com(A_all, *Cn.shape)
    centers_xy = centers[:, ::-1]

    # ---- contours ----
    coors = get_contours(A_all, Cn.shape)
    filtered_coords = []
    for c in coors:
        coords = c['coordinates']
        coords_clean = [[x, y] for x, y in coords if not (math.isnan(x) or math.isnan(y))]
        filtered_coords.append(np.array(coords_clean))
    coords_arr = np.array(filtered_coords, dtype=object)

    # ---- iscell ----
    iscell = np.zeros(len(coords_arr))
    iscell[cnm.estimates.idx_components] = 1

    # ---- save CNMF style ----
    np.save(caiman_folder / 'center.npy', centers_xy)
    np.save(caiman_folder / 'coordinates.npy', coords_arr)
    np.save(caiman_folder / 'iscell.npy', iscell)
    print(f"Saved ROI metadata: {int(iscell.sum())} accepted cells")

    # ---- optional Suite2p-style dict ----
    if match_suite2p:
        coords_dict = process.make_coordinates_into_dict(coords_arr)
        np.save(caiman_folder / 'coordinates_dict.npy', coords_dict)
        print("Saved coordinates_dict.npy (in Suite2p style)")

    return centers_xy, coords_arr, iscell

def extract_traces_with_residuals(movie_paths,cnm_ref, output_folder, bad_frames_dict=None, save_per_sequence=True):
    '''
    # new function from chatgpt for getting the correct C traces, and getting things in the correct format
    :param movie_paths:
    :param cnm_ref:
    :param output_folder:
    :param bad_frames_dict:
    :param save_per_sequence:
    :return:
    '''
    import numpy as np
    import caiman as cm
    from pathlib import Path

    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)

    A = cnm_ref.estimates.A
    n_neurons = A.shape[1]

    all_C = []
    all_YrA = []
    all_raw = []

    print(f"Processing {len(movie_paths)} sequences...")

    for seq_idx, movie_path in enumerate(movie_paths):

        print(f"\n--- Sequence {seq_idx} ---")

        movie = np.array(cm.load(str(movie_path))).astype(np.float32)
        T, H, W = movie.shape

        Y = movie.reshape(-1, T, order='F')
        del movie

        # ---- projection ----
        AtA_diag = np.array(A.multiply(A).sum(axis=0)).ravel()
        AtY = A.T @ Y
        C_proj = AtY / (AtA_diag[:, None] + 1e-9)

        # ---- YrA approximation ----
        Y_hat = A @ C_proj
        R = Y - Y_hat
        YrA_approx = (A.T @ R) / (AtA_diag[:, None] + 1e-9)

        # ---- raw ----
        raw_approx = C_proj + YrA_approx

        # ---- bad frames ----
        if bad_frames_dict and seq_idx in bad_frames_dict:
            bad_frames = bad_frames_dict[seq_idx]
            C_proj[:, bad_frames] = np.nan
            YrA_approx[:, bad_frames] = np.nan
            raw_approx[:, bad_frames] = np.nan

        # store
        all_C.append(C_proj)
        all_YrA.append(YrA_approx)
        all_raw.append(raw_approx)

        # optional saving
        if save_per_sequence:
            np.save(output_folder / f'C_seq{seq_idx}.npy', C_proj.astype(np.float32))
            np.save(output_folder / f'YrA_seq{seq_idx}.npy', YrA_approx.astype(np.float32))
            np.save(output_folder / f'raw_seq{seq_idx}.npy', raw_approx.astype(np.float32))

    # ---- concatenate ----
    C_full = np.concatenate(all_C, axis=1)
    YrA_full = np.concatenate(all_YrA, axis=1)
    raw_full = np.concatenate(all_raw, axis=1)

    print(f"\nFinal shape: {C_full.shape}")

    # ---- iscell (same as CaImAn output) ----
    iscell = np.zeros(n_neurons, dtype=np.uint8)
    iscell[cnm_ref.estimates.idx_components] = 1

    # backup original C & YrA (optional but safer)
    C_backup = cnm_ref.estimates.C.copy()
    YrA_backup = cnm_ref.estimates.YrA if hasattr(cnm_ref.estimates, 'YrA') else None

    cnm_ref.estimates.C = C_full
    # cnm_ref.estimates.detrend_df_f(quantileMin=8, frames_window=250, use_residuals=False)
    # F_dff = cnm_ref.estimates.F_dff.astype(np.float32)
    # baseline = cnm_ref.estimates.bl.astype(np.float32)

    # restore original C (optional)
    cnm_ref.estimates.C = C_backup
    if YrA_backup is not None:
        cnm_ref.estimates.YrA = YrA_backup

    # ---- save (match your naming style) ----
    np.save(output_folder / 'C.npy', C_full.astype(np.float32))
    np.save(output_folder / 'YrA.npy', YrA_full.astype(np.float32))
    np.save(output_folder / 'raw.npy', raw_full.astype(np.float32))
    np.save(output_folder / 'iscell.npy', iscell)
    # np.save(output_folder / 'F_dff.npy', F_dff.astype(np.float32))
    # np.save(output_folder / 'baseline.npy', baseline)

    return C_full, YrA_full, raw_full


### THRESHOLDING IMAGES ###

def threshold_otsu_255_bins(image):
    """
    Compute Otsu's threshold using 255 bins (without clipping).

    Parameters
    ----------
    image : cupy.ndarray
        Grayscale input image as a CuPy array. Can be any dtype or range.
        The histogram is computed with 255 bins over the min->max of 'image'.

    Returns
    -------
    threshold : float
        Otsu threshold in the same scale as the input image data.
    """

    # 1. Quick check if the image is constant:
    first_val = image.ravel()[0]
    if np.all(image == first_val):
        return float(first_val.get())

    # 2. Compute histogram with exactly 255 bins over the entire data range.
    counts, bin_edges = np.histogram(image, bins=255)

    # 3. Compute bin centers (shape: (255,))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 4. Cumulative sums ("weights")
    weight1 = np.cumsum(counts)  # up to bin i
    weight2 = np.cumsum(counts[::-1])[::-1]  # from bin i to the end

    # 5. Compute means for each side of the threshold
    cumsum_val = np.cumsum(counts * bin_centers)
    mean1 = cumsum_val / weight1
    cumsum_val_rev = np.cumsum((counts * bin_centers)[::-1])
    mean2 = (cumsum_val_rev / weight2[::-1])[::-1]

    # 6. Inter-class variance
    #    We skip the last bin in weight1[:-1] and the first bin in weight2[1:]
    #    Otsu's formula: sigma_B^2 = w1*w2*(mean1-mean2)^2
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # 7. Argmax for the best threshold
    idx = np.argmax(variance12)
    threshold = bin_centers[idx]

    # Return as Python float
    return float(threshold)

def normalizeBinarize(image, method = 'otsu', threshold_factor=0.9):
    if method == 'otsu':
        threshold = threshold_otsu_255_bins(image) * threshold_factor
    if method == 'huang':
        threshold = threshold_huang(image) * threshold_factor
    if method == 'mean':
        threshold = np.mean(image) * threshold_factor

    thresholded = (image >= threshold)
    
    return (thresholded)

def threshold_huang(image):
    """Apply Huang-like thresholding (using skimage's threshold_minimum)."""
    from skimage.filters import threshold_minimum

    threshold = threshold_minimum(image)  # Alternative for Huang's method
    return threshold