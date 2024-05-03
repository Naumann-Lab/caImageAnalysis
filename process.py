import os

import numpy as np


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

    rotated_image = [rotate(img, angle=angle) for img in image]
    imwrite(
        base_fish.folder_path.joinpath("img_rotated.tif"), rotated_image, bigtiff=True
    )


def run_movement_correction(
    base_fish,
    caiman_ops=None,
    keep_mmaps=False,
    force=False,
    cropped = False
):
    import caiman as cm
    from tifffile import imsave

    base_fish.process_filestructure()  # why not update :)

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
    output = m_els
    
    if cropped:
        output = m_els[
            :,
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


def run_suite2p(base_fish, input_tau=1.5, spatial_scale = 0, s2p_ops=None, force=False):
    try:
        from suite2p import run_s2p, default_ops
    except:
        try:
            from suite2p.suite2p import run_s2p, default_ops
        except:
            print("failed to import suite2p")

    base_fish.process_filestructure()  # why not update :)

    if "suite2p" in base_fish.data_paths.keys():
        if not force:
            print("suite2p seems already done and not forced")
            return

    imageHz = base_fish.hzReturner(base_fish.frametimes_df)
    try:
        imagepath = base_fish.data_paths["move_corrected_image"]
    except KeyError:
        imagepath = base_fish.data_paths["image"]

    if not s2p_ops:
        s2p_ops = {
            "data_path": [imagepath.parents[0].as_posix()],
            "save_path0": imagepath.parents[0].as_posix(),
            "tau": input_tau,
            "preclassify": 0.15,
            "allow_overlap": True,
            "block_size": [32, 32],
            # "threshold_scaling": 0.9,
            "spatial_scale" : spatial_scale,
            "fs": imageHz,
            "tiff_list": [imagepath.name],
        }

    ops = default_ops()
    db = {}
    for item in s2p_ops:
        ops[item] = s2p_ops[item]

    output_ops = run_s2p(ops=ops, db=db)


def run_caiman_cnmf(base_fish, custom_parameter_dict = None, match_suite2p = True, keep_mmaps = False):
    '''
    base_fish: some BaseFish class that needs to be processed
    custom_parameter_dict: dictionary with custom parameters for caiman source extraction 
    match_suite2p: if you need data to match suite2p output
    keep_mmaps: if you want to keep the caiman memap file, typically don't need this
    '''
    from pathlib import Path
    import caiman as cm
    from caiman.source_extraction.cnmf import cnmf, params
    from caiman.utils.visualization import get_contours
    
    if Path(base_fish.folder_path).joinpath("caiman/cnmf_results.hdf5").exists():
        print('cnmf processed')
    
    movie_path = base_fish.data_paths['move_corrected_image']
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
    np.save( Path(moveto_folder).joinpath('C.npy'), cnmf_refit.estimates.C) # raw calcium 
    np.save( Path(moveto_folder).joinpath('F_dff.npy'),cnmf_refit.estimates.F_dff) # df/f traces
    np.save( Path(moveto_folder).joinpath('baseline.npy'),cnmf_refit.estimates.bl) # baseline
    
    # grabbing coordinates and centers
    centers = cm.base.rois.com(cnmf_refit.estimates.A, *cnmf_refit.estimates.Cn.shape)
    correct_centers = centers[:, ::-1] #need to invert x and y positions in the CoM array
    coors = get_contours(cnmf_refit.estimates.A, correlation_image_orig.shape)
    coordinates_arr = np.array([coors[i]['coordinates'] for i in range(len(coors))])
    
    #saving accepted cells
    accepted_cells_arr = np.zeros(shape = len(coordinates_arr))
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
