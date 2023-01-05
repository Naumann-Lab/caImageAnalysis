import os

import numpy as np


def run_movement_correction(base_fish, caiman_ops=None, keep_mmaps=False, force=False):
    import caiman as cm
    from tifffile import imsave

    base_fish.process_filestructure()  # why not update :)

    if "move_corrected_image" in base_fish.data_paths.keys():
        if not force:
            print("movecorrect seems already done and not forced")
            return

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

    new_path = base_fish.folder_path.joinpath("movement_corr_img.tif")
    imsave(new_path, output)
    return


def run_suite2p(base_fish, input_tau=1.5, s2p_ops=None, force=False):
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
            "fs": imageHz,
            "tiff_list": [imagepath.name],
        }
    ops = default_ops()
    db = {}
    for item in s2p_ops:
        ops[item] = s2p_ops[item]

    output_ops = run_s2p(ops=ops, db=db)
    return
