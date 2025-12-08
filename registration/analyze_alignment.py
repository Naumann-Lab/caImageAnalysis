# script for quickly aligning each fish brain for the OMR2Stim ensemble datasets
# all elavl3:H2B:GCaMP6s, and the same FOV (512 x 512, 3x zoom)

# load imports
import os
import argparse
from tifffile import imread, imwrite
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json

import sys
sys.path.append(r'C:\Users\Kaitlyn\PyCharmProjects\imaging\caImageAnalysis')
from registration import sitkalignment, alignment_utils
from utilities import pathutils

import sys
sys.path.append(r'C:\Users\Kaitlyn\PyCharmProjects\imaging\scopeslip')
import crossCorrelation

# keep all the analysis in the same fish data folder

def load_data(folder_path):
    '''
    Load up all data into images, save into a dictionary to use later
    :param folder_path: path that holds the plane0 target image that you want to align
    :return: dictionary of images necessary for alignment
    '''

    # std image of plane 0
    functional_img = imread(Path(pathutils.pathcrawler(Path(folder_path), inset=set(), inlist=[], mykey = 'std_img_for_alignment')[0]))

    # reference stack path
    # reference_brain_img_path = Path(r"C:\Users\Kaitlyn\Reference_brains\mapzebrain\T_AVG_HuCH2BGCaMP2_croppedFOV.tif")
    reference_brain_img_path = Path(r"C:\Users\Kaitlyn\Reference_brains\mapzebrain\croppedFOV_2\T_AVG_HuCH2BGCaMP2-tg_croppedFOV2.tif")
    reference_stack = imread(reference_brain_img_path)

    # mask paths
    # pt_mask_path = r"C:\Users\Kaitlyn\Reference_brains\mapzebrain\mask_pretectum_croppedFOV.tif"
    pt_mask_path = r"C:\Users\Kaitlyn\Reference_brains\mapzebrain\croppedFOV_2\mask_pretectum_croppedFOV2.tif"
    pt_mask_stack = imread(pt_mask_path)
    # nmlf_mask_path = r"C:\Users\Kaitlyn\Reference_brains\mapzebrain\mask_nmlf_croppedFOV.tif" # had to binarize this since it's actually a marker (s1171)
    # nmlf_mask_path = r"C:\Users\Kaitlyn\Reference_brains\mapzebrain\mask_nmlf_croppedFOV_cleared.tif"
    nmlf_mask_path = r"C:\Users\Kaitlyn\Reference_brains\mapzebrain\croppedFOV_2\mask_nmlf_croppedFOV2_cleared.tif"
    nmlf_mask_stack = imread(nmlf_mask_path)
    # ahb_mask_path = r"C:\Users\Kaitlyn\Reference_brains\mapzebrain\mask_superior_medulla_oblongata_croppedFOV.tif"
    ahb_mask_path = r"C:\Users\Kaitlyn\Reference_brains\mapzebrain\croppedFOV_2\mask_superior_medulla_oblongata_croppedFOV2.tif"
    ahb_mask_stack = imread(ahb_mask_path)

    img_dict = {'functional_img': functional_img,
                'reference_stack': reference_stack,
                'Pt': pt_mask_stack,
                'nMLF': nmlf_mask_stack,
                'aHB': ahb_mask_stack}

    return img_dict

def preprocess_img_for_alignment(img):
    # do the preprocessing steps before embedding the image into the larger pixel space
    scaled_img = img / img.max()
    scaled_img *= 2 ** 12
    return scaled_img

def find_matching_z_in_reference_stack(functional_img, ref_image_stack):
    # run cross correlation metrics just like during imaging, to find the best matching plane in the reference stack to plane 0

    # 1 - resize my reference stack cropped image into the same pixel dimension as the functional image (ref stack is much smaller sized)
    resized_ref_images = []
    start_ref_slice = 210  # since i know that by eye the plane 0 should match something these planes
    end_ref_slice = 250
    for r in ref_image_stack[start_ref_slice:end_ref_slice]:
        new_r = alignment_utils.resize_and_center_crop(r)
        resized_ref_images.append(new_r)
    cropped_ref_image_stack = np.array(resized_ref_images)

    # 2 - run cross correlation between functional image and the mapzebrain images
    xcorr = crossCorrelation.CrossCorrelationAlignment(target_volume=cropped_ref_image_stack,
                                                       image=functional_img,
                                                       correlation_mode='thresholded')
    shiftx, shifty, shiftz = xcorr.align(xcorr.image)
    matched_plane_in_ref_stack = shiftz + start_ref_slice

    # 3 - visually check that this looks good in the original stack
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(functional_img, cmap='gray', vmax=np.percentile(functional_img, 99))
    ax[0].set_title('functional image')
    ax[1].imshow(ref_image_stack[matched_plane_in_ref_stack], cmap='gray',
                 vmax=np.percentile(ref_image_stack[matched_plane_in_ref_stack], 99))
    ax[1].set_title('matched reference stack image')
    [a.axis('off') for a in ax.flatten()]
    plt.show()

    return matched_plane_in_ref_stack

def save_override_match_val(save_path, new_plane_match):
    '''
    saving the matched reference stack image that is overriding automatic selection
    :param save_path: path to save the matched reference stack image
    :param new_plane_match: the new val
    :return:
    '''
    with open(save_path, "w") as f:
        json.dump({"override_plane0": new_plane_match}, f, indent=2)

    return print(f"[INFO] Saved override plane number to {save_path}")

def load_override_match_val(path):
    '''
    Now load in the override matched reference stack value
    :param path: path to save the matched reference stack image json file
    :return: the new mathcing stack val
    '''
    with open(path, "r") as f:
        saved_data = json.load(f)
        saved_override = saved_data.get("override_plane0")
        if saved_override is not None:
            print(f"[INFO] Loaded saved override_plane0 = {saved_override}")

    return saved_override

def prep_for_alignment(folder_path):
    '''
    Prepare for alignment for any plane
    :param folder_path: folder path that has data you will be aligning, saving
    :return: the data dictionary, and matching plane in the reference z stack
    '''
    img_dict = load_data(folder_path)

    matching_plane_num = find_matching_z_in_reference_stack(
        img_dict["functional_img"], img_dict["reference_stack"])

    alignment_fld = Path(folder_path).joinpath('alignment')
    alignment_fld.mkdir(exist_ok=True)

    return img_dict, matching_plane_num, alignment_fld

def find_z_step(master_folder_path):
    # find out the um steps between z steps
    info_xml = Path(master_folder_path).joinpath('plane_0\info.xml')
    with open(info_xml, "r") as f:
        data_str = f.read()
    for j, i in enumerate(data_str.split("\n")):
        if "micronsPerPixel" in i:
            next_line = data_str.split("\n")[j + 1]
            pixel_size = float(next_line.split('value=')[1].split('"')[1])
            z_step_size = float(data_str.split("\n")[j + 3].split('value=')[1].split('"')[1])
    return int(z_step_size)

def plot_and_save_new_ROIs(img_data_dict, roi_save_directory, fwd_transform_path, matching_z_plane):
    '''
    Plot and save new ROIs made from the alignment

    :param img_data_dict:
    :param roi_save_directory:
    :param fwd_transform_path:
    :param matching_z_plane:
    :return:
    '''
    # find out the good transformation paths and save the correct new rois into a folder

    # 1 - get all the ROI masks from the matching plane in the reference stack
    roi_lst = ['Pt', 'nMLF', 'aHB']
    for roi in roi_lst:
        embedded_roi_mask_img = sitkalignment.embed_image((img_data_dict[roi])[matching_z_plane], 1024)
        if roi == 'Pt':
            pt_points, binary_mask, num_contours = alignment_utils.extract_mask_boundaries(embedded_roi_mask_img, points_per_contour=40)
            # pt_points, binary_mask, num_contours = alignment_utils.update_extract_mask_boundaries(embedded_roi_mask_img, points_per_contour=40,
            #                                                merge_contours=True)
        if roi == 'nMLF':
            nmlf_points, binary_mask, num_contours= alignment_utils.update_extract_mask_boundaries(embedded_roi_mask_img, points_per_contour=60, merge_contours=True)
        if roi == 'aHB':
            # if matching_z_plane > 220: # lower than this value in the stack has multiple contours
            ahb_points, binary_mask, num_contours = alignment_utils.extract_mask_boundaries(embedded_roi_mask_img, points_per_contour=60)
            # else:
            #     ahb_points, binary_mask, num_contours = alignment_utils.update_extract_mask_boundaries(embedded_roi_mask_img, points_per_contour=60,
            #                                                merge_contours=True)
    # ahb_points, pt_points = alignment_utils.subtract_overlapping_polygons(ahb_points, pt_points)
    # 1B - need to clean up the nMLF and aHB coordinates, since from the masks they actually overlap
    nmlf_points, ahb_points = alignment_utils.subtract_overlapping_polygons(nmlf_points, ahb_points)
    # # 1C - also need to clean up Pt and nMLF coordinates, since these also overlap
    nmlf_points, pt_points = alignment_utils.subtract_overlapping_polygons(nmlf_points, pt_points)
    #


    # 2 - plot ROIs and save points in alignment files directory
    embedded_ref_img = sitkalignment.embed_image(img_data_dict['reference_stack'][matching_z_plane], 1024)
    embedded_functional_img = sitkalignment.embed_image(img_data_dict['functional_img'], 1024)
    fig, ax = plt.subplots(1, 3, figsize=(14, 6))
    ax[0].imshow(embedded_ref_img, cmap="gray", vmax=np.percentile(embedded_ref_img, 99),
                 vmin=np.percentile(embedded_ref_img, 20))
    ax[1].imshow(embedded_functional_img, cmap="gray", vmax=np.percentile(embedded_functional_img, 99),
                 vmin=np.percentile(embedded_functional_img, 20))
    ax[2].imshow(img_data_dict['functional_img'], cmap="gray", vmax=np.percentile(img_data_dict['functional_img'], 99),
                 vmin=np.percentile(img_data_dict['functional_img'], 20))
    unique_roi_colors = ['tab:blue', 'tab:green','tab:red']
    for n, points in enumerate([pt_points, nmlf_points, ahb_points]):
        [ax[0].scatter(p[0], p[1], s=8, color = unique_roi_colors[n]) for p in points]
        ax[0].fill([p[0] for p in points], [p[1] for p in points], color=unique_roi_colors[n], alpha=0.3)

        fwd_points = sitkalignment.transform_points(Path(fwd_transform_path), points)
        [ax[1].scatter(p[0], p[1], s=8,color = unique_roi_colors[n]) for p in fwd_points]
        ax[1].fill([p[0] for p in fwd_points], [p[1] for p in fwd_points], color=unique_roi_colors[n], alpha=0.3)

        # saving points into a numpy array for ease of ROI use later
        regular_points = alignment_utils.unembed_points_from_space(fwd_points,
                                                                   img_data_dict['functional_img'].shape[1],
                                                                   img_data_dict['functional_img'].shape[0],
                                                                   embed_size=1024)
        [ax[2].scatter(p[0], p[1], s=8, color = unique_roi_colors[n]) for p in regular_points]
        ax[2].fill([p[0] for p in regular_points], [p[1] for p in regular_points], color=unique_roi_colors[n], alpha=0.3)
        np.save(Path(roi_save_directory).joinpath(f'{roi_lst[n]}.npy'), regular_points)

    [a.axis("off") for a in ax]
    ax[0].set_title('original points')
    ax[1].set_title('forward transformed points')
    ax[2].set_title('rois in original space')
    plt.tight_layout()
    plt.savefig(Path(roi_save_directory.parents[0]).joinpath(f'alignment/ROI_masks.png')) # save this figure
    plt.show(block = True)

def main():
    parser = argparse.ArgumentParser(description="run alignment pipeline.")
    parser.add_argument("master_folder", help="Path to the master dataset folder containing plane_0, plane_1, ...")
    parser.add_argument("--scale_penalty", type=float, default=50.0, help="Scale penalty for alignment.")
    parser.add_argument("--mode", choices=["find_transform", "apply_transform"], required=True,
                        help="Mode: find_transform (plane_0) or apply_transform (all planes).")
    parser.add_argument("--override_reference_match", type=int, default=None,
                        help="Optional: manually override the matching plane number for plane_0.")
    args = parser.parse_args()

    master_path = Path(args.master_folder)
    scale_penalty = args.scale_penalty
    mode = args.mode
    override_plane0 = args.override_reference_match

    # --- FIND TRANSFORM MODE --- #
    if mode == "find_transform":
        # 1 - set up for aligning plane 0
        plane0_path = master_path / "plane_0"
        img_dict, matching_plane_num, alignment_fld = prep_for_alignment(plane0_path)

        # Apply override if given
        if override_plane0 is not None:
            print(f"[INFO] Overriding detected matching plane number: {matching_plane_num} → {override_plane0}")
            matching_plane_num = override_plane0
            # Save override to file
            override_path = alignment_fld / "override_reference_match.json"
            save_override_match_val(save_path = override_path, new_plane_match  = matching_plane_num)

        # 2 - run alignment on a couple different scale penalties, plot the best option
        reference_img = preprocess_img_for_alignment(img_dict['reference_stack'][matching_plane_num])
        embedded_reference_img = sitkalignment.embed_image(reference_img, 1024)
        target_img = preprocess_img_for_alignment(img_dict['functional_img'])
        embedded_target_img = sitkalignment.embed_image(target_img, 1024)

        scale_penalty_test_list = [20, 50, 75, 100, 150, 200]
        fwd_alignment_path = str(Path(alignment_fld).joinpath(f'fwd_transform'))
        dice_coeff_lst, best_scale_pen, best_registered_img = alignment_utils.update_test_alignment_parameters(embedded_reference_img,
                                                                                                        embedded_target_img,
                                                                                                        scale_penalities=scale_penalty_test_list,
                                                                                                        iteration_tuple=(15000, 5000),
                                                                                                        master_save_path=fwd_alignment_path,
                                                                                                        manual_select=False)

        imwrite(Path(plane0_path).joinpath('registered_img.tif'), best_registered_img, bigtiff=True)
        alignment_utils.plotting_pre_post_alignment(reference_img=sitkalignment.embed_image(img_dict['reference_stack'][matching_plane_num], 1024),
                                                    target_img=sitkalignment.embed_image(img_dict['functional_img'], 1024),
                                                    registered_target_img=best_registered_img)

    # --- APPLY TRANSFORM MODE --- #
    elif mode == "apply_transform":
        print(f"\n[INFO] Applying existing transform to all planes in {master_path}...")

        plane_step = find_z_step(master_path)

        # Load plane_0 matching plane num to adjust others
        plane0_path = master_path / "plane_0"
        _, matching_plane_num_plane0, _ = prep_for_alignment(plane0_path)

        fwd_transform_dir = plane0_path / "alignment"

        override_ref_val_path = fwd_transform_dir / "override_reference_match.json"
        if override_plane0 is None and override_ref_val_path.exists():
            matching_plane_num_plane0 = load_override_match_val(override_ref_val_path)
            print(f"[INFO] Loaded saved override plane match value: {matching_plane_num_plane0}")
        elif override_plane0 is not None:
            print(f"[INFO] Using CLI override for plane_0: {override_plane0}")
            matching_plane_num_plane0 = override_plane0

        specific_fwd_alignment_path = fwd_transform_dir / f"fwd_transform_sp_{int(scale_penalty)}"

        for plane_dir in sorted(master_path.glob("plane_*")):
            plane_num = int(plane_dir.name.split("_")[-1])
            print(f"\n[INFO] Processing {plane_dir.name}...")

            img_dict = load_data(plane_dir)
            alignment_fld = plane_dir / "alignment"
            alignment_fld.mkdir(exist_ok=True)

            # Adjust matching z-plane number based on plane offset
            matching_plane_num = matching_plane_num_plane0 - (plane_num * plane_step)
            print(f"\n[INFO] matching reference brain plane {matching_plane_num}")

            # Transform + save new ROIs
            roi_save_directory = plane_dir / "rois"
            roi_save_directory.mkdir(exist_ok=True)

            plot_and_save_new_ROIs(
                img_data_dict=img_dict,
                roi_save_directory=roi_save_directory,
                fwd_transform_path=specific_fwd_alignment_path,
                matching_z_plane=matching_plane_num
            )

            print(f"[INFO] Finished applying transform to {plane_dir.name}")

        print("\n[INFO] Alignment pipeline completed successfully.")

if __name__ == "__main__":
    main()