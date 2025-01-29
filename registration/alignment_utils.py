# Making utility functions to help run sitk registration processing #
import numpy as np
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# local imports
from . import sitkalignment

import sys
sys.path.append(r'C:\Users\NaumannLab_KEF\PyCharmProjects\imaging')
from scopeslip.planeAlignment import PlaneAlignment

def test_alignment_parameters(reference_img, target_img, scale_penalities = [20, 50, 100], 
                              iteration_tuple = (5000, 5000), master_save_path = None, plot = True):
    '''
    Test different scale penalities for sitk registration and plot the results, gather the different dice coefficients
    reference_img: np.array (2D), reference image
    target_img: np.array (2D), target image (what you are going to warp)
    scale_penalities: list of ints, scale penalities to test
    iteration_number: int, number of iterations to run the registration
    master_save_path: str, path to save the aligned output
    plot: bool, whether to plot the results

    returns: 
    dice_coeff_lst: list of floats, dice coefficients for each scale penalty
    best_scale_pen: int, best scale penalty
    best_registered_img: np.array (2D), best registered image
    '''
    dice_coeff_lst = []
    registered_img_lst = []
    for sp in scale_penalities:
        save_alignment_path = master_save_path + f'_sp_{sp}'
        print(f'scale penalty = {sp}')
        registered_img = sitkalignment.register_image2(reference_img, target_img, savepath = save_alignment_path, 
                                                        scalePenalty=sp, iterations=iteration_tuple)
        registered_img_lst.append(registered_img)

        if plot:
            fig, ax = plt.subplots( 1,3, figsize=(12,6))
            
            ax[0].imshow(
                reference_img,
                cmap="gray",
                vmax=np.percentile(reference_img, 97),
                vmin=np.percentile(reference_img, 30),
            )
            ax[1].imshow(
                registered_img,
                cmap="gray",
                vmax=np.percentile(registered_img, 97),
                vmin=np.percentile(registered_img, 30),
            )
            
            [a.axis("off") for a in ax]
            
            merge = np.zeros(
                (reference_img.shape[0], reference_img.shape[1], 3)
            )  # assumes same size images
            merge[:, :, 0] = 2 * reference_img / reference_img.max()
            merge[:, :, 1] = 2 * registered_img / registered_img.max()
            ax[-1].imshow(merge)
            
            ax[0].set_title('Reference')
            ax[1].set_title('Target')
            ax[-1].set_title("merge")
            plt.tight_layout()
            plt.show()
    
        # compute similarity between the reference and the registered image
        pa_class = PlaneAlignment(target = reference_img, stack = registered_img, method = 'otsu')
        dice_coeff_lst.append(pa_class.lossReturn()) # gives the loss return of these different penalties

    best_img_ind  = np.where(dice_coeff_lst == np.max(dice_coeff_lst))[0][0]
    best_scale_pen = scale_penalities[best_img_ind]
    best_registered_img = registered_img_lst[best_img_ind]
    
    return dice_coeff_lst, best_scale_pen, best_registered_img

def extract_mask_boundaries(image_array, points_per_contour=20):
    """
    Extracts boundary points from a mask in a TIFF file.

    Args:
        image_array (array): 2D array of pixel values of your mask image
        points_per_contour: int, number of points to extract per contour

    Returns:
        list: A list of boundary points [(x1, y1), (x2, y2), ...].
        np.ndarray: The binary mask used to extract contours.
        int: Number of contours found.
    """


    # Step 2: Rescale pixel values to 0-255
    scaled_mask = (image_array / image_array.max()) * 255  # Normalize and scale
    scaled_mask = scaled_mask.astype(np.uint8)  # Convert to 8-bit integers

    # Step 3: Convert to binary mask
    _, binary_mask = cv2.threshold(scaled_mask, 127, 255, cv2.THRESH_BINARY)

    # Step 4: Find contours (boundaries)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Step 5: Extract points along the boundaries
    points = []
    if contours:
        for contour in contours:
            contour_length = len(contour)  # Number of points in the contour
            if contour_length == 0:
                continue
            # Get evenly spaced indices along the contour
            indices = np.linspace(0, contour_length - 1, points_per_contour, dtype=int)

            # Collect points based on these indices
            contour_points = [list(contour[idx][0]) for idx in indices]
            points.extend(contour_points)

    return np.array(points), binary_mask, len(contours)

def unembed_points_from_space(points, original_width, original_height, embed_size=1024):
    """
    Un-embeds points from a square space of size embed_size x embed_size
    back to the original coordinate space.

    Args:
        points (list): List of points [(x1, y1), (x2, y2), ...].
        original_width (int): Original width of the mask.
        original_height (int): Original height of the mask.
        embed_size (int): Size of the embedding space (default: 1024).

    Returns:
        list: List of un-embedded points [(x1, y1), (x2, y2), ...].
    """
    # Calculate the center of the embedding space
    midpt = embed_size // 2

    # Calculate offsets due to odd dimensions
    y_offset = original_height % 2
    x_offset = original_width % 2

    # Map points back to the original space, accounting for the offsets
    unembedded_points = [
        (
            x - (midpt - original_width // 2) - x_offset,  # Reverse x centering with offset
            y - (midpt - original_height // 2) - y_offset  # Reverse y centering with offset
        )
        for x, y in points
    ]

    return unembedded_points

def process_image_for_alignment(image):
    '''
    Pre processing images for alignment, works well for Cytosolic Gcamp images with Mapzebrain Ref Images
    image: np.array, image to process

    return np.array, processed image
    '''

    from scipy.ndimage import gaussian_filter

    # step 1 - gaussian smoothing
    smoothed_img = gaussian_filter(image, sigma=5)

    # step 2 - linear contrast
    min_val, max_val = smoothed_img.min(), smoothed_img.max()
    scaled_img = ((smoothed_img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    return scaled_img

def minmax_scaler(arr, vmin=0, vmax=1):
    '''
    Scales images by adjusting the min and max values
    arr: np.array, image to scale
    vmin: int, min value
    vmax: int, max value

    return np.array, scaled image
    '''
    arr_min, arr_max = arr.min(), arr.max()
    return ((arr - arr_min) / (arr_max - arr_min)) * (vmax - vmin) + vmin

def plotting_pre_post_alignment(reference_img, target_img, registered_target_img):
    '''
    # Plotting pre and post alignment images
    reference_img: np.array, reference image
    target_img: np.array, target image
    registered_target_img: np.array, registered (aligned) target image

    return: plot
    '''

    fig, ax = plt.subplots(2, 3, figsize=(12,6))

    # pre alignment
    ax[0, 0].imshow(reference_img, cmap="gray", vmax=np.percentile(reference_img, 97), vmin=np.percentile(reference_img, 30),)
    ax[0, 1].imshow(target_img,cmap="gray", vmax=np.percentile(target_img, 97), vmin=np.percentile(target_img, 30),)
    ax[0, 0].set_title('Pre Reference')
    ax[0, 1].set_title('Pre Target')
    ax[0, 2].set_title("Pre Merge")

    pre_merge = np.zeros((reference_img.shape[0], reference_img.shape[1], 3))  # assumes same size images
    pre_merge[:, :, 0] = 2 * reference_img / reference_img.max()
    pre_merge[:, :, 1] = 2 * target_img / target_img.max()
    ax[0, 2].imshow(pre_merge)

    # post alignment
    ax[1, 0].imshow(reference_img, cmap="gray", vmax=np.percentile(reference_img, 97), vmin=np.percentile(reference_img, 30),)
    ax[1, 1].imshow(registered_target_img,cmap="gray", vmax=np.percentile(registered_target_img, 97), vmin=np.percentile(registered_target_img, 30),)
    ax[1, 0].set_title('Post Reference')
    ax[1, 1].set_title('Post Target')

    post_merge = np.zeros((reference_img.shape[0], reference_img.shape[1], 3))  # assumes same size images
    post_merge[:, :, 0] = 2 * reference_img / reference_img.max()
    post_merge[:, :, 1] = 2 * registered_target_img / registered_target_img.max()
    ax[1, 2].imshow(post_merge)
    ax[1, 2].set_title("Post Merge")

    [a.axis("off") for a in ax[0,:]]
    [a.axis("off") for a in ax[1,:]]
    plt.tight_layout()
    
    return plt.show()

def map_points_back(points, angle, image_shape):
    """
    Map points from rotated image back to the original image space.

    Args:
        points: List of (x', y') coordinates in the rotated image.
        angle: Rotation angle (in degrees).
        image_shape: Shape of the original image (height, width).

    Returns:
        List of (x, y) coordinates in the original image space.
    """
    # Convert angle to radians
    theta = np.radians(angle)
    cos_theta = np.cos(-theta)
    sin_theta = np.sin(-theta)
    
    # Center of rotation
    center = np.array(image_shape[::-1]) / 2  # (x_c, y_c)

    # Apply inverse rotation
    original_points = []
    for x_prime, y_prime in points:
        # Translate point to origin
        x_diff = x_prime - center[0]
        y_diff = y_prime - center[1]
        
        # Apply inverse rotation matrix
        x = cos_theta * x_diff - sin_theta * y_diff + center[0]
        y = sin_theta * x_diff + cos_theta * y_diff + center[1]
        
        original_points.append((x, y))
    
    return original_points

