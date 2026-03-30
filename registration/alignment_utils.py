# Making utility functions to help run sitk registration processing #
import numpy as np
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# local imports
from . import sitkalignment

import sys
sys.path.append(r'C:\Users\Kaitlyn\PyCharmProjects\imaging')
from scopeslip.planeAlignment import PlaneAlignment

## ALIGNMENT UTILS FOR FORWARD ALIGNMENT (FROM ZF DC PAPER)
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

def update_test_alignment_parameters(reference_img,
                                        target_img,
                                        scale_penalities=[20, 50, 100],
                                        iteration_tuple=(5000, 5000),
                                        master_save_path=None,
                                        plot=True,
                                        manual_select=True
                                    ):
    """
    Test different scale penalties for sitk registration and plot the results.
    Optionally lets the user pick the best one by eye.
    To be used with the automated analysis script
    """
    dice_coeff_lst = []
    registered_img_lst = []

    print("\n--- Testing scale penalties ---")
    for sp in scale_penalities:
        print(f"Running registration with scale penalty = {sp}")
        save_alignment_path = master_save_path + f'_sp_{sp}'
        registered_img = sitkalignment.register_image2(
            reference_img, target_img, savepath=save_alignment_path,
            scalePenalty=sp, iterations=iteration_tuple
        )
        registered_img_lst.append(registered_img)

        # Compute Dice (optional)
        pa_class = PlaneAlignment(target=reference_img, stack=registered_img, method='otsu')
        dice = pa_class.lossReturn()
        dice_coeff_lst.append(dice)

        if plot:
            fig, ax = plt.subplots(1, 3, figsize=(12, 6))
            ax[0].imshow(reference_img, cmap="gray",
                         vmax=np.percentile(reference_img, 97),
                         vmin=np.percentile(reference_img, 30))
            ax[1].imshow(registered_img, cmap="gray",
                         vmax=np.percentile(registered_img, 97),
                         vmin=np.percentile(registered_img, 30))

            merge = np.zeros((*reference_img.shape, 3))
            merge[..., 0] = 2 * reference_img / reference_img.max()
            merge[..., 1] = 2 * registered_img / registered_img.max()
            ax[2].imshow(merge)

            for a in ax: a.axis("off")
            ax[0].set_title("Reference")
            ax[1].set_title(f"Registered (scale penalty={sp})")
            ax[2].set_title("Merge")
            plt.tight_layout()
            plt.savefig(Path((str(save_alignment_path) + '.png')))
            plt.show(block=False)
            plt.close(fig)

            print(f"Displayed scale penalty {sp}, Dice={dice:.3f}")

    # --- User manually picks best image ---
    if manual_select:
        print("\nAll scale penalty tests complete.")
        print("Dice coefficients:")
        for sp, dice in zip(scale_penalities, dice_coeff_lst):
            print(f"  {sp}: {dice:.3f}")

        while True:
            try:
                user_choice = input(f"\nEnter preferred scale penalty from {scale_penalities}: ").strip()
                user_choice = int(user_choice)
                if user_choice in scale_penalities:
                    best_img_ind = scale_penalities.index(user_choice)
                    print(f"✅ You selected scale penalty {user_choice}")
                    break
                else:
                    print("Invalid input, please enter one of the tested values.")
            except ValueError:
                print("Please enter a valid integer.")
    else:
        # Automatic selection by Dice score
        best_img_ind = int(np.argmax(dice_coeff_lst))
        user_choice = scale_penalities[best_img_ind]
        print(f"Automatically selected best scale penalty by Dice: {user_choice}")

    best_scale_pen = user_choice
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

def update_extract_mask_boundaries(image_array, points_per_contour=200, merge_contours=True):
    """
    Extracts ordered boundary outlines from a mask image using spline fitting,
    supporting multiple disconnected components.

    Args:
        image_array (array): 2D array of pixel values of your mask image.
        points_per_contour (int): number of points to extract per contour.

    Returns:
        list[np.ndarray]: List of ordered boundary arrays [(x1, y1), (x2, y2), ...] per contour.
        np.ndarray: Binary mask used to extract contours.
        int: Number of contours found.
    """
    import numpy as np
    import cv2
    from scipy.interpolate import splprep, splev

    # Step 1: Normalize and convert to 8-bit
    scaled_mask = ((image_array - image_array.min()) /
                   (image_array.max() - image_array.min()) * 255).astype(np.uint8)

    # Step 2: Binarize using Otsu's method
    _, binary_mask = cv2.threshold(scaled_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 3: Connected components to separate each distinct object
    num_labels, labels = cv2.connectedComponents(binary_mask)

    ordered_contours = []

    # Step 4: Extract contour from each connected region
    for label in range(1, num_labels):  # skip background (label 0)
        mask_i = np.uint8(labels == label) * 255
        contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        # Use the largest contour for this component
        contour = max(contours, key=cv2.contourArea)
        contour = contour[:, 0, :]  # shape (N, 2)

        x, y = contour[:, 0], contour[:, 1]

        # Step 5: Fit a spline through contour points to order them smoothly
        try:
            tck, u = splprep([x, y], s=2.0, per=True)
            u_new = np.linspace(0, 1, points_per_contour)
            x_new, y_new = splev(u_new, tck)
            ordered_points = np.vstack([x_new, y_new]).T
        except Exception as e:
            # print(f"Spline fitting failed for component {label}, returning raw contour:", e)
            ordered_points = contour

        ordered_contours.append(ordered_points.astype(int))

    # Step 6: Optionally merge all contours into one combined boundary
    if merge_contours and len(ordered_contours) > 1:
        all_points = np.vstack(ordered_contours)
        # Compute convex hull to make a clean merged outline
        hull = cv2.convexHull(all_points)
        hull = hull[:, 0, :]  # (N, 2)

        x, y = hull[:, 0], hull[:, 1]
        try:
            tck, u = splprep([x, y], s=2.0, per=True)
            u_new = np.linspace(0, 1, points_per_contour)
            x_new, y_new = splev(u_new, tck)
            merged_contour = np.vstack([x_new, y_new]).T.astype(int)
        except Exception as e:
            print("Spline fitting failed for merged contour, returning raw hull:", e)
            merged_contour = hull.astype(int)

        return merged_contour, binary_mask, len(ordered_contours)

    # Step 7: Return separate contours if not merged
    return ordered_contours, binary_mask, len(ordered_contours)

def subtract_overlapping_polygons(poly1_points, poly2_points, verbose=False):
    """
    Removes overlap between two polygons so that poly2 overrides poly1 in any overlapping region.
    Most helpful for nmlf and ahb polygons
    Parameters
    ----------
    poly1_points : array-like
        Nx2 array of (x, y) coordinates for the first polygon.
    poly2_points : array-like
        Mx2 array of (x, y) coordinates for the second polygon.
    verbose : bool, optional
        If True, prints debug information.

    Returns
    -------
    new_poly1_arr : np.ndarray
        Cleaned coordinates of polygon 1 (after removing overlap with polygon 2).
    new_poly2_arr : np.ndarray
        Cleaned coordinates of polygon 2.
    """
    from shapely.geometry import Polygon

    # Convert to shapely polygons
    poly1 = Polygon(poly1_points)
    poly2 = Polygon(poly2_points)

    # Fix invalid polygons
    if not poly1.is_valid:
        if verbose:
            print("Fixing Polygon 1...")
        poly1 = poly1.buffer(0)
    if not poly2.is_valid:
        if verbose:
            print("Fixing Polygon 2...")
        poly2 = poly2.buffer(0)

    # Subtract overlapping area (poly2 overrides poly1)
    poly1_clean = poly1.difference(poly2)
    poly2_clean = poly2

    # Optionally print area stats
    if verbose:
        overlap_area = poly1.intersection(poly2).area
        print(f"Overlap area: {overlap_area:.2f}")
        print(f"Poly1 area before: {poly1.area:.2f}, after cleaning: {poly1_clean.area:.2f}")

    # Convert shapely geometry → numpy arrays
    def poly_to_array(poly):
        """Handles Polygon or MultiPolygon → single array of integer coords."""
        if poly.is_empty:
            return np.empty((0, 2), dtype=int)
        if poly.geom_type == "Polygon":
            coords = np.array(poly.exterior.coords)
        elif poly.geom_type == "MultiPolygon":
            # Take the largest piece if multiple remain
            largest = max(poly.geoms, key=lambda g: g.area)
            coords = np.array(largest.exterior.coords)
        else:
            raise TypeError(f"Unexpected geometry type: {poly.geom_type}")
        return np.round(coords).astype(int)

    new_poly1_arr = poly_to_array(poly1_clean)
    new_poly2_arr = poly_to_array(poly2_clean)

    return new_poly1_arr, new_poly2_arr

## ALIGNMENT UTILS FOR REVERSE ALIGNMENT (FOR OMR2STIM DATASETS) ##

def get_random_points_on_img(test_img, num_points = 25):
    '''
    get some random points across an image for testing purposes
    '''
    ys, xs = np.nonzero(test_img)
    idx = np.random.choice(len(xs), size=num_points, replace=False)
    random_points = np.column_stack((xs[idx], ys[idx]))  # (x, y)
    return random_points

def get_points_from_functional_to_mapzebrain(points, functional_img_dimensions, reference_img_dimensions, embed_space, transform_path):
    '''
    get points from the functional original space to the mapzebrain reference space

    '''

    embed_func_points = embed_points_to_space(points,
                                                functional_img_dimensions[1],
                                                functional_img_dimensions[0],
                                                embed_size=embed_space)
    embed_ref_points = sitkalignment.transform_points(Path(transform_path), embed_func_points)
    ref_points = unembed_points_from_space(embed_ref_points,
                                           reference_img_dimensions[1],
                                           reference_img_dimensions[0],
                                           embed_size=embed_space)
    return ref_points

## GET POINTS POST ALIGNMENT ##

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

def embed_points_to_space(points, original_width, original_height, embed_size=1024):
    """
    Embeds points from original coordinate space into a square embedding space.

    Args:
        points (list): List of points [(x1, y1), (x2, y2), ...]
        original_width (int): Original width of the mask
        original_height (int): Original height of the mask
        embed_size (int): Size of embedding space (default 1024)

    Returns:
        list: List of embedded points [(x1, y1), (x2, y2), ...]
    """
    midpt = embed_size // 2

    # same offsets you used during embedding
    y_offset = original_height % 2
    x_offset = original_width % 2

    embedded_points = [
        (
            x + (midpt - original_width // 2) + x_offset,
            y + (midpt - original_height // 2) + y_offset
        )
        for x, y in points
    ]

    return embedded_points


## PREPROCESSING IMAGES FOR ALIGNMENT ##

def get_brain_std_img(folder_path):
    '''
    get the brain mask over the std img - IMPORTANT FOR WHOLE BRAIN IMAGES
    '''
    from skimage.draw import polygon

    std_img = np.load(folder_path.joinpath('std_img.npy'))
    brain_roi = np.load(folder_path.joinpath('rois/brain.npy'))
    brain_mask = np.zeros(std_img.shape, dtype=bool) # create a brain mask to block out the extra shapes in my functional image
    rr, cc = polygon(brain_roi[:, 1], brain_roi[:, 0], shape=std_img.shape)  # NOTE: (row=y, col=x)
    brain_mask[rr, cc] = True
    std_img_brain = np.where(brain_mask, std_img, 0)

    return std_img_brain

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


## RANDOM POSTPROCESSING UTILS ##
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

def resize_and_center_crop(img, target_size=512):
    """Resize so the smallest side = target_size, then center-crop to target_size×target_size.
    Necessary when going to run cross correlation between the functional image and the reference stack for finding the best match in Z
    Note: works for 512 x 512 images for now, need to troublshoot if it will work for new dimensions
    :param img: np.array
    :param target_size: int (square dimensions that you are changing the image to fit)
    :return: np.array (the cropped image)
    """
    h, w = img.shape[:2]

    # Compute scale factor
    scale = target_size / min(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # Resize
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Center crop
    x_start = (new_w - target_size) // 2
    y_start = (new_h - target_size) // 2

    img_cropped = img_resized[y_start:y_start + target_size, x_start:x_start + target_size]

    return img_cropped

