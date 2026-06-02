import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def create_circular_mask(img_shape, x, y, radius):
    """
    img_shape: (height, width)
    x, y: coordinates in (x, y) format — i.e., (col, row)
    """
    h, w = img_shape
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
    
    return dist_from_center <= radius

def create_polygon_mask(image_shape, polygon_coords):
    """
    Creates a binary mask for a given polygon.

    Parameters:
    - image_shape: Tuple (height, width) of the image.
    - polygon_coords: List of (x, y) tuples defining the polygon.

    Returns:
    - mask: 2D NumPy array (same size as image), where 1=inside, 0=outside.
    """

    import matplotlib.path as mpath

    # Create grid of coordinates
    h, w = image_shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # Flatten the grid
    points = np.vstack((x.ravel(), y.ravel())).T  # Shape (num_pixels, 2)

    # Create a path object
    poly_path = mpath.Path(polygon_coords)

    # Check which points are inside the polygon
    mask_flat = poly_path.contains_points(points)

    # Reshape mask back to image dimensions
    mask = mask_flat.reshape(h, w)

    return mask.astype(np.uint8)


def extract_circular_roi_trace(image_stack, x_center, y_center, diameter_um, um_per_pixel):
    """
    Extracts raw pixel values within a circular ROI across a time series.
    Like getting the raw traces for a specific stim site...
    Parameters:
    -----------
    image_stack : np.ndarray
        3D array with shape (frames, height, width)
    x_center, y_center : float
        The center coordinates of the stimulus in pixels
    diameter_um : float
        Diameter of the spot in micrometers (e.g., 5)
    um_per_pixel : float
        Spatial resolution of your imaging (e.g., 0.8)

    Returns:
    --------
    roi_data : np.ndarray
        2D array of raw pixel values with shape (frames, num_pixels_in_roi)
    """
    # 1. Convert physical diameter to pixel radius
    radius_px = (diameter_um / um_per_pixel) / 2

    # 2. Create a coordinate grid for the image dimensions
    f, h, w = image_stack.shape
    yy, xx = np.mgrid[:h, :w]

    # 3. Calculate Euclidean distance of every pixel from the center
    dist_from_center = np.sqrt((xx - x_center) ** 2 + (yy - y_center) ** 2)

    # 4. Create a boolean mask of pixels within the radius
    roi_mask = dist_from_center <= radius_px

    # 5. Extract values: image_stack[:, roi_mask] returns (frames, N_pixels)
    roi_pixel_values = image_stack[:, roi_mask]

    return roi_pixel_values, roi_mask

def extract_polygon_roi_trace(image_stack, ptlist):
    """
    Extract raw pixel values within a polygon ROI across a time series. - works when you make polygons with the draw_roi function

    Parameters
    ----------
    image_stack : np.ndarray
        3D array with shape (frames, height, width)

    ptlist : list or np.ndarray
        Polygon vertices as [(x1, y1), (x2, y2), ...]

    Returns
    -------
    roi_pixel_values : np.ndarray
        Raw pixel values with shape (frames, num_pixels_in_roi)

    roi_mask : np.ndarray
        Boolean mask of ROI with shape (height, width)
    """

    # image dimensions
    frames, height, width = image_stack.shape

    # make empty mask
    roi_mask = np.zeros((height, width), dtype=np.uint8)

    # convert points to OpenCV format
    pts = np.array(ptlist, dtype=np.int32)

    # fill polygon
    cv2.fillPoly(roi_mask, [pts], 1)

    # convert to boolean
    roi_mask = roi_mask.astype(bool)

    # extract raw pixel traces
    roi_pixel_values = image_stack[:, roi_mask]

    return roi_pixel_values, roi_mask

# making masks out of rsChrmine images (or any red channel image)

def make_red_channel_image_masks(reference_stack_path, otsu_thresh_factor = 1.1, save_mask_directory = None):
    '''
    Make masks from red channel reference images, using Otsu thresholding
    :param reference_stack_path: folder path to the reference stack for the red channel
    :param otsu_thresh_factor: factor to multiply to the otsu threshold value (increase or decrease the mask expression)
    :param save_mask_directory: folder to save the mask, titled 'rschrmine_mask.npy'
    :return: plot of the original red channel images, and the mask images
    '''
    import scipy
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects
    import sys
    sys.path.append(r'C:\Users\Kaitlyn\PyCharmProjects\imaging\caImageAnalysis')
    from bruker_images import collect_img_array_from_individual_volumes

    ch1_img_stack = collect_img_array_from_individual_volumes(reference_stack_path, n=50,
                                                                            plane_idx=None, channel='Ch1')
    ch1_img_stack = [scipy.ndimage.rotate(img, angle=90) for img in ch1_img_stack]

    fig, ax = plt.subplots(2, len(ch1_img_stack), figsize=(20, 10))
    for i in range(len(ch1_img_stack)):
        img = ch1_img_stack[i]
        ax[0, i].imshow(ch1_img_stack[i], cmap='gray', vmax=np.percentile(ch1_img_stack[i], 99))
        # make mask
        thresh_val = threshold_otsu(img)
        mask = img > (thresh_val * otsu_thresh_factor)
        mask = remove_small_objects(mask, min_size=10)  # remove small specks
        ax[1, i].imshow(mask, cmap="gray")
        # save the mask in the folder to use later
        if save_mask_directory is not None:
            np.save(Path(save_mask_directory).joinpath(f'output_folders/plane_{i}/rschrmine_mask.npy'), mask)
    [a.axis('off') for a in ax.flatten()]

    return plt.show()

def cells_per_mask(cell_dicts, mask, min_frac=0.2):
    """
    Identify which cells are overlapping with the red channel/rschrmine masks
    Here we use the masks to find the cells

    cell_dicts: list of cell dictionaries with 'ypix' and 'xpix' (stats attribute in the BaseFish class)
    mask: 2D boolean array
    min_frac: fraction of overlap to call a cell 'positive'
    Returns: list of cell_ids that are overlapping with the mask and all the fraction of overlap for the cell_dicts
    """
    overlapping_cells = []
    overlapping_fracs = []
    for i, cell in enumerate(cell_dicts):
        ypix, xpix = cell['ypix'], cell['xpix']
        overlap = mask[ypix, xpix]
        frac = overlap.mean()
        if frac >= min_frac:
            overlapping_cells.append(i)
        overlapping_fracs.append(frac)

    return overlapping_cells, overlapping_fracs


# these two next functions are essentially the same

def points_within_circle(x_center, y_center, radius):
    '''
    Finds all points within a certain radius of a center point
    x_center: x coordinate of the center point
    y_center: y coordinate of the center point
    radius: radius of the circle

    returns x_points_int, y_points_int: integers of the x and y points within the circle
    '''

    # Define a range of x and y values based on the radius
    x_range = np.arange(x_center - radius, x_center + radius + 1, 1)
    y_range = np.arange(y_center - radius, y_center + radius + 1, 1)
    
    # Collect all x_points and y_points within the radius
    x_points = []
    y_points = []
    for x in x_range:
        for y in y_range:
            if (x - x_center)**2 + (y - y_center)**2 <= radius**2:
                x_points.append(x)
                y_points.append(y)
    x_points_int = np.unique([int(x) for x in x_points])
    y_points_int = np.unique([int(y) for y in y_points])
                
    return x_points_int, y_points_int


def compute_coverage(circle_pixels, polygon_pixels):
    '''
    compute the percentage of overlap between a circle on top of a polygon
    circle_pixels: list of (x, y) tuples for the circle
    polygon_pixels: list of (x, y) tuples for the polygon
    '''
    # Convert to sets of (x, y) tuples
    circle_set = set(circle_pixels)
    polygon_set = set(polygon_pixels)

    # Intersection gives pixels shared by both shapes
    overlap = circle_set & polygon_set

    # Compute percentage
    polygon_covered_by_circle = len(overlap) / len(polygon_set) * 100
    circle_covered_by_polygon = len(overlap) / len(circle_set) * 100

    return polygon_covered_by_circle, circle_covered_by_polygon



# Functions for drawing and saving ROIs outside of the Fish class structure
def draw_roi(ref_img, savePath, title, brightness=50, contrast=30):
    # edited by chatgpt to allow me to see the line

    import cv2
    import numpy as np
    from pathlib import Path

    img_arr = np.zeros((max(ref_img.shape), max(ref_img.shape)))

    for x in np.arange(ref_img.shape[0]):
        for y in np.arange(ref_img.shape[1]):
            img_arr[x, y] = ref_img[x, y]

    plot_img = np.int16(img_arr)
    plot_img = plot_img * (contrast / 127 + 1) - contrast + brightness
    plot_img = np.clip(plot_img, 0, 255).astype(np.uint8)

    # permanent drawing image
    base_img = plot_img.copy()

    ptlist = []

    window_name = f"roiFinder_{title}"

    def roigrabber(event, x, y, flags, params):

        nonlocal base_img

        # -----------------------------
        # LEFT CLICK = add point
        # -----------------------------
        if event == cv2.EVENT_LBUTTONDOWN:

            if len(ptlist) > 0:
                cv2.line(
                    base_img,
                    ptlist[-1],
                    (x, y),
                    color=(0, 0),
                    thickness=1)

            cv2.circle(base_img, (x, y), 2, (0, 0), -1)

            ptlist.append((x, y))

            cv2.imshow(window_name, base_img)

        # -----------------------------
        # MOUSE MOVE = preview line
        # -----------------------------
        elif event == cv2.EVENT_MOUSEMOVE:

            temp_img = base_img.copy()

            if len(ptlist) > 0:
                cv2.line(
                    temp_img,
                    ptlist[-1],
                    (x, y),
                    color=(0, 0),
                    thickness=1,
                )

            cv2.imshow(window_name, temp_img)

        # -----------------------------
        # RIGHT CLICK = close polygon
        # -----------------------------
        elif event == cv2.EVENT_RBUTTONDOWN:

            if len(ptlist) > 2:
                cv2.line(
                    base_img,
                    ptlist[-1],
                    ptlist[0],
                    color=(0, 0),
                    thickness=2,
                )

            cv2.imshow(window_name, base_img)

            cv2.waitKey(300)
            cv2.destroyAllWindows()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1400, 1400)

    cv2.setMouseCallback(window_name, roigrabber)

    cv2.imshow(window_name, base_img)

    try:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        cv2.destroyAllWindows()

    save_roi(Path(savePath), title, ptlist)

def draw_roi_original(ref_img, savePath, title, brightness=50, contrast=30):

    img_arr = np.zeros((max(ref_img.shape), max(ref_img.shape)))

    for x in np.arange(ref_img.shape[0]):
        for y in np.arange(ref_img.shape[1]):
            img_arr[x, y] = ref_img[x, y]

    plot_img = np.int16(img_arr)
    plot_img = plot_img * (contrast / 127 + 1) - contrast + brightness
    plot_img = np.clip(plot_img, 0, 255).astype(np.uint8)

    ptlist = []

    def roigrabber(event, x, y, flags, params):
        if event == 1:  # left click
            if len(ptlist) == 0:
                cv2.line(plot_img, pt1=(x, y), pt2=(x, y), color=(255, 255), thickness=3)
            else:
                cv2.line(
                    plot_img,
                    pt1=(x, y),
                    pt2=ptlist[-1],
                    color=(255, 255),
                    thickness=3,
                )

            ptlist.append((x, y))
        if event == 2:  # right click
            cv2.destroyAllWindows()

    cv2.namedWindow(f"roiFinder_{title}")

    cv2.setMouseCallback(f"roiFinder_{title}", roigrabber)

    cv2.imshow(f"roiFinder_{title}", np.array(plot_img, "uint8"))
    try:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        cv2.destroyAllWindows()

    save_roi(Path(savePath), title, ptlist)


def save_roi(savePath, save_name, ptlist):
    '''
    Save ROI as a .npy file in the rois folder
    savePath: where to save the npy file
    save_name: name of the file
    '''

    savePathFolder = savePath.joinpath("rois")
    if not os.path.exists(savePathFolder):
        os.mkdir(savePathFolder)

    savePath = savePathFolder.joinpath(f"{save_name}.npy")
    np.save(savePath, ptlist)
    print(f"saved {save_name}")


"""
Example of how to use these functions:

load roi:
roi_pts = np.load(roi_path)

plot roi:
import matplotlib.path as mpltPath
import matplotlib.patches as patches

path = mpltPath.Path(roi_pts)
coords = path.to_polygons()
ax.fill([i[1] for i in coords[0]], [i[0] for i in coords[0]], color='white', alpha=0.5)

check points:
path.contains_points(points)

"""


class MultiPlaneCellSelector:
    """
    Interactive selection of cells across multiple planes with optional overlays.
    """

    def __init__(self, masks, overlay_points=None, overlay_colors=None):
        """
        Parameters
        ----------
        masks : list of 2D numpy arrays
            Each array is a binary mask or image for a plane.
        overlay_points : list of lists of (x, y) tuples, optional
            Points to overlay on each plane (same length as masks).
        overlay_colors : list of lists of str, optional
            Colors for each overlay point.
        """
        self.masks = masks
        self.n_planes = len(masks)
        self.overlay_points = overlay_points if overlay_points is not None else [None] * self.n_planes
        self.overlay_colors = overlay_colors if overlay_colors is not None else [None] * self.n_planes
        self.selected_cells = []

    def select_cells(self, n_cells=None, title="Select cells"):
        """
        Interactive matplotlib window to select cells across all planes.

        Parameters
        ----------
        n_cells : int or None
            If set, stops after exactly this many clicks.
            If None, selection continues until Enter is pressed.
        title : str
            Window title.

        Returns
        -------
        selected_cells : list of (plane_id, x, y)
            List of selected coordinates with plane index.
        """
        fig, axes = plt.subplots(1, self.n_planes, figsize=(6 * self.n_planes, 6))
        if self.n_planes == 1:
            axes = [axes]

        fig.suptitle(title)

        # Show each plane
        for i, ax in enumerate(axes):
            ax.imshow(self.masks[i], cmap="gray", vmax = np.percentile(self.masks[i], 99))
            ax.set_title(f"Plane {i}")
            ax.axis("off")

            # Add overlays if provided
            if self.overlay_points[i] is not None:
                colors = self.overlay_colors[i] if self.overlay_colors[i] is not None else ["limegreen"] * len(
                    self.overlay_points[i])
                for (pt, clr) in zip(self.overlay_points[i], colors):
                    ax.scatter(pt[0], pt[1], color=clr, s=20, alpha=0.8)

        print("Click on cells. Press Enter to finish.")

        def onclick(event):
            print("CLICK EVENT FIRED")
            if event.inaxes is None:  # clicked outside axes
                return

            if event.inaxes in axes:
                plane_id = list(axes).index(event.inaxes)
                if event.xdata is None or event.ydata is None:
                    return
                x, y = int(event.xdata), int(event.ydata)
                self.selected_cells.append((x, y, plane_id))
                event.inaxes.scatter(x, y, color="red", s=80, edgecolor="k")
                fig.canvas.draw()

                if n_cells is not None and len(self.selected_cells) >= n_cells:
                    plt.close(fig)

        cid = fig.canvas.mpl_connect("button_press_event", onclick)
        plt.show()

        # After closing interactive window, switch backend back to inline
        # try:
        #     import IPython
        #     ipython = IPython.get_ipython()
        #     if ipython is not None:
        #         ipython.run_line_magic("matplotlib", "inline")
        # except Exception as e:
        #     print("Could not reset backend:", e)

        return self.selected_cells
