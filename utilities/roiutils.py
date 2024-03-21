import cv2
import os

import numpy as np

from pathlib import Path

def create_circular_mask(img_shape, x, y, radius):
    '''
    Creating a circular mask given a x, y coordinate point and radius of the spot over a raw pixel image
    '''
    h,w = img_shape
    Y, X = np.ogrid[:h, :w]
    
    dist_from_center = np.sqrt((X - x)**2 + (Y-y)**2)

    mask = dist_from_center <= radius
    return mask

# Functions for drawing and saving ROIs outside of the Fish class structure

def draw_roi(ref_img, savePath, title):

    img_arr = np.zeros((max(ref_img.shape), max(ref_img.shape)))

    for x in np.arange(ref_img.shape[0]):
        for y in np.arange(ref_img.shape[1]):
            img_arr[x, y] = ref_img[x, y]

    ptlist = []

    def roigrabber(event, x, y, flags, params):
        if event == 1:  # left click
            if len(ptlist) == 0:
                cv2.line(img_arr, pt1=(x, y), pt2=(x, y), color=(255, 255), thickness=3)
            else:
                cv2.line(
                    img_arr,
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

    cv2.imshow(f"roiFinder_{title}", np.array(ref_img, "uint8"))
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

class clickReturner:
    '''
    Use to click a certain point on the image and return the coordinates of the point
    '''
    def __init__(self, img):
        self.img = img
        self.ptlist = []
        
    def click(self, title="blank"):
        import cv2

        img_arr = np.zeros((max(self.img.shape), max(self.img.shape)))

        for x in np.arange(self.img.shape[0]):
            for y in np.arange(self.img.shape[1]):
                img_arr[x, y] = self.img[x, y]

        def roigrabber(event, x, y, flags, params):
            if event == 1:  # left click
                cv2.line(self.img, pt1=(x, y), pt2=(x, y), color=(255, 255), thickness=3)
                self.ptlist.append((x, y))
            if event == 2:  # right click
                cv2.destroyAllWindows()

        cv2.namedWindow(f"roiFinder_{title}")

        cv2.setMouseCallback(f"roiFinder_{title}", roigrabber)

        cv2.imshow(f"roiFinder_{title}", self.img)
        try:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            cv2.destroyAllWindows()
        return 

    def get_pt(self):
        return self.ptlist[-1]
