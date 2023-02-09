import cv2
import os

import numpy as np

from pathlib import Path


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

            ptlist.append((y, x))
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

    savePathFolder = savePath.joinpath("rois")
    if not os.path.exists(savePathFolder):
        os.mkdir(savePathFolder)

    savePath = savePathFolder.joinpath(f"{save_name}.npy")
    np.save(savePath, ptlist)
    print(f"saved {save_name}")


"""
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
