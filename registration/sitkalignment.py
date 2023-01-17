import SimpleITK as sitk
import numpy as np

import os


def load_image(image_path):
    """
    a Kaitlyn-proof solution

    :param image_path: raw str path or Path should work fine
    :return: image
    """
    from tifffile import imread

    return imread(image_path)


def embed_image(image, default_size=1024):
    if image.ndim == 3:
        print("please input 2D image")
        return

    while max(image.shape) >= default_size:
        default_size *= 2
        print(f"increasing default size to {default_size}")

    new_image = np.zeros((default_size, default_size))
    midpt = default_size // 2

    image = np.clip(image, a_min=0, a_max=2**16)

    offset_y = 0
    ydim = image.shape[0]
    if ydim % 2 != 0:  # if its odd kick it one pixel
        offset_y += 1
    offset_x = 0
    xdim = image.shape[1]
    if xdim % 2 != 0:  # if its odd kick it one pixel
        offset_x += 1

    new_image[
        midpt - ydim // 2 : midpt + ydim // 2 + offset_y,
        midpt - xdim // 2 : midpt + xdim // 2 + offset_x,
    ] = image

    return new_image


def trim_image(image):
    """
    uses opencv to get a point list and delete data outside roi

    :param image: as array
    :return: trimmed image array
    """
    import cv2

    if image.ndim == 3:
        image_slice = image[image.shape[0] // 4].copy()
    else:
        image_slice = image.copy()

    list_of_points = []

    def roi_grabber(event, x, y, flags, params):
        if event == 1:  # left click
            list_of_points.append((x, y))
        if event == 2:  # right click
            cv2.destroyAllWindows()

    cv2.namedWindow(f"roi_finding_window")
    cv2.setMouseCallback(f"roi_finding_window", roi_grabber)
    cv2.imshow(f"roi_finding_window", np.array(image_slice, "uint8"))
    try:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        cv2.destroyAllWindows()

    image_mask = np.zeros(image_slice.shape, dtype="int32")
    image_mask = cv2.fillPoly(image_mask, np.int32([list_of_points]), 1, 255)
    if image.ndim == 3:
        images = [np.ma.masked_where(image_mask != 1, i).filled(0) for i in image]
        return np.array(images)
    else:
        return np.ma.masked_where(image_mask != 1, image).filled(0)


def estimate_transform_itk(moving, fixed, tx):
    from SimpleITK import GetImageFromArray
    moving_ = GetImageFromArray(moving.astype('float32'))
    fixed_ = GetImageFromArray(fixed.astype('float32'))
    return tx.Execute(moving_, fixed_)


def calculate_match_value(image_reference, image_target):

    def_size = 1024
    while max(max(image_reference.shape, image_target.shape)) >= def_size:
        def_size *= 2
    image_target = embed_image(image_target, def_size)
    image_reference = embed_image(image_reference, def_size)

    reference_image = sitk.GetImageFromArray(image_reference)
    align_image = sitk.GetImageFromArray(image_target)

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(reference_image)
    elastixImageFilter.SetMovingImage(align_image)

    param_map = sitk.GetDefaultParameterMap("rigid")
    param_map["MaximumNumberOfIterations"] = ["512"]
    elastixImageFilter.SetParameterMap(param_map)

    pmap = sitk.GetDefaultParameterMap("bspline")
    pmap["MaximumNumberOfIterations"] = ["128"]
    pmap["Metric0Weight"] = ["0.1"]
    pmap["Metric1Weight"] = ["20"]
    elastixImageFilter.AddParameterMap(pmap)

    elastixImageFilter.LogToConsoleOn()
    elastixImageFilter.Execute()
    res = elastixImageFilter.GetResultImage()

    r = sitk.ImageRegistrationMethod()
    r.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    r.SetOptimizerAsLBFGSB(maximumNumberOfCorrections=3, numberOfIterations=250)
    r.SetMetricSamplingStrategy(r.RANDOM)
    r.SetMetricSamplingPercentage(0.5)
    tx = sitk.TranslationTransform(2)

    r.SetInitialTransform(tx)
    r.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2])
    r.SetSmoothingSigmasPerLevel(smoothingSigmas=[3, 1])

    res_img = sitk.GetArrayFromImage(res)

    tx = estimate_transform_itk(image_reference, res_img, r)
    return r.GetMetricValue()


def register_image(image_reference, image_target, savepath=None):
    """

    :param image_reference:
    :param image_target:
    :param savepath: directory
    :return:
    """

    def_size = 1024
    while max(max(image_reference.shape, image_target.shape)) >= def_size:
        def_size *= 2
    image_target = embed_image(image_target, def_size)
    image_reference = embed_image(image_reference, def_size)

    reference_image = sitk.GetImageFromArray(image_reference)
    align_image = sitk.GetImageFromArray(image_target)

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(reference_image)
    elastixImageFilter.SetMovingImage(align_image)

    pmap = sitk.GetDefaultParameterMap('rigid')
    pmap['MaximumNumberOfIterations'] = ['4096']
    elastixImageFilter.SetParameterMap(pmap)
    # elastixImageFilter.AddParameterMap(pmap)

    pmap = sitk.GetDefaultParameterMap("bspline")
    pmap['MaximumNumberOfIterations'] = ['4096']
    pmap['Metric0Weight'] = ['0.1']
    pmap['Metric1Weight'] = ['10']
    elastixImageFilter.AddParameterMap(pmap)

    elastixImageFilter.LogToConsoleOn()
    elastixImageFilter.Execute()
    res = elastixImageFilter.GetResultImage()

    if savepath:
        from pathlib import Path

        pmaps = elastixImageFilter.GetTransformParameterMap()

        for n, pmap in enumerate(pmaps):
            sitk.WriteParameterFile(pmap, Path(savepath).joinpath(f'transform_pmap_{n}.txt'))

    return sitk.GetArrayFromImage(res)


def transform_image_from_saved(image, savepath):
    align_image = sitk.GetImageFromArray(image)

    pmap_files = []
    with os.scandir(savepath) as entries:
        for entry in entries:
            if "transform_pmap" in entry.name:
                pmap_files.append(entry.path)

    pmap0 = sitk.ReadParameterFile(pmap_files[0])

    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetTransformParameterMap(pmap0)

    for pmap_file in pmap_files[1:]:
        pmap = sitk.ReadParameterFile(pmap_file)
        transformixImageFilter.AddTransformParameterMap(pmap)

    transformixImageFilter.SetMovingImage(align_image)
    transformixImageFilter.Execute()
    res = transformixImageFilter.GetResultImage()

    return sitk.GetArrayFromImage(res)


def find_best_z_match(stack_reference, image_target, rigorous=False):
    """

    :param stack_reference: 3d image stack to align image target to
    :param image_target: target image array (2d)
    :param rigorous: if you have an abundance of time this can be true
    :return: Z index of reference stack and results dict
    """
    return
