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
    """
    puts images inside a black cube - standardizes size and such (helpful sometimes)
    :param image:
    :param default_size:
    :return:
    """
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

    moving_ = GetImageFromArray(moving.astype("float32"))
    fixed_ = GetImageFromArray(fixed.astype("float32"))
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


def register_image(
    image_reference, image_target, savepath=None, embed=False, scalePenalty=10
):
    """

    :param image_reference:
    :param image_target:
    :param savepath: directory
    :param embed: make embedding image optional
    :param scalePenalty: 10 - default, lower is squishier, higher is more rigid
    :return:
    """
    if embed:
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

    pmap = sitk.GetDefaultParameterMap("rigid")
    pmap["MaximumNumberOfIterations"] = ["4096"]
    elastixImageFilter.SetParameterMap(pmap)

    pmap = sitk.GetDefaultParameterMap("bspline")
    pmap["MaximumNumberOfIterations"] = ["4096"]
    pmap["Metric0Weight"] = ["0.1"]
    pmap["Metric1Weight"] = [str(scalePenalty)]
    elastixImageFilter.AddParameterMap(pmap)

    elastixImageFilter.LogToConsoleOn()
    elastixImageFilter.Execute()
    res = elastixImageFilter.GetResultImage()

    if savepath:
        from pathlib import Path

        pmaps = elastixImageFilter.GetTransformParameterMap()

        for n, pmap in enumerate(pmaps):
            sitk.WriteParameterFile(
                pmap, Path(savepath).joinpath(f"transform_pmap_{n}.txt").as_posix()
            )

    return sitk.GetArrayFromImage(res)


def register_image2(
    image_reference,
    image_target,
    savepath=None,
    embed=False,
    scalePenalty=10,
    iterations=(512, 512),
):
    """
    faster but less clear how well it works :)
    beta version

    :return:
    """
    if embed:
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

    parameterMapVector = sitk.VectorOfParameterMap()
    pmap1 = sitk.GetDefaultParameterMap("affine")
    pmap1["MaximumNumberOfIterations"] = [str(iterations[0])]
    parameterMapVector.append(pmap1)

    pmap2 = sitk.GetDefaultParameterMap("bspline")
    pmap2["MaximumNumberOfIterations"] = [str(iterations[1])]
    pmap2["Metric0Weight"] = ["0.1"]
    pmap2["Metric1Weight"] = [str(scalePenalty)]
    parameterMapVector.append(pmap2)

    elastixImageFilter.SetParameterMap(parameterMapVector)

    elastixImageFilter.Execute()
    res = elastixImageFilter.GetResultImage()

    if savepath:
        from pathlib import Path

        pmaps = elastixImageFilter.GetTransformParameterMap()

        for n, pmap in enumerate(pmaps):
            sitk.WriteParameterFile(
                pmap, Path(savepath).joinpath(f"transform_pmap_{n}.txt").as_posix()
            )

    return sitk.GetArrayFromImage(res)


def calculate_match_value2(image_reference, image_target):
    def_size = 1024
    while max(max(image_reference.shape, image_target.shape)) >= def_size:
        def_size *= 2
    image_target = embed_image(image_target, def_size)
    image_reference = embed_image(image_reference, def_size)

    res = register_image2(image_target, image_reference, iterations=(50, 100))
    r = sitk.ImageRegistrationMethod()
    r.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    r.SetOptimizerAsLBFGSB(maximumNumberOfCorrections=3, numberOfIterations=100)
    r.SetMetricSamplingStrategy(r.RANDOM)
    r.SetMetricSamplingPercentage(0.5)
    tx = sitk.TranslationTransform(2)
    r.SetInitialTransform(tx)
    r.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2])
    r.SetSmoothingSigmasPerLevel(smoothingSigmas=[3, 1])
    # res_img = sitk.GetArrayFromImage(res)
    tx = estimate_transform_itk(image_reference, res, r)
    return r.GetMetricValue()


def transform_points(folderpath: str, points: list, cleanup: bool = True) -> list:
    """
    transforms a list of points

    :param cleanup:
    :param folderpath:
    :param points:
    :return:
    """
    from pathlib import Path

    # write pts to file
    point_path = Path(folderpath).joinpath("point_set.txt")
    if os.path.exists(point_path):
        os.remove(point_path)

    filestream = open(point_path, "a")
    filestream.write("point")
    filestream.write("\n")
    filestream.write(f"{len(points)}")
    filestream.write("\n")

    for pt in points:
        filestream.write(f"{pt[0]} {pt[1]}")
        filestream.write("\n")

    filestream.flush()
    filestream.close()

    # load our saved parameter maps and build filter
    pmap_files = []
    with os.scandir(folderpath) as entries:
        for entry in entries:
            if "transform_pmap" in entry.name:
                pmap_files.append(entry.path)

    pmap0 = sitk.ReadParameterFile(pmap_files[0])

    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetTransformParameterMap(pmap0)

    for pmap_file in pmap_files[1:]:
        pmap = sitk.ReadParameterFile(pmap_file)
        transformixImageFilter.AddTransformParameterMap(pmap)

    transformixImageFilter.SetFixedPointSetFileName(point_path.as_posix())
    transformixImageFilter.SetOutputDirectory(folderpath)
    transformixImageFilter.Execute()

    output_pts_path = Path(folderpath).joinpath("outputpoints.txt")

    with open(output_pts_path) as file:
        contents = file.read()
    lines = contents.split("\n")
    coords = []
    for line in lines:
        if line != "":
            x = int(line.split(";")[3].split("[ ")[1].split(" ]")[0].split(" ")[0])
            y = int(line.split(";")[3].split("[ ")[1].split(" ]")[0].split(" ")[1])
            coord = (x, y)
            coords.append(coord)
    return coords


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


def find_best_z_match(
    stack_reference, image_target, rigorous=False, l=None, r=None, check_distance=3
):
    """

    :param stack_reference: 3d image stack to align image target to
    :param image_target: target image array (2d)
    :param rigorous: if you have an abundance of time this can be true
    :param l:
    :param r:
    :return: Z index of reference stack and results dict
    """

    results_dictionary = {}

    if not l:
        l = 0
    if not r:
        r = len(stack_reference) - 1

    if not rigorous:
        while r - l > 1:

            if l not in results_dictionary.keys():
                try:
                    results_dictionary[l] = abs(
                        calculate_match_value(stack_reference[l], image_target)
                    )
                except:
                    results_dictionary[l] = 0

            if r not in results_dictionary.keys():
                try:
                    results_dictionary[r] = abs(
                        calculate_match_value(stack_reference[r], image_target)
                    )
                except:
                    results_dictionary[r] = 0

            midpt = ((r - l) // 2) + l

            while midpt in results_dictionary.keys():
                midpt += 1

                if midpt >= r:
                    break

            try:
                results_dictionary[midpt] = abs(
                    calculate_match_value(stack_reference[midpt], image_target)
                )
            except:
                results_dictionary[midpt] = 0

            if results_dictionary[r] > results_dictionary[l]:
                if results_dictionary[midpt] >= results_dictionary[l]:
                    l = midpt
                else:
                    break
            elif results_dictionary[l] >= results_dictionary[r]:
                if results_dictionary[midpt] >= results_dictionary[r]:
                    r = midpt
                else:
                    break

            print(l, r)

        maxval = max(results_dictionary.values())
        maxkey = {v: k for k, v in results_dictionary.items()}[maxval]
        for ind in np.arange(maxkey - check_distance, maxkey + check_distance):
            if ind not in results_dictionary.keys():
                results_dictionary[ind] = abs(
                    calculate_match_value(stack_reference[ind], image_target)
                )
        maxval = max(results_dictionary.values())
        maxkey = {v: k for k, v in results_dictionary.items()}[maxval]
        return maxkey, results_dictionary
    else:
        for i in np.arange(l, r):
            results_dictionary[i] = abs(
                calculate_match_value(stack_reference[i], image_target)
            )
        maxval = max(results_dictionary.values())
        maxkey = {v: k for k, v in results_dictionary.items()}[maxval]
        return maxkey, results_dictionary


def find_best_z_match2(
    stack_reference, image_target, rigorous=False, l=None, r=None, check_distance=3
):
    """
    :param stack_reference: 3d image stack to align image target to
    :param image_target: target image array (2d)
    :param rigorous: if you have an abundance of time this can be true
    :param l:
    :param r:
    :return: Z index of reference stack and results dict
    """

    results_dictionary = {}

    if not l:
        l = 0
    if not r:
        r = len(stack_reference) - 1

    if not rigorous:
        while r - l > 1:

            if l not in results_dictionary.keys():
                try:
                    results_dictionary[l] = abs(
                        calculate_match_value2(stack_reference[l], image_target)
                    )
                except:
                    results_dictionary[l] = 0

            if r not in results_dictionary.keys():
                try:
                    results_dictionary[r] = abs(
                        calculate_match_value2(stack_reference[r], image_target)
                    )
                except:
                    results_dictionary[r] = 0

            midpt = ((r - l) // 2) + l

            while midpt in results_dictionary.keys():
                midpt += 1

                if midpt >= r:
                    break

            try:
                results_dictionary[midpt] = abs(
                    calculate_match_value2(stack_reference[midpt], image_target)
                )
            except:
                results_dictionary[midpt] = 0

            if results_dictionary[r] > results_dictionary[l]:
                if results_dictionary[midpt] >= results_dictionary[l]:
                    l = midpt
                else:
                    break
            elif results_dictionary[l] >= results_dictionary[r]:
                if results_dictionary[midpt] >= results_dictionary[r]:
                    r = midpt
                else:
                    break

            print(l, r)

        maxval = max(results_dictionary.values())
        maxkey = {v: k for k, v in results_dictionary.items()}[maxval]
        for ind in np.arange(maxkey - check_distance, maxkey + check_distance):
            if ind not in results_dictionary.keys():
                results_dictionary[ind] = abs(
                    calculate_match_value2(stack_reference[ind], image_target)
                )
        maxval = max(results_dictionary.values())
        maxkey = {v: k for k, v in results_dictionary.items()}[maxval]
        return maxkey, results_dictionary
    else:
        for i in np.arange(l, r):
            results_dictionary[i] = abs(
                calculate_match_value2(stack_reference[i], image_target)
            )
        maxval = max(results_dictionary.values())
        maxkey = {v: k for k, v in results_dictionary.items()}[maxval]
        return maxkey, results_dictionary
