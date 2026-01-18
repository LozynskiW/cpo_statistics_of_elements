import copy
import math
import time

import numpy as np
from skimage.filters import threshold_multiotsu
from skimage.measure import label

from constants import HUMAN_CELLS_MITOSIS_EASY, HUMAN_CELLS_MITOSIS_BINS
from models import ImageObjectsStatistics
from utils import load_image, visualize_objects_statistics, visualize_image, visualize_labels_on_image


"""
    Based on:
    Measuring the blood cells by means of an image segmentation. Philica, 2017. hal-01654006
    By Amelia Carolina Sparavigna.
"""

def preprocess_image(image):
    return copy.deepcopy(image)


def segment_image(image):
    thresholds = threshold_multiotsu(image, classes=2)
    segmented = np.digitize(image, bins=thresholds)

    return segmented


def label_objects(binary_image) -> list[tuple[int, list[float]]]:
    return label(binary_image)


def contour_area(contour):
    """
    Computes the object area based on its contour
    using the polygon area formula.
    """
    x = contour[:, 1]
    y = contour[:, 0]
    return 0.5 * np.abs(
        np.dot(x, np.roll(y, 1)) -
        np.dot(y, np.roll(x, 1))
    )


def contour_perimeter(contour):
    """
    Computes the object perimeter based on its contour.
    """
    diffs = np.diff(contour, axis=0)
    distances = np.sqrt((diffs ** 2).sum(axis=1))
    return distances.sum()


def _initialize_size_dict(image: np.ndarray):
    """
    Collects unique labels and initializes a size dictionary
    in the form label -> object size in pixels.

    Example:
    {
        0: 12,
        1: 94,
        ...
    }
    """
    if image.ndim != 2:
        raise ValueError("Image must be 2D")

    labels = set()

    rows, cols = image.shape
    for y in range(rows):
        for x in range(cols):
            labels.add(int(image[y, x]))

    return {label: 0 for label in labels}


def _measure_single_object(image: np.ndarray, label: int):
    """
    Measures the size of a single labeled object using
    an 8-directional ray casting method (Carnot theorem)
    and polygon area estimation.
    """

    rows, cols = image.shape

    # Locate object pixels and compute centroid
    sum_y = 0.0
    sum_x = 0.0
    count = 0

    for y in range(rows):
        for x in range(cols):
            if image[y, x] == label:
                sum_y += y
                sum_x += x
                count += 1

    if count == 0:
        raise ValueError(f"Label {label} not found")

    cy = sum_y / count
    cx = sum_x / count

    # Ray casting in 8 directions
    angles = [k * math.pi / 4 for k in range(8)]
    radii = []
    max_radius = max(rows, cols)
    step = 0.5

    for theta in angles:
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        r = 0.0

        while True:
            y = cy + r * sin_t
            x = cx + r * cos_t

            iy = int(round(y))
            ix = int(round(x))

            if (iy < 0 or iy >= rows or
                ix < 0 or ix >= cols or
                image[iy, ix] != label):
                break

            r += step
            if r > max_radius:
                break

        radii.append(r)

    # Perimeter estimation using Carnot theorem
    delta_theta = math.pi / 4
    cos_dt = math.cos(delta_theta)

    perimeter = 0.0
    for i in range(8):
        r1 = radii[i]
        r2 = radii[(i + 1) % 8]
        d = math.sqrt(
            r1**2 + r2**2 - 2 * r1 * r2 * cos_dt
        )
        perimeter += d

    # Polygon area estimation (shoelace formula)
    vertices = []
    for k, r in enumerate(radii):
        theta = angles[k]
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        vertices.append((x, y))

    area = 0.0
    for i in range(8):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % 8]
        area += (x1 * y2 - x2 * y1)

    area = abs(area) * 0.5

    return area
    # Optionally, additional properties could be returned:
    # return {
    #     "centroid": (cy, cx),
    #     "radii": radii,
    #     "perimeter": perimeter,
    #     "area": area
    # }


def calculate_statistics(labeled_image) -> ImageObjectsStatistics:

    areas_dict = _initialize_size_dict(labeled_image)

    for label in areas_dict.keys():
        areas_dict[label] = _measure_single_object(labeled_image, label)

    return ImageObjectsStatistics(objs_areas=list(areas_dict.items()))


# MAIN

start = time.time()

image = load_image(HUMAN_CELLS_MITOSIS_EASY)
image_preprocessed = preprocess_image(image)
image_segmented = segment_image(image_preprocessed)
labeled_image = label_objects(image_segmented)
stats = calculate_statistics(labeled_image)

end = time.time()
elapsed = end - start
print(f"Execution time: {elapsed:.6f} s")
print(stats)

visualize_objects_statistics(
    stats=stats,
    bins_num=HUMAN_CELLS_MITOSIS_BINS,
    x_label="object area",
    title="Thresholding method"
)
visualize_image(image_preprocessed, title="Preprocessed image")
visualize_image(image_segmented, title="Segmented image")
visualize_labels_on_image(labeled_image, title="Labeled image")
