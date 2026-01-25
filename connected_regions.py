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

    labels_dict = {label: 0 for label in labels}
    labels_dict.pop(0) # so that background is not label
    return labels_dict


def _measure_single_object(
    image: np.ndarray,
    label: int,
    step: float = 0.25
) -> float:
    """
    Measures object area using 8-directional ray casting
    and Carnot-based triangular area estimation.
    """

    rows, cols = image.shape

    # --- centroid (pixel-accurate) ---
    ys, xs = np.where(image == label)
    if len(xs) == 0:
        raise ValueError(f"Label {label} not found")

    cy = ys.mean()
    cx = xs.mean()

    # --- ray casting ---
    angles = [k * math.pi / 4 for k in range(8)]
    radii = []
    boundary_points = []

    max_radius = math.hypot(rows, cols)

    for theta in angles:
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)

        r = 0.0
        last_inside = None

        while r < max_radius:
            y = cy + r * sin_t
            x = cx + r * cos_t

            iy = int(math.floor(y))
            ix = int(math.floor(x))

            if (
                iy < 0 or iy >= rows or
                ix < 0 or ix >= cols or
                image[iy, ix] != label
            ):
                break

            last_inside = (x, y)
            r += step

        if last_inside is None:
            radii.append(0.0)
            boundary_points.append((cx, cy))
        else:
            bx, by = last_inside
            radii.append(math.hypot(bx - cx, by - cy))
            boundary_points.append((bx, by))

    # --- Carnot-based area estimation ---
    delta_theta = math.pi / 4
    sin_dt = math.sin(delta_theta)

    area_carnot = 0.0
    for i in range(8):
        r1 = radii[i]
        r2 = radii[(i + 1) % 8]
        area_carnot += 0.5 * r1 * r2 * sin_dt

    return area_carnot


def calculate_statistics(labeled_image) -> ImageObjectsStatistics:

    areas_dict = _initialize_size_dict(labeled_image)

    for label in areas_dict.keys():
        areas_dict[label] = _measure_single_object(labeled_image, label)

    return ImageObjectsStatistics(objs_areas=list(areas_dict.items()))


# MAIN
visualize = False

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

if visualize:
    visualize_objects_statistics(
        stats=stats,
        bins_num=HUMAN_CELLS_MITOSIS_BINS,
        x_label="object area",
        title="Connected regions method"
    )
    visualize_image(image_preprocessed, title="Preprocessed image")
    visualize_image(image_segmented, title="Segmented image")
    visualize_labels_on_image(labeled_image, title="Labeled image")
