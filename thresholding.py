import copy
import math
import time

import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import gaussian, butterworth, threshold_multiotsu
from skimage.measure import label

from constants import HUMAN_CELLS_MITOSIS_EASY
from utils import load_image, visualize_image


# Amelia Carolina Sparavigna. Measuring the blood cells by means of an image segmentation. Philica,
# 2017. hal-01654006

def preprocess_image(image):
    return copy.deepcopy(image)


def segment_image(image):
    thresholds = threshold_multiotsu(image, classes=2)
    segmented = np.digitize(image, bins=thresholds)

    return segmented


def label_objects(binary_image):
    return label(binary_image)


def contour_area(contour):
    """
    Oblicza pole powierzchni obiektu na podstawie konturu
    (wzór na pole wielokąta).
    """
    x = contour[:, 1]
    y = contour[:, 0]
    return 0.5 * np.abs(
        np.dot(x, np.roll(y, 1)) -
        np.dot(y, np.roll(x, 1))
    )


def contour_perimeter(contour):
    """
    Oblicza obwód obiektu na podstawie konturu.
    """
    diffs = np.diff(contour, axis=0)
    distances = np.sqrt((diffs ** 2).sum(axis=1))
    return distances.sum()


def _initialize_size_dict(image: np.ndarray):
    """
    Collect unique labels and initialize size dictionary.

    Parameters
    ----------
    image : np.ndarray

    Returns
    -------
    dict
        label -> 0
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
    Measure size of a single labeled object using
    8-directional Carnot theorem and polygon area.

    Parameters
    ----------
    image : np.ndarray
    label : int

    Returns
    -------
    dict
    """

    rows, cols = image.shape

    # --------------------------------------------------
    # Locate object pixels and centroid
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Ray casting
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Carnot perimeter
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Polygon area (shoelace)
    # --------------------------------------------------
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

    return {
        "centroid": (cy, cx),
        "radii": radii,
        "perimeter": perimeter,
        "area": area
    }


def calculate_statistics(labeled_image):
    """
    Ekstrahuje cechy geometryczne i oblicza statystykę obiektów.
    """
    areas = _initialize_size_dict(labeled_image)

    for label in areas.keys():
        areas[label] = _measure_single_object(labeled_image, label)

    stats = {
        "count": len(areas),
        "areas": areas,
        # "mean_area": np.mean(areas.items()) if areas else 0,
        # "min_area": np.min(areas.items()) if areas else 0,
        # "max_area": np.max(areas.items()) if areas else 0,
    }

    return stats

# MAIN

start = time.time()
image = load_image(HUMAN_CELLS_MITOSIS_EASY)
image_preprocessed = preprocess_image(image)

# visualize_image(image_preprocessed, title="image_preprocessed")

image_segmented = segment_image(image_preprocessed)
# visualize_image(image_segmented, title="segmented")

labeled_image = label_objects(image_segmented)
stats = calculate_statistics(labeled_image)
end = time.time()
elapsed = end - start

print(f"Czas wykonania: {elapsed:.6f} s")
print(stats)