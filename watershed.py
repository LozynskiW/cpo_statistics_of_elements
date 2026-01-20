import time

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology, measure, segmentation

from constants import HUMAN_CELLS_MITOSIS_EASY, HUMAN_CELLS_MITOSIS_BINS
from models import ImageObjectsStatistics
from utils import load_image, visualize_objects_statistics, visualize_image, visualize_labels_on_image

"""
    Based on:
    Automated Particle Size and Shape Determination Methods: Application to Proppant Optimization
    By Dongjin Xu, Junting Wang, Zhiwen Li, Changheng Li, Yukai Guo, Xuyi Qiao and Yong Wang 
"""

def preprocess_image(image):
    return filters.gaussian(image, sigma=1.0)


def segment_image(image_preprocessed):
    # Otsu thresholding
    thresh = filters.threshold_otsu(image_preprocessed)
    binary = image_preprocessed > thresh

    # Auto inversion if background is bright
    if np.mean(binary) > 0.5:
        binary = ~binary

    # Morphological cleaning
    binary = morphology.remove_small_objects(binary, max_size=5)
    binary = morphology.opening(binary, morphology.disk(2))
    binary = morphology.closing(binary, morphology.disk(2))

    return binary

def label_objects(image_segmented):
    distance = ndi.distance_transform_edt(image_segmented)
    local_maxi = morphology.local_maxima(distance)
    markers = measure.label(local_maxi)

    return segmentation.watershed(
        -distance,
        markers,
        mask=image_segmented
    )


def calculate_statistics(labeled_image) -> ImageObjectsStatistics:

    props = measure.regionprops(labeled_image)
    areas = []

    for p in props:
        area_circle_px = np.pi * (p.equivalent_diameter / 2) ** 2
        areas.append(area_circle_px)
        objs_areas = {i + 1: v for i, v in enumerate(areas)}

    return ImageObjectsStatistics(objs_areas=objs_areas.items())

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
        title="Watershed method"
    )
    visualize_image(image_preprocessed, title="Preprocessed image")
    visualize_image(image_segmented, title="Segmented image")
    visualize_labels_on_image(labeled_image, title="Labeled image")
