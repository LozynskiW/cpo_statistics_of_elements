import copy
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes

from skimage import filters
from skimage.filters import threshold_otsu, butterworth
from skimage.morphology import closing, disk, erosion
from skimage.measure import find_contours

from constants import HUMAN_CELLS_MITOSIS_EASY, HUMAN_CELLS_MITOSIS_BINS
from models import ImageObjectsStatistics
from utils import load_image, visualize_objects_statistics, visualize_image


"""
    Based on:
    High-Throughput Method for Automated Colony and Cell Counting by Digital Image Analysis Based on Edge Detection
    by Priya Choudhry
"""

def preprocess_image(
    gray_image,
    cutoff_frequency_ratio: float = 0.2,
    filter_order: int = 16,
    num_of_erosions: int = 1
):

    img = copy.deepcopy(gray_image)

    # 1. Noise reduction (low-pass filtering)
    img_smooth = butterworth(
        img,
        cutoff_frequency_ratio=cutoff_frequency_ratio,
        high_pass=False,
        order=filter_order
    )

    # Optional morphological erosion to suppress small artifacts
    for _ in range(num_of_erosions):
        img_smooth = erosion(img_smooth)

    # 2. Contrast normalization
    img_smooth = (img_smooth - img_smooth.min()) / (img_smooth.max() - img_smooth.min())

    # 3. Edge detection
    edges = filters.sobel(img_smooth)

    return edges


def segment_image(image, disk_mask_size=1):
    img = copy.deepcopy(image)

    # 1. Automatic thresholding (Otsu)
    threshold = threshold_otsu(image)
    binary_edges = img > threshold

    # 2. Contour closing using morphological operations
    closed = closing(binary_edges, disk(disk_mask_size))

    # 3. Filling object interiors
    filled = binary_fill_holes(closed)

    return filled


def label_objects(binary_image):
    """
    Detects contours – each contour is treated as a separate object.
    """
    contours = find_contours(
        binary_image,
        level=0.5,
        fully_connected='high'
    )
    return contours


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


def calculate_statistics(contours):
    areas = [contour_area(c) for c in contours]
    # perimeters = [contour_perimeter(c) for c in contours]
    # equivalent_diameters = [np.sqrt(4 * a / np.pi) for a in areas]

    objs_areas = {i + 1: v for i, v in enumerate(areas)}

    return ImageObjectsStatistics(objs_areas=objs_areas.items())


def visualize_contours(gray_image, contours):
    fig, ax = plt.subplots()
    ax.imshow(gray_image, cmap='gray')

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.set_title("Detected object contours")
    ax.axis('off')
    plt.show()


# MAIN
visualize = False

start = time.time()

image = load_image(HUMAN_CELLS_MITOSIS_EASY)
image_preprocessed = preprocess_image(
    image,
    cutoff_frequency_ratio=0.2,
    filter_order=16,
    num_of_erosions=1
)
image_segmented = segment_image(image_preprocessed, disk_mask_size=1)
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
        title="Contour-based method"
    )
    visualize_image(image_preprocessed, title="Preprocessed image")
    visualize_image(image_segmented, title="Segmented image")
    visualize_contours(image_segmented, labeled_image)
    visualize_contours(image, labeled_image)
