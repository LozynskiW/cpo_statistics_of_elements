import copy
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes

from skimage import filters
from skimage.filters import threshold_otsu, butterworth
from skimage.morphology import closing, disk, erosion
from skimage.measure import find_contours

from constants import HUMAN_CELLS_MITOSIS_EASY
from models import ImageObjectsStatistics
from utils import load_image, visualize_objects_statistics, visualize_image


def preprocess_image(gray_image, cutoff_frequency_ratio: float = 0.2, filter_order: int = 16, num_of_erosions: int = 1):
    """
    Based on High-Throughput Method for Automated Colony and Cell Counting by Digital Image Analysis Based on Edge Detection by Priya Choudhry
    """
    img = copy.deepcopy(gray_image)

    # 1. Redukcja szumu (low-pass)
    img_smooth = butterworth(img, cutoff_frequency_ratio=cutoff_frequency_ratio, high_pass=False, order=filter_order)

    for i in range(0, num_of_erosions):
        img_smooth = erosion(img_smooth)

    # 2. Normalizacja kontrastu
    img_smooth = (img_smooth - img_smooth.min()) / (img_smooth.max() - img_smooth.min())

    # 3. Detekcja krawędzi
    edges = filters.sobel(img_smooth)

    return edges


def segment_image(image, disk_mask_size=1):
    img = copy.deepcopy(image)

    threshold = threshold_otsu(image)
    binary_edges = img > threshold

    # 2. Domykanie konturów
    closed = closing(binary_edges, disk(disk_mask_size))

    # 3. Wypełnianie wnętrz obiektów
    filled = binary_fill_holes(closed)

    return filled


def label_objects(binary_image):
    """
    Wykrywa kontury – każdy kontur traktowany jako osobny obiekt.
    """
    contours = find_contours(binary_image,
                             level=0.5,
                             fully_connected='high')
    return contours


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

    ax.set_title("Wykryte kontury obiektów")
    ax.axis('off')
    plt.show()

# MAIN

start = time.time()

image = load_image(HUMAN_CELLS_MITOSIS_EASY)
image_preprocessed = preprocess_image(image, cutoff_frequency_ratio= 0.2, filter_order = 16, num_of_erosions = 1)
image_segmented = segment_image(image_preprocessed, disk_mask_size=1)
labeled_image = label_objects(image_segmented)
stats = calculate_statistics(labeled_image)

end = time.time()
elapsed = end - start

print(f"Czas wykonania: {elapsed:.6f} s")
print(stats)

visualize_objects_statistics(stats=stats, bins_num=30, x_label="object area", title="Contour method")
visualize_image(image_preprocessed, title="image_preprocessed")
visualize_image(image_segmented, title="segmented")
visualize_contours(image_segmented, labeled_image)
visualize_contours(image, labeled_image)
