import copy
import time
import numpy as np
import matplotlib.pyplot as plt

from skimage import io, color, filters
from skimage.filters import gaussian, threshold_otsu, butterworth
from skimage.morphology import skeletonize
from skimage.measure import find_contours

from constants import BLOOD_CELLS_IMAGE_EASY
from utils import visualize_image


def load_image(path):
    """
    Wczytuje obraz z pliku.
    """
    img = io.imread(path)
    return color.rgb2gray(img)


def preprocess_image(gray_image):
    """
    Based on High-Throughput Method for Automated Colony and Cell Counting by Digital Image Analysis Based on Edge Detection by Priya Choudhry
    """
    gray_image_copy = copy.deepcopy(gray_image)

    sharpened = butterworth(gray_image_copy, 0.07, True, 8)
    edges = filters.sobel(sharpened)
    return gaussian(edges, sigma=0.1)


def segment_image(image):
    threshold = threshold_otsu(image)
    binary = image > threshold

    return skeletonize(binary)


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
    """
    Ekstrahuje cechy geometryczne i oblicza statystykę obiektów.
    """
    areas = [contour_area(c) for c in contours]
    perimeters = [contour_perimeter(c) for c in contours]
    equivalent_diameters = [np.sqrt(4 * a / np.pi) for a in areas]

    stats = {
        "count": len(contours),
        "areas": areas,
        "perimeters": perimeters,
        "equivalent_diameters": equivalent_diameters,
        "mean_area": np.mean(areas) if areas else 0,
        "min_area": np.min(areas) if areas else 0,
        "max_area": np.max(areas) if areas else 0,
    }

    return stats


def visualize_contours(gray_image, contours):
    """
    Wyświetla wykryte kontury na obrazie.
    """
    fig, ax = plt.subplots()
    ax.imshow(gray_image, cmap='gray')

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.set_title("Wykryte kontury obiektów")
    ax.axis('off')
    plt.show()

# MAIN

start = time.time()
image = load_image(BLOOD_CELLS_IMAGE_EASY)
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

plt.hist(stats["areas"], bins=20)
plt.xlabel("Pole obiektu [piksele]")
plt.ylabel("Liczba obiektów")
plt.title("Rozkład rozmiarów obiektów – metoda konturowa")
plt.show()

visualize_contours(image_segmented, labeled_image)
visualize_contours(image, labeled_image)
