import io

from matplotlib import pyplot as plt
import skimage as ski
from skimage import io, color

from models import ImageObjectsStatistics


def load_image(path):
    img = io.imread(path)
    if img.ndim == 3 and img.shape[-1] == 4: # is alpha channel in image
        img = img[..., :3]  # drop alpha channel

    return color.rgb2gray(img)

def visualize_image(image, title=""):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
    plt.show()
    plt.close(fig)

def visualize_labels_on_image(labeled_image, title=""):
    fig, ax = plt.subplots()
    ax.imshow(ski.color.label2rgb(labeled_image, bg_label=0), cmap='gray')
    ax.set_title(title)
    ax.axis('off')
    plt.show()
    plt.close(fig)

def visualize_objects_statistics(stats: ImageObjectsStatistics, bins_num: int, x_label: str, title: str):
    fig, ax = plt.subplots()
    plt.hist(stats.areas, bins=bins_num)
    plt.xlabel(x_label)
    plt.ylabel("Number of objects for bin")
    plt.title(title)
    plt.grid()
    plt.show()
    plt.close(fig)
