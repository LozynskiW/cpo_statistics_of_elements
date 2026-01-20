import io

from matplotlib import pyplot as plt
import skimage as ski
from skimage import io, color

from constants import OUTCOMES_PATH_RELATIVE
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
    plt.xticks(bins_num)
    plt.title(title)
    plt.grid()
    plt.show()
    plt.close(fig)

def visualize_multiple_objects_statistics(
        label_stats_dict: dict[str, ImageObjectsStatistics],
        bins_num: int,
        x_label: str,
        title: str
):
    fig, ax = plt.subplots()
    for label in label_stats_dict.keys():
        plt.hist(label_stats_dict[label].areas, bins=bins_num, label=label, alpha=0.5)
    plt.xlabel(x_label)
    plt.ylabel("Number of objects for bin")
    plt.xticks(bins_num)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()
    plt.close(fig)

def save_image(image, title="", file_name="outcome", dpi=200):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
    plt.savefig(fname=f'{OUTCOMES_PATH_RELATIVE}/{file_name}.png', dpi=dpi)
    plt.close(fig)

def save_objects_statistics(
        stats: ImageObjectsStatistics,
        bins_num: int,
        x_label: str,
        title="",
        file_name="outcome",
        dpi=200
):
    fig, ax = plt.subplots()
    plt.hist(stats.areas, bins=bins_num)
    plt.xlabel(x_label)
    plt.ylabel("Number of objects for bin")
    plt.title(title)
    plt.grid()
    plt.savefig(fname=f'{OUTCOMES_PATH_RELATIVE}/{file_name}.png', dpi=dpi)
    plt.close(fig)