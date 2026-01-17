import io

from matplotlib import pyplot as plt
from skimage import io, color


def load_image(path):
    img = io.imread(path)
    if img.ndim == 3 and img.shape[-1] == 4: # is alpha channel in image
        img = img[..., :3]  # drop alpha channel

    return color.rgb2gray(img)

def visualize_image(gray_image, title=""):
    fig, ax = plt.subplots()
    ax.imshow(gray_image, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
    plt.show()