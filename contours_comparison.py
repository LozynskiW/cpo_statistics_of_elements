from matplotlib import pyplot as plt

from constants import HUMAN_CELLS_MITOSIS_EASY, OUTCOMES_PATH_RELATIVE
from contours import preprocess_image, segment_image, label_objects, calculate_statistics, visualize_contours
from utils import load_image, visualize_objects_statistics, visualize_image, save_objects_statistics, save_image

cutoff_frequency_ratios = [0.05, 0.2, 0.4]
filter_orders = [8, 16, 24]
nums_of_erosions = [0, 1, 2]

image = load_image(HUMAN_CELLS_MITOSIS_EASY)

def save_contours(gray_image, contours, file_name, dpi=200):
    fig, ax = plt.subplots()
    ax.imshow(gray_image, cmap='gray')

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.set_title("Detected object contours")
    ax.axis('off')
    plt.savefig(fname=f'{OUTCOMES_PATH_RELATIVE}/{file_name}.png', dpi=dpi)
    plt.close(fig, )

for i in range(0, len(cutoff_frequency_ratios)):

    image_preprocessed = preprocess_image(
        image,
        cutoff_frequency_ratio=cutoff_frequency_ratios[i],
        filter_order=filter_orders[i],
        num_of_erosions=1
    )
    image_segmented = segment_image(image_preprocessed, disk_mask_size=1)
    labeled_image = label_objects(image_segmented)
    stats = calculate_statistics(labeled_image)

    save_objects_statistics(
        stats=stats,
        bins_num=30,
        x_label="object area",
        title=f"cutoff_frequency_ratio: {cutoff_frequency_ratios[i]}, filter_orders: {filter_orders[i]}",
        file_name=f"statistics_{i}_version",
        dpi=200
    )

    save_image(
        image_preprocessed,
        title=f"Preprocessed image, cutoff_frequency_ratio: {cutoff_frequency_ratios[i]}, filter_orders: {filter_orders[i]}",
        file_name=f"processed_{i}_version"
    )
    save_image(
        image_segmented,
        title=f"Segmented image, cutoff_frequency_ratio: {cutoff_frequency_ratios[i]}, filter_orders: {filter_orders[i]}",
        file_name=f"segmented_{i}_version"
    )

    save_contours(image, labeled_image, file_name=f"contours_{i}_version")
