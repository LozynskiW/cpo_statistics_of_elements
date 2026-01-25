import copy

from constants import HUMAN_CELLS_MITOSIS_EQUAL_BINS
from connected_regions import labeled_image as thresholding_labeled_image
from contours import labeled_image as contours_labeled_image
from watershed import labeled_image as watershed_labeled_image
from watershed import calculate_statistics as watershed_calculate_statistics

from utils import visualize_multiple_objects_statistics

thresholding_stats = watershed_calculate_statistics(copy.deepcopy(thresholding_labeled_image))
contours_stats = watershed_calculate_statistics(copy.deepcopy(contours_labeled_image))
watershed_stats = watershed_calculate_statistics(copy.deepcopy(watershed_labeled_image))

label_stats_dict = {
    "carnot": {"data": thresholding_stats, "hatch": "."},
    "contours": {"data": contours_stats, "hatch": "/"},
    "watershed": {"data": watershed_stats, "hatch": "\\"}
}

visualize_multiple_objects_statistics(
    label_stats_dict=label_stats_dict,
    bins_num=HUMAN_CELLS_MITOSIS_EQUAL_BINS,
    x_label="object area",
    title="Methods comparison"
)
