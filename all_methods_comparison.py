from constants import HUMAN_CELLS_MITOSIS_BINS
from thresholding import stats as thresholding_stats
from contours import stats as contours_stats
from watershed import stats as watershed_stats

from utils import visualize_multiple_objects_statistics

label_stats_dict = {
    "thresholding": thresholding_stats,
    "contours": contours_stats,
    "watershed": watershed_stats
}

visualize_multiple_objects_statistics(
    label_stats_dict=label_stats_dict,
    bins_num=HUMAN_CELLS_MITOSIS_BINS,
    x_label="object area",
    title="Methods comparison"
)
