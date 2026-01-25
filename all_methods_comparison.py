from constants import HUMAN_CELLS_MITOSIS_EQUAL_BINS
from connected_regions import stats as thresholding_stats
from contours import stats as contours_stats
from watershed import stats as watershed_stats

from utils import visualize_multiple_objects_statistics

label_stats_dict = {
    "connected-regions": {"data": thresholding_stats, "hatch": "."},
    "contours": {"data": contours_stats, "hatch": "/"},
    "watershed": {"data": watershed_stats, "hatch": "\\"}
}

visualize_multiple_objects_statistics(
    label_stats_dict=label_stats_dict,
    bins_num=HUMAN_CELLS_MITOSIS_EQUAL_BINS,
    x_label="object area",
    title="Methods comparison"
)
