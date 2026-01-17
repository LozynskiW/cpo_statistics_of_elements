import statistics
from dataclasses import dataclass
from typing import Any


@dataclass(init=False)
class ImageObjectsStatistics:
    """
    count: how many objects is on image
    ares: list of all objects areas in pixels in format of list( tuple(obj_label, obj_size)...)
    avg_area: average area, calculated as expected value when all probabilities are equal
    """
    objs_areas: list[tuple[Any, float]]
    areas: list[float]
    count: int
    min_area: float
    max_area: float
    avg_area: float
    std_dev: float
    median: float

    def __init__(self, objs_areas: list[tuple[Any, float]]):
        self.objs_areas = objs_areas
        self.count = len(self.objs_areas)
        self.areas = list(map(lambda x: x[1], self.objs_areas))
        self.min_area = min(self.areas)
        self.max_area = max(self.areas)
        self.avg_area = statistics.mean(self.areas)
        self.std_dev = statistics.stdev(self.areas)
        self.median = statistics.median(self.areas)

    def __str__(self):
        return (f"number_of_objects = {self.count}\n"
                f"average object size in pixels = {self.avg_area}\n"
                f"object size standard deviation in pixels = {self.std_dev}\n"
                f"object size median in pixels = {self.std_dev}\n"
                f"min object size in pixels = {self.min_area}\n"
                f"max object size in pixels = {self.max_area}\n")