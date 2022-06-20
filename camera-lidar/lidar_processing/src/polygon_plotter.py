import csv
import sys
from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt

# This is a debug tool for the landArea.csv files.
# Usage: run python3 polygon_plotter.py <name of your csv>


class Point(NamedTuple):
    x: float
    y: float



def main(argv):
    with open(argv[1], newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        prev_id = 0
        id = 0
        polygon = []
        polygons = []
        for line in reader:
            if not line:
                continue
            prev_id = id
            id = line[0]
            if id == prev_id:
                polygon.append(Point(line[1], line[2]))
            else:
                if len(polygon):
                    polygon.append(polygon[0])
                polygons.append(polygon)
                polygon = []

    for (i, poly) in enumerate(polygons):
        x = []
        y = []
        for point in poly:
            p_x = float(point.x)*1.0e6
            p_y = float(point.y)*1.0e6
            x.append(p_x)
            y.append(p_y)
        plt.plot(x, y, c="blue")
    plt.show()


if __name__ == '__main__':
        main(sys.argv)
