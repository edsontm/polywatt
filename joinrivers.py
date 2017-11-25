#!/usr/bin/python

import os
import csv
import numpy as np
from watt import RiverDataset


if __name__ == '__main__':
    r = RiverDataset()
    r.load_pairs('stations.txt')
    r.load_dir('rcsv')
    r.save_complete_years('river_pairs')
