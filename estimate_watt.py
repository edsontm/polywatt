#!/usr/bin/python

from watt import RiverDataset,RiverPair

if __name__ == '__main__':
   r = RiverDataset()
   r.load_complete_years('river_pairs')
   RiverPair.run_all_pairs(r)


