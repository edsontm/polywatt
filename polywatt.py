#!/usr/bin/python

import getopt
import sys
from pvddtw import TestClass
import os
import csv

if __name__=='__main__':

    t = TestClass()
    if not os.path.isdir('complete_years'):
        t.test_prepare_dataset()

    t.test_pair_river()
