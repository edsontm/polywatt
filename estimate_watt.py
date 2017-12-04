#!/usr/bin/python

from watt import RiverDataset,RiverPair
import sys

def exit_error():
    print("usage:%s <--disable-pip>\n"%sys.argv[0])
    sys.exit(0)

if __name__ == '__main__':
    enable_pip = True
    if len(sys.argv) == 2:
         if sys.argv[1] == '--disable-pip':
            enable_pip = False
         else:
             exit_error()
    if enable_pip:
        print "Running with pip\nTo disable pip run:\n\n    %s --disable-pip\n"%(sys.argv[0]) 
    else:
        print "\n\nPIP disabled\n";
    r = RiverDataset()
    r.load_complete_years('river_pairs')
    RiverPair.run_all_pairs(r,enable_pip=enable_pip)


