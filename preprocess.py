#!/usr/bin/python
from watt import Converter
import sys
import os

if __name__ == '__main__':
    if (len(sys.argv) != 2):
        print("usage: %s <dir>\n"%(sys.argv[0]))
        exit(0)
    main_dir = sys.argv[1]
    if not os.path.isdir(main_dir):
        print("usage: %s <dir>\n"%(sys.argv[0]))
        exit(0)
    tformat = main_dir.replace('/','')
    conv = Converter(tformat)
    
    conv.convert()

        

        

    


