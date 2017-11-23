import os
import csv
import numpy as np
from scipy.interpolate import interp1d
from .dataloader import TablesLoader
import matplotlib.pyplot as plt



 
class Converter:
    def __init__(self,tformat='tables',output_dir='rcsv',missing_interpolate = 5):
        self.tformat = tformat
        self.output_dir = output_dir
        self.input_dir = tformat
        self.missing_interpolate = missing_interpolate
        self.loader = None

    def load(self,file_name):
        self.loader.load(file_name)
        self.river_name = self.loader.river_name
        self.vdates = self.loader.vdates
        self.hdates = self.loader.hdates
        self.levels = self.loader.levels

    def convert(self):
        print self.tformat
        if self.tformat == 'tables':
            self.loader = TablesLoader()
        for file_name in os.listdir(self.input_dir):
            print(file_name)
            self.load(self.input_dir+os.sep+file_name)
            self.interpolate_missing()
            self.save()

    def save(self):
        if not os.path.isdir(self.output_dir):
           os.mkdir(self.output_dir)

        tbuffer = ""
        for i in range(len(self.vdates)):
            v = self.levels[i]
            if type(v) == type(None):
                v = -1
            tbuffer+= self.vdates[i] + ",%f\n"%(v)
            
        f = open(self.output_dir+os.sep+self.river_name,'w')
        f.write(tbuffer)
        f.close()
             
    def missing_dict(self,v):
        anterior = 0
        state = 'None'
        comeco = 0
        counter = dict()
        for i in range(1,len(v)):
            if anterior != None  and  v[i] != None:
                state = 'numero'
            elif anterior != None and v[i] == None:
                state = 'abriu'
            elif anterior == None and v[i] == None:
                state = 'dentro'
            elif anterior == None and v[i] != None:
                state = 'saiu'
            if state == 'abriu':
                comeco = i
                if comeco not in counter:
                    counter[comeco] = 0
                counter[comeco] += 1
            elif state == 'dentro':
                counter[comeco] += 1
            anterior = v[i]
        return counter


            
    def interpolate_missing(self):
        counter = self.missing_dict(self.levels)    
        for comeco in counter.keys():
            nmiss = counter[comeco]
            if nmiss < self.missing_interpolate:
                x = [comeco-1,comeco + nmiss]
                y = [self.levels[comeco-1],self.levels[comeco+nmiss]]
                f = interp1d(x,y)
                for i in range(comeco,comeco+nmiss):
                    self.levels[i] = f(i)




        
             
       
