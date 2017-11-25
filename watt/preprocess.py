import os
import csv
import numpy as np
from scipy.interpolate import interp1d
from .dataloader import TablesLoader,RcsvLoader
import matplotlib.pyplot as plt


 

class RiverDataset:
    def __init__(self):
        self.main_dir = None
        self.river_stations = dict()
        self.pairs = None
        self.verbose = True

    def load_dir(self,main_dir):
        self.main_dir = main_dir
        if self.pairs == None:
            for file_name in os.listdir(main_dir):
                if file_name.find('.csv') > 0:
                    print file_name
                    self.river_stations[file_name] = RiverStation()
                    self.river_stations[file_name].load_data(main_dir+os.sep+file_name)
        else:
            list_files = []
            for s in self.pairs:
                list_files += s

            for file_name in os.listdir(main_dir):
                if file_name.find('.csv') > 0:
                    if file_name in list_files:
                        print file_name
                        self.river_stations[file_name] = RiverStation()
                        self.river_stations[file_name].load_data(main_dir+os.sep+file_name)

    def _save_vectors(self,v1,v2,filename):
        if len(v1) != len(v2):
            raise Exception('sorry, you must give vectors of the same size, not %d %d'%(len(v1),len(v2)))
        else:
            tstr = ''
            for i in range(len(v1)):
                tstr += '%4.3f,%4.3f\n'%(v1[i],v2[i])
        f = open(filename,'w')
        f.write(tstr)
        f.close()



    def save_complete_years(self,complete_dir):
        if self.pairs == None:
            raise Exception('sorry, you must execute load_pairs first')
        else:
            if not os.path.isdir(complete_dir):
                os.mkdir(complete_dir)
            tcount = 0       
            for pair in self.pairs:
                r1 = self.river_stations[pair[0]]
                r2 = self.river_stations[pair[1]]
                common_dates = r1.find_common_intervals(r2)
                for (diff,start1,end1) in common_dates:
                    if diff >= 364:
                        tcount+=1
                        
                        dstart = r1.vdates[start1] 
                        dend   = r1.vdates[end1]

                        start2 = r2.hdates[dstart]
                        end2   = r2.hdates[dend]

                        v1     = r1.levels[start1:end1]
                        v2     = r2.levels[start2:end2]
                        data_name = r1.river_name.replace('.csv','_')+r2.river_name.replace('.csv','') + '_'+dstart+'.csv'
                        self._save_vectors(v1,v2,complete_dir+os.sep+data_name)
            print tcount


    def load_pairs(self,pairs_file):
        f = csv.reader(open(pairs_file),delimiter=' ',quotechar='|')
        rname = []
        for row in f:
            if len(row) > 0:
                rname.append(row)
        if self.verbose: print rname
        self.pairs = rname


                
    def load_complete_years(self,complete_dir):
        self.data = dict()
        self.start_date = dict()
        self.hstart_date = dict()
        for file_name in sorted(os.listdir(complete_dir)):
            if file_name.find('.csv') > 0:
                v = file_name.split('_')
                pair_name = '_'.join(v[:2])
                if pair_name not in self.data:
                    self.data[pair_name] = []
                    self.start_date[pair_name] = []
                    self.hstart_date[pair_name] = dict()
                f = csv.reader(open(complete_dir+os.sep+file_name),delimiter=',')

                start_date  = v[2].replace('.csv','')
                start_index = len(self.data[pair_name])

                self.start_date[pair_name].append((start_date,start_index))
                self.hstart_date[pair_name][start_index] = start_date
                for row in f:
                    self.data[pair_name].append([float(temp) for temp in row])
        if self.verbose:
            for pair_name in self.data:
                print pair_name,len(self.data[pair_name])



        
    def split_train_test(self,pair_name,w_years,test_size):
        result = []
        v = self.start_date[pair_name]
        v.append(('',len(self.data[pair_name])-1))
        iv = []
        for i in range(len(v)):
            iv.append(v[i][1])

        for w in range(w_years,len(self.start_date[pair_name]),1):
            train = iv[w-w_years:w-test_size+1]
            test  = iv[w-test_size:w]
            rtrain = []
            rtest  = []
            for i in range(len(train)-1):
                rtrain.append((train[i],train[i+1]))
            rtest.append((iv[w-test_size],iv[w]))
            result.append((rtrain,rtest))
            if self.verbose: print rtrain,rtest
#iv[w-test_size:w] 
        if self.verbose: print pair_name,len(self.data[pair_name])
        return result
    

    
  
     
    
        


 
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
        elif self.tformat == 'usgs':
            self.loader = TablesLoader('english')
        for file_name in os.listdir(self.input_dir):
            print(file_name)
            self.load(self.input_dir+os.sep+file_name)
            self.interpolate_missing()
            self.save()

    def save(self):
        rcsv = RcsvLoader()
        rcsv.copy(self)
        rcsv.save()
        
        
             
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




        
             
       
