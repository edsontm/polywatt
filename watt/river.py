import os
import csv
import sys
import string


from sklearn.covariance import EllipticEnvelope

from scipy import stats
from numpy import linalg as la
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import collections  as mc
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import colorConverter

from sklearn.neighbors import KNeighborsRegressor

import getopt

import heapq

import dtw
from .empirical import EmpiricalFormulas
from .dataloader import RcsvLoader

class TestClass(object):

    def test_prepare_dataset(self):
        r = RiverDataset()
        r.load_pairs('stations.txt')
        r.load_dir('dataset')
        r.save_complete_years('complete_years')

                
    def test_pair_river(self):
        r = RiverDataset()
        r.load_complete_years('complete_years')
        RiverPair.run_all_pairs(r)

class RiverPairCache:
    def __init__(self):

        self.path = None
        self.dist = None
        self.cost = None
        self.interval = None
        
        self.lr1_raw=None
        self.lr2_raw=None

        self.lr1 = None
        self.lr2 = None

        self.topn1max = None
        self.topn2max = None

class RiverPair:
    def __init__(self,data,pair_name):
        tarray = np.matrix(data)
        self.pair_name = pair_name
        self.gr1_raw = np.array(tarray[:,0].transpose().tolist()[0])
        self.gr2_raw = np.array(tarray[:,1].transpose().tolist()[0])
        self.vcache = dict()
        self.verbose = True
        self.run_all = True
        self.glevels = dict()
        self.w = 30  # windows size
        self.save_with_header = True
        self.w_years=20
        self.test_size=1
   
    @staticmethod
    def run_all_pairs(r):
        out_dir = 'prepared_data'
        out_plot = 'graphs'
        if not os.path.isdir(out_dir):
           os.mkdir(out_dir)
        if not os.path.isdir(out_plot):
           os.mkdir(out_plot)

        e = EmpiricalFormulas() 
        e.load()
        for pair_name in r.data:
            p = RiverPair(r.data[pair_name],pair_name) # all data
            origin  = pair_name.split('_')[0]
 
            destiny = pair_name.split('_')[-1]
            train_test = r.split_train_test(pair_name,w_years=p.w_years,test_size=p.test_size)

            pp    = PdfPages('graphs/poly_'+pair_name+'.pdf')
            ppoly = PdfPages('graphs/matches_poly_'+pair_name+'.pdf')
            pddtw = PdfPages('graphs/matches_ddtw_'+pair_name+'.pdf')
            pdtw  = PdfPages('graphs/matches_dtw_'+pair_name+'.pdf')
            
            for (train,test) in train_test:
                print train,test
                for first_index in sorted(r.hstart_date[pair_name].keys()):
                    print r.hstart_date[pair_name][first_index]

                print r.hstart_date[pair_name]
                p.train_interval(train)
                p.test_interval(test)
                seyamList = []
                for i in range(len(train)):
                     
                    p.set_interval(train[i])

                    p.ddtw()
                    p.pip()
                    p.find_correspondent_points()
                    seyamList.append(p.seyam2014())

                    print 'llevels',p.llevels
                    p.save_cache()
                
                if len(p.glevels) > 5: # it must have at least 5 points to derive the points
                    first_index = train[0][0]
                    first_date = r.hstart_date[pair_name][first_index]
                    last_date = r.hstart_date[pair_name][train[-1][0]]

                    test_index = test[0][0]
                    test_date = r.hstart_date[pair_name][test_index]
                    print "first training date", first_date
                    print "last training date", last_date
                    print "first test date", test_date

                    p.find_poly()
                    training_years_range = first_date[:4]+'->'+last_date[:4]
                    p.save_plot(pp,training_years_range)
                    p.glevels = dict() 
                    

                    name = out_dir+os.sep+pair_name+'_'+test_date
                   
                    train_interval = (p.train_start,p.test_end) 

                    # original
                    method = lambda x:0
                    tname = name+'_train_pair_original.csv'
                    p.save_pair(tname,method,train_interval)



                    # poly
                    method = p.poly
                    tname = name+'_train_pair_poly.csv'
                    p.save_pair(tname,method,train_interval)

                    # seyam

                    print 'seyamList',seyamList
                    method = lambda x:np.mean(seyamList)
                    tname = name+'_train_pair_seyam.csv'
                    p.save_pair(tname,method,train_interval)
                    print 'seyam   %f'%(method(1))



                    L = e.L(origin,destiny)
                    S = e.S(origin,destiny)
                    # kirpich
                    method = lambda x:round(e.kirpich(L,S))
                    tname = name+'_train_pair_kirpich.csv'
                    p.save_pair(tname,method,train_interval)
                    print 'kirpich %f'%(method(1))

                    # johnstone
                    method = lambda x:round(e.johnstone(L,S))
                    tname = name+'_train_pair_johnstone.csv'
                    p.save_pair(tname,method,train_interval)
                    print 'johnstone %f'%(method(1))


                    # chow
                    method = lambda x:round(e.chow(L,S))
                    tname = name+'_train_pair_chow.csv'
                    p.save_pair(tname,method,train_interval)
                    print 'chow %f'%(method(1))
                    


            v = r.start_date[pair_name]
            iv = []
            for i in range(len(v)):
                iv.append(v[i][1])
            for i in range(len(iv)-1):
                tinterval = (iv[i],iv[i+1])
                tkey = RiverPair.interval2key(tinterval)
                if tkey not in p.vcache:
                    Exception('Error! %s is not valid'%(tkey))
                p.set_interval(tinterval)
                first_index = iv[i]
                first_date = r.hstart_date[pair_name][first_index]
                p.save_poly_plot(ppoly,first_date)
                if p.path == None:
                    p.ddtw()
                    p.pip()
                    p.find_correspondent_points()
                p.save_ddtw_plot(pddtw,first_date)
                p.save_dtw_plot(pdtw,first_date)
                #p.save_twoway_plot()


 


            pp.close()
            ppoly.close()
            pdtw.close()
            pddtw.close()


    
    
         
    @staticmethod
    def interval2key(interval):
        return "%d_%d"%(interval[0],interval[1])
                


    def load_cache(self):
        key = self.key
        self.path       = self.vcache[key].path
        self.dist       = self.vcache[key].dist
        self.cost       = self.vcache[key].cost
        self.interval   = self.vcache[key].interval
        
        self.lr1_raw    = self.vcache[key].lr1_raw
        self.lr2_raw    = self.vcache[key].lr2_raw

        self.lr1        = self.vcache[key].lr1
        self.lr2        = self.vcache[key].lr2

        self.topn1max   = self.vcache[key].topn1max
        self.topn2max   = self.vcache[key].topn2max
        
        self.llevels    = self.vcache[key].llevels
        self.seyam      = self.vcache[key].seyam


    def save_cache(self):
        key = self.key
        self.vcache[key].path        =     self.path      
        self.vcache[key].dist        =     self.dist      
        self.vcache[key].cost        =     self.cost      
        self.vcache[key].interval    =     self.interval  
                                           
        self.vcache[key].lr1_raw     =     self.lr1_raw   
        self.vcache[key].lr2_raw     =     self.lr2_raw   
                                                          
        self.vcache[key].lr1         =     self.lr1       
        self.vcache[key].lr2         =     self.lr2       
                                                          
        self.vcache[key].topn1max    =     self.topn1max  
        self.vcache[key].topn2max    =     self.topn2max  

        self.vcache[key].llevels     =     self.llevels 
        self.vcache[key].seyam       =     self.seyam
        
        for tkey in self.llevels:
            if tkey not in self.glevels:
                self.glevels[tkey] = []
            self.glevels[tkey] += self.llevels[tkey]

                 




    def set_interval(self,interval):
        self.path = None
        self.dist = None
        self.cost = None
        self.interval = None
        
        self.lr1_raw=None
        self.lr2_raw=None

        self.lr1 = None
        self.lr2 = None

        self.topn1max = None
        self.topn2max = None
        
        self.llevels = dict()




        self.key  = RiverPair.interval2key(interval)

        if self.verbose: print self.key

        if self.key in self.vcache:
            self.load_cache()
            self.run_all = False
        else:
            self.run_all = True
            self.vcache[self.key] = RiverPairCache()

            self.interval = interval
        
            self.lr1_raw=self.gr1_raw[interval[0]:interval[1]]
            self.lr2_raw=self.gr2_raw[interval[0]:interval[1]]

            self.lr1 = stats.zscore(self.lr1_raw)
            self.lr2 = stats.zscore(self.lr2_raw)
        
    def ddtw(self):
        if self.run_all:
            dist,cost,path = dtw.ddtw(self.lr1,self.lr2)
            self.dist = dist
            self.cost = cost
            self.path = path
        
    def pip(self):
        if self.run_all:
            ssize = len(self.lr1)
            topn = int((ssize/364.0)*20)
            self.topn1max = PairAnalysis.pipe(self.lr1,topn)
            self.topn2max = PairAnalysis.pipe(self.lr2,topn)

    def find_correspondent_points(self):
        if self.run_all:
            sr1 = self.path[0]
            sr2 = self.path[1]
            points = []
            for i in range(len(sr1)):
                diff = sr2[i] - sr1[i]

                ox = sr1[i]
                oy = self.lr1[ox]

                dx = sr2[i]
                dy = self.lr2[dx]

                if ox in self.topn1max and dx in self.topn2max and diff >= 0:
                    r1_raw_level = self.lr1_raw[sr1[i]]
                    if r1_raw_level not in self.llevels:
                        self.llevels[r1_raw_level] = []
                    self.llevels[r1_raw_level].append(diff)

    
    def seyam2014(self):
        if self.run_all:
            maxdayr1 = np.argmax(self.lr1)
            maxdayr2 = np.argmax(self.lr2)
            self.seyam = abs(maxdayr1 - maxdayr2)
        return self.seyam

       
        

    def find_poly(self):
        X = []
        y = []
        for tkey in sorted(self.glevels):    
            X.append(tkey)
            y.append(np.mean(self.glevels[tkey]))
        z = np.polyfit(X, y, 2)
        self.poly = np.poly1d(z)
    def find_knn(self):
        X = []
        y = []
        for tkey in sorted(self.glevels):    
            X.append([tkey])
            y.append(np.mean(self.glevels[tkey]))
        print X,y
        
        #knn = KNeighborsRegressor(weights='distance')    

      #  if len(X) > 7:
      #      param_grid = {"n_neighbors":[1,2,3,4,5]}
      #      random_search = GridSearchCV(knn, param_grid=param_grid,
      #                             scoring='mean_absolute_error') 

      #      random_search.fit(X,y)
      #      self.knn = random_search.best_estimator_

       # else: 
        knn = KNeighborsRegressor(weights='distance',n_neighbors=len(X))    
        knn.fit(X,y)
        self.knn = knn
        knn = KNeighborsRegressor(weights='distance',n_neighbors=1)    
        knn.fit(X,y)
        self.nn1 = knn





    def save_plot(self,pp,title):
        X = []
        y = []
        for tkey in sorted(self.glevels):    
            X.append(tkey)
            y.append(np.mean(self.glevels[tkey]))
        plt.title(title)

        print title
        for i in range(len(X)):
            print X[i],y[i]
        plt.plot(X,y,'o')
        tx = range(int(min(X)),int(max(X)))
        plt.xlabel('River level (cm)')
        plt.ylabel('WaTT (days)')
        plt.plot(tx,self.poly(tx))
        plt.savefig(pp,format='pdf')
        plt.clf()


    def save_ddtw_plot(self,pp,title):
            plot_dist = 3
            fig = plt.figure(2)
            ax = fig.add_subplot(111)
            x1 = np.array(range(len(self.lr1)))
            plt.title(title)
            plt.plot(x1,self.lr1)
            plt.plot(x1,self.lr2-plot_dist)
            sr1 = self.path[0]
            sr2 = self.path[1]
            lines = []
            lines_dtw = []
            points = []
            for i in range(len(sr1)):

                diff = sr2[i] - sr1[i]
                ox = sr1[i]
                oy = self.lr1[ox]

                dx = sr2[i]
                dy = self.lr2[dx] - plot_dist

                line = [(ox,oy),(dx,dy)]
                if ox in self.topn1max and dx in self.topn2max and diff >= 0:
                        lines.append(line)
                        points.append( ((ox+dx)/2.0,(oy+dy)/2.0,diff))
                else:
                        if i%10==0:
                            lines_dtw.append(line)

            l1, = plt.plot(x1,self.lr1,color='b',linewidth=3)                 # serie temporal rio 1
  #          plt.plot(x1[self.topn1max],self.lr1[self.topn1max],'ro')          # pontos de maximo

            l2, = plt.plot(x1,self.lr2-plot_dist,color='g',linewidth=3)       # serie temporal rio 2
  #          plt.plot(x1[self.topn2max],self.lr2[self.topn2max]-plot_dist,'ro') # pontos de minimo
            plt.xlabel("Julian day")

            v = self.pair_name.split('_')
            r1_name = string.capwords(v[0])
            r2_name = string.capwords(v[1])

            plt.legend((l1,l2), (r1_name, r2_name), loc='lower right', shadow=True)
            lc = mc.LineCollection(lines,colors=(1,0,0))
            lc2 = mc.LineCollection(lines_dtw,colors=(0.5019607843137255, 0.5019607843137255, 0.5019607843137255))
       #     ax.add_collection(lc)
            ax.add_collection(lc2)
       #     for (y,x,diff) in points:
       #         ax.text(y+2,x,diff)                
            plt.savefig(pp,format='pdf')
            plt.clf()




    def save_dtw_plot(self,pp,title):
            plot_dist = 3
            fig = plt.figure(2)
            ax = fig.add_subplot(111)
            x1 = np.array(range(len(self.lr1)))
            plt.title(title)
            plt.plot(x1,self.lr1)
            plt.plot(x1,self.lr2-plot_dist)
            dist,cost,path = dtw.dtw(self.lr1,self.lr2)
            sr1 = path[0]
            sr2 = path[1]
            lines = []
            lines_dtw = []
            points = []
            for i in range(len(sr1)):

                diff = sr2[i] - sr1[i]
                ox = sr1[i]
                oy = self.lr1[ox]

                dx = sr2[i]
                dy = self.lr2[dx] - plot_dist

                line = [(ox,oy),(dx,dy)]
                if ox in self.topn1max and dx in self.topn2max and diff >= 0:
                        lines.append(line)
                        points.append( ((ox+dx)/2.0,(oy+dy)/2.0,diff))
                else:
                        if i%10==0:
                            lines_dtw.append(line)

            l1, = plt.plot(x1,self.lr1,color='b',linewidth=3)                 # serie temporal rio 1
    #        plt.plot(x1[self.topn1max],self.lr1[self.topn1max],'ro')          # pontos de maximo

            l2, = plt.plot(x1,self.lr2-plot_dist,color='g',linewidth=3)       # serie temporal rio 2
     #       plt.plot(x1[self.topn2max],self.lr2[self.topn2max]-plot_dist,'ro') # pontos de minimo
            plt.xlabel("Julian day")

            v = self.pair_name.split('_')
            r1_name = string.capwords(v[0])
            r2_name = string.capwords(v[1])

            plt.legend((l1,l2), (r1_name, r2_name), loc='lower right', shadow=True)
            lc = mc.LineCollection(lines,colors=(1,0,0))
            lc2 = mc.LineCollection(lines_dtw,colors=(0.5019607843137255, 0.5019607843137255, 0.5019607843137255))
       #     ax.add_collection(lc)
            ax.add_collection(lc2)
       #     for (y,x,diff) in points:
       #         ax.text(y+2,x,diff)                
            plt.savefig(pp,format='pdf')
            plt.clf()








    def save_twoway_plot(self):
        import rpy2.robjects.numpy2ri
        rpy2.robjects.numpy2ri.activate()
        from rpy2.robjects.packages import importr
        R = rpy2.robjects.r
        DTW = importr('dtw')
        alignment = R.dtw(self.lr1, self.lr2, keep=True) 
        R.plot(alignment,type="threeway")

        
    

    def save_poly_plot(self,pp,title):
        plot_dist = 3
        fig = plt.figure(2)
        ax = fig.add_subplot(111)

        x1 = np.array(range(len(self.lr1)))
        plt.xlabel("Julian day")
        plt.title(title)
        plt.plot(x1,self.lr1,'-',color='b',linewidth=3)
        plt.plot(x1,self.lr2 - plot_dist,color='g',linewidth=3)
        lines = []
        vdiff = []
        points = []
        for i in range(len(self.lr1)):
            diff = int (self.poly(self.lr1_raw[i]))

            if i+diff < len(self.lr2) and i+diff > 0:

                ox = i
                oy = self.lr1[ox]

                dx = i+diff
                dy = self.lr2[dx] - plot_dist

                if i%10 == 0:
                    line = [(ox,oy),(dx,dy)]
                    vdiff.append(diff)   
                    lines.append(line)
                    points.append( ((ox+dx)/2.0,(oy+dy)/2.0,diff,{'fontsize': 10}))

        lc = mc.LineCollection(lines,colors=(0.5019607843137255, 0.5019607843137255, 0.5019607843137255))
        ax.add_collection(lc)
        for (y,x,diff,fsize) in points:
            ax.text(y+1,x,diff,fsize)

        plt.savefig(pp,format='pdf')
        plt.clf()

       


    def train_interval(self,train):
        self.train_start = train[0][0]
        self.train_end    = train[-1][1]
#        print self.train_start
#        print self.train_end

    def test_interval(self,test):
        self.test_start  = test[0][0]
        self.test_end    = test[-1][1]
#        print self.test_start
#        print self.test_end



    def save_pair(self,name,method,interval):
        s = interval[0]
        e = interval[1]
        r1 = self.gr1_raw[s:e]
        r2 = self.gr2_raw[s:e]

        f = open(name,'w')

        if self.save_with_header:f.write('att1,class\n')
        size = len(r1)
        for i in range(size):
            instance = r1[i]
            diff = method(instance)

            if i+diff < size:
                y = r2[int(i+diff)]
                f.write('%4.2f,%4.2f\n'%(instance,y))
        f.close()



    def save_window(self,name,method,interval):
        f = open(name,'w')
        s = interval[0]
        e = interval[1]
        r1 = self.gr1_raw[s:e]
        r2 = self.gr2_raw[s:e]
        size = len(r1)

        w = self.w
        if self.save_with_header:
            theader = ''
            for i in range(w):
                theader +='att%d,'%i
            theader += 'class\n'
            f.write(theader)

        for i in range(w,size):
            instance = r1[i-w:i]
            diff = int(method(r1[i])+0.5)
            if i+diff < size:
                tstr = ','.join([str(v) for v in instance]) 
                f.write('%s,%4.2f\n'%(tstr,r2[i+diff]))
        f.close()

    def save_test_pair(self,name):
        s = self.test_start
        e = self.test_end
        r1 = self.gr1_raw[s:e]
        r2 = self.gr2_raw[s:e]

        f = open(name,'w')
        size = len(r1)
        for i in range(size):
            instance = r1[i]
            diff = method(instance)

            if i+diff < size:
                y = r2[i+diff]
                f.write('%4.2f,%4.2f\n'%(instance,y))
        f.close()

        pass

    def save_test_window(self,name):
        pass





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



    def save_complete_years(self,complete_dir,start_date = '07-01'):
        if self.pairs == None:
            raise Exception('sorry, you must execute load_pairs first')
        else:
            if not os.path.isdir(complete_dir):
                os.mkdir(complete_dir)
            tcount = 0
            v = start_date.split('-')
            split_month = v[0]     
            split_day   = v[1] 
            for pair in self.pairs:
                r1 = self.river_stations[pair[0]]
                r2 = self.river_stations[pair[1]]
                common_dates = r1.find_common_intervals(r2)
                for (diff,start1,end1) in common_dates:
                        split_points = []
                        for i in range(start1,end1):
                            date = r1.vdates[i]
                            v    = date.split('-')
                            month = v[1]
                            day   = v[2]
                            if (day == split_day and month == split_month):
                                split_points.append(i)
                        for i in range(len(split_points) -1):
                            start1 = split_points[i]
                            end1   = split_points[i+1]   
                            tcount+=1
                            
                            dstart = r1.vdates[start1] 
                            dend   = r1.vdates[end1]

                            start2 = r2.hdates[dstart]
                            end2   = r2.hdates[dend]

                            v1     = r1.levels[start1:end1]
                            v2     = r2.levels[start2:end2]
                            if (v1.count(-1) == 0 and v2.count(-1) == 0):
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
    

    


class PairAnalysis:
    def __init__(self,r1,r1_name,r2,r2_name,rlevels = None):
        self.raw_r1 = r1
        self.raw_r2 = r2 
        self.r1 = np.array(self.zscore(r1))
        self.r2 = np.array(self.zscore(r2))
        self.r1_name = r1_name
        self.r2_name = r2_name
        self.rlevels = rlevels
        self.smooth = False
        self.X = []
        self.y = []
        #self.r1 = np.array([0,0,0,0,1,1,2,2,3,2,1,1,0,0,0,0])
        #self.r2 = np.array([0,0,1,1,2,2,3,3,3,3,2,2,1,1,0,0])

    @staticmethod
    def pipmi(y):
        xaxis = []
        pstart = np.array([0,y[0]])
        pend   = np.array([len(y),y[-1]])
        de = []
        for i in range(len(y)):
            dstart = abs(y[i]-y[0])
            dend   = abs(y[-1] -y[i])
#            dstart = np.array([i,y[i]]) - pstart
#            dend   = pend - np.array([i,y[i]])
            de.append(dstart+dend)
        return de

    @staticmethod
    def pipei(y):
        xaxis = []
        pstart = np.array([0,y[0]])
        pend   = np.array([len(y),y[-1]])
        de = []
        for i in range(len(y)):
            dstart = la.norm(np.array([i,y[i]]) - pstart)
            dend   = la.norm(pend - np.array([i,y[i]]))
            de.append(dstart+dend-len(y))
        
        return de
    @staticmethod
    def pipe(y,percentage):
        #print y
        y = np.array(y)
        theap = []
        selected = []
        sizey = len(y)
        if percentage < 1:
            nselected = sizey*percentage
            if nselected < 1:
                nselected = 1
        else:
            nselected = percentage
        heapq.heappush(theap,(-sizey,0,range(sizey)))
        heapq.heappush(theap,(-len(y),0,range(sizey)))
        while len(theap) > 0 and len(selected) < nselected:
            (size,ini,m) = heapq.heappop(theap) 
            #print 'analyzing ',m,y[m]
            piper = PairAnalysis.pipmi(y[m])
            p = np.argmax(piper)
            #print 'piper',piper
            #print 'returned ini %d p %d piper %f'%(ini,p,piper[p])
            if piper[p] > 1.0  and np.std(piper) > 0.01:
                selected.append(p+ini)
                idir = m[:p]
                iesq = m[p+1:]
                if len(idir) >nselected :heapq.heappush(theap,(-np.std(idir),ini,idir))
                if len(iesq) >nselected :heapq.heappush(theap,(-np.std(iesq),p+ini+1,iesq))

#                if len(idir) >sizey*0.1 :heapq.heappush(theap,(-len(idir),ini,idir))
#                if len(iesq) >sizey*0.1 :heapq.heappush(theap,(-len(iesq),p+ini+1,iesq))
                #print selected
        return selected
 
    @staticmethod
    def pipe_test(y,percentage):
        #print y
        y = np.array(y)
        theap = []
        selected = []
        sizey = len(y)
        if percentage < 1:
            nselected = sizey*percentage
            if nselected < 1:
                nselected = 1
        else:
            nselected = percentage
        heapq.heappush(theap,(-sizey,0,range(sizey)))
        heapq.heappush(theap,(-len(y),0,range(sizey)))
        while len(theap) > 0 and len(selected) < nselected:
            (size,ini,m) = heapq.heappop(theap) 
            #print 'analyzing ',m,y[m]
            piper = PairAnalysis.pipmi(y[m])
            p = np.argmax(piper)
            print 'piper',piper
            print 'returned ini %d p %d piper %f'%(ini,p,piper[p])
            if piper[p] > 1.0  and np.std(piper) > 0.01:
                selected.append(p+ini)
                idir = m[:p]
                iesq = m[p+1:]
                if len(idir) >nselected :heapq.heappush(theap,(-np.std(idir),ini,idir))
                if len(iesq) >nselected :heapq.heappush(theap,(-np.std(iesq),p+ini+1,iesq))

#                if len(idir) >sizey*0.1 :heapq.heappush(theap,(-len(idir),ini,idir))
#                if len(iesq) >sizey*0.1 :heapq.heappush(theap,(-len(iesq),p+ini+1,iesq))
                #print selected
        return selected
 


    @staticmethod
    def smooth_static(levels):
        t = levels[:]
        for i in range(1,len(levels)):
#            if None not in levels[i:-1]:
            t[i] = (levels[i]+levels[i-1])/2.0
        return t


    def zscore(self,x):
        mean = np.mean(x)
        std = np.std(x)
        return [(v-mean)/std for v in x]

    def use_poly(self,poly):

        plot_dist = 3
        if self.smooth:
            r1 = PairAnalysis.smooth_static(self.r1)
            r2 = PairAnalysis.smooth_static(self.r2)
        else:
            r1 = PairAnalysis.smooth_static(self.r1)
            r2 = PairAnalysis.smooth_static(self.r2)

     
        ssize = len(self.r1)
        #print topn1max
        #print topn2max

        fig = plt.figure(2)
        ax = fig.add_subplot(111)
        x1 = np.array(range(len(self.r1)))
        plt.plot(x1,r1)
        plt.plot(x1,r2 - plot_dist)
        lines = []
        lines_dtw = []
        points = []
        vdiff = []
        projection = []
         
        for i in range(len(r1)):
            diff = int( poly(self.raw_r1[i]))
#            if i+diff < len(r2):
#                projection.append(r2[i+diff])



            if i+diff < len(r2):

                ox = i
                oy = r1[ox]

                dx = i+diff
                dy = r2[dx] - plot_dist


                instance = [self.raw_r1[ox],diff]
                self.X.append(instance)
                self.y.append(self.raw_r2[dx])



                if i%10 == 0:

                    line = [(ox,oy),(dx,dy)]

                    vdiff.append(diff)   
                    lines.append(line)
                    vdiff.append(diff)
                    r1_raw_level = self.raw_r1[i] 
                    r2_raw_level = self.raw_r2[i+diff] 
                    points.append( ((ox+dx)/2.0,(oy+dy)/2.0,diff,{'fontsize': 10}))

                #print r1_raw_level, r2_raw_level,diff

        lc = mc.LineCollection(lines)
        ax.add_collection(lc)
        for (y,x,diff,fsize) in points:
            ax.text(y+1,x,diff,fsize)




        

    def find_peak_valley(self,training_limit = 0):

        plot_dist = 3
        if self.smooth:
            r1 = PairAnalysis.smooth_static(self.r1)
            r2 = PairAnalysis.smooth_static(self.r2)
        else:
            r1 = PairAnalysis.smooth_static(self.r1)
            r2 = PairAnalysis.smooth_static(self.r2)

     
        dist, cost, path = dtw.ddtw(r1, r2)
        ssize = len(self.r1)
        topn = int((ssize/365.0)*20)
        topn1max = PairAnalysis.pipe(r1,topn)
        topn2max = PairAnalysis.pipe(r2,topn)
        #print topn1max
        #print topn2max

        fig = plt.figure(2)
        ax = fig.add_subplot(111)
        x1 = np.array(range(len(self.r1)))
        plt.plot(x1,self.r1)
        plt.plot(x1,self.r2-plot_dist)
        sr1 = path[0]
        sr2 = path[1]
        lines = []
        lines_dtw = []
        points = []
#        print 'Path'
        vdiff = []
        ant = -1
        hashmap = dict()
        tvdiff = [-20]*10
         
        for i in range(len(sr1)):

            diff = sr2[i] - sr1[i]
            tvdiff.append(diff)


            ox = sr1[i]
            oy = self.r1[ox]

            dx = sr2[i]
            dy = self.r2[dx] - plot_dist
#            sderivative =abs(dr1[ox]) + abs(dr2[dx])


            line = [(ox,oy),(dx,dy)]
            try:
                r1s1 = (self.raw_r1[ox-1] - self.raw_r1[ox])
                r1s2 = (self.raw_r1[ox+1] - self.raw_r1[ox])

                r2s1 = (self.raw_r2[dx-1] - self.raw_r2[dx])
                r2s2 = (self.raw_r2[dx+1] - self.raw_r2[dx])
            except IndexError:
               r1s1 = 0
               r1s2 = 0
               r2s1 = 0
               r2s2 = 0
                
            #print '%04.3f %04.3f %04.3f --- %04.3f %04.3f %04.3f '%(r1s1,r1s2,r1s1*r1s2,r2s1,r2s2,r2s1*r2s2)
            # if  peak1[ox] > 0 or peak2[dx] > 0:
            #if ox != ant:
            #    instance = [self.raw_r1[ox],diff]
            #    self.X.append(instance)
            #    self.y.append(self.raw_r2[dx])
            #    ant = ox

            if tvdiff[-1] == tvdiff[-2] == tvdiff[-3]==tvdiff[-4] and diff > 0 :
                instance = [self.raw_r1[ox],diff]
                print instance
                self.X.append(instance)
                self.y.append(self.raw_r2[dx])


                    
            #if tvdiff[-1] == tvdiff[-2] == tvdiff[-3]==tvdiff[-4]==tvdiff[-5] and diff > 0 :
            if ox in topn1max and dx in topn2max and diff >= 0:
#and r1s1*r1s2 > 0 and r2s1*r2s2 > 0: 
                    vdiff.append(diff)
                    r1_raw_level = self.raw_r1[sr1[i]] 
                    r2_raw_level = self.raw_r2[sr2[i]] 

                    lines.append(line)
                    points.append( ((ox+dx)/2.0,(oy+dy)/2.0,diff))
                    print sr1[i],sr2[i], diff,r1_raw_level, r2_raw_level
                    if training_limit > 0 and sr1[i] < training_limit:

                        print 'added',sr1[i],training_limit
                        if r1_raw_level not in self.rlevels: 
                           self.rlevels[r1_raw_level] = []
                        self.rlevels[r1_raw_level].append(diff)
                    else:
                        print 'not added'
            else:
                    if i%10==0:
                        lines_dtw.append(line)


                    #        print 'R1'
#        for i in range(10):
#            print (i,self.r1[i])
#        print 'R2'
#        for i in range(10):
#            print (i,self.r2[i])

        # plot top and bottom points            
        l1, = plt.plot(x1,self.r1,color='b',linewidth=3)
        plt.plot(x1[topn1max],self.r1[topn1max],'ro')
        l2, = plt.plot(x1,self.r2-plot_dist,color='g',linewidth=3)
        plt.plot(x1[topn2max],self.r2[topn2max]-plot_dist,'ro')
        plt.xlabel("Julian day")

        r1_name = string.capwords(self.r1_name.replace('.csv',''))
        r2_name = string.capwords(self.r2_name.replace('.csv',''))

        plt.legend((l1,l2), (r1_name, r2_name), loc='lower right', shadow=True)


        lc = mc.LineCollection(lines,colors=(1,0,0))
        lc2 = mc.LineCollection(lines_dtw,colors=(0.5019607843137255, 0.5019607843137255, 0.5019607843137255))
        ax.add_collection(lc)
        ax.add_collection(lc2)
        for (y,x,diff) in points:
            ax.text(y+2,x,diff)
        ndiff = len(vdiff)
        for level in sorted(self.rlevels):
            print level,np.mean(self.rlevels[level])
        print '--->',len(self.X),len(self.y)
        return path

    def save_csv(self,file_name):
        f = open(file_name,'a')
        for i in range(len(self.X)):
            #tstr = ','.join([str(v) for v in self.X[i]])
            tstr = ','.join([str(v) for v in self.X[i]])
            f.write(tstr+','+str(self.y[i])+'\n')
        f.close()

    def save_original(self,file_name):
        f = open(file_name,'a')
        for i in range(len(self.raw_r1)):
            f.write("%d,%d\n"%(self.raw_r1[i],self.raw_r2[i]))
        f.close()

        

     
        
    def show_pip_points(self):
        dist, cost, path = dtw.ddtw(self.r1, self.r2)
        ssize = len(self.r1)
        topn = int((ssize/365.0)*20)
        topn1max = PairAnalysis.pipe(self.r1,topn)
        topn2max = PairAnalysis.pipe(self.r2,topn)

        fig = plt.figure(2)
        ax = fig.add_subplot(111)
        x1 = np.array(range(len(self.r1)))
        plt.plot(x1,self.r1)
        #plt.plot(x1[topn1max],self.r1[topn1max],'ro')
        plt.plot(x1,self.r2-5)
        #plt.plot(x1[topn2max],self.r2[topn2max]-5,'ro')

    @staticmethod
    def remove_outliers(data):
        elliptic = EllipticEnvelope(contamination=0.25)
        elliptic.fit(data)
        outlier_training =np.array(data)
        vpred = elliptic.predict(outlier_training)
        return vpred

    @staticmethod
    def remove_outliers_list(tlist):
        data  = zip(np.random.random(len(tlist)),tlist)
        vpred = PairAnalysis.remove_outliers(data)
        rlist = []
        for i in range(len(vpred)):
            if vpred[i] > 0:
                rlist.append(tlist[i])
        return rlist

        
        
    @staticmethod    
    def get_delay(r0,r1):

    
        common_dates = r0.find_common_intervals(r1)

        # record matched levels
        rlevels = dict()
        data_name = r0.river_name.replace('.csv','_')+r1.river_name.replace('.csv','')
        pp = PdfPages('graphs/'+data_name+'.pdf')

        for tdate in sorted(common_dates,key = lambda x: x[0],reverse=True):
            (d,start,end) = tdate
            dstart = r0.vdates[start]
            dend   = r0.vdates[end]
            if (end - start) > 355:
                print r0.str_date(dstart)
                print r0.str_date(dend)
                plt.title(dstart+' ---> '+dend)
                d = PairAnalysis(r0.levels[start:end],r0.river_name,r1.levels[r1.hdates[dstart]:r1.hdates[dend]],r1.river_name,rlevels)
                d.save_original('out/'+data_name+'_'+dstart+'_original.csv')
                dtw_path = d.find_peak_valley(training_limit = (r1.hdates['1990-01-01']- r1.hdates[dstart]  ))
                d.save_csv('out/'+data_name+'_'+dstart+'_ddtw.csv')
                plt.savefig(pp,format='pdf')
                plt.clf()

        pp.close()
        outlier_training = []
        for level in sorted(rlevels):
            outlier_training.append([level,np.mean(rlevels[level])])
        vpred = remove_outliers(outlier_training)
        x = []
        y = []
        for i in range(len(vpred)):
            if vpred[i] >0:
                x.append(outlier_training[i][0])
                y.append(outlier_training[i][1])
                print outlier_training[i]
        z =  np.polyfit(x,y,3)
        xp = np.linspace(min(x),max(x),100)
        p = np.poly1d(z)

        plt.plot(x,y,'.',xp,p(xp),'-')
        plt.savefig('graphs/polyfit'+data_name+'.pdf',format='pdf')
        plt.clf()

        return p,dtw_path
    @staticmethod 
    def eval_delay(r0,r1,poly):
        common_dates = r0.find_common_intervals(r1)

        data_name = r0.river_name.replace('.csv','_')+r1.river_name.replace('.csv','')
        pp = PdfPages('graphs/'+data_name+'_pvddtw.pdf')

        for tdate in sorted(common_dates,key = lambda x: x[0],reverse=True):
            (d,start,end) = tdate
            dstart = r0.vdates[start]
            dend   = r0.vdates[end]
            if (end - start) > 355:
                print r0.str_date(dstart)
                print r0.str_date(dend)
                plt.title(dstart+' ---> '+dend)
                d = PairAnalysis(r0.levels[start:end],r0.river_name,r1.levels[r1.hdates[dstart]:r1.hdates[dend]],r1.river_name)
                d.use_poly(poly)

                d.save_csv('out/'+data_name+'_'+dstart+'_pvddtw.csv')
                plt.savefig(pp,format='pdf')
                plt.clf()


        pp.close()


    def prepare_ndays_data(self,w=30,delay_method=None):
        r1 = self.raw_r1
        r2 = self.raw_r2
        size = len(r1)
        if isinstance(delay_method,np.poly1d):
            for i in range(w,size):
                instance = r1[i-w:i]
                dif = int((delay_method(r1[i])+0.5))
                if i+dif < size:
                    self.X.append(instance)
                    self.y.append(r2[i+dif])
        elif isinstance(delay_method,tuple):
             d1 = delay_method[0]
             d2 = delay_method[1]
             dv = []
             for i in range(len(d1)):
                 dv.append(r1[d1[i]])
           

             for i in range(w,size):
                instance = r1[i-w:i]
                tindex = dv.index(r1[i])
                dif = d2[tindex] - d1[tindex]

                if i+dif < size:
                    self.X.append(instance)
                    self.y.append(r2[i+dif])
     
        else:
            for i in range(w,size):
                instance = r1[i-w:i]

                self.X.append(instance)
                self.y.append(r2[i])
       


    @staticmethod 
    def build_ndays_data(r0,r1,w=30,delay_method=None):
        
        common_dates = r0.find_common_intervals(r1)

        data_name = r0.river_name.replace('.csv','_')+r1.river_name.replace('.csv','')

    
        if isinstance(delay_method,np.poly1d):
            comp_str = "%d_pvddtw"%(w)
        elif isinstance(delay_method,tuple):
            comp_str = "%d_ddtw"%(w)
        else:
            comp_str = "%d_original"%(w)

        for tdate in sorted(common_dates,key = lambda x: x[0],reverse=True):
            (d,start,end) = tdate
            dstart = r0.vdates[start]
            dend   = r0.vdates[end]
            if (end - start) > 355:
                d = PairAnalysis(r0.levels[start:end],r0.river_name,r1.levels[r1.hdates[dstart]:r1.hdates[dend]],r1.river_name)
                d.prepare_ndays_data(w,delay_method=delay_method)
                d.save_csv('out/'+data_name+'_'+dstart+'_window%s.csv'%(comp_str))
        






class RiverStation:
    def __init__(self):
        self.river_name = None
        self.levels = None
        self.vdates = None
        self.hdates = None
        self.cmonth = {'ago': 8, 'dez': 12, 'mar': 3, 'fev': 2, 'jun': 6, 'jul': 7, 'jan': 1, 'abr': 4, 'set': 9, 'mai': 5, 'nov': 11, 'out': 10}
        self.loader = RcsvLoader() 

    def load_data(self,dataset_file):
        self.loader.load(dataset_file)
        self.river_name = self.loader.river_name
        self.vdates = self.loader.vdates
        self.hdates = self.loader.hdates
        self.levels = self.loader.levels
        

    def find_common_intervals_old(self,river):
        
        start_date = self.vdates[0]
        end_date   = self.vdates[-1]
        if self.vdates[0]  < river.vdates[0] : start_date = river.vdates[0]
        if self.vdates[-1] > river.vdates[-1]: end_date   = river.vdates[-1]
        all_dates = dict()
        for date in self.vdates:
            all_dates[date] = 1
        for date in river.vdates:
            if date in all_dates:
                all_dates[date] += 1
        dates_vector = []
        status = 'invalid'
        ret    = []
        index1 = None
        for date in sorted(all_dates.keys()):
            if date >= start_date and date <= end_date: 
                if all_dates[date] > 1: # compute days present in both rivers
                    index1 = self.hdates[date]
                    index2 = river.hdates[date]
                    v1 = self.levels[index1]
                    v2 = river.levels[index2]
                    if status == 'invalid':
                        if v1 != None and v2 !=None:  # Valid pair!!!
                            status = 'valid'
                            start = index1
                    elif status == 'valid':
                        if v1 == None or v2 == None: #  Invalid pair
                            status = 'invalid'
                            end = index1-1
                            ret.append((end-start,start,end))
        if status == 'valid':
                end = index1
                ret.append((end-start,start,end))
                            
        return ret            
        
        
       




    def find_common_intervals(self,river):
        
        start_date = self.vdates[0]
        end_date   = self.vdates[-1]
        if self.vdates[0]  < river.vdates[0] : start_date = river.vdates[0]
        if self.vdates[-1] > river.vdates[-1]: end_date   = river.vdates[-1]
        all_dates = dict()
        for date in self.vdates:
            all_dates[date] = 1

        for date in river.vdates:
            if date in all_dates:
                all_dates[date] += 1

        dates_vector = []
        status = 'invalid'
        ret    = []
        index1 = None
        for date in sorted(all_dates.keys()):
            if date >= start_date and date <= end_date: 
                if all_dates[date] > 1: # compute days present in both rivers
                    index1 = self.hdates[date]
                    index2 = river.hdates[date]
                    v1 = self.levels[index1]
                    v2 = river.levels[index2]
                    if status == 'invalid':
                        if v1 != None and v2 !=None:  # Valid pair!!!
                            status = 'valid'
                            start = index1
                    elif status == 'valid':
                        if v1 == None or v2 == None: #  Invalid pair
                            status = 'invalid'
                            end = index1-1
                            ret.append((end-start,start,end))
        if status == 'valid':
                end = index1
                ret.append((end-start,start,end))
        return ret            
        
        
       


    def plot(self,plt,trange=None):
        x1 = np.array(range(len(self.levels)))
        if trange == None:
            trange = x1
        plt.title(self.river_name)
        plt.plot(x1[trange],np.array(self.levels)[trange])

    def info(self):
        return '%s (%s %s) %d'%(self.river_name,self.vdates[0],self.vdates[-1],len(self.levels))
                        
                    
                     


    
    def str_date(self,sdate):
        index = self.hdates[sdate]
        try:
            return 'date %s index %d value %d'%(sdate,index,self.levels[index])
        except TypeError:
            return 'date %s index %d value None'%(sdate,index)

