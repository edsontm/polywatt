#!/usr/bin/python

import matplotlib.pyplot as plt
import string
import csv
import numpy as np
import sys
import os
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


if len(sys.argv) < 2:
    maindir = 'prepared_data'
else:
    maindir = sys.argv[1]
out_plot = 'igraphs2'
if not os.path.isdir(out_plot):
    os.mkdir(out_plot)

out_data = 'nolearning'
if not os.path.isdir(out_data):
    os.mkdir(out_data)

result = []


fres = open(out_data+os.sep+'nolearning_result.csv','w')
fres.write('origin,destiny,year,method,mae,r2\n')

hdata = dict()

for file_name in sorted(os.listdir(maindir)):
    print file_name
    vfile_name = file_name.split('_')
    r1_name = string.capwords(vfile_name[0])
    r2_name = string.capwords(vfile_name[1])

    datakey = r1_name.lower()+'_'+r2_name.lower()

    year   = vfile_name[2].split('-')[0]
    method = vfile_name[-1].split('.')[0]


    f = csv.reader(open(maindir+os.sep+file_name),delimiter=',')
    original = []
    header = None
    for line in f:
        if header == None:
            header = line
        else:
            v = [float(t) for t in line]
            original.append(v)
    if datakey not in hdata:
        hdata[datakey] = dict()
    if method not in hdata[datakey]:
        hdata[datakey][method] = dict()
        hdata[datakey][method]= []




    original = np.matrix(original)
    x = range(original.shape[0])
    nyears = len(x)/364
    start = (nyears-1)*364
    end = start + 364
    print len(x), end
    if len(x) < end:
        print "Sorry, can't compute %s"%(file_name)
        hdata[datakey][method].append((year,0,0))
    else:



        x = x[start:end]
        plot_dist = 0
        mean = np.mean(original[start:end,0])
        std = np.std(original[start:end,0])
        origin = np.array((original[start:end,0]  -mean)/std)

        mean = np.mean(original[start:end,1])
        std = np.std(original[start:end,1])
        destiny = np.array((original[start:end,1]  -mean)/std)

        pinterval = np.arange(end-start)
        #plt.subplot(2,1,1)
        plt.subplot2grid((4,1), (0,0), rowspan=3)
        l1, = plt.plot(pinterval,origin,'-',color='b',linewidth=3)
        l2, = plt.plot(pinterval,destiny,'-',color='g',linewidth=3)


        plt.legend((l1, l2), (r1_name, r2_name), loc='lower right', shadow=True)
        plt.xlabel('Julian days')
        plt.ylabel('Standard score')

        #plt.subplot(2,1,2)
        plt.subplot2grid((4,1), (3,0))
        pdiff = origin-destiny
        gz = pdiff > 0
        gz = gz.transpose()[0]
        lz = pdiff < 0
        lz = lz.transpose()[0]
        plt.ylim([-2,2])
        plt.bar(pinterval[gz],pdiff[gz],color='b')
        plt.bar(pinterval[lz],pdiff[lz],color='r')
        plt.ylabel('Absolute Error')

        plt.savefig(out_plot+os.sep+r1_name.lower()+'_'+r2_name.lower()+'_'+year+'_'+method+'.pdf')
        plt.clf()
        mae = mean_absolute_error(origin,destiny)
        r2  = r2_score(origin,destiny)
        tstr = r1_name.lower()+','+r2_name.lower()+','+year+','+method+','+str(mae)+','+str(r2)+'\n'
        hdata[datakey][method].append((year,mae,r2))
        fres.write(tstr)
fres.close()


for datakey in sorted(hdata.keys()):

    for metric in ['mae','r2']:
        fres = open(out_data+os.sep+'%s_%s.csv'%(datakey,metric),'w')
        data = []
        header = 'year'
        for method in sorted(hdata[datakey]):
            # add year
            for i in range(len(hdata[datakey][method])):
                tinst = hdata[datakey][method][i]
                if len(header)==4:
                    data.append('%s,%4.2f'%(tinst[0],tinst[1]))
                elif metric == 'mae':
                    data[i] += ',%4.2f'%(tinst[1]) 
                elif metric == 'r2':
                    data[i] += ',%4.2f'%(tinst[2]) 

            header += ','+method
        # save
        fres.write(header+'\n')
        for i in range(len(data)):
            fres.write(data[i]+'\n')
        fres.close()


            



     
    

