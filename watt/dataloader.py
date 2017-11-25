import os
import csv
import numpy as np

class TablesLoader:
    def __init__(self,variant='portuguese'):
        if variant == 'english':
            self.cmonth = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
        elif variant == 'portuguese':
            self.cmonth = {'ago': 8, 'dez': 12, 'mar': 3, 'fev': 2, 'jun': 6, 'jul': 7, 'jan': 1, 'abr': 4, 'set': 9, 'mai': 5, 'nov': 11, 'out': 10}

    def load(self,dataset_file):
        self.river_name = dataset_file.split(os.sep)[-1]
        f = csv.reader(open(dataset_file),delimiter=',')
        tmatrix = []
        for row in f:
            tmatrix.append(row)
        matrix = np.matrix(tmatrix)
        years = matrix[0].tolist()[0][1:]
        days = matrix[1:,0].transpose().tolist()[0]
        i = 0
        self.vdates = []
        self.hdates = dict()
        for year in years:
            for date in days:
                v = date.split('-')
                if len(v) == 2:
                    day = v[0]
                    month = self.cmonth[v[1].lower()]
                    sdate = '%s-%02d-%02d'%(year,month,int(day))
                    self.vdates.append(sdate)
                    self.hdates[sdate] = i
                    i+=1
        data = matrix[1:,1:]
        (lines,cols) = data.shape
        tvector = data.transpose().reshape(lines*cols).tolist()[0]
        self.levels = []
        for i in range(len(tvector)):
            if i < len(self.vdates):
                v = tvector[i]
                sdate = self.vdates[i]
                vd = sdate.split('-')
                try:
                    tint = int(v)
                except ValueError:
                    sdate = self.vdates[i]
                    if vd[1] == '02' and vd[2] == '29':
                        tint = self.levels[-1]
                    else:
                        tint = None
                self.levels.append(tint)

class RcsvLoader:
    def __init__(self,):
        self.river_name = None
        self.vdates = None 
        self.hdates = None
        self.levels = None
        self.output_dir = None

    def copy(self,obj):
        self.output_dir = obj.output_dir
        self.river_name = obj.river_name
        self.vdates = obj.vdates
        self.hdates = obj.hdates
        self.levels = obj.levels

    def load(self,file_name):
        self.river_name = file_name.split(os.sep)[-1]
        f = csv.reader(open(file_name),delimiter=',')
        i = 0
        self.vdates = []
        self.hdates = dict()
        self.levels = []
        for v in f:
            date = v[0]
            value = v[1]
            self.vdates.append(date)
            self.levels.append(float(value))
            self.hdates[date] = i
            i += 1
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

            


