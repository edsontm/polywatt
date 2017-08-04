import csv

class MorphometricSpecification:
    def __init__(self,station,elevation,distance,pos):
        self.station = station
        self.e       = elevation
        self.l       = distance
        self.pos     = pos

class EmpiricalFormulas:
    def __init__(self):
        self.morpho = dict()
        self.vmorpho = []

    def load(self,filename='morphometric2.txt'):
        counter = 0
        f = csv.reader(open(filename),delimiter=',')
        header = None
        
        for row in f:
            if header == None:
                header = row
            else:
                station       = row[0].strip(' ')
                elevationdiff = float(row[1])
                distance      = float(row[2])
                self.morpho[station] = MorphometricSpecification(station,elevationdiff,distance,counter)
                self.vmorpho.append(station)
                counter+=1

    def L(self,station1,station2):
        if station1 not in self.morpho or station2 not in self.morpho:
            print "Error, %s or %s not found in memory\n"%(station1,station2)
        p1 = self.morpho[station1].pos 
        p2 = self.morpho[station2].pos
        if p1 > p2:
            aux = p2
            p2 = p1
            p1 = p2
        acc = 0
        for i in range(p1+1,p2+1):
            istation = self.vmorpho[i]
            acc += self.morpho[istation].l
        return acc

    def S(self,station1,station2):
        if station1 not in self.morpho or station2 not in self.morpho:
            print "Error, %s or %s not found in memory\n"%(station1,station2)
        E1 = self.morpho[station1].e 
        E2 = self.morpho[station2].e
        H = abs(E1-E2)
        L = self.L(station1,station2)
        return H/L


        
        
    def lag_time(self,L,S):
        return self.a*(L**self.b)*(S**self.c)/(24)

    def kirpich(self,L,S):
        self.a=0.0663
        self.b=0.77
        self.c=-0.385
        return self.lag_time(L,S)

    def johnstone(self,L,S):
        self.a=0.4623
        self.b=0.5
        self.c=-0.25  
        return self.lag_time(L,S)
         
        
    def chow(self,L,S):
        self.a = 0.1602
        self.b = 0.64
        self.c = -0.32   
        return self.lag_time(L,S)

