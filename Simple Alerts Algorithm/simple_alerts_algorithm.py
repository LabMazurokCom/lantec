from py2neo import Graph
import numpy as np
import pandas as pd
import queue

host = '192.168.1.111'
password = '666666'
gr = Graph(host=host, bolt=True, password=password)


class GraphReader:
    def __init__(self, graph):
        self.graph = graph
        self.last_id = 0  # 307293#

    def get_next_alert(self):
        query = "match (a:Alert{id: {id}}), (c:Camera)-[:detectionCamera]->(a)-[:detectedVehicle]-(v:Vehicle) return a.time as time, v.number as number, c.hash as camera"
        cursor = self.graph.run(query, id=self.last_id)
        self.last_id += 1
        # print(self.last_id)
        if cursor.forward():
            return cursor.current['time'], cursor.current['number'], cursor.current[
                'camera']  # {'time': cursor.current['time'], 'number': cursor.current['number'], 'camera': cursor.current['camera']}
        else:
            return None


def SimpleAlert(gr: Graph, deltaT: int, suspicious_level: float):
    suspicious_level2 = suspicious_level / 2
    GR = GraphReader(gr)
    # AllCameras=gr.run('MATCH (b:Camera) return b').to_ndarray()
    # print('All cameras:',AllCameras)
    # lenCams=len(AllCameras)
    # AllCars=gr.run('MATCH (b:Vehicle) return b').to_ndarray()#doesn't work
    # print('All cars:',AllCars)
    # lenCars=len(AllCars)
    # time,car,camera=GR.get_next_alert()
    # TimesCarCameras=pd.DataFrame(np.zeros((1,lenCams)),index=[car],columns=AllCameras)
    # TimesCameras={}#{x:[] for x in AllCameras}#queue.Queue()
    # CarsCameras={}#{x:[] for x in AllCameras}
    TimesCarsCameras = {}
    CarsMutual = {}
    CarsCounter = {}
    # CarsCameras={}#for bigrams ?
    # CarsTimes={}#for bigrams ?
    # pd.SparseDataFrame([0],index=[car],columns=[car],default_fill_value=0.0)
    Cars = []
    alpha = 0.3
    eps = alpha / 5
    flag = False
    FinalSuspicious = []
    for i in range(10000):  # while True:#
        # read new alert
        time, car, camera = GR.get_next_alert()

        # print(car,camera)
        TimesCarsCameras.update(
            {(camera, car): time})  # TimesCameras[camera].append(time)#CarsCameras[camera].append(car)
        # TimesCarCameras.loc[car,camera]=time
        Cars.append(car)
        """#for bigrams
        if car in CarsCameras.keys():
            CarsCameras[car].append(camera)
            CarsTimes[car].append(time)
            #print(CarsCameras[car])
        else:
            CarsCameras.update({car:[camera]})
            CarsTimes.update({car:[time]})
        for x in Cars:
            if x!=car and (camera,x) in TimesCarsCameras.keys():
#                if len(CarsCameras[car])>2:
#                    print(CarsCameras[car][-2])
#                    #print(TimesCarsCameras[CarsCameras[car][:-2],car])
                if (TimesCarsCameras[camera,x]>=time-deltaT and len(CarsCameras[car])>1 and len(CarsCameras[x])>1 
                    and CarsCameras[car][-2]==CarsCameras[x][-2]
                    and abs(CarsTimes[car][-2]-CarsTimes[x][-2])<=2*deltaT):
                    if (car,x) in CarsMutual.keys():
                        CarsMutual[car,x]+=alpha*(1-CarsMutual[car,x])
                        CarsCounter[car,x]+=1
                        #symmetric?
                        if (x,car) in CarsMutual.keys():
                            CarsMutual[x,car]+=alpha*(1-CarsMutual[x,car])
                            CarsCounter[x,car]+=1
                        else:
                            CarsMutual.update({(x,car):alpha})
                            CarsCounter.update({(x,car):1})
                        #print('+:',CarsMutual[car,x])
                    else:
                        CarsMutual.update({(car,x):alpha})
                        CarsCounter.update({(car,x):1})
                        #symmetric?
                        if (x,car) in CarsMutual.keys():
                            CarsMutual[x,car]+=alpha*(1-CarsMutual[x,car])
                            CarsCounter[x,car]+=1
                        else:
                            CarsMutual.update({(x,car):alpha})
                            CarsCounter.update({(x,car):1})
                elif TimesCarsCameras[camera,x]<time-2*deltaT:
                    if (car,x) in CarsMutual.keys():
                        CarsMutual[car,x]-=alpha*(CarsMutual[car,x])
                        CarsCounter[car,x]-=1
                        #print('-:',CarsMutual[car,x])
                    if (x in CarsCameras) and len(CarsCameras[x])>1 and TimesCarsCameras[camera,x]<time-2*deltaT:
                        del TimesCarsCameras[camera,x]
                        #j=0
                        #while j<len(CarsCameras[x])-1:                            
                        del CarsCameras[x]#[j]
                if ((car,x) in CarsMutual.keys()) and CarsMutual[car,x]<eps:
                    del CarsMutual[car,x]
                    del CarsCounter[car,x]"""
        for x in Cars:
            if x != car and (camera, x) in TimesCarsCameras.keys():
                if TimesCarsCameras[camera, x] >= time - deltaT:
                    if (car, x) in CarsMutual.keys():
                        CarsMutual[car, x] += alpha * (1 - CarsMutual[car, x])
                        CarsCounter[car, x] += 1
                        # symmetric?
                        if (x, car) in CarsMutual.keys():
                            CarsMutual[x, car] += alpha * (1 - CarsMutual[x, car])
                            CarsCounter[x, car] += 1
                        else:
                            CarsMutual.update({(x, car): alpha})
                            CarsCounter.update({(x, car): 1})
                        # print('+:',CarsMutual[car,x])
                    else:
                        CarsMutual.update({(car, x): alpha})
                        CarsCounter.update({(car, x): 1})
                        # symmetric?
                        if (x, car) in CarsMutual.keys():
                            CarsMutual[x, car] += alpha * (1 - CarsMutual[x, car])
                            CarsCounter[x, car] += 1
                        else:
                            CarsMutual.update({(x, car): alpha})
                            CarsCounter.update({(x, car): 1})
                elif TimesCarsCameras[camera, x] < time - 2 * deltaT:
                    if (car, x) in CarsMutual.keys():
                        CarsMutual[car, x] -= alpha * (CarsMutual[car, x])
                        CarsCounter[car, x] -= 1
                        # print('-:',CarsMutual[car,x])
                    del TimesCarsCameras[camera, x]
                if ((car, x) in CarsMutual.keys()) and CarsMutual[car, x] < eps:
                    del CarsMutual[car, x]
                    del CarsCounter[car, x]
        # print('CarsMutual:',CarsMutual,'\n')

        # calculation of mu_i for current car
        lenCars = len(CarsMutual)
        if lenCars > 1:
            AVG = 0
            for value in CarsMutual.values():
                AVG += value
            AVG /= (lenCars - 1)
            # print('AVG=',AVG)
            Susp = set()
            Key1 = []
            Key2 = []
            Weights = []
            Count = []
            for key1, key2 in CarsMutual.keys():
                if CarsMutual[key1, key2] - AVG >= suspicious_level2:
                    Key1.append(key1)
                    Key2.append(key2)
                    Weights.append(CarsMutual[key1, key2])
                    Count.append(CarsCounter[key1, key2])
                    print(datetime.datetime.fromtimestamp(time / 1000), key1, key2, CarsMutual[key1, key2],
                          CarsCounter[key1, key2])
            #                    Susp.add(key1)
            #                    Susp.add(key2)
            '''KA=Key1+Key2#KA=np.array(Key1+Key2)#
#            KA1=np.array(Key1)
#            KA2=np.array(Key2)
#            WA=np.array(Weights)
#            Connect=np.zeros(KA1.shape)
#            Ind=WA.argsort()[::-1]
#            KA1=KA1[Ind]
#            KA2=KA2[Ind]
#            WA=WA[Ind]


#            Ind=KA1.argsort()
#            KA1=KA1[Ind]
#            KA2=KA2[Ind]

            #division by subgroups
            SuspK=list(set(KA))#SuspK=np.unique(KA)
            sno=0
            while sno<len(SuspK):#for sno in range(len(SuspK)):
                #print(Susp,sno)
                Susp=[SuspK[sno]]#set
                del SuspK[sno]
                sno-=1
                SuspW=[]
                SuspC=[]
                #curSusp=SuspK[sno]
                j=0
                while j<len(Susp):
                    k=0
                    while k<len(Key1):
                        x=Key1[k]
                        y=Key2[k]#for x,y in zip(Key1,Key2):
                        if x==Susp[j]:
                            if y not in Susp:
                                Susp.append(y)
                                SuspW.append(Weights[k])
                                SuspC.append(Count[k])
                            Key1.remove(x)
                            Key2.remove(y)
                            del Weights[k]
                            del Count[k]
                            k-=1
                            if y in SuspK:
                                SuspK.remove(y)
                        elif y==Susp[j]:
                            if x not in Susp:
                                Susp.append(x)
                                SuspW.append(Weights[k])
                                SuspC.append(Count[k])
                            Key1.remove(x)
                            Key2.remove(y)
                            del Weights[k]
                            del Count[k]
                            k-=1
                            if x in SuspK:
                                SuspK.remove(x)
                        k+=1
                    j+=1
                sno+=1
#                while j<len(KA1) and KA1[j]==Susp1[sno]:
#                    Susp.add(KA2[j])
#                    j+=1
                #if 1<len(Susp)<=6 and (Susp not in FinalSuspicious):#only new shadowings
                if 1<len(Susp)<=6:# and (Susp not in FinalSuspicious):
                    print(datetime.datetime.fromtimestamp(time/1000),Susp,SuspW,SuspC,i)
                    FinalSuspicious.append(Susp)
                    flag=True'''
    #        if i % 100 == 0:
    #            print(i)
    # print('\n')

    #            #if len(Susp)>0:
    #            #    print('Susp:',Susp)
    #            #Susp=np.unique(np.array(Susp).flatten())
    #            Mu=len(Susp)
    #            if 0<Mu<=5:
    #                print(True,Susp)
    #                flag=True
    #                #return True,Susp

    # print('CarsMutual:',CarsMutual,'\n')
    #    print('Max=',max(CarsMutual.values()))
    #    print('AVG=',AVG)
    #    print('Susp:',Susp)
    return flag  # False


host = '192.168.1.111'
password = '666666'
gr = Graph(host=host, password=password, bolt=True)
hash = '2786849839b0aa78fcfef8754fdbe877'
time = 1545033444680
deltaT = 120000

print(SimpleAlert(gr, deltaT, 0.9))