#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 10:13:42 2019

@author: admin
"""
from py2neo import Graph
import numpy as np
import pandas as pd

host = '192.168.1.111'
password = '666666'
gr = Graph(host=host, bolt=True, password=password)


def ReadDataTime(vip: str, vipTuple: list, time: int, deltaT: int, Times: np.array, Cameras: np.array,
                 # Cars:pd.Series,
                 CarsCur: pd.Series, CamerasCur: np.array, TimesCur: np.array):  # AllCameras:list,
    A = gr.run(
        "match (a:Alert)-[:detectedVehicle]->(v:Vehicle), (c:Camera)-[:detectionCamera]->(a) where a.time >= {leftTime} and a.time <= {rightTime} return v.number, c.hash, a.time",
        leftTime=time - deltaT, rightTime=time).to_ndarray()  # + deltaT
    if len(A) == 0:
        return Times, Cameras, CarsCur, CamerasCur, TimesCur
    CarsA = A[:, 0]
    CamerasA = A[:, 1]
    TimesA = np.array(A[:, 2], dtype=int)
    Cameras1 = CamerasA[CarsA == vip]
    Times1 = TimesA[CarsA == vip]
    CamerasCur = np.concatenate((CamerasCur, CamerasA))
    TimesCur = np.concatenate((TimesCur, TimesA))
    CarsCur = pd.concat([CarsCur, pd.Series(CarsA)])  # np.concatenate((CarsCur,CarsA))
    Cameras = np.concatenate((Cameras, Cameras1))
    Times = np.concatenate((Times, Times1))
    return Times, Cameras, CarsCur, CamerasCur, TimesCur


#    CamerasCur=pd.concat([CamerasCur,pd.Series(CamerasA)])
#    TimesCur=pd.concat([TimesCur,pd.Series(TimesA)])
#    CarsCur=pd.concat([CarsCur,pd.Series(CarsA)])
# Cameras=pd.concat([Cameras,Cameras1])
#    for i,cam in enumerate(Cameras1):
##        if not (CarsA[i] in vipTuple):
##            Cars.loc[cam].append(CarsA[i])
#        Cameras.append(cam)
#        Times.append(int(TimesA[i])) 

def RegressionBigramTimeStream(vip: str, vipTuple: list, startTime: int, deltaT: int, suspicious_level: float):
    i = 0
    dT = 1000  # deltaT
    MaxTInter = deltaT / dT * 10
    period = 604800000  # 100000#?#the time length of period in past
    suspicious_level_2 = suspicious_level / 2
    CamerasVIP = np.array([])
    TimesVIP = np.array([])
    CarsCur = pd.Series([])  # np.array([])
    CamerasCur = np.array([])
    TimesCur = np.array([])
    #    CameraPrev=0
    #    CameraCur=0
    pattern = ['00', '01', '10', '11']
    prediction_init = [0, 0.7, 0.4, 1]
    alpha = 0.6
    table = pd.DataFrame({'prediction': prediction_init, 'real': 0}, index=pattern)
    CarsTable = {}
    CarsHistoryTable = {}
    countCarsPast = 0
    df0 = pd.DataFrame()
    #    cursor = gr.run('MATCH (b:Camera) return b')
    #    AllCameras=[]
    #    for row in cursor:
    #        AllCameras.append(row['b']['hash'])
    #    CarsNew=pd.Series([[]*len(AllCameras)],index=AllCameras)
    while True:
        # read current data about next car detection on any camera at current time
        tk = 0
        while True:
            TimesVIP, CamerasVIP, CarsCur, CamerasCur, TimesCur = ReadDataTime(vip, vipTuple, startTime, dT, TimesVIP,
                                                                               CamerasVIP, CarsCur, CamerasCur,
                                                                               TimesCur)  # CarsNew,
            print('TimesVIP:', TimesVIP)
            if len(TimesVIP) > 0:
                break
            else:
                startTime += dT
                tk += 1
                if tk >= MaxTInter:
                    return False
                # print('startTime=',startTime)
        # print('*')
        #        print('TimesVIP:',TimesVIP)
        #        print('CamerasVIP:',CamerasVIP)
        #        print('CamerasCur:',CamerasCur)
        #        print('CarsCur:',CarsCur)
        #        print('TimesCur:',TimesCur)
        # compare data with vip, save camera id and detection time
        #        if len(TimesVIP)==0:#didn't find any vip appearance at the time period
        #            return False
        for io, cam in enumerate(CamerasVIP):  # new cameras
            CarsNew = np.unique(CarsCur[(CamerasCur == cam) & (~(CarsCur.isin(vipTuple))) &
                                        (TimesCur >= TimesVIP[io] - deltaT) & (TimesCur <= TimesVIP[io] + deltaT)])
            # print('CarsNew:',CarsNew)
            # cam=CamerasVIP[io]
            # create tables for each car detected together with vip
            if i == 0:
                History = pd.Series('1', index=CarsNew)  # [cam]#index - camera id
                HistoryBigram = History.copy()
                # Cars=#.loc[cam]#list
                for x in CarsNew:
                    CarsTable.update({x: table.copy()})
                    # print(x,len(CarsTable))
                    CarsHistoryTable.update({x: np.array([])})  #
                # print(len(CarsTable),len(CarsNew))
                # print('if i==0')
            else:
                # compare new cars with old cars which are already in CarsTable
                for x in CarsNew:  # .loc[cam]:
                    # print('History:',History[x][-1])
                    if x in CarsTable.keys():
                        if History.loc[x][-1] == '1':  # car was on the previous camera AND on this one
                            HistoryBigram.loc[x] += '1'
                            # real value for PREVIOUS state
                            # print(x,ReadPatterns(HistoryBigram[x][:-1]))
                            CarsTable[x].loc[ReadPatterns(HistoryBigram[x][:-1]), 'real'] = 1
                        else:  # car wasn't on the previous camera, only on this one
                            HistoryBigram.loc[x] += '0'
                        History.loc[x] += '1'
                # old cars which do not appear now
                CarsOld = set(CarsTable.keys()).difference(CarsNew)
                # print('CarsOld:',CarsOld)
                for x in CarsOld:
                    History.loc[x] += '0'
                    HistoryBigram.loc[x] += '0'
                    CarsTable[x].loc[ReadPatterns(HistoryBigram[x][:-1]), 'real'] = 0
                # print('else')
            # print('CarsTable:',CarsTable)
            # print('History:',History)
            # CURRENT pattern for each old car which were in CarsTable on the previous step
            LastN = pd.Series([ReadPatterns(x) for x in History.values], index=History.index)
            # print('LastN:',LastN)
            # print(len(LastN),len(CarsTable),len(CarsNew))
            sumPredict = 0
            Suspicious = {}
            # print('len=',len(LastN))
            for x in LastN.index:
                y = LastN[x]
                # change alpha with regard to the car's history on this bigram in past
                dt = (TimesVIP[io] - TimesVIP[io - 1]) if io > 0 else 0
                countCarsPast, CarsHistoryTable[x] = HistoryCarBigram(CarsHistoryTable[x], x, CamerasVIP[io],
                                                                      TimesVIP[io] - period,
                                                                      TimesVIP[io] - deltaT, dt)  #

                # recalculate prediction
                CarsTable[x].loc[y, 'prediction'] += alpha / (np.log(countCarsPast + 1) + 1) * (
                            CarsTable[x].loc[y, 'real'] - CarsTable[x].loc[y, 'prediction'])
                # calculate sum of predictions for all cars
                sumPredict += CarsTable[x].loc[y, 'prediction']
                if CarsTable[x].loc[y, 'prediction'] >= suspicious_level:  # 1st condition for suspicious
                    Suspicious.update({x: y})
            lenCars = len(LastN) - 1
            # print('Suspicious:',Suspicious)
            # print('sumPredictAVG=',sumPredict/(lenCars+1))
            # print('1 for x in LastN.index')

            # find if some car's prediction is significantly bigger than others
            SuspCars = []
            for x in Suspicious.keys():  # LastN.index:#
                y = Suspicious[x]  # LastN[x]#
                if (CarsTable[x].loc[y, 'prediction'] - (sumPredict - CarsTable[x].loc[
                    y, 'prediction']) / lenCars) >= suspicious_level_2:  # 2nd conditions for suspicious
                    SuspCars.append(x)
            #                    print('Final CarsTable:',CarsTable)
            #                    print('Final History:',History)
            #                    print('Final HistoryBigram:',HistoryBigram)
            #                    return True,x
            if len(SuspCars) > 0:
                return True, SuspCars
            # print('2 for x in LastN.index')
            # insert new data
            for x in CarsNew:  # [cam]:
                if not (x in CarsTable.keys()):  # new car
                    CarsTable.update({x: table.copy()})
                    CarsHistoryTable.update({x: df0.copy()})
                    History.loc[x] = '1'  # .loc
                    HistoryBigram.loc[x] = '1'
            # print('*CarsTable:',CarsTable)
            # print('*History:',History)
            # print('*HistoryBigram:',HistoryBigram)

            # remove old data
            mask = TimesCur >= TimesVIP[io] - deltaT
            TimesCur = TimesCur[mask]
            CarsCur = CarsCur[mask]
            CamerasCur = CamerasCur[mask]
        # previous data
        # for io in range(CameraPrev):
        #    cam=CamerasVIP[io]            

        # remove old data
        mask = TimesVIP >= startTime - deltaT
        TimesVIP = TimesVIP[mask]
        CamerasVIP = CamerasVIP[mask]
        #        j=0
        #        while j<len(TimesVIP):
        #            if TimesVIP[j]<startTime-deltaT:
        #                del TimesVIP[j]
        #                del CamerasVIP[j]
        #                j-=1
        #            j+=1
        #        lastTime=TimesVIP[-1]#it is supposed to be the maximal time
        #        TimesVIP=[lastTime]
        #        for i in range(len(CamerasVIP)-1):
        #            CarsNew.iloc[i].clear()
        #        CamerasVIP=[CamerasVIP[-1]]
        ##        while i<len(Times):
        ##            if Times[i]<lastTime:#-deltaT
        ##                del Times[i]
        ##                del Cameras[i]
        ##                CarsNew.iloc[i].clear()
        ##                i-=1
        ##            i+=1
        startTime += dT  # =lastTime+deltaT
        i += 1
    return False


def ReadPatterns(s: str):
    # pattern2=['00','01','10','11']
    ptn00 = '000'
    ptn01 = ['001', '0010', '0011']
    ptn10 = ['1010', '100', '110']
    # ptn11=['1011','101','111']
    # print('s=',s[-4:])
    sptn4 = ('000' + s[-4:])[-4:]
    sptn3 = sptn4[-3:]
    if sptn3 == ptn00:
        return '00'
    if sptn3 == ptn01[0] or sptn4 == ptn01[1] or sptn4 == ptn01[2]:
        return '01'
    if sptn4 == ptn10[0] or sptn3 == ptn10[1] or sptn3 == ptn10[1]:
        return '10'
    # print(sptn4)
    return '11'


def HistoryCarBigram(State1: np.array, car: str, cam2: str, startTime: int, endTime: int, deltaT: float):
    # deltaT - time from cam1 to cam2 in vip case by car
    State2 = gr.run(
        "match (c:Camera{hash: {hash}})-[:detectionCamera]->(a:Alert)-[:detectedVehicle]->(v:Vehicle{number: {number}}) where a.time >= {leftTime} and a.time <= {rightTime}return distinct a.time",
        hash=cam2, number=car, leftTime=startTime, rightTime=endTime).to_ndarray()
    St1, St2 = np.meshgrid(State1, State2)
    #    print('State1:',State1)
    #    print('State2:',State2)
    T = np.where(((St2 - St1) <= 2 * deltaT) | ((St1 - St2) <= 2 * deltaT), 1, 0)
    return np.sum(T) / 2, State2


class GraphReader:
    def __init__(self, graph):
        self.graph = graph
        self.last_id = 21939336  # 0#21939636#21500186#317542#307293#

    def get_next_alert(self):
        query = "match (a:Alert{id: {id}}), (c:Camera)-[:detectionCamera]->(a)-[:detectedVehicle]-(v:Vehicle) return a.time as time, v.number as number, c.hash as camera"
        cursor = self.graph.run(query, id=self.last_id)
        self.last_id += 1
        # print(self.last_id)
        if cursor.forward():
            return {'time': cursor.current['time'], 'number': cursor.current['number'],
                    'camera': cursor.current['camera']}
        else:
            return None


def RegressionBigramOneAlert(gr: Graph, vip: str, vipTuple: list, startTime: int, deltaT: int, suspicious_level: float):
    i = 0
    k = 0
    period = 604800000  # the time length of period in past (1 week)
    suspicious_level_2 = suspicious_level / 2
    CamerasVIP = []  # np.array([])
    TimesVIP = []  # np.array([])
    CarsCur = []  # pd.Series([])#np.array([])
    CamerasCur = []  # np.array([])
    TimesCur = []  # np.array([])
    pattern = ['00', '01', '10', '11']
    prediction_init = [0, 0.7, 0.4, 1]
    alpha = 0.6
    table = pd.DataFrame({'prediction': prediction_init, 'real': 0}, index=pattern)
    CarsTable = {}
    CarsHistoryTable = {}
    countCarsPast = 0
    # df0=pd.DataFrame()
    # CarsNew=[]
    GR = GraphReader(gr)
    flag = False
    FinalSuspicious = []
    for q in range(20000):  # while True: #
        # read 1 alert
        alert = GR.get_next_alert()
        time = alert['time']
        car = alert['number']
        camera = alert['camera']
        # print(time,car,camera)
        if car == vip:
            CamerasVIP.append(camera)
            TimesVIP.append(time)
            print('CamerasVIP:', CamerasVIP)
        elif not (car in vipTuple):
            CamerasCur.append(camera)
            TimesCur.append(time)
            CarsCur.append(car)
        curTime = time
        # print('curTime=',curTime)
        # remove old cars
        #        j=0
        #        while j<len(CarsCur):
        #            if TimesCur[j]<time-2*deltaT:
        #                del CarsCur[j]
        #                del CamerasCur[j]
        #                del TimesCur[j]
        #                j-=1
        #            j+=1
        if len(TimesVIP) > 0:  # if VIP passed some camera in recent 2 minutes
            k = 0
            # print('TimesVIP:',TimesVIP)
            # add cars which were at VIP cameras in CarsNew array
            for io, cam in enumerate(CamerasVIP):
                CarsNew = []
                j = 0
                while j < len(CarsCur):
                    if CamerasCur[j] == cam and TimesCur[j] >= TimesVIP[io] - deltaT and TimesCur[j] <= TimesVIP[
                        io] + deltaT:
                        CarsNew.append(CarsCur[j])
                        # remove cars which are put in CarsNew array
                        del CarsCur[j]
                        del CamerasCur[j]
                        del TimesCur[j]
                        j -= 1
                    j += 1
                CarsNew = list(set(CarsNew))  # to remove repeated items!
                # print('CarsNew:',CarsNew)
                # print('CarsCur:',CarsCur)
                if len(CarsNew) > 0:  # if there are some cars which were at VIP cameras
                    # print('CarsNew:',CarsNew)
                    # print('\nCarsNew:')
                    if i == 0:  # initialization of History, HistoryBigram, CarsTable
                        History = pd.Series('1', index=CarsNew)  # [cam]#index - camera id
                        HistoryBigram = History.copy()
                        for x in CarsNew:
                            if not (x in CarsTable.keys()):
                                CarsTable.update({x: table.copy()})
                                CarsHistoryTable.update({x: np.array([])})
                        i = 1
                    else:
                        # compare new cars with old cars which are already in CarsTable
                        for x in CarsNew:
                            if x in CarsTable.keys():
                                # if (x in History.keys()):
                                if History.loc[x][-1] == '1':  # car was on the previous camera AND on this one
                                    HistoryBigram.loc[x] += '1'
                                    # real value for the PREVIOUS state
                                    # print('x=',x,'hist:',HistoryBigram[x])
                                    # CarsTable[x].loc[ReadPatterns(History[x][:-1]),'real']=1
                                    CarsTable[x].loc[ReadPatterns(HistoryBigram[x][:-1]), 'real'] = 1
                                else:  # car wasn't on the previous camera, only on this one
                                    HistoryBigram.loc[x] += '0'
                                History.loc[x] += '1'
                            else:  # elif x not in History.index:#
                                History.loc[x] = '1'
                                HistoryBigram.loc[x] = '1'
                                CarsTable.update({x: table.copy()})
                                CarsHistoryTable.update({x: np.array([])})
                                # print('CarsTable:',CarsTable.keys())
                        # print('CarsNew:',CarsNew)
                        # print('CarsOld:',CarsOld)
                        # print('History:',History)
                    # print('HistoryBigram:',HistoryBigram)#print(CarsNew,History)

                    # patterns for new cars by bigrams
                    patterns = [''] * len(CarsNew)
                    for j in range(len(CarsNew)):
                        x = HistoryBigram.loc[CarsNew[j]][:-1]  # x=History.loc[CarsNew[j]]#
                        patterns[j] = ReadPatterns(x)
                    LastN = pd.Series(patterns, index=CarsNew)
                    # LastN=pd.Series([ReadPatterns(x) for x in HistoryBigram.values],index=CarsNew)#History
                    sumPredict = 0
                    Suspicious = {}
                    for x in LastN.index:
                        y = LastN[x]
                        # CarsTable[x].loc[y,'real']=1
                        # change alpha with regard to the car's history on this bigram in past
                        dt = (TimesVIP[io] - TimesVIP[io - 1]) if io > 0 else 0
                        countCarsPast, CarsHistoryTable[x] = HistoryCarBigram(CarsHistoryTable[x], x, CamerasVIP[io],
                                                                              TimesVIP[io] - period,
                                                                              TimesVIP[io] - deltaT, dt)  #
                        # countCarsPast=0

                        # recalculate prediction
                        # print(CarsTable[x].loc[y])
                        # y2=ReadPatterns(HistoryBigram[x])
                        # CarsTable[x].loc[y2,'prediction']=(CarsTable[x].loc[y,'prediction']+
                        #        alpha/(np.log(countCarsPast+1)+1)*(CarsTable[x].loc[y,'real']-CarsTable[x].loc[y,'prediction']))
                        CarsTable[x].loc[y, 'prediction'] += alpha / (np.log(countCarsPast + 1) + 1) * (
                                    CarsTable[x].loc[y, 'real'] - CarsTable[x].loc[y, 'prediction'])
                    #                        if x=='AE3308HO' or x=='AE7445BK' or x=='AE8990CP':
                #                            print(cam)
                #                            print('CarsNew:',x,CarsTable[x].loc[y,'prediction'])
                # if HistoryBigram.loc[x]=='111':
                # print(x,HistoryBigram.loc[x],CarsTable[x].loc[y,'prediction'])
                # calculate sum of predictions for all cars
                #                        sumPredict+=CarsTable[x].loc[y,'prediction']
                #                        if CarsTable[x].loc[y,'prediction']>=suspicious_level:#1st condition for suspicious
                #                            Suspicious.update({x:y})
                #                    lenCars=len(LastN)-1
                #                print('Suspicious:',Suspicious)
                #                print('sumPredictAVG=',sumPredict/(lenCars+1))

                # find if some car's prediction is significantly bigger than others
                #                    SuspCars=[]
                #                    for x in Suspicious.keys():#LastN.index:#
                #                        y=Suspicious[x]#LastN[x]#
                #                        if (CarsTable[x].loc[y,'prediction']-(sumPredict-CarsTable[x].loc[y,'prediction'])/lenCars)>=suspicious_level_2:#2nd conditions for suspicious
                #                            SuspCars.append(x)
                #        #                    print('Final CarsTable:',CarsTable)
                #        #                    print('Final History:',History)
                #        #                    print('Final HistoryBigram:',HistoryBigram)
                #        #                    return True,x
                #                    if len(SuspCars)>0:
                #                        return True,SuspCars
                # insert new data
                #                for x in CarsNew:
                #                    if not (x in CarsTable.keys()): #new car
                #                        CarsTable.update({x:table.copy()})
                #                        CarsHistoryTable.update({x:df0.copy()})
                #                        History.loc[x]='1'
                #                        HistoryBigram.loc[x]='1'
                # print('*CarsTable:',CarsTable)
                # print('*History:',History)
                # print('*HistoryBigram:',HistoryBigram)

                if len(CarsTable) > 0 and len(CarsCur) > 0:
                    # old cars which do not appear for TimesVIP[io]+deltaT
                    # print('\nCarsCur:')
                    j = 0
                    while j < len(TimesCur):
                        if CamerasCur[j] == cam and (TimesCur[j] < TimesVIP[io] - deltaT) and (
                                CarsCur[j] in History.index):
                            x = CarsCur[j]
                            History.loc[x] += '0'
                            HistoryBigram.loc[x] += '0'
                            y = ReadPatterns(HistoryBigram[x][:-1])
                            # print('x=',x,'hist:',HistoryBigram[x])
                            CarsTable[x].loc[y, 'real'] = 0
                            # CarsTable[x].loc[ReadPatterns(HistoryBigram[x][:-1]),'real']=0

                            # change alpha with regard to the car's history on this bigram in past
                            dt = (TimesVIP[io] - TimesVIP[io - 1]) if io > 0 else 0
                            countCarsPast, CarsHistoryTable[x] = HistoryCarBigram(CarsHistoryTable[x], x,
                                                                                  CamerasVIP[io], TimesVIP[io] - period,
                                                                                  TimesVIP[io] - deltaT, dt)  #
                            # countCarsPast=0
                            # recalculate prediction for old cars
                            # print(CarsTable[x].loc[y])
                            # y2=ReadPatterns(HistoryBigram[x])
                            # CarsTable[x].loc[y2,'prediction']=(CarsTable[x].loc[y,'prediction']+
                            #         alpha/(np.log(countCarsPast+1)+1)*(CarsTable[x].loc[y,'real']-CarsTable[x].loc[y,'prediction']))  
                            CarsTable[x].loc[y, 'prediction'] += alpha / (np.log(countCarsPast + 1) + 1) * (
                                        CarsTable[x].loc[y, 'real'] - CarsTable[x].loc[y, 'prediction'])
                            #                            if x=='AE3308HO' or x=='AE7445BK' or x=='AE8990CP':
                            #                                print(cam)
                            #                                print('CarsCur:',x,CarsTable[x].loc[y,'prediction'])
                            # if HistoryBigram.loc[x]=='111':
                            # print(x,HistoryBigram.loc[x],CarsTable[x].loc[y,'prediction'])
                            # remove cars for which we recalculated predictions
                            del TimesCur[j]
                            del CarsCur[j]
                            del CamerasCur[j]
                        else:
                            j += 1

                # find suspicious cars
                sumPredict = 0
                Suspicious = {}
                SuspPred = []
                for x in CarsTable.keys():
                    y = ReadPatterns(HistoryBigram[x][:-1])  #
                    # calculate sum of predictions for all cars
                    sumPredict += CarsTable[x].loc[y, 'prediction']
                    if CarsTable[x].loc[y, 'prediction'] >= suspicious_level:  # 1st condition for suspicious
                        Suspicious.update({x: y})
                        SuspPred.append(CarsTable[x].loc[y, 'prediction'])
                #                print('Suspicious:',Suspicious)
                #                print('sumPredictAVG=',sumPredict/(lenCars+1))
                #                if len(Suspicious)>0:
                #                    print('Suspicious:',Suspicious)
                #                    print('Suspicious Predictions:',SuspPred)
                # print('sumPredict=',sumPredict/(lenCars+1))

                # find if some car's prediction is significantly bigger than others
                lenCars = len(CarsTable) - 1
                SuspCars = []
                SuspPred = []
                if lenCars > 0:
                    for x in Suspicious.keys():  # LastN.index:#
                        y = Suspicious[x]  # LastN[x]#
                        # if (CarsTable[x].loc[y,'prediction']-(sumPredict)/(lenCars))>=suspicious_level_2:
                        if (CarsTable[x].loc[y, 'prediction'] - (sumPredict - CarsTable[x].loc[
                            y, 'prediction']) / lenCars) >= suspicious_level_2:  # 2nd conditions for suspicious
                            SuspCars.append(x)
                            SuspPred.append(CarsTable[x].loc[y, 'prediction'])
                    #                    print('Final CarsTable:',CarsTable)
                    #                    print('Final History:',History)
                    #                    print('Final HistoryBigram:',HistoryBigram)
                    #                    return True,x
                    if 0 < len(SuspCars) <= 5 and (SuspCars not in FinalSuspicious):  # len(SuspCars)>0:#
                        print(True, SuspCars, SuspPred, q)
                        FinalSuspicious.append(SuspCars)
                        for x in SuspCars:
                            print('History:', History.loc[x])
                            print('History Bigram:', HistoryBigram.loc[x])
                        # print('VIP cameras:',CamerasVIP)
                        flag = True
                        # return True,SuspCars

                    # print('**CarsTable:',CarsTable)
                    # print('*History:',History)
                    # print('**HistoryBigram:',HistoryBigram)
            #            CarsOld=set(CarsTable.keys()).difference(CarsNew)
            #            for x in CarsOld:
            #                History.loc[x]+='0'
            #                HistoryBigram.loc[x]+='0'
            #                CarsTable[x].loc[ReadPatterns(HistoryBigram[x][:-1]),'real']=0
            # remove old data about other cars
            #            j=0
            #            while j<len(TimesCur):
            #                if TimesCur[j]<curTime-deltaT:#TimesVIP[io]-deltaT:
            #                    del TimesCur[j]
            #                    del CamerasCur[j]
            #                    del CarsCur[j]
            #                    j-=1
            #                j+=1

            # remove old VIP data
            j = 0
            while j < len(TimesVIP):
                if TimesVIP[j] < curTime - deltaT:
                    del TimesVIP[j]
                    del CamerasVIP[j]
                    j -= 1
                j += 1
        else:
            k += 1
        #            if k>1000:
    #                return False

    return flag  # False


'''
#Получить все бренды в базе:
cursor = gr.run('MATCH (b:Brand) return b')
for row in cursor:
    print(row['b']['name'])

#Получить все модели в базе, представителем которых является ТС с номером 'BM2472AX':
cursor = gr.run("MATCH (:Vehicle{number: 'BM2472AX'})-[:isOfModel]->(r:Model) return r.name as name")
for row in cursor:
    print(row['name'])
'''
###
# for time intervals
# 'AE8726BT'#shadowing!
# TimesVIP: [1.54503353e+12 1.54503356e+12]
# 'AP4830AO'#shadowing when dt=1000
# 'AE4711CE'#not shadowing by avg creteria
# 'AE2828EO'#'AE4027EI'#'AA6117TO'#'00261AI'#'AE1333EM'#'AE6393AA'#'AP4830AO'#'AE5239HP'#'AE5239HP'#'AE1441BK'#'AE2282IX'#'AE7547IM'#'AE5022IE'#'AH6684KB'#no shadowing

# shadowing from SimpleAlert.py
# 'AE2978CE', 'AE2577IM'
# 'AE2978CE', 'AE8581EK', 'AE2577IM'
# 'AE1645IX', 'AE2978CE', 'AE8581EK', 'AE2577IM'
# 'AE2577IM', 'AX0674EK', 'AX8044EO', 'AE8581EK', 'AE2978CE'

# {'AX0674EK', 'AE2577IM'}
# ['AE2577IM', 'AX0674EK', 'AE2978CE', 'AX8044EO', 'AE8581EK', 'AE9868AH']


# current shadowing
# '0717' <- 'AE3308HO'

# ['AES9301', 'AE5930', 'KM0379', 'AE1252EX', 'AE9094EI', 'AE5933']
# 'AES9301' <- ['AE5930'] - 1
# 'AE5930' <- False
# 'KM0379' <- ['AE5930'] - 1
# 'AE1252EX' <- ['AE5930'] - 1
# 'AE9094EI' <- False
# 'AE5933' <- False

# ['AE6147EM', 'AE0989IP', 'AE7471CB', 'AE0710BA', 'AE4434BI', 'AE0990EP']
# 'AE6147EM' <- False - 0
# 'AE0989IP <- ['AE0990EP', 'AE9080HA', 'AX9702AA', 'AE5923IP', 'AE7188CB'] - 1
# 'AE7471CB' <- ['AE4434BI', 'AE1112BE', 'AE6688EE', 'AE7175HE', 'AE4017'] - 1
# 'AE0710BA' <- False
# 'AE4434BI' <- ['AE7471CB', 'AE6688EE', 'AE7175HE', 'AE4017'] - 1
# 'AE0990EP' <- ['AE0989IP', 'AE9080HA', 'AE7188CB'] - 1

vip = '0717'  #
# startTime=1545033444680
deltaT = 120000
# To obtain all car ids
# cars = gr.run('MATCH (a:Alert)-[:detectedVehicle]->(v:Vehicle) where a.time >= {leftTime} and a.time <= {rightTime} return v.number',
#                leftTime=startTime - deltaT, rightTime=startTime).to_ndarray()
# print(cars)

host = '192.168.1.111'
password = '666666'
gr = Graph(host=host, password=password, bolt=True)
hash = '2786849839b0aa78fcfef8754fdbe877'
time = 1545033444680
delta = 120000
# cursor = gr.run("match (c:Camera{hash: {hash}})-[:detectionCamera]->(a:Alert), (a)-[:detectedVehicle]->(v:Vehicle) using index a:Alert(time) where a.time >= {left_time} and a.time <= {right_time} return distinct v.number"
# , hash=hash, leftTime=time - delta, rightTime=time + delta)
# for row in cursor:
#   print(row["number"])

print("\nResult=", RegressionBigramOneAlert(gr, vip, [vip], time, deltaT, 0.9), '\n')
# print("\nResult=",RegressionBigramTimeStream(vip,[vip],time,deltaT,0.9),'\n')

# A=gr.run("match (a:Alert)-[:detectedVehicle]->(v:Vehicle), (c:Camera)-[:detectionCamera]->(a) where a.time >= {leftTime} and a.time <= {rightTime} return v.number, c.hash",
#           leftTime=startTime - deltaT, rightTime=startTime + deltaT).to_ndarray()
# CarsA=A[:,0]
# CamerasA=pd.Series(A[:,1])
# Cameras1=CamerasA[CarsA==vip]
##Cameras1=np.array(['2786849839b0aa78fcfef8754fdbe877'])
# Cars1=CarsA[CamerasA.isin(Cameras1)]
# print("Cameras:",Cameras1)
# print('Cars:',Cars1)

# To obtain all car ids
# cursor = gr.run('MATCH (b:Vehicle) return b')
# for row in cursor:
#    print(row['b']['number'])

# curTime=0
# cursor = gr.run("MATCH (a:Alert{time: {time}})-[:detectedVehicle]->(r:Vehicle) return r.number as number, a.alert_id as alert_id",
#                time=curTime)


# To obtain all camera ids
'''
cursor = gr.run('MATCH (b:Camera) return b')
AllCameras=[]
for row in cursor:
    AllCameras.append(row['b']['hash'])
    #print(row['b']['camera_id'])
print('All cameras:',AllCameras)

startTime=1545033444680
deltaT=120000
Cars=[]
Cameras=[]
Times=[]
vipTuple=[]'''
# hash='2786849839b0aa78fcfef8754fdbe877'
# cursor = gr.run("match (c:Camera{hash: {hash}})-[:detectionCamera]->(a:Alert)-[:detectedVehicle]->(v:Vehicle) where a.time >= {leftTime} and a.time <= {rightTime}return distinct v.number as number",
#                hash=hash, leftTime=startTime - deltaT, rightTime=startTime + deltaT)
# for row in cursor:
#   print(row["number"])

# cursor = gr.run("match (c:Camera{})-[:detectionCamera]->(a:Alert)-[:detectedVehicle]->(v:Vehicle) where a.time >= {leftTime} and a.time <= {rightTime}return distinct v.number as number, c.hash as hash",
#                leftTime=startTime - deltaT, rightTime=startTime + deltaT)
# for row in cursor:
#   print(row["number"],row['hash'])

'''
for hash in AllCameras:
    cursor = gr.run("match (c:Camera{hash: {hash}})-[:detectionCamera]->(a:Alert)-[:detectedVehicle]->(v:Vehicle) where a.time >= {leftTime} and a.time <= {rightTime}return distinct v.number as number", 
                hash=hash, leftTime=startTime - deltaT, rightTime=startTime + deltaT)
    for row in cursor:
        car=row['number']
        if not (car in vipTuple):
            if car==vip:
                Times.append(startTime)#not exact time, the middle of the time segment
                Cameras.append(hash)
            else:
                Cars.append(car)'''

'''
#for cam in Cameras:
#cursor = gr.run("MATCH (:Vehicle{number: {number}})-[:isOfModel]->(r:Model) return r.name as name", number='BM2472AX')
cursor = gr.run("MATCH (:Camera{hash: {hash}})-[d:detected]->(r:Vehicle) return r.number as number, d.time as time", 
                hash=Cameras[0])
Cars=[]
Times=[]
i=0
for row in cursor:
    car,time=row['number'], row['time']    
    if car==vip:
        Times.append(time)
        i+=1
    elif (Times[i]-deltaT)<=time<=(Times[i]+deltaT):
        Cars.append(car)#suspicious car
    #print(row['number','time'])'''