from py2neo import Graph
import numpy as np
import pandas as pd

host = '192.168.1.111'
password = '666666'
gr = Graph(host=host, bolt=True, password=password)


def ReadPatterns(s: str):
    # pattern2=['00','01','10','11']
    ptn00 = '000'
    ptn01 = ['001', '0010', '0011']
    ptn10 = ['1010', '100', '110']
    # ptn11=['1011','101','111']
    sptn4 = ('000' + s[-4:])[-4:]
    sptn3 = sptn4[-3:]
    if sptn3 == ptn00:
        return '00'
    if sptn3 == ptn01[0] or sptn4 == ptn01[1] or sptn4 == ptn01[2]:
        return '01'
    if sptn4 == ptn10[0] or sptn3 == ptn10[1] or sptn3 == ptn10[1]:
        return '10'
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
        self.last_id = 307293  # 0

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

# 'AE8726BT'#shadowing!
# TimesVIP: [1.54503353e+12 1.54503356e+12]
# 'AP4830AO'#shadowing when dt=1000
# 'AE4711CE'#not shadowing by avg creteria
# 'AE2828EO'#'AE4027EI'#'AA6117TO'#'00261AI'#'AE1333EM'#'AE6393AA'#'AP4830AO'#'AE5239HP'#'AE5239HP'#'AE1441BK'#'AE2282IX'#'AE7547IM'#'AE5022IE'#'AH6684KB'#no shadowing
vip = 'AE8726BT'  #
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