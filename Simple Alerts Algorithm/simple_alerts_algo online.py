from py2neo import Graph
import datetime
import csv
class GraphReader:
    def __init__(self, graph):
        self.graph = graph
        self.last_id = 21500186  # 0# 307293#

    def get_next_alert(self):
        query = "match (a:Alert{id: {id}}), (c:Camera)-[:detectionCamera]->(a)-[:detectedVehicle]-(v:Vehicle) return a.time as time, v.number as number, c.hash as camera"
        cursor = self.graph.run(query, id=self.last_id)
        self.last_id += 1
        if cursor.forward():
            return cursor.current['time'], cursor.current['number'], cursor.current[
                'camera']
        else:
            return None

def SimpleAlert(gr: Graph, deltaT: int, suspicious_level: float):
    suspicious_level2 = suspicious_level / 2
    GR = GraphReader(gr)
    TimesCarsCameras = {}
    CarsMutual = {}
    CarsCounter = {}
    Cars = set()
    alpha = 0.3
    eps = alpha / 5
    flag = False
    FinalSuspicious = []
    j = 0
    avg = 0
    n = 0
    while True:
        if j % 10000 == 0:
            print(j)
        j += 1
        # read new alert
        alert = GR.get_next_alert()
        time, car, camera = alert

        TimesCarsCameras.update(
            {(camera, car): time})
        Cars.add(car)
        updated = []
        for x in Cars:
            if x != car and (camera, x) in TimesCarsCameras.keys():
                if TimesCarsCameras[camera, x] >= time - deltaT:
                    if car not in CarsMutual:
                        CarsMutual[car] = {}
                    if x not in CarsMutual:
                        CarsMutual[x] = {}
                    if car not in CarsMutual[x]:
                        CarsMutual[x][car] = 0
                        n += 1
                    if x not in CarsMutual[car]:
                        CarsMutual[car][x] = 0
                        n += 1
                    CarsMutual[car][x] += alpha * (1 - CarsMutual[car][x])
                    CarsMutual[x][car] += alpha * (1 - CarsMutual[x][car])
                elif TimesCarsCameras[camera, x] < time - 2 * deltaT:
                    if car in CarsMutual and x in CarsMutual[car]:
                        CarsMutual[car][x] -= alpha * (CarsMutual[car][x])
                    del TimesCarsCameras[camera, x]
                if car in CarsMutual and x in CarsMutual[car] and CarsMutual[car][x] < eps:
                    del CarsMutual[car][x]

        # calculation of mu_i for current car
        lenCars = len(CarsMutual)
        if lenCars > 1:
            AVG = 0
            for value in CarsMutual.values():
                AVG += value
            AVG /= (lenCars - 1)
            Susp = set()
            """Key1 = []
            Key2 = []
            Weights = []
            Count = []"""
            for key1, key2 in CarsMutual.keys():
                if CarsMutual[key1, key2] - AVG >= suspicious_level2:
                    """Key1.append(key1)
                    Key2.append(key2)
                    Weights.append(CarsMutual[key1, key2])
                    Count.append(CarsCounter[key1, key2])"""
                    #print(datetime.datetime.fromtimestamp(time / 1000), key1, key2, CarsMutual[key1, key2],
                              #CarsCounter[key1, key2])
    return flag

host = 'localhost'
password = '666666'
gr = Graph(host=host, password=password, bolt=True)
hash = '2786849839b0aa78fcfef8754fdbe877'
time = 1545033444680
deltaT = 120000

print(SimpleAlert(gr, deltaT, 0.9))