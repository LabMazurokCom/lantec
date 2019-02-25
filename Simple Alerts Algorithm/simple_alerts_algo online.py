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
    Cars = set()
    alpha = 0.3
    eps = alpha / 5
    flag = False
    j = 0
    sum = 0
    n = 0
    t = datetime.datetime.now()
    while True:
        j += 1
        if j % 1000 == 0:
            t1 = datetime.datetime.now()
            print(j, t1 - t)
            t = t1
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
                    sum += alpha * (1 - CarsMutual[car][x])
                    CarsMutual[car][x] += alpha * (1 - CarsMutual[car][x])
                    updated.append((car, x, CarsMutual[car][x], '↑'))
                    sum += alpha * (1 - CarsMutual[x][car])
                    CarsMutual[x][car] += alpha * (1 - CarsMutual[x][car])
                    updated.append((x, car, CarsMutual[x][car], '↑'))
                elif TimesCarsCameras[camera, x] < time - 2 * deltaT:
                    if car in CarsMutual and x in CarsMutual[car]:
                        sum -= alpha * (CarsMutual[car][x])
                        CarsMutual[car][x] -= alpha * (CarsMutual[car][x])
                        updated.append((x, car, CarsMutual[car][x], '↓'))
                    del TimesCarsCameras[camera, x]
                if car in CarsMutual and x in CarsMutual[car] and CarsMutual[car][x] < eps:
                    sum -= CarsMutual[car][x]
                    n -= 1
                    del CarsMutual[car][x]

        if len(updated) > 0 and n > 0:
            AVG = sum / n
            for x, car, value, increase in updated:
                if value - AVG >= suspicious_level2:
                    print(x, car, value, increase)
    return flag

host = '192.168.1.111'
password = '666666'
gr = Graph(host=host, password=password, bolt=True)
hash = '2786849839b0aa78fcfef8754fdbe877'
time = 1545033444680
deltaT = 120000

print(SimpleAlert(gr, deltaT, 0.9))