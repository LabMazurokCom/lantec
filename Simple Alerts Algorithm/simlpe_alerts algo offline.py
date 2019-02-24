from py2neo import Graph
import csv

class GraphReader:
    def __init__(self, graph):
        self.graph = graph
        self.last_id = 21500186  # 0# 307293#

    def get_next_alert(self):
        query = "match (a:Alert{id: {id}}), (c:Camera)-[:detectionCamera]->(a)-[:detectedVehicle]-(v:Vehicle) return a.time as time, v.number as number, c.hash as camera"
        cursor = self.graph.run(query, id=self.last_id)
        self.last_id += 1
        # print(self.last_id)
        if cursor.forward():
            return cursor.current['time'], cursor.current['number'], cursor.current[
                'camera']
        else:
            return None


def SimpleAlert(gr: Graph, deltaT: int, suspicious_level: float):
    suspicious_level2 = suspicious_level / 2
    GR = GraphReader(gr)
    CarsMutual = {}
    CarsCounter = {}
    cameras_queue = {}
    Cars = set()
    alpha = 0.3
    while True:#
        # read new alert
        alert = GR.get_next_alert()
        if alert is None:
            break
        time, car, camera = alert

        # print(car,camera)
        if camera not in cameras_queue:
            cameras_queue[camera] = {}
        cameras_queue[camera][car] = time  # TimesCameras[camera].append(time)#CarsCameras[camera].append(car)
        # TimesCarCameras.loc[car,camera]=time
        Cars.add(car)
        to_del = []
        for x, t in cameras_queue[camera].items():
            if x != car:
                if t >= time - deltaT:
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
                elif t < time - 2 * deltaT:
                    to_del.append(x)
        for x in to_del:
            del cameras_queue[camera][x]

        # calculation of mu_i for current car
    with open('out.csv', 'w') as out:
        writer = csv.writer(out, lineterminator = '\n')
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
            ratios = []
            for key1, key2 in CarsMutual.keys():
                if CarsMutual[key1, key2] - AVG >= suspicious_level2 and key1 < key2:
                    Key1.append(key1)
                    Key2.append(key2)
                    Weights.append(CarsMutual[key1, key2])
                    Count.append(CarsCounter[key1, key2])
                    ratios.append((key1, key2, CarsMutual[key1, key2]))
            ratios.sort(key=lambda x: x[2], reverse=True)
            for x in ratios:
                writer.writerow(x)
    return CarsMutual

host = 'localhost'
password = '666666'
gr = Graph(host=host, password=password, bolt=True)
deltaT = 120000

res = SimpleAlert(gr, deltaT, 0.9)