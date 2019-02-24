from py2neo import Graph
import pandas as pd
from collections import deque
import math
from scipy.stats import norm as norm


class GraphReader:
    def __init__(self, graph, start_id=0, last_id=math.inf):
        self.graph = graph
        self.alerts_read = start_id
        self.last_id = last_id

    def get_next_alerts(self, rows):
        query = "match (c:Camera)-[:detectedCamera]->(a:Alert)-[:detectedVehicle]->" \
                "(v:Vehicle) where $from_ind <= a.id < $to_ind return c.hash as camera, " \
                "a.time as time, v.number as number order by a.time"
        frame = self.graph.run(query, from_ind = self.alerts_read, to_ind = min(self.alerts_read + rows, self.last_id)).to_data_frame()
        self.alerts_read += len(frame)
        return frame

class Vehicle:
    def __init__(self, number):
        self.pre_last_alert = None
        self.last_alert = None
        self.number = number
        self.pursuit_ratios = {}

    def add_meeting(self, number, weight):
        if number != self.number:
            if number not in self.pursuit_ratios:
                self.pursuit_ratios[number] = 0
            self.pursuit_ratios[number] += weight

    def add_alert(self, alert):
        self.pre_last_alert = self.last_alert
        self.last_alert = alert


class SmartTrigramAlgorithm:
    CAMERA_COLUMN = 0
    NUMBER_COLUMN = 1
    TIME_COLUMN = 2

    ONE_TWO_WEIGHT = 0.5
    ONE_THREE_WEIGHT = 1

    MAX_ROWS_PER_READ = 10000

    def __init__(self, delta_time, graph_reader, percent=0.75, max_pursuers=5):
        self.delta_time = delta_time
        self.graph_reader = graph_reader
        self.vehicles = {}
        self.bigrams = {}
        self.percent = percent
        self.max_pursuers = max_pursuers

    def find_pursuit(self):
        alerts = grr.get_next_alerts(SmartTrigramAlgorithm.MAX_ROWS_PER_READ)
        j = 0
        while len(alerts) > 0:
            print(j)
            j += 1
            alerts = alerts.values
            for i in range(len(alerts)):
                row = alerts[i]
                self.process_alert(row)
            alerts = grr.get_next_alerts(SmartTrigramAlgorithm.MAX_ROWS_PER_READ)
        suspect_ratios = []
        for number, vehicle in self.vehicles.items():
            if len(vehicle.pursuit_ratios) > 0:
                res = self.find_pursuit_ratio(vehicle)
                if 0 < len(res) <= self.max_pursuers:
                    for k, v in res.items():
                        suspect_ratios.append((k, vehicle.number, v))
        suspect_ratios.sort(key=lambda x: x[2], reverse=True)
        return pd.DataFrame(suspect_ratios, columns=['pursuer', 'victim', 'coefficient'])

    def find_pursuit_ratio(self, vehicle):
        if len(vehicle.pursuit_ratios) > 0:
            pursuit_ratios = pd.Series(vehicle.pursuit_ratios)
            ratios_sum = pursuit_ratios.sum()
            pursuit_ratios /= ratios_sum
            t_coeff = norm.ppf(self.percent)
            d = 1 / len(pursuit_ratios)
            d_up = (d + t_coeff ** 2 / (2 * ratios_sum) + t_coeff * math.sqrt(d * (1 - d) / ratios_sum + t_coeff ** 2 / (4 * ratios_sum ** 2))) / (1 + t_coeff ** 2 / ratios_sum)
            return pursuit_ratios[pursuit_ratios > d_up]
        else:
            return []


    def process_alert(self, alert):
        number = alert[SmartTrigramAlgorithm.NUMBER_COLUMN]
        if number not in self.vehicles:
            self.vehicles[number] = Vehicle(number)
        vehicle = self.vehicles[number]
        alert_to_add = (alert[SmartTrigramAlgorithm.CAMERA_COLUMN], alert[SmartTrigramAlgorithm.TIME_COLUMN])
        if vehicle.pre_last_alert is not None:
            self.add_bigram(vehicle, vehicle.pre_last_alert[0], alert_to_add[0], SmartTrigramAlgorithm.ONE_THREE_WEIGHT,
                            alert_to_add[1])
        if vehicle.last_alert is not None:
            self.add_bigram(vehicle, vehicle.last_alert[0], alert_to_add[0], SmartTrigramAlgorithm.ONE_TWO_WEIGHT,
                            alert_to_add[1])
        vehicle.add_alert((alert[SmartTrigramAlgorithm.CAMERA_COLUMN], alert[SmartTrigramAlgorithm.TIME_COLUMN]))

    def add_bigram(self, vehicle, source, dest, weight, time):
        if (source, dest) not in self.bigrams:
            self.bigrams[(source, dest)] = deque([(vehicle.number, time, weight)])
        else:
            current_bigram = self.bigrams[(source, dest)]
            to_del = True
            while to_del and len(current_bigram) > 0:
                loop_number, loop_time, loop_weight = current_bigram[0]
                if loop_time + self.delta_time < time:
                    current_bigram.popleft()
                else:
                    to_del = False
            for loop_number, loop_time, loop_weight in current_bigram:
                vehicle.add_meeting(loop_number, weight)
                self.vehicles[loop_number].add_meeting(vehicle.number, loop_weight)
            current_bigram.append((vehicle.number, time, weight))


host = 'localhost'
password='666666'
gr = Graph(host=host, bolt=True, password=password)
delta_time = 120000
start_from = 21500186
last_id = start_from + 20000
grr = GraphReader(gr, start_from)
sta = SmartTrigramAlgorithm(delta_time=delta_time, graph_reader=grr)
suspects = sta.find_pursuit()
suspects.to_csv('2019-02-19 00_00_04 --- 2019-02-20 18_58_41 suspects.csv')

