from py2neo import Graph
import numpy as np
from collections import deque
from collections import OrderedDict
import heapq

class GraphReader:
    def __init__(self, graph, start_from=0):
        self.graph = graph
        self.last_id = start_from
        self.start = self.last_id

    def get_next_alert(self):
        query = "match (a:Alert{id: {id}}), (c:Camera)-[:detectionCamera]->(a)-[:detectedVehicle]-(v:Vehicle) return a.time as time, v.number as number, c.hash as camera"
        cursor = self.graph.run(query, id=self.last_id)
        self.last_id += 1
        if cursor.forward():
            return {'time': cursor.current['time'], 'number': cursor.current['number'], 'camera': cursor.current['camera']}
        else:
            return None

    def get_cameras(self):
        query = "match (c:Camera) return c.hash"
        return self.graph.run(query).to_series()

    def get_vehicle_alerts(self, vehicle):
        cameras_sequence_query = "match (c:Camera)-[:detectionCamera]->(a:Alert)-[:detectedVehicle]->(v:Vehicle{number: $number}) using index v:Vehicle(number) return c.hash order by a.time"
        k = self.graph.run(cameras_sequence_query, number=vehicle).to_series()
        return k


class Vehicle:
    def __init__(self, number, graph_reader, k, last_vip_bigrams_num):
        self.graph_reader = graph_reader
        self.number = number
        self.last_alerts = deque()
        self.bigrams_ratio = None
        self._suspect_ratios = deque(np.zeros(max(last_vip_bigrams_num, 0), dtype=np.float64))
        self.ratio = 0
        self.k = k

    def recover(self, last_vip_bigrams_num):
        self._suspect_ratios = deque(np.zeros(max(last_vip_bigrams_num, 0), dtype=np.float64))
        self.ratio = 0

    def _get_bigrams_ratio(self):
        bigrams_ratio = {}
        bigrams_row_ratio = {}
        vehicle_alerts = self.graph_reader.get_vehicle_alerts(self.number)
        prev_camera = vehicle_alerts[0]
        for camera in vehicle_alerts[1:]:
            if prev_camera not in bigrams_ratio:
                bigrams_ratio[prev_camera] = {}
                bigrams_row_ratio[prev_camera] = 0
            if camera not in bigrams_ratio[prev_camera]:
                bigrams_ratio[prev_camera][camera] = 0
            bigrams_ratio[prev_camera][camera] += 1
            bigrams_row_ratio[prev_camera] += 1
            prev_camera = camera
        for prev_camera, row in bigrams_ratio.items():
            for camera, value in row.items():
                bigrams_ratio[prev_camera][camera] = value / bigrams_row_ratio[prev_camera]
        return bigrams_ratio

    def get_bigram_suspicion(self, source, to):
        if self.bigrams_ratio is None:
            self.bigrams_ratio = self._get_bigrams_ratio()
        p = 0
        if source in self.bigrams_ratio:
            row = self.bigrams_ratio[source]
            if to in row:
                p = row[to]
        return (1 - p) / (1 + self.k * p)

    def set_suspect_ratios(self, ratios):
        self._suspect_ratios = deque(ratios)
        if len(self._suspect_ratios) == 0:
            raise Exception
        self.ratio = min(1, np.sum(ratios))

    def set_last_ratio(self, ratio):
        self._suspect_ratios[0] = ratio
        self.ratio = min(1, np.sum(self._suspect_ratios))

    def get_suspect_ratios(self):
        return self._suspect_ratios


class WMA:
    def __init__(self, graph_reader, vip, n=5, alpha=2, beta=2, k=5, delta_time_minus=120000, delta_time_plus=120000, min_cameras_life=120000, a=2, type=1):
        self.graph_reader = graph_reader
        self.n = n
        self.k = k
        self.cameras = self.graph_reader.get_cameras()
        self.weights = np.array([np.array([(1 - (k / m) ** alpha) ** alpha for k in range(m)]) for m in range(2 * n)])
        weights_sum = np.array([np.sum(self.weights[i]) for i in range(2 * n)])
        self.weights /= weights_sum
        self.delta_time_minus = delta_time_minus
        self.delta_time_plus = delta_time_plus
        self.vehicles = {}
        self.suspects = {}
        self.vip = vip
        self.last_vip_alerts = deque()
        self.buffer = {}
        self.vehicle = {}
        self.beta = beta
        self.a = a
        self.type = type
        self.min_cameras_life = min_cameras_life
        for camera in self.cameras:
            self.buffer[camera] = OrderedDict()
        self.start_inspection()

    def clear_buffer(self, camera, time):
        dict = self.buffer[camera]
        to_del = True
        while to_del and len(dict) > 0:
            vehicle = dict.popitem()
            if vehicle[1][1] >= time - self.delta_time_plus:
                dict[vehicle[0]] = vehicle[1]
                to_del = False

    def clear_last_vip_alerts(self, time):
        has_out_of_range_cameras = True
        deleted = 0
        while has_out_of_range_cameras and len(self.last_vip_alerts) > self.n + 1:
            camera, vip_time = self.last_vip_alerts.pop()
            deleted += 1
            if vip_time >= time - self.min_cameras_life:
                deleted -= 1
                self.last_vip_alerts.append((camera, vip_time))
                has_out_of_range_cameras = False
        if deleted > 0:
            for number, vehicle in self.suspects.items():
                self.delete_from_ratio(vehicle, deleted)

    def clear_vehicle_alerts(self, vehicle):
        alerts = vehicle.last_alerts
        to_del = True
        while to_del and len(alerts) > 2:
            alert_time = alerts[-1][1]
            alert_camera = alerts[-1][0]
            if self.may_be_cleared_alert(alert_camera, alert_time):
                alerts.pop()
            else:
                to_del = False

    def delete_from_ratio(self, vehicle, deleted):
        ratios = np.array(vehicle.get_suspect_ratios())
        ratios /= self.weights[len(ratios)]
        ratios = ratios[:len(ratios) - deleted]
        ratios *= self.weights[len(ratios)]
        vehicle.set_suspect_ratios(ratios)

    def fix_vehicle(self, number, camera, time):
        if number not in self.vehicles:
            self.vehicles[number] = Vehicle(number, self.graph_reader, self.k, len(self.last_vip_alerts) - 1)
        camera_buffer = self.buffer[camera]
        camera_buffer[number] = (self.vehicles[number], time)
        camera_buffer.move_to_end(number, False)
        if number not in self.suspects:
            self.suspects[number] = self.vehicles[number]
            self.suspects[number].recover(len(self.last_vip_alerts) - 1)
        cur_vehicle = self.vehicles[number]
        ratios = np.array(cur_vehicle.get_suspect_ratios(), dtype=np.float_)
        cur_vehicle.last_alerts.appendleft((camera, time))
        if len(self.last_vip_alerts) > 1:
            vip_alerts_list = list(self.last_vip_alerts)
            for i in range(len(vip_alerts_list)):
                if vip_alerts_list[i][0] == camera and time <= vip_alerts_list[i][1] + self.delta_time_plus:
                    if i > 0 and ratios[i - 1] == 0:
                        for next_camera, next_time in cur_vehicle.last_alerts:
                            if next_camera == vip_alerts_list[i - 1][0] \
                                    and vip_alerts_list[i - 1][1] - self.delta_time_minus <= next_time <= vip_alerts_list[i - 1][1] + self.delta_time_plus:
                                ratios[i - 1] = self.get_suspect_ratio(i - 1, cur_vehicle, vip_alerts_list[i][0],
                                                           vip_alerts_list[i - 1][0])
                                break
                    if i < len(ratios) and ratios[i] == 0:
                        for last_camera, last_time in cur_vehicle.last_alerts:
                            if last_camera == vip_alerts_list[i + 1][0] and \
                                    vip_alerts_list[i + 1][1] - self.delta_time_minus <= last_time <= vip_alerts_list[i + 1][1] + self.delta_time_plus:
                                ratios[i] = self.get_suspect_ratio(i, cur_vehicle, vip_alerts_list[i + 1][0],
                                                                       vip_alerts_list[i][0])
                                break
            cur_vehicle.set_suspect_ratios(ratios)
            cur_vehicle.set_suspect_ratios(ratios)

    def get_suspect_ratio(self, i, vehicle, source, to):
        return self.weights[len(self.last_vip_alerts) - 1][i] * vehicle.get_bigram_suspicion(source, to)

    def clear_suspects(self, time):
        to_del = []
        for number, vehicle in self.suspects.items():
            if self.may_be_cleared_vehicle(vehicle, time):
                to_del.append(vehicle.number)
        for number in to_del:
            del self.suspects[number]

    def may_be_cleared_vehicle(self, vehicle, time):
        to_del = (vehicle.ratio == 0) and (vehicle.last_alerts[0][1] <= time - self.delta_time_minus)
        if to_del:
            for cam, time in vehicle.last_alerts:
                to_del = self.may_be_cleared_alert(cam, time)
                if not to_del:
                    break
        return to_del

    def may_be_cleared_alert(self, cam, time):
        for vip_cam, vip_time in self.last_vip_alerts:
            if vip_cam == cam and vip_time - self.delta_time_minus <= time <= vip_time + self.delta_time_plus:
                return False
        return True

    def get_max_wma_list_finite_differences(self):
        ratios = [(-v.ratio, k) for k, v in self.suspects.items()]
        ratios.append((0, 'FICTIVE'))
        heapq.heapify(ratios)
        max_wma_list = []
        last_ratio, last_number = heapq.heappop(ratios)
        last_ratio = -last_ratio
        if last_ratio != 0:
            max_wma_list.append((last_number, last_ratio))
            while len(ratios) > 0 and len(max_wma_list) <= 2 * self.a + 1:
                cur_ratio, cur_number = heapq.heappop(ratios)
                cur_ratio = -cur_ratio
                if cur_ratio != 0:
                    buffer = [(cur_number, cur_ratio)]
                    next_ratio, next_number = heapq.heappop(ratios)
                    while next_ratio == -cur_ratio:
                        buffer.append((next_number, -next_ratio))
                        next_ratio, next_number = heapq.heappop(ratios)
                    heapq.heappush(ratios, (next_ratio, next_number))
                    next_ratio = -next_ratio
                    if (last_ratio - cur_ratio) / (cur_ratio - next_ratio) < self.beta:
                        max_wma_list += buffer
                        last_ratio, last_number = cur_ratio, cur_number
                    else:
                        break
                else:
                    break
        return max_wma_list

    def get_max_wma_list_averages(self):
        ratios = [(-v.ratio, k) for k, v in self.suspects.items()]
        N = 2 * (2 * self.a + 1)
        heapq.heapify(ratios)
        ratios_len = len(ratios)
        n_largest = [heapq.heappop(ratios) for i in range(min(N, ratios_len))]
        n_largest = [(v[1], -v[0]) for v in n_largest]
        if n_largest[0][1] > 0:
            avg_left = n_largest[0][1]
            if len(n_largest) > 1:
                avg_right = np.mean([k[1] for k in n_largest[1:]])
            else:
                avg_right = 0
            max_res = avg_left - avg_right
            best_cut = 0
            for i in range(1, min(len(n_largest) - 1, 2 * self.a + 2)):
                r = (avg_left * i + n_largest[i][1]) / (i + 1) - (avg_right * (N - i) - n_largest[i][1]) / (len(n_largest) - i - 1)
                if r > max_res:
                    max_res = r
                    best_cut = i
            return n_largest[:best_cut + 1]
        else:
            return []

    def start_inspection(self):
        alert = self.graph_reader.get_next_alert()
        print(alert)
        j = 0
        while alert is not None:
            if j == 12000:
                break
            j += 1
            number = alert['number']
            camera = alert['camera']
            time = alert['time']
            self.clear_buffer(camera, time)
            self.clear_last_vip_alerts(time)
            self.clear_suspects(time)
            if number == self.vip:
                self.fix_vip_auto(camera, time)
            else:
                self.fix_vehicle(number, camera, time)
            if self.type == 1:
                max_wma_list = self.get_max_wma_list_finite_differences()
            else:
                max_wma_list = self.get_max_wma_list_averages()
            if number == self.vip or number in [i[0] for i in max_wma_list]:
                if len(max_wma_list) != 0:
                    if len(max_wma_list) <= self.a:
                        print('RED')
                    elif len(max_wma_list) <= 2 * self.a + 1:
                        print('YELLOW')
                    else:
                        print('GREEN')
                    print(j, max_wma_list)
            alert = self.graph_reader.get_next_alert()

    def aging(self, vehicle):
        ratios = np.array(vehicle.get_suspect_ratios(), dtype=np.float64)
        ratios /= self.weights[len(ratios)]
        ratios *= self.weights[len(ratios) + 1][1:]
        ratios = deque(ratios)
        ratios.appendleft(0)
        vehicle.set_suspect_ratios(ratios)

    def fix_vip_auto(self, vip_camera, vip_time):
        self.last_vip_alerts.appendleft((vip_camera, vip_time))
        if len(self.last_vip_alerts) > 1:
            prev_vip_camera, prev_vip_time = self.last_vip_alerts[1]
            camera_buffer = self.buffer[vip_camera]
            for number, vehicle in self.suspects.items():
                if len(vehicle.get_suspect_ratios()) < len(self.last_vip_alerts):
                    self.aging(vehicle)
            for number, vehicle in camera_buffer.items():
                vehicle = self.suspects[number]
                self.clear_vehicle_alerts(vehicle)
                for camera, time in vehicle.last_alerts:
                    if camera == prev_vip_camera and prev_vip_time - self.delta_time_minus <= time <= prev_vip_time + self.delta_time_plus:
                        vehicle.set_last_ratio(self.get_suspect_ratio(0, vehicle, prev_vip_camera, vip_camera))


host = 'localhost'
password='666666'
gr = Graph(host=host, bolt=True, password=password)
start_from = 21939336 - 500
grr = GraphReader(gr, start_from)
a = WMA(grr, '0717', delta_time_minus=120000, delta_time_plus=120000, min_cameras_life=120000, type=2, k=2, alpha=3)