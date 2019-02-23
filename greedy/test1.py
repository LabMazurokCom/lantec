import pandas as pd
from greedy import GreedyProbabilisticAlgorithm
import json
data = pd.read_csv('nct1.csv', names = ['number', 'camera', 'time'])
greedy = GreedyProbabilisticAlgorithm(alerts=data, vip=6)
print(json.dumps(greedy.findCovers(5, 3)))