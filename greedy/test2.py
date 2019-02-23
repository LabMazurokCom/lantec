import pandas as pd
from greedy import GreedyProbabilisticAlgorithm
import numpy as np
import json
import datetime
def f(time):
    return datetime.datetime.fromtimestamp(time).year

data = pd.read_csv('out.csv', names = ['number', 'camera', 'time'])
#data = data.iloc[0:1000]
#groups = data.groupby('number')
vip = '0717'
#getYearVector = np.vectorize(f)
#data = data[getYearVector(data['time']) == 2019]
greedy = GreedyProbabilisticAlgorithm(alerts=data, vip=vip)
print(json.dumps(greedy.findCovers(15, 5)))