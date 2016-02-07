import csv
from config import config
from config import get_arrays
import algorithm
import k_prototype_utils

data = []
words = []
num_iterations = 15
num_clusters = 5

numbers, strings = get_arrays()  # indexes of types input data

fname = '..\data\dataset.txt'
with open(fname) as f:
    next(f)
    for line in f:
        reader = csv.reader(f, delimiter=",")
        for line in reader:
            point = []
            for i in range(len(config.attributes)):
                if len(line[i]) == 0:
                    point.append(None)
                    continue

                if config.attributes[i] == 'float':
                    point.append(float(line[i]))

                if config.attributes[i] == 'string':
                    point.append(str(line[i]))

            data.append(point)
with open(fname) as f:
    head = f.readline()
    words = head.split(",")

    # print words

# replace empty values in data array
average = k_prototype_utils.find_avg(data, numbers, strings)
for d in data:
    for n in numbers:
        if d[n] is None:
            d[n] = average[n]

    for s in strings:
        if d[s] is None:
            d[s] = average[s]


clusters = algorithm.k_prototype(num_clusters, num_iterations, data, numbers, strings, 0.5)

k_prototype_utils.data_to_json(clusters)
