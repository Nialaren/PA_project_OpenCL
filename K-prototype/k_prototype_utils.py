import math
import pandas as pd
from astropy.table import Table, Column


def find_min(data, numbers):
    minimum = [None] * len(data[0])
    for d in data:
        for n in numbers:
            if minimum[n] is None or minimum[n] > d[n]:
                minimum[n] = d[n]

    return minimum


def find_max(data, numbers):
    maximum = [None] * len(data[0])
    for d in data:
        for n in numbers:
            if maximum[n] is None or maximum[n] < d[n]:
                maximum[n] = d[n]

    return maximum


def find_avg(data, numbers, strings):
    summary = [0] * len(data[0])
    sum_string = [None] * len(data[0])
    avg_sum = [0] * len(data[0])
    for child in data:
        for num in numbers:
            if child[num] is None:
                continue

            summary[num] += child[num]

        for s in strings:
            if sum_string[s] is None:
                sum_string[s] = dict()
            if child[s] not in sum_string[s]:
                tmp = sum_string[s]
                tmp.update({child[s]: 1})
            else:
                sum_string[s][child[s]] += 1

    for i in numbers:
        avg_sum[i] = summary[i]/len(data)

    for i in strings:
        actual_dictionary = sum_string[i]
        avg_sum[i] = max(actual_dictionary, key=lambda j: actual_dictionary[j])

    return avg_sum


def find_dispersion(data, numbers, strings):
    """
    :param data:
    :param numbers:
    :param strings:
    :return:
    """
    average = find_avg(data, numbers, strings)
    dis_sum = 0
    for d in data:
        dis_sum += math.pow(euclidean_distance(average, d, numbers, strings, 0.5), 2.0)

    return dis_sum


def find_variance(data, numbers, strings):
    variance = math.sqrt(find_dispersion(data, numbers, strings))

    return variance


def euclidean_distance(a, b, numbers, strings, w):
    """

    :param a:
    :param b:
    :param numbers:
    :param strings:
    :param w: gama
    :return:
    """
    num_sum = 0
    str_sum = 0
    for n in numbers:
        num_sum += math.pow(a[n] - b[n], 2.0)
    for s in strings:
        if a[s] != b[s]:
            str_sum += 1

    distance = num_sum + str_sum * w

    return distance


def clean_data():
        file = open("..\data\data.json", "r")
        filedata = file.read()
        file.close()

        newdata = filedata.replace(",]", "]")

        file = open("..\data\data_clean.json", "w")
        file.write(newdata)
        file.close()


def data_to_json(clusters):
    text_file = open("..\data\data.json", "w")

    num = 1
    text_file.write('{"clusters":[')
    for child in clusters:
        text_file.write('[')
        num += 1
        for i in range(len(child.children)):
            text_file.write("[{0}],".format(",".join(map(repr, child.children[i]))))
        text_file.write('],')

    text_file.write(']}')
    text_file.close()

    clean_data()

    print "Data zapisane"


def data_to_output(words, data, clusters, num_clusters, num_iterations, numbers, strings):
    text_file = open("output.arff", "w")
    minimum = find_min(data, numbers)
    maximum = find_max(data, numbers)
    average = find_avg(data, numbers, strings)
    dispersion = find_dispersion(data, numbers, strings)
    variance = find_variance(data, numbers, strings)

    text_file.write("Output from data\n")
    template = "{0:<15} {1:^20} {2:^20} {3:^20}"
    text_file.write(template.format("Name", "Minimum", "Maximum", "Mean/Modus"))
    text_file.write("\n")
    for i in range(len(words)):
        text_file.write(template.format(words[i], minimum[i], maximum[i], average[i]))
        text_file.write("\n")

    text_file.write('\nDispersion' + ' {0}'.format(dispersion))
    text_file.write('\nVariance' + ' {0}'.format(variance))

    text_file.write('\n\n#######################################################################\n\n')
    text_file.write('Pocet iteraci: {0}'.format(num_iterations))
    text_file.write("\n")
    text_file.write('Pocet zhlukov: {0}'.format(num_clusters))
    text_file.write("\n")
    c = 0
    for cluster in clusters:
        # print cluster.children
        c = cluster.compute_sse()
        c *= c
    text_file.write('SSE: {0}\n'.format(c))


    text_file.write('\n\n#######################################################################\n')
    text_file.write('##############################  ZHLUKY  ###############################\n')
    text_file.write('#######################################################################\n\n')


    num = 1
    for child in clusters:
        text_file.write('\ncluster #{0}\n'.format(num))
        num += 1
        for i in range(len(child.children)):
            text_file.write('{0}\n'.format(child.children[i]))

    print "Data zapisane"
    # text_file.close()

