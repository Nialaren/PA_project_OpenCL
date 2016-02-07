import random
import k_prototype_utils
from Cluster import Cluster


def k_prototype(k, n, data, numbers, strings, w):
    """
    :param k:
    :param n:
    :param data:
    :param numbers:
    :param strings:
    :param w: gama
    :return:
    """
    clusters = []
    change = True
    iterations = 0

    for i in range(k):
        clusters.append(Cluster(random.choice(data), numbers, strings))

    while(change is True and n > iterations):
        for cluster in clusters:
            cluster.clean_cluster()

        for point in data:
            nearest_cluster = None
            nearest_cluster_value = None
            for cluster in clusters:
                distance = k_prototype_utils.euclidean_distance(point, cluster.centroid, numbers, strings, w)
                if nearest_cluster_value is None or nearest_cluster_value > distance:
                    nearest_cluster_value = distance
                    nearest_cluster = cluster

            nearest_cluster.add_child(point)

        change = False
        for cluster in clusters:
            if cluster.update_centroid() is True:
                change = True

        iterations += 1

    return clusters
