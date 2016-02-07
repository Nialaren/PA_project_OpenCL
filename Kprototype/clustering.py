import random as rand
import math
import numpy as np
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D

class clustering:
    def __init__(self, locs_, k_):
        self.locations = locs_
        self.k = k_
        self.clusters = []      #clusters of nodes
        self.means = []         #means of clusters
        self.debug = False      #debug

    # returns the next random node
    def next_random(self, index, points, clusters):
        dist = {}
        for point1 in points:
            if self.debug:
                print 'bod 1: %f %f' % (point1.X, point1.Y)

            for cluster in clusters.values():       # compute distance from this node to all others points in cluster
                point2 = cluster[0]
                if self.debug:
                    print 'bod 2: %f %f' % (point2.X, point2.Y)
                if point1 not in dist:
                    dist[point1] = math.sqrt(math.pow(point1.X - point2.X, 2.0) + math.pow(point1.Y - point2.Y, 2.0))
                else:
                    dist[point1] += math.sqrt(math.pow(point1.X - point2.X, 2.0) + math.pow(point1.Y - point2.Y, 2.0))
        if self.debug:
            for key, value in dist.items():
                print "(%f %f)==> %f" % (key.X, key.Y, value)

        count_ = 0
        max_ = 0
        for key, value in dist.items():
            if count_ == 0:
                max_ = value
                max_point = key
                count_ += 1
            else:
                if value > max_:
                    max_ = value
                    max_point = key
        return max_point        # max_point has the maximum distance from previous nodes

    def initial_means(self, points):
        point_ = rand.choice(points)
        if self.debug:
            print 'bod #0: %f %f' % (point_.X, point_.Y)

        clusters = dict()
        clusters.setdefault(0, []).append(point_)
        points.remove(point_)

        for i in range(1, self.k):
            point_ = self.next_random(i, points, clusters)
            if self.debug:
                print 'bod #%d: %f %f' % (i, point_.X, point_.Y)

            clusters.setdefault(i, []).append(point_)
            points.remove(point_)

        self.means = self.compute_mean(clusters)
        if self.debug:
            print "Pociatocne uzly:"
            self.print_means(self.means)

    def compute_mean(self, clusters):
        means = []

        for cluster in clusters.values():
            mean_point = Point(0.0, 0.0)
            c = 0.0

            for point in cluster:
                mean_point.X += point.X
                mean_point.Y += point.Y
                c += 1.0
            mean_point.X = mean_point.X / c
            mean_point.Y = mean_point.Y / c
            means.append(mean_point)

        return means
    # check current mean and the previous and say if we should stop
    def update_means(self, means, threshold):
        for i in range(len(self.means)):
            mean_1 = self.means[i]
            mean_2 = means[i]

            if self.debug:
                print "uzol_1(%f, %f)" % (mean_1.X, mean_1.Y)
                print "uzol_2(%f, %f)" % (mean_2.X, mean_2.Y)

            if math.sqrt(math.pow(mean_1.X - mean_2.X, 2.0) + math.pow(mean_1.Y - mean_2.Y, 2.0)) > threshold :
                return False
        return True

    # print means
    def print_means(self, means):
        for point in means:
            print "%f %f" % (point.X, point.Y)

    def assign_points(self, points):
        if self.debug:
            print "priradenie bodov"

        clusters = dict()
        for point in points:
            dist = []
            if self.debug:
                print "bod(%f, %f)" % (point.X, point.Y)
            for mean in self.means: # find best cluster in this node
                dist.append(math.sqrt(math.pow(point.X - mean.X, 2.0) + (math.pow(point.Y - mean.Y, 2.0))))

            if self.debug:
                print dist

            c_ = 0
            index = 0
            min_ = dist[0]

            for d in dist:
                if d < min_:
                    min_ = d
                    index = c_
                c_ += 1

            if self.debug:
                print "index: %d" % index

            clusters.setdefault(index, []).append(point)

        return clusters

    # print cluster points
    def print_clusters(self, clusters):
        cluster_cn = 1
        for cluster in clusters.values():
            print "Uzly v zhluku #%d: " % cluster_cn
            cluster_cn += 1

            for point in cluster:
                print "bod(%f, %f)" % (point.X, point.Y)

    def save_clusters(self, clusters):
        file = open('iris.txt', 'w');
        cluster_cn = 1
        for cluster in clusters.values():
            file.write("\nUzly v zhluku #%d:\n" % cluster_cn)
            cluster_cn += 1

            for point in cluster:
                file.write("bod(%f, %f),\n" % (point.X, point.Y))

    def k_means(self):
        if len(self.locations) < self.k:
            return -1   #error

        points_ = [point for point in self.locations]

        self.initial_means(points_)     # compute initial means
        stop = False

        while not stop:
            points_ = [point for point in self.locations]
            clusters = self.assign_points(points_)

            if self.debug:
                self.print_clusters(clusters)

            means = self.compute_mean(clusters)

            if self.debug:
                print "uzly:"
                print self.print_means(means)
                print "aktualizovat uzol:"
            stop = self.update_means(means, 0.01)       # este domysliet
            if not stop:
                self.means = []
                self.means = means

        self.clusters = clusters

        return 0

    def min(self):
        minimumX = []
        minimumY = []
        for point in self.locations:
           minimumX.append(point.X)
           minimumY.append(point.Y)

        minX = min(minimumX)
        minY = min(minimumY)


        print(minX)