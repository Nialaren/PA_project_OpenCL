import k_prototype_utils


class Cluster(object):
    def __init__(self, centroid, numbers, strings):
        self.centroid = centroid
        self.numbers = numbers
        self.strings = strings

        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def update_centroid(self):
        if len(self.children) < 2:
            return False

        num_sum = [0] * len(self.children[0])
        sum_string = [None] * len(self.children[0])
        avg_sum = [0] * len(self.children[0])
        for child in self.children:
            for num in self.numbers:
                num_sum[num] += child[num]

            for s in self.strings:
                if sum_string[s] is None:
                    sum_string[s] = dict()
                if child[s] not in sum_string[s]:
                    tmp = sum_string[s]
                    tmp.update({child[s]: 1})
                else:
                    sum_string[s][child[s]] += 1

        for i in self.numbers:
            avg_sum[i] = num_sum[i]/len(self.children)

        for i in self.strings:
            actual_dictionary = sum_string[i]
            avg_sum[i] = max(actual_dictionary, key=lambda j: actual_dictionary[j])

        change = False
        for i in self.numbers:
            if avg_sum[i] != self.centroid[i]:
                change = True

        for i in self.strings:
            if avg_sum[i] != self.centroid[i]:
                change = True

        self.centroid = avg_sum

        return change

    def clean_cluster(self):
        self.children = []

    def compute_sse(self):
        sse = k_prototype_utils.find_dispersion(self.children, self.numbers, self.strings)
        return sse

    def print_cluster(self):
        for child in self.children:
            return child
    # def __str__(self):
    #     buff = 'CHILDREN\n'
    #     # print self.children
    #     for child in self.children:
    #         for c in child:
    #             # print c
    #             buff = buff + '{0} '.format(c)
    #         buff += '\n'
    #     # print buff
    #     return buff
