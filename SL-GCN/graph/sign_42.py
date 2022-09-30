import sys

sys.path.extend(['../'])
from graph import tools

num_node = 42
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(0,0),(0, 1), (1, 2),(2, 3),(3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (0,9),(9,10),(10,11),(11,12),
                    (0,13),(13,14),(14,15),(15,16),
                    (0,17),(17,18),(18,19),(19,20),
                    (21,21),
                    (21, 22), (22,23),(23, 24),(24, 25),
                    (21, 26), (26, 27), (27, 28), (28, 29),
                    (21,30),(30,31),(31,32),(32,33),
                    (21,34),(34,35),(35,36),(36,37),
                    (21,38),(38,39),(39,40),(40,41)
                   ]

inward = [(i , j ) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)
