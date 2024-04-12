import numpy as np

def scanpath_edges(nodes):
    return np.array([[1.0 if abs(i - j) <= 1 else 0.0
                      for j in range(len(nodes))]
                     for i in range(len(nodes))])


def iou_edges(nodes):
    result = [[0.0 for j in range(len(nodes))] for i in range(len(nodes))]
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if i == j:
                result[i][j] = 1.0 # TODO review self edges
                continue
            xA = max(node_i.viewed_x_min, node_j.viewed_x_min)
            yA = max(node_i.viewed_y_min, node_j.viewed_y_min)
            xB = min(node_i.viewed_x_max, node_j.viewed_x_max)
            yB = min(node_i.viewed_y_max, node_j.viewed_y_max)
            
            intersec = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            
            i_area = ((node_i.viewed_x_max - node_i.viewed_x_min + 1) *
                      (node_i.viewed_y_max - node_i.viewed_y_min + 1))
            j_area = ((node_j.viewed_x_max - node_j.viewed_x_min + 1) *
                      (node_j.viewed_y_max - node_j.viewed_y_min + 1))
            
            iou = intersec / (i_area + j_area - intersec)

            result[i][j] = iou

    return np.array(result)


def euclidean_edges(nodes): # TODO review min value being 0
    return np.array([[(2 ** 0.5 - 
                       ((nodes[i].norm_x - nodes[j].norm_x) ** 2 +
                       (nodes[i].norm_y - nodes[j].norm_y) ** 2
                       ) ** 0.5
                      ) / (2 ** 0.5)
                      for j in range(len(nodes))]
                     for i in range(len(nodes))]) # TODO review self edges