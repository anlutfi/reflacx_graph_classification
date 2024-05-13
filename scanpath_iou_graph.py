import numpy as np

from feature_extraction.dense_feature_extraction import DenseFeatureExtractor
from gaze_tracking_graph import GazeTrackingGraph


class ScanPathIOUGraph(GazeTrackingGraph):
    """Graph with edges determined by fixations' viewing order
    """
    edge_type_names = ['scanpath', 'iou']

    @staticmethod
    def get_edge_type_names():
        return ScanPathIOUGraph.edge_type_names
    
    
    def __init__(self,
                 dicom_id,
                 reflacx_id,
                 reflacx_sample=None,
                 metadata=None,
                 stdevs=1,
                 feature_extractor=DenseFeatureExtractor(),
                 mean_features=None,
                 self_edges=True,
                 bidirectional=True):
        super().__init__(dicom_id,
                         reflacx_id,
                         reflacx_sample,
                         metadata,
                         stdevs,
                         feature_extractor,
                         mean_features,
                         self_edges=self_edges,
                         bidirectional=bidirectional)
        self.name = 'ScanPathGraph_{}_{}'.format(self.dicom_id, self.reflacx_id)


    def calc_edge(self):
        """A node is neighbor to another if they are next or previous in the scanpath
        """
        nodes = self.nodes
        result = [[0.0 for j in range(len(nodes))] for i in range(len(nodes))]
        for i in range(len(nodes)):
            for j in range(i, len(nodes)):
                if i == j and self.self_edges:
                    result[i][j] = 1.0
                elif j - i == 1:
                    result[i][j] = 1.0
                    if self.bidirectional:
                        result[j][i] = result[i][j]

        self.edges['scanpath'] = np.array(result)

        result = [[0.0 for j in range(len(nodes))] for i in range(len(nodes))]
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes[i:], i):
                if i == j:
                    result[i][j] = 1.0 if self.self_edges else 0.0
                    continue
                i_x_min, i_y_min = node_i.topleft
                i_x_max, i_y_max = node_i.bottomright
                j_x_min, j_y_min = node_j.topleft
                j_x_max, j_y_max = node_j.bottomright
                
                xA = max(i_x_min, j_x_min)
                yA = max(i_y_min, j_y_min)
                xB = min(i_x_max, j_x_max)
                yB = min(i_y_max, j_y_max)
                
                intersec = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                
                i_area = ((i_x_max - i_x_min + 1) * (i_y_max - i_y_min + 1))
                j_area = ((j_x_max - j_x_min + 1) * (j_y_max - j_y_min + 1))
                
                iou = intersec / (i_area + j_area - intersec)

                result[i][j] = result[j][i] = iou

        self.edges['iou'] = np.array(result)