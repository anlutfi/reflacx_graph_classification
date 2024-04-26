import numpy as np

from feature_extraction.dense_feature_extraction import DenseFeatureExtractor
from gaze_tracking_graph import GazeTrackingGraph


class IOUGraph(GazeTrackingGraph):
    """Graph with edges porportional to IOU area between to fixations' crops.
    """
    def __init__(self,
                 dicom_id,
                 reflacx_id,
                 reflacx_sample=None,
                 metadata=None,
                 stdevs=1,
                 feature_extractor=DenseFeatureExtractor(),
                 mean_features=None,
                 self_edges=True):
        super().__init__(dicom_id,
                         reflacx_id,
                         reflacx_sample,
                         metadata,
                         stdevs,
                         feature_extractor,
                         mean_features,
                         self_edges=self_edges)
        self.name = 'IOUGraph_{}_{}'.format(self.dicom_id, self.reflacx_id)


    def calc_edge(self):
        """Edges are 0 for non-intersecting fixation crops and 1 for equal,
        based on their intersection over union.
        """
        nodes = self.nodes
        result = [[0.0 for j in range(len(nodes))] for i in range(len(nodes))]
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i == j:
                    result[i][j] = 1.0 if self.self_edges else 0.0
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

        self.adj_mat = np.array(result)