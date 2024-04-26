import numpy as np

from feature_extraction.dense_feature_extraction import DenseFeatureExtractor
from gaze_tracking_graph import GazeTrackingGraph


class EuclideanGraph(GazeTrackingGraph):
    """Complete graph with euclidean edges
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
        self.name = 'EuclideanGraph_{}_{}'.format(self.dicom_id, self.reflacx_id)


    def calc_edge(self):
        """Edge weights are inverse to distance, normalized to a [0, 1] interval.
        """
        nodes = self.nodes
        result = [[0.0 for j in range(len(nodes))] for i in range(len(nodes))]
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes[i:], i):
                if i == j:
                    result[i][j] = 1.0 if self.self_edges else 0.0
                    continue
                
                result[i][j] = (2 ** 0.5 - 
                                ((self.nodes[i].norm_x - self.nodes[j].norm_x) ** 2 +
                                (self.nodes[i].norm_y - self.nodes[j].norm_y) ** 2
                                ) ** 0.5
                               ) / (2 ** 0.5)
                result[j][i] = result[i][j]

        self.adj_mat = np.array(result)