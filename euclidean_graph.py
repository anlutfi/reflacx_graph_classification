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
                 mean_features=None):
        super().__init__(dicom_id,
                         reflacx_id,
                         reflacx_sample,
                         metadata,
                         stdevs,
                         feature_extractor,
                         mean_features)
        self.name = 'EuclideanGraph_{}_{}'.format(self.dicom_id, self.reflacx_id)


    def calc_edge(self):
        """Edge weights are inverse to distance, normalized to a [0, 1] interval.
        """
        nodes = self.nodes
        self.adj_mat = np.array([[(2 ** 0.5 - 
                                ((nodes[i].norm_x - nodes[j].norm_x) ** 2 +
                                (nodes[i].norm_y - nodes[j].norm_y) ** 2
                                ) ** 0.5
                               ) / (2 ** 0.5)
                               for j in range(len(nodes))]
                              for i in range(len(nodes))]) # TODO review self edges