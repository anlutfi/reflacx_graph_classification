import numpy as np

from feature_extraction.dense_feature_extraction import DenseFeatureExtractor
from gaze_tracking_graph import GazeTrackingGraph


class ScanPathGraph(GazeTrackingGraph):
    """Graph with edges determined by fixations' viewing order
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
        self.name = 'ScanPathGraph_{}_{}'.format(self.dicom_id, self.reflacx_id)


    def calc_edge(self, self_edges=True):
        """A node is neighbor to another if they are next or previous in the scanpath
        """
        def w(i, j):
            if i == j:
                return 1.0 if self_edges else 0.0
            return 1.0 if abs(i - j) == 1 else 0.0
        
        self.adj_mat =  np.array([[w(i, j)
                                   for j in range(len(self.nodes))]
                                  for i in range(len(self.nodes))])