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

        self.adj_mat = np.array(result)