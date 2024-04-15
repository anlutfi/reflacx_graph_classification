from feature_extraction.dense_feature_extraction import DenseFeatureExtractor
from fixation_node import FixationNode
from pyvis.network import Network
from os.path import sep

from reflacx_labels import PHASE_1_LABELS, PHASE_2_3_LABELS

class GazeTrackingGraph:
    """Represents a REFLACX datapoint as a graph of its gaze fixations
    Nodes are each fixation and edges are to be defined by inheritance
    """
    @staticmethod
    def edge_csv_header():
        """returns a header of class attributes' names
        to be used as header of csv file
        """
        return "src_id, dst_id, weight"
    
    
    def __init__(self,
                 dicom_id,
                 reflacx_id,
                 reflacx_sample=None,
                 metadata=None,
                 stdevs=1,
                 feature_extractor=DenseFeatureExtractor(),
                 mean_features=None):
        """param:reflacx_sample is a REFLACX datapoint. If none is provided,
        it will be loaded from param:metadata

        The region of the image observed by a fixation is a gaussian bell with the fixation's position as the mean (center point), extending for a number of standard deviations(param:stdevs). 1 standard deviation = 1 degree.

        if param:mean_features is provided, it will be subtracted from all feature extractions (mean normalization).
        """
        assert reflacx_sample is not None or metadata is not None
        if reflacx_sample is None:
            reflacx_sample = metadata.get_sample(dicom_id, reflacx_id)
        
        self.dicom_id = dicom_id
        self.reflacx_id = reflacx_id
        
        self.name = 'GazeTrackingGraph_{}_{}'.format(self.dicom_id, self.reflacx_id)
        
        self.xray = reflacx_sample.get_dicom_img()
        self.chest_bb = reflacx_sample.get_chest_bounding_box()

        #TODO normalize labels' intervals to [0, 1]. i think its now [1, 5]
        self.phase1_labels = {k: reflacx_sample.data[k] 
                              for k in PHASE_1_LABELS
                              if k in reflacx_sample.data}
        
        self.phase2_3_labels = {k: reflacx_sample.data[k] 
                                for k in PHASE_2_3_LABELS
                                if k in reflacx_sample.data}
        
        self.common_labels = {k: reflacx_sample.data[k]
                              for k in PHASE_1_LABELS
                              if k in PHASE_2_3_LABELS
                              and k in reflacx_sample.data}

        self.nodes = []
        img_features = feature_extractor.get_img_features(self.xray,
                                                          to_numpy=True, 
                                                          mean_features=mean_features)
        for i, fixation in enumerate(reflacx_sample.get_fixations()):
            node = FixationNode.new_node(i,
                                         fixation,
                                         self.chest_bb,
                                         self.xray,
                                         feature_extractor=feature_extractor,
                                         img_features=img_features,
                                         stdevs=stdevs)
            if node is None:
                continue
            if node.features is None:
                self.log('bad features for fixation {}'.format(i))
                raise IndexError
            self.nodes.append(node)
        
        self.adj_mat = None
        self.calc_edge()


    def calc_edge(self, self_edges=True):
        """Classes inheriting GazeTrackingGraph need
        to fill a adjacency matrix (self.adj_mat).
        param:self_edges determine whether or not to add edges from a node to itself
        """
        pass

    
    def draw(self, out_dir= None, fname=None, color='#88cccc', edge_labels=True):
        """Draws the graph and saves it as a html file
        in param:out_dir under param:fname
        """
        g = Network(notebook=True, cdn_resources='remote')
        g.toggle_physics(False)

        for k, n in enumerate(self.nodes):
            g.add_node(k,
                    label=str(k),
                    title=str(k),
                    value=int(n.duration * 1000),
                    x=n.norm_x * 1000,
                    y=n.norm_y * 1000,
                    color=('orange'
                        if k == 0
                        else ('cyan' if k == len(self.nodes) - 1
                                else "#97c2fc")),
                    shape='circle')
        
        for i in range(len(self.adj_mat)):
            for j in range(i + 1, len(self.adj_mat)):
                if i == j or self.adj_mat[i][j] == 0:
                    continue
                g.add_edge(i,
                           j,
                           label=("{:.2f}".format(self.adj_mat[i][j])
                                  if edge_labels
                                  else ""),
                           color=color)
        if out_dir is None:
            out_dir = '.'
        
        if fname is None:
            fname = sep.join([out_dir, '{}.html'.format(self.name)])
        
        g.save_graph(fname)

    
    def write_nodes_csv(self, csv_file, makeline=lambda x: x):
        for node in self.nodes:
            csv_file.write(makeline(str(node)))

    
    def write_edges_csv(self, csv_file, makeline=lambda x: x):
        for i, node in enumerate(self.nodes):
            for j, node in enumerate(self.nodes):
                if self.adj_mat[i, j] != 0:
                    csv_file.write(makeline("{}, {}, {}".format(i, j, self.adj_mat[i, j])))

    
    def graph_csv(self, labels='common'):
        assert labels in ['common', 'phase_1', 'phase_2', 'phase_3']
        if labels == 'common':
            result = self.common_labels
        elif labels == 'phase_1':
            result = self.phase1_labels
        else:
            result = self.phase2_3_labels

        return str(list(result.values()))

    
    def __str__(self):
        ids = '{}  dicom-id: {}  reflacx-id: {}'.format(self.name,
                                                        self.dicom_id,
                                                        self.reflacx_id)
        
        ph1 = ('phase 1:\n' + '\n'.join(['{}: {}'.format(k, self.phase1_labels[k])
                                         for k in self.phase1_labels]))
        
        ph23 = ('phase 2/3:\n' + '\n'.join(['{}: {}'.format(k, self.phase2_3_labels[k])
                                            for k in self.phase2_3_labels]))
        
        return '\n'.join([ids, ph1, ph23])
