from feature_extraction.dense_feature_extraction import DenseFeatureExtractor
from fixation_node import FixationNode
from pyvis.network import Network
from os.path import sep

class GazeTrackingGraph:
    def calc_edge(self):
        pass
    
    def __init__(self,
                 dicom_id,
                 reflacx_id,
                 reflacx_sample=None,
                 metadata=None,
                 stdevs=1,
                 feature_extractor=DenseFeatureExtractor(),
                 mean_features=None):
        assert reflacx_sample is not None or metadata is not None
        if reflacx_sample is None:
            reflacx_sample = metadata.get_sample(dicom_id, reflacx_id)
        
        self.dicom_id = dicom_id
        self.reflacx_id = reflacx_id
        
        self.name = 'GazeTrackingGraph_{}_{}'.format(self.dicom_id, self.reflacx_id)
        
        self.xray = reflacx_sample.get_dicom_img()
        self.chest_bb = reflacx_sample.get_chest_bounding_box()

        self.phase1_labels = {k: reflacx_sample.data[k] 
                              for k in ["Airway wall thickening",
                                        "Atelectasis",
                                        "Consolidation",
                                        "Emphysema",
                                        "Enlarged cardiac silhouette",
                                        "Fibrosis",
                                        "Fracture",
                                        "Groundglass opacity",
                                        "Mass",
                                        "Nodule",
                                        "Pleural effusion",
                                        "Pleural thickening",
                                        "Pneumothorax",
                                        "Pulmonary edema",
                                        "Wide mediastinum"]
                              if k in reflacx_sample.data}
        
        self.phase2_3_labels = {k: reflacx_sample.data[k] 
                                for k in ["Abnormal mediastinal contour",
                                          "Acute fracture",
                                          "Atelectasis",
                                          "Consolidation",
                                          "Enlarged cardiac silhouette",
                                          "Enlarged hilum",
                                          "Groundglass opacity",
                                          "Hiatal hernia",
                                          "High lung volume / emphysema, Interstitial lung disease",
                                          "Lung nodule or mass",
                                          "Pleural abnormality",
                                          "Pneumothorax",
                                          "Pulmonary edema"]
                                if k in reflacx_sample.data}
        
        #TODO add common labels for simpler test case with most datapoints

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
            if node is not None:
                self.nodes.append(node)
            else:
                #print('Void Node at {}'.format(i))
                pass #TODO add in logging
        
        self.calc_edge()


    def draw(self, out_dir= None, fname=None, color='#88cccc', edge_labels=True):
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

    
    def get_nodes_csv(self, csv_path):
        #TODO replace with pandas
        with open(csv_path, 'w') as f:
            f.write(self.nodes[0].get_csv_header())
            for node in self.nodes:
                f.write(str(node))

    def get_edges_csv(self, csv_path):
        #TODO replace with pandas
        with open(csv_path, 'w') as f:
            f.write("src, dst, w")
            for i, node in enumerate(self.nodes):
                for j, node in enumerate(self.nodes):
                    if self.adj_mat[i, j] != 0:
                        f.write("{}, {}, {}".format(i, j, self.adj_mat[i, j]))

    def get_graph_csv(self):
        pass #TODO

    def __str__(self):
        ids = '{}  dicom-id: {}  reflacx-id: {}'.format(self.name,
                                                        self.dicom_id,
                                                        self.reflacx_id)
        
        ph1 = ('phase 1:\n' + '\n'.join(['{}: {}'.format(k, self.phase1_labels[k])
                                         for k in self.phase1_labels]))
        
        ph23 = ('phase 2/3:\n' + '\n'.join(['{}: {}'.format(k, self.phase2_3_labels[k])
                                            for k in self.phase2_3_labels]))
        
        return '\n'.join([ids, ph1, ph23])
