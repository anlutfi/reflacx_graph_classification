from feature_extraction.dense_feature_extraction import DenseFeatureExtractor
from fixation_node import FixationNode

class ScanPathGraph:
    def __init__(self,
                 calc_edge,
                 dicom_id,
                 reflacx_id,
                 reflacx_sample=None,
                 metadata=None,
                 std_devs=1,
                 feature_extractor=DenseFeatureExtractor(),
                 mean_features=None):
        assert reflacx_sample is not None or metadata is not None
        if reflacx_sample is None:
            reflacx_sample = metadata.get_sample(dicom_id, reflacx_id)
        
        self.dicom_id = dicom_id
        self.reflacx_id = reflacx_id
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
                                                          mean_features=mean_features)
        for i, fixation in enumerate(reflacx_sample.get_fixations()):
            node = FixationNode.new_node(i,
                                         fixation,
                                         self.chest_bb,
                                         self.xray,
                                         feature_extractor=feature_extractor,
                                         img_features=img_features,
                                         std_devs=std_devs)
            if node is not None:
                self.nodes.append(node)
        
        self.adj_mat = calc_edge(self.nodes)


    def draw(self, fpath='./graph.html', color='#88cccc', edge_labels=True):
        pass #TODO

    
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