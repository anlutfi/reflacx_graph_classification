import dgl

from regularization.regularizer import Regularizer


class RegularizationPipeline:
    def __init__(self, dataset, device, node_feat_nms=None, edge_feat_nms=None):
        assert node_feat_nms is not None or edge_feat_nms is not None
        
        all_graphs = dgl.batch([g for g, _ in dataset]).to(device)
        
        self.node_regularizers = {}
        if node_feat_nms is not None:
            for feat_nm in node_feat_nms:
                self.node_regularizers[feat_nm] = Regularizer(all_graphs,
                                                              feat_nm,
                                                              'nodes')
        
        self.edge_regularizers = {}
        if edge_feat_nms is not None:
            for feat_nm in edge_feat_nms:
                self.edge_regularizers[feat_nm] = Regularizer(all_graphs,
                                                              feat_nm,
                                                              'edges')


    def __call__(self, g):
        for k in self.node_regularizers:
            g.ndata[k] = self.node_regularizers[k](g)
        for k in self.edge_regularizers:
            g.edata[k] = self.edge_regularizers[k](g)