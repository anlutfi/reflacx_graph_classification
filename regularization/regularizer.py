import torch

class Regularizer:
    def __init__(self, batch, feat_nm, nodes_or_edges='nodes'):
        assert nodes_or_edges in ('nodes', 'edges')
        data = (batch.ndata[feat_nm]
                if nodes_or_edges == 'nodes'
                else batch.ndata[feat_nm])
        self.fmin = torch.min(data, dim=0).values
        fmax = torch.max(data, dim=0).values
        
        self.finterval = fmax - self.fmin
        self.feat_nm = feat_nm
        self.nodes = nodes_or_edges == 'nodes'
    
    def __call__(self, g):
        data = g.ndata[self.feat_nm] if self.nodes else g.edata[self.feat_nm]
        return (data - self.fmin) / self.finterval