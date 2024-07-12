import torch
from dgl import mean_nodes

def gridify_indices(g,
                    gridsize,
                    x_nm='norm_x',
                    y_nm='norm_y'):
    batch_xs = (g.ndata[x_nm] * gridsize).int()
    batch_ys = (g.ndata[y_nm] * gridsize).int()
    batch_xs[torch.where(batch_xs > gridsize - 1)] = gridsize - 1
    batch_ys[torch.where(batch_ys > gridsize - 1)] = gridsize - 1
    
    result = []
    for i in range(gridsize):
        line = []
        result.append(line)
        for j in range(gridsize):
            yis = torch.where(batch_ys == j)[0]
            xis = torch.where(batch_xs[yis] == i)
            nis = yis[xis]
            line.append(nis)
    
    return result


def gridify_by_indices(g,
                       indices):
    n_tail = torch.cumsum(g.batch_num_nodes(), dim=0) - 1
    n_head = torch.cat([torch.tensor([0]).to(n_tail.device), n_tail[:-1] + 1])
    
    result = []
    for i, i_line in enumerate(indices):
        line = []
        result.append(line)
        for j, nis in enumerate(i_line):

            sg = g.subgraph(nis)
            #preserving NODE batch info for new subgraph
            nis = nis.unsqueeze(1).tile((1, len(n_tail)))
            mask = (nis >= n_head) & (nis <= n_tail)
            bnn = torch.count_nonzero(mask, dim=0)
            sg.set_batch_num_nodes(bnn)

            #preserving EDGE batch info for new subgraph
            e_tail = torch.cumsum(bnn, dim=0) - 1
            e_head = torch.cat([torch.tensor([0]).to(e_tail.device), e_tail[:-1] + 1])
            source, dest = sg.edges()
            source = source.unsqueeze(1).tile((1, len(e_tail)))
            dest = dest.unsqueeze(1).tile((1, len(e_tail)))
            mask = (source >= e_head) & (source <= e_tail) & (dest >= e_head) & (dest <= e_tail)
            bne = torch.count_nonzero(mask, dim=0)
            sg.set_batch_num_edges(bne)

            line.append(sg)
    
    return result


def gridify(g,
            gridsize,
            x_nm='norm_x',
            y_nm='norm_y'):
    indices = gridify_indices(g, gridsize, x_nm, y_nm)
    return indices, gridify_by_indices(g, indices)
    

def grid_readout(grid,
                 name,
                 aggr=mean_nodes,
                 replace_nan=True):
    result = None
    for line in grid:
        result_line = None
        for sg in line:
            readout = aggr(sg, name)
            # if a subgraph for a grid cell is empty (0 nodes)
            if replace_nan and torch.any(readout.isnan()):
                readout = list(torch.split(readout, 1))
                for i, sg_readout in enumerate(readout):
                    if torch.all(sg_readout.isnan()):
                        readout[i] = torch.zeros_like(sg_readout)
                readout = torch.cat(readout)
            
            readout = readout.unsqueeze(1)
            if result_line is None:
                result_line = readout
            else:
                result_line = torch.cat((result_line, readout), 1)
        
        result_line = result_line.unsqueeze(1)
        if result is None:
            result = result_line
        else:
            result = torch.cat((result, result_line), 1)

    return result
    

class Readout:
    def __init__(self, feats_and_aggrs, replace_nan=True):
        self.replace_nan = replace_nan
        self.readouts = []
        for feat_nm, aggr in feats_and_aggrs:
            self.readouts.append(lambda grid,
                                        f=feat_nm,
                                        a=aggr,
                                        r=replace_nan: grid_readout(grid, f, a, r))
            
    def __call__(self, grid, flatten=True):
        result = None
        for readout in self.readouts:
            ro = readout(grid)
            if ro.dim() < 4:
                ro = ro.unsqueeze(-1)
            if result is None:
                result = ro
            else:
                result = torch.cat([result, ro], dim=-1)
        return result if not flatten else result.flatten(1, -1)