import dgl
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def get_node(graph, i):
    assert 0 <= i < len(graph.nodes())
    result = {}
    for k in graph.ndata.keys():
        result[k] = graph.ndata[k][i]
    return result


def get_edge(graph, i, j, non_edge_value=0.0, field='weight'): # TODO review field name str
    try:
        e_i = graph.edge_ids(i, j)
    except dgl.DGLError: # inexistent edge in sparse adj matrix
        return non_edge_value
    return graph.edata[field][e_i]


def draw(graph,
         options={'node_color': 'black',
                  'node_size': 20,
                  'width': 1},
         figsize=[15, 7]):
    g = dgl.to_networkx(graph.cpu())
    plt.figure(figsize=figsize)
    nx.draw(g, **options)


def get_connected_component(adj, start=0):
    result = []
    q = [start]
    while len(q) > 0:
        n = q.pop(0)
        result.append(n)
        adj[:, n] = 0
        q = q + list(np.where(adj[n] != 0)[0])
    return set(result)


def gridify(g,
            gridsize,
            x_nm='norm_x',
            y_nm='norm_y'):
    batch_xs = (g.ndata[x_nm] * gridsize).int()
    batch_ys = (g.ndata[y_nm] * gridsize).int()
    batch_xs[torch.where(batch_xs > gridsize - 1)] = gridsize - 1
    batch_ys[torch.where(batch_ys > gridsize - 1)] = gridsize - 1
    n_tail = torch.cumsum(g.batch_num_nodes(), dim=0) - 1
    n_head = torch.cat([torch.tensor([0]).to(n_tail.device), n_tail[:-1] + 1])
    
    result = []
    for i in range(gridsize):
        line = []
        result.append(line)
        for j in range(gridsize):
            yis = torch.where(batch_ys == j)[0]
            xis = torch.where(batch_xs[yis] == i)
            nis = yis[xis]
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


def grid_readout(grid_or_batch,
                 name,
                 aggr=dgl.mean_nodes,
                 replace_nan=True,
                 leaf_class_name='DGLGraph'):
    def call(grid):
        result = None
        for line in grid:
            result_line = None
            for sg in line:
                readout = aggr(sg, name)
                # if a subgraph for a grid cell is empty (0 nodes)
                if replace_nan and torch.all(readout.isnan()):
                    readout = torch.zeros_like(readout)
                
                if result_line is None:
                    result_line = readout
                else:
                    result_line = torch.cat((result_line,readout), 0)
            
            result_line = result_line.unsqueeze(0)
            if result is None:
                result = result_line
            else:
                result = torch.cat((result, result_line), 0)

        return result
    
    dims = 0
    g = grid_or_batch
    while g.__class__.__name__ != leaf_class_name:
        g = g[0]
        dims += 1

    if dims == 2: # is a single grid
        return call(grid_or_batch)
    else: # dims == 3 -> is a grid batch
        readouts = [call(grid) for grid in grid_or_batch]
        return torch.cat([r.unsqueeze(0) for r in readouts], 0)
    

