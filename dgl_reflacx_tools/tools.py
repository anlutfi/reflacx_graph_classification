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
    
    result = []
    offset = 0
    for xs, ys in zip(torch.split(batch_xs, tuple(g.batch_num_nodes())),
                      torch.split(batch_ys, tuple(g.batch_num_nodes()))):
        grid = []
        for i in range(gridsize):
            line = []
            grid.append(line)
            for j in range(gridsize):
                yis = torch.where(ys == j)[0]
                xis = torch.where(xs[yis] == i)
                nis = yis[xis] + offset
                line.append(g.subgraph(nis))
        result.append(grid)
        offset += len(xs)
    
    return result if len(result) > 1 else result[0]


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
    

