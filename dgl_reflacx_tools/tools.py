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
    node_f = lambda i: get_node(g, i)
    
    grid = [[[] for j in range(gridsize)] for i in range(gridsize)]
    for i in (int(i) for i in g.nodes()):
        n = node_f(i)
        x = min(int(n[x_nm] * gridsize), gridsize - 1)
        y = min(int(n[y_nm] * gridsize), gridsize - 1)
        grid[x][y].append(i)

    return [[g.subgraph(grid[i][j]) for j in range(len(grid[0]))]
            for i in range(len(grid))]


def grid_readout(grid, name, aggr=dgl.mean_nodes): # TODO preserve grads
    sz = len(grid)
    node = None
    for line in grid:
        for sg in line:
            try:
                node = get_node(sg, 0)
                break
            except AssertionError:
                continue
    shape = (sz, sz) + node[name].shape
    result = torch.zeros(shape)

    result = None
    for i, line in enumerate(grid):
        result_line = None
        for j, sg in enumerate(line):
            readout = aggr(sg, name).unsqueeze(0)
            if result_line is None:
                result_line = readout.clone()
            else:
                result_line = torch.cat((result_line, readout), 0)
        if result is None:
            result = result_line.clone()#.unsqueeze...
        else:
            pass # TODO concatenate lines


#            if len(shape) == 2:
#                result[i, j] = aggr(sg, name)
#            else: # 3D
#                result[i, j, :] = aggr(sg, name)

    return result
    

