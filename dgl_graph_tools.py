import dgl
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