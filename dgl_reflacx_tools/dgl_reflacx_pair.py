
import cv2
import numpy as np
from dgl_reflacx_tools.tools import get_edge

class GraphPair:
    def __init__(self, reflacx_graph, dgl_graph):
        self.reflacx_graph = reflacx_graph
        self.dgl_graph, self.dgl_labels = dgl_graph

    def draw_dgl(self):
        pass

    def draw_reflacx(self):
        pass

    def dgl_ious(self, field='weight'): # TODO review field
        result = {}
        edge = lambda i, j: get_edge(self.dgl_graph, i, j, field=field)
        for i in (int(i) for i in self.dgl_graph.nodes()):
            for j in (int(j) for j in self.dgl_graph.nodes()):
                result[(i, j)] = np.float32(edge(i, j))

        return result

    def reflacx_ious(self, canvas_sz=500):
        def get_coords(node):
            tlx, tly = node.topleft
            brx, bry = node.bottomright
            tlx = int(tlx * canvas_sz)
            tly = int(tly * canvas_sz)
            brx = int(brx * canvas_sz)
            bry = int(bry * canvas_sz)
            return tlx, tly, brx, bry
        
        def get_mask(tlx, tly, brx, bry):
            mask = np.zeros((canvas_sz, canvas_sz))
            mask = cv2.rectangle(mask, (tlx, tly), (brx, bry), 255, -1)
            mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
            return mask
        
        result = {}
        g = self.reflacx_graph
        for i, node_i in enumerate(g.nodes):
            tlx_i, tly_i, brx_i, bry_i = get_coords(node_i)
            for j, node_j in enumerate(g.nodes):
                if i == j:
                    result[(i, j)] = 1.0
                    continue
                tlx_j, tly_j, brx_j, bry_j = get_coords(node_j)
                imask = get_mask(tlx_i, tly_i, brx_i, bry_i)
                umask = np.copy(imask)
                jmask = get_mask(tlx_j, tly_j, brx_j, bry_j)

                imask[imask != jmask] = 0
                umask[umask != jmask] = 255
                inter = np.count_nonzero(imask)
                union = np.count_nonzero(umask)
                
                result[(i, j)] = inter / union
        
        return result
                

    def get_ious(self):
        return {'dgl': self.dgl_iou(), 'reflacx': self.reflacx_iou()}