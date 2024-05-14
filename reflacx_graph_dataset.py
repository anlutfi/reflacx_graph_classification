import numpy as np
import torch
from feature_extraction.dense_feature_extraction import DenseFeatureExtractor
from gaze_tracking_graph import GazeTrackingGraph
from rlogger import RLogger
import os
from fixation_node import FixationNode

from consts import CSV_SEP

def generate_csv_dataset(name,
                         metadata,
                         outdir=None,
                         filenames={'meta': 'meta.yaml',
                                    'edges': 'edges.csv',
                                    'nodes': 'nodes.csv',
                                    'graphs': 'graphs.csv',
                                    'index': 'index.csv'
                                    },
                         g_id = 'graph_id',
                         sep=CSV_SEP,
                         graph_class=GazeTrackingGraph,
                         stdevs=1,
                         feature_extractor=DenseFeatureExtractor(),
                         mean_normalize_features=True,
                         mean_features_fpath=None,
                         log_dir='.'):
    log = RLogger(__name__)
    
    outdir = './{}'.format(name) if outdir is None else outdir
    os.makedirs(outdir, exist_ok=True)

    os.makedirs(log_dir, exist_ok=True)
    RLogger.start(os.path.sep.join([log_dir, '{}.log'.format(name)]))

    mean_features = None
    if mean_normalize_features:
        if mean_features_fpath is None:
            mean_features = feature_extractor.get_reflacx_avg_features(metadata)
        elif not os.path.exists(mean_features_fpath):
            mean_features = feature_extractor.get_reflacx_avg_features(metadata,
                                                                        fname=mean_features_fpath)
        elif mean_features_fpath.split('.')[-1] == 'pt':
            mean_features = torch.load(mean_features_fpath)
        else: # .npy
            mean_features = torch.from_numpy(np.load(mean_features_fpath))

    edge_types = graph_class.get_edge_type_names()
    prefix, ext = filenames['edges'].split('.')
    edge_csv_lines = []
    edge_csv_paths = {}
    for i, edge_type in enumerate(edge_types):
        edge_csv_paths[edge_type] = '{}_{}.{}'.format(prefix, i, ext)
        edge_csv_lines.append('\n- file_name: {}'.format(edge_csv_paths[edge_type]))
        edge_csv_lines.append('\n  etype: [fixation, {}, fixation]'.format(edge_type))
        
    with open(os.sep.join([outdir, filenames['meta']]), 'w') as f:
        f.writelines(['dataset_name: {}'.format(name),
                      '\nedge_data:']
                      + edge_csv_lines
                      + ['\nnode_data:',
                         '\n- file_name: {}'.format(filenames['nodes']),
                         '\n  ntype: fixation',
                         '\ngraph_data:',
                         '\n  file_name: {}'.format(filenames['graphs'])])

    e_csvs = {e_type: open(os.sep.join([outdir, edge_csv_paths[e_type]]), 'w')
              for e_type in edge_csv_paths}
    #e_csv = open(os.sep.join([outdir, filenames['edges']]), 'w')
    n_csv = open(os.sep.join([outdir, filenames['nodes']]), 'w')
    g_csv = open(os.sep.join([outdir, filenames['graphs']]), 'w') 
    i_csv = open(os.sep.join([outdir, filenames['index']]), 'w') 
           
    
    csv_line = lambda prefix, line: sep.join([str(prefix), line]) + '\n'
    csv_header = lambda line: csv_line(g_id, line)
    
    n_csv.write(csv_header(FixationNode.csv_header()))
    #e_csv.write(csv_header(graph_class.edge_csv_header()))
    for e_type in e_csvs:
        e_csvs[e_type].write(csv_header(graph_class.edge_csv_header()))
    g_csv.write(csv_header('labels'))
    i_csv.write(csv_header(sep.join(['dicom_id', 'reflacx_id'])))
    
    i = 0
    dicom_ids = metadata.list_dicom_ids()
    d_size = len(dicom_ids)
    for di, dicom_id in enumerate(dicom_ids):
        xray = metadata.get_dicom_img(dicom_id)
        if xray is None:
            log('missing dicom img for if {}'.format(dicom_id),
                exception=True)
            continue
        img_features = feature_extractor.get_img_features(xray,
                                                          to_numpy=True,
                                                          mean_features=mean_features)
        reflacx_ids = metadata.list_reflacx_ids(dicom_id)
        for ri, reflacx_id in enumerate(reflacx_ids):
            r_size = len(reflacx_ids)
            print('dicom_id {} of {}  ---  reflacx_id {} of {}'.format(di + 1,
                                                                       d_size,
                                                                       ri + 1,
                                                                       r_size),
                  end='\r')
            try:
                curr_line = lambda line: csv_line(i, line)
                g = graph_class(dicom_id,
                                reflacx_id,
                                reflacx_sample=metadata.get_sample(dicom_id,
                                                                   reflacx_id),
                                metadata=metadata,
                                stdevs=stdevs,
                                feature_extractor=feature_extractor,
                                img_features=img_features,
                                mean_features=mean_features)
                g_csv.write(curr_line(g.graph_csv(labels='common')))
                i_csv.write(curr_line(sep.join([dicom_id, reflacx_id])))
                g.write_nodes_csv(n_csv, curr_line)
                g.write_edges_csv(e_csvs, curr_line)
            except (IndexError, AttributeError):
                log('bad graph for pair {} --- {}'.format(dicom_id,
                                                          reflacx_id),
                    exception=True)
                i += 1
                continue
            i += 1
   
    n_csv.close()
    for e_csv in e_csvs.values():
        e_csv.close()
    g_csv.close()

        