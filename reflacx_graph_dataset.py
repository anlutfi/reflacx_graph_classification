import numpy as np
import torch
from feature_extraction.dense_feature_extraction import DenseFeatureExtractor
from gaze_tracking_graph import GazeTrackingGraph
from rlogger import RLogger
import os
from fixation_node import FixationNode

def generate_dataset(name,
                     metadata,
                     outdir=None,
                     filenames={'meta': 'meta.yaml',
                                'edges': 'edges.csv',
                                'nodes': 'nodes.csv',
                                'graphs': 'graphs.csv',
                                'index': 'index.csv'
                                },
                     g_id = 'graph_id',
                     sep=', ',
                     graph_class=GazeTrackingGraph,
                     stdevs=1,
                     feature_extractor=DenseFeatureExtractor(),
                     mean_normalize_features=True,
                     mean_features_fpath=None,
                     log_dir='.'):
    log = RLogger(__name__)
    
    outdir = './{}'.format(name) if outdir is None else outdir
    os.makedirs(outdir, exist_ok=True)

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

    with open(os.sep.join([outdir, filenames['meta']]), 'w') as f:
        f.writelines(['dataset_name: {}'.format(name),
                        '\nedge_data:',
                        '\n- file_name: {}'.format(filenames['edges']),
                        '\nnode_data:',
                        '\n- file_name: {}'.format(filenames['nodes']),
                        '\ngraph_data:',
                        '\nfile_name: {}'.format(filenames['graphs'])])

    e_csv = open(os.sep.join([outdir, filenames['edges']]), 'w')
    n_csv = open(os.sep.join([outdir, filenames['nodes']]), 'w')
    g_csv = open(os.sep.join([outdir, filenames['graphs']]), 'w') 
    i_csv = open(os.sep.join([outdir, filenames['index']]), 'w') 
           
    
    csv_line = lambda prefix, line: sep.join([str(prefix), line]) + '\n'
    csv_header = lambda line: csv_line(g_id, line)
    
    n_csv.write(csv_header(FixationNode.csv_header()))
    e_csv.write(csv_header(graph_class.edge_csv_header()))
    g_csv.write(csv_header('labels'))
    i_csv.write(csv_header(sep.join(['dicom_id', 'reflacx_id'])))
    
    
    i = 0
    dicom_ids = metadata.list_dicom_ids()
    size = len(dicom_ids)
    last_percent = 0
    for dicom_id in dicom_ids:
        for reflacx_id in metadata.list_reflacx_ids(dicom_id):
            os.makedirs(log_dir)
            RLogger.start(os.path.sep.join([log_dir,
                                    '{}__{}.log'.format(dicom_id, reflacx_id)]))
            try:
                curr_line = lambda line: csv_line(i, line)
                g = graph_class(dicom_id,
                            reflacx_id,
                            reflacx_sample=metadata.get_sample(dicom_id, reflacx_id),
                            metadata=metadata,
                            stdevs=stdevs,
                            feature_extractor=feature_extractor,
                            mean_features=mean_features)
                g_csv.write(curr_line(g.graph_csv(labels='common')))
                i_csv.write(curr_line(sep.join([dicom_id, reflacx_id])))
                g.write_nodes_csv(n_csv, curr_line)
                g.write_edges_csv(e_csv, curr_line)
            except:
                log('bad graph for pair {} --- {}'.format(dicom_id,
                                                            reflacx_id),
                    exception=True)
                continue
            percent = int((i / size) * 100)
            if percent > last_percent:
                print('{} \% of dicom ids'.format(percent))
                last_percent = percent
            i += 1

    n_csv.close()
    e_csv.close()
    g_csv.close()

        