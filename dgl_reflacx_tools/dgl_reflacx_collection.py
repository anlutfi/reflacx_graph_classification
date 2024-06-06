import dgl
import os
import numpy as np
from feature_extraction.dense_feature_extraction import DenseFeatureExtractor
from dgl_reflacx_tools.dgl_reflacx_pair import GraphPair

from metadata import Metadata

class GraphCollection:
    def __init__(self,
                 dgl_dataset_dir,
                 graph_class,
                 extractor=DenseFeatureExtractor(),
                 avg_fpath='./avg_DensNet_REFLACX_features.npy',
                 reflacx_dir="../data/reflacx",
                 mimic_dir="../data/mimic/reflacx_imgs",
                 full_meta_path="../reflacx_lib/full_meta.json"):
        self.reflacx = Metadata(reflacx_dir,
                                mimic_dir,
                                full_meta_path,
                                max_dicom_lib_ram_percent=20)
        self.dataset = dgl.data.CSVDataset(dgl_dataset_dir)
        self.graph_class = graph_class
        self.extractor = extractor
        self.index = {}
        with open(os.sep.join([dgl_dataset_dir, 'index.csv'])) as f:
            for line in (l.strip() for l in f.readlines()[1:]):
                i, did, rid = line.split(',')
                self.index[int(i)] = (did, rid)

        self.mean_feats = np.load(avg_fpath) if avg_fpath is not None else None


    def _fetch(self, i, sample):
        features = self.extractor.get_reflacx_img_features(sample, to_numpy=True)
        reflacx_g = self.graph_class(sample.dicom_id,
                                     sample.reflacx_id,
                                     sample,
                                     metadata=self.reflacx,
                                     stdevs=1,
                                     feature_extractor=self.extractor,
                                     img_features=features,
                                     mean_features=self.mean_feats)
        
        return GraphPair(reflacx_g, self.dataset[i])

    
    def fetch_by_dgl_index(self, i):
        assert i in self.index
        did, rid = self.index[i]
        sample = self.reflacx.get_sample(did, rid)
        return self._fetch(i, sample)
    
    
    def fetch_by_reflacx(self, did=None, rid=None, sample=None):
        assert sample is not None or (did is not None and rid is not None)
        if sample is None:
            sample = self.reflacx.get_sample(did, rid)
        reverse_index = {v: k for (k, v) in self.index.items()}
        assert (did, rid) in reverse_index
        i = reverse_index[(did, rid)]
        return self._fetch(i, sample)