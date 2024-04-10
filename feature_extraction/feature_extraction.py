import torchxrayvision as xrv
import numpy as np
import torch


class FeatureExtractor:
    def __init__(self, model_url):
        self.model = xrv.models.DenseNet(weights=model_url)

    def transform_img(self, img, to_numpy):
        pass

    def transform_fixation(self, fixation_pos, ang_x, ang_y, stdevs, xray_size):
        pass

    def get_img_features(self, img, to_numpy):
        pass

    def get_fixation_features(self, fixation):
        pass

    def get_all_fixations_features(self, reflacx_sample):
        pass

    def get_reflacx_img_features(self, reflacx_sample, to_numpy=False):
        return self.get_img_features(reflacx_sample.get_dicom_img(),
                                     to_numpy=to_numpy)
    
    def get_reflacx_avg_features(self, reflacx_meta, to_numpy=False):
        all_features = []
        dicom_ids = reflacx_meta.list_dicom_ids()
        for i, dicom_id in enumerate(dicom_ids):
            sample = reflacx_meta.get_sample(dicom_id, reflacx_meta.list_reflacx_ids(dicom_id)[0])
            try:
                img = sample.get_dicom_img()
            except ValueError:
                continue
            all_features.append(self.get_img_features(img, True))
        result = np.average(np.array(all_features), axis=0)
        if to_numpy:
            return result
        return torch.from_numpy(result)

    def __call__(self, img, to_numpy=False):
        result = self.get_img_features(img)
        if to_numpy:
            return result.detach().numpy()
        return result