import torchxrayvision as xrv


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

    def __call__(self, img, to_numpy=False):
        result = self.get_img_features(img)
        if to_numpy:
            return result.detach().numpy()
        return result