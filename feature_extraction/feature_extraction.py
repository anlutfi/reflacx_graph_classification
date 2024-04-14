from rlogger import RLogger
import torchxrayvision as xrv
import numpy as np
import torch


class FeatureExtractor:
    """Class for extracting image/fixation features from DICOM chest xray images.
    Images features, in this case, are the result of some convolutional layer of a model,
    before the last flattened tensor. The ideia is to maintain 2d image information,
    so that the features of a specific gaze fixation can be a crop of these image features.
    """
    def __init__(self):
        self.model = None

        self.log = RLogger(__name__, self.__class__.__name__)
        

    def transform_img(self, img, to_numpy):
        """Transforms an image into the input format expected by self.model
        Returns an np.array if param:to_numpy is true and a torch.tensor otherwise
        """
        pass


    def transform_fixation(self, fixation_pos, ang_x, ang_y, stdevs, xray_size):
        """Transforms a REFLACX fixation into its correspondent in a transformed image (self.transform_img()).
        Returns a tuple (transformed position, transformed crop)
        transformed crop is the region of the image observed by a fixation,
        a gaussian bell with the fixation's position as the mean (center point), extending for a number of standard deviations(param:stdevs). param:ang_x and param:ang_y are how many pixels per degree of vision are observed. 1 standard deviation = 1 degree.
        """
        pass


    def get_img_features(self, img, to_numpy):
        """Feeds param:img to self.model and returns its features.
        Returns an np.array if param:to_numpy is true and a torch.tensor otherwise
        """
        pass


    def get_fixation_features(self, fixation, img_features, to_numpy):
        """Gets fixation features based on fixation's position relative to
        param:img_features
        Returns an np.array if param:to_numpy is true and a torch.tensor otherwise
        """
        pass


    def get_all_fixations_features(self, reflacx_sample):
        """Returns a list of a param:reflacx_sample's features for all fixations
        """
        pass


    def get_reflacx_img_features(self,
                                 reflacx_sample,
                                 to_numpy=False,
                                 mean_features=None):
        """Runs self.get_img_features for a given param:reflacx_sample.
        Just a wrapper that loads the image from the sample internally
        param:mean_features is a tensor with the average features across all REFLACX images
            is it is not None, the resulting features will be subtracted from this mean (mean normalization)
        Returns an np.array if param:to_numpy is true and a torch.tensor otherwise
        """
        return self.get_img_features(reflacx_sample.get_dicom_img(),
                                     to_numpy=to_numpy,
                                     mean_features=mean_features)
    

    def get_reflacx_avg_features(self, reflacx_meta, fname=None, to_numpy=False):
        """Computes features for all of REFLACX images and returns their average.
        The resulting tensor can be used to perform mean normalization. Specifically, the result of this method can be passed as the mean_features argument of this class' other methods.
        param:reflacx_meta is a Metadata object containing REFLACX data.
        Returns an np.array if param:to_numpy is true and a torch.tensor otherwise
        If param:fname is set to a string, saves the tensor in a format consistent
            with param:to_numpy.
        """
        all_features = []
        dicom_ids = reflacx_meta.list_dicom_ids()
        for dicom_id in dicom_ids:
            sample = reflacx_meta.get_sample(dicom_id, reflacx_meta.list_reflacx_ids(dicom_id)[0])
            try:
                img = sample.get_dicom_img()
            except ValueError:
                continue
            all_features.append(self.get_img_features(img, True))
        result = np.average(np.array(all_features), axis=0)
        if to_numpy:
            if fname is not None:
                np.save(fname, result)
            return result
        
        result = torch.from_numpy(result)
        if fname is not None:
            torch.save(result, fname)
        return result

    def __call__(self, img, to_numpy=False):
        result = self.get_img_features(img)
        return result.detach().numpy() if to_numpy else result