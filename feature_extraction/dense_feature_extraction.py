import torchxrayvision as xrv
import numpy as np
import cv2
import torch
from math import ceil, floor

from feature_extraction.feature_extraction import FeatureExtractor

class DenseFeatureExtractor(FeatureExtractor):
    """Class for extracting image/fixation features from DICOM chest xray images
    using DensNet.
    This class is only documented to the extent it differs from its superclass.
    Check FeatureExtractor documentation also
    """
    def __init__(self,
                 model_url = "densenet121-res224-mimic_ch",
                 resize=224):
        """Loads DenseNet weights specified by param:model_url
        param:resize is the image dimensions expected by the model
        """
        super().__init__()
        self.model = xrv.models.DenseNet(weights=model_url)
        self.resize = resize
        self.last_img_size = None


    def transform_img(self, img, to_numpy=False):
        if self.last_img_size != img.shape:
            self.last_img_size = img.shape
        
        img = xrv.datasets.normalize(img, np.max(img))
        
        # crop center
        y, x = img.shape
        crop_size = np.min([y, x])
        startx = x // 2 - (crop_size // 2)
        starty = y // 2 - (crop_size // 2)
        img = img[starty:starty + crop_size, startx:startx + crop_size]

        # resize
        result = cv2.resize(img,
                            (self.resize, self.resize),
                            interpolation=cv2.INTER_AREA
                           ).reshape(1, self.resize, self.resize).astype(np.float32)
        
        return result if to_numpy else torch.from_numpy(result)
        
    
    def transform_fixation(self,
                           fixation_pos,
                           ang_x,
                           ang_y,
                           stdevs=1,
                           img_size=None,
                           normalize=False):
        """Calculates a fixation's position in a transformed image.
        if an original image size (param:image_size) is not provided,
        it uses the last processed image's size instead, if available. Throws an assertion error otherwise.
        if param:normalize is True, returns position in a ([0, 1], [0, 1]) space
        """
        assert img_size is not None or self.last_img_size is not None
        if img_size is None:
            y, x = self.last_img_size
        else:
            y, x = img_size

        crop_size = np.min([y, x])
        startx = x // 2 - (crop_size // 2)
        starty = y // 2 - (crop_size // 2)

        # if fixation out of bounds, adjust to border
        fixation_pos = (max(0, fixation_pos[0] - startx),
                        max(0, fixation_pos[1] - starty))
        
        if fixation_pos[0] < 0 or fixation_pos[1] < 0:
            return None
        
        fixation_crop = [(max(0, fixation_pos[0] - stdevs * ang_x),
                          max(0, fixation_pos[1] - stdevs * ang_y)), 
                         (min(x, fixation_pos[0] + stdevs * ang_x),
                          min(y, fixation_pos[1] + stdevs * ang_y))]
        
        fixation_pos = (fixation_pos[0] / crop_size, fixation_pos[1] / crop_size)
        
        fixation_crop[0] = (fixation_crop[0][0] / crop_size,
                            fixation_crop[0][1] / crop_size)
        fixation_crop[1] = (fixation_crop[1][0] / crop_size,
                            fixation_crop[1][1] / crop_size)
        
        if not normalize:
            fixation_pos = (int(fixation_pos[0] * self.resize),
                            int(fixation_pos[1] * self.resize))
            
            fixation_crop[0] = (int(fixation_crop[0][0] * self.resize),
                                int(fixation_crop[0][1] * self.resize))
            fixation_crop[1] = (int(fixation_crop[1][0] * self.resize),
                                int(fixation_crop[1][1] * self.resize))
        
        fixation_crop = tuple(fixation_crop)
        return fixation_pos, fixation_crop
    
    
    def get_img_features(self, img, to_numpy=False, mean_features=None):
        img = self.transform_img(img)
        result = self.model.features(img[None,...])[0]
        if mean_features is not None:
            result -= mean_features
        return result.detach().numpy() if to_numpy else result
    

    def get_fixation_features(self,
                              fixation_pos,
                              ang_x,
                              ang_y,
                              img_features=None,
                              img=None,
                              stdevs=1,
                              img_size=None,
                              mean_features=None,
                              to_numpy=False):
        assert img_features is not None or img is not None
        if img_features is None:
            img_features = self.get_img_features(img,
                                                 to_numpy=True,
                                                 mean_features=mean_features)
        
        trans_fix = self.transform_fixation(fixation_pos,
                                            ang_x,
                                            ang_y,
                                            stdevs=stdevs,
                                            img_size=img_size,
                                            normalize=True)
        
        if trans_fix is None:
            return None
        
        fixation_pos, fixation_crop = trans_fix

        tl = fixation_crop[0]
        br = fixation_crop[1]
        tr = (br[0], tl[1])
        bl = (tl[0], br[1])

        adjustpos = lambda point:(point[0] * img_features.shape[2],
                                  point[1] * img_features.shape[1])
        
        tl = adjustpos(tl)
        bl = adjustpos(bl)
        tr = adjustpos(tr)
        br = adjustpos(br)

        h_region_count = ceil(tr[0]) - floor(tl[0])
        v_region_count = ceil(bl[1]) - floor(tl[1])

        crop_area = (tr[0] - tl[0]) * (bl[1] - tl[1])

        # calculate intersection between fixation crop
        # and each of the feature regions
        result = np.zeros(img_features.shape[0], dtype=img_features.dtype)

        for i in range(floor(tl[0]), ceil(tr[0])):
            for j in range(floor(tl[1]), ceil(bl[1])):
                xmin = max(tl[0], i)
                ymin = max(tl[1], j)
                xmax = min(br[0], i + 1)
                ymax = min(br[1], j + 1)

                coef = (xmax - xmin) * (ymax - ymin) / crop_area
                try:
                    result = np.sum([result,
                                     img_features[:, i, j] * coef],
                                     axis=0)
                except IndexError:
                    print('h_region_count {}\n v_region_count {}\n i {}\n j {}\n img_features.shape {}'.format(h_region_count, v_region_count, i, j, img_features.shape))
                    raise IndexError
                
        return result if to_numpy else torch.from_numpy(result)
    

    def get_all_fixations_features(self,
                                   reflacx_sample,
                                   stdevs=1,
                                   mean_features=None):
        img = reflacx_sample.get_dicom_img()
        img_features = self.get_img_features(img,
                                             to_numpy=True,
                                             mean_features=mean_features)
        result = {}

        for i, fixation in enumerate(reflacx_sample.get_fixations()):
            position = (fixation['x_position'], fixation['y_position'])
            ang_x = fixation['angular_resolution_x_pixels_per_degree']
            ang_y = fixation['angular_resolution_y_pixels_per_degree']
            fix_features = self.get_fixation_features(position,
                                                      ang_x,
                                                      ang_y,
                                                      img_features=img_features,
                                                      stdevs=stdevs,
                                                      mean_features=mean_features)
            if fix_features is not None: #TODO log when this happens
                result[i] = fix_features
        return result
    
    