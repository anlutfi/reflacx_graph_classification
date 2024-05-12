from consts import CSV_SEP, FIX_OUT_OF_CHEST

class FixationNode:
    """A graph's node representing a REFLACX fixation
    """
    @staticmethod
    def csv_header():
        """returns a header of class attributes' names
        to be used as header of csv file
        """
        return CSV_SEP.join(['node_id',
                             'norm_x',
                             'norm_y',
                             'norm_top_left',
                             'norm_bottom_right',
                             'duration',
                             'feats'])
        
    
    @staticmethod
    def new_node(id,
                 fixation,
                 chest_bb,
                 img,
                 feature_extractor,
                 img_features=None,
                 stdevs=1):
        """Transforms REFLACX fixation data into a node.
        if param:fixation's position lies outside of the xary's chest bounding box (param:chest_bb),
        returns None, as it's an invalid node. For this reason, this function should be called instead of the constructor.

        param:img is the reflacx xray, used by param:feature_extractor to obtain a tensor
        of image features. If features were already calculated, pass them to param:img_features, to avoid calculation at each fixation.

        param:stdevs is the number of standard deviations to be considered to determine
            the fixation's crop.
        """
        bb_xrange = chest_bb['xmax'] - chest_bb['xmin']
        bb_yrange = chest_bb['ymax'] - chest_bb['ymin']
        
        norm_x = ((fixation['x_position'] - chest_bb['xmin']) / bb_xrange)
        norm_y = ((fixation['y_position'] - chest_bb['ymin']) / bb_yrange)
        
        if norm_x < 0 or norm_x > 1 or norm_y < 0 or norm_y > 1:
            return FIX_OUT_OF_CHEST
        
        ang_x = fixation['angular_resolution_x_pixels_per_degree']
        ang_y = fixation['angular_resolution_y_pixels_per_degree']

        viewed_x_min = int(max(0, fixation['x_position'] - ang_x * stdevs))
        viewed_x_max = int(min(img.shape[1],
                               fixation['x_position'] + ang_x * stdevs))
        viewed_y_min = int(max(0, fixation['y_position'] - ang_y * stdevs))
        viewed_y_max = int(min(img.shape[0],
                               fixation['y_position'] + ang_y * stdevs))
        
        viewed_x_min = max(0, viewed_x_min - chest_bb['xmin']) / bb_xrange
        viewed_x_max = ((min(chest_bb['xmax'], viewed_x_max) - chest_bb['xmin'])
                        / bb_xrange)
        viewed_y_min = max(0, viewed_y_min - chest_bb['ymin']) / bb_yrange
        viewed_y_max = ((min(chest_bb['ymax'], viewed_y_max) - chest_bb['ymin'])
                        / bb_yrange)

        return FixationNode(id,
                            fixation,
                            norm_x,
                            norm_y,
                            (viewed_x_min, viewed_y_min),
                            (viewed_x_max, viewed_y_max),
                            ang_x,
                            ang_y,
                            img,
                            feature_extractor,
                            img_features=img_features,
                            stdevs=stdevs)
        
    
    def __init__ (self,
                  id,
                  fixation,
                  norm_x,
                  norm_y,
                  norm_topleft,
                  norm_bottomright,
                  ang_x,
                  ang_y,
                  img,
                  feature_extractor,
                  img_features=None,
                  stdevs=1):
        """See FixationNode.new_node()
        """
        self.id = id
        self.duration = (fixation['timestamp_end_fixation'] -
                         fixation['timestamp_start_fixation'])
        self.norm_x = norm_x
        self.norm_y = norm_y
        
        self.topleft = norm_topleft
        self.bottomright = norm_bottomright

        try:
            fs = feature_extractor.get_fixation_features((fixation['x_position'],
                                                          fixation['y_position']),
                                                          ang_x,
                                                          ang_y,
                                                          img=img,
                                                          img_features=img_features,
                                                          stdevs=stdevs)
        except IndexError:
            fs = None
        
        self.features = fs

    
    def __str__(self):
        feature_str = '\"{}\"'.format(CSV_SEP.join([str(float(x)) for x in self.features]))
        crop_str = lambda p: '\"{}\"'.format(CSV_SEP.join([str(x) for x in p]))
        return CSV_SEP.join([str(self.norm_x),
                             str(self.norm_y),
                             crop_str(self.topleft),
                             crop_str(self.bottomright),
                             str(self.duration),
                             feature_str])
