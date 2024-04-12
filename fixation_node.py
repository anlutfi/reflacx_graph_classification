class FixationNode:
    @staticmethod
    def new_node(id,
                 fixation,
                 chest_bb,
                 img,
                 feature_extractor,
                 img_features=None,
                 stdevs=1):
        norm_x = ((fixation['x_position'] - chest_bb['xmin']) /
                  (chest_bb['xmax'] - chest_bb['xmin']))
        norm_y = ((fixation['y_position'] - chest_bb['ymin']) /
                  (chest_bb['ymax'] - chest_bb['ymin']))

        if norm_x < 0 or norm_x > 1 or norm_y < 0 or norm_y > 1:
            return None
        return FixationNode(id,
                            fixation,
                            norm_x,
                            norm_y,
                            img,
                            feature_extractor,
                            img_features=img_features,
                            stdevs=stdevs)
        
    
    def __init__ (self,
                  id,
                  fixation,
                  norm_x,
                  norm_y,
                  img,
                  feature_extractor,
                  img_features=None,
                  stdevs=1):
        self.id = id
        self.duration = (fixation['timestamp_end_fixation'] -
                         fixation['timestamp_start_fixation'])
        self.norm_x = norm_x
        self.norm_y = norm_y
        
        ang_x = fixation['angular_resolution_x_pixels_per_degree']
        ang_y = fixation['angular_resolution_y_pixels_per_degree']
        
        self.viewed_x_min = int(max(0, fixation['x_position'] - ang_x * stdevs))
        self.viewed_x_max = int(min(img.shape[1],
                                    fixation['x_position'] + ang_x * stdevs))
        self.viewed_y_min = int(max(0, fixation['y_position'] - ang_y * stdevs))
        self.viewed_y_max = int(min(img.shape[0],
                                    fixation['y_position'] + ang_y * stdevs))
        
        fs = feature_extractor.get_fixation_features((fixation['x_position'],
                                                      fixation['y_position']),
                                                      ang_x,
                                                      ang_y,
                                                      img=img,
                                                      img_features=img_features,
                                                      stdevs=stdevs)
        
        self.features = fs

    
    def get_csv_header(self):
        return "id, norm_x, norm_y, duration, features"

    
    def __str__(self):
        feature_str = '[{}]'.format(', '.join([str(x) for x in self.features])) #TODO review csv array format
        return ', '.join([str(self.id),
                          str(self.norm_x),
                          str(self.norm_y),
                          str(self.duration),
                          feature_str])