"""Function(s) that regularize REFLACX diagnoses labels into a [0, 1] interval
"""

def linear_regularization(labels, original_interval=(0, 5)):
    """takes an array of labels, each known to be inside param:original_interval,
    and adjusts it to [0, 1]
    """
    low, high = original_interval
    interval_sz = high - low
    return [(x - low) / interval_sz for x in labels]