#! /usr/bin/env python
##########################################################################
# CAPS - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
import nibabel
import numpy as np


def signal_loss(normalized_file, mask_file, roi_mask_file, verbose=0):
    """
    Control if a signal loss occured in the 4D volume in the region specified
    by the mask file (lower anterio quadrant of the brain by default)

    NOTE: it is recommended to run this metric BEFORE any smoothing is
    performed on the normalized volume
    Parameters
    ----------

    Inputs:
        normalized_file: str, a 4D or mean (3D) of the normalized volume
        mask_file: str, a 3D binary volume (mandatory)
    output:
        mean_intensity: float, the mean signal intensity in the mask area
    """

    mask_roi_data = nibabel.load(roi_mask_file).get_data()
    mask_data = nibabel.load(mask_file).get_data()
    data = nibabel.load(normalized_file).get_data()

    if len(data.shape) == 4:
        if verbose > 0:
            print "Compute mean image from a 4D-volume"
        mean_volume = np.mean(data, axis=3)
    else:
        mean_volume = data

    if verbose:
        print "the volume shape is {}".format(mean_volume.shape)

    return float(np.mean(data * mask_roi_data) / np.mean(data * mask_data))
