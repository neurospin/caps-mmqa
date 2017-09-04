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


def control_fov(mask_file, threshold=0, verbose=0):
    """
    Control if the brain is fully covered by the FOV for a 3D volume
    Can be used with mean realigned image or on each 3d volume of a 4D serie

    Parameters
    ----------

    Inputs:
        mask_file: str, a 3D binary volume (mandatory)
        threshold: int, the number of voxels that are allowed to intersect
            the edge of the FoV before we consider the image flawed
    output:
        fov_coverage: bool, True if the mask is entirely contained within the
            FoV, False if not (to many pixels on the edge)
        intersect_score: number of voxels that are on the edge of the FoV
    """

    mask_data = nibabel.load(mask_file).get_data()

    if verbose:
        print "the volume shape is {}".format(mask_data.shape)

    posterior = np.sum(mask_data[:, 0, :])
    anterior = np.sum(mask_data[:, -1, :])
    inferior = np.sum(mask_data[:, :, 0])
    superior = np.sum(mask_data[:, :, -1])
    left = np.sum(mask_data[0, :, :])
    right = np.sum(mask_data[-1, :, :])

    intersec_score = sum([posterior, anterior, inferior, superior,
                          left, right])

    if verbose > 0:
        print "intersection scores are:"
        print "posterior:", posterior
        print "anterior:", anterior
        print "inferior:", inferior
        print "superior:", superior
        print "left:", left
        print "right:", right
        print "total:", intersec_score

    fov_coverage = True

    if intersec_score > threshold:
        fov_coverage = False

    return fov_coverage, intersec_score
