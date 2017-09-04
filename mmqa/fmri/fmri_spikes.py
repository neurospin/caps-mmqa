#! /usr/bin/env python
##########################################################################
# Nsap - Neurospin - Berkeley - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import os
import json
import numpy as np
import nibabel
import logging
from scipy.ndimage.morphology import binary_dilation

# Matplotlib backend: deal with no X terminal
import matplotlib as mpl
mpl.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# Define logger
logger = logging.getLogger(os.path.basename(__file__))

# CAPS import
from stats_utils import median_absolute_deviation
from stats_utils import time_slice_diffs
from stats_utils import format_time_serie
from stats_utils import mutual_information


def spike_detector(image_file, output_directory, zalph=5., time_axis=-1,
                   slice_axis=-2, title=None, roi_mask_file=None):
    """ Detect spiked slices.

    Raises
    ------
    ValueError: if the image dimension is different than 4.

    <unit>
        <input name="image_file" type="File" desc="A functional volume."/>
        <input name="output_directory" type="Directory" desc="The destination
            folder."/>
        <input name="zalph" type="Float" desc="Cut off for the sum of
            square."/>
        <input name="time_axis" type="Int" desc="Axis of the input array that
            varies over time. The default is the last axis."/>
        <input name="slice_axis" type="Int" desc="Axis of the array that
            varies over image slice. The default is the last non-time axis."/>
        <input name="title" type="String" desc="The first part of the figures'
            title (optional)" optional="True"/>
        <output name="snap_spikes" type="File" desc="A snap with the dectected
            spikes."/>
        <output name="spikes_file" type="File" desc="The detected spikes
            array."/>
    </unit>
    """
    # Load the image and get the associated numpy array
    image = nibabel.load(image_file)
    array = image.get_data()

    # get_roi only if necessary
    if roi_mask_file is not None:
        roi_mask_nif = nibabel.load(roi_mask_file)
        roi_mask = roi_mask_nif.get_data()
        roi_mask = binary_dilation(roi_mask, iterations=4)
        roi_mask = roi_mask.astype(int)
        nx = roi_mask.shape[0]
        ny = roi_mask.shape[1]
        roi_mask[int(round(float(nx) / 5, 0)):4 * int(round(float(nx) / 5, 0)),
                 int(round(float(ny) / 5, 0)):4 * int(round(float(ny) / 5, 0)),
                 :] = 1
        roi_mask = np.repeat(roi_mask[:, :, :, np.newaxis],
                             array.shape[3], axis=3)
        roi_nifti = nibabel.Nifti1Image(roi_mask,
                                        affine=roi_mask_nif.get_affine())
        nibabel.save(roi_nifti,
                     os.path.join(output_directory, "roi_spike_mask.nii.gz"))
        array = array * (1. - roi_mask)

    # Check the input specified axis parameters
    if array.ndim != 4:
        raise ValueError("Time image dimension is '{0}', expect a 4d "
                         "volume.".format(array.ndim))

    # Run the spike detection
    snap_spikes = os.path.join(output_directory, "spikes.pdf")
    slices_to_correct, spikes = detect_spikes_time_slice_diffs(
        array, zalph=zalph, time_axis=time_axis, slice_axis=slice_axis,
        output_fname=snap_spikes, title=title)

    # Save the result in a json
    spikes_file = os.path.join(output_directory, "spikes.json")
    for key, value in slices_to_correct.iteritems():
        slices_to_correct[key] = value.tolist()

    out_dict = {"spikes": slices_to_correct}
    with open(spikes_file, "w") as json_data:
        json.dump(out_dict, json_data)

    return snap_spikes, spikes_file


def detect_spikes_time_slice_vals(array, time_axis=-1, slice_axis=-2, sigma=2,
                                  output_fname=None):
    """ Detect spiked slices using the time slice values as a criterion.

    Parameters
    ----------
    array: array (mandatory)
        array where the time is the last dimension.
    time_axis: int (optional, default -1)
        axis of the input array that varies over time. The default is the last
        axis.
    slice_axis: int (optional default -2)
        axis of the array that varies over image slice. The default is the last
        non-time axis.
    sigma: float (optional default 3)
        cut off for the sum of square.
    output_fname: str (optional default None)
        the output file name where the image that represents the detected
        spikes is saved.

    Returns
    -------
    slices_to_correct : dict
        the timepoints where spikes are detected as keys and the corresponding
        spiked slices.
    """
    # Reshape the input array: roll time and slice axis
    logger.info("Input array shape is '%s'", array.shape)
    array = format_time_serie(array, time_axis, slice_axis)
    logger.info("Reshape array shape is '%s' when time axis is '%s' and slice "
                "axis is '%s'", array.shape, time_axis, slice_axis)

    # Go through all time-points
    gaussians = []
    for timepoint in range(array.shape[0]):

        # Go through all slices and compute the mutual information to the
        # the referecne first slice
        reference_slice = array[timepoint, 0]
        mi_values = []
        for sliceid in range(array.shape[1]):

            # Compute the mutual information between slices
            mi_values.append(mutual_information(
                reference_slice, array[timepoint, sliceid], bins=256))

        # Remove the mi corresponding to MI(X, X) = H(X)
        mi_values = np.asarray(mi_values[1:])

        # Make the assumption of a gaussian distribution
        mean = np.mean(mi_values)
        std = np.std(mi_values)

        # Reject slice to far from the distribution +/-n sigma
        lower_bound = mean - sigma * std
        upper_bound = mean + sigma * std

        # Store the result
        gaussians.append({
            "dist": mi_values,
            "mean": mean,
            "bounds": (lower_bound, upper_bound),
            "outliers": []
        })
        print np.where(mi_values < lower_bound)
        print np.where(mi_values > upper_bound)

    # Display nicely the slice distributions
    if output_fname is not None:
        display_slice_distributions(gaussians, 18, output_fname)


def display_slice_distributions(gaussians, bins, output_fname):
    """ Display the computed slice distance distribution.

    Parameters
    ----------
    gaussians: list of dict
        Keys are dist: array (N) containing the the slice to slice distances,
        mean: float under the gaussian assumtpion, the mean of the
        distribution, bounds: 2-uplet under the gaussian assumtpion, the lower
        and uppper cut-offs.
    bins: int
        the number of histogram bins.
    output_fname: str
        the output file name where the image is saved.

    Raises
    ------
    ValueError: if the base directory of `output_fname` does not exists.
    """
    # Check the input destination file parameter
    if not os.path.isdir(os.path.dirname(output_fname)):
        raise ValueError("The output file name '{0}' point to an invalid "
                         "directory.".format(output_fname))

    # Go through all timepoints
    pdf = PdfPages(output_fname)
    try:
        for cnt, dataset in enumerate(gaussians):

            fig = plt.figure()
            plot = fig.add_subplot(111)
            plot.grid()
            plt.title("Spikes at slice {0}".format(cnt))
            plt.hist(dataset["dist"], bins, normed=1, facecolor="green",
                     alpha=0.5)
            plot.plot((dataset["mean"], dataset["mean"]), (0, 1), "r")
            plot.plot((dataset["bounds"][0], dataset["bounds"][0]),
                      (0, 1), "b")
            plot.plot((dataset["bounds"][1], dataset["bounds"][1]),
                      (0, 1), "b")
            plot.axes.set_xlabel("Smarts")
            plot.axes.set_ylabel("Probability")
            pdf.savefig(fig)
            plt.close()

        # Close pdf file
        pdf.close()

    except:
        pdf.close()
        raise


def detect_spikes_time_slice_diffs(array, zalph=5., time_axis=-1,
                                   slice_axis=-2, output_fname=None,
                                   title=None):
    """ Detect spiked slices using the time slice difference as a criterion.

    Parameters
    ----------
    array: array (mandatory)
        array where the time is the last dimension.
    zalph: float (optional default 5)
        cut off for the sum of square.
    time_axis: int (optional, default -1)
        axis of the input array that varies over time. The default is the last
        axis.
    slice_axis: int (optional default -2)
        axis of the array that varies over image slice. The default is the last
        non-time axis.
    output_fname: str (optional default None)
        the output file name where the image that represents the detected
        spikes is saved.

    Returns
    -------
    slices_to_correct : dict
        the timepoints where spikes are detected as keys and the corresponding
        spiked slices.
    spikes: array (T-1, S)
        all the detected spikes.
    """
    # Reshape the input array: roll time and slice axis
    logger.info("Input array shape is '%s'", array.shape)
    array = format_time_serie(array, time_axis, slice_axis)
    logger.info("Reshape array shape is '%s' when time axis is '%s' and slice "
                "axis is '%s'", array.shape, time_axis, slice_axis)

    # Time-point to time-point differences over and slices
    logger.info("Computing time-point to time-point differences over "
                "slices...")
    smd2 = time_slice_diffs(array)
    logger.info("Metric smd2 shape is '%s', ie. (number of timepoints - 1, "
                "number of slices).", smd2.shape)

    # Detect spikes from quared difference
    spikes, meds, mads = spikes_from_slice_diff(smd2, zalph)

    # Filter the spikes to preserve outliers only
    final_spikes = final_detection(spikes)

    # Find which timepoints and which slices are affected
    times_to_correct = np.where(final_spikes.sum(axis=1) > 0)[0]

    slices_to_correct = {}
    for timepoint in times_to_correct:
        slices_to_correct[timepoint] = np.where(
            final_spikes[timepoint, :] > 0)[0]

    # Information message
    logger.info("Total number of outliers found: '%s'.", spikes.sum())
    logger.info("Total number of slices to be corrected: '%s'.",
                slices_to_correct)

    # Display detected spikes
    if output_fname is not None:
        display_spikes(array, smd2, spikes, meds, mads, zalph,
                       output_fname, title)

    return slices_to_correct, spikes


def display_spikes(array, smd2, spikes, meds, mads, zalph, output_fname, title,
                   figsize=(11.7, 8.3)):
    """ Display the detected spikes.

    Parameters
    ----------
    smd2: array (T-1, S) (mandatory)
        array containing the mean (over voxels in volume) of the
        squared difference from one time point to the next.
    spikes: array (T-1, S)
        the detected spikes array.
    output_fname: str
        the output file name where the image is saved.

    Raises
    ------
    ValueError: if the base directory of `output_fname` does not exists.
    """
    # Check the input destination file parameter
    if not os.path.isdir(os.path.dirname(output_fname)):
        raise ValueError("The output file name '{0}' point to an invalid "
                         "directory.".format(output_fname))

    # Plot information
    cnt = 1
    nb_of_timepoints = smd2.shape[0]

    # Go through all timepoints
    pdf = PdfPages(output_fname)
    try:
        for timepoint_smd2, timepoint_spikes, med, mad in zip(smd2.T,
                                                              spikes.T,
                                                              meds, mads):

            # If at least one spike is detected, generate a subplot
            if timepoint_spikes.sum() > 0:

                fig = plt.figure()
                plot = fig.add_subplot(111)
                plot.grid()
                if title:
                    plt.title("{1}\nSpikes for slice {0}".format(cnt - 1,
                                                                 title))
                else:
                    plt.title("Spikes for slice {0}".format(cnt - 1))

                plt.plot(range(nb_of_timepoints), timepoint_smd2, "yo-",
                         range(nb_of_timepoints),
                         [med] * nb_of_timepoints, 'g',
                         range(nb_of_timepoints),
                         [med + mad * zalph] * nb_of_timepoints, 'r')
                for spike_index in np.where(timepoint_spikes > 0)[0]:

                    plot.axes.annotate("",
                                       (spike_index,
                                        1.02 * timepoint_smd2[spike_index]),
                                       xytext=(
                                           spike_index,
                                           1.08 * timepoint_smd2[spike_index]),
                                       xycoords="data",
                                       size=2, va="top", ha="center",
                                       arrowprops=dict(facecolor='blue',
                                                       width=1.5,
                                                       frac=0.1,
                                                       headwidth=2.5)
                                       )

                plot.axes.set_xlabel("Volumes")
                plot.axes.set_ylabel("Slice mean squared difference")
                pdf.savefig(fig)
                plt.close(fig)
                # Display a mosaic with the spiked slices
                spikes_indices = np.where(timepoint_spikes > 0)[0]
                spike_nb = len(spikes_indices)
                nb_rows_perpage = 1
                nb_cols_perpage = 2
                nb_im_perpage = nb_cols_perpage * nb_rows_perpage
                page_index = 1
                fig = plt.figure(figsize=figsize)
                fig.subplots_adjust(top=0.85)
                for index, spike_index in enumerate(spikes_indices):
                    if not index // nb_im_perpage < page_index:
                        pdf.savefig(fig)
                        fig = plt.figure(figsize=figsize)
                        fig.subplots_adjust(top=0.85)
                        page_index += 1
                    ax = fig.add_subplot(
                        nb_rows_perpage, nb_cols_perpage,
                        index + 1 - (page_index - 1) * nb_im_perpage)
                    plt.title("Slice {0} - volume {1}".format(
                        cnt - 1, spike_index, meds[cnt - 1], mads[cnt - 1]))
                    ax.imshow(array[spike_index, cnt - 1, :, :],
                              cmap=mpl.cm.Greys_r)
                    frame = plt.gca()
                    ax.axis("off")
                pdf.savefig(fig)
                plt.clf()
                plt.close(fig)

            # Increment slice number
            cnt += 1

        # Close pdf file
        pdf.close()

    except:
        pdf.close()
        raise


def spikes_from_slice_diff(smd2, zalph=5., lower_zalph=3.):
    """ Detect spiked slices.

    Notation: T is the number of time points (TRs) and S is the number of
    slices.

    Parameters
    ----------
    smd2: array (T-1, S) (mandatory)
        array containing the mean (over voxels in volume) of the
        squared difference from one time point to the next
    zalph: float (optional default 5)
        cut off for the sum of square.
    lower_zalph: float (optional default 3)
        lower cut off for the sum of square. Used to detect histeresis spikes.
        Value must be above this threshold to be a candidate for histeresis.

    Returns
    -------
    spikes: array (T-1, S)
        the detected spikes array.
    """
    # Information message
    logger.info("Entering slice spikes detection...")

    # Initialize the detection result
    shape = smd2.shape
    spikes = np.zeros(shape=shape, dtype=np.int)
    lower_spikes = np.zeros(shape=shape, dtype=np.int)

    locs = []
    scales = []

    # Go through all slices
    for slice_index in range(shape[1]):

        # Information splitter
        logger.info("{0} Computing slice '{1}'...".format(
            "-" * 10, slice_index))

        # Compute distribution mean and dispertion
        loc = np.median(smd2[:, slice_index])
        scale = median_absolute_deviation(smd2[:, slice_index])

        locs.append(loc)
        scales.append(scale)

        # Detect the outliers
        spikes[:, slice_index] = (smd2[:, slice_index] > loc + zalph * scale)
        lower_spikes[:, slice_index] = (smd2[:, slice_index] > loc +
                                        lower_zalph * scale)
        nb_spikes = spikes[:, slice_index].sum()
        logger.info("Found '%s' spike(s) at slice '%s' between timepoints "
                    "'%s' . The lower spikes are '%s'.", nb_spikes,
                    slice_index,
                    spikes[:, slice_index], lower_spikes[:, slice_index])

    return spikes, locs, scales


def detect_pattern(array, pattern, ppos=None, dpos=0):
    """ Detect a pattern in the fisrt axis of a numpy array.

    Parameters
    ----------
    array: array (N, M)
        the input data - pattern is search over axis 0.
    pattern: 1-dimension array or list
        the pattern to detect.
    ppos: 's'|'e'|integer | None
        pattern position: a specific position in time to detect the pattern,
        None means over all possible axis 0 positions.
    dpos: integer
        where to put '1' or 'True' in the result array when pattern is detected
        (0: start of pattern).

    Returns
    -------
    hits: array (N, M)
        the match result.

    Raises
    ------
    ValueError: if a wrong pattern is specified.
    """
    # Inner parameters
    shape = array.shape
    hits = np.zeros(shape, dtype=np.bool)
    pattern = np.asarray(pattern)
    pshape = pattern.shape

    # Check the input parameters
    if pattern.ndim != 1:
        raise ValueError("Invalid pattern '{0}'.".format(pattern))

    # Pattern instersection
    nb_of_hits = shape[0] - pshape[0] + 1
    hits = np.ones((nb_of_hits, shape[1]), dtype=np.bool)
    for cnt, pattern_value in enumerate(pattern):
        local_match = (array[cnt: cnt + nb_of_hits, :] == pattern_value)
        hits = np.logical_and(hits, local_match)

    return hits


def final_detection(spikes):
    """ This function takes an array with zeros or ones, look at when two
    "ones" follow each other in the time direction (first dimension), and
    return an array of ones in these cases. These are the slices that we can
    potentially correct if they are isolated.

    Parameters
    ----------
    spikes: array (T-1, S)
        the detected spikes array.

    Returns
    -------
    final: array (T, S)
        the spikes array.
    """
    # Initialize the detection result
    shape = spikes.shape
    final = -1 * np.ones(shape=(shape[0] + 1, shape[1]), dtype=np.int)
    # Detect patterns of interest
    final[0] = detect_pattern(spikes[:2, :], [1, 0])
    final[1] = detect_pattern(spikes[:3, :], [1, 1, 0])
    final[2: shape[0] - 1] = detect_pattern(spikes, [0, 1, 1, 0])
    final[-2] = detect_pattern(spikes[-3:], [0, 1, 1])
    final[-1] = detect_pattern(spikes[-2:], [0, 1])
    # Detect if all values in the final array have been set
    if np.min(final) < 0:
        raise ValueError("Detection failed, check matrices shapes")

    # Information message
    logger.info("The final spike detection matrix is '%s' when looking for "
                "global pattern [0, 1, 1, 0], begining patterns [1, 0] and "
                "[1, 1, 0] and final patterns [0, 1, 1] and [0, 1].", final)

    return final


if __name__ == "__main__":

    from caps.toy_datasets import get_sample_data
    import logging

    logging.basicConfig(level=logging.INFO)

    localizer_dataset = get_sample_data("localizer")

    array = nibabel.load(localizer_dataset.fmri).get_data()
    detect_spikes_time_slice_vals(
        array, time_axis=-1, slice_axis=-2, sigma=2,
        output_fname="/volatile/nsap/catalogue/quality_assurance/spikes.pdf")

    slices_to_correct, _ = spike_detector(
        localizer_dataset.fmri, zalph=5., time_axis=-1, slice_axis=-2,
        output_fname="/volatile/nsap/catalogue/quality_assurance/spikes.pdf")
    print slices_to_correct
