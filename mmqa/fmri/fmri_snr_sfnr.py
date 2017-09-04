#! /usr/bin/env python
##########################################################################
# CAPS - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import os
import json
import nibabel
import numpy
import matplotlib
import scipy.ndimage as ndim

matplotlib.use("AGG")

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def signal_to_noise_ratio(image_file, output_directory,
                          exclude_volumes, roi_size=20):
    """
    Compute signal to noise ratio of a 4D image (from Velasco 2014)
    "It takes as 'signal' the average voxel intensity in all the ROIs
    defined in the object, averaged across time. It takes as 'noise' the
    standard deviation (acress space) of the signal in the ROIs defined in
    the background, and then averages across time.

    <unit>
        <input name="image_file" type="File" desc="A functional volume."/>
        <input name="output_directory" type="Directory" desc="The output
            directory"/>
        <input name="roi_size" type="Int" desc="Size of the central ROI
            (optional)" optional="True"/>
        <input name="exclude_volumes" type="List" content="Int" desc="Exclude
            some temporal positions (optional)" optional="True"/>
        <output name="snr0_file" type="File" desc="The SNR0 score in a
            json file"/>
    </unit>
    """
    # Get image data
    img = nibabel.load(image_file)
    array_image = img.get_data()

    if len(exclude_volumes) > 0:
        to_keep = sorted(set(range(
            array_image.shape[3])).difference(exclude_volumes))
        array_image = array_image[:, :, :, to_keep]

    # Create a central roi
    center = numpy.round(numpy.asarray(array_image.shape) / 2)
    if roi_size == 1:
        signal_roi_array = array_image[center[0],
                                       center[1],
                                       center[2] + 3]
    # Even dimension, cannot be perfectly centered
    elif roi_size % 2 == 0:
        signal_roi_array = array_image[
            center[0] - (roi_size / 2 - 1): center[0] + (roi_size / 2) + 1,
            center[1] - (roi_size / 2 - 1): center[1] + (roi_size / 2) + 1,
            center[2] + 3]
    else:
        signal_roi_array = array_image[
            center[0] - (roi_size - 1) / 2: center[0] + (roi_size - 1) / 2 + 1,
            center[1] - (roi_size - 1) / 2: center[1] + (roi_size - 1) / 2 + 1,
            center[2] + 3]

    # average over time and space
    signal_summary = numpy.average(signal_roi_array)

    average_img = numpy.average(array_image, axis=3)

    mask = average_img > (signal_summary / 3)
    mask = mask.astype(int)

    # Dilation of the mask
    mask = ndim.binary_dilation(mask, iterations=5).astype(
        mask.dtype)
    # save the mask
    nif = nibabel.Nifti1Image(mask, img.get_affine())
    nibabel.save(nif, os.path.join(output_directory, "mask.nii.gz"))

    # compute the standard deviation for each volume
    stds = [numpy.std(numpy.ma.masked_array(array_image[:, :, :, x],
                                            mask=mask))
            for x in range(array_image.shape[3])]

    maxs = [numpy.max(numpy.ma.masked_array(array_image[:, :, :, x],
                                            mask=mask))
            for x in range(array_image.shape[3])]

    means = [numpy.mean(numpy.ma.masked_array(array_image[:, :, :, x],
                                              mask=mask))
             for x in range(array_image.shape[3])]

    # average over time
    noise_summary = numpy.average(stds)
    max_summary = numpy.average(maxs)
    mean_summary = numpy.average(means)

    # comute scores and save it in json file
    snr = signal_summary / (1.53 * noise_summary)
    sog_max = signal_summary / max_summary
    sog_mean = signal_summary / mean_summary

    results = {
        "snr0": float(snr),
        "sog_max": float(sog_max),
        "sog_mean": float(sog_mean)
        }
    with open(os.path.join(output_directory, "snr0.json"), "w") as _file:
        json.dump(results, _file)

    snr0_file = os.path.join(output_directory, "snr0.json")

    return snr0_file


def snr_percent_fluctuation_and_drift(image_file, repetition_time, roi_size,
                                      output_directory, title=None,
                                      exclude_volumes=[]):
    """ compute the SFNR (Signal-to-fluctuation-Noise Ratio)
        from Friedman et al. (2006)

    <unit>
        <input name="image_file" type="File" desc="A functional volume."/>
        <input name="repetition_time" type="Float" desc="The fMRI sequence
            repetition time (in seconds)."/>
        <input name="roi_size" type="Int" desc="The central ROI size used to
            compute the fluuctation and drift."/>
        <input name="output_directory" type="Directory" desc="The destination
            folder."/>
        <input name="title" type="String" desc="The first part of the figure's
            title (optional)" optional="True"/>
        <input name="exclude_volumes" type="List" content="Int" desc="Exclude
            some temporal positions (optional)" optional="True"/>
        <output name="snap_fluctuation_drift" type="File" desc="A functional
        volume."/>
        <output name="fluctuation_drift_file" type="File" desc="A score
        in a json file."/>
    </unit>
    """
    # Get image data
    array_image = load_fmri_dataset(image_file)

    if len(exclude_volumes) > 0:
        to_keep = sorted(set(range(
            array_image.shape[3])).difference(exclude_volumes))
        array_image = array_image[:, :, :, to_keep]

    # compute the SFNR (Signal-to-fluctuation-Noise Ratio)
    fmri_summary_signal = get_fmri_signal(array_image)
    temporal_fluct_noise_image = get_fmri_temporal_fluctuation_noise(
        array_image)
    sfnr_image, sfnr_score = get_signal_to_fluctuation_noise_ratio(
        fmri_summary_signal,
        temporal_fluct_noise_image)

    # Compute Static Spatial Noise Image
    ssn_array = get_static_spatial_noise(array_image)

    snr = get_spatial_noise_ratio(fmri_summary_signal, ssn_array,
                                  array_image.shape[3],
                                  roi_size=roi_size)

    # Compute the drift and fluctuation
    (mean_signal_intensity, average_intensity, polynomial, residuals,
     fluctuation, drift) = get_snr_percent_fluctuation_and_drift(
        array_image,
        roi_size=roi_size)

    spectrum = get_residuals_spectrum(residuals, mean_signal_intensity,
                                      repetition_time)

    # Compute a weisskoff analysis on the fluctuation
    (fluctuations, theoretical_fluctuations,
     rdc, max_fluctuation, max_roi_size) = get_weisskoff_analysis(array_image)

    # Save the result in a json
    fluctuation_drift_file = os.path.join(
        output_directory, "snr_fluctuation_drift.json")
    results = {
        "drift": drift * 100.,
        "snr": snr,
        "sfnr": sfnr_score,
        "weisskoff_rdc": rdc,
        "fluctuation": fluctuation * 100.
    }

    with open(fluctuation_drift_file, "w") as json_data:
        json.dump(results, json_data)

    # Display result in a pdf
    snap_fluctuation_drift = os.path.join(
        output_directory, "snr_fluctuation_drift.pdf")
    pdf = PdfPages(snap_fluctuation_drift)
    try:
        # Plot one figure per page
        fig = time_series_figure(average_intensity, polynomial, drift, snr,
                                 title)
        pdf.savefig(fig)
        plt.close(fig)
        fig = spectrum_figure(spectrum, title)
        pdf.savefig(fig)
        plt.close(fig)
        fig = weisskoff_figure(fluctuations, theoretical_fluctuations, rdc,
                               max_fluctuation, max_roi_size, title)
        pdf.savefig(fig)
        plt.close(fig)

        # Close the pdf
        pdf.close()
    except:
        pdf.close()
        raise

    return snap_fluctuation_drift, fluctuation_drift_file


def load_fmri_dataset(fmri_file):
    """ Load a functional volume

    Parameters
    ----------
    fmri_file: str (mandatory)
        the path to the functional volume.

    Returns
    -------
    array: array [X,Y,Z,N]
        the functional volume.
    """
    img = nibabel.load(fmri_file)
    return img.get_data()


def get_fmri_signal(array):
    """ The fmri signal is the the average voxel intensity across time

    A signal summary value is obtained from an ROI placed in the center.

    Parameters
    ----------
    array: array [X,Y,Z,N]
        the functional volume.

    Returns
    -------
    signal: array [X,Y,Z]
        the fmri signal.

    Notes
    -----
    FMRI Data Quality, Pablo Velasco, 2014
    Report on a Multicenter fMRI Quality Assurance Protocol, Lee Friedman, 2006
    SINAPSE fMRI Quality Assurance, Katherine Lymer
    """
    return numpy.average(array, 3)


def get_fmri_temporal_fluctuation_noise(array):
    """ The fluctuation noise image is produced by substracting for each voxel
    a trend line estimated from the data (the BIRN function uses a second
    order polynomial - It's effectively a standard deviation image).

    By removing the trend from the data, the fluctuation noise image is an
    image of the standard deviation of the residuals, voxel by voxel.

    A linear trend may indicate a systematic increase or decrease in the data
    (caused by eg sensor drift)

    Parameters
    ----------
    array: array [X,Y,Z,N]
        the functional volume.

    Returns
    -------
    tfn: array [X,Y,Z]
        the temporal fluctuation noise array.
    """
    voxels_per_volume = reduce(lambda x, y: x * y, array.shape[: -1], 1)

    # Flatten input array
    flat_array = []
    for t in range(array.shape[-1]):
        flat_array.extend(array[:, :, :, t].ravel())

    # Compute the temporal fluctuation noise
    tfn = numpy.ndarray((voxels_per_volume,), dtype=numpy.single)
    x = numpy.arange(array.shape[3])

    for i in range(voxels_per_volume):

        # Get the temporal signal at one voxel
        y = flat_array[i::voxels_per_volume]

        # Estimate a second order polynomial trend
        polynomial = numpy.polyfit(x, y, 2)
        model = numpy.polyval(polynomial, x)

        # Compute the residuals
        residuals = y - model

        # Compute the temporal fluctuation noise
        tfn[i] = numpy.std(residuals)

    return tfn.reshape(array.shape[:-1])


def get_signal_to_fluctuation_noise_ratio(signal_array, tfn_array,
                                          roi_size=10):
    """ The SFNR image is is obtained by dividing, voxel by voxel,
    the mean fMRI signal image by the temporal fluctuation image.

    A 21 x 21 voxel ROI, placed in the center of the image, is created.
    The average SFNR across these voxels is the SFNR summary value.

    Parameters
    ----------
    signal_array: array [X,Y,Z]
        the fmri signal.
    tfn_array: array [X,Y,Z]
        the temporal fluctuation noise array.
    roi_size: int (default 10)
        the size of the central roi used to get the summary indice.

    Returns
    -------
    sfnr_array: array [X,Y,Z]
        the signal to fluctuation noise ratio array.
    sfnr_summary: float
        the signal to fluctuation noise ratio average on a central roi.
    """
    # Compute the signal to fluctuation noise ratio
    sfnr_array = signal_array / (tfn_array + numpy.finfo(float).eps)

    # Create a central roi
    center = numpy.round(numpy.asarray(sfnr_array.shape) / 2)
    if roi_size == 1:
        roi = sfnr_array[center[0],
                         center[1],
                         center[2] + 3]
    # even dimension, cannot be perfectly centered
    elif roi_size % 2 == 0:
        roi = sfnr_array[
            center[0] - (roi_size / 2 - 1): center[0] + (roi_size / 2) + 1,
            center[1] - (roi_size / 2 - 1): center[1] + (roi_size / 2) + 1,
            center[2] + 3]
    else:
        roi = sfnr_array[
            center[0] - (roi_size - 1) / 2: center[0] + (roi_size - 1) / 2 + 1,
            center[1] - (roi_size - 1) / 2: center[1] + (roi_size - 1) / 2 + 1,
            center[2] + 3]

    # Compute the signal to fluctuation noise ratio summary
    sfnr_summary = numpy.average(roi)

    return sfnr_array, sfnr_summary


def get_static_spatial_noise(array):
    """ In order to measure the spatial noise, the sum of the odd and even
    numbered volumes are calculated separately.
    The static patial noise is then approximated by the differences of
    these two sums.

    If there is no drift in either amplitude or geometry across time series,
    then the difference image will show no structure.

    Parameters
    ----------
    array: array [X,Y,Z,N]
        the functional volume.

    Returns
    -------
    ssn_array: array [X,Y,Z]
        the static spatial noise.
    """
    shape_t = array.shape[3]
    odd_array = array[..., range(1, shape_t, 2)]
    odd_sum_array = numpy.sum(odd_array, 3)
    even_array = array[..., range(0, shape_t, 2)]
    even_sum_array = numpy.sum(even_array, 3)
    ssn_array = odd_sum_array - even_sum_array

    return ssn_array


def get_spatial_noise_ratio(signal_array, ssn_array, nb_time_points,
                            roi_size=21):
    """ A central ROI is placed in the center of the static spatial noise
    image. The SNR is the signal summary value divided by by the square root
    of the variance summary value divided by the numbre of time points

    SNR = (signal summary value)/sqrt((variance summary value)/#time points).

    Parameters
    ----------
    signal_array: array [X,Y,Z]
        the fmri signal.
    ssn_array: array [X,Y,Z]
        the static spatial noise.
    nb_time_points: int
        the number of time points in the fMRI serie.
    roi_size: int (default 10)
        the size of the central roi used to get the summary indice.

    Returns
    -------
    snr: float
        the fMRI image signal to noise ratio.

    Notes
    -----
    From National Electric Manufacturer Association (NEMA). Determination of
    signal-to-noise ratio (SNR) in diagnosis magnetic resonance images.
    Nema: Rosslyn, VA, 1988

    """
    # Create a central roi
    center = numpy.round(numpy.asarray(signal_array.shape) / 2)
    if roi_size == 1:
        signal_roi_array = signal_array[center[0],
                                        center[1],
                                        center[2] + 3,
                                        :]
        ssn_roi_array = ssn_array[center[0],
                                  center[1],
                                  center[2] + 3,
                                  :]
    # even dimension, cannot be perfectly centered
    elif roi_size % 2 == 0:
        signal_roi_array = signal_array[
            center[0] - (roi_size / 2 - 1): center[0] + (roi_size / 2) + 1,
            center[1] - (roi_size / 2 - 1): center[1] + (roi_size / 2) + 1,
            center[2] + 3]
        ssn_roi_array = ssn_array[
            center[0] - (roi_size / 2 - 1): center[0] + (roi_size / 2) + 1,
            center[1] - (roi_size / 2 - 1): center[1] + (roi_size / 2) + 1,
            center[2] + 3]
    else:
        signal_roi_array = signal_array[
            center[0] - (roi_size - 1) / 2: center[0] + (roi_size - 1) / 2 + 1,
            center[1] - (roi_size - 1) / 2: center[1] + (roi_size - 1) / 2 + 1,
            center[2] + 3]
        ssn_roi_array = ssn_array[
            center[0] - (roi_size - 1) / 2: center[0] + (roi_size - 1) / 2 + 1,
            center[1] - (roi_size - 1) / 2: center[1] + (roi_size - 1) / 2 + 1,
            center[2] + 3]

    # Get the signal and variance summaries
    signal_summary = numpy.average(signal_roi_array)
    variance_summary = numpy.var(ssn_roi_array)

    # Compute the SNR
    snr = signal_summary / numpy.sqrt(variance_summary / nb_time_points)

    return snr


def get_snr_percent_fluctuation_and_drift(array, roi_size=21):
    """ A time-series of the average intensity within a 21 x 21 voxel ROI
    centered in the image is calculated.

    A second-order polynomial trend is fit to the volume number vs
    average intensity.

    The mean signal intensity of the time-series (prior to detrending) and SD
    of the residuals after subtracting the fit line from the data
    are calculated.

    fluctuation = 100*(SD of the residuals)/(mean signal intensity)
    drift = ((max fit value) - (min fit value)) /
             (mean signal intensity signal)

    Parameters
    ----------
    array: array [X,Y,Z,N]
        the functional volume.
    roi_size: int (default 10)
        the size of the central roi used to get the summary indice.

    Returns
    -------
    average_intensity: array [N]
        the average voxel intensity across time.
    polynomial: array [N]
        the second-order polynomial used to remove the slow drift.
    residuals: array [N]
        the signal/model residuals.
    fluctuation: float
        the fluctuation value computed on a central roi.
    drift: float
        the signal temporal drift on a central roi.
    """
    # Create a central roi
    center = numpy.round(numpy.asarray(array.shape) / 2)
    if roi_size == 1:
        roi = array[center[0],
                    center[1],
                    center[2] + 3,
                    :]
    # even dimension, cannot be perfectly centered
    elif roi_size % 2 == 0:
        roi = array[
            center[0] - (roi_size / 2 - 1): center[0] + (roi_size / 2) + 1,
            center[1] - (roi_size / 2 - 1): center[1] + (roi_size / 2) + 1,
            center[2] + 3,
            :]
    else:
        roi = array[
            center[0] - (roi_size - 1) / 2: center[0] + (roi_size - 1) / 2 + 1,
            center[1] - (roi_size - 1) / 2: center[1] + (roi_size - 1) / 2 + 1,
            center[2] + 3,
            :]

    shape = roi.shape
    # Compute the mean signal intensity
    mean_signal_intensity = numpy.average(roi)

    # Compute the average voxel intensity across time
    if roi_size == 1:
        average_intensity = roi
    else:
        average_intensity = numpy.sum(
            numpy.sum(roi, 0), 0) + numpy.finfo(float).eps
        average_intensity /= (shape[0] * shape[1])

    # Compute the temporal fluctuation noise
    # > a second-order polynomial detrending to remove the slow drift
    x = numpy.arange(array.shape[3])
    polynomial = numpy.polyfit(x, average_intensity, 2)
    average_intensity_model = numpy.polyval(polynomial, x)
    # > a fluctuation value is calculated as the temporal standard deviation
    # of the residual variance of each voxel after the detrending
    residuals = average_intensity - average_intensity_model
    fluctuation = numpy.std(residuals) / mean_signal_intensity

    # Compute the drift
    drift = (average_intensity_model.max() -
             average_intensity_model.min()) / mean_signal_intensity

    return (mean_signal_intensity, average_intensity, polynomial, residuals,
            fluctuation, drift)


def get_residuals_spectrum(residuals, mean_signal_intensity, repetition_time):
    """ Residuals of the mean signal intensity fit are submitted to
    a fast Fourier transform (FFT).

    To allow multi-site comparison, we divide the residuals by the mean
    signal intensity.

    Parameters
    ----------
    residuals: array [N]
        the residuals between the time serie values and the model.
    repetition_time: float
        the repetition time in s.
    """
    fft = numpy.fft.rfft(residuals / mean_signal_intensity)
    if residuals.size % 2 == 0:
        # Discard real term for frequency n/2
        fft = fft[:-1]
    fftfreq = numpy.fft.fftfreq(len(residuals), repetition_time)
    return numpy.vstack((fftfreq[:len(fft)], numpy.abs(fft)))


def get_gaussian_model(residuals, n_bins=50):

    from scipy.stats import norm
    import matplotlib.mlab as mlab

    mu, std = norm.fit(residuals)

    plt.figure()
    plt.subplot(211)
    result = plt.hist(residuals, n_bins, histtype='bar')

    born_inf, born_supp = norm.interval(0.95, mu, std)

    x = numpy.linspace(min(residuals), max(residuals), 100)

    dx = result[1][1] - result[1][0]
    scale = len(residuals) * dx
    plt.plot(x, mlab.normpdf(x, mu, std) * scale, "r--", linewidth=2.0)
    plt.axvline(born_inf, color='g', linewidth=2.0)
    plt.axvline(born_supp, color='g', linewidth=2.0)
    plt.subplot(212)
    plt.plot(residuals)

    plt.show()


def get_weisskoff_analysis(array, max_roi_size=25):
    """ The Weisskoff analysis provides another measure of scanner
    stability included in the GSQAP.

    Parameters
    ----------
    array: array [X,Y,Z,N]
        the functional volume.
    max_roi_size: int
        the max size of the central size that will be used to compute the
        drift and fluctuation.

    Returns
    -------
    fluctuation: 2-uplet
        the roi sizes array and the fluctuation array.
    theoretical_fluctuation: 2-uplet
        the roi sizes array and the theoretical fluctuation array.
    """
    # Generate all the roi sizes we want to consider
    roi_sizes = numpy.arange(1, max_roi_size + 1)

    # Compute the fluctuation for each roi size
    fluctuation = [get_snr_percent_fluctuation_and_drift(array, s)[-2]
                   for s in roi_sizes]

    # The theorical fluctuation is given by the ine voxel roi fluctuation
    theoretical_fluctuation = fluctuation[0] / roi_sizes

    # Get the RDC (Radius of Decorrelation)
    # get the straight line equation y = ax + b for theorical decrease
    # where y = log(fluctuation) and x = log(roi_width)
    # from the theorical fluctuation equation, we have:
    # log(y) = log(x0) - log(x) (a = -1, b = log(x0))
    # so log(rdc) = log(x0) - log(y_rdc) with y_rdc =  fluctuation[-1]
    # and x0 = theoretical_fluctuation[0]

    rdc_log = numpy.log(theoretical_fluctuation[0]) - \
        numpy.log(fluctuation[-1])

    # get back to 'real' value
    rdc = numpy.exp(rdc_log)

    return (numpy.vstack((roi_sizes, fluctuation)),
            numpy.vstack((roi_sizes, theoretical_fluctuation)),
            rdc,
            fluctuation[-1],
            max_roi_size)


def weisskoff_figure(fluctuations, theoretical_fluctuations, rdc,
                     max_fluctuation, max_roi_size, title=None):
    """ Return a matplotlib figure containing the Weisskoff analysis.

    Parameters
    ----------
    fluctuation: 2-uplet
        the roi sizes array and the fluctuation array.
    theoretical_fluctuation: 2-uplet
        the roi sizes array and the theoretical fluctuation array.
    """
    figure = plt.figure()
    plot = figure.add_subplot(111)
    plot.grid(True, which="both", ls="-")
    plot.axes.loglog()

    if title:
        plt.title("{0}\nWeisskoff analysis".format(title))
    else:
        plt.title("Weisskoff analysis")
    plot.plot(fluctuations[0, :], 100 * fluctuations[1, :], "ko-",
              fillstyle="full")
    plot.plot(theoretical_fluctuations[0, :],
              100 * theoretical_fluctuations[1, :],
              "ko-", markerfacecolor="w")

    ymin, ymax = plt.ylim()
    plot.plot([rdc, rdc], [ymin, 100 * max_fluctuation], "r-")
    plot.plot([rdc, max_roi_size],
              [100 * max_fluctuation, 100 * max_fluctuation], "r-")
    plt.ylim((ymin, ymax))
    plt.xlim((1, max_roi_size + 10))

    plot.text(rdc, ymin, 'rdc = {0}'.format(round(rdc, 2)),
              verticalalignment='bottom', horizontalalignment='right',
              color='red', fontsize=10)
#    plt.xticks(list(plt.xticks()[0]) + [rdc])
    plot.axes.set_xlabel("ROI width (pixels)")
    plot.axes.set_ylabel("Fluctuation (%)")
    plot.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    plot.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    plot.legend(("Measured", "Theoretical"), "upper right")
    return figure


def spectrum_figure(spectrum, title=None):
    """ Return a matplotlib figure containing the Fourier spectrum, without its
    DC coefficient.
    """
    figure = plt.figure()
    plot = figure.add_subplot(111)
    plot.grid()
    if title:
        plt.title("{0}\nResidual Spectrum".format(title))
    else:
        plt.title("Residual Spectrum")
    plot.plot(spectrum[0, 1:], spectrum[1, 1:], "k-")
    plot.axes.set_xlabel("Frequency (Hz)")
    plot.axes.set_ylabel("Magnitude")
    return figure


def time_series_figure(time_series, polynomial, drift, snr, title=None):
    """ Return a matplotlib figure containing the time series and its
    polynomial model.
    """
    figure = plt.figure()
    plot = figure.add_subplot(111)
    plot.grid()
    if title:
        plt.title("{2}\nDrift: {0: .1f}% - SNR: {1: .1f}dB".format(
            drift * 100, 10 * numpy.log10(snr), title))
    else:
        plt.title("Drift: {0: .1f}% - SNR: {1: .1f}dB".format(
            drift * 100, 10 * numpy.log10(snr)))
    x = numpy.arange(2, 2 + len(time_series))
    model = numpy.polyval(polynomial, x)
    plot.plot(x, time_series, "k-")
    plot.plot(x, model, "k-")
    plot.axes.set_xlabel("Volume number")
    plot.axes.set_ylabel("Intensity")
    return figure


def aggregate_results(snr_score, sfnr_score, spike_score, output_directory):
    """
    This function takes the dictionaries outputed by other processed and
    merge them into one general result json file
    NOTE: if 2 files share the same key, a value will be overwritten !
    <unit>
        <input name="snr_score" type="File" desc="the snr json file"/>
        <input name="sfnr_score" type="File" desc="the sfnr json file"/>
        <input name="spike_score" type="File" desc="the spike json file"/>
        <input name="output_directory" type="Directory" desc="The output
            dir containing the score file"/>
        <output name="scores_file" type="File" desc="All scores in a
            json file"/>
    </unit>
    """
    scores_file = os.path.join(output_directory, "scores.json")
    out = {}
    with open(snr_score, "r") as _file:
        temp = json.load(_file)
    out.update(temp)
    with open(sfnr_score, "r") as _file:
        temp = json.load(_file)
    out.update(temp)
    with open(spike_score, "r") as _file:
        temp = json.load(_file)
    out.update(temp)

    with open(scores_file, "w") as _file:
        json.dump(out, _file)

    return scores_file
