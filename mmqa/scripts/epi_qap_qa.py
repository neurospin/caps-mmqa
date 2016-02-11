#! /usr/bin/env python
##########################################################################
# NSAp - Copyright (C) CEA, 2013-2016
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System imports
from __future__ import print_function
import argparse
import os
import shutil
import nibabel
import numpy
import json


# Bredala import
try:
    import bredala
    bredala.USE_PROFILER = False
    bredala.register("clinfmri.quality_control.movement_quantity",
                     names=["time_serie_mq"])
    bredala.register("nipype.algorithms.misc", names=["TSNR.run"])
    bredala.register("qap.temporal_qc", names=["fd_jenkinson"])
    bredala.register("qap.qap_workflows_utils",
                     names=["qap_functional_spatial",
                            "qap_functional_temporal"])
    bredala.register("qap.spatial_qc", names=["snr", "cnr", "fber", "efc",
                                              "artifacts", "fwhm",
                                              "ghost_direction", "ghost_all",
                                              "summary_mask", "get_background",
                                              "check_datatype",
                                              "convert_negatives"])
    bredala.register("mmqa.fmri.fmri_spikes", names=["spike_detector"])
    bredala.register("mmqa.fmri.movement_quantity", names=["get_rigid_matrix"])
    bredala.register("qap.temporal_qc", names=["mean_dvars_wrapper",
                                               "mean_outlier_timepoints",
                                               "mean_quality_timepoints",
                                               "global_correlation"])
    bredala.register("qap.viz.plotting", names=["plot_mosaic", "plot_fd"])
    bredala.register("qap.viz.reports", names=["concat_pdf"])
    bredala.register("mmqa.plot.plotting", names=["plot_measuresf"])
except:
    pass

# Clinfmri imports
from clinfmri.quality_control.movement_quantity import time_serie_mq

# Nipype import
import nipype.algorithms.misc as nam

# Qap import
from qap.viz.plotting import plot_mosaic
from qap.viz.plotting import plot_fd
from qap.viz.reports import concat_pdf
from qap.temporal_qc import fd_jenkinson
from qap.qap_workflows_utils import qap_functional_spatial
from qap.qap_workflows_utils import qap_functional_temporal

# Mmqa import
from mmqa.normative_measures import abide
from mmqa.normative_measures import corr
from mmqa.fmri.fmri_spikes import spike_detector
from mmqa.fmri.movement_quantity import get_rigid_matrix
from mmqa.plot.plotting import plot_measures

# Script documentation
doc = """
EPI quality assurance pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Definition of quality assurance: a check that occurs before any
preprocessing/processing.

This scipt depends on two Python modules:
* QAP: collection of three quality assessment pipelines for anatomical MRI and
  functional MRI scans.
  https://github.com/preprocessed-connectomes-project/quality-assessment-protocol
* CLINFMRI: a tool dedicated to functional processings.
  https://github.com/neurospin/caps-clinfmri

Here is a brief description of the computed scores that regrouped in two
catagories.

Spatial QA metrics:
* Entropy Focus Criterion [efc]: Uses the Shannon entropy of voxel intensities
as an indication of ghosting and blurring induced by head motion.
Lower values are better:
Atkinson D, Hill DL, Stoyle PN, Summers PE, Keevil SF (1997). Automatic
correction of motion artifacts in magnetic resonance images using an entropy
focus criterion. IEEE Trans Med Imaging. 16(6):903-10.
* Foreground to Background Energy Ratio [fber]: Mean energy of image values
(i.e., mean of squares) within the head relative to outside the head. Higher
values are better.
* Smoothness of Voxels [fwhm, fwhm_x, fwhm_y, fwhm_z]: The full-width half
maximum (FWHM) of the spatial distribution of the image intensity values in
units of voxels. Lower values are better.
* Ghost to Signal Ratio (GSR) [ghost_x, ghost_y or ghost_z]: A measure of the
mean signal in the 'ghost' image (signal present outside the brain due to
acquisition in the phase encoding direction) relative to mean signal within
the brain. Lower values are better.
* Summary Measures [fg_mean, fg_std, fg_size, bg_mean, bg_std, bg_size]:
Intermediate measures used to calculate the metrics above. Mean, standard
deviation, and mask size are given for foreground and background masks.

Temporal QA metrics:
* Standardized DVARS [dvars]: The spatial standard deviation of the temporal
derivative of the data, normalized by the temporal standard deviation and
temporal autocorrelation. Lower values are better:
Power, J. D., Barnes, K. A., Snyder, A. Z., Schlaggar, B. L. and Petersen, S.
E. (2012) Spurious but systematic correlations in functional connectivity MRI
networks arise from subject motion. Neuroimage 59, 2142-2154
Nichols, T. (2012, Oct 28). Standardizing DVARS. Retrieved from
http://blogs.warwick.ac.uk/nichols/entry/standardizing_dvars
* Outlier Detection [outlier]: The mean fraction of outliers found in each
volume using the 3dToutcount command from AFNI. Lower values are better:
Cox, R.W. (1996) AFNI: Software for analysis and visualization of functional
magnetic resonance neuroimages. Computers and Biomedical Research, 29:162-173
* Median Distance Index [quality]: The mean distance (1-spearman's rho)
between each time-point's volume and the median volume using AFNI's 3dTqual
command. Lower values are better 7.
* Mean Fractional Displacement - Jenkinson [mean_fd]: A measure of subject
head motion, which compares the motion between the current and previous
volumes. This is calculated by summing the absolute value of displacement
changes in the x, y and z directions and rotational changes about those
three axes. The rotational changes are given distance values based on the
changes across the surface of a 80mm radius sphere. Lower values are better:
Jenkinson, M., Bannister, P., Brady, M., and Smith, S. (2002). Improved
optimization for the robust and accurate linear registration and motion
correction of brain images. Neuroimage, 17(2), 825-841
Yan CG, Cheung B, Kelly C, Colcombe S, Craddock RC, Di Martino A, Li Q, Zuo XN,
Castellanos FX, Milham MP (2013). A comprehensive assessment of regional
variation in the impact of head micromovements on functional connectomics.
Neuroimage. 76:183-201
* Number of volumes with FD greater than 0.2mm [num_fd]: Lower values
are better.
* Percent of volumes with FD greater than 0.2mm [perc_fd]: Lower values are
better.
* Spike detection: home made.
* Movement quantity: home made.

The QA results are given with the ABIDE (1,110+ subject across 20+ sites) and
CoRR (1,400+ subjects across 30+ sites) normative metrics.

Command:

python $HOME/git/caps-mmqa/mmqa/scripts/epi_qap_qa.py \
    -v 2 \
    -e \
    -o /volatile/nsap/qap \
    -s mysid \
    -m /volatile/nsap/catalogue/pclinfmri/fmri_preproc_spm_fmri/realign/meanaufmri_localizer.nii \
    -a /volatile/nsap/catalogue/pclinfmri/fmri_preproc_spm_fmri/bet.fsl_bet/fmri_localizer_brain_mask.nii.gz \
    -f /volatile/nsap/catalogue/pclinfmri/fmri_preproc_spm_fmri/ungzip_adapter/ufmri_localizer.nii \
    -t /volatile/nsap/catalogue/pclinfmri/fmri_preproc_spm_fmri/realign/rp_aufmri_localizer.txt \
    -r /volatile/nsap/catalogue/pclinfmri/fmri_preproc_spm_fmri/realign/raufmri_localizer.nii \
    -d all


Local multi-processing:

from hopla import hopla
import os
myhome = os.environ["HOME"]
status, exitcodes = hopla(
    os.path.join(myhome, "git", "caps-clindmri", "clindmri", "scripts",
                 "freesurfer_conversion.py"),
    c="/i2bm/local/freesurfer/SetUpFreeSurfer.sh",
    d="/volatile/imagen/dmritest/freesurfer",
    s=["000043561374", "000085724167", "000052904972"],
    e=True,
    hopla_iterative_kwargs=["s"],
    hopla_cpus=3,
    hopla_logfile="/volatile/imagen/dmritest/freesurfer/conversion.log",
    hopla_verbose=1)

Cluster multi-processing:

from hopla.converter import hopla
import hopla.demo as demo
import os

apath = os.path.abspath(os.path.dirname(demo.__file__))
script = os.path.join(os.path.dirname(demo.__file__),
                      "my_ls_script.py")
status, exitcodes = hopla(
    script, hopla_iterative_kwargs=["d"], d=[apath, apath], v=1,
    hopla_verbose=1, hopla_cpus=2, hopla_cluster=True,
    hopla_cluster_logdir="/home/ag239446/test/log",
    hopla_cluster_python_cmd="/usr/bin/python2.7",
    hopla_cluster_queue="Cati_LowPrio")
"""


def is_file(filearg):
    """ Type for argparse - checks that file exists but does not open.
    """
    if not os.path.isfile(filearg):
        raise argparse.ArgumentError(
            "The file '{0}' does not exist!".format(filearg))
    return filearg


def is_directory(dirarg):
    """ Type for argparse - checks that directory exists.
    """
    if not os.path.isdir(dirarg):
        raise argparse.ArgumentError(
            "The directory '{0}' does not exist!".format(dirarg))
    return dirarg


parser = argparse.ArgumentParser(description=doc)
parser.add_argument(
    "-v", "--verbose", dest="verbose", type=int, choices=[0, 1, 2], default=0,
    help="increase the verbosity level: 0 silent, [1, 2] verbose.")
parser.add_argument(
    "-e", "--erase", dest="erase", action="store_true",
    help="if activated, clean the result folder.")
parser.add_argument(
    "-o", "--outdir", dest="outdir", required=True, metavar="PATH",
    help="the destination output directory: will create an extra subject"
         "level",
    type=is_directory)
parser.add_argument(
    "-s", "--subjectid", dest="subjectid", required=True,
    help="the subject identifier.")
parser.add_argument(
    "-m", "--meanepi", dest="meanepi", required=True, metavar="FILE",
    help="the mean epi volume after slice time correction and realignement.",
    type=is_file)
parser.add_argument(
    "-g", "--wrappedmean", dest="wrappedmean", required=True, metavar="FILE",
    help=("the mean wrapped epi volume after slice time correction and"
          " realignement."),
    type=is_file)
parser.add_argument(
    "-f", "--func", dest="func", required=True, metavar="FILE",
    help="the functional serie.", type=is_file)
parser.add_argument(
    "-a", "--funcmask", dest="funcmask", required=True, metavar="FILE",
    help="the template brain mask.", type=is_file)
parser.add_argument(
    "-t", "--transformations", dest="transformations", required=True,
    metavar="FILE",
    help="the realignment parameters file.", type=is_file)
parser.add_argument(
    "-r", "--funcrealign", dest="funcrealign", required=True,
    metavar="FILE",
    help="the functional realignes serie.", type=is_file)
parser.add_argument(
    "-d", "--direction", dest="direction", choices=["x", "y", "z", "all"],
    default="y",
    help="phase encoding (x - RL/LR, y - AP/PA, z - SI/IS, or "
    "all) used to acquire the scan.", type=str)
parser.add_argument(
    "-c", "--crop_volume", dest="crop", type=int, default=0,
    help=("set how many of the first volumes have been cut out"
          " for preprocessing"))
parser.add_argument(
    "-b", "--basemask", dest="basemask", required=True,
    metavar="FILE",
    help="the reference template mask for normalization",
    type=is_file)
parser.add_argument(
    "-w", "--meanwrappedmask", dest="meanwrappedmask", required=True,
    metavar="FILE",
    help="the binary mask of the wrapped mean volume",
    type=is_file)


args = parser.parse_args()


"""
Create first the subject directory properly and display some information
depending of the verbosity.
"""
if args.verbose > 0:
    print("#" * 20)
    print("Computing spatio temporal epi QA")
    print("#" * 20)
    print("Mean EPI: ", args.meanepi)
    print("Functional mask: ", args.funcmask)
    print("Phase encoding direction: ", args.direction)
figures = []
subjectdir = os.path.join(args.outdir, args.subjectid)
if not os.path.isdir(subjectdir):
    os.mkdir(subjectdir)
if args.erase:
    shutil.rmtree(subjectdir)
    os.mkdir(subjectdir)

"""
QAP spatial
"""

# out_vox: output the FWHM as # of voxels (otherwise as mm)
# direction: used to compute signal present outside the brain due to
#            acquisition in the phase encoding direction

# > compute functional spatial scores from QAP library
qc = qap_functional_spatial(args.meanepi, args.funcmask, args.direction,
                            args.subjectid, "mysession", "myscan",
                            site_name="mysite", out_vox=True)

# > save scores as a Json file
scores_json = os.path.join(subjectdir, "qap_functional_spatial.json")
qc.pop("session")
qc.pop("scan")
qc.pop("site")
for key, value in qc.items():
    if isinstance(value, numpy.double) or isinstance(value, numpy.single):
        qc[key] = float(value)
    if isinstance(value, numpy.core.memmap):
        if value.dtype == numpy.single or value.dtype == numpy.double:
            qc[key] = float(value)
        elif value.dtype == numpy.int:
            qc[key] = int(value)
        else:
            raise ValueError("Unexpected value type '{0}:{1}'".format(key,
                                                                      value))
with open(scores_json, "w") as open_file:
    json.dump(qc, open_file, indent=4)

# Control FOV
mean_snap = os.path.join(subjectdir, "mean_epi.pdf")
figures.append(mean_snap)
fig = plot_mosaic(args.meanepi, title="Mean EPI")
fig.savefig(mean_snap, dpi=300)


# Control wrapping on template
mean_snap_norm = os.path.join(subjectdir, "mean_normalized_epi_masked.pdf")
figures.append(mean_snap_norm)
fig = plot_mosaic(args.wrappedmean,
                  title="Mean wrapped EPI",
                  overlay_mask=args.basemask)
fig.savefig(mean_snap_norm, dpi=300)
# compute overlapping score between template masn and mean mask
# load masks
subject_mask_data = nibabel.load(args.meanwrappedmask).get_data()
template_mask_data = nibabel.load(args.basemask).get_data()

intersect = subject_mask_data + template_mask_data

intersect = 2 * float((intersect == 2).sum()) / \
    (2 * (intersect == 2).sum() + (intersect == 1).sum())

# final step: save scores in dict
scores = {"intersection_score": "{0}".format(round(
    100 * float(intersect), 2))}

scores_json = os.path.join(subjectdir, "extra_scores.json")
with open(scores_json, "w") as open_file:
    json.dump(scores, open_file, indent=4)

"""
QAP temporal
"""
# > Jenkinson Frame Displacement (FD) (Jenkinson et al., 2002)
fd_file = os.path.join(subjectdir, "fd.txt")
r12_file = os.path.join(
    subjectdir, "r12_" + os.path.basename(args.transformations))
rparams = numpy.loadtxt(args.transformations)
r12 = []
for rigid_params in rparams:
    r12.append(get_rigid_matrix(rigid_params, "SPM")[:-1].ravel())
r12 = numpy.asarray(r12)
numpy.savetxt(r12_file, r12)
fd_jenkinson(r12_file, rmax=80., out_file=fd_file)

# > computes the time-course SNR for a time series,
# typically you want to run this on a realigned time-series.
funcrealign_file = os.path.join(
    subjectdir, "nonan_" + os.path.basename(args.funcrealign))
im = nibabel.load(args.funcrealign)
data_array = im.get_data()
data_array[numpy.isnan(data_array)] = 0
nibabel.save(im, funcrealign_file)
cwd = os.getcwd()
os.chdir(subjectdir)
tsnr = nam.TSNR()
tsnr.inputs.in_file = funcrealign_file
tsnr.run()
tsnr = tsnr.aggregate_outputs()
os.chdir(cwd)
for out_name in ["tsnr_file", "mean_file", "stddev_file"]:
    path = getattr(tsnr, out_name)
    tmp_path = os.path.join(subjectdir, "tmp_" + os.path.basename(path))
    shutil.copyfile(path, tmp_path)
    im = nibabel.load(tmp_path)
    data_array = im.get_data()
    data_array[numpy.isnan(data_array)] = 0
    nibabel.save(im, path)
    os.remove(tmp_path)


# > compute functional temporal scores from QAP library
qc = qap_functional_temporal(funcrealign_file, args.funcmask, tsnr.tsnr_file,
                             fd_file, args.subjectid, "mysession", "myscan",
                             site_name="mysite", motion_threshold=1.0)
# > compute snaps
tsnr_snap = os.path.join(subjectdir, "tsnr_volume.pdf")
figures.append(tsnr_snap)
fig = plot_mosaic(getattr(tsnr, "tsnr_file"), title="tSNR volume")
fig.savefig(tsnr_snap, dpi=300)
fd_snap = os.path.join(subjectdir, "plot_fd.pdf")
figures.append(fd_snap)
fig = plot_fd(fd_file, title="FD plot")
fig.savefig(fd_snap, dpi=300)

# > save scores as a CSV file
scores_json = os.path.join(subjectdir, "qap_functional_temporal.json")
qc.pop("session")
qc.pop("scan")
qc.pop("site")
for key, value in qc.items():
    if isinstance(value, numpy.double) or isinstance(value, numpy.single):
        qc[key] = float(value)
    if isinstance(value, numpy.core.memmap):
        if value.dtype == numpy.single or value.dtype == numpy.double:
            qc[key] = float(value)
        elif value.dtype == numpy.int:
            qc[key] = int(value)
        else:
            raise ValueError("Unexpected value type '{0}:{1}'".format(key,
                                                                      value))
with open(scores_json, "w") as open_file:
    json.dump(qc, open_file, indent=4)


"""
Normative measures

Reference group-level data are depicted as violin plots, with each of the
plots being a representation of the corresponding values from the column with
the same name in the normative group-level CSV file.

Parse the abidde normaitve measures and represent the new measure in this
distribution
"""
abide_struct = abide()
corr_struct = corr()
subj_struct = {}
for fname in ["qap_functional_spatial.json", "qap_functional_temporal.json"]:
    scores_json = os.path.join(subjectdir, fname)
    with open(scores_json) as data_file:
        subj_struct.update(json.load(data_file))
for fname, group_struct in [("abide_normative_measures.pdf", abide_struct),
                            ("corr_normative_measures.pdf", corr_struct)]:
    dist_snap = os.path.join(subjectdir, fname)
    figures.append(dist_snap)
    fig = plot_measures(
        group_struct, ncols=4, subject_measures=subj_struct,
        title="QC measures: " + fname.split("_")[0] + " " + args.subjectid,
        figsize=(8.27, 11.69), display_type="violin")
    fig.savefig(dist_snap, dpi=300)


"""
Movement quantity and spike detection
"""
snap_mvt, displacement_file = time_serie_mq(
    args.func, args.transformations, "SPM", subjectdir, time_axis=-1,
    slice_axis=-2, mvt_thr=1.5, rot_thr=0.5, volumes_to_ignore=args.crop)
figures.append(snap_mvt)
snap_spike, spikes_file = spike_detector(args.func, subjectdir)
figures.append(snap_spike)


"""
Create a report
"""
report_snap = os.path.join(subjectdir, "report_" + args.subjectid + ".pdf")
concat_pdf(figures, out_file=report_snap)
