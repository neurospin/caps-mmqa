# -*- coding: utf-8 -*-

# fMRI quality assurance pipeline.
# Occurs before any preprocessing/processing

# system imports
import os
import shutil
import nibabel
import numpy
import subprocess
import json
import traceback
import multiprocessing as mp

# library imports
from clinfmri.quality_control.movement_quantity import time_serie_mq
from qap.temporal_qc import (outlier_timepoints,
                             quality_timepoints)
from qap.spatial_qc import (efc,
                            fber,
                            fwhm,
                            ghost_all)

from mmqa.fmri.fmri_spikes import spike_detector

# poison pills
FLAG_ALL_DONE = b"WORK_FINISHED"
FLAG_WORKER_FINISHED_PROCESSING = b"WORKER_FINISHED_PROCESSING"


# WORKERS
def run_worker(inputs_queue, outputs_queue, index):

    # get something from the pile
    while True:
        inputs = inputs_queue.get()
        # stop condition
        if inputs == FLAG_ALL_DONE:
            outputs_queue.put(FLAG_WORKER_FINISHED_PROCESSING)
            break

        subj_id = inputs[0]
        fmri_file = inputs[1]
        rp_file = inputs[2]
        root_output = inputs[3]

        # define working directory
        working_directory = os.path.join(root_output,
                                         subj_id,
                                         "outputs")
        if os.path.isdir(working_directory):
                shutil.rmtree(working_directory)
        os.makedirs(working_directory)

        try:

            # get data array
            fmri_file_data = nibabel.load(fmri_file).get_data()

            # step 1: get movement snap and parameters
            snap_mvt, displacement_file = time_serie_mq(fmri_file,
                                                        rp_file,
                                                        "SPM",
                                                        working_directory,
                                                        time_axis=-1,
                                                        slice_axis=-2,
                                                        mvt_thr=1.5,
                                                        rot_thr=0.5)

            # step 2: get efc score (entropy focus criterion)
            r_efc = efc(fmri_file_data)

            # step 3: get masks from afni
            mask_file = os.path.join(working_directory, "mask.nii")
            cmd = ["3dAutomask", "-prefix", mask_file, fmri_file]
            subprocess.check_call(cmd)
            mask_data = nibabel.load(mask_file).get_data()

            # step 4: get fber score (foreground to background energy ratio)
            r_fber = fber(fmri_file_data, mask_data)

            #step 5: get smoothness of voxels score
            r_fwhm = fwhm(fmri_file, mask_file)

            # step 6: detect outlier timepoints in each volume
            outliers = outlier_timepoints(fmri_file, mask_file)
            mean_outliers = numpy.mean(outliers)

            # step 7: ghost scores
            gsrs = ghost_all(fmri_file_data, mask_data)

            # step 8: quality timepoints
            qt = quality_timepoints(fmri_file, automask=True)
            mean_qt = numpy.mean(qt)

            # step 9: spike detection
            snap_spikes, spikes_file = spike_detector(
                fmri_file, working_directory)

            with open(spikes_file) as _file:
                spikes_dict = json.load(_file)

            # final step: save scores in dict
            scores = {"efc": "{0}".format(r_efc),
                      "fber": "{0}".format(r_fber),
                      "fwhm": "{0}".format(r_fwhm),
                      "outliers": "{0}".format(outliers),
                      "mean_outliers": "{0}".format(mean_outliers),
                      "x_gsr": "{0}".format(gsrs[0]),
                      "y_gsr": "{0}".format(gsrs[1]),
                      "quality": "{0}".format(qt),
                      "mean_quality": "{0}".format(mean_qt)}

            scores.update(spikes_dict)

            scores_file = os.path.join(working_directory, "qa_scores.json")
            with open(scores_file, "w") as _file:
                json.dump(scores, _file, indent=4)

            outputs_queue.put("{0} - Success".format(subj_id))
        except:
            outputs_queue.put("{0} - FAIL:".format(subj_id))
            traceback.print_exc()


# Main script
if __name__ == '__main__':
    # get the data, set pathes
    # TODO data should be provided in a context of a project, via a parser...
    root_output = "/volatile/local_disk/QC_pipeline_WD"
    root_preproc_path = ("/volatile/local_disk/QC_pipeline_WD/000086100102/"
                         "000086100102_preproc/")
    fmri_file = os.path.join(root_preproc_path,
                             "BL_wea000086100102s006a001.nii.gz")
    subj_id = "000086100102"
    rp_file = os.path.join(root_preproc_path,
                           "BL_rp_a000086100102s006a001_model1.txt")

    # XXX MULTIPROCESSING
    n_process = 1

    # define queues
    manager = mp.Manager()
    input_queue = manager.Queue()
    output_queue = manager.Queue()

    # fill input queue
    input_queue.put([subj_id,
                     fmri_file,
                     rp_file,
                     root_output])

    # add poison pills
    for _ in range(n_process):
        input_queue.put(FLAG_ALL_DONE)

    processes = []

    # create processes
    for proc_number in range(n_process):
        p = mp.Process(target=run_worker,
                       args=(input_queue, output_queue, proc_number + 1))
        p.daemon = True
        processes.append(p)
        p.start()

    # get output
    proc_finished = 0
    while True:
        results = output_queue.get()
        if results == FLAG_WORKER_FINISHED_PROCESSING:
            proc_finished += 1

            if proc_finished == n_process:
                break

        else:
            print results
