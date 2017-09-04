#! /usr/bin/env python
##########################################################################
# NSAp - Copyright (C) CEA, 2015
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import os
import sys
import unittest
import tempfile
import json


def pilot_qa_fmri():
    """
    Imports
    -------

    This code needs 'capsul' and 'mmutils' package in order to instanciate and
    execute the pipeline and to get a toy dataset.
    These packages are available in the 'neurospin' source list or in pypi.
    """
    # Capsul import
    from capsul.study_config.study_config import StudyConfig
    from capsul.process.loader import get_process_instance

    # Mmutils import
    from mmutils.toy_datasets import get_sample_data

    """
    Parameters
    ----------

    The 'pipeline_name' parameter contains the location of the pipeline XML
    description that will perform the DICOMs conversion, and the 'outdir' the
    location of the pipeline's results: in this case a temporary directory.
    """

    pipeline_name = "mmqa.fmri.fmri_quality_assurance_bbox.xml"
    outdir = tempfile.mkdtemp()

    """
    Capsul configuration
    --------------------

    A 'StudyConfig' has to be instantiated in order to execute the pipeline
    properly. It enables us to define the results directory through the
    'output_directory' attribute, the number of CPUs to be used through the
    'number_of_cpus' attributes, and to specify that we want a log of the
    processing step through the 'generate_logging'. The 'use_scheduler'
    must be set to True if more than 1 CPU is used.
    """
    study_config = StudyConfig(
        number_of_cpus=1,
        generate_logging=True,
        use_scheduler=True,
        output_directory=outdir)
    """
    Get the toy dataset
    -------------------

    The toy dataset is composed of a functional image that is downloaded
    if it is necessary throught the 'get_sample_data' function and exported
    locally.
    """

    localizer_dataset = get_sample_data("localizer_extra")

    """
    Pipeline definition
    -------------------

    The pipeline XML description is first imported throught the
    'get_process_instance' method, and the resulting pipeline instance is
    parametrized: in this example we decided to set the date in the converted
    file name and we set two DICOM directories to be converted in Nifti
    format.
    """

    pipeline = get_process_instance(pipeline_name)
    pipeline.image_file = localizer_dataset.fmri
    pipeline.repetition_time = 2.0
    pipeline.exclude_volume = []
    pipeline.roi_size = 21
    pipeline.score_file = os.path.join(outdir, "scores.json")

    """
    Pipeline representation
    -----------------------

    By executing this block of code, a pipeline representation can be
    displayed. This representation is composed of boxes connected to each
    other.
    """
    if 0:
        from capsul.qt_gui.widgets import PipelineDevelopperView
        from PySide import QtGui
        app = QtGui.QApplication(sys.argv)
        view1 = PipelineDevelopperView(pipeline)
        view1.show()
        app.exec_()

    """
    Pipeline execution
    ------------------

    Finally the pipeline is eecuted in the defined 'study_config'.
    """
    study_config.run(pipeline)

    """
    Access the result
    -----------------

    Display the computed scores
    """

    scores_file = pipeline.scores_file

    with open(scores_file, "r") as _file:
        scores = json.load(_file)

    for key, value in scores.iteritems():
        print "{0} = {1}".format(key, value)


class TestQAFmri(unittest.TestCase):
    """ Class to test dicom to nifti pipeline.
    """
    def test_qa_fmri(self):
        """ Method to test a simple 1 cpu call with the scheduler.
        """
        pilot_qa_fmri()


def test():
    """ Function to execute unitest
    """
    suite = unittest.TestLoader().loadTestsFromTestCase(TestQAFmri)
    runtime = unittest.TextTestRunner(verbosity=2).run(suite)
    return runtime.wasSuccessful()


if __name__ == "__main__":
    test()
