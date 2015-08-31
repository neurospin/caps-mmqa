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

# Capsul import
from capsul.study_config.study_config import StudyConfig
from capsul.process.loader import get_process_instance

# Mmutils import
# from mmutils.toy_datasets import get_sample_data


class TestFmriQA(unittest.TestCase):
    """ Class to test dicom to nifti pipeline.
    """
    def setUp(self):
        self.outdir = tempfile.mkdtemp()
        self.pipeline_name = "mmqa.fmri.fmri_quality_assurance.xml"

    def test_simple_run(self):
        """ Method to test a simple 1 cpu call with the scheduler.
        """
        # Configure the environment
        study_config = StudyConfig(
            modules=[],
            use_smart_caching=True,
            number_of_cpus=1,
            generate_logging=True,
            use_fsl=True,
            output_directory=self.outdir,
            use_scheduler=True)

        # Create pipeline
        pipeline = get_process_instance(self.pipeline_name)

        # Set pipeline input parameters
        # localizer_dataset = get_sample_data("localizer")
        pipeline.image_file = os.path.join(os.path.dirname(
                                               os.path.realpath(__file__)),
                                           "raw_fMRI_raw_bold.nii.gz")
        pipeline.repetition_time = 2400.
        pipeline.score_file = os.path.join(self.outdir, "scores.json")

        # View pipeline
        if 0:
            from capsul.qt_gui.widgets import PipelineDevelopperView
            from PySide import QtGui
            app = QtGui.QApplication(sys.argv)
            view1 = PipelineDevelopperView(pipeline)
            view1.show()
            app.exec_()

        # Execute the pipeline in the configured study
        # study_config.run(pipeline, executer_qc_nodes=True, verbose=1)


def test():
    """ Function to execute unitest
    """
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFmriQA)
    runtime = unittest.TextTestRunner(verbosity=2).run(suite)
    return runtime.wasSuccessful()


if __name__ == "__main__":
    test()
