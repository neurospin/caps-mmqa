#! /usr/bin/env python
##########################################################################
# CAPS - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import sys
from PySide import QtGui
import logging
import datetime

logging.basicConfig(level=logging.INFO)

# CAPSUL import
from capsul.qt_gui.widgets import PipelineDevelopperView
from capsul.study_config.study_config import StudyConfig
from capsul.process.loader import get_process_instance

# CAPS import
from caps.toy_datasets import get_sample_data


# Configure the environment
start_time = datetime.datetime.now()
print "Start Configuration", start_time
study_config = StudyConfig(
    modules=["SmartCachingConfig"],
    use_smart_caching=True,
    output_directory="/volatile/nsap/catalogue/quality_assurance/")
print "Done in {0} seconds".format(datetime.datetime.now() - start_time)


# Create pipeline
start_time = datetime.datetime.now()
print "Start Pipeline Creation", start_time
pipeline = get_process_instance("mmqa.fmri.fmri_quality_assurance.xml")
print "Done in {0} seconds.".format(datetime.datetime.now() - start_time)


# Set pipeline input parameters
start_time = datetime.datetime.now()
print "Start Parametrization", start_time
localizer_dataset = get_sample_data("localizer")
pipeline.image_file = localizer_dataset.fmri
pipeline.repetition_time = localizer_dataset.TR
print "Done in {0} seconds.".format(datetime.datetime.now() - start_time)


# View pipeline
app = QtGui.QApplication(sys.argv)
view1 = PipelineDevelopperView(pipeline)
view1.show()
app.exec_()

# Execute the pipeline in the configured study
study_config.run(pipeline, executer_qc_nodes=True, verbose=1)
