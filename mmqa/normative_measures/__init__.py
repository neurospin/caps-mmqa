##########################################################################
# NSAP - Copyright (C) CEA, 2013 - 2016
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import os
import csv


abide_file = os.path.join(
    os.path.dirname(__file__), "abide_func.csv")
corr_file = os.path.join(
    os.path.dirname(__file__), "corr_func.csv")


scores = ["efc", "fber", "fwhm", "dvars", "outlier", "quality",
          "mean_fd", "num_fd", "perc_fd", "gsr"]


def abide():
    """ Parse the abide normative measures.
    """
    return parse_csv(abide_file)


def corr():
    """ Parse the corr normative measures.
    """
    return parse_csv(corr_file)


def parse_csv(csv_file):
    """ Parse a csv file.

    Parameters
    ----------
    csv_file: str
        a csv file.

    Returns
    -------
    struct: dict
        the parsed csv as a dictionary with column names as keys.
    """
    struct = dict((score_name, []) for score_name in scores
                  if score_name != "gsr")
    for dim in ["x", "y", "z"]:
        struct["ghost_{0}".format(dim)] = []
    with open(csv_file) as open_file:
        reference = csv.DictReader(open_file, delimiter=",")
        for dict_row in reference:
            for column_name, value in dict_row.items():
                score_name = column_name[5:]
                if score_name in scores and value not in ["NA"]:
                    if score_name == "gsr":
                        for dim in ["x", "y", "z"]:
                            struct["ghost_{0}".format(dim)].append(float(value))  
                    else:
                        struct[score_name].append(float(value))

    return struct    
