#! /usr/bin/env python
##########################################################################
# NSAp - Copyright (C) CEA, 2013-2016
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import math
import os
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import FigureCanvasPdf as FigureCanvas
import seaborn as sns

# Qap import
from qap.viz.plotting import plot_vline


def plot_measures(group_measures, ncols=4, title="Group level report",
                  subject_measures=None, figsize=(8.27, 11.69),
                  display_type="violin"):
    """ Normative measures

    Display the distribution of normative measures and a measured new value.

    Parameters
    ----------
    group_measure: dict
        a dictionary with score names as keys and a list of float values.
    ncols: int (optional, default 4)
        the number of plot per line.
    title: str (optional, default 'Group level report')
        the title of the plot.
    subject_measures: dict (optional, default None)
        the subject measures, if None display the distribution only.
    figsize: 2-uplet (optional, default (8.27, 11.69))
        the plot size.
    display_type: str (optional, default 'violin')
        the distribution rendering type, one of 'violin' or 'hist'.

    Returns
    -------
    fig
        a matplotlib figure.    
    """
    # Check display type
    if display_type not in ["hist", "violin"]:
        raise ValueError("Unexpected '{0}' display type.".format(display_type))

    # Create a grid plot
    header = group_measures.keys()
    nmeasures = len(header)
    nrows = nmeasures // ncols
    if nmeasures % ncols > 0:
        nrows += 1
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nrows, ncols)


    # Build the distribution representation and identify the subject measure
    # in this distribution
    axes = []
    for i, score_name in enumerate(header):
        axes.append(plt.subplot(gs[i]))
        axes[-1].set_xlabel(score_name)

        if display_type == "hist":
            sns.distplot(
                group_measures[score_name], ax=axes[-1], color="b", rug=True,
                norm_hist=True)
        else:
            sns.violinplot(
                group_measures[score_name], ax=axes[-1], orient="v",
                linewidth=1)

        if subject_measures is not None:
            if score_name in subject_measures:
                if display_type == "hist":
                    plot_vline(
                        subject_measures[score_name], "*", axes[-1])
                else:
                    axes[-1].plot(
                        [0], [subject_measures[score_name]], ms=9, mew=.8,
                        linestyle="None", color="w", marker="*",
                        markeredgecolor="k", zorder=10)

    # Adjust the figure
    fig.suptitle(title)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.subplots_adjust(top=0.85)

    return fig
