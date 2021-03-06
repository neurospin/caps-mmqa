# -*- coding: utf-8 -*-

# get QA results and generate raw statistics images

# General imports
import json
import pylab as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
import datetime
import csv


def plot_results(file_list, selection, output_directory, groups=None,
                 n_bins=10):
    """
    Plot the results of a QA process.

    <unit>
        <input name="file_list" type="List" content="File" desc="The list of
            the result's file paths"/>
        <input name="selection" type="List" content="Str" desc="The list of
            scores to be plotted as they are defined in subject-related
            jsons"/>
        <input name="output_directory" type="Directory" desc="The directory
        that will contain the images and the aggregated csv file"/>
        <input name="groups" type="List" content="Str" desc="The list of group
            names corresponding to the filepath in file_list (same order,
            same length)
        (optional)" optional="True"/>
        <input name="n_bins" type="Int" desc="Number of bins of the histograms
        (optional)" optional="True"/>
        <output name="output_dir" type="Directory" desc="The outpout
            directory"/>
    </unit>

    """
    # create a batch file of results
    results_file = os.path.join(output_directory, "results.csv")
    batch_file = aggregate_results(file_list, results_file, groups)

    # parce the aggregated result files and compute images
    plot_images(batch_file, selection, output_directory, n_bins=n_bins)

    output_dir = output_directory
    return output_dir


def aggregate_results(file_list, out_file, groups=None):
    """
    Aggregator of result files generated by a QA process.

    Parameters
    ----------
    Inputs:
        file_list (mandatory): a list of the subject-related result
            files that come from the QA process (json files)
        out_file (mandatory): the output file
        selection (mandatory): The list of result names to aggregate and plot
        subject_groups (optional): the definition of subgroups. Dictionary that
            states, for each result file, its corresponding group.
    outputs:
        out_file: the outcoming agregated file

    """
    # open results file
    csvfile = open(out_file, "w")

    # generate header
    if groups:
        fieldnames = ["Group", ]
    else:
        fieldnames = []

    # get the header field
    rdm_file = file_list[0]
    with open(rdm_file, "r") as _file:
        file_content = json.load(_file)
        fieldnames.extend(file_content.keys())
    # write the header
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # navigate the individual result files
    for index in range(len(file_list)):
        if groups:
            row_dict = {"Group": groups[index]}
        else:
            row_dict = {}
        with open(file_list[index], 'r') as _file:
            results = json.load(_file)
            row_dict.update(results)
        writer.writerow(row_dict)

    csvfile.close()
    return out_file


def plot_images(data_file, selection, out_dir, n_bins=10):
    """
    PLot diagram and mean/standard deviation graphs from an aggregated results
    files. One serie per groups (if any)
    """
    # generate a set of images per feature
    for cnt, feature in enumerate(selection):

        with open(data_file, "r") as csv_file:
            reader = csv.DictReader(csv_file)
            # generate a dictionary of sub-values
            group_dict = {}
            if "Group" not in reader.fieldnames:
                group_dict[feature] = []
            for row in reader:
                if "Group" in reader.fieldnames:
                    if row['Group'] in group_dict:
                        group_dict[row['Group']].append(float(row[feature]))
                    else:
                        group_dict[row['Group']] = []
                else:
                    group_dict[feature].append(float(row[feature]))
            # create the graphs
            mean_list = []
            std_list = []
            values_list = []
            # legend
            group_names = []
            # normalisation per subset
            weights = []
            for group, values in group_dict.iteritems():
                group_names.append("{0} ({1} values)".format(group,
                                                             len(values)))
                mean_list.append(np.mean(values))
                std_list.append(np.std(values))
                values_list.append(values)
                weights.append((1. / float(len(values)) *
                                np.ones(len(values))).tolist())

            plt.figure(cnt, figsize=(26, 14))
            plt.subplot(211)
            n, bins, patches = plt.hist(values_list, n_bins, histtype='bar',
                                        weights=weights, label=group_names)
            plt.xlabel(feature)
            plt.ylabel("Number of occurences (%)")
            plt.title("{0} score histogram ({1} bins)".format(feature, n_bins))
            plt.legend()

            # set the formatter
            formatter = FuncFormatter(to_percent)
            plt.gca().yaxis.set_major_formatter(formatter)

            # get axes to align subplots scales
            ax = plt.axis()

            _axes = plt.subplot(212)

            # plot the segment (standard deviation values)
            level = 0.5
            _min = None
            _max = float("inf")
            for mean, std in zip(mean_list, std_list):
                if mean + std < _max:
                    _max = mean + std
                if mean - std > _min:
                    _min = mean - std
                plt.plot([mean - std, mean + std],
                         [level, level],
                         linewidth=2.0)
                level += 1

            # reset color cycle
            plt.gca().set_color_cycle(None)

            # now plot the mean values
            level = 0.5
            y_ticks = [level]
            for mean in mean_list:
                plt.plot(mean, level, 's', linewidth=2.0)
                _axes.text(mean, level + 0.1, "{0}".format(round(mean, 2)))
                level += 1
                y_ticks.append(level)

            _axes.set_xlim(ax[:2])
            _axes.set_ylim([0, len(mean_list)])

            # plot vertical lines
            plt.plot([_min, _min], [0, level + 0.5], 'k--')
            plt.plot([_max, _max], [0, level + 0.5], 'k--')

            # change yticks
            _axes.set_yticks(y_ticks)
            _axes.set_yticklabels(group_dict.keys())

            # legend and text
            plt.title("{0} - means and standard deviation values".format(
                feature))
            plt.xlabel(feature)

            # save the figure
            plt.savefig(os.path.join(out_dir, '{0} - {1}.png'.format(
                datetime.date.today().strftime("%Y%m%d"),
                feature)), bbox_inches='tight')


def to_percent(y, position):
    # plot in percentage in the diagram instead of value from 0 to one
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if plt.rcParams['text.usetex']:
        return s + r'$\%$'
    else:
        return s + '%'
