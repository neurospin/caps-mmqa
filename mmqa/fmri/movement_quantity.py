#! /usr/bin/env python
##########################################################################
# Nsap - Neurospin - Berkeley - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import numpy
import nibabel


def time_serie_mq(image_file, realignment_parameters, package,
                  output_directory, time_axis=-1, slice_axis=-2):
    """ Movement quantity (MQ) - Description of the amount of motion in a
    temporal sequence.

    Raises
    ------
    ValueError: if the image dimension is different than 4 or the realignement
    parameters file is not valid.

    <unit>
        <input name="image_file" type="File" desc="A functional volume."/>
        <input name="spm_realignment_parameters" type="File" desc="Estimated
            spm translation and rotation parameters during the realignment."/>
        <input name="package" type="Str" desc="The package that generated
            the parameters, SPM or FSL."/>
        <input name="output_directory" type="Directory" desc="The destination
            folder."/>
        <input name="time_axis" type="Int" desc="Axis of the input array that
            varies over time. The default is the last axis."/>
        <input name="slice_axis" type="Int" desc="Axis of the array that
            varies over image slice. The default is the last non-time axis."/>
        <output name="snap_spikes" type="File" desc="A snap with the dectected
            spikes."/>
        <output name="spikes_file" type="File" desc="The detected spikes
            array."/>
    </unit>
    """
    # Load the image and get the associated numpy array
    image = nibabel.load(image_file)
    shape = image.get_shape()
    array = image.get_data()
    affine = image.get_affine()

    # Load realign parameters
    rparams = numpy.loadtxt(realignment_parameters)

    # Check the input shape
    if array.ndim != 4:
        raise ValueError("Time image dimension is '{0}', expect a 4d "
                         "volume.".format(array.ndim))

    # Check the realignment parameters
    if array.shape[time_axis] != rparams.shape[0]:
        raise ValueError("Realignement parameters in '{0}' are not "
                         "valid.".format(realignment_parameters))

    # Go through all volumes
    displacments = []
    for rigid_params in rparams:

        # Get the rigid transformation
        rigid_matrix = get_rigid_matrix(rigid_params, package)

        # Estimate the max displacment
        displacments.append(
            movement_quantity(rigid_matrix, shape, affine, metric="max")[0])

    print displacments


def movement_quantity(rigid, shape, affine, metric="max"):
    """ Movement quantity (MQ) - Description of the amount of motion.

    Parameters
    ----------
    rigid: array (4, 4)
        a rigid transformation.
    shape: array (3,)
        the image shape.
    affine: array (4, 4)
        the transformation matrix.
    metric: str (optional, default 'max')
        the output metric type: 'max', 'mean' or 'stat' (mean and std).

    Returns
    -------
    mq: tuple
        the required global displacement metric.
    """
    # Compute the local displacement
    displacement = deformation_field(rigid, shape, affine)

    # Compute the norm of each displacement
    norm_displacement = numpy.sqrt(
        numpy.sum(displacement[..., i]**2 for i in range(3)))

    # Return the requested metric
    if metric == "max":
        return (numpy.max(norm_displacement), )
    elif metric == "mean":
        return (numpy.mean(norm_displacement), )
    elif metric == "stat":
        return (numpy.mean(norm_displacement), numpy.std(norm_displacement))
    else:
        raise ValueError("Unkown metric.")


def deformation_field(rigid, shape, affine):
    """ Evaluate the deformation field associated to a rigid transform.

    Parameters
    ----------
    rigid: array (4, 4)
        a rigid transformation.
    shape: array (3,)
        the image shape.
    affine: array (4, 4)
        the transformation matrix.

    Returns
    -------
    deformation_field: array (shape, 3)
        the deformation field associated with the rigid transformation.
    """
    # Go through all the image voxels
    x = numpy.arange(0, shape[0], 1)
    y = numpy.arange(0, shape[1], 1)
    z = numpy.arange(0, shape[2], 1)
    mesh = numpy.meshgrid(x, y, z)
    for item in mesh:
        item.shape += (1, )
    mesh = numpy.concatenate(mesh, axis=3)

    # Apply the rigid transform
    points = field_dot(affine[:3, :3], mesh)
    points = field_add(points, affine[:3, 3])
    wrap_points = field_add(field_dot(rigid[:3, :3], points), rigid[:3, 3])
    deformation_field = wrap_points - points

    return deformation_field


def field_dot(matrix, field):
    """ Dot product between a rotation matrix and a field of 3d vectors.

    Parameters
    ----------
    matrix: array (3, 3)
        a rotation matrix.
    field: array (x, y, z, 3)
        an image of vectors to rotate.

    Returns
    -------
    dot: array (x, y, z, 3)
        the rotated field.
    """
    dot = numpy.zeros(field.shape, dtype=numpy.single)
    dot[..., 0] = numpy.sum(matrix[0, i] * field[..., i] for i in range(3))
    dot[..., 1] = numpy.sum(matrix[1, i] * field[..., i] for i in range(3))
    dot[..., 2] = numpy.sum(matrix[2, i] * field[..., i] for i in range(3))
    return dot


def field_add(field, vector):
    """ Add the vector to the field.

    Parameters
    ----------
    field: array (x, y, z, 3)
        an image of vectors.
    vector: array (3, )
        the vector that will be added to the field.

    Returns
    -------
    field: array (x, y, z, 3)
        the incremented image of vectors.
    """
    field[..., 0] += vector[0]
    field[..., 1] += vector[1]
    field[..., 2] += vector[2]
    return field


def get_rigid_matrix(rigid_params, package):
    """ Return rigid matrix given a set of translation and rotation parameters.

    In FSL order of fields is three angles (pitch - alpha, roll - beta, yaw -
    gamma in radians) then three translation params (x,y,z in mm).
    In SPM order of fields is three translation params (x,y,z in mm) then three
    angles (pitch - alpha, roll - beta, yaw - gamma in radians).

    Parameters
    ----------
    rigid_params: array [6,]
        the affine matrix coefficients.
    package: str
        the package that generated the parameters, SPM or FSL.

    Returns
    -------
    rigid: array [4, 4]
        a rigid transformation.

    Raises
    ------
    ValueError: if the srcpckg is not valid.
    """
    # Check if a valide package has been specified
    if package not in ["FSL", "SPM"]:
        raise ValueError("Uknown package '{0}'.".format(package))

    # FSL reorganization
    if package == "FSL":
        rigid_params = rigid_params[3, 4, 5, 0, 1, 2]

    # Get the translation
    T = rigid_params[0:3]

    # Get the rotation part from Euler description
    """
    Given 3 Euler angles x, y, z, the rotation matrix is calculated as follows:
           |1      0         0    |
     Rx =  |0     cos(x)   -sin(x)|
           |0     sin(x)   cos(x) |

           |cos(y)   0      sin(y)|
     Ry =  |0        1        0   |
           |-sin(y)  0      cos(y)|

           |cos(z)   -sin(z)    0 |
     Rz =  |sin(z)    cos(z)    0 |
           |0         0         1 |

     R = Rz.Ry.Rx
    """

    R = numpy.eye(3)
    Rx = numpy.eye(3)
    Rx[1:3, 1:3] = rotfunc1(rigid_params[5])
    Ry = numpy.eye(3)
    Ry[(0, 0, 2, 2), (0, 2, 0, 2)] = rotfunc2(rigid_params[4]).ravel()
    Rz = numpy.eye(3)
    Rz[0:2, 0:2] = rotfunc1(rigid_params[3])
    R = numpy.dot(Rz, numpy.dot(Ry, Rx))

    # Create the rigid transformation
    rigid = numpy.eye(4)
    rigid[:3, :3] = R
    rigid[:3, 3] = T

    return rigid


def rotfunc1(x):
    return numpy.array([[numpy.cos(x), -numpy.sin(x)],
                       [numpy.sin(x), numpy.cos(x)]])


def rotfunc2(x):
    return numpy.array([[numpy.cos(x), numpy.sin(x)],
                       [-numpy.sin(x), numpy.cos(x)]])


if __name__ == "__main__":

    # Z-rotation of alpha + translation
    alpha = numpy.pi / 2
    trans = [0, 0, 0]
    rigid = numpy.array([
        [numpy.cos(alpha), -numpy.sin(alpha), 0, trans[0]],
        [numpy.sin(alpha), numpy.cos(alpha), 0, trans[1]],
        [0, 0, 1, trans[2]],
        [0, 0, 0, 1]
    ], dtype=numpy.single)

    # Compute the dispalcement
    dispalcement = deformation_field(rigid, (2, 2, 2), numpy.eye(4))
    print dispalcement

    # Compute the mq
    mq = movement_quantity(rigid, (2, 2, 2), numpy.eye(4), "stat")
    print mq

    # Real data test
    from caps.toy_datasets import get_sample_data
    import logging

    logging.basicConfig(level=logging.INFO)

    localizer_dataset = get_sample_data("localizer")
    time_serie_mq(localizer_dataset.fmri,
                  localizer_dataset.mouvment_parameters,
                  "SPM", "/volatile/nsap/catalogue/quality_assurance/",
                  time_axis=-1, slice_axis=-2)
