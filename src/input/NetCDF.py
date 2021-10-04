# BEGIN GPL LICENSE BLOCK
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# END GPL LICENSE BLOCK

"""
File that contains the functions to read files in NetCDF4 format.
"""
from typing import Union

import numpy as np
from netCDF4 import Dataset

from src.error.netcdf_import_error import NetCDFImportError
from src.utils import HEIGHT_KEYS, LATITUDE_KEYS, LONGITUDE_KEYS, get_logger

log = get_logger(module='NETCDF')


def get_variables_from_grp(grp, key_values: list) -> Union[list, None]:
    """
    Get the list of values from a grp object.

    Args:
        key_values: Possible key values to search for
        grp: Grp object generated by the netcdf4 library

    Returns: List with values
    """
    grp_keys = grp.variables.keys()

    # Check for the key to be in the object
    for key in key_values:
        if key in grp_keys:
            return grp.variables[key]

    return None


def read_info(file_name: str) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Extract the information of X, Y and Z from a NetCDF4 file.

    The keys to use for extracting the information of the longitude, latitude and height are stored in the files
    longitude_keys.json, latitude_keys.json and height_keys.json respectively.

    In case that the file  does not have a key to extract either the longitude, latitude or height, then an
    NetCDFImportError is raised proportioning as data a dictionary with the keys accepted as the parameter and the
    keys stored in the read file.

    Important:
        Due to how some files are generated, if the longitude or latitude keys are missing, then the method ask
        for the parameters x_range, y_range and spacing in the file. if they are defined in the file, then the program
        generates the X and Y values from the values defined in the parameters. If they are not defined, then the
        NetCDFImportError exception is raised with the information of the keys accepted and the keys that are inside the
        file.

    Return a tuple with 3 elements in the format (X, Y, Z):
        X: 1-dimensional array.
        Y: 1-dimensional array.
        Z: 2-dimensional array.

    Args:
        file_name (str): Filename to analyze.

    Returns: 
        Tuple with the values of the variables X, Y and Z in the file.
    """
    root_grp = Dataset(file_name, "r", format="NETCDF4")

    x = get_longitude_list_from_file(root_grp)
    y = get_latitude_list_from_file(root_grp)
    z = get_height_list_from_file(root_grp)

    # If the Z variable is defined as unidimensional array, then it is necessary to flip the contents of the array once
    # it is converted to a 2D matrix since the order of the y-axis is inverted.
    if z.ndim == 1:
        log.debug("Height of file is unidimensional.")
        z = np.array(z)
        z = z.reshape((len(y), len(x)))
        z = np.flipud(z)

    # Convert the variables to arrays to return
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # Change the order of the arrays if they are not sorted with ascending values
    # ---------------------------------------------------------------------------
    x_is_descending = x[0] > x[-1]
    y_is_descending = y[0] > y[-1]

    x = x if not x_is_descending else np.flip(x)
    y = y if not y_is_descending else np.flip(y)

    z = np.flip(z, 0) if y_is_descending else z
    z = np.flip(z, 1) if x_is_descending else z

    log.debug(f"Where X values descending: {x_is_descending}")
    log.debug(f"Where Y values descending: {y_is_descending}")
    log.debug(f"X values: {x}")
    log.debug(f"Y values: {y}")

    # Close the file
    root_grp.close()

    return x, y, z


def get_height_list_from_file(root_grp):
    """
    Get the heights values from the Dataset as it is stored in the file.

    This method gets the heights data stored in a netCDF4.Dataset object. The data retrieved is the same as it is
    stored in the file, in the same order that the elements are defined in its interior and with the same shape.

    Args:
        root_grp: Dataset object from the class netcdf4.Dataset.

    Returns: List with the latitude values defined in the Dataset object.
    """
    z = get_variables_from_grp(root_grp, HEIGHT_KEYS)
    # Variable Z must be defined as an array on the netcdf files.
    # Raise error if it is not defined.
    if z is None:
        raise NetCDFImportError(4, {'accepted_keys': HEIGHT_KEYS,
                                    'file_keys': root_grp.variables.keys()})
    return z


def get_latitude_list_from_file(root_grp) -> list:
    """
    Get the latitude values from the Dataset as it is stored in the file.

    This method gets the latitude data stored in a netCDF4.Dataset object. The data retrieved is the same as it is
    stored in the file, in the same order that the elements are defined in its interior.

    Args:
        root_grp: Dataset object from the class netcdf4.Dataset.

    Returns: List with the latitude values defined in the Dataset object.
    """

    y = get_variables_from_grp(root_grp, LATITUDE_KEYS)
    # Ask if the file have the values for the y values defined as ranges, spacing and dimensions.
    # Raise error if it is not defined as ranges/spacing/dimensions.
    if y is None:
        y_range_values = get_variables_from_grp(root_grp, ['y_range'])
        y_range_array = np.array(y_range_values)
        if y_range_values is None or len(y_range_array) < 2:
            raise NetCDFImportError(2, {'accepted_keys': LATITUDE_KEYS,
                                        'file_keys': root_grp.variables.keys()})

        spacing_values = get_variables_from_grp(root_grp, ['spacing'])
        spacing_array = np.array(spacing_values)
        if spacing_values is None or len(spacing_array) < 2:
            raise NetCDFImportError(2, {'accepted_keys': LATITUDE_KEYS,
                                        'file_keys': root_grp.variables.keys()})

        dimension_values = get_variables_from_grp(root_grp, ['dimension'])
        dimension_array = np.array(dimension_values)
        if dimension_values is None or len(dimension_array) < 2:
            raise NetCDFImportError(2, {'accepted_keys': LATITUDE_KEYS,
                                        'file_keys': root_grp.variables.keys()})

        y = np.arange(y_range_array[0], y_range_array[1], spacing_array[1]).tolist()
        y = [y] if type(y) is not list else y

        if len(y) + 1 == dimension_array[1]:
            y += [y_range_array[1]]
    return y


def get_longitude_list_from_file(root_grp) -> list:
    """
    Get the longitude values from the Dataset as it is stored in the file.

    This method gets the longitude data stored in a netCDF4.Dataset object. The data retrieved is the same as it is
    stored in the file, in the same order that the elements are defined in its interior.

    Args:
        root_grp: Dataset object from the class netcdf4.Dataset.

    Returns: List with the latitude values defined in the Dataset object.
    """
    # Ask if the arrays that contains the information related to the arrays of the map is defined on the netcdf file
    # with the names that are recognized by the program.
    x = get_variables_from_grp(root_grp, LONGITUDE_KEYS)

    # Ask if the file have the values for the x values defined as ranges, spacing and dimensions.
    # Raise error if it is not defined as ranges/spacing/dimensions
    if x is None:
        x_range_values = get_variables_from_grp(root_grp, ['x_range'])
        x_range_array = np.array(x_range_values)
        if x_range_values is None or len(x_range_array) < 2:
            raise NetCDFImportError(3, {'accepted_keys': LONGITUDE_KEYS,
                                        'file_keys': root_grp.variables.keys()})

        spacing_values = get_variables_from_grp(root_grp, ['spacing'])
        spacing_array = np.array(spacing_values)
        if spacing_values is None or len(spacing_array) < 2:
            raise NetCDFImportError(3, {'accepted_keys': LONGITUDE_KEYS,
                                        'file_keys': root_grp.variables.keys()})

        dimension_values = get_variables_from_grp(root_grp, ['dimension'])
        dimension_array = np.array(dimension_values)
        if dimension_values is None or len(dimension_array) < 2:
            raise NetCDFImportError(3, {'accepted_keys': LONGITUDE_KEYS,
                                        'file_keys': root_grp.variables.keys()})

        # Generate the x-values given the ranges and spacing.
        x = np.arange(x_range_array[0], x_range_array[1], spacing_array[0]).tolist()
        x = [x] if type(x) is not list else x

        # Add the last value of the range to the list if the x-value list is one short than the specified in the
        # dimensions
        if len(x) + 1 == dimension_array[0]:
            x += [x_range_array[1]]

    return x
