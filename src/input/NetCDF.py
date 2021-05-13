"""
File that contains the functions to read files in NetCDF4 format.
"""
from typing import Union
from src.utils import get_logger
from src.error.netcdf_import_error import NetCDFImportError

import numpy as np
from netCDF4 import Dataset

# Variables to use to check for data in the netcdf files
LONGITUDE_KEYS = ['x', 'lon']
LATITUDE_KEYS = ['y', 'lat']
HEIGHT_KEYS = ['z', 'Band1']

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

    Args:
        file_name (str): Filename to analyze.

    Returns:
        np.array, np.array, np.array: Values of the variables X, Y
                                      and Z in the file.
    """
    root_grp = Dataset(file_name, "r", format="NETCDF4")

    x = get_variables_from_grp(root_grp, LONGITUDE_KEYS)
    y = get_variables_from_grp(root_grp, LATITUDE_KEYS)
    z = get_variables_from_grp(root_grp, HEIGHT_KEYS)

    # ask if the file have the values defined as ranges and spacing/dimensions
    if x is None:
        x_range = np.array(get_variables_from_grp(root_grp, ['x_range']))
        if x_range is None:
            raise NetCDFImportError(1, LONGITUDE_KEYS)

        spacing = np.array(get_variables_from_grp(root_grp, ['spacing']))
        if spacing is None:
            raise NetCDFImportError(1, LONGITUDE_KEYS)

        x = np.arange(x_range[0], x_range[1], spacing[0]).tolist() + [x_range[1]]

    # ask if the file have the values defined as ranges and spacing/dimensions
    if y is None:
        y_range = np.array(get_variables_from_grp(root_grp, ['y_range']))
        if y_range is None:
            raise NetCDFImportError(1, LATITUDE_KEYS)

        spacing = np.array(get_variables_from_grp(root_grp, ['spacing']))
        if spacing is None:
            raise NetCDFImportError(1, LATITUDE_KEYS)

        y = np.arange(y_range[0], y_range[1], spacing[1]).tolist() + [y_range[1]]

    # shape the arrays to work
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # check the case the z_array is unidimensional
    if z.ndim == 1:
        z = z.reshape((len(y), len(x)))
        z = np.flipud(z)

    return x, y, z


if __name__ == "__main__":
    # filename = "../../test/input/files/test_file_2.nc"
    filename = "./test_inputs/ETOPO_IceSurfacec_6m.nc"
    rootgrp = Dataset(filename, "r", format="NETCDF4")

    print("Dimensiones del archivo:")
    print(rootgrp.dimensions)

    print("Grupos del archivo:")
    print(rootgrp.groups)

    print("Variables del archivo:")
    print(rootgrp.variables)

    X, Y, Z = read_info(filename)

    print("X values")
    print(X)

    print("Y values")
    print(Y)

    print("Z values.")
    print(Z)
