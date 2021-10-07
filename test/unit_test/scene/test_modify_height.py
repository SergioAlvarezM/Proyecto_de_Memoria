#  BEGIN GPL LICENSE BLOCK
#
#      This program is free software: you can redistribute it and/or modify
#      it under the terms of the GNU General Public License as published by
#      the Free Software Foundation, either version 3 of the License, or
#      (at your option) any later version.
#
#      This program is distributed in the hope that it will be useful,
#      but WITHOUT ANY WARRANTY; without even the implied warranty of
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#      GNU General Public License for more details.
#
#      You should have received a copy of the GNU General Public License
#      along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#  END GPL LICENSE BLOCK

"""
File with the tests related to the modifications of the height to the points inside the polygons.
"""
import os

import numpy as np

from src.engine.scene.transformation.linear_transformation import LinearTransformation
from src.input.NetCDF import read_info
from test.test_case import ProgramTestCase


class TestLinearTransformation(ProgramTestCase):

    def test_linear_transformation(self):
        self.engine.create_model_from_file('resources/test_resources/cpt/cpt_1.cpt',
                                           'resources/test_resources/netcdf/test_file_1.nc')

        # load list of polygons
        self.engine.create_polygon_from_file('resources/test_resources/polygons/shape_one_polygon_south_america.shp')

        # apply transformation with filters
        transformation = LinearTransformation(self.engine.get_active_model_id(),
                                              self.engine.get_active_polygon_id(),
                                              2000,
                                              3000)
        self.engine.transform_points(transformation)

        # export model to compare data
        self.engine.export_model_as_netcdf(self.engine.get_active_model_id(),
                                           'resources/test_resources/temp/temp_transformation_1')

        # read data and compare
        info_written = read_info('resources/test_resources/temp/temp_transformation_1.nc')
        info_expected = read_info('resources/test_resources/expected_data/netcdf/expected_transformation_1.nc')

        np.testing.assert_array_almost_equal(info_written[0], info_expected[0], 3,
                                             'Info on the x array is not equal to the expected.')
        np.testing.assert_array_almost_equal(info_written[1], info_expected[1], 3,
                                             'Info on the y array is not equal to the expected.')
        np.testing.assert_array_almost_equal(info_written[2], info_expected[2], 3,
                                             'Info on the height matrix is not equal to the expected.')

        os.remove('resources/test_resources/temp/temp_transformation_1.nc')
