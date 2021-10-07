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
Module that tests the transformation used by the program to modify the height of the loaded maps.
"""

import unittest

import numpy as np

from src.engine.scene.transformation_helper import TransformationHelper
from src.input.NetCDF import read_info
from src.input.shapefile_importer import ShapefileImporter


class TestMinMaxPolygon(unittest.TestCase):

    def setUp(self) -> None:
        """
        Logic that runs at the beginning of every method.

        Initialize the common variables.
        """
        self.helper = TransformationHelper()

    def test_min_max_normal(self):

        # create and set the points on the grid
        points = np.zeros((10, 10, 3))
        for row in range(10):
            for col in range(10):
                points[row, col, 0] = col
                points[row, col, 1] = row

        # create the polygon
        polygon_points = [0.9, 0, 0,
                          5, 0, 0,
                          1, 5, 0]

        # create heights of the map
        height = np.zeros((10, 10))
        height[1, 1] = -15
        height[2, 2] = 30

        self.assertEqual((30, -15), self.helper.get_max_min_inside_polygon(points,
                                                                           polygon_points,
                                                                           height),
                         'Minimum and maximum values are not the one inside the polygon.')

    def test_min_max_same_value(self):

        # create and set the points on the grid
        points = np.zeros((10, 10, 3))
        for row in range(10):
            for col in range(10):
                points[row, col, 0] = col
                points[row, col, 1] = row

        # create the polygon
        polygon_points = [1, 0, 0,
                          5, 0, 0,
                          1, 5, 0]

        # create heights of the map
        height = np.ones((10, 10))
        height = height * 150

        self.assertEqual((150, 150), self.helper.get_max_min_inside_polygon(points,
                                                                            polygon_points,
                                                                            height),
                         'Minimum and maximum values are not the one inside the polygon.')

    def test_min_max_polygon_outside(self):

        # create and set the points on the grid
        points = np.zeros((10, 10, 3))
        for row in range(10):
            for col in range(10):
                points[row, col, 0] = col
                points[row, col, 1] = row

        # create the polygon
        polygon_points = [100, 0, 0,
                          150, 0, 0,
                          125, 50, 0]

        # create heights of the map
        height = np.ones((10, 10))
        height = height * 150

        (max_, min_) = self.helper.get_max_min_inside_polygon(points,
                                                              polygon_points,
                                                              height)
        self.assertEqual((np.nan, np.nan), (max_, min_),
                         'Minimum and maximum values are not the one inside the polygon.')


class TestInterpolateExternalPoints(unittest.TestCase):

    def setUp(self) -> None:
        """
        Logic that runs at the beginning of every method.

        Initialize the common variables.
        """
        self.helper = TransformationHelper()

    def setUp_real_case(self):
        """
        Prepare the variables to use in the real case tests.

        These variables are only used in the tests of real cases, and thus, they must not be defined on the
        setUp method.

        Returns: None
        """
        x, y, z = read_info('resources/test_resources/netcdf/test_file_1.nc')
        polygon_points, _ = ShapefileImporter().get_polygon_information('resources/test_resources/polygons/'
                                                                        'shape_two_concentric_polygons.shp')

        # get the polygon points
        polygon_points_formatted = []
        external_polygon_points_formatted = []

        # polygon[1] is an external polygon to polygon[0]
        for point in polygon_points[1]:
            polygon_points_formatted.append(point[0])
            polygon_points_formatted.append(point[1])
            polygon_points_formatted.append(0.5)

        # polygon[0] in an internal polygon to polygon[1]
        for point in polygon_points[0]:
            external_polygon_points_formatted.append(point[0])
            external_polygon_points_formatted.append(point[1])
            external_polygon_points_formatted.append(0.5)

        # format the points of the map
        width = len(x)
        height = len(y)

        x = np.tile(x, height)
        y = np.tile(y, width)

        x = x.reshape(z.shape)

        y = y.reshape((z.shape[1], z.shape[0]))
        y = y.transpose()

        points = np.zeros((height, width, 3))
        points[:, :, 0] = x
        points[:, :, 1] = y

        self.points = points
        self.polygon_points = polygon_points_formatted
        self.height = z
        self.external_polygon_points = external_polygon_points_formatted

    def plot_points(self, points: list) -> None:
        """
        Plot the points given using matplotlib.

        Args:
            points: Points to plot. format: [x1, y1, z1, x2, y2, z2, ...]

        Returns: None
        """
        array_points = np.array(points)
        array_points = array_points.reshape((-1, 3))

        from matplotlib import pyplot as plt
        plt.scatter(array_points[:, 0], array_points[:, 1])
        plt.show()

    def test_interpolate_external_points_real_case_linear(self):
        self.setUp_real_case()

        # apply interpolation
        new_height = self.helper.interpolate_points_external_to_polygon(self.points,
                                                                        self.polygon_points,
                                                                        self.height,
                                                                        self.external_polygon_points,
                                                                        'linear')

        expected_data = np.load('resources/test_resources/expected_data/npy_data/test_transformation_2.npy')
        self.assertTrue((new_height == expected_data).all(), 'Expected data is not equal to the one returned by the'
                                                             'method modify_points_inside_polygon_linear.')

    def test_interpolate_external_points_real_case_nearest(self):
        self.setUp_real_case()

        # apply interpolation
        new_height = self.helper.interpolate_points_external_to_polygon(self.points,
                                                                        self.polygon_points,
                                                                        self.height,
                                                                        self.external_polygon_points,
                                                                        'nearest')

        expected_data = np.load('resources/test_resources/expected_data/npy_data/test_transformation_4.npy')
        self.assertTrue((new_height == expected_data).all(), 'Expected data is not equal to the one returned by the'
                                                             'method modify_points_inside_polygon_linear.')

    def test_interpolate_external_points_real_case_cubic(self):
        self.setUp_real_case()

        # apply interpolation
        new_height = self.helper.interpolate_points_external_to_polygon(self.points,
                                                                        self.polygon_points,
                                                                        self.height,
                                                                        self.external_polygon_points,
                                                                        'cubic')

        expected_data = np.load('resources/test_resources/expected_data/npy_data/test_transformation_3.npy')
        self.assertTrue((new_height == expected_data).all(), 'Expected data is not equal to the one returned by the'
                                                             'method modify_points_inside_polygon_linear.')


if __name__ == '__main__':
    unittest.main()
