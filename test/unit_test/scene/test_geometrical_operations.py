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
Module in charge of the testing of the module geometrical operations.
"""

import unittest

import numpy as np

from src.engine.scene.geometrical_operations import merge_matrices


class TestMergeMatrices(unittest.TestCase):

    def test_merge_normal_matrices(self):
        first_matrix = np.array([[1, 2, 3],
                                 [4, np.nan, 6],
                                 [7, 8, 9]])
        second_matrix = np.array([[np.nan, np.nan, np.nan],
                                  [np.nan, 5, np.nan],
                                  [np.nan, np.nan, np.nan]])

        np.testing.assert_array_equal(np.array([[1, 2, 3],
                                                [4, 5, 6],
                                                [7, 8, 9]]),
                                      merge_matrices(first_matrix,
                                                     second_matrix))

    def test_do_not_modify_matrices(self):
        first_matrix = np.array([[1, 2, 3],
                                 [4, np.nan, 6],
                                 [7, 8, 9]])
        second_matrix = np.array([[np.nan, np.nan, np.nan],
                                  [np.nan, 5, np.nan],
                                  [np.nan, np.nan, np.nan]])

        merge_matrices(first_matrix, second_matrix)

        np.testing.assert_array_equal(np.array([[1, 2, 3],
                                                [4, np.nan, 6],
                                                [7, 8, 9]]),
                                      first_matrix,
                                      "Base matrix was modified on the process.")

        np.testing.assert_array_equal(np.array([[np.nan, np.nan, np.nan],
                                                [np.nan, 5, np.nan],
                                                [np.nan, np.nan, np.nan]]),
                                      second_matrix,
                                      "Second matrix was modified on the process.")

    def test_merge_shared_nan_values(self):
        first_matrix = np.array([[1, 2, 3],
                                 [4, np.nan, np.nan],
                                 [np.nan, np.nan, np.nan]])
        second_matrix = np.array([[np.nan, np.nan, np.nan],
                                  [np.nan, 5, np.nan],
                                  [np.nan, np.nan, 15]])

        np.testing.assert_array_equal(np.array([[1, 2, 3],
                                                [4, 5, np.nan],
                                                [np.nan, np.nan, 15]]),
                                      merge_matrices(first_matrix,
                                                     second_matrix),
                                      "Matrix generated is not equal to the expected.")

    def test_merge_shared_numeric_values(self):
        first_matrix = np.array([[1, 2, 3],
                                 [4, np.nan, np.nan],
                                 [np.nan, np.nan, np.nan]])
        second_matrix = np.array([[20, 15, 30],
                                  [np.nan, 5, 6],
                                  [7, 8, 9]])

        np.testing.assert_array_equal(np.array([[1, 2, 3],
                                                [4, 5, 6],
                                                [7, 8, 9]]),
                                      merge_matrices(first_matrix,
                                                     second_matrix),
                                      "Matrix generated is not equal to the expected.")


if __name__ == '__main__':
    unittest.main()
