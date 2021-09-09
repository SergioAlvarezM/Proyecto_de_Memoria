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
Module with test related to the engine of the program.
"""

import unittest
import warnings

from src.engine.engine import Engine
from src.program.program import Program
from src.utils import dict_to_serializable_dict, json_to_dict


class TestViewMode(unittest.TestCase):

    def test_default_view_mode(self):
        engine = Engine()
        program = Program(engine, initialize_engine=False)

        self.assertEqual('2D',
                         program.get_view_mode(),
                         '2D is not the default mode when creating the program.')

    def test_view_mode_3D(self):
        engine = Engine()
        program = Program(engine, initialize_engine=False)

        engine.set_program_view_mode('3D')
        self.assertEqual('3D',
                         program.get_view_mode(),
                         'Mode was not changed to 3D after calling set_program_view_mode')

    def test_view_mode_2D(self):
        engine = Engine()
        program = Program(engine, initialize_engine=False)

        engine.set_program_view_mode('3D')
        engine.set_program_view_mode('2D')
        self.assertEqual('2D',
                         program.get_view_mode(),
                         'Mode was not changed to 2D after calling set_program_view_mode')


class TestModelInformation(unittest.TestCase):
    engine: Engine = None
    program: Program = None

    @classmethod
    def setUpClass(cls) -> None:
        """
        Method executed once before the tests begins.

        Create program and load a map into it.
        """
        warnings.simplefilter('ignore', category=ResourceWarning)
        cls.engine = Engine()
        cls.engine.should_use_threads(False)
        cls.program = Program(cls.engine)

        cls.engine.load_netcdf_file('resources/test_resources/cpt/colors_0_100_200.cpt',
                                    'resources/test_resources/netcdf/test_file_1.nc')

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Method executed once after all tests.
        """
        cls.program.close()

    def test_information_keys(self):
        model_list = self.engine.get_model_list()
        self.assertEqual(1,
                         len(model_list),
                         'Model list does not have only one element.')
        self.assertEqual(
            ['height_array', 'coordinates_array', 'projection_matrix', 'showed_limits', 'shape', 'filename'],
            list(self.engine.get_model_information(model_list[0]).keys()),
            'Keys in the dictionary are not equal to the expected keys.'
        )

    def test_information_values(self):
        model_list = self.engine.get_model_list()
        model_information = self.engine.get_model_information(model_list[0])

        model_information_serialized = dict_to_serializable_dict(model_information)
        model_information_expected = json_to_dict('resources/test_resources/expected_data/json_data/model_info.json')

        self.assertEqual(model_information_serialized,
                         model_information_expected,
                         'Model information generated is not equal to the expected.')


if __name__ == '__main__':
    unittest.main()