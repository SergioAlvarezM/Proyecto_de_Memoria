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
File with the class ReliefTools. Class in charge of render the Relief tools inside another frame.
"""

from dataclasses import dataclass
from typing import List

import imgui

from src.type_hinting import *
from src.utils import get_logger

log = get_logger(module="RELIEF_TOOLS")


@dataclass
class Filter:
    """
    Data class to represent the filters on the GUI.

    The type and the corresponding filters are as follows:
        0: height_less_than
        1: height_greater_than
        2: is_in
        3: is_not_in

    The arguments that each filter must have are as follows:
        0: int/float
        1: int/float
        2: String (polygon id)
        3: String (polygon id)
    """
    selected_type: int  # int representing the filter to use. The description for the filter is in the list of options.
    arguments: any  # arguments to apply the filter, can be a number, a polygon, or anything.

    def get_filter_tuple(self):
        """
        Get a tuple with the filter ID and the argument to use in the filter. The ID returned is the one used by the
        scene of the program.

        Example:
            ('height_less_than', 80)
            ('is_not_in', 'some_polygon')

        Returns: Tuple with the filter ID and the argument to use to apply the filter.
        """
        if self.selected_type == 0:
            return 'height_less_than', self.arguments

        elif self.selected_type == 1:
            return 'height_greater_than', self.arguments

        elif self.selected_type == 2:
            return 'is_in', self.arguments

        elif self.selected_type == 3:
            return 'is_not_in', self.arguments
        else:
            raise NotImplementedError(f'Conversion of filter with id {self.selected_type} not implemented.')


class ReliefTools:
    """
    Class that render the ReliefTools inside another frame.
    """

    # noinspection PyUnresolvedReferences
    def __init__(self, gui_manager: 'GUIManager'):
        """
        Constructor of the class.
        """
        self.__gui_manager: 'GUIManager' = gui_manager

        self.__combo_options = ["Linear"]
        self.__current_combo_option = 0

        self.__max_height_value = 0
        self.__min_height_value = 0

        self.__polygon_data = {}

        # filter data
        # options available. The filter IDs represent their index on this list.
        self.__filter_name_list: List[str] = ['Height <=', 'Height >= ', 'Is is ', 'Is not in ']
        self.__filters: List[Filter] = []  # filters to apply on the polygon if the interpolation is triggered

        # auxiliary variables

        # Variable that stores the values of the minimum and maximum height inside a polygon.
        # The values are [None, None] if there is nothing being calculated, ['waiting', 'waiting'] when the values
        # are being calculated, and [int, int] once the values are calculated.
        self.__max_min_data = [None, None]

    def __get_filters_dictionary_list(self) -> list:
        """
        Covert the filters to a list of tuples with the ID of the filter and the argument used to apply the filter.

        Tuples generated are as follows (filter_id, arguments).

        Returns: List with the information of the filters.
        """
        filter_dictionary_list = []
        for filter_obj in self.__filters:
            filter_dictionary_list.append(filter_obj.get_filter_tuple())

        return filter_dictionary_list

    def filter_menu(self):
        """
        Method with the logic to render the filter menu.

        Returns: None
        """

        # variable to store if it is necessary to delete a filter
        filter_to_remove = None

        imgui.text_wrapped('Filters:')

        # render the filters on the GUI
        for filter_ind in range(len(self.__filters)):
            filter_data = self.__filters[filter_ind]

            # push the ID since the elements of the filter will all have the same ID
            imgui.push_id(f"relief_tools_filter_{filter_ind}")

            # Selection of the filter
            _, filter_data.selected_type = imgui.combo('Filter',
                                                       filter_data.selected_type,
                                                       self.__filter_name_list)

            # Selection of the argument for the filter.
            # this vary depending on the filter selected
            # ------------------------------------------

            # height <= or height >=
            if filter_data.selected_type == 0 or filter_data.selected_type == 1:
                if not isinstance(filter_data.arguments, float):
                    filter_data.arguments = 0
                _, filter_data.arguments = imgui.input_float('Value', filter_data.arguments)

            # is in or is not in
            elif filter_data.selected_type == 2 or filter_data.selected_type == 3:
                # get list with the polygons on the program and remove the active one
                polygon_list = self.__gui_manager.get_polygon_id_list()
                polygon_list.remove(self.__gui_manager.get_active_polygon_id())
                polygon_list_names = list(map(lambda x: self.__gui_manager.get_polygon_name(x), polygon_list))

                # empty list case
                if len(polygon_list) == 0:
                    filter_data.arguments = 0
                    _, filter_data.arguments = imgui.combo('Value',
                                                           filter_data.arguments,
                                                           polygon_list_names)

                else:
                    # change the current polygon to the first on the list if it is not selected
                    if filter_data.arguments not in polygon_list:
                        filter_data.arguments = polygon_list[0]

                    # show the combo options for the rendering. Must be at least one polygon for the filter to work.
                    _, selected_polygon = imgui.combo('Value',
                                                      polygon_list.index(filter_data.arguments),
                                                      polygon_list_names)
                    filter_data.arguments = polygon_list[selected_polygon]

            # button to remove the filter
            if imgui.button('Remove Filter'):
                filter_to_remove = filter_data

            imgui.pop_id()

        # remove the filter if the button to remove was pressed
        if filter_to_remove is not None:
            self.__filters.remove(filter_to_remove)

        # button to add more filters
        if imgui.button('Add Filter', -1):
            self.__filters.append(Filter(0, 0))

    def render(self) -> None:
        """
        Render the relief tools to modify the relief of the model.
        Returns: None
        """
        # get all the data necessary
        active_polygon_id = self.__gui_manager.get_active_polygon_id()
        active_model_id = self.__gui_manager.get_active_model_id()

        if active_polygon_id not in self.__polygon_data:
            self.__polygon_data[active_polygon_id] = {
                'max_height': None,
                'min_height': None
            }

        # Update the values of the minimum and maximum height if they were calculated.
        if self.__max_min_data != [None, None] and self.__max_min_data != ['waiting', 'waiting']:
            self.__polygon_data[active_polygon_id]['max_height'] = max(self.__max_min_data)
            self.__polygon_data[active_polygon_id]['min_height'] = min(self.__max_min_data)
            self.__max_min_data = [None, None]

        self.__gui_manager.set_tool_title_font()
        imgui.text('Relief Tools')
        self.__gui_manager.set_regular_font()

        imgui.text('Current polygon information:')
        imgui.columns(2, None, False)
        imgui.text(f'Max height:')
        imgui.next_column()
        imgui.text(str(self.__polygon_data[active_polygon_id]['max_height']))
        imgui.next_column()
        imgui.text(f'Min height:')
        imgui.next_column()
        imgui.text(str(self.__polygon_data[active_polygon_id]['min_height']))
        imgui.columns(1)

        if imgui.button('Recalculate Information', -1):
            log.debug('Recalculate polygon information')

            if active_model_id is None:
                self.__gui_manager.set_modal_text('Error', 'You must load a model to try to calculate the '
                                                           'height of the '
                                                           'points inside it.')
            else:
                # Change the values of the max_min_data to waiting and execute the function to calculate the values.
                # The method to calculate max_min_height is asynchronous, so it returns immediately.
                self.__max_min_data = ['waiting', 'waiting']
                self.__gui_manager.calculate_max_min_height(active_model_id,
                                                            active_polygon_id,
                                                            self.__max_min_data)

        # Filter Logic
        # ------------
        self.filter_menu()

        # Transformation Menu
        # -------------------
        self.transformation_menu(active_model_id, active_polygon_id)

    def transformation_menu(self, active_model_id, active_polygon_id):
        """
        Method with the logic to render the menu that applies the interpolation on the maps.

        Args:
            active_model_id: Model id to use.
            active_polygon_id: Active polygon ID.

        Returns: None

        """

        clicked, self.__current_combo_option = imgui.combo(
            "Transformation", self.__current_combo_option, self.__combo_options
        )
        _, self.__min_height_value = imgui.input_float('Min Height', self.__min_height_value)
        _, self.__max_height_value = imgui.input_float('Max Height', self.__max_height_value)
        if imgui.button('Change Height', -1):
            if self.__min_height_value >= self.__max_height_value:
                self.__gui_manager.set_modal_text('Error', 'The new minimum value is higher or equal to'
                                                           ' the maximum value.')
            elif active_model_id is None:
                self.__gui_manager.set_modal_text('Error', 'You must load a model to try to calculate the '
                                                           'height of the '
                                                           'points inside it.')
            else:
                if self.__current_combo_option == 0:
                    self.__gui_manager.change_points_height(active_polygon_id,
                                                            active_model_id,
                                                            min_height=self.__min_height_value,
                                                            max_height=self.__max_height_value,
                                                            transformation_type='linear',
                                                            filters=self.__get_filters_dictionary_list())
