"""
Frame that indicate the parameters of the polygons
"""

import imgui
import psutil
import os

from src.engine.GUI.frames.frame import Frame
from src.utils import get_logger

log = get_logger(module="Polygon Information")


class PolygonInformation(Frame):
    """
    Class that render a frame to store the parameters of the active polygon.
    """

    def __init__(self, gui_manager: 'GUIManager'):
        """
        Constructor of the class.
        """
        super().__init__(gui_manager)
        self.__height = 300
        self.__width = 200

        # auxiliary variables
        self.__key_string_value = 'Name of parameter'
        self.__value_string_value = 'Value of parameter'

        self.__should_open_edit_dialog = False
        self.__parameter_to_edit = None

        self.__should_open_add_dialog = False

    def render(self) -> None:
        """
        Render the frame.
        Returns: None
        """
        # Do not draw the screen if there is no active polygon.
        if self._GUI_manager.get_active_polygon_id() is not None:

            # set the flags if the windows should be collapsable or not
            if self._GUI_manager.are_frame_fixed():
                imgui.begin('Polygon Information', False,
                            imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE)
                self.change_position([self._GUI_manager.get_window_width() - self.__width,
                                      self._GUI_manager.get_window_height() - self.__height])
                imgui.set_window_position(self.get_position()[0], self.get_position()[1])
                imgui.set_window_size(self.__width, self.__height, 0)

            else:
                imgui.begin('Polygon Information')

            # First row
            self._GUI_manager.set_bold_font()
            imgui.columns(2, 'Data List')
            imgui.separator()
            imgui.text("Field Name")
            imgui.next_column()
            imgui.text("Value")
            imgui.separator()
            self._GUI_manager.set_regular_font()

            # parameters
            for parameter in self._GUI_manager.get_polygon_parameters(self._GUI_manager.get_active_polygon_id()):

                # key
                imgui.next_column()
                imgui.text(parameter[0])
                if imgui.is_item_hovered() and imgui.is_mouse_clicked(1):
                    imgui.open_popup(f'options for parameter {parameter[0]}')

                # value
                imgui.next_column()
                imgui.text(str(parameter[1]))
                if imgui.is_item_hovered() and imgui.is_mouse_clicked(1):
                    imgui.open_popup(f'options for parameter {parameter[0]}')
                imgui.separator()

                # popup with the options
                if imgui.begin_popup(f'options for parameter {parameter[0]}'):

                    # edit option
                    imgui.selectable('Edit')
                    if imgui.is_item_clicked():
                        self.__should_open_edit_dialog = True
                        self.__parameter_to_edit = parameter[0]

                    # delete option
                    imgui.selectable('Delete')
                    if imgui.is_item_clicked():
                        self._GUI_manager.delete_polygon_parameter(self._GUI_manager.get_active_polygon_id(),
                                                                   parameter[0])

                    imgui.end_popup()

            # return to one column
            imgui.columns(1)

            # button to add a new parameter.
            if imgui.button("Add new", -1):
                self.__should_open_add_dialog = True

            # popup to add a new parameter
            self.__add_parameter_popup()

            # popup to edit a parameter
            self.__edit_parameter_popup()

            imgui.end()

    def __add_parameter_popup(self):
        # popup modal to add
        imgui.set_next_window_size(-1, -1)

        # in case of opening
        if self.__should_open_add_dialog:
            imgui.open_popup('Add new parameter')

            # once open this variable should be changed to false
            self.__key_string_value = 'Name of parameter'
            self.__value_string_value = 'Value of parameter'
            self.__should_open_add_dialog = False

        # popup to add a new parameter
        if imgui.begin_popup_modal('Add new parameter')[0]:

            # name
            # note: the input item text is the id and should be different from the ones that shows at the same time
            imgui.text('Name of the parameter:')
            imgui.same_line()
            _, self.__key_string_value = imgui.input_text('',
                                                          self.__key_string_value,
                                                          20)

            # value
            # note: the input item text is the id and should be different from the ones that shows at the same time
            imgui.text('Value of the parameter:')
            imgui.same_line()
            _, self.__value_string_value = imgui.input_text(' ',
                                                            self.__value_string_value,
                                                            50)

            if imgui.button('Done', -1):

                polygon_id = self._GUI_manager.get_active_polygon_id()
                dict_parameters = dict(self._GUI_manager.get_polygon_parameters(polygon_id))
                if self.__key_string_value in dict_parameters:

                    # close the popup
                    imgui.close_current_popup()

                    # open a modal
                    self._GUI_manager.set_modal_text('Error',
                                                     f'{self._GUI_manager.get_polygon_name(polygon_id)} '
                                                     f'already has a parameter named:'
                                                     f' {self.__key_string_value}')

                else:
                    # set the new parameter
                    self._GUI_manager.set_polygon_parameter(self._GUI_manager.get_active_polygon_id(),
                                                            self.__key_string_value,
                                                            self.__value_string_value)

                    # close the popup
                    imgui.close_current_popup()

            imgui.end_popup()

    def __edit_parameter_popup(self):

        # popup modal to edit
        imgui.set_next_window_size(-1, -1)

        # ask if open it
        if self.__should_open_edit_dialog:
            imgui.open_popup('Edit parameter')

            # once open this variable should be changed to false
            self.__should_open_edit_dialog = False

            # change default values for the fields
            self.__key_string_value = self.__parameter_to_edit
            self.__value_string_value = ''

        # popup to edit a parameter
        if imgui.begin_popup_modal('Edit parameter')[0]:

            # should modify the key
            imgui.text(f'Parameter: {self.__key_string_value}')

            # new parameter
            imgui.text('New value of the parameter:')
            imgui.same_line()
            _, self.__value_string_value = imgui.input_text('',
                                                            self.__value_string_value,
                                                            50)

            # change the parameter once done
            if imgui.button('Done', -1):
                # set the new parameter
                self._GUI_manager.set_polygon_parameter(self._GUI_manager.get_active_polygon_id(),
                                                        self.__key_string_value,
                                                        self.__value_string_value)

                # reset the variables
                imgui.close_current_popup()

            imgui.end_popup()
