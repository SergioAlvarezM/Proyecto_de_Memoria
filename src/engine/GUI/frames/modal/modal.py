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
Module that defines the Modal class, base class to use when defining new modals in the GUI.
"""
from typing import TYPE_CHECKING, Union

import imgui

from src.engine.GUI.frames.frame import Frame

if TYPE_CHECKING:
    from src.engine.GUI.guimanager import GUIManager


class Modal(Frame):
    """
    Base class to use when defining modals on the GUI.
    """

    def __init__(self, gui_manager: 'GUIManager'):
        super().__init__(gui_manager)
        self.__should_show = False
        self.__tool_before_pop_up: Union[str, None] = None

    @property
    def should_show(self) -> bool:
        """Get if the frame should be shown on the program."""
        return self.__should_show

    @should_show.setter
    def should_show(self, value: bool) -> None:
        """Specify if the frame should be shown on the program."""
        self.__should_show = value

    def begin_modal(self, modal_title: str) -> bool:
        """
        Begin a new modal.

        Open a new modal, executing all the logic related to the opening of a new modal.

        Example:
            if self.begin_modal('my_new_modal'):
                ...
                if button(...):
                    self.close_modal()

                imgui.end_popup()

        Args:
            modal_title: Title of the modal to open.

        Returns: Boolean indicating if the frame will be rendered.
        """
        if self.should_show:
            # Ask imgui to open the popup modal
            imgui.open_popup(modal_title)

            # Stores the active tool and deactivate it
            self.__tool_before_pop_up = self._GUI_manager.get_active_tool()
            self._GUI_manager.set_active_tool(None)

            # Disable keyboard input
            self._GUI_manager.set_controller_keyboard_callback_state(False)

            # Return the variable should_show to false since the modal was already opened
            self.should_show = False

        # Fix the size for the frame to be shown in the center of the program
        # -------------------------------------------------------------------
        imgui.set_next_window_size(self.size[0], -1)
        imgui.set_next_window_position(imgui.get_io().display_size.x * 0.5,
                                       imgui.get_io().display_size.y * 0.5,
                                       imgui.ALWAYS,
                                       0.5,
                                       0.5)
        return imgui.begin_popup_modal(modal_title)[0]

    def close_modal(self) -> None:
        """
        Close the current active modal.

        Close the modal being rendered, executing all the logic related to the closing process of a modal.

        Returns: None
        """
        imgui.close_current_popup()

        self._GUI_manager.set_active_tool(self.__tool_before_pop_up)
        self._GUI_manager.set_controller_keyboard_callback_state(True)
        self.should_show = False

    def render(self) -> None:
        """
        Do nothing since modals should not be draw in every frame.

        Returns: None
        """
        pass

    def post_render(self) -> None:
        """
        Since the modals should be shown over the other frames, they should be defined in this method to keep them
        from being closed by other frame being rendered after them.

        Returns: None
        """
        raise NotImplementedError('Logic for the frame was not implemented.')
