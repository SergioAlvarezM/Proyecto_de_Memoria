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
Main menu bar frame in the GUI
"""
from typing import TYPE_CHECKING

import OpenGL.GL as GL
import imgui

from src.engine.GUI.frames.frame import Frame
from src.program.view_mode import ViewMode
from src.utils import get_logger

if TYPE_CHECKING:
    from engine.GUI.guimanager import GUIManager

log = get_logger(module="MAIN_MENU_BAR")


class MainMenuBar(Frame):
    """
    Frame that controls the top menu bar of the application.
    """

    # noinspection PyUnresolvedReferences
    def __init__(self, gui_manager: 'GUIManager'):
        """
        Constructor of the class.

        Args:
            gui_manager: GuiManager of the application.
        """
        super().__init__(gui_manager)

    def render(self) -> None:
        """
        Render the main menu bar on the screen.
        Returns: None
        """
        current_model = self._GUI_manager.get_active_model_id()
        model_loaded = current_model is not None

        if imgui.begin_main_menu_bar():
            # File menu
            self.__file_menu(model_loaded)

            # Edit menu
            self.__edit_menu(model_loaded)

            # View menu
            self.__view_menu(model_loaded)

            imgui.end_main_menu_bar()

    def __file_menu(self, model_loaded: bool):
        """
        Options that appear on the File option of the main menu bar.

        Args:
            model_loaded: Boolean indicating if there is a model loaded in the program.
        """
        if imgui.begin_menu('File', True):

            # Option to open a NetCDF file
            imgui.menu_item('Open NetCDF file...', 'Ctrl+O', False, True)
            if imgui.is_item_clicked():
                self._GUI_manager.load_netcdf_file_with_dialog()

            # Option to open a CPT file
            imgui.menu_item('Change CPT file...', 'Ctrl+T', False, model_loaded)
            if imgui.is_item_clicked() and model_loaded:
                self._GUI_manager.change_color_file_with_dialog()

            # Option to load a Shapefile file
            imgui.separator()
            imgui.menu_item('Load shapefile file...', 'Ctrl+L', False, model_loaded)
            if imgui.is_item_clicked() and model_loaded:
                log.debug('Clicked load shapefile...')
                self._GUI_manager.load_shapefile_file_with_dialog()

            # Option to export the current model to NetCDF file
            imgui.separator()
            imgui.menu_item('Export current model...', enabled=model_loaded)
            if imgui.is_item_clicked() and model_loaded:
                self._GUI_manager.export_model_as_netcdf(self._GUI_manager.get_active_model_id())
                imgui.close_current_popup()

            imgui.end_menu()

    def __view_menu(self, model_loaded: bool):
        """
        Options that appear on the View option from the main menu bar

        Args:
            model_loaded: Boolean specifying if there is a model loaded in the program.
        """
        if imgui.begin_menu('View'):

            # Option to fix/unfix the frames of the application
            if self._GUI_manager.get_frame_fixed_state():
                imgui.menu_item('Unfix windows positions')
                if imgui.is_item_clicked():
                    self._GUI_manager.fix_frames_position(False)

            else:
                imgui.menu_item('Fix windows positions')
                if imgui.is_item_clicked():
                    self._GUI_manager.fix_frames_position(True)

            # Options to show the points of the map, lines, or to render the model of the map
            imgui.separator()
            imgui.menu_item('Use points', enabled=model_loaded)
            if imgui.is_item_clicked() and model_loaded:
                log.info("Rendering points")
                self._GUI_manager.set_models_polygon_mode(GL.GL_POINT)

            imgui.menu_item('Use wireframes', enabled=model_loaded)
            if imgui.is_item_clicked() and model_loaded:
                log.info("Rendering wireframes")
                self._GUI_manager.set_models_polygon_mode(GL.GL_LINE)

            imgui.menu_item('Fill polygons', enabled=model_loaded)
            if imgui.is_item_clicked() and model_loaded:
                log.info("Rendering filled polygons")
                self._GUI_manager.set_models_polygon_mode(GL.GL_FILL)

            # Option to change to 2D/3D mode. Raise error if the program is in another mode other than 3D or 2D.
            imgui.separator()
            program_view_mode = self._GUI_manager.get_program_view_mode()
            if program_view_mode == ViewMode.mode_3d:
                imgui.menu_item('Change to 2D view', enabled=model_loaded)
                if imgui.is_item_clicked() and model_loaded:
                    self._GUI_manager.set_program_view_mode(ViewMode.mode_2d)

            elif program_view_mode == ViewMode.mode_2d:
                imgui.menu_item('Change to 3D view', enabled=model_loaded)
                if imgui.is_item_clicked() and model_loaded:
                    self._GUI_manager.set_program_view_mode(ViewMode.mode_3d)

            else:
                raise ValueError('That mode is not configured yet.')

            imgui.end_menu()

    def __edit_menu(self, model_loaded: bool):
        """
        Options that appear when opening the Edit option from the main menu bar.

        Args:
            model_loaded: Boolean indicating if there is a model loaded in the program.
        """
        if imgui.begin_menu('Edit', True):

            # Option to undo the last executed action
            imgui.menu_item('Undo', 'CTRL+Z', False, model_loaded)
            if imgui.is_item_clicked() and model_loaded:
                self._GUI_manager.undo_action()
            imgui.end_menu()
