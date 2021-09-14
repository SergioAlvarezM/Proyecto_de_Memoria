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
File that contains the Engine class. Class in charge of the management of all the logic of the application.
"""
from typing import List, TYPE_CHECKING, Union

import glfw
from PIL import Image

from src.engine.GUI.guimanager import GUIManager
from src.engine.controller.controller import Controller
from src.engine.process_manager import ProcessManager
from src.engine.render.render import Render
from src.engine.scene.scene import Scene
from src.engine.settings import Settings
from src.engine.task_manager import TaskManager
from src.engine.thread_manager import ThreadManager
from src.error.export_error import ExportError
from src.error.interpolation_error import InterpolationError
from src.error.line_intersection_error import LineIntersectionError
from src.error.model_transformation_error import ModelTransformationError
from src.error.netcdf_import_error import NetCDFImportError
from src.error.repeated_point_error import RepeatedPointError
from src.error.scene_error import SceneError
from src.input.NetCDF import read_info
from src.input.shapefile_importer import ShapefileImporter
from src.output.netcdf_exporter import NetcdfExporter
from src.output.shapefile_exporter import ShapefileExporter
from src.utils import get_logger

if TYPE_CHECKING:
    import numpy as np

log = get_logger(module='ENGINE')


class Engine:
    """
    Main class of the program, controls and connect every component of the program.

    The engine is the one in charge of the management of the resources of the application. This class controls the
    scene, the GUI, the program, the render and the controller. Also, this programs uses components from the input and
    output modules to load new files into the application or to export data respectively.
    """

    def __init__(self):
        """
        Constructor of the program.
        """
        self.render = Render()
        self.gui_manager = GUIManager(self)
        self.window = None
        self.scene = Scene(self)
        self.controller = Controller(self)
        self.program = None

        self.__use_threads = True
        self.__process_manager = ProcessManager()
        self.__thread_manager = ThreadManager()
        self.__task_manager = TaskManager()

    def add_new_vertex_to_active_polygon_using_real_coords(self, position_x: float, position_y: float) -> None:
        """
        Add a new point to the polygon using map coordinates.

        The points added using this method will not be modified before adding them to the polygons.

        Args:
            position_x: Position on the X axis.
            position_y: Position on the Y axis. (from bottom to top)

        Returns: None
        """
        try:
            self.scene.add_new_vertex_to_active_polygon_using_map_coords(position_x, position_y)
        except RepeatedPointError:
            log.info('Handling repeated point.')
            self.set_modal_text('Error', 'Point already exist in polygon.')

        except LineIntersectionError:
            log.info('Handling line intersection.')
            self.set_modal_text('Error', 'Line intersect another one already in the polygon.')

    def add_new_vertex_to_active_polygon_using_window_coords(self, position_x: int, position_y: int) -> None:
        """
        Ask the scene to add a vertex in the active polygon of the engine.

        Args:
            position_x: Position X of the point
            position_y: Position Y of the point (from top to bottom)

        Returns: None
        """

        # Check for the polygon and model to exist. if not, then open a modal text with a message explaining the error
        if self.get_active_polygon_id() is None:
            self.set_modal_text('Error', 'Please select a polygon before adding a new vertex to it.')
            return

        if self.get_active_model_id() is None:
            self.set_modal_text('Error', 'Please load a model before adding vertices to the polygon.')
            return

        # Add  the new vertex to the polygon located on the scene. In case of error, then open a modal text with a
        # message explaining the error.
        try:
            self.scene.add_new_vertex_to_active_polygon_using_window_coords(position_x, position_y)

        except RepeatedPointError:
            log.info('Handling repeated point.')
            self.set_modal_text('Error', 'Point already exist in polygon.')

        except LineIntersectionError:
            log.info('Handling line intersection.')
            self.set_modal_text('Error', 'Line intersect another one already in the polygon.')

    def add_zoom(self) -> None:
        """
        Add zoom to the current map being watched.

        Returns: None
        """
        self.program.add_zoom()
        self.scene.update_projection_matrix_2D()

    def apply_smoothing(self, polygon_id: str, model_id: str, distance_to_polygon: float) -> None:
        """
        Ask the scene to apply smoothing over the indicated polygon.

        A loading window will appear in the program when realizing the smoothing process on the loaded map.

        NOTE: The smoothing process will not be done in the same frame that this method is called. It will be
              done roughly 3 frames after this method is called.

        Args:
            polygon_id: Polygon to use to apply smoothing.
            model_id: Model to use.
            distance_to_polygon: Distance to generate the external polygon and the smoothing area.

        Returns: None
        """
        if model_id is None:
            self.set_modal_text('Error', 'Please load a model before smoothing.')
            return

        if polygon_id is None:
            self.set_modal_text('Error', 'Please select a polygon to use for the interpolation')
            return

        if not self.scene.is_polygon_planar(polygon_id):
            self.set_modal_text('Error', 'Polygon selected is not planar.')
            return

        self.set_loading_message('Applying smoothing, This may take a while.')
        self.set_task_with_loading_frame(lambda: self.scene.apply_smoothing_algorithm(polygon_id,
                                                                                      model_id,
                                                                                      distance_to_polygon))

    def are_frames_fixed(self) -> bool:
        """
        Return if the frames are fixed or not in the application.

        The data is asked directly to the Settings static class.

        Returns: boolean indicating if the frames are fixed
        """
        return Settings.FIXED_FRAMES

    def calculate_max_min_height(self, model_id: str, polygon_id: str, return_data: list) -> None:
        """
        Ask the scene for max and min values of the vertices that are inside the polygon.

        This method is executed asynchronously, returning immediately. When the asynchronous task end, the calculated
         values will be stored in the return_data variable. In case of error, [None, None] will be set in the
         return_data variable.

        This method is asynchronous, but it does not use threads or new process, it just set the execution of the code
        some frames in the future to give time to the Loading windows to appear on the program.

        Args:
            return_data: List with length 2 where to store the data.
            model_id: ID of the model to use.
            polygon_id: ID of the polygon to use.

        Returns: tuple with the max and min value.
        """
        assert len(return_data) == 2, 'List to use as return value must be of length 2'

        def asynchronous_task(return_values):
            """Task to execute in a future frame"""

            try:
                max_value, min_value = self.scene.calculate_max_min_height(model_id, polygon_id)
                return_values[0] = max_value
                return_values[1] = min_value

            except SceneError as e:
                if e.code == 1:
                    self.set_modal_text('Error',
                                        'The polygon is not planar. Try using a planar polygon.')
                    return_values[0] = None
                    return_values[1] = None

                elif e.code == 2:
                    self.set_modal_text('Error',
                                        'The polygon must have at least 3 points to be able to '
                                        'calculate the information.')
                    return_values[0] = None
                    return_values[1] = None

                elif e.code == 3:
                    self.set_modal_text('Error',
                                        'The current model is not supported to use to update the '
                                        'height of the vertices, try using another type of model.')
                    return_values[0] = None
                    return_values[1] = None

                else:
                    raise e

        # Set the message for the loading frame and set the task to be executed behind a loading frame
        self.set_loading_message('Calculating heights...')
        self.set_task_with_loading_frame(lambda: asynchronous_task(return_data))

    def change_3D_model_height_unit(self, model_id: str, measure_unit: str) -> None:
        """
        Ask the scene to change the measure unit of the specified model.

        Args:
            model_id: id of the model to change the measure unit of the height.
            measure_unit: new measure unit to use in the model.

        Returns: None
        """
        self.scene.change_height_unit_3D_model(model_id, measure_unit)

    def change_3D_model_position_unit(self, model_id: str, measure_unit: str) -> None:
        """
        Ask the scene to change the measure unit of the points on the model.

        Args:
            model_id: id of the model to change the measure unit to.
            measure_unit: new measure unit to use.

        Returns: None
        """
        self.scene.change_map_unit_3D_model(model_id, measure_unit)

    def change_camera_elevation(self, angle) -> None:
        """
        Ask the scene to change the camera elevation.

        Args:
            angle: Angle to add to the elevation of the camera.

        Returns: None
        """
        self.scene.change_camera_elevation(angle)

    def change_camera_xy_angle(self, angle) -> None:
        """
        Ask the scene to change the azimuthal angle of the camera.

        Args:
            angle: angle to add to the angle of the camera.

        Returns: None
        """
        self.scene.change_camera_azimuthal_angle(angle)

    def change_color_file_with_dialog(self) -> None:
        """
        Change the color file (CPT file) to the one selected.
        This change all the models using the color file.

        Returns: None
        """
        try:
            self.program.load_cpt_file_with_dialog()
        except FileNotFoundError:
            self.set_modal_text('Error', 'File not loaded.')
        except IOError:
            self.set_modal_text('Error', 'File is not a cpt file.')

    def change_color_of_polygon(self, polygon_id: str, color: list) -> None:
        """
        Change the color of the polygon with the specified id.

        Only change the color of the lines of the polygon.

        The colors must be defined in the order RGBA and with values between 0 and 1.

        Args:
            polygon_id: Id of the polygon to change the color.
            color: List-like object with the colors to use.

        Returns: None
        """
        self.scene.change_color_of_polygon(polygon_id, color)

    def change_current_3D_model_normalization_factor(self, new_factor: float) -> None:
        """
        Ask the scene to change the height normalization factor of the current 3D model.

        Args:
            new_factor: new height normalization factor to use in the model.

        Returns: None
        """
        active_model = self.get_active_model_id()
        self.scene.change_normalization_height_factor(active_model, new_factor)

    def change_dot_color_of_polygon(self, polygon_id: str, color: list) -> None:
        """
        Change the color of the dots of the polygon with the specified id.

        Only change the color of the dots of the polygon.

        The colors must be defined in the order RGBA and with values between 0 and 1.

        Args:
            polygon_id: Id of the polygon to change the color.
            color: List-like object with the colors to use.

        Returns: None
        """
        self.scene.change_dot_color_of_polygon(polygon_id, color)

    def change_height_window(self, height: int) -> None:
        """
        Change the engine settings height for the windows.
        Args:
            height: New height

        Returns: None
        """
        Settings.HEIGHT = height

    def change_polygon_draw_priority(self, polygon_id: str, new_priority_value: int) -> None:
        """
        Ask the scene to change the order in which the polygons are draw.

        The closer the priority is to 0, the higher the priority. Polygons with high priority will be draw over 
        polygons with less priority.

        Args:
            polygon_id: Polygon to change the order.
            new_priority_value: New position in the order of drawing. If value is negative, then the polygon will be the
                                draw over all the other polygons.

        Returns: None
        """
        return self.scene.change_polygon_draw_priority(polygon_id, new_priority_value)

    def change_model_draw_priority(self, model_id: str, new_priority_value: int) -> None:
        """
        Ask the scene to change the order in which the models are draw.

        The closer the priority is to 0, the higher the priority. Models with high priority will be draw over
        models with less priority.

        Args:
            model_id: Model to change the order.
            new_priority_value: New position in the order of drawing. If value is negative, then the model will be the
                                draw over all the other models.

        Returns: None
        """
        return self.scene.change_model_draw_priority(model_id, new_priority_value)

    def change_quality(self, quality: int) -> None:
        """
        Change the quality used to render the maps.

        Args:
            quality: Quality to use in the rendering process

        Returns: None
        """
        Settings.QUALITY = quality

    def change_width_window(self, width: int) -> None:
        """
        Change the engine settings width for the windows
        Args:
            width: New width

        Returns: None
        """
        Settings.WIDTH = width

    def create_new_polygon(self) -> str:
        """
        Create a new polygon on the scene.

        Returns: the id of the new polygon
        """
        return self.scene.create_new_polygon()

    def delete_parameter_from_polygon(self, polygon_id: str, key: str) -> None:
        """
        Delete a parameter from a polygon.

        Args:
            polygon_id: ID of the polygon.
            key: Parameter to be deleted.

        Returns: None
        """
        self.scene.delete_polygon_param(polygon_id, key)

    def delete_polygon_by_id(self, polygon_id: str) -> None:
        """
        Delete the polygon with the specified id from the scene

        Args:
            polygon_id: Polygon id to use to delete

        Returns: None
        """
        self.scene.delete_polygon_by_id(polygon_id)

    def disable_only_gui_keyboard_callback(self) -> None:
        """
        Enable the glfw callback defined in the controller.

        The GUI callback is not affected.

        Returns: None
        """
        self.controller.enable_only_gui_keyboard_callback()

    def enable_only_gui_keyboard_callback(self) -> None:
        """
        Disable the glfw callback defined in the controller.

        The GUI callback is not affected.

        Returns: None
        """
        self.controller.disable_only_gui_keyboard_callback()

    def exit(self):
        """
        Terminate the process in charge of rendering the windows and the scene, closing the windows and returning the
        resources to the OS.

        This method do not close the program, just exit the process executed by the engine, to close the program
        completely call the method close() from the Program class.

        Returns: None
        """
        # Terminate process external to the engine, returning the resources to the OS.
        glfw.terminate()

    def export_model_as_netcdf(self, model_id: str, directory_file: str = None) -> None:
        """
        Save the information of a model in a netcdf file.

        Args:
            model_id: ID of the model to export.
            directory_file: Directory and filename to use to store the file. If not selected, then a popup is opened.

        Returns: None
        """
        try:
            # Select a directory to store the file.
            if directory_file is None:
                directory_file = self.program.open_file_save_box_dialog(
                    'Select a directory and filename for the shapefile file.',
                    'Relief Creator',
                    'Model')

            # Add the .nc extension if the selected directory does not have it.
            if directory_file[-3:] != '.nc':
                directory_file += '.nc'

            # Ask the scene for information of the model
            vertices = self.scene.get_map2dmodel_vertices_array(model_id)

            # Check if the temporary file exists, if it exists then store the new data on the file,
            # otherwise, show an error message.
            if self.program.check_model_temp_file_exists():
                NetcdfExporter().modify_heights_existent_netcdf_file(vertices[:, :, 2],
                                                                     self.program.get_model_temp_file())
            else:
                self.set_modal_text('Error', 'Temporary file storing the original data not found, try loading the '
                                             'map again.')
                return

            # Export temporary file to the directory selected
            self.program.copy_model_temp_file(directory_file)

            # Show success window  to the user
            self.set_modal_text('Information', 'Model exported successfully')

        except TypeError:
            self.set_modal_text('Error', 'This model can not be exported.')

        except ValueError:
            self.set_modal_text('Error', 'You must select a directory to save the model.')

    def export_polygon_list_id(self, polygon_id_list: list, filename_placeholder: str = 'polygons',
                               directory_filename: str = None) -> None:
        """
        Export the polygons to a shapefile file.

        Args:
            filename_placeholder: Name to use as placeholder in the box to store files.
            polygon_id_list: List with the polygons IDs.
            directory_filename: directory and filename to use to store the files.

        Returns: None
        """
        points_list = []
        parameters_list = []
        names_list = []
        for polygon_id in polygon_id_list:
            points_list.append(self.scene.get_point_list_from_polygon(polygon_id))
            parameters_list.append(dict(self.scene.get_polygon_params(polygon_id)))
            names_list.append(self.scene.get_polygon_name(polygon_id))

        try:
            if directory_filename is None:
                directory_filename = self.program.open_file_save_box_dialog(
                    'Select a directory and filename for the shapefile file.',
                    'Relief Creator',
                    filename_placeholder)
        except ValueError:
            self.set_modal_text('Error', 'Polygons not exported.')
            return

        try:
            ShapefileExporter().export_list_of_polygons(points_list,
                                                        parameters_list,
                                                        names_list,
                                                        directory_filename)
        except ExportError as e:
            if e.code == 1:
                self.set_modal_text('Error', 'One or more polygons does not have enough points to be exported.')
                return
            else:
                raise e

        self.set_modal_text('Information', 'Polygons exported successfully.')

    def export_polygon_with_id(self, polygon_id: str, directory_filename: str = None) -> None:
        """
        Export the polygon with the given ID to a shapefile file

        Args:
            polygon_id: Id of the polygon to export to shapefile
            directory_filename: directory and filename to use to store the  shapefile file. If none then a window
                                to select the directory is opened.

        Returns: None
        """

        # Ask for the points of the polygon
        points = self.scene.get_point_list_from_polygon(polygon_id)

        try:
            if directory_filename is None:
                directory_filename = self.program.open_file_save_box_dialog(
                    'Select a filename and directory for the new polygon',
                    'Relief Creator',
                    self.scene.get_polygon_name(polygon_id))
        except ValueError:
            self.set_modal_text('Error', 'Polygon not exported.')
            return

        # Ask the exporter to export the list of points
        try:
            ShapefileExporter().export_polygon_to_shapefile(points,
                                                            directory_filename,
                                                            self.scene.get_polygon_name(polygon_id),
                                                            dict(self.scene.get_polygon_params(polygon_id)))
        except ExportError as e:
            if e.code == 2:
                self.set_modal_text("Error", "The polygon does not have enough points.")
                return
            else:
                raise e

        self.set_modal_text('Information', 'Polygon exported successfully')

    def fix_frames(self, fix: bool) -> None:
        """
        Fixes/unfix the frames in the application.
        Args:
            fix: boolean indicating if fix or not the frames.

        Returns: None
        """
        Settings.fix_frames(fix)

    def get_3d_model_list(self) -> List[str]:
        """
        Get the list of all 3D models generated in the program.

        Returns: List with the ID of the 3D models in the program.
        """
        return self.scene.get_3d_model_list()

    def get_active_model_id(self) -> str:
        """
        Returns the active model being used by the program.

        Returns: active model id
        """
        return self.program.get_active_model()

    def get_active_polygon_id(self) -> str:
        """
        Get the id of the active polygon.

        Returns: id of the active polygon
        """
        return self.program.get_active_polygon_id()

    def get_active_tool(self) -> str:
        """
        Get the active tool in the program.

        Returns: String with the active tool being used.
        """
        return self.program.get_active_tool()

    def get_camera_data(self) -> dict:
        """
        Ask the scene for the camera data.

        Returns: Dictionary with the data related to the camera.
        """
        return self.scene.get_camera_data()

    def get_camera_settings(self) -> dict:
        """
        Get all the settings related to the camera.

        Returns: Dictionary with the settings related to the camera.
        """
        return {
            'FIELD_OF_VIEW': Settings.FIELD_OF_VIEW,
            'PROJECTION_NEAR': Settings.PROJECTION_NEAR,
            'PROJECTION_FAR': Settings.PROJECTION_FAR
        }

    def get_clear_color(self) -> list:
        """
        Get the clear color used.

        Returns:list with the clear color
        """
        return Settings.CLEAR_COLOR

    def get_cpt_file(self) -> str:
        """
        Get the CTP file currently being used in the program

        Returns: String with the path to the file
        """
        return self.program.get_cpt_file()

    def get_extra_reload_proportion_setting(self) -> float:
        """
        Get the extra reload proportion stored in the settings of the program.

        Returns: Float with the value of the extra reload proportion.
        """
        return Settings.EXTRA_RELOAD_PROPORTION

    def get_model_information(self, model_id: str) -> dict:
        """
        Get the information of a model in a dictionary.

        The dictionary generated has the following shape:
        {
            'height_array': Numpy array
            'coordinates_array': (Numpy array, Numpy array),
            'projection_matrix': Numpy array,
            'showed_limits': {
                'left': Number,
                'right': Number,
                'top': Number,
                'bottom': Number
            },
            'shape': (Int, Int, Int),
            'name': string
        }

        Args:
            model_id: ID of the model on the program.

        Returns: Information of the model
        """
        return self.scene.get_model_information(model_id)

    def get_float_bytes(self) -> int:
        """
        Return the number of bytes used to represent a float number in opengl.

        Returns: Number of bytes used to represent a float.
        """
        return Settings.FLOAT_BYTES

    def get_font_size(self) -> int:
        """
        Get the font size to use in the program.

        Returns: Font size used by the program.
        """
        return Settings.FONT_SIZE

    def get_gui_key_callback(self) -> callable:
        """
        Get the key callback used by the gui

        Returns: Function used as the key callback in the gui
        """
        return self.gui_manager.get_gui_key_callback()

    def get_gui_scroll_callback(self):
        """
        Ask the gui manager for the callback used in the scrolling.

        Returns: Function used in the callback.
        """
        return self.gui_manager.get_gui_mouse_scroll_callback()

    def get_gui_setting_data(self) -> dict:
        """
        Get the GUI setting data.

        Returns: Dictionary with the data related to the GUI.
        """
        return {
            'LEFT_FRAME_WIDTH': Settings.LEFT_FRAME_WIDTH,
            'TOP_FRAME_HEIGHT': Settings.TOP_FRAME_HEIGHT,
            'MAIN_MENU_BAR_HEIGHT': Settings.MAIN_MENU_BAR_HEIGHT
        }

    def get_height_normalization_factor_of_active_3D_model(self) -> float:
        """
        Ask the scene for the normalization factor being used by the active 3D model.

        Return -1 if the model is not in the list of models of the scene.

        Returns: normalization factor being used by the model
        """
        try:
            active_model = self.get_active_model_id()
            return self.scene.get_height_normalization_factor(active_model)
        except KeyError:
            return -1

    def get_map_coordinates_from_window_coordinates(self, x_coordinate: int, y_coordinate: int) -> (float, float):
        """
        Get the position of a point in the map given in screen coordinates.

        Screen coordinates have the origin of the system at the top-left of the window, being the x-axis positive to
        the right and the y-axis positive to the bottom.

        The returned tuple is the real coordinates (coordinates used in the map) of the point specified in screen
        coordinates.

        If there is no map loaded on the program, then (None, None) is returned.

        Args:
            x_coordinate: x-axis component of the screen coordinate to evaluate on the map.
            y_coordinate: y-axis component of the screen coordinate to evaluate on the map.

        Returns: (x, y) tuple with the coordinates of the point on the map.
        """
        return self.scene.calculate_map_position_from_window(x_coordinate, y_coordinate)

    def get_map_height_on_coordinates(self, x_coordinate: float, y_coordinate: float) -> float:
        """
        Get the height of the current active map on the specified coordinates.

        If there is no map loaded or the coordinates are outside of the map, then None is returned.

        Args:
            x_coordinate: x-axis coordinate.
            y_coordinate: y-axis coordinate.

        Returns: Height of the active model on the specified location.
        """
        return self.scene.get_active_model_height_on_coordinates(x_coordinate, y_coordinate)

    def get_map_position(self) -> list:
        """
        Get the map position on the program.

        Returns: List with the position of the map.
        """
        return self.program.get_map_position()

    def get_model_list(self) -> List[str]:
        """
        Get a list with the id of all the 2D models loaded into the program.

        Returns: List of models loaded into the program.
        """
        return self.scene.get_model_list()

    def get_parameters_from_polygon(self, polygon_id: str) -> list:
        """
        Ask the scene for the parameters of certain polygon.

        Args:
            polygon_id: ID of the polygon to ask for

        Returns: List with the parameters of the polygon.
        """
        return self.scene.get_polygon_params(polygon_id)

    def get_points_from_polygon(self, polygon_id) -> list:
        """
        Get the list of points of a polygon.

        Args:
            polygon_id: ID of the polygon to ask for.

        Returns: List with the points of the polygon.
        """
        return self.scene.get_polygon_points(polygon_id)

    def get_polygon_id_list(self) -> list:
        """
        Get the full list of polygon ids currently being used on the program.

        Returns: List of polygons in the program.
        """
        return self.scene.get_polygon_id_list()

    def get_polygon_name(self, polygon_id: str) -> str:
        """
        Get the name of a polygon given its id

        Args:
            polygon_id: Id of the polygon

        Returns: Name of the polygon.
        """
        return self.scene.get_polygon_name(polygon_id)

    def get_program_view_mode(self) -> str:
        """
        Ask the program for the view mode being used.

        Returns: view mode being used by the program.
        """
        return self.program.get_view_mode()

    def get_quality(self) -> int:
        """
        Get the quality value stored in the settings.

        Returns: Quality setting.
        """
        return Settings.QUALITY

    def get_render_settings(self):
        """
        Return a dictionary with the settings related to the render.

        Returns: Dictionary with the render settings.
        """
        return {
            "LINE_WIDTH": Settings.LINE_WIDTH,
            "POLYGON_LINE_WIDTH": Settings.POLYGON_LINE_WIDTH,
            "QUALITY": Settings.QUALITY,
            "DOT_SIZE": Settings.DOT_SIZE,
            "POLYGON_DOT_SIZE": Settings.POLYGON_DOT_SIZE,
            "ACTIVE_POLYGON_LINE_WIDTH": Settings.ACTIVE_POLYGON_LINE_WIDTH
        }

    def get_scene_setting_data(self) -> dict:
        """
        Get the scene setting data.

        Returns: dict with the data.
        """
        return {
            'SCENE_BEGIN_X': Settings.SCENE_BEGIN_X, 'SCENE_BEGIN_Y': Settings.SCENE_BEGIN_Y,
            'SCENE_WIDTH_X': Settings.SCENE_WIDTH_X, 'SCENE_HEIGHT_Y': Settings.SCENE_HEIGHT_Y
        }

    def get_tool_title_font_size(self) -> int:
        """
        Get the font size to use for the tool titles.

        Returns: Int with the font size to use.
        """
        return Settings.TOOL_TITLE_FONT_SIZE

    def get_window_setting_data(self) -> dict:
        """
        Get the window setting data.

        Returns: dict with the data.
        """
        return {
            'HEIGHT': Settings.HEIGHT,
            'WIDTH': Settings.WIDTH,
            'MAX_WIDTH': Settings.MAX_WIDTH,
            'MAX_HEIGHT': Settings.MAX_HEIGHT,
            'MIN_WIDTH': Settings.MIN_WIDTH,
            'MIN_HEIGHT': Settings.MIN_HEIGHT
        }

    def get_zoom_level(self) -> float:
        """
        Get the zoom level currently being used in the program.

        Returns: Zoom level
        """
        return self.program.get_zoom_level()

    # noinspection PyUnresolvedReferences
    def initialize(self, engine: 'Engine', program: 'Program') -> None:
        """
        Initialize the components of the program.

        Args:
            program: Program that runs the engine and the application.
            engine: Engine to initialize.

        Returns: None
        """
        log.info('Starting Program')
        self.program = program

        # GLFW CODE
        log.debug("Creating windows.")
        self.window = self.render.init("Relief Creator", engine)

        # Icon of the program
        program_icon = Image.open('resources/icons/program_icons/icon_program.png')
        glfw.set_window_icon(self.window, 1, [program_icon])

        # GUI CODE
        log.debug("Loading GUI")
        self.gui_manager.initialize(self.window, engine, self.gui_manager)

        # CONTROLLER CODE
        glfw.set_key_callback(self.window, self.controller.get_on_key_callback())
        glfw.set_window_size_callback(self.window, self.controller.get_resize_callback())
        glfw.set_mouse_button_callback(self.window, self.controller.get_mouse_button_callback())
        glfw.set_cursor_pos_callback(self.window, self.controller.get_cursor_position_callback())
        glfw.set_scroll_callback(self.window, self.controller.get_mouse_scroll_callback())

    def interpolate_points(self, polygon_id: str, model_id: str, distance: float, type_interpolation: str) -> None:
        """
        Ask the scene to interpolate the points using the specified parameters.

        Possible interpolation types:
            - linear
            - nearest
            - cubic

        Args:
            type_interpolation: Type of interpolation to use.
            polygon_id: ID of the polygon to use.
            model_id: ID of the model to use.
            distance: Distance to use for the interpolation.

        Returns: None
        """
        if model_id is None:
            self.set_modal_text('Error', 'Please load a model before interpolating points.')
            return

        if polygon_id is None:
            self.set_modal_text('Error', 'Please select a polygon to use for the interpolation')
            return

        if not self.scene.is_polygon_planar(polygon_id):
            self.set_modal_text('Error', 'Polygon selected is not planar.')
            return

        try:
            self.scene.interpolate_points(polygon_id, model_id, distance, type_interpolation)

        except InterpolationError as e:
            if e.code == 1:
                self.set_modal_text('Error', 'There is not enough points in the polygon to do'
                                             ' the interpolation.')
            elif e.code == 2:
                self.set_modal_text('Error', 'Distance must be greater than 0 to do the '
                                             'interpolation')
            elif e.code == 3:
                self.set_modal_text('Error', 'Model used for interpolation is not accepted by '
                                             'the program.')

    def is_mouse_hovering_frame(self) -> bool:
        """
        Ask the GUIManager if the mouse is hovering a frame.

        Returns: Boolean indicating if mouse is hovering a frame or not.
        """
        return self.gui_manager.is_mouse_inside_frame()

    def is_polygon_planar(self, polygon_id: str) -> bool:
        """
        Ask to the scene if the polygon is planar or not.

        Args:
            polygon_id: Id of the polygon

        Returns: Boolean indicating if the polygon is planar or not
        """
        return self.scene.is_polygon_planar(polygon_id)

    def is_program_debug_mode(self) -> bool:
        """
        Get if the program was initiated in debug mode or not.

        Returns: Boolean indicating if the program is in debug mode or not.
        """
        return self.program.get_debug_mode()

    def is_program_loading(self) -> bool:
        """
        Return if the program is loading or not.

        Returns: Boolean representing if the program is running or not.
        """
        return self.program.is_loading()

    def less_zoom(self) -> None:
        """
        Reduce on 1 the level of zoom.

        Returns: None
        """
        self.program.less_zoom()
        self.scene.update_projection_matrix_2D()

    def load_netcdf_file(self, path_color_file: str, path_model: str, then: callable = lambda: None) -> None:
        """
        Load a netcdf file on the engine, refreshing the scene and showing it.

        This method also creates a copy of the loaded file in the directory specified by the program module. This file
        is used in the export process of the maps.

        This method executes asynchronously, to keep executing things in the same thread use the then  function.

        Args:
            path_color_file: Path to the color file to use.
            path_model: Path to the model file (NetCDF) to use.
            then: Function to call after the execution of the method.

        Returns: none
        """

        # noinspection PyMissingOrEmptyDocstring
        def then_routine(model_id):
            # Update the GUI
            # --------------
            self.gui_manager.add_model_to_gui(model_id)

            # Update the program and create 3D model if required
            # --------------------------------------------------
            self.program.set_active_model(model_id)
            self.program.set_loading(False)

            if self.program.get_view_mode() == '3D':
                self.set_task_with_loading_frame(lambda: self.scene.create_3D_model_if_not_exists())

            # Create temporary file with the information of the model
            # -------------------------------------------------------
            self.program.create_model_temp_file(path_model)

            then()

        self.program.set_loading(True)
        self.set_loading_message("Please wait a moment...")

        try:
            self.scene.load_model_from_file_async(path_color_file, path_model, then_routine)

        except OSError:
            self.program.set_loading(False)
            self.set_modal_text('Error', 'Error reading selected file. Is the file a netcdf file?')

        except SceneError as e:
            self.program.set_loading(False)
            active_model_info = self.scene.get_model_information(self.program.get_active_model())

            if e.code == 9:
                self.set_modal_text('Error',
                                    'The model loaded does not use the same values for the x-axis as the active '
                                    'model in the application.\n'
                                    f'Current model x-axis: {e.data.get("expected", None)}\n'
                                    f'Loaded model x-axis: {e.data.get("actual", None)}')
            if e.code == 10:
                self.set_modal_text('Error',
                                    'The model loaded does not use the same values for the y-axis as the active '
                                    'model in the application.\n'
                                    f'Current model y-axis: {e.data.get("expected", None)}\n'
                                    f'Loaded model y-axis: {e.data.get("actual", None)}')
            if e.code == 11:
                self.set_modal_text('Error',
                                    'The resolution of the model loaded is no the same as the active model.\n'
                                    f'Current model shape: {e.data.get("expected", None)}\n'
                                    f'Loaded model shape: {e.data.get("actual", None)}'
                                    )

        except NetCDFImportError as e:
            self.program.set_loading(False)

            if e.code == 2:
                self.set_modal_text('Error',
                                    f'{e.get_code_message()}\n\n'
                                    f'Current keys on the file are: {list(e.data["file_keys"])}\n\n'
                                    f'Keys accepted by the program for latitude are: {list(e.data["accepted_keys"])}'
                                    f'\n\nTry adding a key to the latitude_keys.json file located in the resources '
                                    f'folder and restarting the application.')
            if e.code == 3:
                self.set_modal_text('Error',
                                    f'{e.get_code_message()}\n\n'
                                    f'Current keys on the file are: {list(e.data["file_keys"])}\n\n'
                                    f'Keys accepted by the program for longitude are: {list(e.data["accepted_keys"])}'
                                    f'\n\nTry adding a key to the longitude_keys.json file located in the resources '
                                    f'folder and restarting the application.')
            if e.code == 4:
                self.set_modal_text('Error',
                                    f'{e.get_code_message()}\n\n'
                                    f'Current keys on the file are: {list(e.data["file_keys"])}\n\n'
                                    f'Keys accepted by the program for height are: {list(e.data["accepted_keys"])}'
                                    f'\n\nTry adding a key to the height_keys.json file located in the resources folder'
                                    f' and restarting the application.')

        except KeyError:
            self.program.set_loading(False)
            self.set_modal_text('Error', 'Error reading selected file. Is the key used in the file inside the '
                                         'list of keys?')

    def load_netcdf_file_with_dialog(self) -> None:
        """
        Open a dialog to load a new netcdf model into the program.

        Returns: None
        """
        try:
            self.program.load_netcdf_file_with_dialog()

        except FileNotFoundError:
            self.set_modal_text('Error', 'File not loaded.')

    def load_preview_interpolation_area(self, distance: float) -> None:
        """
        Ask the scene to load the interpolation area for the active polygon.

        Args:
            distance: Distance to use to calculate the interpolation area.

        Returns: None
        """
        if not self.scene.is_polygon_planar(self.get_active_polygon_id()):
            self.set_modal_text('Error', 'Polygon selected is not planar.')
            return

        def load_preview_logic():
            """Logic to load the preview of the polygons."""
            try:
                self.scene.load_preview_interpolation_area(distance, self.get_active_polygon_id())

            except SceneError as e:
                if e.code == 2:
                    self.set_modal_text('Error', 'The polygon must have at least 3 vertices to load the '
                                                 'interpolation area.')

        self.set_loading_message('Loading preview, this may take a while.')
        self.set_task_with_loading_frame(load_preview_logic)

    def load_shapefile_file(self, filename: str) -> None:
        """
        Load the data from a shapefile file and tell the scene to create one or more polygons with the data.

        In case of error, this method shows modal texts with messages.

        The last polygon added is set as teh active polygon.

        Args:
            filename: Name of the shapefile file.

        Returns: None
        """
        polygons_point_list, polygons_param_list = ShapefileImporter().get_polygon_information(filename)

        if polygons_point_list is None and polygons_param_list is None:
            self.set_modal_text('Error', 'An error happened while loading file.')
            return

        if self.get_active_model_id() is None:
            self.set_modal_text('Error', 'Please load a model before loading polygons.')
            return

        try:
            for polygon_points, params in zip(polygons_point_list, polygons_param_list):
                polygon_id = self.scene.create_new_polygon(polygon_points, params)
                self.gui_manager.add_imported_polygon(polygon_id)
                self.set_active_polygon(polygon_id)

        except LineIntersectionError:
            self.set_modal_text('Error', 'One of the polygon loaded intersect itself.')

        except RepeatedPointError:
            self.set_modal_text('Error', 'One of the polygon loaded has repeated points.')

    def load_shapefile_file_with_dialog(self) -> None:
        """
        Call the program to open the dialog to load a shapefile file.

        Returns: None
        """
        try:
            self.program.load_shapefile_file_with_dialog()

        except FileNotFoundError:
            self.set_modal_text('Error', 'File not loaded.')

    def modify_camera_radius(self, distance: float) -> None:
        """
        Ask the scene to get the camera closer to the model.

        Args:
            distance: Distance to get the camera closer.

        Returns: None
        """
        self.scene.modify_camera_radius(distance)

    def move_camera_position(self, movement: tuple) -> None:
        """
        Ask the scene to move the camera position the given movement.

        Args:
            movement: offset to add to the position of the camera. Tuple must have 3 values.

        Returns: None
        """
        self.scene.move_camera(movement)

    def move_map_position(self, x_movement: int, y_movement: int) -> None:
        """
        Tell the scene to move given the parameters specified.

        Args:
            x_movement: Movement in the x-axis
            y_movement: Movement in the y-axis

        Returns: None
        """

        # Get the data to move the maps
        width_scene = self.get_scene_setting_data()['SCENE_WIDTH_X']
        height_scene = self.get_scene_setting_data()['SCENE_HEIGHT_Y']
        showed_limits = self.scene.get_2D_showed_limits()
        map_position = self.get_map_position()

        # Calculate the amount to move the scene depending on the coordinates showed on the screen
        # The more coordinates are showing on the scene, the bigger the movement.
        map_position[0] += (x_movement * (showed_limits['right'] - showed_limits['left'])) / width_scene
        map_position[1] += (y_movement * (showed_limits['top'] - showed_limits['bottom'])) / height_scene

        # Update the position on the program
        self.program.set_map_position(map_position)

        # Update projection matrix
        self.scene.update_projection_matrix_2D()

    def optimize_gpu_memory(self) -> None:
        """
        Call the scene to optimize the GPU memory.

        Optimize the memory regenerating the indices for the current points of the map being showed on the application.
        This method only works when the application is in 2D mode.

        Make an asynchronous call, setting the loading screen.

        Returns: None
        """
        log.debug("Optimizing gpu memory")
        self.program.set_loading(True)
        self.set_loading_message("Deleting triangles from the memory")

        # noinspection PyMissingOrEmptyDocstring
        def then_routine():
            self.program.set_loading(False)

        self.scene.optimize_gpu_memory_async(then_routine)

    def read_netcdf_info(self, filename: str) -> ('np.array', 'np.array', 'np.array'):
        """
        Read the information of a netcdf file and return its contents.

        In case of error reading the file, then NetCDFImportError is raised. This method uses the functionality defined
        in the Input module of the program.

        Returns: Values of the variables X, Y and Z in the file.

        Args:
            filename: Path and name of the file to use.

        """
        return read_info(filename)

    def reload_models(self) -> None:
        """
        Ask the Scene to reload the models to better the definitions.

        This method recalculate the indices array used on the maps in the 2D mode to generate the triangles on the
        model that will be rendered. The new indices are generated so they only generate just the sufficient amount of
        triangles to fill the scene on the current level of zoom.

        After the process, the old indices that were in the same position than the new triangles will be deleted
        using the method optimize_gpu_memory.

        NOTE:
            This method will create a loading frame on the application while the models are being reloaded.

        IMPORTANT:
            This method is asynchronous, this is, the logic that make the reload of the models run in another thread
            while the main thread is still in charge of rendering the program.

        Returns: None
        """
        self.program.set_loading(True)
        self.set_loading_message("Please wait a moment...")

        # noinspection PyMissingOrEmptyDocstring
        def then_routine():
            self.optimize_gpu_memory()  # async function
            self.program.set_loading(False)

        self.scene.reload_models_async(then_routine)

    def remove_interpolation_preview(self, polygon_id: str) -> None:
        """
        Ask the scene to remove the interpolation area of the specified polygon.

        Args:
            polygon_id: Polygon if of the polygon to remove the area to.

        Returns: None
        """
        self.scene.remove_interpolation_preview(polygon_id)

    def remove_model(self, model_id: str) -> None:
        """
        Removes the model from the program.

        This method removes the 2D and 3D if exists. Do nothing if the model id does not exists on the program.

        Args:
            model_id: ID of the model to remove from the program.

        Returns: None
        """
        self.scene.remove_model(model_id)
        self.scene.remove_model_3d(model_id)

        if model_id == self.program.get_active_model():
            self.program.set_active_model(None)

    def reset_camera_values(self) -> None:
        """
        Ask the scene to reset the values of the camera.

        Set the camera values to the  default values.

        Returns: None
        """
        self.scene.reset_camera_values()

    def reset_zoom_level(self) -> None:
        """
        Reset the zoom level of the program.

        Returns: None
        """
        self.program.reset_zoom_level()

    def reset_map_position(self) -> None:
        """
        Set the position of the map to the coordinates (0, 0)

        Returns: None
        """
        self.program.set_map_position([0, 0])

    def run(self, n_frames: int = None, terminate_process: bool = True) -> None:
        """
        Run the main logic of the application.

        Args:
            n_frames: Number of frames to run in the application. None to run until window is closed.
            terminate_process: If terminate the program after the execution of the frames.

        Returns: None
        """
        log.debug("Starting main loop.")

        # Run the app for a fixed number of frames or until the user closes
        if n_frames is not None:
            assert type(n_frames) == int and n_frames > 0

            # Run for the specified amount of frames
            for _ in range(n_frames):
                if glfw.window_should_close(self.window):
                    break

                self.__task_manager.update_tasks()
                self.__thread_manager.update_threads()
                self.__process_manager.update_process()
                self.render.on_loop([lambda: self.scene.draw()])

        else:
            while not glfw.window_should_close(self.window):
                self.__task_manager.update_tasks()
                self.__thread_manager.update_threads()
                self.__process_manager.update_process()
                self.render.on_loop([lambda: self.scene.draw()])

        # Terminate the process if the app ended the process.
        if terminate_process:
            self.program.close()

    def set_active_polygon(self, polygon_id: str or None) -> None:
        """
        Set a new active polygon on the program. Set None to remove the active polygon.

        Args:
            polygon_id: ID of the polygon or None.

        Returns: None
        """
        self.program.set_active_polygon(polygon_id)

    def set_active_tool(self, tool: Union[str, None]) -> None:
        """
        Set the active tool in the program.

        Args:
            tool: String representing the new tool.

        Returns: None
        """
        self.program.set_active_tool(tool)

    def set_loading_message(self, new_msg: str) -> None:
        """
        Change the loading message shown on the screen.

        Args:
            new_msg: New message to show

        Returns: None
        """
        self.gui_manager.set_loading_message(new_msg)

    def set_modal_text(self, title_modal, msg) -> None:
        """
        Set a modal in the program.

        Args:
            title_modal: title of the modal
            msg: message to show in the modal

        Returns: None
        """
        self.gui_manager.set_modal_text(title_modal, msg)

    # noinspection PyUnresolvedReferences
    def set_models_polygon_mode(self, polygon_mode: 'OGLConstant.IntConstant') -> None:
        """
        Call the scene to change the polygon mode used by the models.

        The polygon mode must be one of the following constants defined in the opengl library:
            - GL_POINT
            - GL_LINE
            - GL_FILL

        Args:
            polygon_mode: Polygon mode to use.

        Returns: None
        """
        self.scene.set_models_polygon_mode(polygon_mode)

    def set_new_parameter_to_polygon(self, polygon_id: str, key: str, value: any) -> None:
        """
        Set a new parameter to an existent polygon.

        Args:
            value: value of the parameter.
            key: key of the new value.
            polygon_id: ID of the polygon.

        Returns: None
        """
        self.scene.set_polygon_param(polygon_id, key, value)

    def set_polygon_name(self, polygon_id: str, new_name: str) -> None:
        """
        Change the name of a polygon.

        Args:
            polygon_id: Old polygon id
            new_name: New polygon id

        Returns: None
        """
        self.scene.set_polygon_name(polygon_id, new_name)

    def set_process_task(self, parallel_task: callable, then_task: callable, parallel_task_args=None,
                         then_task_args=None) -> None:
        """
        Creates a new process with the given tasks and start it.

        Args:
            parallel_task: Task to execute in another process.
            then_task: Task to execute after the process. (the return object from the parallel task will be passed as
                       first parameter to this function)
            parallel_task_args: Arguments to give to the parallel task.
            then_task_args: Arguments to give to the then task.

        Returns: None
        """
        if parallel_task_args is None:
            parallel_task_args = []
        if then_task_args is None:
            then_task_args = []

        self.__process_manager.create_parallel_process(parallel_task,
                                                       parallel_task_args,
                                                       then_task,
                                                       then_task_args)

    def set_program_loading(self, new_state: bool = True) -> None:
        """
        Tell the program to set the loading state.

        Args:
            new_state: Boolean indicating the state of the program (if it is loading or not)

        Returns: None
        """
        self.program.set_loading(new_state)

    def set_program_view_mode(self, mode: str = '2D') -> None:
        """
        Set the program view mode to the selected mode.

        Raise ValueError if the mode is an invalid value.

        Args:
            mode: New mode to change to.

        Returns: None
        """
        if mode == '2D':
            self.program.set_view_mode_2D()
            self.render.enable_depth_buffer(False)

        elif mode == '3D':
            self.program.set_view_mode_3D()
            self.render.enable_depth_buffer(True)

            self.set_loading_message('Generating 3D model...')
            self.set_task_with_loading_frame(lambda: self.scene.create_3D_model_if_not_exists())

        else:
            raise ValueError(f'Can not change program view mode to {mode}.')

    def set_task_for_next_frame(self, task: callable) -> None:
        """
        Store a function and executes it in the next frame of the application.

        Args:
            task: Function to execute.

        Returns: None
        """
        self.__task_manager.set_task(task, 2)

    def set_task_with_loading_frame(self, task: callable) -> None:
        """
        Set a task to be executed at the end of the next frame. Also configures the loading setting of
        the program to show the loading frame on the screen.

        Args:
            task: Task to be called in while showing a loading frame.

        Returns: None
        """
        self.program.set_loading(True)

        # noinspection PyMissingOrEmptyDocstring
        def task_loading():
            task()
            self.program.set_loading(False)

        self.__task_manager.set_task(task_loading, 3)

    def set_thread_task(self, parallel_task, then, parallel_task_args=None, then_task_args=None) -> None:
        """
        Add and start a new thread with the current task. At the end of the thread, the then
        function is called.

        If the parallel task return something other than None, then the object returned is given as the first
        parameter to the then task.

        Args:
            then_task_args: List of argument to use in the then task
            parallel_task_args: List of argument to use in the parallel task
            parallel_task: Task to be executed in parallel
            then: Task to be executed in the main thread after the parallel task

        Returns: None
        """
        if self.__use_threads:
            self.__thread_manager.set_thread_task(parallel_task, then, parallel_task_args, then_task_args)
        else:

            if parallel_task_args is None:
                parallel_task_args = []
            if then_task_args is None:
                then_task_args = []

            ret_val = parallel_task(*parallel_task_args)
            if ret_val is not None:
                then(ret_val, *then_task_args)
            else:
                then(*then_task_args)

    def should_use_threads(self, value: bool) -> None:
        """
        Set if the engine should use threads or not.

        If not, then threads logic is executed the moment the thread is set.

        Args:
            value: If to use threads or not.

        Returns: None
        """
        self.__use_threads = value

    def transform_points(self,
                         polygon_id: str,
                         model_id: str,
                         min_height: float,
                         max_height: float,
                         transformation_type: str = 'linear',
                         filters=None) -> None:
        """
        Ask the scene to transform the height of the points of the specified polygon using a linear transformation.

        Transformation types available:
            - linear

        Args:
            filters: List with the filters to use in the modification of the points. List must be in the
                format [(filter_id, args),...]
            transformation_type: Type of transformation to do.
            model_id: ID of the model to use for the interpolation.
            polygon_id: ID of the polygon to use.
            min_height: Min height of the points once converted.
            max_height: Max height of the points once converted.

        Returns: None
        """
        if filters is None:
            filters = []

        try:
            self.scene.transform_points(polygon_id,
                                        model_id,
                                        min_height,
                                        max_height,
                                        transformation_type,
                                        filters)

        except ModelTransformationError as e:
            if e.code == 4:
                self.set_modal_text('Error',
                                    'The current model is not supported to use to update the '
                                    'height of the vertices, try using another type of '
                                    'model.')
            elif e.code == 2:
                self.set_modal_text('Error',
                                    'The polygon must have at least 3 points to be able to '
                                    'modify the heights.')
            elif e.code == 3:
                self.set_modal_text('Error',
                                    'The polygon is not planar. Try using a planar polygon.')
            elif e.code == 6:
                self.set_modal_text('Error',
                                    'Polygon not selected or invalid in filter.')
            elif e.code == 7:
                self.set_modal_text('Error',
                                    'Polygons used in filters must have at least 3 vertices.')
            elif e.code == 8:
                self.set_modal_text('Error',
                                    'One of the polygons used in a filter is not simple/planar.')
            else:
                raise NotImplementedError(f'ModelTransformationError with code {e.code} not handled.')

    def undo_action(self) -> None:
        """
        Undo the most recent action made in the program.

        The logic executed depends on the active tool of the program.

        Returns: None
        """
        active_tool = self.get_active_tool()

        if active_tool == 'create_polygon':
            log.debug('Undoing actions for tool create_polygon.')

            # Ask for the active polygon and call the scene to remove the last added point
            if self.get_active_polygon_id() is not None:
                self.scene.remove_last_point_from_active_polygon()

    def update_current_3D_model(self) -> None:
        """
        Update the 3D model with the information of the model 2D.

        A frame with the loading message is displayed while the process is executed.

        Returns: None
        """
        self.set_loading_message('Getting data from the map 2D...')
        self.set_task_with_loading_frame(lambda: self.scene.update_3D_model(self.program.get_active_model()))

    def update_scene_models_colors(self):
        """
        Update the colors of the models in the scene with the colors that are
        in the ctp file stored in the program.

        Returns: None
        """
        self.scene.update_models_colors()

    def update_scene_values(self) -> None:
        """
        Update the configuration values related to the scene.

        Returns: None
        """
        Settings.update_scene_values()

    def update_scene_viewport(self) -> None:
        """
        Update the scene viewport with the new values that exist in the Settings.

        Returns: None
        """
        self.scene.update_viewport()
