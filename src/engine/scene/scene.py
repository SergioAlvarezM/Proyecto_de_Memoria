"""
File that contain the Scene class. This class is in charge of the management of the models of the scene.
"""
import OpenGL.GL as GL
import OpenGL.constant as OGLConstant

from src.engine.scene.model.map2dmodel import Map2DModel
from src.engine.scene.model.model import Model
from src.input.NetCDF import read_info
from src.utils import get_logger

log = get_logger(module="SCENE")


class Scene:

    def __init__(self):
        """
        Constructor of the class.
        """
        self.__model_list = []
        self.__engine = None

        self.__width_viewport = None
        self.__height_viewport = None

        # auxiliary variables
        # -------------------
        self.__should_execute_then_reload = 0

    def initialize(self, engine: 'Engine') -> None:
        """
        Initialize the component in the engine.
        Args:
            engine: Engine to use

        Returns: None
        """
        self.__engine = engine

    def update_viewport_variables(self):
        """
        Update the viewport variables.

        Returns: None
        """
        viewport_data = self.__engine.get_scene_setting_data()
        self.__width_viewport = viewport_data['SCENE_WIDTH_X']
        self.__height_viewport = viewport_data['SCENE_HEIGHT_Y']

    def draw(self) -> None:
        """
        Draw the models in the list.

        Draw the models in the order of the list.
        Returns: None
        """
        for model in self.__model_list:
            model.draw()

    def add_model(self, model: Model) -> None:
        """
        Add a model to the list of models.
        Args:
            model: Model to add to the list

        Returns: None
        """
        self.__model_list.append(model)

    def remove_all_models(self) -> None:
        """
        Remove all models from the list of models.
        Returns: None
        """
        self.__model_list = []

    def reload_models_async(self, then):
        """
        Ask the 2D models to reload with the new resolution of the screen.

        Returns: None

        Args:
            then: Function to be called after all the async routine.
        """

        # set up the then routine that should executes only once.
        # -------------------------------------------------------
        self.__should_execute_then_reload = len(self.__model_list)

        def then_routine():
            self.__should_execute_then_reload -= 1

            if self.__should_execute_then_reload == 0:
                self.__should_execute_then_reload = len(self.__model_list)
                then()

        for model in self.__model_list:
            if isinstance(model, Map2DModel):
                model.recalculate_vertices_from_grid_async(quality=self.__engine.get_quality(), then=then_routine)
            else:
                raise NotImplementedError("This type of model doesnt have a method to be reloaded.")

    def remove_model(self, id_model: str) -> None:
        """
        Return the model with the specified id.
        Args:
            id_model: Id of the model to remove.

        Returns: None
        """
        model_to_remove = None
        for model in self.__model_list:
            if model.id == id_model:
                model_to_remove = model

        self.__model_list.remove(model_to_remove)

    def set_polygon_mode(self, polygon_mode: OGLConstant.IntConstant) -> None:
        """
        Select if the models uses the wireframe mode or not.
        Args:
            polygon_mode: OpenGL polygon mode to draw the model.

        Returns: None
        """
        for model in self.__model_list:
            model.polygon_mode = polygon_mode

    def update_viewport(self) -> None:
        """
        Update the viewport with the new values that exist in the Settings.
        """
        log.debug("Updating viewport")
        self.update_viewport_variables()

        scene_data = self.__engine.get_scene_setting_data()
        GL.glViewport(scene_data['SCENE_BEGIN_X'], scene_data['SCENE_BEGIN_Y'], scene_data['SCENE_WIDTH_X'],
                      scene_data['SCENE_HEIGHT_Y'])

        self.update_models_projection_matrix()

    def update_models_projection_matrix(self) -> None:
        """
        Update the projection matrix of the models.

        Returns: None
        """
        log.debug("Updating projection matrix of models.")
        scene_data = self.__engine.get_scene_setting_data()

        for model in self.__model_list:
            if isinstance(model, Map2DModel):
                model.calculate_projection_matrix(scene_data, self.__engine.get_zoom_level())
            else:
                raise NotImplementedError("Not implemented method to update the projection matrix in this model.")

    def set_parallel_task(self, parallel_task, then):
        """
        Set a parallel task in the engine.

        Args:
            parallel_task: Task to execute in parallel
            then: Task to execute after the parallel task

        Returns: None
        """
        self.__engine.set_thread_task(parallel_task, then)

    def refresh_with_model_2d_async(self, path_color_file: str, path_model: str, model_id: str = 'main',
                                    then=lambda: None) -> 'Model':
        """
        Refresh the scene, removing all the models, and adding the new model specified
        in the input.

        The model added will be added as a 2d model to the program with the id 'main'.

        The model must  be in netCDF format.

        The color file must be in CTP format.

        Args:
            then: Function to be executed at the end of the async routine
            model_id: Model ID to use in the model added
            path_color_file: Path to the CTP file with the colors
            path_model: Path to the model to use in the application

        Returns: None

        """

        log.debug("Reading information from file.")
        X, Y, Z = read_info(path_model)

        log.debug("Generating model")
        model = Map2DModel(self)

        def then_routine():
            log.debug("Settings colors from file.")
            model.set_color_file(path_color_file)
            model.calculate_projection_matrix(self.__engine.get_scene_setting_data())
            model.wireframes = False
            model.id = model_id

            self.remove_all_models()
            self.add_model(model)

            self.__engine.reset_zoom_level()

            # call the then routine
            then()

        log.debug("Setting vertices from grid.")
        model.set_vertices_from_grid_async(X, Y, Z, self.__engine.get_quality(), then_routine)

    def get_float_bytes(self) -> int:
        """
        Get the float bytes used in a float to render.
        Ask the engine for this information (that is stored in the settings).

        Returns: Number of bytes used for store float numbers.
        """
        return self.__engine.get_float_bytes()

    def get_scene_setting_data(self) -> dict:
        """
        Get the scene settings.

        Returns: None
        """
        return self.__engine.get_scene_setting_data()

    def update_models_colors(self) -> None:
        """
        Update the colors of the models reloading the colors from the file used in the program.

        Returns: None
        """
        color_file = self.__engine.get_CPT_file()
        for model in self.__model_list:
            model.set_color_file(color_file)
