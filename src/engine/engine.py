"""
File that contains the program class, class that will be the main class of the program.
"""

import glfw

from src.engine.render.render import Render
from src.utils import get_logger
from src.engine.GUI.guimanager import GUIManager

from src.engine.GUI.frames.sample_text import SampleText
from src.engine.GUI.frames.main_menu_bar import MainMenuBar
from src.engine.scene.scene import Scene
from src.engine.controller.controller import Controller
from src.engine.settings import Settings

log = get_logger(module='PROGRAM')


# TODO: ADD this class to the class diagram.
# TODO: Solve problem in code consistency related to the types in the definitions

class Engine:
    """
    Main class of the program, controls and connect every component of the program.
    """

    def __init__(self):
        """
        Constructor of the program.
        """
        self.render = Render()
        self.gui_manager = GUIManager()
        self.window = None
        self.scene = Scene()
        self.controller = Controller()

    def initialize(self, engine: 'Engine') -> None:
        """
        Initialize the components of the program.
        Returns: None

        Args:
            engine: Engine to initialize.
        """
        log.info('Starting Program')

        # GLFW CODE
        # ---------
        log.debug("Creating windows.")
        self.window = self.render.init("Relief Creator", engine)

        # GUI CODE
        # --------
        log.debug("Loading GUI")
        self.gui_manager.initialize(self.window, engine)
        self.gui_manager.add_frames(
            [
                MainMenuBar(self.gui_manager),
                # TestWindow(),
                SampleText(self.gui_manager)
            ]
        )

        # CONTROLLER CODE
        # ---------------
        self.controller.init(engine)
        glfw.set_key_callback(self.window, self.controller.get_on_key_callback())
        glfw.set_window_size_callback(self.window, self.controller.get_resize_callback())

    @staticmethod
    def change_height_window(height: int) -> None:
        """
        Change the engine settings height for the windows.
        Args:
            height: New height

        Returns: None
        """
        Settings.HEIGHT = height

    @staticmethod
    def change_width_window(width: int)->None:
        """
        Change the engine settings width for the windows
        Args:
            width: New width

        Returns: None
        """
        Settings.WIDTH = width

    @staticmethod
    def update_scene_values()->None:
        """
        Update the configuration values related to the scene.
        Returns: None
        """
        Settings.update_scene_values()

    def run(self) -> None:
        """
        Run the program
        Returns: None
        """
        log.debug("Starting main loop.")
        while not glfw.window_should_close(self.window):
            self.render.on_loop([lambda: self.scene.draw()])

        glfw.terminate()
