"""
File that do fake imports of the classes used in type hinting to make them available for the
pycharm linting.

usage:

from src.type_hinting import *

or

from src.type_hinting import SOMECLASS
"""

if False:
    from src.engine.GUI.guimanager import GUIManager
    from src.engine.engine import Engine