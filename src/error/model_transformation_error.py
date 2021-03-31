"""
File with the class ModelTransformationError, class to raise when there is an error in the transformation process.
"""

from src.error.scene_error import SceneError


class ModelTransformationError(SceneError):
    """
    Class to use when there is an error in the transformation process of a model.
    """

    def __init__(self, code: int = 0):
        """
        Constructor of the class.
        """
        super().__init__(code)

        self.codes = {
            0: 'Error description not selected.',
            1: 'Transformation type not recognized by the program.',
            2: 'The polygon used doesnt have at least 3 vertices.',
            3: 'Polygon used is not planar.',
            4: 'Can not use that model for transforming points. Try using a Map2DModel.'
        }

