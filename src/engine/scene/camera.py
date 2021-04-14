"""
File with the definition of the camera class, class in charge of the management of the visualization of the 3D models.
"""

import numpy as np
from math import sin, cos, pi
from src.engine.scene.model.tranformations.transformations import lookAt


class Camera:
    """
    Class in charge of generating view matrices for the 3D models to use.
    """

    def __init__(self):
        """
        Constructor of the class.
        """
        self.__radius = 100
        self.__phi = 0  # along the xy plane
        self.__theta = pi/2  # perpendicular to xy plane

        self.__camera_pos = self.__spherical_to_cartesian(self.__radius, self.__phi, self.__theta)
        self.__look_at = np.array([0, 0, 0])
        self.__normal = np.array([0, 0, 1])

    def __spherical_to_cartesian(self, radius, phi, theta) -> np.ndarray:
        """
        Transform spherical coordinates to cartesian ones.

        Args:
            radius: Radius to use.
            phi: phi grades to use. (along xy plane)
            theta: rho grades to use. (perpendicular to xy plane)

        Returns: Cartesian coordinates.
        """
        return np.array([
            radius * sin(theta) * cos(phi),
            radius * sin(theta) * sin(phi),
            radius * cos(theta)
        ])

    def get_view_matrix(self) -> np.ndarray:
        """
        Get the view matrix generated by the camera.

        Returns: Matrix generated by the camera. Numpy Array
        """
        return lookAt(self.__camera_pos,
                      self.__look_at,
                      self.__normal)

    def reset_values(self) -> None:
        """
        Set the camera to its initial configuration.

        Returns: None
        """
        self.__camera_pos = np.array([0, 0, 100])
        self.__look_at = np.array([0, 0, 0])
        self.__normal = np.array([0, 1, 0])

    def modify_radius(self, change_value: float = 1) -> None:
        """
        Make the radius of the camera bigger.

        Use the default value if change_value is none.

        Returns: None
        """
        self.__radius += change_value

        if self.__radius <= 0:
            self.__radius -= change_value

        self.__camera_pos = self.__spherical_to_cartesian(self.__radius,
                                                          self.__phi,
                                                          self.__theta)

    def modify_elevation(self, angle) -> None:
        """
        Modify the elevation of the camera a given angle.

        Args:
            angle: angle to add to the elevation of the camera.

        Returns: None
        """

        self.__theta += angle
        if self.__theta >= pi or self.__theta <= 0:
            self.__theta -= angle

        self.__camera_pos = self.__spherical_to_cartesian(self.__radius,
                                                          self.__phi,
                                                          self.__theta)
        print(self.__camera_pos)
        print(self.__theta / pi)

    def modify_azimuthal_angle(self, angle) -> None:
        """
        Modify the azimuthal angle of the camera. (the angle parallel to the xy plane)

        Args:
            angle: Angle to add to the camera.

        Returns: None
        """
        self.__phi += angle
        self.__camera_pos = self.__spherical_to_cartesian(self.__radius,
                                                          self.__phi,
                                                          self.__theta)
        print(self.__camera_pos)
        print(self.__phi / pi)
