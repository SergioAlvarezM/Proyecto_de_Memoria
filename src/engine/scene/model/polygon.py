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
File containing the class polygon.

This class stores all the information related to the polygons that can be draw on the screen of the program.
"""
from src.engine.scene.model.model import Model
from src.utils import get_logger
from src.engine.scene.model.points import Points
from src.engine.scene.model.lines import Lines
from src.engine.scene.model.dashed_lines import DashedLines
from src.error.line_intersection_error import LineIntersectionError
from src.error.repeated_point_error import RepeatedPointError

import numpy as np
import OpenGL.GL as GL
from shapely.geometry import LineString
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point as ShapelyPoint

log = get_logger(module="POLYGON")


class Polygon(Model):
    """
    Class in charge of the polygons of the program.

    Default values for z-axis coordinate of points in the polygon is 0.5. Unless in 3D mode, this value must be
    set to the third coordinate so the methods defined inside the class works.
    """

    def __init__(self, scene, id_polygon: str, point_list: list = None, parameters: dict = None):
        """
        Constructor of the class.

        Generate a new polygon with or without data.

        The default

        Args:
            id_polygon: Id to use to identify the polygon on the program.
            point_list: List of initial points to use in the polygon. [[x,y],[x,y],...]
            parameters: Dictionary with initial parameters to set in the polygon. {parameter_name: value,...}
        """
        super().__init__(scene)

        self.id = id_polygon
        self.draw_mode = GL.GL_LINES

        self.update_uniform_values = False

        self.__name = self.get_id()

        self.__point_model = Points(scene)  # model to use to draw the points
        self.__lines_model = Lines(scene)  # model to use to draw the lines
        self.__last_line_model = DashedLines(scene)  # model to use to render the last line of the polygon

        self.__parameters = parameters if parameters is not None else {}

        self.__is_planar = True

        # initialize polygon if data is given
        if point_list is not None:

            # check for consistency on the data
            test_line = LineString(point_list)
            if not test_line.is_simple:
                raise LineIntersectionError()

            if len(point_list) != len(set(point_list)):
                raise RepeatedPointError()

            # prepare the data
            point_array = np.array(point_list)
            point_array = point_array.reshape((-1, 2))

            points = np.empty((len(point_list), 3))
            points[:, 0] = point_array[:, 0]
            points[:, 1] = point_array[:, 1]
            points[:, 2] = 0.5  # unless in 3D, height of the points must be kept in 0.5

            self.__point_model = Points(scene, points)
            self.__lines_model = Lines(scene, point_list=points)
            self.update_last_line(False)

    def __check_intersection(self, points: list = None) -> bool:
        """
        Join the points given in a line and check if there is intersections.

        Args:
            points: List with the points to check. [x,y,z,x,y,z,x,y,z,...]
        """
        points_array = np.array(points)
        points_array = points_array.reshape((-1, 3))
        return not LineString(points_array).is_simple

    def __check_repeated_point(self, x: float, y: float, z: float) -> bool:
        """
        Check if a point already exist on the polygon.

        Args:
            x: x-coordinate of the point
            y: y-coordinate of the point
            z: z-coordinate of the point

        Returns: Boolean representing if point already exist in the polygon.
        """
        point_list_array = np.array(self.get_point_list() + [x, y, z])
        point_list_array = point_list_array.reshape((-1, 3))
        return len(np.unique(point_list_array, axis=0)) != len(point_list_array)

    def __str__(self):
        """
        Format how polygons are printed on the console.

        Returns: String representing the polygon.
        """
        return f"Points model: {self.__point_model}\nLines model: {self.__lines_model}"

    def __update_planar_state(self):
        """
        Update the variable indicating if the polygon is planar or not.

        Returns: None
        """
        if self.get_point_number() > 2:
            # ask for intersections on the closed polygon.
            point_list = self.get_point_list()
            if self.__check_intersection(points=point_list + [point_list[0], point_list[1], point_list[2]]):
                self.__is_planar = False
            else:
                self.__is_planar = True

    def __get_internal_points_in_shapely_format(self) -> ShapelyPolygon:
        """
        Get the points in the polygon in the shapely.Polygon format.

        Returns: Polygon with the points of the polygon.
        """
        list_of_points = self.get_point_list()

        # delete the z axis from the points
        new_list = []
        pair_used = []
        for component_ind in range(len(list_of_points)):
            if component_ind % 3 == 0:
                pair_used.append(list_of_points[component_ind])
            elif component_ind % 3 == 1:
                pair_used.append(list_of_points[component_ind])
            elif component_ind % 3 == 2:
                new_list.append(pair_used)
                pair_used = []

        return ShapelyPolygon(new_list)

    def __get_external_polygon_points_in_shapely_format(self, distance: float) -> ShapelyPolygon:
        """
        Get the external polygon given the specified distance in shapely.Polygon format.

        Args:
            distance: Distance to use to calculate the external polygon.

        Returns: External polygon in shapely.Polygon format.
        """
        polygon_shapely = self.__get_internal_points_in_shapely_format()
        external_polygon = polygon_shapely.buffer(distance).exterior
        return ShapelyPolygon(external_polygon)

    def get_exterior_polygon_points(self, distance: float) -> list:
        """
        Get the external polygon generated using the distance specified.

        Polygon is returned in shapely format.

        WARNING: this method create the polygon in each call. Try not to use this in loops.

        Args:
            distance: Distance to use to generate the external polygon.

        Returns: Points used for the exterior polygon.
        """
        external_polygon = self.__get_external_polygon_points_in_shapely_format(distance).exterior
        x_coords, y_coords = external_polygon.xy

        polygon_exterior = []
        for x_coordinate, y_coordinate in zip(x_coords, y_coords):
            polygon_exterior.append(x_coordinate)
            polygon_exterior.append(y_coordinate)
            polygon_exterior.append(0.5)

        return polygon_exterior

    def add_point(self, x: float, y: float, z: float = 0.5) -> None:
        """
        Add a new point to the list of points.

        Unless used in 3D, the z value must be keep 0.5.

        Args:
            x: x position of the point
            y: y position of the point
            z: z position of the point (default to 0.5)

        Returns: None
        """

        # check if point is already on the polygon
        if self.__check_repeated_point(x, y, z):
            raise RepeatedPointError()

        # check if lines intersect
        if self.get_point_number() > 2:
            point_list = self.get_point_list()

            # do not let the creation of lines that intersect
            if self.__check_intersection(point_list + [x, y, z]):
                raise LineIntersectionError()

            # if the completion line intersect, then change the state of the polygon.
            if self.__check_intersection(point_list + [x, y, z, point_list[0], point_list[1], point_list[2]]):
                self.__is_planar = False
            else:
                self.__is_planar = True

        # add points to the point model
        self.__point_model.add_point(x, y, z)

        # list of points
        point_list = self.get_point_list()

        # draw line only if there is more than one point
        if self.get_point_number() > 1:
            self.__lines_model.add_line((point_list[-6], point_list[-5], point_list[-4]),
                                        (point_list[-3], point_list[-2], point_list[-1]))

        # if there is 3 points, just add a new line connecting the polygon
        if self.get_point_number() == 3:
            self.update_last_line(False)

        # if there is more than 3 points, then delete the line and add a new one
        if self.get_point_number() > 3:
            self.update_last_line()

    def delete_parameter(self, key: str) -> None:
        """
        Delete a parameter from the dictionary of parameters.

        Args:
            key: Key to be deleted.

        Returns: None
        """
        self.__parameters.pop(key)

    def draw(self) -> None:
        """
        Set how and when to draw the polygons.
        """

        # draw the components on the screen
        if self.scene.get_active_polygon_id() == self.id:
            self.__lines_model.set_use_borders(True)
            self.__last_line_model.set_use_borders(True)
        else:
            self.__lines_model.set_use_borders(False)
            self.__last_line_model.set_use_borders(False)

        self.__lines_model.draw()
        self.__last_line_model.draw()
        self.__point_model.draw()

    def get_id(self) -> str:
        """
        Get the id of the polygon.

        Returns: Id of the polygon
        """
        return self.id

    def get_name(self) -> str:
        """
        Get the name of the polygon.

        Returns: Name of the polygon
        """
        return self.__name

    def get_parameter(self, key: str) -> any:
        """
        Get the parameter from the polygon.

        Args:
            key: Key of the parameter.

        Returns: Value of the parameter. None if parameter does not exist.
        """
        return self.__parameters.get(key)

    def get_parameter_list(self) -> list:
        """
        Return all the parameters of the polygon as a list.

        Returns: List with the parameters [(key, value), (key, value), ...]
        """
        return [(k, v) for k, v in self.__parameters.items()]

    def get_point_list(self) -> list:
        """
        Get the list of points.
        The format of the list is as follows: [x1, y1, z1, x2, y2, z2, ...]

        Returns: List of points
        """
        return self.__point_model.get_point_list()

    def get_point_number(self) -> int:
        """
        Return the number of points of the polygon.

        Returns: Number of points of the polygon.
        """
        # return int(len(self.__point_list) / 3)
        return int(len(self.get_point_list()) / 3)

    def is_planar(self) -> bool:
        """
        Check if the polygon is planar or not

        Returns: Boolean indicating if the polygon is planar or not.
        """
        return self.__is_planar

    def remove_last_added_point(self) -> None:
        """
        Remove the last point added to the polygon.

        Does nothing if there is no points.

        Returns: None
        """

        # only works when there is points in the model
        if self.get_point_number() > 0:
            self.__point_model.remove_last_added_point()
            self.__lines_model.remove_last_added_line()

            if self.get_point_number() < 3:
                self.__last_line_model.remove_last_added_line()
            else:
                self.update_last_line()

            self.__update_planar_state()

    def set_dot_color(self, color: list) -> None:
        """
        Set the color to draw the dots of the polygon.

        The color must be in a list-like object in the order of RGBA with values between 0 and 1.

        Args:
            color: Color to be used by the polygon

        Returns: None
        """
        log.debug(f"Changing polygon dot color to {color}")
        self.__point_model.set_normal_color(tuple(color))

    def set_id(self, new_id: str) -> None:
        """
        Set a new id for the polygon.

        Args:
            new_id: New ID of the polygon.

        Returns: None
        """
        self.id = new_id

    def set_line_color(self, color: list) -> None:
        """
        Set the color to draw the lines of the polygon.

        The color must be in a list-like object in the order of RGBA with values between 0 and 1.

        Args:
            color: Color to be used by the polygon

        Returns: None
        """
        log.debug(f"Changing polygon color to {color}")
        self.__lines_model.set_line_color(color)
        self.__last_line_model.set_line_color(color)

    def set_name(self, new_name: str) -> None:
        """
        Set a new name for the polygon.

        Args:
            new_name: New name of the polygon

        Returns: None
        """
        self.__name = new_name

    def set_new_parameter(self, key: str, value: str) -> None:
        """
        Set a new parameter to be stored in the polygon.

        Args:
            key: Key for the parameter
            value: Value to store

        Returns: None
        """
        self.__parameters[key] = value

    def update_last_line(self, remove_last_line: bool = True) -> None:
        """
        Update the last line of the polygon.

        Args:
            remove_last_line: True to remove the last line from the line model, False to just add a new line to the
                model.

        Returns: None
        """

        # remove the last line if exist
        point_list = self.get_point_list()
        if remove_last_line:
            self.__last_line_model.remove_last_added_line()

        # add a new line
        self.__last_line_model.add_line((point_list[-3], point_list[-2], point_list[-1]),
                                        (point_list[0], point_list[1], point_list[2]))
