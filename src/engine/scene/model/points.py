"""
File with the class Points, class in charge of storing all the information related to the models that draw points on
the scene
"""

from src.engine.scene.model.model import Model
from src.utils import get_logger

import numpy as np
import OpenGL.GL as GL
import ctypes as ctypes

log = get_logger(module="Points")


class Points(Model):
    """
    Class in charge of the modeling of points in the program.
    """

    def __init__(self, scene):
        """
        Constructor of the class
        """
        super().__init__(scene)
        self.cbo = GL.glGenBuffers(1)

        self.draw_mode = GL.GL_POINTS

        self.update_uniform_values = False

        self.__vertex_shader_file = './engine/shaders/point_vertex.glsl'
        self.__fragment_shader_file = './engine/shaders/point_fragment.glsl'

        self.__point_list = []
        self.__indices_list = []
        self.__color_list = []
        self.__normal_color = (1, 1, 0, 1)  # RGBA
        self.__first_point_color = (1, 0, 0, 1)  # RGBA
        self.__last_point_color = (0, 0, 1, 1)  # RGBA

        self.set_shaders(self.__vertex_shader_file, self.__fragment_shader_file)

    def _update_uniforms(self) -> None:
        """
        Update the uniforms values for the model.

        Returns: None
        """

        # update values for the polygon shader
        # ------------------------------------
        projection_location = GL.glGetUniformLocation(self.shader_program, "projection")

        # set the color and projection matrix to use
        # ------------------------------------------
        GL.glUniformMatrix4fv(projection_location, 1, GL.GL_TRUE, self.scene.get_active_model_projection_matrix())

    def add_point(self, x: float, y: float, z: float) -> None:
        """
        Add a point to the list to draw

        Args:
            x: x position of the point
            y: y position of the point
            z: z position of the point

        Returns: None
        """

        # update the vertices buffer
        self.__point_list.append(x)
        self.__point_list.append(y)
        self.__point_list.append(z)
        self.set_vertices(np.array(self.__point_list, dtype=np.float32))

        # update  the color buffer
        if len(self.__color_list) / 4 == 0:
            self.__color_list.append(self.__first_point_color[0])
            self.__color_list.append(self.__first_point_color[1])
            self.__color_list.append(self.__first_point_color[2])
            self.__color_list.append(self.__first_point_color[3])

        elif len(self.__color_list) / 4 == 1:
            self.__color_list.append(self.__last_point_color[0])
            self.__color_list.append(self.__last_point_color[1])
            self.__color_list.append(self.__last_point_color[2])
            self.__color_list.append(self.__last_point_color[3])

        else:
            self.__color_list.pop()
            self.__color_list.pop()
            self.__color_list.pop()
            self.__color_list.pop()
            self.__color_list.append(self.__normal_color[0])
            self.__color_list.append(self.__normal_color[1])
            self.__color_list.append(self.__normal_color[2])
            self.__color_list.append(self.__normal_color[3])
            self.__color_list.append(self.__last_point_color[0])
            self.__color_list.append(self.__last_point_color[1])
            self.__color_list.append(self.__last_point_color[2])
            self.__color_list.append(self.__last_point_color[3])

        self.set_color_buffer(np.array(self.__color_list, dtype=np.float32))

        # update the indices buffer
        self.__indices_list.append(len(self.__point_list) / 3 - 1)
        self.set_indices(np.array(self.__indices_list, dtype=np.uint32))

    def draw(self) -> None:
        """
        Draw the points on the scene

        Returns: None
        """

        # draw if there is at least one point
        if len(self.__point_list) / 3 > 0:
            # get the settings of the points to draw
            render_settings = self.scene.get_render_settings()
            dot_size = render_settings["DOT_SIZE"]
            polygon_dot_size = render_settings["POLYGON_DOT_SIZE"]

            # draw the points
            GL.glPointSize(polygon_dot_size)
            super().draw()
            GL.glPointSize(dot_size)

    def get_first_point_color(self) -> tuple:
        """
        get the color to use in the drawing of the first point

        Returns: tuple with the color in rgb format
        """
        return self.__first_point_color

    def get_last_point_color(self) -> tuple:
        """
        get the color to use in the drawing of the last point

        Returns: tuple with the color in rgb format
        """
        return self.__last_point_color

    def get_normal_color(self) -> tuple:
        """
        get the normal color to use in the drawing of the points

        Returns: tuple with the color in rgb format
        """
        return self.__normal_color

    def get_point_list(self) -> list:
        """
        Get the point list

        Returns: Point list
        """
        return self.__point_list

    def set_color_buffer(self, colors: np.ndarray) -> None:
        """
        Set the color buffer in opengl with the colors of the array

        Args:
            colors: Colors to use in the buffer

        Returns: None
        """
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.cbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            len(colors) * self.scene.get_float_bytes(),
            colors,
            GL.GL_STATIC_DRAW
        )
        GL.glVertexAttribPointer(
            1, 4, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0)
        )
        GL.glEnableVertexAttribArray(1)

    def set_first_point_color(self, new_color: tuple) -> None:
        """
        Set the color to use to coloring the first point.

        Args:
            new_color: New color to use for the first point. (R, G, B)

        Returns: None
        """
        self.__first_point_color = new_color

        # update the color buffer if there is at least one point defined
        if len(self.__point_list) / 3 > 0:
            self.__color_list[0] = new_color[0]
            self.__color_list[1] = new_color[1]
            self.__color_list[2] = new_color[2]
            self.__color_list[3] = new_color[3]

            self.set_color_buffer(np.array(self.__color_list, dtype=np.float32))

    def set_last_point_color(self, new_color: tuple) -> None:
        """
        Set the color to use in the last point

        Args:
            new_color: New color to use in the last point. (R, G, B)

        Returns: None
        """
        self.__last_point_color = new_color

        # update the color buffer if there is at least two point defined
        if len(self.__point_list) / 3 > 1:
            self.__color_list.pop()
            self.__color_list.pop()
            self.__color_list.pop()
            self.__color_list.pop()

            self.__color_list.append(new_color[0])
            self.__color_list.append(new_color[1])
            self.__color_list.append(new_color[2])
            self.__color_list.append(new_color[3])

            self.set_color_buffer(np.array(self.__color_list, dtype=np.float32))

    def set_normal_color(self, new_color: tuple) -> None:
        """
        Set the normal color to use to coloring the points.

        Args:
            new_color: New color to use. (R, G, B)

        Returns: None
        """
        self.__normal_color = new_color

        # update the color buffer if there is at least three points defined
        if len(self.__point_list) / 3 > 2:
            number_of_point = int(len(self.__point_list) / 3)

            self.__color_list = []

            # append the initial color
            self.__color_list.append(self.__first_point_color[0])
            self.__color_list.append(self.__first_point_color[1])
            self.__color_list.append(self.__first_point_color[2])
            self.__color_list.append(self.__first_point_color[3])

            # append the new color to the list
            for _ in range(number_of_point - 2):
                self.__color_list.append(self.__normal_color[0])
                self.__color_list.append(self.__normal_color[1])
                self.__color_list.append(self.__normal_color[2])
                self.__color_list.append(self.__normal_color[3])

            # append the last color
            self.__color_list.append(self.__last_point_color[0])
            self.__color_list.append(self.__last_point_color[1])
            self.__color_list.append(self.__last_point_color[2])
            self.__color_list.append(self.__last_point_color[3])

            # update the color buffer
            self.set_color_buffer(np.array(self.__color_list, dtype=np.float32))
