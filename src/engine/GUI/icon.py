"""
File that contains the class Icon, class in charge of representing the icons to be drawn on the GUI.

Icons are represented as textures of opengl, so this module is OpenGL dependant.
"""
import OpenGL.GL as GL
import numpy as np
from PIL import Image


class Icon:
    """
    Class in charge of load the icons as OpenGL textures.
    """

    def __init__(self, file_dir):
        """
        Constructor of the class
        """
        self.__texture_id = GL.glGenTextures(1)

        # load the image data
        image = Image.open(file_dir)
        img_data = np.array(list(image.getdata()), np.uint8)

        # check the image format
        if image.mode == "RGB":
            internalFormat = GL.GL_RGB
            glformat = GL.GL_RGB
        elif image.mode == "RGBA":
            internalFormat = GL.GL_RGBA
            glformat = GL.GL_RGBA
        else:
            print("Image mode not supported.")
            raise Exception()

        # bind and configure the texture
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.__texture_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_BORDER)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_BORDER)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internalFormat, image.size[0], image.size[1], 0, glformat,
                        GL.GL_UNSIGNED_BYTE,
                        img_data)

    def get_texture_id(self) -> int:
        """
        Get the id of the texture given by opengl.

        Returns: Id of hte texture
        """
        return self.__texture_id