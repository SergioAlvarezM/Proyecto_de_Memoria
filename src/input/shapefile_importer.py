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
File with the class ShapefileImporter, class in charge of import shapefile files and generate polygons.
"""

import shapefile
from src.utils import is_numeric


class ShapefileImporter:
    """
    Class in charge of the import process of the shapefile files.
    """

    def __init__(self):
        """
        Class constructor.
        """
        pass

    def get_polygon_information(self, filename: str) -> tuple:
        """
        Retrieve the information stored in a shapefile file.

        Return (None, None) if there was a problem while getting the information from the file.

        Convert parameters to the types accepted by the program.
            - str
            - float
            - boolean

        Returns: (list, list) A tuple with the points of the polygons and the parameters stored in them.
        """
        try:
            sf = shapefile.Reader(filename)
        except shapefile.ShapefileException:
            return None, None

        point_list = []
        parameter_list = []

        for shape_record in sf.shapeRecords():
            if shape_record.shape.shapeType == shapefile.POLYGON:

                # dont store the last points 'cause it's the same than the first
                point_list.append(shape_record.shape.points[:-1])

                record_dict = shape_record.record.as_dict()
                for k, v in record_dict.items():
                    # convert the parameter to the types managed for the exporter
                    if v is None:
                        record_dict[k] = ''
                    elif type(v) == bool:
                        pass
                    elif is_numeric(str(v)):
                        record_dict[k] = float(v)
                    else:
                        record_dict[k] = str(v)
                parameter_list.append(record_dict)

        return point_list, parameter_list

if __name__ == '__main__':
    importer = ShapefileImporter()

    points, parameters = importer.get_polygon_information('../../test/input/files/polygons/shape_multiple_parameters.shp')

    data = {
        'points': points,
        'parameteres': parameters
    }

    import json
    with open('../../test/input/files/polygons/shapefile_data_parameters.json', 'w') as f:
        json.dump(data, f)