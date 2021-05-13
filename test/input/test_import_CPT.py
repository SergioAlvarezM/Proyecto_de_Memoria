import unittest
import os
import json

from src.input.CTP import read_file

FILES_DIRECTORY = './test/input/files/'

class TestReadCPTFile(unittest.TestCase):

    def test_reading_normal_files(self):

        with open(os.path.join(FILES_DIRECTORY, 'test_data_CPT_1.json')) as f:
            data_1 = json.load(f)
            data_read = read_file(os.path.join(FILES_DIRECTORY, 'test_cpt_1.cpt'))
            self.assertEqual(data_1, data_read, 'Data readed from CPT file is not what is expected.')

        with open(os.path.join(FILES_DIRECTORY, 'test_data_CPT_2.json')) as f:
            data_2 = json.load(f)
            data_read = read_file(os.path.join(FILES_DIRECTORY, 'test_cpt_2.cpt'))
            self.assertEqual(data_2, data_read, 'Data readed from CPT file is not what is expected.')

if __name__ == '__main__':
    unittest.main()
