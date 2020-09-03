import sys
import unittest
data_path = "/home/chris/code/adlml/datasets/casas_aruba/corrected_data.csv"
sys.path.append("/home/chris/code/adlml/pyadlml")


import pyadlml.dataset.casas_aruba as ca

class TestCasasAruba(unittest.TestCase):
    def setUp(self):
        self.data = ca.load(data_path) 
        print(self.data)
        

    def test_create_raw(self):
        #tmp = self.data.create_raw()
        pass