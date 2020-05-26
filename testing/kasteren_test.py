import sys
import unittest

deviceData = "/home/chris/code/adlml/datasets/kasteren/kasterenSenseData.txt"
activityData = "/home/chris/code/adlml/datasets/kasteren/kasterenActData.txt"
sys.path.append("/home/chris/code/adlml/pyadlml")

import pyadlml.dataset.kasteren as kasteren

class TestKasteren(unittest.TestCase):
    def setUp(self):
        self.data = kasteren.load(deviceData, activityData)

    def test_create_raw(self):
        tmp = self.data.create_raw()