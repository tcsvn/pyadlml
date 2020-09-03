import sys
import unittest

deviceData = "/home/chris/code/adlml/datasets/kasteren/kasterenSenseData.txt"
activityData = "/home/chris/code/adlml/datasets/kasteren/kasterenActData.txt"
sys.path.append("/home/chris/code/adlml/pyadlml")

import pyadlml.dataset.kasteren as kasteren
import pyadlml.dataset._dataset as ds

class TestKasteren(unittest.TestCase):
    def setUp(self):
        self.data = kasteren.load(deviceData, activityData) 
        

    def test_set_up(self):
        pass

    def test_create_raw(self):
        t_res = None
        idle = False
        raw = ds.create_raw(self.data.df_devices,self.data.df_activities,t_res=t_res,idle=idle)
        
        # TODO asserts
        assert raw.shape[0] == 2638 and raw.shape[1] == 15

        # test filling nans with idle
        idle = True
        t_res = None
        raw = ds.create_raw(self.data.df_devices,self.data.df_activities,t_res=t_res,idle=idle)

        assert not raw['activity'].isnull().values.any()
        assert raw.shape[0] == 2638 and raw.shape[1] == 15

        # test time discretization 
        idle = False
        t_res = '30s'
        raw = ds.create_raw(self.data.df_devices,self.data.df_activities,t_res=t_res,idle=idle)
        print(raw)

    
    def test_raw_corr(self):
        """ tests the correlation between raw activities"""
        pass