from pyadlml.dataset._representations.raw import create_raw
from pyadlml.dataset._representations.changepoint import create_changepoint
from pyadlml.dataset.activities import check_activities

class Data():
    def __init__(self, activities, devices):
        #assert check_activities(activities) 
        #assert check_devices(devices)

        self.df_activities = activities
        self.df_devices = devices

        # list of activities and devices
        #self.activities = list(activities.activity.unique())
        #self.devices = list(devices.device.unique())
        self.df_raw = None
        self.df_cp = None
        self.df_lf = None

    def create_cp(self, t_res):
        raise NotImplementedError

    def create_raw(self, t_res=None, idle=False):
        self.df_raw = create_raw(self.df_devices, self.df_activities, t_res)

    def create_lastfired(self):
        raise NotImplementedError