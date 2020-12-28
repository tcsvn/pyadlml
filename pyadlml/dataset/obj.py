from pyadlml.dataset._representations.raw import create_raw
from pyadlml.dataset._representations.changepoint import create_changepoint
from pyadlml.dataset.activities import check_activities

class Data():
    def __init__(self, activities, devices, activity_list, device_list):
        #assert check_activities(activities) 
        #assert check_devices(devices)

        self.df_activities = activities
        self.df_devices = devices

        # list of activities and devices
        self.lst_activities = activity_list
        self.lst_devices = device_list

    def create_cp(self, t_res):
        raise NotImplementedError

    def create_raw(self, t_res=None, idle=False):
        self.df_raw = create_raw(self.df_devices, self.df_activities, t_res)

    def create_lastfired(self):
        raise NotImplementedError