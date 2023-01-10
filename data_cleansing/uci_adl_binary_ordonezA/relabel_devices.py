# Awesome script
import pandas as pd
from pyadlml.constants import TIME, DEVICE, VALUE
from pyadlml.dataset._core.devices import get_index_matching_rows

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2011 10:38:33.000000','Kitchen Fridge Magnetic',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2011 10:38:40.000000','Kitchen Fridge Magnetic',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['29.11.2011 11:31:04.000000','Bedroom Bed Pressure',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2011-11-29 08:51:07.077100'), DEVICE:'Bedroom Bed Pressure', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['29.11.2011 16:34:17.000000','Living Seat Pressure',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['29.11.2011 17:08:07.000000','Living Seat Pressure',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['29.11.2011 17:09:09.000000','Living Seat Pressure',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2011-11-29 16:28:42.890600'), DEVICE:'Living Seat Pressure', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['30.11.2011 01:22:33.000000','Bedroom Bed Pressure',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2011-11-30 00:25:46.104400'), DEVICE:'Bedroom Bed Pressure', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['30.11.2011 14:11:16.000000','Bathroom Toilet Flush',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['30.11.2011 14:11:48.000000','Bathroom Toilet Flush',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['30.11.2011 14:11:22.000000','Bathroom Basin PIR',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['30.11.2011 14:11:24.000000','Bathroom Basin PIR',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2011-11-30 14:11:25.118800'), DEVICE:'Bathroom Basin PIR', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2011-11-30 14:11:20.495200'), DEVICE:'Bathroom Basin PIR', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['05.12.2011 16:02:10.000000','Living Seat Pressure',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['05.12.2011 16:37:15.000000','Living Seat Pressure',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2011-12-05 16:51:44.486700'), DEVICE:'Living Seat Pressure', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2011-12-05 16:09:55.222400'), DEVICE:'Living Seat Pressure', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['05.12.2011 16:51:44.486700','Living Seat Pressure',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2011-12-05 17:02:39.901500'), DEVICE:'Living Seat Pressure', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['05.12.2011 19:01:42.000000','Living Seat Pressure',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['05.12.2011 19:24:02.000000','Living Seat Pressure',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['05.12.2011 19:24:22.000000','Living Seat Pressure',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2011-12-05 18:58:02.218300'), DEVICE:'Living Seat Pressure', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
