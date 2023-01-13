# Awesome script
import pandas as pd
from pyadlml.constants import TIME, DEVICE, VALUE
from pyadlml.dataset._core.devices import get_index_matching_rows


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['22.11.2012 09:29:14.000000','Bedroom Bed Pressure',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2012-11-22 06:26:27.194985900'), DEVICE:'Bedroom Bed Pressure', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2012-11-30 10:30:29.892600'), DEVICE:'Bathroom Toilet Flush', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2012-11-30 10:27:00.195400'), DEVICE:'Bathroom Toilet Flush', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['29.11.2012 16:18:03.000000','Bathroom Toilet Flush',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2012-11-29 16:13:15.715100080'), DEVICE:'Bathroom Toilet Flush', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['29.11.2012 09:56:53.000000','Bathroom Toilet Flush',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2012-11-29 09:51:54.713100'), DEVICE:'Bathroom Toilet Flush', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['25.11.2012 13:45:27.000000','Bathroom Toilet Flush',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2012-11-25 13:42:59.498300'), DEVICE:'Bathroom Toilet Flush', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2012 17:40:24.000000','Bathroom Toilet Flush',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2012-11-28 17:44:32.240800'), DEVICE:'Bathroom Toilet Flush', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['27.11.2012 10:22:54.000000','Kitchen Fridge Magnetic',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['27.11.2012 10:22:55.000000','Kitchen Fridge Magnetic',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['17.11.2012 20:46:41.000000','Bedroom Door PIR',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['17.11.2012 20:46:42.000000','Bedroom Door PIR',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['01.12.2012 21:13:53.000000','Bedroom Door PIR',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['01.12.2012 21:13:54.000000','Bedroom Door PIR',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2012-11-24 00:24:51.160761200'), DEVICE:'Living Door PIR', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2012-11-23 22:17:41.404723400'), DEVICE:'Living Door PIR', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
