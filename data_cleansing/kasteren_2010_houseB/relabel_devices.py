# Awesome script
import pandas as pd
from pyadlml.constants import TIME, DEVICE, VALUE
from pyadlml.dataset._core.devices import get_index_matching_rows


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['28.07.2009 18:35:30.000004','cupboard plates',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.07.2009 18:36:04.999995','cupboard plates',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['28.07.2009 18:34:42.000003','cupboard plates',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.07.2009 18:34:44.000002','cupboard plates',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.07.2009 18:34:46.999996','cupboard plates',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.07.2009 18:34:51.999999','cupboard plates',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.07.2009 18:34:53.000003','cupboard plates',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.07.2009 18:35:04.999999','cupboard plates',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['01.08.2009 22:24:37.000002','cupboard plates',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['01.08.2009 22:24:38.009996','cupboard plates',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['02.08.2009 09:33:48.999998','cupboard plates',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['02.08.2009 09:33:50.999998','cupboard plates',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['03.08.2009 15:11:39.999995','cupboard plates',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.08.2009 15:11:41.999994','cupboard plates',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['04.08.2009 12:42:53.999999','cupboard plates',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['04.08.2009 12:43:11.999992','cupboard plates',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['04.08.2009 12:43:36.999998','cupboard plates',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2009-08-04 12:43:52.117000'), DEVICE:'cupboard plates', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['25.07.2009 13:21:28.999996','toaster',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['25.07.2009 13:21:30.000000','toaster',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['25.07.2009 13:21:46.999999','toaster',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['25.07.2009 13:21:48.000003','toaster',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['25.07.2009 13:23:09.999991','toaster',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['25.07.2009 13:23:10.999996','toaster',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['25.07.2009 13:26:09.999990','toaster',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['25.07.2009 13:26:10.999995','toaster',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)



#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['28.07.2009 10:14:23.000003','toaster',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2009-07-26 12:21:19.986000'), DEVICE:'toaster', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2009-07-30 11:24:47.955600'), DEVICE:'toaster', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2009-07-28 10:22:20.979800'), DEVICE:'toaster', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2009-07-31 09:49:41.030300'), DEVICE:'toaster', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2009-07-30 11:31:29.980700'), DEVICE:'toaster', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2009-08-01 11:20:42.036600'), DEVICE:'toaster', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2009-07-31 09:58:56.967200'), DEVICE:'toaster', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['01.08.2009 11:25:52.999999','toaster',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['01.08.2009 11:25:54.999998','toaster',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2009-08-01 22:23:12.054300'), DEVICE:'toaster', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2009-08-01 11:27:22.023900'), DEVICE:'toaster', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2009-08-04 14:27:07.904900'), DEVICE:'toaster', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2009-08-04 14:27:06.388600'), DEVICE:'toaster', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2009-08-04 12:50:55.972900'), DEVICE:'toaster', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#--------------------------------------------------
# Hit_nr.: 0
# Clean device debounce Bedroom door 

#idx_to_del = get_index_matching_rows(df_devs, 	[['25.07.2009 02:55:26.999992','Bedroom door','True']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['25.07.2009 02:55:27.999997','Bedroom door','False']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#
##--------------------------------------------------
## Hit_nr.: 1
## 
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['26.07.2009 03:35:34.999997','Bedroom door','True']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['26.07.2009 03:35:36.000001','Bedroom door','False']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#
##--------------------------------------------------
## Hit_nr.: 3
## 
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['28.07.2009 11:02:16.000000','Bedroom door','True']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['28.07.2009 11:02:17.000005','Bedroom door','False']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#
##--------------------------------------------------
## Hit_nr.: 5
## 
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['29.07.2009 18:08:41.999995','Bedroom door','True']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['29.07.2009 18:08:43.000000','Bedroom door','False']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#
##--------------------------------------------------
## Hit_nr.: 5
## 
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['29.07.2009 18:08:47.999993','Bedroom door','True']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['29.07.2009 18:08:48.999997','Bedroom door','False']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#
##--------------------------------------------------
## Hit_nr.: 6
## 
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['31.07.2009 09:45:22.999996','Bedroom door','True']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['31.07.2009 09:45:24.000001','Bedroom door','False']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#
##--------------------------------------------------
## Hit_nr.: 7
## 
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['01.08.2009 05:20:40.999991','Bedroom door','True']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['01.08.2009 05:20:41.999996','Bedroom door','False']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#
##--------------------------------------------------
## Hit_nr.: 7
## 
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['01.08.2009 05:20:27.999991','Bedroom door','True']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['01.08.2009 05:20:29.009996','Bedroom door','False']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#
##--------------------------------------------------
## Hit_nr.: 7
## 
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['01.08.2009 05:20:21.999994','Bedroom door','True']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['01.08.2009 05:20:22.999998','Bedroom door','False']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#
##--------------------------------------------------
## Hit_nr.: 7
## 
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['01.08.2009 05:20:19.000000','Bedroom door','True']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['01.08.2009 05:20:20.000004','Bedroom door','False']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#
##--------------------------------------------------
## Hit_nr.: 8
## 
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['02.08.2009 04:44:27.999999','Bedroom door','True']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['02.08.2009 04:44:29.000004','Bedroom door','False']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#
##--------------------------------------------------
## Hit_nr.: 9
## 
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['02.08.2009 05:02:01.999995','Bedroom door','True']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['02.08.2009 05:02:03.000000','Bedroom door','False']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#
##--------------------------------------------------
## Hit_nr.: 10
## 
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['02.08.2009 10:33:56.999992','Bedroom door','True']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['02.08.2009 10:33:57.999997','Bedroom door','False']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#
##--------------------------------------------------
## Hit_nr.: 10
## 
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['02.08.2009 10:33:58.999991','Bedroom door','True']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['02.08.2009 10:33:59.999996','Bedroom door','False']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#
##--------------------------------------------------
## Hit_nr.: 10
## 
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['02.08.2009 10:34:20.999992','Bedroom door','True']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['02.08.2009 10:34:21.999997','Bedroom door','False']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#
##--------------------------------------------------
## Hit_nr.: 11
## 
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['03.08.2009 04:25:55.999996','Bedroom door','True']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['03.08.2009 04:25:57.000000','Bedroom door','False']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#
##--------------------------------------------------
## Hit_nr.: 12
## 
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['04.08.2009 00:12:05.000000','Bedroom door','True']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['04.08.2009 00:12:06.000005','Bedroom door','False']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#
##--------------------------------------------------
## Hit_nr.: 13
## 
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['05.08.2009 04:25:39.999992','Bedroom door','True']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)
#
#idx_to_del = get_index_matching_rows(df_devs, 	[['05.08.2009 04:25:40.999997','Bedroom door','False']])
#df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['29.07.2009 11:51:35.999998','frontdoor',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['29.07.2009 11:51:37.000002','frontdoor',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2009-07-29 11:51:38.522000'), DEVICE:'frontdoor', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2009-07-29 11:51:35.152800'), DEVICE:'frontdoor', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'frontdoor')\
     & ('2009-08-03 04:05:13.481300' < df_devs[TIME])\
     & (df_devs[TIME] < '2009-08-03 04:05:15.972400')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['04.08.2009 16:27:00.000001','frontdoor',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['04.08.2009 16:27:00.999995','frontdoor',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'frontdoor')\
     & ('2009-08-05 04:28:05.209800' < df_devs[TIME])\
     & (df_devs[TIME] < '2009-08-05 04:28:07.397500')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)



#--------------------------------------------------
# Hit_nr.: 0
# Clean debounce signal fridge 

idx_to_del = get_index_matching_rows(df_devs, 	[['25.07.2009 11:13:26.999991','fridge','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['25.07.2009 11:13:27.999996','fridge','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 2
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['27.07.2009 12:53:20.999998','fridge','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['27.07.2009 12:53:22.000003','fridge','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 13
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['29.07.2009 18:09:55.999996','fridge','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['29.07.2009 18:09:57.000001','fridge','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['29.07.2009 18:09:57.999995','fridge','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['29.07.2009 18:09:59.000000','fridge','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 16
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['29.07.2009 18:15:29.999995','fridge','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['29.07.2009 18:15:30.999999','fridge','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 16
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['29.07.2009 18:15:31.999994','fridge','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['29.07.2009 18:15:32.999999','fridge','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 21
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['31.07.2009 09:50:39.999996','fridge','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['31.07.2009 09:50:41.000001','fridge','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 31
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['02.08.2009 09:34:55.999997','fridge','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['02.08.2009 09:34:57.000002','fridge','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 32
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['04.08.2009 12:43:15.999991','fridge','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['04.08.2009 12:43:16.999995','fridge','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 32
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['04.08.2009 12:43:21.999998','fridge','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['04.08.2009 12:43:23.000003','fridge','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 33
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['04.08.2009 12:45:00.000000','fridge','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['04.08.2009 12:45:01.000005','fridge','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 33
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['04.08.2009 12:45:22.999996','fridge','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['04.08.2009 12:45:24.000001','fridge','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'cutlery drawer kwick')\
     & ('2009-07-25 03:10:47.160800' < df_devs[TIME])\
     & (df_devs[TIME] < '2009-07-25 03:11:25.724800')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'cutlery drawer kwick')\
     & ('2009-07-24 21:25:20.754998' < df_devs[TIME])\
     & (df_devs[TIME] < '2009-07-25 02:54:19.545400')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'cutlery drawer kwick')\
     & ('2009-07-25 02:54:31.722762700' < df_devs[TIME])\
     & (df_devs[TIME] < '2009-07-25 06:54:01.584992')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'cutlery drawer kwick')\
     & ('2009-07-25 07:04:37.957187200' < df_devs[TIME])\
     & (df_devs[TIME] < '2009-07-25 11:08:25.158991')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'cutlery drawer kwick')\
     & ('2009-07-24 21:19:31.875200' < df_devs[TIME])\
     & (df_devs[TIME] < '2009-07-24 21:20:37.605400')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['24.07.2009 20:33:19.999998','cutlery drawer kwick',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2009-07-24 21:14:42.988800'), DEVICE:'cutlery drawer kwick', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'kwik dresser')\
     & ('2009-08-04 00:12:33.475916' < df_devs[TIME])\
     & (df_devs[TIME] < '2009-08-04 05:52:47.583596')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'kwik dresser')\
     & ('2009-08-04 05:55:08.960976' < df_devs[TIME])\
     & (df_devs[TIME] < '2009-08-04 12:40:37.748990500')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'kwik stove lid')\
     & ('2009-07-24 20:22:22.179100' < df_devs[TIME])\
     & (df_devs[TIME] < '2009-07-24 20:22:25.234300')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'fridge')\
     & ('2009-07-31 22:22:41.749200' < df_devs[TIME])\
     & (df_devs[TIME] < '2009-07-31 22:22:43.569100')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2009-07-29 18:11:01.969200'), DEVICE:'microwave', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2009-07-29 11:55:33.985400'), DEVICE:'microwave', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2009-07-29 18:16:51.998900'), DEVICE:'microwave', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2009-07-29 18:11:11.982300'), DEVICE:'microwave', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'microwave')\
     & ('2009-07-30 11:23:28.501700' < df_devs[TIME])\
     & (df_devs[TIME] < '2009-07-31 09:48:21.595300')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2009-07-31 09:50:15.014100'), DEVICE:'microwave', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2009-07-31 09:48:30.979100'), DEVICE:'microwave', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['31.07.2009 09:50:19.999994','microwave',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['31.07.2009 09:50:20.999999','microwave',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['31.07.2009 09:50:26.999997','microwave',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2009-08-02 09:30:20.016100'), DEVICE:'microwave', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'microwave')\
     & ('2009-08-02 09:30:58.467000' < df_devs[TIME])\
     & (df_devs[TIME] < '2009-08-02 09:34:24.694000')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'microwave')\
     & ('2009-08-02 09:30:45.112300' < df_devs[TIME])\
     & (df_devs[TIME] < '2009-08-02 09:30:50.217500')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['27.07.2009 23:24:20.000000','toilet flush',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2009-07-27 13:21:59.974100'), DEVICE:'toilet flush', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['26.07.2009 14:46:09.000001','balcony door',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['27.07.2009 12:53:46.999998','balcony door',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['27.07.2009 12:53:59.999998','balcony door',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.07.2009 11:06:07.999999','balcony door',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['28.07.2009 20:14:26.999998','balcony door',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.07.2009 20:14:28.000003','balcony door',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.07.2009 20:14:32.999996','balcony door',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.07.2009 20:14:34.000000','balcony door',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.07.2009 20:14:45.999995','balcony door',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.07.2009 20:14:47.000000','balcony door',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['29.07.2009 18:32:17.999999','balcony door',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2009-07-29 11:56:17.117700'), DEVICE:'balcony door', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'balcony door')\
     & ('2009-08-03 15:13:19.850400' < df_devs[TIME])\
     & (df_devs[TIME] < '2009-08-03 15:13:20.667800')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'balcony door')\
     & ('2009-07-26 14:45:56.432700' < df_devs[TIME])\
     & (df_devs[TIME] < '2009-07-26 14:46:19.483100')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['02.08.2009 10:53:22.999994','balcony door',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['02.08.2009 10:53:24.009999','balcony door',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'frontdoor')\
     & ('2009-07-29 11:51:33.900600' < df_devs[TIME])\
     & (df_devs[TIME] < '2009-07-29 11:51:36.322600')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['30.07.2009 11:22:54.999999','fridge',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2009-07-30 11:21:39.984100'), DEVICE:'fridge', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['25.07.2009 11:24:08.999999','fridge',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['25.07.2009 14:29:42.000004','fridge',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['04.08.2009 14:22:43.999998','kwik dresser',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2009-08-04 14:22:35.985000'), DEVICE:'kwik dresser', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat server corner')\
     & ('2009-08-04 14:27:06.866400' < df_devs[TIME])\
     & (df_devs[TIME] < '2009-08-04 14:27:07.272500')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 0
# Clean debounce toilet door 

idx_to_del = get_index_matching_rows(df_devs, 	[['24.07.2009 20:22:04.999998','toilet door','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['24.07.2009 20:22:06.000003','toilet door','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 1
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['25.07.2009 13:03:34.999997','toilet door','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['25.07.2009 13:03:36.000002','toilet door','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 4
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['27.07.2009 12:30:26.999998','toilet door','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['27.07.2009 12:30:28.000002','toilet door','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 5
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['28.07.2009 10:51:16.999994','toilet door','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['28.07.2009 10:51:17.999998','toilet door','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 7
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['29.07.2009 10:41:13.999995','toilet door','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['29.07.2009 10:41:15.000000','toilet door','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 7
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['29.07.2009 10:41:25.999991','toilet door','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['29.07.2009 10:41:26.999995','toilet door','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 17
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['02.08.2009 09:45:16.999998','toilet door','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['02.08.2009 09:45:18.000003','toilet door','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 18
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['03.08.2009 04:23:54.999998','toilet door','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['03.08.2009 04:23:56.000003','toilet door','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 19
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['03.08.2009 05:40:13.999996','toilet door','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['03.08.2009 05:40:15.000000','toilet door','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 19
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['03.08.2009 05:40:23.999992','toilet door','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['03.08.2009 05:40:24.999996','toilet door','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 19
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['03.08.2009 05:40:17.999994','toilet door','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['03.08.2009 05:40:18.999999','toilet door','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 21
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['04.08.2009 23:45:37.999998','toilet door','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['04.08.2009 23:45:39.000003','toilet door','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'press bed right')\
     & ('2009-07-29 04:59:51.726500' < df_devs[TIME])\
     & (df_devs[TIME] < '2009-07-30 11:01:47.533000')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'press bed left')\
     & ('2009-08-01 05:19:28.353300' < df_devs[TIME])\
     & (df_devs[TIME] < '2009-08-01 05:19:41.276000')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'press bed left')\
     & ('2009-07-28 20:28:28.577700' < df_devs[TIME])\
     & (df_devs[TIME] < '2009-07-28 20:29:03.506300')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'press bed left')\
     & ('2009-08-05 04:10:22.065700' < df_devs[TIME])\
     & (df_devs[TIME] < '2009-08-05 04:11:09.650600')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'press bed right')\
     & ('2009-08-05 04:10:44.020700' < df_devs[TIME])\
     & (df_devs[TIME] < '2009-08-05 04:11:21.389200')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'press bed right')\
     & ('2009-08-02 04:44:26.764000' < df_devs[TIME])\
     & (df_devs[TIME] < '2009-08-02 04:44:45.134900')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)

