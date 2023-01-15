# Awesome script
import pandas as pd
from pyadlml.constants import TIME, DEVICE, VALUE
from pyadlml.dataset._core.devices import get_index_matching_rows


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'ToiletFlush')\
     & ('2008-03-04 23:33:56.390200' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-03-04 23:34:01.153100')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'Hall-Bathroom door')\
     & ('2008-03-09 20:37:09.675900' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-03-09 20:49:21.836800')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'Hall-Bathroom door')\
     & ('2008-02-26 00:39:44.407200' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-02-26 00:41:09.807300')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'ToiletFlush')\
     & ('2008-03-04 23:33:50.423000' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-03-04 23:34:15.971400')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'ToiletFlush')\
     & ('2008-03-11 17:54:12.657800' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-03-11 17:54:15.423000')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'ToiletFlush')\
     & ('2008-03-16 00:49:09.691000' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-03-16 00:49:11.566200')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'Washingmachine')\
     & ('2008-03-13 19:34:29.483000' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-03-13 19:34:32.153900')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'Washingmachine')\
     & ('2008-03-13 19:33:58.147700' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-03-13 19:34:03.505600')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'Washingmachine')\
     & ('2008-03-01 21:49:37.563000' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-03-01 21:49:39.919100')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-03-05 19:06:29.692300'), DEVICE:'Microwave', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-03-05 19:05:04.003400'), DEVICE:'Microwave', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['11.03.2008 18:56:00.000001','Microwave',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-03-05 19:06:30.992200'), DEVICE:'Microwave', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-03-11 18:59:07.977600'), DEVICE:'Microwave', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-03-11 18:56:04.002000'), DEVICE:'Microwave', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'Microwave')\
     & ('2008-03-05 09:09:24.720700' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-03-05 09:09:39.490400')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)



#--------------------------------------------------
# Hit_nr.: 1
# Clean cups cupboard debounce 1s 

idx_to_del = get_index_matching_rows(df_devs, 	[['25.02.2008 17:55:10.999998','Cups cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['25.02.2008 17:55:11.999993','Cups cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 1
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['25.02.2008 17:55:12.999998','Cups cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['25.02.2008 17:55:14.999997','Cups cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 2
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['25.02.2008 19:56:20.000000','Cups cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['25.02.2008 19:56:21.000004','Cups cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 3
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['05.03.2008 19:57:32.999996','Cups cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['05.03.2008 19:57:34.000001','Cups cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 3
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['05.03.2008 19:57:39.999998','Cups cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['05.03.2008 19:57:41.000003','Cups cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 4
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['07.03.2008 01:12:21.999995','Cups cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['07.03.2008 01:12:23.000000','Cups cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 4
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['07.03.2008 01:12:25.999994','Cups cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['07.03.2008 01:12:26.999998','Cups cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 4
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['07.03.2008 01:12:32.999996','Cups cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['07.03.2008 01:12:43.999997','Cups cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 5
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['07.03.2008 01:14:19.999999','Cups cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['07.03.2008 01:14:21.000004','Cups cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 8
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['14.03.2008 21:46:22.999992','Cups cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['14.03.2008 21:46:23.999997','Cups cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

#--------------------------------------------------
# Hit_nr.: 1
# Clean debounce Pans Cupboard 

idx_to_del = get_index_matching_rows(df_devs, 	[['25.02.2008 17:17:39.999997','Pans Cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['25.02.2008 17:17:42.999991','Pans Cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 1
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['25.02.2008 17:17:43.999996','Pans Cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['25.02.2008 17:17:48.999999','Pans Cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 2
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['25.02.2008 19:55:59.999998','Pans Cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['25.02.2008 19:56:01.000002','Pans Cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 3
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['27.02.2008 23:42:19.999996','Pans Cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['27.02.2008 23:42:21.000001','Pans Cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 4
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['29.02.2008 20:52:01.999998','Pans Cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['29.02.2008 20:52:03.000002','Pans Cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 4
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['29.02.2008 20:52:14.000003','Pans Cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['29.02.2008 20:52:24.999994','Pans Cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 6
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['05.03.2008 19:56:52.999992','Pans Cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['05.03.2008 19:56:53.999996','Pans Cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 7
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['05.03.2008 20:12:37.000001','Pans Cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['05.03.2008 20:12:39.999995','Pans Cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 8
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['05.03.2008 20:31:06.000001','Pans Cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['05.03.2008 20:31:08.999995','Pans Cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 9
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['15.03.2008 19:43:37.999999','Pans Cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['15.03.2008 19:43:39.000003','Pans Cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 10
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['25.02.2008 09:58:34.000000','Pans Cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['25.02.2008 17:17:38.999993','Pans Cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 9
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['15.03.2008 20:08:32.999997','Pans Cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['15.03.2008 20:08:38.999995','Pans Cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

#--------------------------------------------------
# Hit_nr.: 0
# Clean deboucne plates cupboard 

idx_to_del = get_index_matching_rows(df_devs, 	[['25.02.2008 09:49:27.000000','Plates cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['25.02.2008 09:49:28.000005','Plates cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 1
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['01.03.2008 21:00:32.999992','Plates cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['01.03.2008 21:00:33.999997','Plates cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 5
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['08.03.2008 20:09:49.999992','Plates cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['08.03.2008 20:09:50.999996','Plates cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 6
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['14.03.2008 10:04:02.999996','Plates cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['14.03.2008 10:04:06.000000','Plates cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 6
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['18.03.2008 09:12:13.999992','Plates cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['18.03.2008 09:12:14.999996','Plates cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 0
# Clean debounce Groceries Cupboard 

idx_to_del = get_index_matching_rows(df_devs, 	[['28.02.2008 09:50:33.999999','Groceries Cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['28.02.2008 09:50:35.000004','Groceries Cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 0
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['28.02.2008 09:50:37.999997','Groceries Cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['28.02.2008 09:50:43.000000','Groceries Cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 1
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['28.02.2008 09:51:18.999996','Groceries Cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['28.02.2008 09:51:20.000001','Groceries Cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 2
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['28.02.2008 22:20:58.999997','Groceries Cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['28.02.2008 22:21:00.000002','Groceries Cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 3
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['04.03.2008 09:30:34.999995','Groceries Cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['04.03.2008 09:30:35.999999','Groceries Cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 4
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['05.03.2008 21:35:10.999996','Groceries Cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['05.03.2008 21:35:12.000001','Groceries Cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 4
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['05.03.2008 21:35:17.999998','Groceries Cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['05.03.2008 21:35:28.999999','Groceries Cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 5
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['11.03.2008 09:20:14.999998','Groceries Cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['11.03.2008 09:20:16.000003','Groceries Cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 6
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['11.03.2008 18:39:17.999994','Groceries Cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['11.03.2008 18:39:18.999999','Groceries Cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 7
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['14.03.2008 10:04:20.999999','Groceries Cupboard','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['14.03.2008 10:04:22.000004','Groceries Cupboard','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-02-25 17:17:45.002900'), DEVICE:'Pans Cupboard', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-02-25 09:58:37.988500'), DEVICE:'Pans Cupboard', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-02-25 23:11:36.732500'), DEVICE:'Pans Cupboard', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-02-25 19:56:08.006200'), DEVICE:'Pans Cupboard', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['25.02.2008 23:11:37.999999','Pans Cupboard',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['25.02.2008 23:11:40.999993','Pans Cupboard',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'Pans Cupboard')\
     & ('2008-03-15 20:08:50.540500' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-03-16 08:29:15.684800')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-03-14 10:04:07.964500'), DEVICE:'Groceries Cupboard', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-03-14 00:32:06.579350'), DEVICE:'Groceries Cupboard', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 0
# Debounce front door 

idx_to_del = get_index_matching_rows(df_devs, 	[['26.02.2008 20:31:34.999994','Frontdoor','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['26.02.2008 20:31:35.999999','Frontdoor','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


idx_to_del = get_index_matching_rows(df_devs, 	[['26.02.2008 22:10:50.999996','Frontdoor','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['26.02.2008 22:10:52.000001','Frontdoor','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['06.03.2008 09:29:10.999998','Frontdoor','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['06.03.2008 09:29:12.000002','Frontdoor','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)



#--------------------------------------------------
# Hit_nr.: 2
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['29.02.2008 22:03:23.000002','Frontdoor','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['29.02.2008 22:03:46.999993','Frontdoor','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 3
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['07.03.2008 11:36:02.999999','Frontdoor','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['07.03.2008 11:36:05.999993','Frontdoor','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 5
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['11.03.2008 20:26:56.999993','Frontdoor','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['11.03.2008 20:26:57.999998','Frontdoor','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)



#--------------------------------------------------
# Hit_nr.: 5
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['09.03.2008 18:35:12.999995','Frontdoor','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['09.03.2008 18:35:15.999999','Frontdoor','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 7
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['16.03.2008 23:39:19.000002','Frontdoor','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['16.03.2008 23:39:23.999995','Frontdoor','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 9
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['18.03.2008 22:49:22.000004','Frontdoor','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['18.03.2008 22:49:23.999993','Frontdoor','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 3
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['08.03.2008 10:29:41.999997','Frontdoor','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['08.03.2008 10:29:43.000002','Frontdoor','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 3
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['11.03.2008 19:07:24.999997','Frontdoor','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['11.03.2008 19:07:26.000002','Frontdoor','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 4
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['17.03.2008 10:09:10.999995','Frontdoor','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['17.03.2008 10:09:12.000000','Frontdoor','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 21
# Merge bedroom door small events 

idx_to_del = get_index_matching_rows(df_devs, 	[['06.03.2008 08:38:55.999996','Hall-Bedroom door','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['06.03.2008 08:38:59.000000','Hall-Bedroom door','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 22
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['07.03.2008 10:26:10.999995','Hall-Bedroom door','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['07.03.2008 10:26:13.999999','Hall-Bedroom door','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 46
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['18.03.2008 06:57:51.999997','Hall-Bedroom door','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['18.03.2008 06:57:54.999991','Hall-Bedroom door','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['25.02.2008 21:08:50.000002','Hall-Bathroom door',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['25.02.2008 21:08:50.999997','Hall-Bathroom door',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'Hall-Bathroom door')\
     & ('2008-02-26 00:38:59.612200' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-02-26 00:39:13.076000')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['27.02.2008 08:26:35.000002','Hall-Bathroom door',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['27.02.2008 08:26:37.999996','Hall-Bathroom door',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['28.02.2008 10:46:22.000005','Hall-Bathroom door',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.02.2008 10:46:24.999998','Hall-Bathroom door',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'Hall-Bathroom door')\
     & ('2008-02-29 10:31:06.986500' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-02-29 10:31:14.754300')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['05.03.2008 10:30:14.999999','Hall-Bathroom door',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['05.03.2008 10:30:17.999993','Hall-Bathroom door',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['05.03.2008 23:14:50.000004','Hall-Bathroom door',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['05.03.2008 23:14:53.999992','Hall-Bathroom door',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'Hall-Bathroom door')\
     & ('2008-03-13 19:44:09.516800' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-03-13 19:44:11.614800')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'Hall-Bathroom door')\
     & ('2008-03-14 22:02:29.495200' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-03-14 22:02:36.397800')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['19.03.2008 18:24:39.000004','Hall-Bathroom door',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['19.03.2008 18:24:49.999995','Hall-Bathroom door',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'Hall-Toilet door')\
     & ('2008-02-25 18:32:06.141800' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-02-25 18:32:14.435700')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'Hall-Toilet door')\
     & ('2008-02-28 00:13:50.814100' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-02-28 00:13:54.002900')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['01.03.2008 21:58:00.000004','Hall-Toilet door',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['01.03.2008 21:58:02.999998','Hall-Toilet door',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'Hall-Toilet door')\
     & ('2008-03-15 09:14:08.614700' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-03-15 09:14:13.876500')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['18.03.2008 23:58:47.000004','Hall-Toilet door',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['18.03.2008 23:58:47.999998','Hall-Toilet door',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

