# Awesome script
import pandas as pd
from pyadlml.constants import TIME, DEVICE, VALUE
from pyadlml.dataset._core.devices import get_index_matching_rows


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['21.11.2008 21:30:54.999997','front door reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['21.11.2008 21:32:12.000002','front door reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-24 13:10:50.349734100'), DEVICE:'cupboard leftovers reed', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-05 15:53:25.739733'), DEVICE:'cupboard leftovers reed', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-04 16:55:11.621000'), DEVICE:'cupboard leftovers reed', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-05 15:56:26.102200'), DEVICE:'cupboard leftovers reed', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['05.12.2008 15:52:49.000001','pans cupboard reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-07 10:24:58.569080'), DEVICE:'pans cupboard reed', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-20 17:33:00.724177200'), DEVICE:'pans cupboard reed', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['01.12.2008 17:35:09.000000','pans cupboard reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['01.12.2008 17:39:56.999997','pans cupboard reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 
new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-06 00:07:46.541100'), DEVICE:'freezer reed', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-20 17:26:58.494800'), DEVICE:'freezer reed', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)



#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:58:38.000002','freezer reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['04.12.2008 17:05:15.999999','freezer reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['04.12.2008 17:08:47.000001','freezer reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['04.12.2008 17:12:02.000000','freezer reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['04.12.2008 17:08:39.999999','freezer reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['04.12.2008 17:08:40.999994','freezer reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)



#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['30.11.2008 14:28:32.999996','freezer reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:48:06.000000','freezer reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-30 14:28:28.443200'), DEVICE:'freezer reed', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-28 20:03:52.184104'), DEVICE:'freezer reed', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['26.11.2008 23:52:29.999997','freezer reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-28 20:03:48.970661600'), DEVICE:'freezer reed', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2008 20:03:48.970661','freezer reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2008 20:03:48.999998','freezer reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:49:09.999995','freezer reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:53:04.999998','freezer reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['06.12.2008 00:07:46.541100','freezer reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-06 00:04:25.028400'), DEVICE:'freezer reed', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['04.12.2008 17:09:59.000003','microwave reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['24.11.2008 07:33:39.999994','microwave reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['24.11.2008 07:56:18.999995','microwave reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['24.11.2008 13:11:25.000003','microwave reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['24.11.2008 13:12:00.000004','microwave reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2008 20:04:27.999998','microwave reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['30.11.2008 14:28:50.000004','microwave reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['30.11.2008 14:28:50.999999','microwave reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['04.12.2008 17:05:43.000004','microwave reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['04.12.2008 17:05:43.999998','microwave reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['01.12.2008 08:42:54.000003','cabinet cups/bowl/tuna reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-02 23:17:21.023900'), DEVICE:'cabinet cups/bowl/tuna reed', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['04.12.2008 22:39:57.000001','cabinet cups/bowl/tuna reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['05.12.2008 16:57:32.999996','cabinet cups/bowl/tuna reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['06.12.2008 00:06:43.999991','cabinet cups/bowl/tuna reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-05 16:57:42.034700'), DEVICE:'cabinet cups/bowl/tuna reed', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['20.11.2008 10:42:49.999998','cabinet cups/bowl/tuna reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-20 10:42:50.169500'), DEVICE:'cabinet cups/bowl/tuna reed', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-20 10:42:42.978900'), DEVICE:'cabinet cups/bowl/tuna reed', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['06.12.2008 10:57:03.999992','cabinet cups/bowl/tuna reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-06 08:46:36.982100'), DEVICE:'cabinet cups/bowl/tuna reed', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['21.11.2008 11:39:21.999996','cabinet cups/bowl/tuna reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['21.11.2008 11:39:23.999995','cabinet cups/bowl/tuna reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['21.11.2008 11:39:25.000000','cabinet cups/bowl/tuna reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['21.11.2008 11:39:25.999994','cabinet cups/bowl/tuna reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['04.12.2008 22:38:54.000000','cabinet cups/bowl/tuna reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['04.12.2008 22:38:54.999995','cabinet cups/bowl/tuna reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-23 05:31:41.725900'), DEVICE:'toilet flush upstairs', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-23 05:30:43.649400'), DEVICE:'toilet flush upstairs', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['23.11.2008 05:30:43.000005','toilet flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-22 09:39:39.817600'), DEVICE:'toilet flush upstairs', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['26.11.2008 08:17:26.999997','toilet flush upstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['26.11.2008 08:17:29.999991','toilet flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2008 07:15:04.999996','toilet flush upstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2008 07:16:02.999994','toilet flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2008 07:16:03.999998','toilet flush upstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2008 07:16:04.999993','toilet flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2008 07:16:05.999997','toilet flush upstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2008 07:16:07.999997','toilet flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2008 07:16:09.000001','toilet flush upstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2008 07:16:13.999994','toilet flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2008 07:16:14.999999','toilet flush upstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2008 07:16:26.999994','toilet flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2008 07:16:27.999999','toilet flush upstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2008 07:16:50.999995','toilet flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2008 07:16:51.999999','toilet flush upstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2008 07:17:27.999995','toilet flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-20 01:32:55.077200'), DEVICE:'toilet flush upstairs', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-20 01:32:08.013200'), DEVICE:'toilet flush upstairs', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['20.11.2008 01:32:08.000000','toilet flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2008 07:15:03.999992','toilet flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-28 07:14:44.452900'), DEVICE:'toilet flush upstairs', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['29.11.2008 16:58:07.000003','toilet flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-29 16:56:53.554400'), DEVICE:'toilet flush upstairs', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['22.11.2008 09:39:39.817600','toilet flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-22 09:39:59.552200'), DEVICE:'toilet flush upstairs', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)



#--------------------------------------------------
# Relabel dresser pir debounce
# Hit_nr.: 0
#

idx_to_del = get_index_matching_rows(df_devs, 	[['20.11.2008 09:33:39.999997','dresser pir','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['20.11.2008 09:33:41.000002','dresser pir','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 6
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['24.11.2008 07:43:34.999995','dresser pir','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['24.11.2008 07:43:36.000000','dresser pir','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['24.11.2008 07:43:39.999998','dresser pir','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['24.11.2008 07:43:41.000003','dresser pir','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 7
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['26.11.2008 08:24:09.999994','dresser pir','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['26.11.2008 08:24:10.999999','dresser pir','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 9
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['29.11.2008 19:12:51.999994','dresser pir','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['29.11.2008 19:12:52.999998','dresser pir','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 17
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['30.11.2008 13:49:38.999992','dresser pir','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['30.11.2008 13:49:39.999997','dresser pir','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 23
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['04.12.2008 07:06:48.999991','dresser pir','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['04.12.2008 07:06:49.999996','dresser pir','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['27.11.2008 07:05:27.000003','dresser pir',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2008 07:20:49.000001','dresser pir',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-28 07:20:49.991400'), DEVICE:'dresser pir', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-28 07:20:48.995500'), DEVICE:'dresser pir', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['02.12.2008 07:10:44.999998','dresser pir',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-30 13:49:43.127200'), DEVICE:'dresser pir', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['05.12.2008 07:17:25.000001','dresser pir',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-04 07:06:53.022600'), DEVICE:'dresser pir', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['06.12.2008 12:15:58.000004','dresser pir',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-06 08:32:21.004100'), DEVICE:'dresser pir', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['20.11.2008 09:34:07.000001','dresser pir',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-20 09:33:44.841700'), DEVICE:'dresser pir', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['25.11.2008 22:03:49.999996','toilet flush downstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-25 22:03:45.022400'), DEVICE:'toilet flush downstairs', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['27.11.2008 20:20:07.999999','toilet flush downstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-27 20:19:52.015800'), DEVICE:'toilet flush downstairs', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 22:05:49.000005','toilet flush downstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-03 22:05:33.257000'), DEVICE:'toilet flush downstairs', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-20 01:37:10.463600'), DEVICE:'door bedroom', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['21.11.2008 06:05:26.000002','door bedroom',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-21 11:19:23.269500'), DEVICE:'door bedroom', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-23 09:22:36.389000'), DEVICE:'door bedroom', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-23 06:19:14.576200'), DEVICE:'door bedroom', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['26.11.2008 19:18:52.000002','door bedroom',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['27.11.2008 00:10:56.000002','door bedroom',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['27.11.2008 18:14:14.000005','door bedroom',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-28 00:23:14.984400'), DEVICE:'door bedroom', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-29 07:20:04.663900'), DEVICE:'door bedroom', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-28 23:48:14.429700'), DEVICE:'door bedroom', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['30.11.2008 09:39:26.999996','door bedroom',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-30 08:53:56.088800'), DEVICE:'door bedroom', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 00:37:46.999995','door bedroom',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 06:00:48.999996','door bedroom',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 06:38:34.999996','door bedroom',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['04.12.2008 00:14:56.999993','door bedroom',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['04.12.2008 00:15:06.999999','door bedroom',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['04.12.2008 06:57:47.000004','door bedroom',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-05 06:59:09.038400'), DEVICE:'door bedroom', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-05 03:15:30.307200'), DEVICE:'door bedroom', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-05 03:15:10.282000'), DEVICE:'door bedroom', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-04 19:39:55.178600'), DEVICE:'door bedroom', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['05.12.2008 07:00:01.999996','door bedroom',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['05.12.2008 18:18:53.000000','door bedroom',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-06 08:29:47.075500'), DEVICE:'door bedroom', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-06 04:50:26.761800'), DEVICE:'door bedroom', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 
new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-07 08:10:12.873600'), DEVICE:'door bedroom', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['20.11.2008 13:19:07.999996','toilet door downstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-20 11:04:12.754200'), DEVICE:'toilet door downstairs', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-29 11:26:30.750500'), DEVICE:'toilet door downstairs', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-28 22:41:35.868100'), DEVICE:'toilet door downstairs', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['29.11.2008 11:26:52.999995','toilet door downstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['29.11.2008 11:26:54.000000','toilet door downstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['29.11.2008 11:26:54.999994','toilet door downstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-29 11:26:35.017700'), DEVICE:'toilet door downstairs', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['02.12.2008 00:42:30.999993','toilet door downstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-01 18:07:25.921900'), DEVICE:'toilet door downstairs', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['05.12.2008 15:40:55.999996','toilet door downstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-05 07:04:19.069700'), DEVICE:'toilet door downstairs', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-05 15:40:19.975400'), DEVICE:'toilet door downstairs', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-05 15:40:16.550900'), DEVICE:'toilet door downstairs', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['24.11.2008 16:47:11.000000','toilet door downstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['24.11.2008 16:47:12.000005','toilet door downstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['01.12.2008 08:33:35.999992','toilet door downstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['01.12.2008 08:33:36.999996','toilet door downstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['19.11.2008 22:51:02.999999','toilet door downstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['23.11.2008 21:58:28.999997','toilet door downstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['23.11.2008 21:58:30.000002','toilet door downstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['19.11.2008 22:51:04.000004','toilet door downstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-19 22:51:01.169600'), DEVICE:'toilet door downstairs', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#--------------------------------------------------
# Hit_nr.: 0
# Cleanup washbasin deboucne

idx_to_del = get_index_matching_rows(df_devs, 	[['20.11.2008 01:33:42.999997','washbasin flush upstairs','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['20.11.2008 01:33:44.000002','washbasin flush upstairs','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 6
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['22.11.2008 18:23:03.999997','washbasin flush upstairs','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['22.11.2008 18:23:05.000001','washbasin flush upstairs','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 6
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['22.11.2008 18:23:09.999994','washbasin flush upstairs','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['22.11.2008 18:23:10.999999','washbasin flush upstairs','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 9
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['22.11.2008 18:29:57.000000','washbasin flush upstairs','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['22.11.2008 18:29:58.000004','washbasin flush upstairs','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 12
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['23.11.2008 09:38:49.999995','washbasin flush upstairs','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['23.11.2008 09:38:51.000000','washbasin flush upstairs','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 13
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['24.11.2008 00:17:42.999998','washbasin flush upstairs','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['24.11.2008 00:17:44.999997','washbasin flush upstairs','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 13
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['24.11.2008 00:17:14.999999','washbasin flush upstairs','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['24.11.2008 00:17:16.000003','washbasin flush upstairs','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 13
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['24.11.2008 00:17:18.999997','washbasin flush upstairs','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['24.11.2008 00:17:20.000002','washbasin flush upstairs','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 14
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['24.11.2008 00:20:42.999997','washbasin flush upstairs','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['24.11.2008 00:20:44.000002','washbasin flush upstairs','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 15
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['24.11.2008 09:34:35.000000','washbasin flush upstairs','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['24.11.2008 09:34:36.000005','washbasin flush upstairs','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 16
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['24.11.2008 18:25:07.999998','washbasin flush upstairs','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['24.11.2008 18:25:09.000003','washbasin flush upstairs','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 16
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['24.11.2008 18:25:09.999997','washbasin flush upstairs','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['24.11.2008 18:25:11.000002','washbasin flush upstairs','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 16
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['24.11.2008 18:25:12.999991','washbasin flush upstairs','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['24.11.2008 18:25:13.999996','washbasin flush upstairs','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 16
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['24.11.2008 18:24:25.999994','washbasin flush upstairs','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['24.11.2008 18:24:26.999999','washbasin flush upstairs','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 20
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['26.11.2008 00:51:40.999998','washbasin flush upstairs','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['26.11.2008 00:51:42.000002','washbasin flush upstairs','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 22
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['27.11.2008 00:09:07.999995','washbasin flush upstairs','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['27.11.2008 00:09:08.999999','washbasin flush upstairs','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 26
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['28.11.2008 19:33:12.999999','washbasin flush upstairs','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['28.11.2008 19:33:14.000004','washbasin flush upstairs','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 31
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['29.11.2008 18:58:42.999992','washbasin flush upstairs','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['29.11.2008 18:58:43.999996','washbasin flush upstairs','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 35
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['30.11.2008 14:53:23.999995','washbasin flush upstairs','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['30.11.2008 14:53:25.000000','washbasin flush upstairs','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 43
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['04.12.2008 00:09:43.999991','washbasin flush upstairs','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['04.12.2008 00:09:44.999995','washbasin flush upstairs','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['20.11.2008 17:20:19.999994','front door reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['20.11.2008 17:20:20.999999','front door reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['20.11.2008 01:34:05.999993','washbasin flush upstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['20.11.2008 01:34:06.999998','washbasin flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['21.11.2008 06:04:28.999999','washbasin flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['21.11.2008 06:04:30.999999','washbasin flush upstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['21.11.2008 06:04:32.000003','washbasin flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['21.11.2008 06:04:32.999998','washbasin flush upstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['21.11.2008 18:24:08.000002','washbasin flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-21 18:24:10.503100'), DEVICE:'washbasin flush upstairs', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['22.11.2008 01:29:58.000001','washbasin flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-22 01:29:59.994000'), DEVICE:'washbasin flush upstairs', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['22.11.2008 18:33:58.000000','washbasin flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['22.11.2008 18:35:09.999992','washbasin flush upstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['22.11.2008 18:31:42.999998','washbasin flush upstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['22.11.2008 18:31:44.000003','washbasin flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['22.11.2008 18:29:22.999993','washbasin flush upstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['22.11.2008 18:29:23.999997','washbasin flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['22.11.2008 18:28:46.999997','washbasin flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['22.11.2008 18:28:47.999992','washbasin flush upstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['23.11.2008 09:39:13.999996','washbasin flush upstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['23.11.2008 09:39:15.000000','washbasin flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['24.11.2008 00:19:42.999991','washbasin flush upstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['24.11.2008 00:19:43.999995','washbasin flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['24.11.2008 00:23:20.000000','washbasin flush upstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['24.11.2008 00:23:21.000005','washbasin flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['26.11.2008 00:48:42.999997','washbasin flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['26.11.2008 00:48:43.999992','washbasin flush upstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['26.11.2008 08:14:58.999995','washbasin flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-26 00:51:46.610400'), DEVICE:'washbasin flush upstairs', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['27.11.2008 18:10:19.999997','washbasin flush upstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['27.11.2008 18:12:06.000005','washbasin flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2008 00:21:54.999999','washbasin flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2008 00:21:58.999997','washbasin flush upstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2008 00:22:00.000002','washbasin flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2008 00:22:02.999996','washbasin flush upstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2008 19:35:15.000001','washbasin flush upstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2008 19:35:15.999996','washbasin flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['29.11.2008 17:03:02.999996','washbasin flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['29.11.2008 17:03:13.999997','washbasin flush upstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['29.11.2008 18:57:21.000004','washbasin flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['29.11.2008 18:57:23.999998','washbasin flush upstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['29.11.2008 19:01:05.999991','washbasin flush upstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['29.11.2008 19:01:06.999995','washbasin flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['22.11.2008 09:46:32.999998','pressure mat bed left',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-22 09:46:23.002000'), DEVICE:'pressure mat bed left', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-24 07:43:42.985400'), DEVICE:'pressure mat bed left', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-24 00:26:38.997700'), DEVICE:'pressure mat bed left', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-26 08:24:16.918400'), DEVICE:'pressure mat bed left', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-25 07:07:37.980400'), DEVICE:'pressure mat bed left', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['29.11.2008 07:25:12.000003','pressure mat bed left',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['29.11.2008 07:25:31.999995','pressure mat bed left',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-30 13:49:47.022200'), DEVICE:'pressure mat bed left', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-30 08:57:05.975500'), DEVICE:'pressure mat bed left', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-02 07:10:49.995300'), DEVICE:'pressure mat bed left', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-01 08:27:29.006700'), DEVICE:'pressure mat bed left', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-02 07:11:02.996100'), DEVICE:'pressure mat bed left', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-02 07:10:53.975600'), DEVICE:'pressure mat bed left', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-05 07:17:32.978400'), DEVICE:'pressure mat bed left', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-04 07:07:00.007300'), DEVICE:'pressure mat bed left', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['06.12.2008 08:32:29.999998','pressure mat bed left',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['02.12.2008 01:00:30.000005','washbasin flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-01 18:38:32.541400'), DEVICE:'washbasin flush upstairs', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['05.12.2008 07:15:26.000003','washbasin flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-05 03:14:24.033800'), DEVICE:'washbasin flush upstairs', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['05.12.2008 07:17:04.999999','washbasin flush upstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['05.12.2008 17:48:15.000002','washbasin flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['21.11.2008 21:21:51.000000','front door reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['21.11.2008 21:32:17.000005','front door reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2008 18:54:44.999995','front door reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-28 07:31:27.613300'), DEVICE:'front door reed', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['28.11.2008 22:39:53.999997','front door reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-28 22:49:52.847000'), DEVICE:'front door reed', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['29.11.2008 11:46:24.000000','front door reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-29 11:23:22.444900'), DEVICE:'front door reed', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['01.12.2008 11:17:30.999996','front door reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-01 09:49:56.997000'), DEVICE:'front door reed', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['02.12.2008 23:03:24.999998','front door reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['02.12.2008 23:09:07.999998','front door reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['02.12.2008 23:09:09.000003','front door reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['02.12.2008 23:09:22.999997','front door reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['02.12.2008 23:09:24.000002','front door reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['02.12.2008 23:09:26.999996','front door reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['02.12.2008 23:09:29.000005','front door reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['02.12.2008 23:09:39.999996','front door reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['02.12.2008 23:09:46.999998','front door reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['02.12.2008 23:52:44.000001','front door reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['02.12.2008 23:52:46.999995','front door reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['02.12.2008 23:56:36.000000','front door reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['02.12.2008 23:56:37.000005','front door reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 06:05:58.999994','front door reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 06:05:59.999999','front door reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 06:06:01.999998','front door reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 06:06:03.999997','front door reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 06:09:31.000001','front door reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['04.12.2008 22:28:32.000005','front door reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['05.12.2008 07:20:42.999994','front door reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['05.12.2008 07:20:43.999998','front door reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['05.12.2008 07:21:31.000005','front door reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['25.11.2008 16:59:55.999995','front door reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-25 16:52:34.884000'), DEVICE:'front door reed', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 06:47:28.999996','front door reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-03 06:50:44.930200'), DEVICE:'front door reed', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:34:25.999996','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:34:27.000000','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:41:59.999997','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:42:01.000002','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:42:50.999992','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:42:51.999997','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:43:11.000004','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:43:21.999995','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:44:22.999996','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:44:24.000001','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:44:39.000000','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:44:40.000005','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:45:02.999990','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:45:05.000000','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:45:21.999998','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:45:23.000003','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:45:43.999999','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:45:45.000004','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:46:00.999998','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:46:06.999995','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:46:08.000000','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:46:09.000005','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:46:09.999999','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:46:11.000004','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:46:13.999998','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:46:15.999997','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:46:16.999991','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:46:17.999996','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:46:18.999991','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:46:19.999995','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:46:21.000000','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:46:22.000005','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:46:22.999999','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:46:24.000004','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:46:39.999997','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:46:43.999996','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:46:50.999998','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:46:52.000003','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:46:58.000000','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:47:00.000000','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:47:14.999999','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:47:16.000003','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:47:17.999992','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:47:18.999997','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:47:25.999999','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:47:27.000004','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:47:40.999998','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:47:42.000003','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:47:56.999992','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:47:57.999997','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:48:10.999997','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:48:12.000001','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:48:16.000000','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:48:17.000004','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:48:17.999999','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:48:19.000004','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:48:19.999998','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:48:21.000003','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:48:35.999992','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:48:36.999996','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:48:44.999993','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:48:45.999998','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:48:46.999993','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:48:47.999997','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:48:48.999992','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:48:49.999996','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:48:50.999991','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:48:51.999996','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:49:11.999998','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:49:13.000002','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:49:29.999991','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:49:30.999995','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:51:14.000000','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:51:15.000004','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:51:29.999994','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:51:30.999998','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:51:45.999997','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:51:47.000002','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:52:01.999991','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:52:02.999996','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:52:19.999994','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:52:20.999999','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:52:35.999998','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:52:37.000002','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:52:37.999997','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:52:39.000001','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:52:53.999991','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:52:54.999995','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:53:17.999991','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:53:18.999996','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:53:46.999995','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:53:47.999999','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:54:03.999993','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:54:04.999998','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:54:27.999994','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:54:28.999998','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:54:44.999992','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:54:45.999997','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:55:00.999996','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:55:02.000000','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:55:16.999999','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:55:18.000004','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:55:33.999998','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:55:35.000002','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:55:55.999999','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:55:57.000004','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:56:14.999997','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:56:16.000001','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:56:31.999995','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:56:33.000000','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:56:49.999998','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:56:51.000003','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:57:13.999998','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 16:57:15.000003','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:00:21.000000','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:00:22.000005','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:03:59.999999','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:04:01.000004','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:05:36.999996','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:05:38.000001','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:11:33.999991','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:11:34.999995','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:11:37.999999','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:11:39.000004','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:12:03.999999','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:12:05.000004','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:12:30.999993','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:12:31.999998','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:12:32.999993','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:12:33.999997','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:12:58.999992','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:12:59.999997','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:13:52.999991','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:13:53.999996','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:18:37.000000','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:18:38.000004','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:18:40.999998','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:18:42.000003','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:34:40.999991','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:34:41.999996','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:36:02.999999','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:36:04.000004','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:37:39.999996','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:37:41.000001','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:41:49.999998','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:41:51.000003','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:47:04.999999','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:47:06.000004','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:47:40.999995','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:47:42.000000','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:47:58.999998','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:48:00.000003','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:48:15.999996','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:48:17.000001','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:48:32.000000','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:48:33.000005','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:51:06.999994','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:51:07.999999','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:52:00.999993','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:52:01.999998','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:52:17.999991','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:52:18.999996','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:52:33.999995','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:52:35.000000','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:52:52.999993','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:52:53.999997','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:53:10.999996','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:53:12.000000','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:53:27.999994','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:53:28.999998','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:53:43.999998','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:53:45.000002','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:54:35.999997','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:54:37.000002','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:55:39.999992','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:55:40.999997','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:56:06.999996','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:56:08.000001','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:56:23.000000','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:56:24.000005','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:56:41.999998','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:56:43.000002','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:57:03.000005','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:57:18.000004','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:57:18.999998','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:57:20.000003','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:57:20.999997','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:57:22.000002','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:57:28.999994','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:57:29.999999','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:57:30.999993','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:57:31.999998','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:57:32.999993','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:57:33.999997','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:57:34.999992','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:57:35.999997','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:57:50.999996','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:57:52.000000','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:58:07.999994','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:58:08.999999','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:58:23.999998','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:58:25.000002','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:58:39.999991','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:58:40.999996','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:58:44.000000','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:58:45.000004','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:59:01.999993','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:59:02.999997','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:59:17.999996','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:59:19.000001','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:59:34.000000','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 17:59:35.000005','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:00:02.999994','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:00:03.999998','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:00:18.999998','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:00:20.000002','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:01:11.999992','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:01:12.999996','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:01:29.000000','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:01:30.000005','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:01:45.999998','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:01:47.000003','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:02:00.999997','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:02:02.000002','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:02:21.000000','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:02:22.000004','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:02:26.999997','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:02:35.000004','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:02:35.999999','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:02:37.000003','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:02:50.999998','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:02:52.000002','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:03:08.999991','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:03:09.999995','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:03:22.999995','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:03:27.000004','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:03:27.999998','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:03:29.000003','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:03:51.000004','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:03:57.999997','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:04:03.999994','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:04:04.999999','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:04:07.999993','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:04:08.999997','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:04:09.999992','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:04:10.999996','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:04:12.999996','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:04:15.000005','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:04:20.999992','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:04:21.999997','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:04:23.000002','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:04:25.999996','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:04:53.000000','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:04:54.000005','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:04:54.999999','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:04:56.000004','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:05:04.999995','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:05:07.000005','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:05:07.999999','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:05:09.000004','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:05:13.999997','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:05:15.000001','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:05:26.999997','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:05:28.000001','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:05:28.999996','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:05:30.000000','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:05:32.000000','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:05:33.999999','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:06:05.000002','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:06:07.999995','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:06:09.000000','keys',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['03.12.2008 18:06:10.000005','keys',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-20 17:27:38.036500'), DEVICE:'cabinet plates spices reed', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-20 13:22:43.973300'), DEVICE:'cabinet plates spices reed', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-22 10:21:28.992000'), DEVICE:'cabinet plates spices reed', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-21 13:44:04.966000'), DEVICE:'cabinet plates spices reed', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-26 19:31:16.965200'), DEVICE:'cabinet plates spices reed', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-25 23:21:37.975500'), DEVICE:'cabinet plates spices reed', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-28 20:06:07.941900'), DEVICE:'cabinet plates spices reed', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-27 17:28:54.975800'), DEVICE:'cabinet plates spices reed', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-01 08:42:26.015200'), DEVICE:'cabinet plates spices reed', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-30 14:27:30.020800'), DEVICE:'cabinet plates spices reed', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-01 17:49:22.007700'), DEVICE:'cabinet plates spices reed', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-01 08:42:35.969900'), DEVICE:'cabinet plates spices reed', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-02 07:17:25.994500'), DEVICE:'cabinet plates spices reed', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-02 00:37:01.005400'), DEVICE:'cabinet plates spices reed', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-04 17:27:33.948100'), DEVICE:'cabinet plates spices reed', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-03 18:21:05.974700'), DEVICE:'cabinet plates spices reed', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-05 16:54:42.986100'), DEVICE:'cabinet plates spices reed', VALUE:True})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)
new_row=pd.Series(
{TIME: pd.Timestamp('2008-12-05 15:53:41.003300'), DEVICE:'cabinet plates spices reed', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['05.12.2008 16:56:42.999996','cabinet plates spices reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['05.12.2008 16:56:44.000000','cabinet plates spices reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['05.12.2008 16:57:15.000003','cabinet plates spices reed',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['05.12.2008 19:44:33.000002','cabinet plates spices reed',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['04.12.2008 17:11:34.000001','refrigerator',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['04.12.2008 17:20:06.999994','refrigerator',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['27.11.2008 18:02:54.000002','refrigerator',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['27.11.2008 22:43:56.999996','refrigerator',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['22.11.2008 10:43:58.999996','refrigerator',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['22.11.2008 10:51:25.000001','refrigerator',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['21.11.2008 13:46:17.000001','refrigerator',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['21.11.2008 14:38:07.999995','refrigerator',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

#--------------------------------------------------
# Hit_nr.: 0
# Filtering refrigerator signal

idx_to_del = get_index_matching_rows(df_devs, 	[['21.11.2008 17:34:22.000004','refrigerator','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['21.11.2008 17:34:58.000000','refrigerator','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 0
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['21.11.2008 17:33:58.000003','refrigerator','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['21.11.2008 17:34:20.999999','refrigerator','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 0
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['21.11.2008 17:33:45.000003','refrigerator','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['21.11.2008 17:33:56.999999','refrigerator','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 0
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['21.11.2008 17:33:38.999996','refrigerator','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['21.11.2008 17:33:43.999999','refrigerator','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 0
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['21.11.2008 17:33:25.000001','refrigerator','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['21.11.2008 17:33:32.999998','refrigerator','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 0
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['21.11.2008 17:33:34.000003','refrigerator','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['21.11.2008 17:33:34.999997','refrigerator','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 0
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['21.11.2008 17:32:31.999997','refrigerator','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['21.11.2008 17:32:44.999997','refrigerator','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 0
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['21.11.2008 17:33:36.000002','refrigerator','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['21.11.2008 17:33:37.999991','refrigerator','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 1
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['24.11.2008 13:14:19.999999','refrigerator','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['24.11.2008 13:14:21.000004','refrigerator','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 2
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['01.12.2008 17:38:53.999997','refrigerator','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['01.12.2008 17:38:54.999992','refrigerator','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 2
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['01.12.2008 17:44:06.000005','refrigerator','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['01.12.2008 17:44:10.999998','refrigerator','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 3
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['04.12.2008 17:27:32.999999','refrigerator','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['04.12.2008 17:27:34.000004','refrigerator','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 4
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['05.12.2008 17:04:14.999998','refrigerator','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['05.12.2008 17:04:15.999993','refrigerator','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#--------------------------------------------------
# Hit_nr.: 3
# 

idx_to_del = get_index_matching_rows(df_devs, 	[['04.12.2008 22:44:24.000001','refrigerator','True']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, 	[['04.12.2008 22:44:24.999995','refrigerator','False']])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat couch')\
     & ('2008-11-30 14:47:49.970800' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-11-30 14:55:12.545200')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat couch') & ('2008-11-22 01:27:19.507100' < df_devs[TIME]) & (df_devs[TIME] < '2008-11-22 01:38:12.712500')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat couch') & ('2008-11-21 18:16:27.584300' < df_devs[TIME]) & (df_devs[TIME] < '2008-11-21 18:25:47.733000')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['04.12.2008 00:12:01.999997','washbasin flush upstairs',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['04.12.2008 00:15:16.000000','washbasin flush upstairs',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat couch') & ('2008-11-22 01:37:49.679500' < df_devs[TIME]) & (df_devs[TIME] < '2008-11-22 03:06:06.353700')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat couch') & ('2008-11-21 19:09:36.083100' < df_devs[TIME]) & (df_devs[TIME] < '2008-11-21 20:05:34.955400')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat couch') & ('2008-11-30 15:01:19.720500' < df_devs[TIME]) & (df_devs[TIME] < '2008-11-30 21:15:17.796600')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat couch') & ('2008-12-01 10:50:40.347500' < df_devs[TIME]) & (df_devs[TIME] < '2008-12-01 11:15:37.825600')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat couch') & ('2008-12-01 14:03:26.318100' < df_devs[TIME]) & (df_devs[TIME] < '2008-12-01 16:47:08.374900')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat couch') & ('2008-12-02 15:38:13.743000' < df_devs[TIME]) & (df_devs[TIME] < '2008-12-02 22:40:57.487700')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat couch') & ('2008-12-03 08:19:01.098900' < df_devs[TIME]) & (df_devs[TIME] < '2008-12-03 08:33:31.538500')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat couch') & ('2008-11-22 18:21:14.748100' < df_devs[TIME]) & (df_devs[TIME] < '2008-11-22 18:35:07.559600')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat couch') & ('2008-11-30 13:15:27.023300' < df_devs[TIME]) & (df_devs[TIME] < '2008-11-30 13:23:04.562300')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['24.11.2008 18:16:47.000000','door bedroom',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['24.11.2008 18:27:00.999999','door bedroom',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat bed right')\
     & ('2008-11-20 01:39:44.538900' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-11-20 08:48:01.553700')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat bed right')\
     & ('2008-11-20 08:48:02.688200' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-11-20 08:50:33.626500')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat bed right')\
     & ('2008-11-20 08:50:35.500500' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-11-20 09:31:47.396500')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat bed right')\
     & ('2008-11-21 00:52:28.485100' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-11-21 04:17:34.735200')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['21.11.2008 04:17:36.000002','pressure mat bed right',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-21 04:17:29.117100'), DEVICE:'pressure mat bed right', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['21.11.2008 04:17:29.117100','pressure mat bed right',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-21 06:00:08.636600'), DEVICE:'pressure mat bed right', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat bed right')\
     & ('2008-11-21 06:05:41.759800' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-11-21 07:30:48.401100')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['21.11.2008 07:30:48.999996','pressure mat bed right',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series(
{TIME: pd.Timestamp('2008-11-21 11:20:23.116000'), DEVICE:'pressure mat bed right', VALUE:False})
df_devs = pd.concat([df_devs, new_row.to_frame().T], axis=0).sort_values(by=TIME).reset_index(drop=True)

#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat bed right')\
     & ('2008-11-22 01:38:17.352700' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-11-22 09:36:39.398300')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat bed right')\
     & ('2008-11-23 02:15:20.737400' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-11-23 05:28:38.589800')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat bed right')\
     & ('2008-11-23 05:31:14.503000' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-11-23 06:18:21.457700')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat bed right')\
     & ('2008-11-23 06:19:15.038000' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-11-23 09:23:23.803300')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat bed right')\
     & ('2008-11-24 00:27:42.738600' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-11-24 07:28:01.442200')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat bed right')\
     & ('2008-11-25 00:51:45.637500' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-11-25 06:55:58.407900')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat bed right')\
     & ('2008-11-26 01:07:52.260500' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-11-26 08:12:25.517000')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat bed right')\
     & ('2008-11-27 00:22:43.499700' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-11-27 06:53:08.226200')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat bed right')\
     & ('2008-11-28 00:24:20.944400' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-11-28 07:11:51.339000')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat bed right')\
     & ('2008-11-28 23:49:33.265800' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-11-29 07:15:25.478800')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat bed right')\
     & ('2008-11-30 01:54:36.790800' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-11-30 08:53:33.943700')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat bed right')\
     & ('2008-12-01 00:23:26.643600' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-12-01 08:22:08.054700')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat bed right')\
     & ('2008-12-02 01:02:12.492400' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-12-02 07:01:53.738800')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat bed right')\
     & ('2008-12-03 00:39:01.670100' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-12-03 06:00:29.660300')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat bed right')\
     & ('2008-12-04 00:16:53.805500' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-12-04 06:56:07.789400')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat bed right')\
     & ('2008-12-05 03:17:33.618500' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-12-05 06:44:55.525300')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat bed right')\
     & ('2008-12-06 04:51:46.828300' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-12-06 08:29:06.479100')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)


#----------------------------------------
# 

mask = (df_devs[DEVICE] == 'pressure mat bed right')\
     & ('2008-12-07 02:07:49.456100' < df_devs[TIME])\
     & (df_devs[TIME] < '2008-12-07 08:07:47.349100')
df_devs = df_devs[~mask].sort_values(by=TIME).reset_index(drop=True)

