# Awesome script 
import pandas as pd
from pyadlml.dataset._core.activities import get_index_matching_rows

# TODO add original activity dataframe in line below
idxs_to_delete = []

# Content
#----------------------------------------------------------------------------------------------------

# id=0
# enveloping activity is split by inserting intermediate activities

lst_to_add_0 = [
	['19.11.2008 23:00:50.000004','20.11.2008 00:14:50.000001','Relax'],
	['20.11.2008 00:14:50.001001','20.11.2008 00:15:19.999999','Get drink'],
	['20.11.2008 00:15:20.000999','20.11.2008 00:39:00.000001','Relax'],
	['20.11.2008 00:39:00.001001','20.11.2008 00:39:59.999998','Get snack'],
	['20.11.2008 00:40:00.000998','20.11.2008 01:24:59.999998','Relax'],
]

lst_to_del = [
	['19.11.2008 23:00:50.000004','20.11.2008 01:24:59.999998','Relax'],
	['20.11.2008 00:14:50.000001','20.11.2008 00:15:19.999999','Get drink'],
	['20.11.2008 00:39:00.000001','20.11.2008 00:39:59.999998','Get snack'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=1
# enveloping activity is split by inserting intermediate activity

lst_to_add_1 = [
	['20.11.2008 01:36:59.999995','20.11.2008 08:48:39.999997','Go to bed'],
	['20.11.2008 08:48:40.000997','20.11.2008 08:49:40.000003','Use toilet upstairs'],
	['20.11.2008 08:49:40.001003','20.11.2008 09:31:40.000004','Go to bed'],
]

lst_to_del = [
	['20.11.2008 01:36:59.999995','20.11.2008 09:31:40.000004','Go to bed'],
	['20.11.2008 08:48:39.999997','20.11.2008 08:49:40.000003','Use toilet upstairs'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=2
# enveloping activity is split by inserting intermediate activities. 
# It is reasonable to resume prepare lunch for 2 minutes after visiting toilet

lst_to_add_2 = [
	['20.11.2008 13:11:10.000004','20.11.2008 13:18:50.000003','Prepare Lunch'],
	['20.11.2008 13:18:50.001003','20.11.2008 13:20:40.000000','Use toilet downstairs'],
	['20.11.2008 13:20:40.001000','20.11.2008 13:22:29.999997','Prepare Lunch'],
	['20.11.2008 13:22:30.000997','20.11.2008 13:33:59.999996','Eating'],
]

lst_to_del = [
	['20.11.2008 13:11:10.000004','20.11.2008 13:22:29.999997','Prepare Lunch'],
	['20.11.2008 13:18:50.000003','20.11.2008 13:20:40.000000','Use toilet downstairs'],
	['20.11.2008 13:22:29.999997','20.11.2008 13:33:59.999996','Eating'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=3
# enlarged eating <- smaller activity than relax.
# Resuming relax for 2 minutes after toilet is reasonable

lst_to_add_3 = [
	['20.11.2008 18:30:30.998000','20.11.2008 19:50:10.000005','Relax'],
	['20.11.2008 19:50:10.001005','20.11.2008 19:51:15.000004','Use toilet downstairs'],
	['20.11.2008 19:51:15.001004','20.11.2008 19:52:59.999998','Relax'],
]

lst_to_del = [
	['20.11.2008 18:14:29.999998','20.11.2008 19:52:59.999998','Relax'],
	['20.11.2008 19:50:10.000005','20.11.2008 19:51:15.000004','Use toilet downstairs'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=4
# enveloping activity is split by inserting intermediate activities

lst_to_add_4 = [
	['21.11.2008 00:50:59.999999','21.11.2008 06:00:59.999996','Go to bed'],
	['21.11.2008 06:01:00.000996','21.11.2008 06:04:40.000000','Use toilet upstairs'],
	['21.11.2008 06:04:40.001000','21.11.2008 11:20:00.000002','Go to bed'],
]

lst_to_del = [
	['21.11.2008 00:50:59.999999','21.11.2008 11:20:00.000002','Go to bed'],
	['21.11.2008 06:00:59.999996','21.11.2008 06:04:40.000000','Use toilet upstairs'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=5
# enveloping activity is split by inserting intermediate activities

lst_to_add_5 = [
	['23.11.2008 02:11:19.999996','23.11.2008 05:28:50.000004','Go to bed'],
	['23.11.2008 05:28:50.001004','23.11.2008 05:29:50.000001','Use toilet upstairs'],
	['23.11.2008 05:29:50.001001','23.11.2008 09:23:29.999996','Go to bed'],
]

lst_to_del = [
	['23.11.2008 02:11:19.999996','23.11.2008 09:23:29.999996','Go to bed'],
	['23.11.2008 05:28:50.000004','23.11.2008 05:29:50.000001','Use toilet upstairs'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=6
# Find middle ground between two activities 

lst_to_add_6 = [
	['23.11.2008 09:24:00.000005','23.11.2008 09:26:20.000002','Use toilet upstairs'],
	['23.11.2008 09:26:20.999998','23.11.2008 09:30:00.000003','Take shower'],
]

lst_to_del = [
	['23.11.2008 09:24:00.000005','23.11.2008 09:26:40.000002','Use toilet upstairs'],
	['23.11.2008 09:25:59.999998','23.11.2008 09:30:00.000003','Take shower'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=7
# enveloping activity is split by inserting intermediate activities

lst_to_add_7 = [
	['23.11.2008 22:00:30.000005','23.11.2008 23:14:00.000004','Relax'],
	['23.11.2008 23:14:00.001004','23.11.2008 23:15:09.999996','Get drink'],
	['23.11.2008 23:15:10.000996','24.11.2008 00:01:19.999999','Relax'],
]

lst_to_del = [
	['23.11.2008 22:00:30.000005','24.11.2008 00:01:19.999999','Relax'],
	['23.11.2008 23:14:00.000004','23.11.2008 23:15:09.999996','Get drink'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=8
# Enveloping activity is split by inserting intermediate activity. 
# Second activity begins 1ms later

lst_to_add_8 = [
	['24.11.2008 07:34:10.000002','24.11.2008 07:43:10.000000','Prepare Breakfast'],
	['24.11.2008 07:43:10.001000','24.11.2008 07:45:30.000005','Get dressed'],
	['24.11.2008 07:45:30.001005','24.11.2008 07:57:10.000000','Prepare Breakfast'],
	['24.11.2008 07:57:10.001000','24.11.2008 08:08:10.000001','Eating'],
]

lst_to_del = [
	['24.11.2008 07:34:10.000002','24.11.2008 07:57:10.000000','Prepare Breakfast'],
	['24.11.2008 07:43:10.000000','24.11.2008 07:45:30.000005','Get dressed'],
	['24.11.2008 07:57:10.000000','24.11.2008 08:08:10.000001','Eating'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=9
# enveloping activity is split by inserting intermediate activities

lst_to_add_9 = [
	['24.11.2008 18:33:39.999997','24.11.2008 18:38:19.999997','Relax'],
	['24.11.2008 18:38:20.000997','24.11.2008 18:40:20.000000','Use toilet downstairs'],
	['24.11.2008 18:40:20.001000','24.11.2008 18:42:00.000001','Relax'],
]

lst_to_del = [
	['24.11.2008 18:33:39.999997','24.11.2008 18:42:00.000001','Relax'],
	['24.11.2008 18:38:19.999997','24.11.2008 18:40:20.000000','Use toilet downstairs'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=10
# find middleground between activities

lst_to_add_10 = [
	['25.11.2008 00:44:30.000002','25.11.2008 00:48:15.999995','Brush teeth'],
	['25.11.2008 00:48:16.000000','25.11.2008 00:48:50.000000','Use toilet upstairs'],
]

lst_to_del = [
	['25.11.2008 00:44:30.000002','25.11.2008 00:48:20.000001','Brush teeth'],
	['25.11.2008 00:48:09.999995','25.11.2008 00:48:50.000000','Use toilet upstairs'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=11
# enveloping activity is split by inserting intermediate activities

lst_to_add_11 = [
	['25.11.2008 17:01:00.000000','25.11.2008 17:05:30.000004','Prepare Dinner'],
	['25.11.2008 17:05:30.001004','25.11.2008 17:06:30.000000','Use toilet downstairs'],
	['25.11.2008 17:06:30.001000','25.11.2008 17:35:59.999995','Prepare Dinner'],
]

lst_to_del = [
	['25.11.2008 17:01:00.000000','25.11.2008 17:35:59.999995','Prepare Dinner'],
	['25.11.2008 17:05:30.000004','25.11.2008 17:06:30.000000','Use toilet downstairs'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=12
# enveloping activity is split by inserting intermediate activities

lst_to_add_12 = [
	['25.11.2008 23:23:22.999998','26.11.2008 00:13:19.999996','Relax'],
	['26.11.2008 00:13:20.000996','26.11.2008 00:14:35.000001','Use toilet upstairs'],
	['26.11.2008 00:14:35.001001','26.11.2008 00:19:30.000001','Relax'],
	['26.11.2008 00:19:30.001001','26.11.2008 00:19:39.999997','Get snack'],
	['26.11.2008 00:19:40.000997','26.11.2008 00:45:00.000000','Relax'],
]

lst_to_del = [
	['25.11.2008 23:23:22.999998','26.11.2008 00:45:00.000000','Relax'],
	['26.11.2008 00:13:19.999996','26.11.2008 00:14:35.000001','Use toilet upstairs'],
	['26.11.2008 00:19:30.000001','26.11.2008 00:19:39.999997','Get snack'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=13
# enveloping activity is split by inserting intermediate activities

lst_to_add_13 = [
	['26.11.2008 00:49:10.000002','26.11.2008 00:51:10.000005','Go to bed'],
	['26.11.2008 00:51:10.001005','26.11.2008 00:51:29.999997','Take medication'],
	['26.11.2008 00:51:30.000997','26.11.2008 08:12:59.999997','Go to bed'],
]

lst_to_del = [
	['26.11.2008 00:49:10.000002','26.11.2008 08:12:59.999997','Go to bed'],
	['26.11.2008 00:51:10.000005','26.11.2008 00:51:29.999997','Take medication'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=14
# enveloping activity is split by inserting intermediate activities

lst_to_add_14 = [
	['26.11.2008 21:49:30.000004','26.11.2008 21:51:10.000005','Relax'],
	['26.11.2008 21:51:10.001005','26.11.2008 21:51:59.999995','Get snack'],
	['26.11.2008 21:52:00.000995','26.11.2008 22:08:49.999999','Relax'],
]

lst_to_del = [
	['26.11.2008 21:49:30.000004','26.11.2008 22:08:49.999999','Relax'],
	['26.11.2008 21:51:10.000005','26.11.2008 21:51:59.999995','Get snack'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=15
# enveloping activity is split by inserting intermediate activities

lst_to_add_15 = [
	['26.11.2008 23:04:29.999997','26.11.2008 23:38:49.999999','Relax'],
	['26.11.2008 23:38:50.000999','26.11.2008 23:39:30.000003','Get snack'],
	['26.11.2008 23:39:30.001003','26.11.2008 23:48:20.000005','Relax'],
	['26.11.2008 23:48:20.001005','26.11.2008 23:49:09.999995','Use toilet downstairs'],
	['26.11.2008 23:49:10.000995','27.11.2008 00:03:29.999998','Relax'],
]

lst_to_del = [
	['26.11.2008 23:04:29.999997','27.11.2008 00:03:29.999998','Relax'],
	['26.11.2008 23:38:49.999999','26.11.2008 23:39:30.000003','Get snack'],
	['26.11.2008 23:48:20.000005','26.11.2008 23:49:09.999995','Use toilet downstairs'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=16
# enveloping activity is split by inserting intermediate activities
# Between the two find middle ground

lst_to_add_16 = [
	['27.11.2008 00:07:49.999995','27.11.2008 00:11:20.000003','Go to bed'],
	['27.11.2008 00:11:20.001003','27.11.2008 00:12:10.000000','Use toilet downstairs'],
	['27.11.2008 00:12:10.001000','27.11.2008 00:12:20.000999','Take medication'],
	['27.11.2008 00:12:20.001999','27.11.2008 06:51:49.999999','Go to bed'],
]

lst_to_del = [
	['27.11.2008 00:07:49.999995','27.11.2008 06:51:49.999999','Go to bed'],
	['27.11.2008 00:11:20.000003','27.11.2008 00:12:19.999999','Use toilet downstairs'],
	['27.11.2008 00:11:59.999997','27.11.2008 00:13:00.000004','Take medication'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=17
# enveloping activity is split by inserting intermediate activities

lst_to_add_17 = [
	['27.11.2008 22:38:59.999998','27.11.2008 22:41:55.000004','Relax'],
	['27.11.2008 22:41:55.001004','27.11.2008 22:42:25.000003','Get snack'],
	['27.11.2008 22:42:25.001003','28.11.2008 00:11:09.999997','Relax'],
]

lst_to_del = [
	['27.11.2008 22:38:59.999998','28.11.2008 00:11:09.999997','Relax'],
	['27.11.2008 22:41:55.000004','27.11.2008 22:42:25.000003','Get snack'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=18
# 

lst_to_add_18 = [
	['29.11.2008 18:50:00.000002','29.11.2008 18:54:29.999996','Take shower'],
	['29.11.2008 18:54:30.000996','29.11.2008 18:59:20.000002','Brush teeth'],
	['29.11.2008 18:59:30.000998','29.11.2008 18:59:59.999997','Use toilet upstairs'],
	['29.11.2008 19:00:00.000997','29.11.2008 19:09:49.999995','Take shower'],
]

lst_to_del = [
	['29.11.2008 18:50:00.000002','29.11.2008 19:09:49.999995','Take shower'],
	['29.11.2008 18:54:29.999996','29.11.2008 18:59:20.000002','Brush teeth'],
	['29.11.2008 18:59:29.999998','29.11.2008 18:59:59.999997','Use toilet upstairs'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=19
# smaller activity brush teeth wins

lst_to_add_19 = [
	['01.12.2008 00:19:20.000100','01.12.2008 08:20:00.000002','Go to bed'],
]

lst_to_del = [
	['01.12.2008 00:19:09.999999','01.12.2008 08:20:00.000002','Go to bed'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=20
# 

lst_to_add_20 = [
	['01.12.2008 17:32:29.999998','01.12.2008 18:04:10.000002','Prepare Dinner'],
	['01.12.2008 18:04:10.001002','01.12.2008 18:05:09.999998','Use toilet downstairs'],
	['01.12.2008 18:05:10.000998','01.12.2008 18:11:00.000001','Prepare Dinner'],
	['01.12.2008 18:11:00.001001','01.12.2008 18:23:49.999999','Eating'],
]

lst_to_del = [
	['01.12.2008 17:32:29.999998','01.12.2008 18:11:39.999995','Prepare Dinner'],
	['01.12.2008 18:04:10.000002','01.12.2008 18:05:09.999998','Use toilet downstairs'],
	['01.12.2008 18:11:00.000001','01.12.2008 18:23:49.999999','Eating'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=21
# 

lst_to_add_21 = [
	['01.12.2008 23:51:20.000004','02.12.2008 00:33:30.000001','Relax'],
	['02.12.2008 00:33:30.001001','02.12.2008 00:39:09.999997','Eating'],
	['02.12.2008 00:40:52.000000','02.12.2008 00:53:00.000002','Relax'],
]

lst_to_del = [
	['01.12.2008 23:51:20.000004','02.12.2008 00:53:00.000002','Relax'],
	['02.12.2008 00:33:30.000001','02.12.2008 00:39:09.999997','Eating'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=22
# enveloping activity is split by inserting intermediate activities

lst_to_add_22 = [
	['02.12.2008 00:59:00.000000','02.12.2008 06:09:10.000004','Go to bed'],
	['02.12.2008 06:09:10.001004','02.12.2008 06:14:00.000000','Take shower'],
	['02.12.2008 06:14:00.001000','02.12.2008 06:59:44.999998','Go to bed'],
]

lst_to_del = [
	['02.12.2008 00:59:00.000000','02.12.2008 06:59:44.999998','Go to bed'],
	['02.12.2008 06:09:10.000004','02.12.2008 06:14:00.000000','Take shower'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=23
# 

lst_to_add_23 = [
	['03.12.2008 00:01:49.999997','03.12.2008 00:09:10.000004','Shave'],
	['03.12.2008 00:09:10.001004','03.12.2008 00:13:45.000001','Brush teeth'],
	['03.12.2008 00:13:45.001001','03.12.2008 00:34:09.999995','Relax'],
]

lst_to_del = [
	['03.12.2008 00:01:49.999997','03.12.2008 00:09:14.999997','Shave'],
	['03.12.2008 00:09:10.000004','03.12.2008 00:13:45.000001','Brush teeth'],
	['03.12.2008 00:09:55.000001','03.12.2008 00:34:09.999995','Relax'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=24
# enveloping activity is split by inserting intermediate activities

lst_to_add_24 = [
	['03.12.2008 18:50:30.000000','03.12.2008 19:58:15.000000','Relax'],
	['03.12.2008 19:58:15.001000','03.12.2008 19:58:55.000004','Use toilet downstairs'],
	['03.12.2008 19:58:55.001004','03.12.2008 21:49:30.000004','Relax'],
	['03.12.2008 21:49:30.001004','03.12.2008 21:51:49.999999','Use toilet downstairs'],
	['03.12.2008 21:51:50.000999','03.12.2008 22:04:35.000004','Relax'],
	['03.12.2008 22:04:35.001004','03.12.2008 22:06:30.000004','Use toilet downstairs'],
	['03.12.2008 22:06:30.001004','04.12.2008 00:05:39.999996','Relax'],
]

lst_to_del = [
	['03.12.2008 18:50:30.000000','04.12.2008 00:05:39.999996','Relax'],
	['03.12.2008 19:58:15.000000','03.12.2008 19:58:55.000004','Use toilet downstairs'],
	['03.12.2008 21:49:30.000004','03.12.2008 21:51:49.999999','Use toilet downstairs'],
	['03.12.2008 22:04:35.000004','03.12.2008 22:06:30.000004','Use toilet downstairs'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=25
# middelground between activities

lst_to_add_25 = [
	['05.12.2008 16:52:54.999995','05.12.2008 17:11:15.000000','Prepare Dinner'],
	['05.12.2008 17:11:15.001000','05.12.2008 17:25:45.000002','Eating'],
]

lst_to_del = [
	['05.12.2008 16:52:54.999995','05.12.2008 17:13:49.999997','Prepare Dinner'],
	['05.12.2008 17:09:15.000000','05.12.2008 17:25:45.000002','Eating'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=26
# middelground between activities

lst_to_add_26 = [
	['06.12.2008 08:43:59.999997','06.12.2008 08:51:30.000000','Prepare Breakfast'],
	['06.12.2008 08:51:30.001000','06.12.2008 09:02:39.999997','Eating'],
]

lst_to_del = [
	['06.12.2008 08:43:59.999997','06.12.2008 08:54:59.999998','Prepare Breakfast'],
	['06.12.2008 08:51:30.000000','06.12.2008 09:02:39.999997','Eating'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=27
# smaller activity wins

lst_to_add_27 = [
	['06.12.2008 11:03:00.999998','06.12.2008 11:42:00.000004','Put clothes in washingmachine'],
]

lst_to_del = [
	['06.12.2008 11:00:39.999998','06.12.2008 11:42:00.000004','Put clothes in washingmachine'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=28
# enveloping activity is split by inserting intermediate activities

lst_to_add_28 = [
	['07.12.2008 02:08:39.999999','07.12.2008 08:10:05.000001','Go to bed'],
	['07.12.2008 08:10:05.001001','07.12.2008 08:11:10.000000','Use toilet upstairs'],
	['07.12.2008 08:11:10.001000','07.12.2008 09:37:05.000001','Go to bed'],
]

lst_to_del = [
	['07.12.2008 02:08:39.999999','07.12.2008 09:37:05.000001','Go to bed'],
	['07.12.2008 08:10:05.000001','07.12.2008 08:11:10.000000','Use toilet upstairs'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=29
# enveloping activity is split by inserting intermediate activities

lst_to_add_29 = [
	['07.12.2008 12:37:29.999997','07.12.2008 12:43:25.000002','Take shower'],
	['07.12.2008 12:43:25.001002','07.12.2008 12:44:00.000004','Use toilet upstairs'],
	['07.12.2008 12:44:00.001004','07.12.2008 13:10:50.000001','Take shower'],
]

lst_to_del = [
	['07.12.2008 12:37:29.999997','07.12.2008 13:10:50.000001','Take shower'],
	['07.12.2008 12:43:25.000002','07.12.2008 12:44:00.000004','Use toilet upstairs'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=30
# enveloping activity is split by inserting intermediate activities

lst_to_add_30 = [
	['07.12.2008 21:27:00.000004','07.12.2008 21:29:19.999999','Relax'],
	['07.12.2008 21:29:20.000999','07.12.2008 21:30:00.000003','Get drink'],
	['07.12.2008 21:30:00.001003','08.12.2008 02:17:30.000001','Relax'],
]

lst_to_del = [
	['07.12.2008 21:27:00.000004','08.12.2008 02:17:30.000001','Relax'],
	['07.12.2008 21:29:19.999999','07.12.2008 21:30:00.000003','Get drink'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)

#----------------------------------------------------------------------------------------------------
new_rows_to_add = [*lst_to_add_0, *lst_to_add_1, *lst_to_add_2, *lst_to_add_3, *lst_to_add_4, *lst_to_add_5, *lst_to_add_6, *lst_to_add_7, *lst_to_add_8, *lst_to_add_9, *lst_to_add_10, *lst_to_add_11, *lst_to_add_12, *lst_to_add_13, *lst_to_add_14, *lst_to_add_15, *lst_to_add_16, *lst_to_add_17, *lst_to_add_18, *lst_to_add_19, *lst_to_add_20, *lst_to_add_21, *lst_to_add_22, *lst_to_add_23, *lst_to_add_24, *lst_to_add_25, *lst_to_add_26, *lst_to_add_27, *lst_to_add_28, *lst_to_add_29, *lst_to_add_30, ]
df_acts = df_acts.drop(idxs_to_delete)
df_news = pd.DataFrame(data=new_rows_to_add, columns=['start_time', 'end_time', 'activity'])
df_news['start_time'] = pd.to_datetime(df_news['start_time'], dayfirst=True)
df_news['end_time'] = pd.to_datetime(df_news['end_time'], dayfirst=True)
df_acts = pd.concat([df_acts, df_news], axis=0)
