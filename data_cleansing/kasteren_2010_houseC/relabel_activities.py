# Awesome script
import pandas as pd
from pyadlml.dataset._core.activities import get_index_matching_rows
from pyadlml.constants import START_TIME, END_TIME, ACTIVITY

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2008-11-19 22:59:25.000002','2008-11-19 23:00:00.000003','Get drink']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2008-11-19 22:59:12.595900'), 'end_time': pd.Timestamp('2008-11-19 23:00:30.014900'), 'activity': 'Get drink'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2008-11-20 01:31:30.000005','2008-11-20 01:36:20.000001','Brush teeth']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2008-11-20 01:32:12.588198800'), 'end_time': pd.Timestamp('2008-11-20 01:36:20.000001'), 'activity': 'Brush teeth'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2008-11-20 01:30:29.999998','2008-11-20 01:31:10.000003','Use toilet upstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2008-11-20 01:30:29.999998'), 'end_time': pd.Timestamp('2008-11-20 01:31:52.940412400'), 'activity': 'Use toilet upstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2008-11-20 01:30:29.999998','2008-11-20 01:31:52.940412400','Use toilet upstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2008-11-20 01:30:24.545080500'), 'end_time': pd.Timestamp('2008-11-20 01:32:09.188393200'), 'activity': 'Use toilet upstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2008-11-20 11:03:00.000003','2008-11-20 11:03:59.999999','Use toilet downstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2008-11-20 11:03:29.293200'), 'end_time': pd.Timestamp('2008-11-20 11:04:04.004000'), 'activity': 'Use toilet downstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2008-11-20 17:21:59.999995','2008-11-20 17:23:20.000004','Use toilet downstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2008-11-20 17:22:16.858200'), 'end_time': pd.Timestamp('2008-11-20 17:22:56.980900'), 'activity': 'Use toilet downstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2008-11-21 00:44:30.000002','2008-11-21 00:45:00','Use toilet upstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2008-11-21 00:44:22.601100'), 'end_time': pd.Timestamp('2008-11-21 00:45:29.313200'), 'activity': 'Use toilet upstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2008-11-21 11:21:29.999997','2008-11-21 11:21:59.999995','Use toilet upstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2008-11-21 11:21:35.403400'), 'end_time': pd.Timestamp('2008-11-21 11:22:09.590300'), 'activity': 'Use toilet upstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2008-11-21 16:43:50.000004','2008-11-21 17:29:29.999998','Leave house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2008-11-21 16:44:11.777659'), 'end_time': pd.Timestamp('2008-11-21 17:29:25.990621600'), 'activity': 'Leave house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['20.11.2008 00:15:20.000999','20.11.2008 00:39:00.000001','Relax']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-20 00:16:11.228888'), END_TIME: pd.Timestamp('2008-11-20 00:39:00.000001'), ACTIVITY: 'Relax'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['20.11.2008 00:14:50.001001','20.11.2008 00:15:19.999999','Get drink']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-20 00:14:50.001001'), END_TIME: pd.Timestamp('2008-11-20 00:16:09.340918400'), ACTIVITY: 'Get drink'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['20.11.2008 01:36:59.999995','20.11.2008 08:48:39.999997','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-20 01:36:59.999995'), END_TIME: pd.Timestamp('2008-11-20 08:48:28.000697600'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['20.11.2008 08:49:40.001003','20.11.2008 09:31:40.000004','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-20 08:50:28.185979200'), END_TIME: pd.Timestamp('2008-11-20 09:31:40.000004'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['20.11.2008 13:20:40.001000','20.11.2008 13:22:29.999997','Prepare Lunch']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-20 13:21:08.657451400'), END_TIME: pd.Timestamp('2008-11-20 13:22:29.999997'), ACTIVITY: 'Prepare Lunch'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['20.11.2008 13:18:50.001003','20.11.2008 13:20:40.000000','Use toilet downstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-20 13:18:50.001003'), END_TIME: pd.Timestamp('2008-11-20 13:20:47.827295'), ACTIVITY: 'Use toilet downstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['20.11.2008 15:28:00.000000','20.11.2008 17:20:00.000002','Leave house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-20 15:28:00'), END_TIME: pd.Timestamp('2008-11-20 17:20:17.494364'), ACTIVITY: 'Leave house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['20.11.2008 17:22:16.858200','20.11.2008 17:22:56.980900','Use toilet downstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-20 17:22:16.858200'), END_TIME: pd.Timestamp('2008-11-20 17:23:00.868852800'), ACTIVITY: 'Use toilet downstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['20.11.2008 19:51:15.001004','20.11.2008 19:52:59.999998','Relax']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-20 19:52:00.069516900'), END_TIME: pd.Timestamp('2008-11-20 19:52:59.999998'), ACTIVITY: 'Relax'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['20.11.2008 19:50:10.001005','20.11.2008 19:51:15.000004','Use toilet downstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-20 19:50:34.674448200'), END_TIME: pd.Timestamp('2008-11-20 19:51:15.000004'), ACTIVITY: 'Use toilet downstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['20.11.2008 19:57:10.000000','20.11.2008 22:38:00.000002','Leave house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-20 19:57:10'), END_TIME: pd.Timestamp('2008-11-20 22:38:16.072719'), ACTIVITY: 'Leave house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['21.11.2008 06:04:40.001000','21.11.2008 11:20:00.000002','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-21 06:05:21.606872500'), END_TIME: pd.Timestamp('2008-11-21 11:20:00.000002'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['21.11.2008 06:05:21.606872','21.11.2008 11:20:00.000002','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-21 06:05:21.606872500'), END_TIME: pd.Timestamp('2008-11-21 11:20:21.592262700'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['21.11.2008 14:52:20.000001','21.11.2008 14:54:59.999998','Use toilet downstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-21 14:52:20.000001'), END_TIME: pd.Timestamp('2008-11-21 14:55:16.283738400'), ACTIVITY: 'Use toilet downstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['21.11.2008 18:22:10.000998','21.11.2008 18:23:10.000004','Use toilet upstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-21 18:22:10.000998'), END_TIME: pd.Timestamp('2008-11-21 18:23:18.497100200'), ACTIVITY: 'Use toilet upstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['21.11.2008 18:30:30.000002','21.11.2008 21:21:30.000004','Leave house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-21 18:30:30.000002'), END_TIME: pd.Timestamp('2008-11-21 21:21:49.108980400'), ACTIVITY: 'Leave house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['21.11.2008 21:22:30.000000','21.11.2008 21:23:20.000000','Use toilet downstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-21 21:22:43.778290600'), END_TIME: pd.Timestamp('2008-11-21 21:23:33.070291600'), ACTIVITY: 'Use toilet downstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['21.11.2008 21:30:30.000002','21.11.2008 21:31:59.999996','Leave house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-21 21:30:55.659900'), END_TIME: pd.Timestamp('2008-11-21 21:32:12.639200'), ACTIVITY: 'Leave house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['22.11.2008 01:28:19.999999','22.11.2008 01:29:00.000004','Use toilet upstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-22 01:28:19.999999'), END_TIME: pd.Timestamp('2008-11-22 01:29:10.985134500'), ACTIVITY: 'Use toilet upstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['22.11.2008 09:38:49.999995','22.11.2008 09:45:09.999996','Take shower']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-22 09:39:37.853128900'), END_TIME: pd.Timestamp('2008-11-22 09:45:09.999996'), ACTIVITY: 'Take shower'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['22.11.2008 10:44:29.999998','22.11.2008 10:46:30.000001','Use toilet downstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-22 10:44:45.901153600'), END_TIME: pd.Timestamp('2008-11-22 10:46:30.000001'), ACTIVITY: 'Use toilet downstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['22.11.2008 10:56:40.000002','22.11.2008 12:29:29.999995','Leave house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-22 10:56:40.000002'), END_TIME: pd.Timestamp('2008-11-22 12:29:41.240037800'), ACTIVITY: 'Leave house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['22.11.2008 13:04:00.000002','22.11.2008 13:09:00.000005','Leave house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-22 13:04:00.000002'), END_TIME: pd.Timestamp('2008-11-22 13:09:15.907232800'), ACTIVITY: 'Leave house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['22.11.2008 20:13:29.999995','23.11.2008 02:02:30.000004','Leave house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-22 20:13:29.999995'), END_TIME: pd.Timestamp('2008-11-23 02:02:47.077564'), ACTIVITY: 'Leave house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['23.11.2008 05:29:50.001001','23.11.2008 09:23:29.999996','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-23 05:31:02.747668'), END_TIME: pd.Timestamp('2008-11-23 06:18:35.250434400'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Create operation:

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-23 06:19:11.934392800'), END_TIME: pd.Timestamp('2008-11-23 09:23:30.693300'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['23.11.2008 09:26:20.999998','23.11.2008 09:30:00.000003','Take shower']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-23 09:26:59.818232400'), END_TIME: pd.Timestamp('2008-11-23 09:30:00.000003'), ACTIVITY: 'Take shower'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['23.11.2008 09:37:40.000003','23.11.2008 09:40:20.000000','Brush teeth']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-23 09:38:36.887460'), END_TIME: pd.Timestamp('2008-11-23 09:40:20'), ACTIVITY: 'Brush teeth'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['23.11.2008 21:56:44.999998','23.11.2008 21:58:19.999996','Use toilet downstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-23 21:56:52.681457599'), END_TIME: pd.Timestamp('2008-11-23 21:58:32.339224'), ACTIVITY: 'Use toilet downstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['23.11.2008 23:15:10.000996','24.11.2008 00:01:19.999999','Relax']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-23 23:15:19.396149'), END_TIME: pd.Timestamp('2008-11-24 00:01:19.999999'), ACTIVITY: 'Relax'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['23.11.2008 23:14:00.001004','23.11.2008 23:15:09.999996','Get drink']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-23 23:14:00.001004'), END_TIME: pd.Timestamp('2008-11-23 23:15:17.925519200'), ACTIVITY: 'Get drink'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['24.11.2008 09:39:20.000004','24.11.2008 09:39:50.000002','Use toilet upstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-24 09:38:50.856392'), END_TIME: pd.Timestamp('2008-11-24 09:39:50.000002'), ACTIVITY: 'Use toilet upstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['24.11.2008 18:43:19.999999','24.11.2008 22:22:20.000001','Leave house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-24 18:43:19.999999'), END_TIME: pd.Timestamp('2008-11-24 22:22:28.059861'), ACTIVITY: 'Leave house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['25.11.2008 00:48:16.000000','25.11.2008 00:48:50.000000','Use toilet upstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-25 00:48:16'), END_TIME: pd.Timestamp('2008-11-25 00:48:51.862618400'), ACTIVITY: 'Use toilet upstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)


#----------------------------------------
# 

# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['26.11.2008 00:13:20.000996','26.11.2008 00:14:35.000001','Use toilet upstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

#----------------------------------------
# 


# Join operation: 
idx_to_del = get_index_matching_rows(df_acts, 	[['25.11.2008 23:23:22.999998','26.11.2008 00:13:19.999996','Relax']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)
idx_to_del = get_index_matching_rows(df_acts, 	[['26.11.2008 00:14:35.001001','26.11.2008 00:19:30.000001','Relax']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)
new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-25 23:23:22.999998'), END_TIME: pd.Timestamp('2008-11-26 00:19:30.000001'), ACTIVITY: 'Relax'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['25.11.2008 23:23:22.999998','26.11.2008 00:19:30.000001','Relax']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-25 23:23:22.999998'), END_TIME: pd.Timestamp('2008-11-26 00:19:13.877721500'), ACTIVITY: 'Relax'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['26.11.2008 00:19:30.001001','26.11.2008 00:19:39.999997','Get snack']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-26 00:19:14.665467500'), END_TIME: pd.Timestamp('2008-11-26 00:19:31.470715500'), ACTIVITY: 'Get snack'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['26.11.2008 00:19:40.000997','26.11.2008 00:45:00.000000','Relax']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-26 00:19:32.521043500'), END_TIME: pd.Timestamp('2008-11-26 00:45:00'), ACTIVITY: 'Relax'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['26.11.2008 00:49:10.000002','26.11.2008 00:51:10.000005','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-26 00:49:22.083608600'), END_TIME: pd.Timestamp('2008-11-26 00:51:10.000005'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Create operation:

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-26 08:13:44.627900'), END_TIME: pd.Timestamp('2008-11-26 08:14:46.248400'), ACTIVITY: 'Use toilet upstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['26.11.2008 08:42:00.000004','26.11.2008 16:58:59.999997','Leave house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-26 08:42:00.000004'), END_TIME: pd.Timestamp('2008-11-26 16:59:57.817065800'), ACTIVITY: 'Leave house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['26.11.2008 17:08:10.000001','26.11.2008 17:08:54.999998','Use toilet downstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

#----------------------------------------
# 


# Create operation:

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-26 17:09:25.267000'), END_TIME: pd.Timestamp('2008-11-26 17:10:01.302600'), ACTIVITY: 'Use toilet downstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['26.11.2008 19:13:09.999996','26.11.2008 19:14:10.000003','Use toilet downstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

#----------------------------------------
# 


# Create operation:

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-26 19:14:54.313100'), END_TIME: pd.Timestamp('2008-11-26 19:15:32.180800'), ACTIVITY: 'Use toilet downstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['26.11.2008 17:15:29.999998','26.11.2008 19:13:00.000000','Relax']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-26 17:15:29.999998'), END_TIME: pd.Timestamp('2008-11-26 19:14:46.228312800'), ACTIVITY: 'Relax'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['26.11.2008 21:49:30.000004','26.11.2008 21:51:10.000005','Relax']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)
#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['26.11.2008 19:33:59.999996','26.11.2008 21:48:40.000004','Leave house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-26 19:35:46.726362'), END_TIME: pd.Timestamp('2008-11-26 21:50:21.305788400'), ACTIVITY: 'Leave house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['26.11.2008 21:52:00.000995','26.11.2008 22:08:49.999999','Relax']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-26 22:00:03.233700'), END_TIME: pd.Timestamp('2008-11-26 22:08:51.663400'), ACTIVITY: 'Relax'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Create operation:

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-26 21:52:56.203500'), END_TIME: pd.Timestamp('2008-11-26 21:53:16.068000'), ACTIVITY: 'Get snack'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['26.11.2008 21:51:10.001005','26.11.2008 21:51:59.999995','Get snack']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

#----------------------------------------
# 


# Create operation:

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-26 21:54:00.291700'), END_TIME: pd.Timestamp('2008-11-26 21:59:36.026900'), ACTIVITY: 'Relax'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['26.11.2008 23:39:30.001003','26.11.2008 23:48:20.000005','Relax']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)



#----------------------------------------
# 


# Create operation:

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-26 23:40:06.084800'), END_TIME: pd.Timestamp('2008-11-26 23:42:12.036400'), ACTIVITY: 'Get snack'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['26.11.2008 23:38:50.000999','26.11.2008 23:39:30.000003','Get snack']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)
#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['26.11.2008 23:49:10.000995','27.11.2008 00:03:29.999998','Relax']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-26 23:52:50.794400'), END_TIME: pd.Timestamp('2008-11-27 00:03:29.678600'), ACTIVITY: 'Relax'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Create operation:

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-26 23:50:40.435600'), END_TIME: pd.Timestamp('2008-11-26 23:51:18.744600'), ACTIVITY: 'Use toilet downstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['26.11.2008 23:48:20.001005','26.11.2008 23:49:09.999995','Use toilet downstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['26.11.2008 23:04:29.999997','26.11.2008 23:38:49.999999','Relax']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-26 23:04:29.999997'), END_TIME: pd.Timestamp('2008-11-26 23:40:05.456205800'), ACTIVITY: 'Relax'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Create operation:

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-26 23:42:12.428100'), END_TIME: pd.Timestamp('2008-11-26 23:50:34.603400'), ACTIVITY: 'Relax'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['27.11.2008 00:07:49.999995','27.11.2008 00:11:20.000003','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

#----------------------------------------
# 


# Create operation:

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-27 00:07:26.039900'), END_TIME: pd.Timestamp('2008-11-27 00:09:32.352100'), ACTIVITY: 'Brush teeth'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['27.11.2008 00:04:49.999996','27.11.2008 00:07:15.000004','Brush teeth']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['27.11.2008 00:11:20.001003','27.11.2008 00:12:10.000000','Use toilet downstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['27.11.2008 07:02:55.000003','27.11.2008 07:05:59.999995','Get dressed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-27 07:05:01.128300'), END_TIME: pd.Timestamp('2008-11-27 07:08:09.926000'), ACTIVITY: 'Get dressed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Create operation:

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-27 07:02:33.454500'), END_TIME: pd.Timestamp('2008-11-27 07:04:04.449400'), ACTIVITY: 'Brush teeth'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['27.11.2008 06:59:59.999997','27.11.2008 07:02:00.000000','Brush teeth']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['27.11.2008 06:53:59.999998','27.11.2008 06:59:29.999998','Take shower']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-27 06:56:52.881800'), END_TIME: pd.Timestamp('2008-11-27 07:01:52.830600'), ACTIVITY: 'Take shower'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Create operation:

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-27 06:54:25.589600'), END_TIME: pd.Timestamp('2008-11-27 06:56:31.222000'), ACTIVITY: 'Use toilet upstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['27.11.2008 06:52:10.000001','27.11.2008 06:53:50.000002','Use toilet upstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['27.11.2008 00:12:20.001999','27.11.2008 06:51:49.999999','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-27 00:20:28.950700'), END_TIME: pd.Timestamp('2008-11-27 06:53:35.296500'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['27.11.2008 18:07:10.000001','27.11.2008 18:11:20.000003','Brush teeth']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-27 18:09:34.635300'), END_TIME: pd.Timestamp('2008-11-27 18:13:34.119600'), ACTIVITY: 'Brush teeth'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Create operation:

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-27 18:07:06.857000'), END_TIME: pd.Timestamp('2008-11-27 18:07:54.093200'), ACTIVITY: 'Use toilet downstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['27.11.2008 18:04:50.000996','27.11.2008 18:05:39.999996','Use toilet downstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['27.11.2008 07:20:30.000004','27.11.2008 17:15:29.999998','Leave house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-27 07:21:52.518600970'), END_TIME: pd.Timestamp('2008-11-27 17:17:56.515855'), ACTIVITY: 'Leave house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['27.11.2008 17:58:19.999999','27.11.2008 18:04:49.999996','Relax']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-27 18:00:40.316600'), END_TIME: pd.Timestamp('2008-11-27 18:06:35.651200'), ACTIVITY: 'Relax'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['27.11.2008 17:45:50.001004','27.11.2008 17:57:29.999999','Eating']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-27 17:48:19.494000'), END_TIME: pd.Timestamp('2008-11-27 18:00:37.446600'), ACTIVITY: 'Eating'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['27.11.2008 17:18:29.999998','27.11.2008 17:45:50.000004','Prepare Dinner']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-27 17:18:29.999998'), END_TIME: pd.Timestamp('2008-11-27 17:47:50.696030660'), ACTIVITY: 'Prepare Dinner'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['27.11.2008 20:18:50.000000','27.11.2008 22:36:49.999999','Leave house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-27 20:21:15.394958500'), END_TIME: pd.Timestamp('2008-11-27 22:38:55.690468'), ACTIVITY: 'Leave house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['27.11.2008 20:14:39.999998','27.11.2008 20:17:30.000001','Use toilet downstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-27 20:17:02.448100'), END_TIME: pd.Timestamp('2008-11-27 20:19:38.463400'), ACTIVITY: 'Use toilet downstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['27.11.2008 18:22:09.999998','27.11.2008 20:10:40.000002','Leave house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-27 18:24:27.640400'), END_TIME: pd.Timestamp('2008-11-27 20:13:26.220600'), ACTIVITY: 'Leave house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['27.11.2008 22:42:25.001003','28.11.2008 00:11:09.999997','Relax']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-27 22:44:46.994973600'), END_TIME: pd.Timestamp('2008-11-28 00:12:59.898843'), ACTIVITY: 'Relax'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Create operation:

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-27 22:43:38.778900'), END_TIME: pd.Timestamp('2008-11-27 22:44:16.311800'), ACTIVITY: 'Get snack'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['27.11.2008 22:41:55.001004','27.11.2008 22:42:25.000003','Get snack']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['27.11.2008 22:38:59.999998','27.11.2008 22:41:55.000004','Relax']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-27 22:39:53.811500'), END_TIME: pd.Timestamp('2008-11-27 22:43:34.929400'), ACTIVITY: 'Relax'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Create operation:

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-28 00:13:49.427300'), END_TIME: pd.Timestamp('2008-11-28 00:14:30.317100'), ACTIVITY: 'Use toilet downstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['28.11.2008 22:41:00.000000','28.11.2008 22:41:50.000000','Use toilet downstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-28 22:40:49.833280800'), END_TIME: pd.Timestamp('2008-11-28 22:41:40.210547999'), ACTIVITY: 'Use toilet downstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['28.11.2008 22:45:40.000000','28.11.2008 22:48:45.000002','Get drink']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-28 22:45:17.132698'), END_TIME: pd.Timestamp('2008-11-28 22:48:20.744842800'), ACTIVITY: 'Get drink'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['28.11.2008 22:50:10.000004','28.11.2008 23:45:04.999995','Leave house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-28 22:49:39.867116500'), END_TIME: pd.Timestamp('2008-11-28 23:45:04.999995'), ACTIVITY: 'Leave house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['29.11.2008 11:46:40.000003','29.11.2008 16:43:50.000003','Leave house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-29 11:46:22.804121900'), END_TIME: pd.Timestamp('2008-11-29 16:43:50.000003'), ACTIVITY: 'Leave house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['29.11.2008 16:55:20.000002','29.11.2008 16:56:34.999998','Use toilet upstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-29 16:55:06.818274400'), END_TIME: pd.Timestamp('2008-11-29 16:56:22.001785100'), ACTIVITY: 'Use toilet upstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['29.11.2008 17:09:59.999997','29.11.2008 18:00:20.000001','Leave house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-29 17:09:44.494077700'), END_TIME: pd.Timestamp('2008-11-29 18:00:20.000001'), ACTIVITY: 'Leave house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['29.11.2008 19:21:55.000004','30.11.2008 01:43:54.999996','Leave house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-29 19:21:08.599485'), END_TIME: pd.Timestamp('2008-11-30 01:43:54.999996'), ACTIVITY: 'Leave house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['30.11.2008 01:52:29.999999','30.11.2008 08:54:20.000002','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-30 01:53:58.000232500'), END_TIME: pd.Timestamp('2008-11-30 08:53:38.640214900'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['02.12.2008 00:41:30.000997','02.12.2008 00:43:11.999997','Use toilet downstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-12-02 00:42:32.840141600'), END_TIME: pd.Timestamp('2008-12-02 00:43:11.999997'), ACTIVITY: 'Use toilet downstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['02.12.2008 06:11:30.001004','02.12.2008 06:16:20.000000','Take shower']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

#----------------------------------------
# 


# Join operation: 
idx_to_del = get_index_matching_rows(df_acts, 	[['02.12.2008 01:01:20.000000','02.12.2008 06:11:30.000004','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)
idx_to_del = get_index_matching_rows(df_acts, 	[['02.12.2008 06:16:20.001000','02.12.2008 07:02:04.999998','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)
new_row=pd.Series({START_TIME: pd.Timestamp('2008-12-02 01:01:20'), END_TIME: pd.Timestamp('2008-12-02 07:02:04.999998'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['03.12.2008 18:50:10.000000','03.12.2008 19:57:55.000000','Relax']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-12-03 18:50:10'), END_TIME: pd.Timestamp('2008-12-03 19:56:41.484823600'), ACTIVITY: 'Relax'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['03.12.2008 19:57:55.001000','03.12.2008 19:58:35.000004','Use toilet downstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-12-03 19:57:01.853000'), END_TIME: pd.Timestamp('2008-12-03 19:58:21.291200'), ACTIVITY: 'Use toilet downstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['03.12.2008 19:58:35.001004','03.12.2008 21:49:10.000004','Relax']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-12-03 19:58:35.001004'), END_TIME: pd.Timestamp('2008-12-03 21:47:54.275700500'), ACTIVITY: 'Relax'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['03.12.2008 21:49:10.001004','03.12.2008 21:51:29.999999','Use toilet downstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-12-03 21:48:27.540100'), END_TIME: pd.Timestamp('2008-12-03 21:49:16.037200'), ACTIVITY: 'Use toilet downstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['03.12.2008 21:51:30.000999','03.12.2008 22:04:15.000004','Relax']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-12-03 21:49:39.038400'), END_TIME: pd.Timestamp('2008-12-03 22:03:25.931600'), ACTIVITY: 'Relax'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['03.12.2008 22:04:15.001004','03.12.2008 22:06:10.000004','Use toilet downstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-12-03 22:03:31.318800'), END_TIME: pd.Timestamp('2008-12-03 22:05:46.554500'), ACTIVITY: 'Use toilet downstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['04.12.2008 22:44:45.000002','04.12.2008 22:45:30.000000','Get drink']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-12-04 22:44:18.493700'), END_TIME: pd.Timestamp('2008-12-04 22:44:54.911500'), ACTIVITY: 'Get drink'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['05.12.2008 16:51:24.999995','05.12.2008 17:09:45.000000','Prepare Dinner']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-12-05 16:51:24.999995'), END_TIME: pd.Timestamp('2008-12-05 17:06:14.384521600'), ACTIVITY: 'Prepare Dinner'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['06.12.2008 12:16:20.000000','06.12.2008 13:24:19.999998','Leave house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-12-06 12:16:50.670415'), END_TIME: pd.Timestamp('2008-12-06 13:24:19.999998'), ACTIVITY: 'Leave house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['07.12.2008 01:05:00.000004','07.12.2008 01:59:20.000004','Relax']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-12-07 01:05:00.000004'), END_TIME: pd.Timestamp('2008-12-07 02:01:29.916585600'), ACTIVITY: 'Relax'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['21.11.2008 15:42:59.999997','21.11.2008 16:29:39.999998','Take shower']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-21 15:42:59.999997'), END_TIME: pd.Timestamp('2008-11-21 16:04:12.685900'), ACTIVITY: 'Take shower'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-21 16:05:03.753300'), END_TIME: pd.Timestamp('2008-11-21 16:29:39.999998'), ACTIVITY: 'Take shower'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['26.11.2008 00:49:22.083608','26.11.2008 00:51:10.000005','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['03.12.2008 06:24:14.999999','03.12.2008 06:27:40.000003','Eating']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-12-03 06:24:14.999999'), END_TIME: pd.Timestamp('2008-12-03 06:29:27.116142400'), ACTIVITY: 'Eating'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['28.11.2008 07:13:29.999998','28.11.2008 07:13:59.999996','Use toilet upstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-28 07:13:16.125366500'), END_TIME: pd.Timestamp('2008-11-28 07:13:59.999996'), ACTIVITY: 'Use toilet upstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['22.11.2008 13:04:00.000002','22.11.2008 13:09:15.907232','Leave house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['21.11.2008 21:30:55.659900','21.11.2008 21:32:12.639200','Leave house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['26.11.2008 08:16:10.000003','26.11.2008 08:17:30.000001','Use toilet upstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-26 08:16:10.000003'), END_TIME: pd.Timestamp('2008-11-26 08:17:34.236662700'), ACTIVITY: 'Use toilet upstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['20.11.2008 10:47:00.001000','20.11.2008 11:02:10.000002','Eating']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-20 10:49:47.676400'), END_TIME: pd.Timestamp('2008-11-20 11:02:10.000002'), ACTIVITY: 'Eating'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['26.11.2008 23:40:06.084800','26.11.2008 23:42:12.036400','Get snack']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-26 23:40:06.084800'), END_TIME: pd.Timestamp('2008-11-26 23:42:11.375000'), ACTIVITY: 'Get snack'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['26.11.2008 00:51:30.000997','26.11.2008 08:12:59.999997','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-26 00:57:40.017500'), END_TIME: pd.Timestamp('2008-11-26 08:12:59.999997'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['23.11.2008 05:28:50.001004','23.11.2008 05:29:50.000001','Use toilet upstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-23 05:28:56.771100'), END_TIME: pd.Timestamp('2008-11-23 05:30:54.480400'), ACTIVITY: 'Use toilet upstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['05.12.2008 16:51:24.999995','05.12.2008 17:06:14.384521','Prepare Dinner']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-12-05 16:51:24.999995'), END_TIME: pd.Timestamp('2008-12-05 17:05:45.802200'), ACTIVITY: 'Prepare Dinner'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['25.11.2008 23:23:22.999998','26.11.2008 00:19:13.877721','Relax']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-25 23:23:22.999998'), END_TIME: pd.Timestamp('2008-11-26 00:17:02.590247799'), ACTIVITY: 'Relax'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['26.11.2008 00:19:32.521043','26.11.2008 00:45:00.000000','Relax']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-26 00:21:56.990500'), END_TIME: pd.Timestamp('2008-11-26 00:45:00'), ACTIVITY: 'Relax'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['26.11.2008 00:19:14.665467','26.11.2008 00:19:31.470715','Get snack']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-26 00:17:22.332600'), END_TIME: pd.Timestamp('2008-11-26 00:21:49.866300'), ACTIVITY: 'Get snack'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['01.12.2008 23:53:40.000004','02.12.2008 00:35:50.000001','Relax']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-12-01 23:53:40.000004'), END_TIME: pd.Timestamp('2008-12-02 00:35:29.043600'), ACTIVITY: 'Relax'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['02.12.2008 00:35:50.001001','02.12.2008 00:41:29.999997','Eating']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-12-02 00:35:35.019400'), END_TIME: pd.Timestamp('2008-12-02 00:41:29.999997'), ACTIVITY: 'Eating'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['29.11.2008 18:34:49.999998','29.11.2008 18:52:00.000004','Relax']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-29 18:34:49.999998'), END_TIME: pd.Timestamp('2008-11-29 18:51:13.637900'), ACTIVITY: 'Relax'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['04.12.2008 07:00:00.000001','04.12.2008 07:05:30.000002','Take shower']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-12-04 07:00:54.889100'), END_TIME: pd.Timestamp('2008-12-04 07:05:30.000002'), ACTIVITY: 'Take shower'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['22.11.2008 09:37:20.000001','22.11.2008 09:38:19.999997','Use toilet upstairs']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-22 09:37:34.585000'), END_TIME: pd.Timestamp('2008-11-22 09:39:25.490500'), ACTIVITY: 'Use toilet upstairs'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['03.12.2008 00:04:09.999997','03.12.2008 00:11:30.000004','Shave']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-12-03 00:04:09.999997'), END_TIME: pd.Timestamp('2008-12-03 00:07:58.168200'), ACTIVITY: 'Shave'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['03.12.2008 00:11:30.001004','03.12.2008 00:16:05.000001','Brush teeth']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-12-03 00:08:01.893300'), END_TIME: pd.Timestamp('2008-12-03 00:11:34.934800'), ACTIVITY: 'Brush teeth'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['01.12.2008 11:26:19.999998','01.12.2008 17:28:14.999998','Leave house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-12-01 11:28:30.781900'), END_TIME: pd.Timestamp('2008-12-01 17:28:14.999998'), ACTIVITY: 'Leave house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['02.12.2008 07:24:10.000002','02.12.2008 22:54:20.000002','Leave house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-12-02 07:32:42.030200'), END_TIME: pd.Timestamp('2008-12-02 22:54:20.000002'), ACTIVITY: 'Leave house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['28.11.2008 20:12:00.001000','28.11.2008 20:22:04.999997','Eating']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-28 20:13:08.529500'), END_TIME: pd.Timestamp('2008-11-28 20:22:04.999997'), ACTIVITY: 'Eating'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['28.11.2008 20:03:30.000000','28.11.2008 20:12:00.000000','Prepare Dinner']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-28 20:03:30'), END_TIME: pd.Timestamp('2008-11-28 20:13:03.564900'), ACTIVITY: 'Prepare Dinner'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['24.11.2008 07:34:10.000002','24.11.2008 07:43:10.000000','Prepare Breakfast']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-24 07:34:10.000002'), END_TIME: pd.Timestamp('2008-11-24 07:42:57.206000'), ACTIVITY: 'Prepare Breakfast'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['24.11.2008 07:45:30.001005','24.11.2008 07:57:10.000000','Prepare Breakfast']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-24 07:45:39.951823300'), END_TIME: pd.Timestamp('2008-11-24 07:57:10'), ACTIVITY: 'Prepare Breakfast'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['24.11.2008 07:43:10.001000','24.11.2008 07:45:30.000005','Get dressed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-24 07:43:02.716600'), END_TIME: pd.Timestamp('2008-11-24 07:45:34.979200'), ACTIVITY: 'Get dressed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['26.11.2008 17:15:29.999998','26.11.2008 19:14:46.228312','Relax']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-26 17:20:18.573816800'), END_TIME: pd.Timestamp('2008-11-26 19:14:46.228312800'), ACTIVITY: 'Relax'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['24.11.2008 18:00:00.000000','24.11.2008 18:15:30.000005','Relax']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-24 18:00:00'), END_TIME: pd.Timestamp('2008-11-24 18:05:03.939500'), ACTIVITY: 'Relax'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-24 18:08:14.552500'), END_TIME: pd.Timestamp('2008-11-24 18:15:30.000005'), ACTIVITY: 'Relax'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Create operation:

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-24 18:05:05.940100'), END_TIME: pd.Timestamp('2008-11-24 18:08:13.586300'), ACTIVITY: 'Get snack'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['27.11.2008 22:44:46.994973','28.11.2008 00:12:59.898843','Relax']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-27 22:44:46.994973600'), END_TIME: pd.Timestamp('2008-11-27 23:11:36.333200'), ACTIVITY: 'Relax'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-27 23:14:28.528800'), END_TIME: pd.Timestamp('2008-11-28 00:12:59.898843'), ACTIVITY: 'Relax'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Create operation:

new_row=pd.Series({START_TIME: pd.Timestamp('2008-11-27 23:11:41.236100'), END_TIME: pd.Timestamp('2008-11-27 23:14:26.470600'), ACTIVITY: 'Get snack'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)
