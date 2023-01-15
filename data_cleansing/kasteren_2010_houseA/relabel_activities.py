# Awesome script
import pandas as pd
from pyadlml.constants import START_TIME, END_TIME, ACTIVITY
from pyadlml.dataset._core.activities import get_index_matching_rows


#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['07.03.2008 11:35:57.000002','08.03.2008 06:21:26.999000','Leave house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-07 11:35:57.000002'), END_TIME: pd.Timestamp('2008-03-08 06:18:58.487900'), ACTIVITY: 'Leave house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)



#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['29.02.2008 10:48:43.000004','29.02.2008 18:18:34.000003','Leave house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-02-29 10:48:43.000004'), END_TIME: pd.Timestamp('2008-02-29 18:18:00.156100'), ACTIVITY: 'Leave house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['11.03.2008 20:34:13.999997','12.03.2008 00:38:02.000004','Leave house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-11 20:34:13.999997'), END_TIME: pd.Timestamp('2008-03-12 00:35:38.789900'), ACTIVITY: 'Leave house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['16.03.2008 00:49:08.999997','16.03.2008 00:50:39.999996','Brush teeth']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-16 00:49:13.755800'), END_TIME: pd.Timestamp('2008-03-16 00:50:39.999996'), ACTIVITY: 'Brush teeth'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['25.02.2008 00:19:32.000000','25.02.2008 00:21:23.999996','Brush teeth']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['25.02.2008 23:29:14.001005','26.02.2008 00:39:24.000002','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-02-25 23:32:20.653700'), END_TIME: pd.Timestamp('2008-02-26 00:38:50.893800'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['25.02.2008 23:28:30.001002','25.02.2008 23:29:14.000005','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

#----------------------------------------
# 


# Create operation:

new_row=pd.Series({START_TIME: pd.Timestamp('2008-02-25 23:30:11.305600'), END_TIME: pd.Timestamp('2008-02-25 23:30:53.470800'), ACTIVITY: 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['26.02.2008 00:39:24.001002','26.02.2008 00:39:39.999996','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-02-26 00:38:51.292600'), END_TIME: pd.Timestamp('2008-02-26 00:39:39.999996'), ACTIVITY: 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['26.02.2008 10:05:59.000001','26.02.2008 20:34:20.000005','Leave house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-02-26 10:05:59.000001'), END_TIME: pd.Timestamp('2008-02-26 20:32:41.514200'), ACTIVITY: 'Leave house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['27.02.2008 05:29:11.001001','27.02.2008 05:30:25.999996','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-02-27 05:29:31.590300'), END_TIME: pd.Timestamp('2008-02-27 05:30:17.745900'), ACTIVITY: 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['26.02.2008 23:01:33.000002','27.02.2008 05:29:11.000001','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-02-26 23:02:17.092900'), END_TIME: pd.Timestamp('2008-02-27 05:29:24.714600'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['27.02.2008 05:30:26.000996','27.02.2008 08:00:48.000005','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-02-27 05:30:23.645400'), END_TIME: pd.Timestamp('2008-02-27 08:00:48.000005'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['27.02.2008 08:13:55.000001','27.02.2008 08:16:06.999999','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-02-27 08:13:42.437400'), END_TIME: pd.Timestamp('2008-02-27 08:15:42.329200'), ACTIVITY: 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Create operation:

new_row=pd.Series({START_TIME: pd.Timestamp('2008-02-27 20:35:29.920100'), END_TIME: pd.Timestamp('2008-02-27 20:35:58.767700'), ACTIVITY: 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['28.02.2008 00:16:15.999997','28.02.2008 07:06:27.000000','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-02-28 00:20:12.485000'), END_TIME: pd.Timestamp('2008-02-28 07:06:27'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['28.02.2008 09:49:58.999998','28.02.2008 09:51:46.999995','Prepare Breakfast']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-02-28 09:49:58.999998'), END_TIME: pd.Timestamp('2008-02-28 09:52:11.137100'), ACTIVITY: 'Prepare Breakfast'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['29.02.2008 08:03:13.001003','29.02.2008 08:04:22.999995','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-02-29 08:03:26.748800'), END_TIME: pd.Timestamp('2008-02-29 08:04:22.999995'), ACTIVITY: 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['28.02.2008 23:24:40.999997','29.02.2008 08:03:13.000003','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-02-28 23:25:14.867300'), END_TIME: pd.Timestamp('2008-02-29 08:03:19.629600'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['05.03.2008 23:18:33.999996','06.03.2008 06:59:52.000000','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-05 23:19:01.079700'), END_TIME: pd.Timestamp('2008-03-06 06:59:52'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Create operation:

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-05 23:18:29.508600'), END_TIME: pd.Timestamp('2008-03-05 23:18:41.767500'), ACTIVITY: 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['05.03.2008 23:17:37.000003','05.03.2008 23:18:25.000005','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['07.03.2008 01:11:53.000002','07.03.2008 01:12:43.000002','Get drink']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-07 01:11:53.000002'), END_TIME: pd.Timestamp('2008-03-07 01:12:43.000002'), ACTIVITY: 'Get drink'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['07.03.2008 01:11:53.000002','07.03.2008 01:12:43.000002','Get drink']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-07 01:11:53.000002'), END_TIME: pd.Timestamp('2008-03-07 01:14:30.573900'), ACTIVITY: 'Get drink'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Create operation:

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-08 06:20:27.242700'), END_TIME: pd.Timestamp('2008-03-08 06:20:59.652800'), ACTIVITY: 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['08.03.2008 08:49:48.000000','08.03.2008 08:51:57.999999','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-08 08:49:48'), END_TIME: pd.Timestamp('2008-03-08 08:51:30.649400'), ACTIVITY: 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['08.03.2008 22:05:36.000005','09.03.2008 06:51:36.000005','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-08 22:05:30.494700'), END_TIME: pd.Timestamp('2008-03-09 06:51:13.165700'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['09.03.2008 21:53:31.000004','10.03.2008 03:04:56.999998','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-09 21:55:00.358500'), END_TIME: pd.Timestamp('2008-03-10 03:04:56.999998'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['11.03.2008 06:29:10.001003','11.03.2008 08:22:10.999996','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-11 06:30:23.767000'), END_TIME: pd.Timestamp('2008-03-11 08:21:56.830900'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Create operation:

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-11 19:07:37.759600'), END_TIME: pd.Timestamp('2008-03-11 20:27:45.597200'), ACTIVITY: 'Leave house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['12.03.2008 00:45:01.000005','12.03.2008 05:29:09.000002','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-12 00:46:20.033000'), END_TIME: pd.Timestamp('2008-03-12 05:29:09.000002'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['12.03.2008 05:29:09.001002','12.03.2008 05:43:33.000003','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-12 05:29:09.001002'), END_TIME: pd.Timestamp('2008-03-12 05:39:18.350500'), ACTIVITY: 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['13.03.2008 04:06:50.000996','13.03.2008 04:07:52.999996','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-13 04:07:01.762700'), END_TIME: pd.Timestamp('2008-03-13 04:07:50.014600'), ACTIVITY: 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['13.03.2008 08:59:43.001002','13.03.2008 09:00:31.999997','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-13 08:59:49.450900'), END_TIME: pd.Timestamp('2008-03-13 09:00:24.512900'), ACTIVITY: 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['13.03.2008 04:07:53.000996','13.03.2008 08:59:43.000002','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-13 04:07:53.000996'), END_TIME: pd.Timestamp('2008-03-13 08:59:45.484300'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['13.03.2008 09:00:32.000997','13.03.2008 09:53:24.000002','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-13 09:01:37.841100'), END_TIME: pd.Timestamp('2008-03-13 09:53:32.302100'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['15.03.2008 05:44:24.000997','15.03.2008 05:45:09.999999','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-15 05:44:33.089000'), END_TIME: pd.Timestamp('2008-03-15 05:45:08.374900'), ACTIVITY: 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['17.03.2008 00:44:11.000004','17.03.2008 07:50:59.999995','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-17 00:43:42.409600'), END_TIME: pd.Timestamp('2008-03-17 07:50:59.999995'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['17.03.2008 07:51:58.001003','17.03.2008 08:43:24.000001','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-17 07:52:20.112100'), END_TIME: pd.Timestamp('2008-03-17 08:43:24.000001'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['17.03.2008 21:54:29.999996','18.03.2008 00:35:06.999998','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-17 21:54:42.211100'), END_TIME: pd.Timestamp('2008-03-18 00:35:06.999998'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['19.03.2008 03:49:46.000998','19.03.2008 03:50:46.999999','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-19 03:49:53.258500'), END_TIME: pd.Timestamp('2008-03-19 03:50:45.714900'), ACTIVITY: 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['18.03.2008 23:59:37.999999','19.03.2008 03:49:45.999998','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-19 00:00:10.211500'), END_TIME: pd.Timestamp('2008-03-19 03:49:47.341800'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['19.03.2008 08:30:56.001005','19.03.2008 08:31:35.999999','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-19 08:31:01.242200'), END_TIME: pd.Timestamp('2008-03-19 08:31:29.835500'), ACTIVITY: 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['08.03.2008 21:31:57.999997','08.03.2008 21:34:53.999998','Get drink']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-08 21:31:57.999997'), END_TIME: pd.Timestamp('2008-03-08 21:33:22.861800'), ACTIVITY: 'Get drink'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['04.03.2008 10:11:38.000003','04.03.2008 19:48:47.999997','Leave house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-04 10:11:38.000003'), END_TIME: pd.Timestamp('2008-03-04 19:48:33.325800'), ACTIVITY: 'Leave house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['01.03.2008 21:57:57.000000','01.03.2008 22:03:12.000001','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-01 21:57:59.487800'), END_TIME: pd.Timestamp('2008-03-01 21:58:28.996900'), ACTIVITY: 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Create operation:

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-01 22:03:15.401300'), END_TIME: pd.Timestamp('2008-03-01 22:03:44.135200'), ACTIVITY: 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['27.02.2008 23:06:09.999998','27.02.2008 23:14:26.000003','Unload dishwasher']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-02-27 23:06:09.999998'), END_TIME: pd.Timestamp('2008-02-27 23:17:15.948800'), ACTIVITY: 'Unload dishwasher'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['04.03.2008 09:45:37.000000','04.03.2008 09:53:46.999998','Take shower']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-04 09:45:58.749400'), END_TIME: pd.Timestamp('2008-03-04 09:53:46.999998'), ACTIVITY: 'Take shower'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['04.03.2008 00:21:43.000003','04.03.2008 06:00:13.000000','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-04 00:22:06.249000'), END_TIME: pd.Timestamp('2008-03-04 06:00:13'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['11.03.2008 01:23:56.999997','11.03.2008 06:28:00.999995','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-11 01:24:07.028700'), END_TIME: pd.Timestamp('2008-03-11 06:28:00.999995'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['12.03.2008 00:46:20.033000','12.03.2008 05:29:09.000002','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-12 00:46:49.928400'), END_TIME: pd.Timestamp('2008-03-12 05:29:09.000002'), ACTIVITY: 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['08.03.2008 09:25:57.999998','08.03.2008 09:27:24.000005','Prepare Breakfast']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-08 09:26:04.689500'), END_TIME: pd.Timestamp('2008-03-08 09:27:24.000005'), ACTIVITY: 'Prepare Breakfast'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['01.03.2008 20:32:09.999996','01.03.2008 20:34:45.000000','Doing laundry']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-01 20:32:27.888200'), END_TIME: pd.Timestamp('2008-03-01 20:34:45'), ACTIVITY: 'Doing laundry'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['04.03.2008 23:32:46.999996','04.03.2008 23:33:19.000004','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2008-03-04 23:32:46.999996'), END_TIME: pd.Timestamp('2008-03-04 23:33:36.838800'), ACTIVITY: 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Create operation:

new_row=pd.Series({START_TIME: pd.Timestamp('2008-02-27 23:26:28.316000'), END_TIME: pd.Timestamp('2008-02-27 23:37:08.156500'), ACTIVITY: 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)
