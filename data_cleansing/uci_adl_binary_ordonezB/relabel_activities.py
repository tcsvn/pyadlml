# Awesome script
import pandas as pd
from pyadlml.constants import START_TIME, END_TIME, ACTIVITY
from pyadlml.dataset._core.activities import get_index_matching_rows


#----------------------------------------
# Activity ends earlier


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['11.11.2012 21:14:00.000000','12.11.2012 00:22:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-11 21:14:00'), END_TIME: pd.Timestamp('2012-11-12 00:22:23.091297'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['12.11.2012 00:24:00.000000','12.11.2012 00:43:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-12 00:25:12.041324400'), END_TIME: pd.Timestamp('2012-11-12 00:43:25.345096900'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['12.11.2012 01:54:00.000000','12.11.2012 09:31:59.000000','Sleeping']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-12 01:54:47.661768800'), END_TIME: pd.Timestamp('2012-11-12 09:30:57.025579200'), ACTIVITY: 'Sleeping'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['12.11.2012 01:53:00.000000','12.11.2012 01:53:59.000000','Toileting']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-12 01:53:21.134423800'), END_TIME: pd.Timestamp('2012-11-12 01:53:49.079900800'), ACTIVITY: 'Toileting'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['12.11.2012 01:52:00.000000','12.11.2012 01:52:59.000000','Grooming']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-12 01:52:38.765474800'), END_TIME: pd.Timestamp('2012-11-12 01:53:20.954130400'), ACTIVITY: 'Grooming'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['12.11.2012 00:50:00.000000','12.11.2012 01:51:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-12 00:50:44.317153800'), END_TIME: pd.Timestamp('2012-11-12 01:52:37.323127600'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['12.11.2012 00:48:00.000000','12.11.2012 00:49:59.000000','Grooming']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-12 00:48:28.721684500'), END_TIME: pd.Timestamp('2012-11-12 00:50:20.617086100'), ACTIVITY: 'Grooming'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# No place for this activity to happen


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['12.11.2012 10:33:00.000000','12.11.2012 10:34:59.000000','Grooming']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['12.11.2012 10:35:00.000000','12.11.2012 10:35:59.000000','Toileting']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-12 10:34:10.337484'), END_TIME: pd.Timestamp('2012-11-12 10:35:11.673786'), ACTIVITY: 'Toileting'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['12.11.2012 10:36:00.000000','12.11.2012 10:40:59.000000','Grooming']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-12 10:35:13.845066800'), END_TIME: pd.Timestamp('2012-11-12 10:40:04.670399100'), ACTIVITY: 'Grooming'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['12.11.2012 12:52:58.000000','12.11.2012 12:55:59.000000','Toileting']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-12 12:51:55.571046600'), END_TIME: pd.Timestamp('2012-11-12 12:55:58.898710900'), ACTIVITY: 'Toileting'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Join operation: 
idx_to_del = get_index_matching_rows(df_acts, 	[['12.11.2012 13:00:00.000000','12.11.2012 13:07:59.000000','Grooming']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)
idx_to_del = get_index_matching_rows(df_acts, 	[['12.11.2012 13:12:00.000000','12.11.2012 13:18:59.000000','Grooming']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)
new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-12 13:00:00'), END_TIME: pd.Timestamp('2012-11-12 13:18:59'), ACTIVITY: 'Grooming'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['12.11.2012 17:10:00.000000','12.11.2012 17:29:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-12 17:10:00'), END_TIME: pd.Timestamp('2012-11-12 17:29:25.267774700'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['12.11.2012 21:14:00.000000','12.11.2012 21:15:59.000000','Grooming']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-12 21:14:00'), END_TIME: pd.Timestamp('2012-11-12 21:15:08.576382800'), ACTIVITY: 'Grooming'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['12.11.2012 23:46:00.000000','13.11.2012 01:27:00.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-12 23:46:48.182534300'), END_TIME: pd.Timestamp('2012-11-13 01:27:00'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['13.11.2012 09:53:00.000000','13.11.2012 10:11:59.000000','Leaving']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-13 09:53:18.093000'), END_TIME: pd.Timestamp('2012-11-13 10:11:05.872900'), ACTIVITY: 'Leaving'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['13.11.2012 17:37:00.000000','13.11.2012 18:48:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-13 17:38:20.964335700'), END_TIME: pd.Timestamp('2012-11-13 18:47:38.595504400'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['13.11.2012 23:45:00.000000','14.11.2012 00:21:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-13 23:45:42.234142600'), END_TIME: pd.Timestamp('2012-11-14 00:20:46.826240200'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['14.11.2012 00:28:00.000000','14.11.2012 00:29:59.000000','Toileting']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-14 00:28:00'), END_TIME: pd.Timestamp('2012-11-14 00:29:33.998746'), ACTIVITY: 'Toileting'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['14.11.2012 00:29:59.999999','14.11.2012 05:12:59.000000','Sleeping']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-14 00:29:36.774867'), END_TIME: pd.Timestamp('2012-11-14 05:12:00.446810400'), ACTIVITY: 'Sleeping'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['14.11.2012 09:06:00.000000','14.11.2012 09:17:59.000000','Breakfast']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-14 09:06:28.643565600'), END_TIME: pd.Timestamp('2012-11-14 09:17:59'), ACTIVITY: 'Breakfast'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['14.11.2012 12:29:00.000000','14.11.2012 12:52:00.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-14 12:29:00'), END_TIME: pd.Timestamp('2012-11-14 12:51:59.997597600'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['14.11.2012 12:52:00.000100','14.11.2012 12:54:59.000000','Leaving']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-14 12:52:20.710836'), END_TIME: pd.Timestamp('2012-11-14 12:54:33.707514'), ACTIVITY: 'Leaving'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['14.11.2012 12:52:20.710836','14.11.2012 12:54:33.707514','Leaving']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['14.11.2012 15:58:00.000000','14.11.2012 19:48:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-14 15:59:04.842049800'), END_TIME: pd.Timestamp('2012-11-14 19:48:23.568886'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['14.11.2012 20:11:00.000000','14.11.2012 21:37:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-14 20:11:00'), END_TIME: pd.Timestamp('2012-11-14 21:37:14.427525300'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['14.11.2012 21:37:59.001000','14.11.2012 21:47:59.000000','Dinner']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-14 21:37:21.864757400'), END_TIME: pd.Timestamp('2012-11-14 21:47:59'), ACTIVITY: 'Dinner'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['14.11.2012 22:18:00.000000','14.11.2012 23:06:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-14 22:19:00.371865600'), END_TIME: pd.Timestamp('2012-11-14 23:05:51.703644500'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['15.11.2012 00:10:00.000000','15.11.2012 00:10:59.000000','Grooming']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-15 00:10:00'), END_TIME: pd.Timestamp('2012-11-15 00:10:21.924747200'), ACTIVITY: 'Grooming'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['15.11.2012 00:10:59.001000','15.11.2012 00:39:59.000000','Sleeping']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-15 00:10:38.216860400'), END_TIME: pd.Timestamp('2012-11-15 00:39:59'), ACTIVITY: 'Sleeping'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['15.11.2012 00:40:00.000000','15.11.2012 00:43:59.000000','Sleeping']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-15 00:40:41.298500'), END_TIME: pd.Timestamp('2012-11-15 00:43:08.272300'), ACTIVITY: 'Sleeping'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['15.11.2012 09:53:00.000000','15.11.2012 09:56:59.000000','Grooming']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-15 09:53:43.576653600'), END_TIME: pd.Timestamp('2012-11-15 09:56:59'), ACTIVITY: 'Grooming'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['15.11.2012 10:00:00.000000','15.11.2012 10:07:59.000000','Grooming']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-15 10:05:24.277790500'), END_TIME: pd.Timestamp('2012-11-15 10:07:59'), ACTIVITY: 'Grooming'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Create operation:

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-15 09:58:17.367100'), END_TIME: pd.Timestamp('2012-11-15 10:05:06.683800'), ACTIVITY: 'Showering'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['16.11.2012 00:51:00.000000','16.11.2012 00:53:59.000000','Grooming']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-16 00:51:26.213914'), END_TIME: pd.Timestamp('2012-11-16 00:53:59'), ACTIVITY: 'Grooming'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['16.11.2012 12:09:00.000000','16.11.2012 12:09:59.000000','Snack']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-16 12:09:32.548800'), END_TIME: pd.Timestamp('2012-11-16 12:09:56.329700'), ACTIVITY: 'Snack'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['16.11.2012 12:56:00.000000','16.11.2012 13:52:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-16 12:56:57.244800'), END_TIME: pd.Timestamp('2012-11-16 13:52:17.738288'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['16.11.2012 15:17:00.000000','16.11.2012 18:04:59.000000','Leaving']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-16 15:17:00'), END_TIME: pd.Timestamp('2012-11-16 18:04:11.591601600'), ACTIVITY: 'Leaving'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['16.11.2012 18:16:00.000000','16.11.2012 21:38:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-16 18:17:39.038162900'), END_TIME: pd.Timestamp('2012-11-16 21:38:59'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['17.11.2012 02:31:00.000000','17.11.2012 02:31:59.000000','Toileting']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-17 02:30:32.494627200'), END_TIME: pd.Timestamp('2012-11-17 02:31:24.263612'), ACTIVITY: 'Toileting'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['17.11.2012 02:32:00.000000','17.11.2012 02:33:59.000000','Grooming']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-17 02:31:29.182836800'), END_TIME: pd.Timestamp('2012-11-17 02:33:59'), ACTIVITY: 'Grooming'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['17.11.2012 12:45:00.000000','17.11.2012 14:31:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-17 12:45:00'), END_TIME: pd.Timestamp('2012-11-17 14:08:26.318600'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-17 14:09:05.596300'), END_TIME: pd.Timestamp('2012-11-17 14:31:59'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['17.11.2012 16:32:00.000000','17.11.2012 17:28:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-17 16:33:09.189130800'), END_TIME: pd.Timestamp('2012-11-17 17:28:59'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['17.11.2012 17:39:00.000000','17.11.2012 20:26:00.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-17 17:39:00'), END_TIME: pd.Timestamp('2012-11-17 19:38:13.764800'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-17 19:42:59.990400'), END_TIME: pd.Timestamp('2012-11-17 20:26:00'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['18.11.2012 01:39:00.000000','18.11.2012 08:57:59.000000','Sleeping']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-18 01:39:35.024472'), END_TIME: pd.Timestamp('2012-11-18 08:57:20.423336'), ACTIVITY: 'Sleeping'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['18.11.2012 17:45:00.000000','18.11.2012 19:40:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-18 17:45:54.641955400'), END_TIME: pd.Timestamp('2012-11-18 18:24:23.852620500'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['18.11.2012 17:44:00.000000','18.11.2012 17:44:59.000000','Grooming']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-18 17:44:20.729735'), END_TIME: pd.Timestamp('2012-11-18 17:45:21.354234'), ACTIVITY: 'Grooming'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Create operation:

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-18 18:53:11.409561'), END_TIME: pd.Timestamp('2012-11-18 19:39:55.307100'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['18.11.2012 23:00:00.000000','18.11.2012 23:02:59.000000','Snack']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-18 23:00:39.175000'), END_TIME: pd.Timestamp('2012-11-18 23:02:41.912100'), ACTIVITY: 'Snack'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['18.11.2012 23:05:00.000000','19.11.2012 00:08:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-18 23:05:52.281800'), END_TIME: pd.Timestamp('2012-11-19 00:08:10.252300'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['19.11.2012 14:29:00.000000','19.11.2012 14:29:59.000000','Grooming']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['19.11.2012 14:30:00.000000','19.11.2012 14:30:59.000000','Toileting']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-19 14:29:02.734900'), END_TIME: pd.Timestamp('2012-11-19 14:30:10.268700'), ACTIVITY: 'Toileting'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['19.11.2012 14:31:00.000000','19.11.2012 15:52:59.000000','Leaving']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-19 14:31:22.752225'), END_TIME: pd.Timestamp('2012-11-19 15:52:31.596600600'), ACTIVITY: 'Leaving'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['19.11.2012 22:32:59.000100','20.11.2012 01:22:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-19 22:32:59.000100'), END_TIME: pd.Timestamp('2012-11-20 00:22:25.318300'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-20 00:24:30.219800'), END_TIME: pd.Timestamp('2012-11-20 01:22:59'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['20.11.2012 10:21:00.000000','20.11.2012 10:30:59.000000','Breakfast']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-20 10:21:37.305100'), END_TIME: pd.Timestamp('2012-11-20 10:30:49.325900'), ACTIVITY: 'Breakfast'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['20.11.2012 13:03:00.000000','20.11.2012 13:09:59.000000','Snack']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-20 13:03:24.403200'), END_TIME: pd.Timestamp('2012-11-20 13:09:59.862100'), ACTIVITY: 'Snack'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['21.11.2012 15:57:00.000000','21.11.2012 17:49:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-21 15:58:01.960469600'), END_TIME: pd.Timestamp('2012-11-21 17:49:59'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['22.11.2012 01:40:00.001000','22.11.2012 01:42:59.000000','Snack']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-22 01:40:43.494900'), END_TIME: pd.Timestamp('2012-11-22 01:42:47.402000'), ACTIVITY: 'Snack'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['21.11.2012 21:18:00.000000','22.11.2012 01:40:00.000000','Leaving']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-21 21:18:10.151270800'), END_TIME: pd.Timestamp('2012-11-22 01:40:11.574344900'), ACTIVITY: 'Leaving'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['22.11.2012 11:56:00.000000','22.11.2012 12:09:59.000000','Snack']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-22 11:56:35.168700'), END_TIME: pd.Timestamp('2012-11-22 12:09:28.060600'), ACTIVITY: 'Snack'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['22.11.2012 18:19:00.000000','22.11.2012 20:23:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-22 18:20:02.640405200'), END_TIME: pd.Timestamp('2012-11-22 20:23:59'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['22.11.2012 18:15:00.001000','22.11.2012 18:18:59.000000','Snack']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-22 18:15:41.093591600'), END_TIME: pd.Timestamp('2012-11-22 18:19:30.830657600'), ACTIVITY: 'Snack'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['22.11.2012 21:06:00.000000','23.11.2012 00:34:59.000000','Leaving']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-22 21:15:17.392339600'), END_TIME: pd.Timestamp('2012-11-23 00:34:59'), ACTIVITY: 'Leaving'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['23.11.2012 17:11:00.000000','23.11.2012 17:57:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-23 17:11:51.926835200'), END_TIME: pd.Timestamp('2012-11-23 17:57:12.934138800'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['23.11.2012 16:57:00.000000','23.11.2012 17:10:59.000000','Snack']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-23 16:57:36.035000800'), END_TIME: pd.Timestamp('2012-11-23 17:11:34.768033300'), ACTIVITY: 'Snack'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['24.11.2012 10:19:00.000000','24.11.2012 11:13:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-24 10:19:41.641268600'), END_TIME: pd.Timestamp('2012-11-24 11:12:38.148989200'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['24.11.2012 14:36:00.000000','24.11.2012 15:59:59.000000','Leaving']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-24 14:36:23.618490'), END_TIME: pd.Timestamp('2012-11-24 15:59:10.786163'), ACTIVITY: 'Leaving'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['24.11.2012 19:58:00.000000','24.11.2012 19:59:59.000000','Snack']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-24 19:58:24.265500'), END_TIME: pd.Timestamp('2012-11-24 19:59:40.359200'), ACTIVITY: 'Snack'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['24.11.2012 23:23:00.000000','25.11.2012 00:44:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-24 23:24:05.531402200'), END_TIME: pd.Timestamp('2012-11-25 00:43:49.231914200'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['24.11.2012 23:21:00.000000','24.11.2012 23:22:59.000000','Snack']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-24 23:21:09.471313700'), END_TIME: pd.Timestamp('2012-11-24 23:23:14.094976100'), ACTIVITY: 'Snack'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['25.11.2012 00:45:00.000000','25.11.2012 00:47:59.000000','Grooming']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-25 00:44:15.919816900'), END_TIME: pd.Timestamp('2012-11-25 00:47:59'), ACTIVITY: 'Grooming'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['25.11.2012 15:47:00.000000','25.11.2012 15:48:59.000000','Snack']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-25 15:47:15.592514800'), END_TIME: pd.Timestamp('2012-11-25 15:48:59'), ACTIVITY: 'Snack'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['25.11.2012 15:58:00.000000','25.11.2012 20:10:59.000000','Leaving']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-25 15:58:46.749143'), END_TIME: pd.Timestamp('2012-11-25 20:10:08.301868400'), ACTIVITY: 'Leaving'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['25.11.2012 22:14:59.000100','25.11.2012 23:20:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-25 22:14:59.000100'), END_TIME: pd.Timestamp('2012-11-25 23:20:07.602724'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['25.11.2012 23:25:34.000000','26.11.2012 01:28:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-25 23:25:34'), END_TIME: pd.Timestamp('2012-11-26 00:21:44.485400'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-26 00:28:30.173000'), END_TIME: pd.Timestamp('2012-11-26 01:28:59'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['26.11.2012 01:32:00.000000','26.11.2012 08:50:59.000000','Sleeping']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-26 01:33:15.720251'), END_TIME: pd.Timestamp('2012-11-26 08:48:13.381602800'), ACTIVITY: 'Sleeping'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['26.11.2012 21:17:00.000000','27.11.2012 01:27:59.000000','Leaving']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-26 21:17:45.581634'), END_TIME: pd.Timestamp('2012-11-27 01:27:59'), ACTIVITY: 'Leaving'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['27.11.2012 11:58:00.000000','27.11.2012 12:23:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-27 11:59:05.855284399'), END_TIME: pd.Timestamp('2012-11-27 12:22:54.838410500'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['27.11.2012 12:26:00.000000','27.11.2012 13:22:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-27 12:27:04.823946'), END_TIME: pd.Timestamp('2012-11-27 13:21:47.313758300'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['27.11.2012 19:11:00.000000','27.11.2012 19:44:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-27 19:11:31.756930200'), END_TIME: pd.Timestamp('2012-11-27 19:44:59'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['27.11.2012 19:46:00.000000','27.11.2012 19:46:59.000000','Snack']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-27 19:46:16.801400'), END_TIME: pd.Timestamp('2012-11-27 19:46:54.203700'), ACTIVITY: 'Snack'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['28.11.2012 12:58:00.000000','28.11.2012 13:14:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-28 12:58:38.093300'), END_TIME: pd.Timestamp('2012-11-28 13:14:06.330200'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['28.11.2012 16:05:00.001000','28.11.2012 17:38:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-28 16:05:35.228419200'), END_TIME: pd.Timestamp('2012-11-28 17:37:37.852558'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['28.11.2012 14:28:00.000000','28.11.2012 16:05:00.000000','Leaving']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-28 14:28:00'), END_TIME: pd.Timestamp('2012-11-28 16:05:16.842199800'), ACTIVITY: 'Leaving'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['29.11.2012 00:59:00.000000','29.11.2012 09:35:59.000000','Sleeping']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-29 01:00:00.336817200'), END_TIME: pd.Timestamp('2012-11-29 09:34:25.149130'), ACTIVITY: 'Sleeping'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['29.11.2012 00:58:00.000000','29.11.2012 00:58:59.000000','Grooming']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-29 00:58:00'), END_TIME: pd.Timestamp('2012-11-29 00:59:13.120973'), ACTIVITY: 'Grooming'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['29.11.2012 00:58:00.000000','29.11.2012 00:59:13.120973','Grooming']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-29 00:58:27.179066'), END_TIME: pd.Timestamp('2012-11-29 00:59:13.120973'), ACTIVITY: 'Grooming'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['29.11.2012 14:27:00.000000','29.11.2012 15:05:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-29 14:27:00'), END_TIME: pd.Timestamp('2012-11-29 15:05:32.600990'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['29.11.2012 15:08:00.000000','29.11.2012 15:32:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-29 15:08:45.445224800'), END_TIME: pd.Timestamp('2012-11-29 15:31:54.833979'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['29.11.2012 16:12:00.000000','29.11.2012 16:12:59.000000','Grooming']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

#----------------------------------------
# 


# Create operation:

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-29 16:12:21.744198740'), END_TIME: pd.Timestamp('2012-11-29 16:12:59.715240899'), ACTIVITY: 'Toileting'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['29.11.2012 16:13:00.000000','29.11.2012 16:17:59.000000','Toileting']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['30.11.2012 12:56:00.000000','30.11.2012 14:35:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-30 12:56:00'), END_TIME: pd.Timestamp('2012-11-30 14:34:17.961068300'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['01.12.2012 11:29:00.000000','01.12.2012 11:36:59.000000','Snack']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-12-01 11:29:27.869300'), END_TIME: pd.Timestamp('2012-12-01 11:36:00.109400'), ACTIVITY: 'Snack'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['02.12.2012 09:26:22.000000','02.12.2012 10:16:59.000000','Sleeping']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-12-02 09:26:22'), END_TIME: pd.Timestamp('2012-12-02 10:15:43.384401300'), ACTIVITY: 'Sleeping'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['29.11.2012 09:50:00.000000','29.11.2012 09:56:59.000000','Toileting']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-29 09:49:34.991572800'), END_TIME: pd.Timestamp('2012-11-29 09:52:48.626185400'), ACTIVITY: 'Toileting'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['12.11.2012 21:06:00.000000','12.11.2012 21:11:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['17.11.2012 10:21:00.000000','17.11.2012 10:34:59.000000','Grooming']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-17 10:21:18.254916'), END_TIME: pd.Timestamp('2012-11-17 10:34:59'), ACTIVITY: 'Grooming'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['17.11.2012 10:18:00.000000','17.11.2012 10:20:59.000000','Showering']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-17 10:18:00'), END_TIME: pd.Timestamp('2012-11-17 10:21:13.611138'), ACTIVITY: 'Showering'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['30.11.2012 01:40:00.000000','30.11.2012 01:40:59.000000','Toileting']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-30 01:40:00'), END_TIME: pd.Timestamp('2012-11-30 01:40:28.778495'), ACTIVITY: 'Toileting'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['30.11.2012 01:41:00.000000','30.11.2012 10:20:59.000000','Sleeping']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-30 01:40:43.283263'), END_TIME: pd.Timestamp('2012-11-30 10:20:59'), ACTIVITY: 'Sleeping'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['02.12.2012 09:25:00.000000','02.12.2012 09:25:59.000000','Toileting']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-12-02 09:25:00'), END_TIME: pd.Timestamp('2012-12-02 09:25:40.616523900'), ACTIVITY: 'Toileting'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['23.11.2012 10:56:00.000000','23.11.2012 12:35:59.000000','Sleeping']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

#----------------------------------------
# 


# Create operation:

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-23 10:56:04.785100'), END_TIME: pd.Timestamp('2012-11-23 12:36:07.231600'), ACTIVITY: 'Leaving'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['25.11.2012 13:41:00.000000','25.11.2012 13:45:59.000000','Toileting']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-25 13:41:00'), END_TIME: pd.Timestamp('2012-11-25 13:42:19.070850'), ACTIVITY: 'Toileting'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['30.11.2012 12:26:00.000000','30.11.2012 12:30:59.000000','Grooming']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-30 12:26:37.573425'), END_TIME: pd.Timestamp('2012-11-30 12:30:59'), ACTIVITY: 'Grooming'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['17.11.2012 14:42:00.000000','17.11.2012 16:24:59.000000','Leaving']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-17 14:42:00'), END_TIME: pd.Timestamp('2012-11-17 16:24:31.504720'), ACTIVITY: 'Leaving'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['26.11.2012 11:34:00.000000','26.11.2012 11:59:59.000000','Leaving']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-26 11:34:00'), END_TIME: pd.Timestamp('2012-11-26 11:59:45.948074'), ACTIVITY: 'Leaving'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['15.11.2012 11:42:00.000000','15.11.2012 13:02:59.000000','Leaving']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-15 11:42:00'), END_TIME: pd.Timestamp('2012-11-15 13:02:49.973523'), ACTIVITY: 'Leaving'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['15.11.2012 15:54:00.000000','15.11.2012 20:10:59.000000','Leaving']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-15 15:54:16.229537400'), END_TIME: pd.Timestamp('2012-11-15 20:10:59'), ACTIVITY: 'Leaving'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['16.11.2012 22:18:00.000000','17.11.2012 02:19:59.000000','Leaving']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-16 22:18:31.377550'), END_TIME: pd.Timestamp('2012-11-17 02:19:59'), ACTIVITY: 'Leaving'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['18.11.2012 10:53:00.000000','18.11.2012 14:34:59.000000','Leaving']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-18 10:53:00'), END_TIME: pd.Timestamp('2012-11-18 14:34:13.654231600'), ACTIVITY: 'Leaving'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['27.11.2012 21:12:00.000000','28.11.2012 01:37:59.000000','Leaving']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-27 21:12:00'), END_TIME: pd.Timestamp('2012-11-28 01:37:38.947296600'), ACTIVITY: 'Leaving'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['27.11.2012 11:53:00.000000','27.11.2012 11:53:59.000000','Grooming']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-27 11:53:00'), END_TIME: pd.Timestamp('2012-11-27 11:53:50.765519600'), ACTIVITY: 'Grooming'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['24.11.2012 00:33:00.000000','24.11.2012 10:02:59.000000','Sleeping']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-24 00:33:00'), END_TIME: pd.Timestamp('2012-11-24 10:02:29.793869300'), ACTIVITY: 'Sleeping'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['02.12.2012 01:41:00.000000','02.12.2012 09:20:59.000000','Sleeping']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-12-02 01:41:06.055957200'), END_TIME: pd.Timestamp('2012-12-02 09:20:59'), ACTIVITY: 'Sleeping'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['23.11.2012 14:40:00.000000','23.11.2012 14:41:59.000000','Toileting']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-11-23 14:40:00'), END_TIME: pd.Timestamp('2012-11-23 14:40:43.002957200'), ACTIVITY: 'Toileting'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['02.12.2012 19:10:00.000000','02.12.2012 19:11:59.000000','Toileting']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2012-12-02 19:10:00'), END_TIME: pd.Timestamp('2012-12-02 19:11:24.928314'), ACTIVITY: 'Toileting'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)
