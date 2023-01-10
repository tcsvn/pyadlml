# Awesome script
import pandas as pd
from pyadlml.constants import START_TIME, END_TIME, ACTIVITY
from pyadlml.dataset._core.activities import get_index_matching_rows


#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['28.11.2011 10:25:44.000000','28.11.2011 10:33:00.000000','Showering']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2011-11-28 10:25:44'), END_TIME: pd.Timestamp('2011-11-28 10:32:12.877373400'), ACTIVITY: 'Showering'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['29.11.2011 11:31:55.000000','29.11.2011 11:36:55.000000','Toileting']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2011-11-29 11:31:55'), END_TIME: pd.Timestamp('2011-11-29 11:37:32.756100'), ACTIVITY: 'Toileting'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['29.11.2011 15:13:28.000000','29.11.2011 15:13:57.000000','Toileting']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2011-11-29 15:13:09.091021700'), END_TIME: pd.Timestamp('2011-11-29 15:13:57'), ACTIVITY: 'Toileting'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['30.11.2011 10:11:07.000000','30.11.2011 10:13:59.000000','Toileting']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2011-11-30 10:11:07'), END_TIME: pd.Timestamp('2011-11-30 10:14:37.392878400'), ACTIVITY: 'Toileting'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['30.11.2011 14:11:16.000000','30.11.2011 14:11:48.000000','Toileting']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2011-11-30 14:10:23.624283999'), END_TIME: pd.Timestamp('2011-11-30 14:11:48'), ACTIVITY: 'Toileting'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['30.11.2011 18:57:08.000000','30.11.2011 18:57:34.000000','Toileting']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2011-11-30 18:56:46.977020999'), END_TIME: pd.Timestamp('2011-11-30 18:57:34'), ACTIVITY: 'Toileting'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['30.11.2011 19:37:36.000000','30.11.2011 19:38:01.000000','Toileting']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2011-11-30 19:37:20.371340'), END_TIME: pd.Timestamp('2011-11-30 19:38:01'), ACTIVITY: 'Toileting'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['01.12.2011 16:26:35.000000','01.12.2011 16:27:03.000000','Toileting']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2011-12-01 16:26:24.301848500'), END_TIME: pd.Timestamp('2011-12-01 16:27:03'), ACTIVITY: 'Toileting'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['01.12.2011 19:28:51.000000','01.12.2011 19:29:59.000000','Toileting']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2011-12-01 19:27:52.079036800'), END_TIME: pd.Timestamp('2011-12-01 19:29:59'), ACTIVITY: 'Toileting'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['05.12.2011 11:53:01.000000','05.12.2011 11:53:31.000000','Toileting']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2011-12-05 11:52:33.276102400'), END_TIME: pd.Timestamp('2011-12-05 11:53:31'), ACTIVITY: 'Toileting'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['09.12.2011 10:51:56.000000','09.12.2011 10:52:27.000000','Toileting']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2011-12-09 10:51:39.014276700'), END_TIME: pd.Timestamp('2011-12-09 10:52:27'), ACTIVITY: 'Toileting'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['11.12.2011 15:28:59.000000','11.12.2011 15:30:14.000000','Toileting']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2011-12-11 15:28:36.647065200'), END_TIME: pd.Timestamp('2011-12-11 15:30:14'), ACTIVITY: 'Toileting'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)


#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['29.11.2011 13:25:29.000000','29.11.2011 13:25:32.000000','Snack']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2011-11-29 13:25:29'), END_TIME: pd.Timestamp('2011-11-29 13:25:36.493850'), ACTIVITY: 'Snack'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['30.11.2011 18:01:44.000000','30.11.2011 18:01:47.000000','Toileting']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2011-11-30 18:01:39.036641'), END_TIME: pd.Timestamp('2011-11-30 18:01:46.082940400'), ACTIVITY: 'Toileting'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['05.12.2011 21:06:35.000000','05.12.2011 21:06:43.000000','Toileting']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2011-12-05 21:06:30.576174100'), END_TIME: pd.Timestamp('2011-12-05 21:06:43'), ACTIVITY: 'Toileting'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------
# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['29.11.2011 12:22:38.000000','29.11.2011 12:24:59.000000','Spare_Time/TV']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({START_TIME: pd.Timestamp('2011-11-29 12:22:38'), END_TIME: pd.Timestamp('2011-11-29 13:25:15.371700'), ACTIVITY: 'Spare_Time/TV'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)
