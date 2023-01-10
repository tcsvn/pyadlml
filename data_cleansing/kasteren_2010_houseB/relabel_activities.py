# Awesome script
import pandas as pd
from pyadlml.dataset._core.activities import get_index_matching_rows


#----------------------------------------# Toilet activity -> therefore cut activity and add "Use toilet" act below


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['2009-07-24 20:21:50.642995','2009-07-24 20:39:39.524997','Prepare dinner']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-24 20:25:04.151135'), 'end_time': pd.Timestamp('2009-07-24 20:39:39.524997'), 'activity': 'Prepare dinner'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------#  add "Use toilet" since toilet door and flush are activated


# Create operation:

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-24 20:22:01.741500'), 'end_time': pd.Timestamp('2009-07-24 20:24:58.162608890'), 'activity': 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# coming home earlier


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['2009-07-24 21:25:20.754998','2009-07-25 02:46:34.712998','Leaving the house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-24 21:25:20.754998'), 'end_time': pd.Timestamp('2009-07-25 02:44:26.991100'), 'activity': 'Leaving the house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Create operation:

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-25 02:45:43.079600'), 'end_time': pd.Timestamp('2009-07-25 02:46:31.681400'), 'activity': 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-07-25 02:54:18.841993','2009-07-25 06:54:01.584992','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-25 02:54:31.722762700'), 'end_time': pd.Timestamp('2009-07-25 06:54:01.584992'), 'activity': 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# Adjusting to later onset of activity


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-07-25 07:04:19.742000','2009-07-25 11:08:25.158991','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-25 07:04:37.957187200'), 'end_time': pd.Timestamp('2009-07-25 11:08:25.158991'), 'activity': 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-07-25 15:12:05.317993','2009-07-26 03:33:10.088994','Leaving the house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-25 15:12:05.317993'), 'end_time': pd.Timestamp('2009-07-26 03:31:07.941174800'), 'activity': 'Leaving the house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Create operation:

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-26 03:31:55.591900'), 'end_time': pd.Timestamp('2009-07-26 03:32:57.775900'), 'activity': 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# Adjusting to later onset of activity


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-07-26 03:33:25.654993','2009-07-26 09:32:18.156999','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-26 03:35:26.698188800'), 'end_time': pd.Timestamp('2009-07-26 09:32:18.156999'), 'activity': 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# no toileting sensor behaviour


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['2009-07-26 15:41:14.468996','2009-07-26 15:41:18.848997','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

#----------------------------------------# Adjusting to later onset of activity


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-07-26 15:41:34.893991','2009-07-26 15:49:58.455995','Take shower']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-26 15:41:56.228998'), 'end_time': pd.Timestamp('2009-07-26 15:49:58.455995'), 'activity': 'Take shower'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# Since  toilet activity was near this has to be the frame where it happened


# Create operation:

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-26 15:41:41.143700'), 'end_time': pd.Timestamp('2009-07-26 15:41:55.818700'), 'activity': 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# Adjusting to later ending of activity


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-07-26 15:54:27.212001','2009-07-26 15:56:07.771001','Brush teeth']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-26 15:54:27.212001'), 'end_time': pd.Timestamp('2009-07-26 15:56:23.389565400'), 'activity': 'Brush teeth'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# Adjusting to later onset of activity


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-07-26 16:19:19.705998','2009-07-27 04:58:12.477000','Leaving the house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-26 16:20:07.791482500'), 'end_time': pd.Timestamp('2009-07-27 04:58:12.477000'), 'activity': 'Leaving the house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# Adjusting to later onset of activity


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-07-27 05:03:24.415998','2009-07-27 12:15:38.414996','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-27 05:10:36.278768'), 'end_time': pd.Timestamp('2009-07-27 12:15:38.414996'), 'activity': 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# Adjusting to earlier onset of activity


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-07-27 12:17:38.284000','2009-07-27 12:17:42.087993','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-27 12:16:28.286083200'), 'end_time': pd.Timestamp('2009-07-27 12:17:42.087993'), 'activity': 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# Adjusting to later onset of activity (toileting is coming before)


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-07-27 13:18:13.256992','2009-07-27 23:04:32.660997','Leaving the house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-27 13:22:33.738936400'), 'end_time': pd.Timestamp('2009-07-27 23:04:32.660997'), 'activity': 'Leaving the house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# activity is not happening


# Delete operation:
idx_to_del = get_index_matching_rows(df_acts, 	[['2009-07-27 04:58:44.282993','2009-07-27 05:02:46.282998','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

#----------------------------------------# shifting whole toilet activity to match behaviour


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-07-28 09:56:17.264995','2009-07-28 09:57:11.997991','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-28 09:56:27.669044'), 'end_time': pd.Timestamp('2009-07-28 09:57:14.046818'), 'activity': 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# Adjusting to earlier ending of activity


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-07-28 19:21:28.733994','2009-07-28 20:07:52.030998','Play piano']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-28 19:21:28.733994'), 'end_time': pd.Timestamp('2009-07-28 19:51:15.423271600'), 'activity': 'Play piano'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# activity happens


# Create operation:

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-28 19:51:22.395700'), 'end_time': pd.Timestamp('2009-07-28 19:52:27.344800'), 'activity': 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Create operation:

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-28 22:33:16.130800'), 'end_time': pd.Timestamp('2009-07-28 22:34:45.791000'), 'activity': 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# later onset + toileting during night => earlier ending


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-07-28 22:37:05.866992','2009-07-29 09:48:20.144992','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-28 22:37:57.560242700'), 'end_time': pd.Timestamp('2009-07-29 04:39:20.699954'), 'activity': 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# activity not labeled


# Create operation:

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-29 04:39:29.801900'), 'end_time': pd.Timestamp('2009-07-29 04:40:19.373000'), 'activity': 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# resume sleeping after toileting


# Create operation:

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-29 04:40:38.428188800'), 'end_time': pd.Timestamp('2009-07-29 05:00:21.158770200'), 'activity': 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# adding "missing" activity (maybe sbd. is not sleeping)


# Create operation:

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-29 05:51:23.480262500'), 'end_time': pd.Timestamp('2009-07-29 09:48:27.246120100'), 'activity': 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# adding missing activity


# Create operation:

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-29 09:48:38.824300'), 'end_time': pd.Timestamp('2009-07-29 09:49:30.007100'), 'activity': 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# shifting activity


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-07-29 14:08:28.570993','2009-07-29 18:08:13.497997','Leaving the house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-29 14:08:42.714761'), 'end_time': pd.Timestamp('2009-07-29 18:08:19.720921700'), 'activity': 'Leaving the house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-07-30 03:09:00.711994','2009-07-30 11:18:18.325991','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-30 03:09:24.597345'), 'end_time': pd.Timestamp('2009-07-30 11:18:25.873257400'), 'activity': 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# shift


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-07-30 11:18:27.665993','2009-07-30 11:19:12.543993','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-30 11:18:41.372412'), 'end_time': pd.Timestamp('2009-07-30 11:19:16.842184800'), 'activity': 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-07-30 13:11:43.527995','2009-07-30 17:10:20.680996','Leaving the house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-30 13:12:06.772195'), 'end_time': pd.Timestamp('2009-07-30 17:10:28.858121600'), 'activity': 'Leaving the house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-07-31 01:33:11.454001','2009-07-31 09:45:20.159999','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-31 01:26:01.247920500'), 'end_time': pd.Timestamp('2009-07-31 09:45:23.031870200'), 'activity': 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-07-31 11:09:41.497000','2009-07-31 11:12:05.657001','Prepare for leaving']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-31 11:09:48.687930'), 'end_time': pd.Timestamp('2009-07-31 11:12:05.657001'), 'activity': 'Prepare for leaving'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-07-31 11:07:36.933001','2009-07-31 11:09:38.597992','Get dressed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-31 11:07:36.933001'), 'end_time': pd.Timestamp('2009-07-31 11:09:46.971253'), 'activity': 'Get dressed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-07-31 22:34:00.472001','2009-07-31 22:34:47.464997','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-31 22:33:46.302488500'), 'end_time': pd.Timestamp('2009-07-31 22:34:36.349219100'), 'activity': 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-07-31 22:35:15.441993','2009-08-01 05:19:06.419997','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-07-31 22:35:39.755030'), 'end_time': pd.Timestamp('2009-08-01 05:19:03.260586499'), 'activity': 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-08-01 05:19:06.420997','2009-08-01 05:20:08.461000','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-08-01 05:19:13.755600'), 'end_time': pd.Timestamp('2009-08-01 05:19:59.497300'), 'activity': 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-08-01 11:14:00.419997','2009-08-01 11:17:18.460997','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-08-01 11:14:16.067200'), 'end_time': pd.Timestamp('2009-08-01 11:17:25.582700'), 'activity': 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-08-01 12:58:27.527998','2009-08-01 13:01:24.505000','Prepare for leaving']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-08-01 12:58:36.003562800'), 'end_time': pd.Timestamp('2009-08-01 13:01:24.505000'), 'activity': 'Prepare for leaving'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-08-01 12:56:49.750998','2009-08-01 12:58:23.599996','Get dressed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-08-01 12:57:05.819000'), 'end_time': pd.Timestamp('2009-08-01 12:58:33.148200'), 'activity': 'Get dressed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-08-02 00:08:49.032999','2009-08-02 00:09:55.286994','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-08-02 00:09:07.413100'), 'end_time': pd.Timestamp('2009-08-02 00:09:49.589500'), 'activity': 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-08-02 00:12:33.606997','2009-08-02 09:26:32.485991','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-08-02 00:13:19.689009900'), 'end_time': pd.Timestamp('2009-08-02 04:44:30.415727600'), 'activity': 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Create operation:

new_row=pd.Series({'start_time': pd.Timestamp('2009-08-02 04:44:44.483600'), 'end_time': pd.Timestamp('2009-08-02 04:46:50.554800'), 'activity': 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Create operation:

new_row=pd.Series({'start_time': pd.Timestamp('2009-08-02 04:47:45.991275'), 'end_time': pd.Timestamp('2009-08-02 09:26:41.778861'), 'activity': 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-08-02 09:27:11.227000','2009-08-02 09:28:05.184995','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-08-02 09:27:30.177400'), 'end_time': pd.Timestamp('2009-08-02 09:28:09.829100'), 'activity': 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-08-02 09:45:03.889991','2009-08-02 09:45:33.838994','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-08-02 09:45:17.498100'), 'end_time': pd.Timestamp('2009-08-02 09:45:33.813000'), 'activity': 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-08-02 14:41:06.944991','2009-08-02 14:43:02.022993','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-08-02 14:41:24.721600'), 'end_time': pd.Timestamp('2009-08-02 14:43:11.753200'), 'activity': 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-08-02 14:46:15.927996','2009-08-03 04:05:01.692999','Leaving the house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-08-02 14:46:30.649641600'), 'end_time': pd.Timestamp('2009-08-03 04:05:05.455439999'), 'activity': 'Leaving the house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Create operation:

new_row=pd.Series({'start_time': pd.Timestamp('2009-08-03 04:05:40.092200'), 'end_time': pd.Timestamp('2009-08-03 04:06:29.860600'), 'activity': 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-08-03 05:47:46.123997','2009-08-03 15:11:53.866995','Leaving the house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-08-03 05:47:22.623200'), 'end_time': pd.Timestamp('2009-08-03 15:10:23.976425200'), 'activity': 'Leaving the house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-08-03 15:13:29.736998','2009-08-03 15:14:44.225992','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-08-03 15:13:53.502900'), 'end_time': pd.Timestamp('2009-08-03 15:15:13.017500'), 'activity': 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-08-03 15:41:39.666666','2009-08-04 00:09:59.737996','Leaving the house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-08-03 15:41:53.286875'), 'end_time': pd.Timestamp('2009-08-04 00:10:01.822829600'), 'activity': 'Leaving the house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-08-04 00:11:06.200993','2009-08-04 05:53:01.324993','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-08-04 00:12:33.475916'), 'end_time': pd.Timestamp('2009-08-04 05:52:47.583596'), 'activity': 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-08-04 05:53:01.325993','2009-08-04 05:54:47.140995','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-08-04 05:53:37.734600'), 'end_time': pd.Timestamp('2009-08-04 05:54:22.704300'), 'activity': 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-08-04 05:54:47.141995','2009-08-04 12:40:30.691000','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-08-04 05:55:08.960976'), 'end_time': pd.Timestamp('2009-08-04 12:40:37.748990500'), 'activity': 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-08-04 12:40:38.324996','2009-08-04 12:42:08.140995','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-08-04 12:40:58.588800'), 'end_time': pd.Timestamp('2009-08-04 12:42:19.272300'), 'activity': 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-08-04 13:47:53.097992','2009-08-04 13:48:06.265995','Use toilet']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-08-04 13:45:02.761400'), 'end_time': pd.Timestamp('2009-08-04 13:48:08.045400'), 'activity': 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-08-04 14:21:52.132999','2009-08-04 14:23:30.752993','Get dressed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-08-04 14:22:05.323400'), 'end_time': pd.Timestamp('2009-08-04 14:23:43.489000'), 'activity': 'Get dressed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-08-04 14:27:38.505994','2009-08-04 16:26:48.405998','Leaving the house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-08-04 14:27:57.354011200'), 'end_time': pd.Timestamp('2009-08-04 16:27:02.865838'), 'activity': 'Leaving the house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-08-04 16:40:03.887994','2009-08-04 22:54:17.006994','Leaving the house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-08-04 16:40:27.604521899'), 'end_time': pd.Timestamp('2009-08-04 22:54:20.730383600'), 'activity': 'Leaving the house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-08-04 23:49:48.948999','2009-08-05 04:25:53.783995','Go to bed']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-08-04 23:51:22.710462800'), 'end_time': pd.Timestamp('2009-08-05 04:08:36.665509800'), 'activity': 'Go to bed'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Create operation:

new_row=pd.Series({'start_time': pd.Timestamp('2009-08-05 04:08:53.765800'), 'end_time': pd.Timestamp('2009-08-05 04:09:58.138500'), 'activity': 'Use toilet'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)

#----------------------------------------# 


# Modify operation:

idx_to_del = get_index_matching_rows(df_acts, 	[['2009-08-05 04:27:38.585991','2009-08-05 15:17:30.755998','Leaving the house']],
)
df_acts = df_acts.drop(index=idx_to_del).reset_index(drop=True)

new_row=pd.Series({'start_time': pd.Timestamp('2009-08-05 04:28:04.839904700'), 'end_time': pd.Timestamp('2009-08-05 15:17:30.755998'), 'activity': 'Leaving the house'})
df_acts = pd.concat([df_acts, new_row.to_frame().T], axis=0).reset_index(drop=True)
