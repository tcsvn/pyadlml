# Awesome script 
import pandas as pd
from pyadlml.constants import START_TIME, END_TIME, ACTIVITY
from pyadlml.dataset._core.activities import get_index_matching_rows

# TODO add original activity dataframe in line below
idxs_to_delete = []

# Content
#----------------------------------------------------------------------------------------------------

# id=0
# 

lst_to_add_0 = [
	['14.11.2012 00:29:59.999999','14.11.2012 05:12:59.000000','Sleeping'],
]

lst_to_del = [
	['14.11.2012 00:29:00.000000','14.11.2012 05:12:59.000000','Sleeping'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=1
# 

lst_to_add_1 = [
	['14.11.2012 12:29:00.000000','14.11.2012 12:52:00.000000','Spare_Time/TV'],
	['14.11.2012 12:52:00.000100','14.11.2012 12:54:59.000000','Leaving'],
]

lst_to_del = [
	['14.11.2012 12:29:00.000000','14.11.2012 12:52:59.000000','Spare_Time/TV'],
	['14.11.2012 12:52:00.000000','14.11.2012 12:54:59.000000','Leaving'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=2
# 

lst_to_add_2 = [
	['14.11.2012 21:37:59.001000','14.11.2012 21:47:59.000000','Dinner'],
]

lst_to_del = [
	['14.11.2012 21:37:00.000000','14.11.2012 21:47:59.000000','Dinner'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=3
# 

lst_to_add_3 = [
	['14.11.2012 22:14:59.001000','14.11.2012 22:17:59.000000','Snack'],
]

lst_to_del = [
	['14.11.2012 22:14:00.000000','14.11.2012 22:17:59.000000','Snack'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=4
# 

lst_to_add_4 = [
	['15.11.2012 00:10:59.001000','15.11.2012 00:39:59.000000','Sleeping'],
]

lst_to_del = [
	['15.11.2012 00:10:00.000000','15.11.2012 00:39:59.000000','Sleeping'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=5
# 

lst_to_add_5 = [
	['15.11.2012 00:43:59.000100','15.11.2012 01:54:59.000000','Spare_Time/TV'],
]

lst_to_del = [
	['15.11.2012 00:43:00.000000','15.11.2012 01:54:59.000000','Spare_Time/TV'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=6
# 

lst_to_add_6 = [
	['17.11.2012 17:39:00.000000','17.11.2012 20:26:00.000000','Spare_Time/TV'],
	['17.11.2012 20:26:00.001000','17.11.2012 20:31:59.000000','Snack'],
]

lst_to_del = [
	['17.11.2012 17:39:00.000000','17.11.2012 20:26:59.000000','Spare_Time/TV'],
	['17.11.2012 20:26:00.000000','17.11.2012 20:31:59.000000','Snack'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=7
# 

lst_to_add_7 = [
	['19.11.2012 19:56:00.000000','19.11.2012 21:59:00.000000','Spare_Time/TV'],
	['19.11.2012 21:59:00.001000','19.11.2012 22:04:59.000000','Dinner'],
]

lst_to_del = [
	['19.11.2012 19:56:00.000000','19.11.2012 21:59:59.000000','Spare_Time/TV'],
	['19.11.2012 21:59:00.000000','19.11.2012 22:04:59.000000','Dinner'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=8
# 

lst_to_add_8 = [
	['19.11.2012 22:32:59.000100','20.11.2012 01:22:59.000000','Spare_Time/TV'],
]

lst_to_del = [
	['19.11.2012 22:32:00.000000','20.11.2012 01:22:59.000000','Spare_Time/TV'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=9
# 

lst_to_add_9 = [
	['21.11.2012 21:18:00.000000','22.11.2012 01:40:00.000000','Leaving'],
	['22.11.2012 01:40:00.001000','22.11.2012 01:42:59.000000','Snack'],
]

lst_to_del = [
	['21.11.2012 21:18:00.000000','22.11.2012 01:40:59.000000','Leaving'],
	['22.11.2012 01:40:00.000000','22.11.2012 01:42:59.000000','Snack'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=10
# 

lst_to_add_10 = [
	['22.11.2012 11:11:00.000000','22.11.2012 11:55:59.000000','Spare_Time/TV'],
]

lst_to_del = [
	['22.11.2012 11:11:00.000000','22.11.2012 11:56:59.000000','Spare_Time/TV'],
	['22.11.2012 11:12:00.000000','22.11.2012 11:55:59.000000','Spare_Time/TV'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=11
# 

lst_to_add_11 = [
	['22.11.2012 15:27:59.000100','22.11.2012 16:42:59.000000','Spare_Time/TV'],
]

lst_to_del = [
	['22.11.2012 15:27:00.000000','22.11.2012 16:42:59.000000','Spare_Time/TV'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=12
# 

lst_to_add_12 = [
	['22.11.2012 16:53:00.000000','22.11.2012 18:15:00.000000','Spare_Time/TV'],
	['22.11.2012 18:15:00.001000','22.11.2012 18:18:59.000000','Snack'],
]

lst_to_del = [
	['22.11.2012 16:53:00.000000','22.11.2012 18:15:59.000000','Spare_Time/TV'],
	['22.11.2012 18:15:00.000000','22.11.2012 18:18:59.000000','Snack'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=13
# 

lst_to_add_13 = [
	['23.11.2012 13:05:00.000000','23.11.2012 14:23:00.000000','Spare_Time/TV'],
	['23.11.2012 14:23:00.001000','23.11.2012 14:38:59.000000','Lunch'],
]

lst_to_del = [
	['23.11.2012 13:05:00.000000','23.11.2012 14:23:59.000000','Spare_Time/TV'],
	['23.11.2012 14:23:00.000000','23.11.2012 14:38:59.000000','Lunch'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=14
# 

lst_to_add_14 = [
	['23.11.2012 22:17:00.000000','24.11.2012 00:27:00.000000','Spare_Time/TV'],
	['24.11.2012 00:27:00.001000','24.11.2012 00:32:59.000000','Grooming'],
]

lst_to_del = [
	['23.11.2012 22:17:00.000000','24.11.2012 00:27:59.000000','Spare_Time/TV'],
	['24.11.2012 00:27:00.000000','24.11.2012 00:32:59.000000','Grooming'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=15
# 

lst_to_add_15 = [
	['25.11.2012 22:14:59.000100','25.11.2012 23:20:59.000000','Spare_Time/TV'],
]

lst_to_del = [
	['25.11.2012 22:14:00.000000','25.11.2012 23:20:59.000000','Spare_Time/TV'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=16
# 

lst_to_add_16 = [
	['27.11.2012 01:43:59.001000','27.11.2012 10:12:59.000000','Sleeping'],
]

lst_to_del = [
	['27.11.2012 01:43:00.000000','27.11.2012 10:12:59.000000','Sleeping'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=17
# 

lst_to_add_17 = [
	['28.11.2012 14:28:00.000000','28.11.2012 16:05:00.000000','Leaving'],
	['28.11.2012 16:05:00.001000','28.11.2012 17:38:59.000000','Spare_Time/TV'],
]

lst_to_del = [
	['28.11.2012 14:28:00.000000','28.11.2012 16:05:59.000000','Leaving'],
	['28.11.2012 16:05:00.000000','28.11.2012 17:38:59.000000','Spare_Time/TV'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=18
# 

lst_to_add_18 = [
	['29.11.2012 00:00:59.001000','29.11.2012 00:57:59.000000','Sleeping'],
]

lst_to_del = [
	['29.11.2012 00:00:00.000000','29.11.2012 00:57:59.000000','Sleeping'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)

# id=19
# 

lst_to_add_19 = [
	['30.11.2012 10:25:00.000000','30.11.2012 10:27:00.000000','Toileting'],
	['30.11.2012 10:27:00.001000','30.11.2012 10:28:59.000000','Grooming'],
	['30.11.2012 10:28:59.001000','30.11.2012 10:31:59.000000','Toileting'],
]

lst_to_del = [
	['30.11.2012 10:25:00.000000','30.11.2012 10:31:59.000000','Toileting'],
	['30.11.2012 10:27:00.000000','30.11.2012 10:28:59.000000','Grooming'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=20
# 

lst_to_add_20 = [
	['30.11.2012 15:20:59.000100','30.11.2012 17:29:59.000000','Spare_Time/TV'],
]

lst_to_del = [
	['30.11.2012 15:20:00.000000','30.11.2012 17:29:59.000000','Spare_Time/TV'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=21
# 

lst_to_add_21 = [
	['30.11.2012 17:51:00.000000','30.11.2012 22:12:00.000000','Spare_Time/TV'],
	['30.11.2012 22:12:00.001000','30.11.2012 22:20:59.000000','Dinner'],
]

lst_to_del = [
	['30.11.2012 17:51:00.000000','30.11.2012 22:12:59.000000','Spare_Time/TV'],
	['30.11.2012 22:12:00.000000','30.11.2012 22:20:59.000000','Dinner'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)



# id=22
# 

lst_to_add_22 = [
	['02.12.2012 11:34:00.000000','02.12.2012 11:35:30.000000','Toileting'],
	['02.12.2012 11:35:30.001000','02.12.2012 11:49:59.000000','Grooming'],
]

lst_to_del = [
	['02.12.2012 11:34:00.000000','02.12.2012 11:35:59.000000','Toileting'],
	['02.12.2012 11:35:00.000000','02.12.2012 11:49:59.000000','Grooming'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)

#----------------------------------------------------------------------------------------------------
new_rows_to_add = [*lst_to_add_0, *lst_to_add_1, *lst_to_add_2, *lst_to_add_3, *lst_to_add_4, *lst_to_add_5, *lst_to_add_6, *lst_to_add_7, *lst_to_add_8, *lst_to_add_9, *lst_to_add_10, *lst_to_add_11, *lst_to_add_12, *lst_to_add_13, *lst_to_add_14, *lst_to_add_15, *lst_to_add_16, *lst_to_add_17, *lst_to_add_18, *lst_to_add_19, *lst_to_add_20, *lst_to_add_21, *lst_to_add_22, ]
df_acts = df_acts.drop(idxs_to_delete)
df_news = pd.DataFrame(data=new_rows_to_add, columns=[START_TIME, END_TIME, ACTIVITY])
df_news[START_TIME] = pd.to_datetime(df_news[START_TIME], dayfirst=True)
df_news[END_TIME] = pd.to_datetime(df_news[END_TIME], dayfirst=True)
df_acts = pd.concat([df_acts, df_news], axis=0)
