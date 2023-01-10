# Awesome script 
import pandas as pd
from pyadlml.dataset._core.activities import get_index_matching_rows

# TODO add original activity dataframe in line below
#df_acts = ... 
idxs_to_delete = []

# Content
#----------------------------------------------------------------------------------------------------

# id=0
# Bigger activity envelopes smaller

lst_to_add_0 = [
	['27.07.2009 04:58:44.282993','27.07.2009 05:02:46.282998','Go to bed'],
	['27.07.2009 05:02:46.283998','27.07.2009 05:03:24.414998','Use toilet'],
	['27.07.2009 05:03:24.415998','27.07.2009 12:15:38.414996','Go to bed'],
]

lst_to_del = [
	['27.07.2009 04:58:44.282993','27.07.2009 12:15:38.414996','Go to bed'],
	['27.07.2009 05:02:46.282998','27.07.2009 05:03:24.414998','Use toilet'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=1
# Smaller activity trumps bigger

lst_to_add_1 = [
	['27.07.2009 23:25:08.868997','27.07.2009 23:25:50.553993','Use toilet'],
	['27.07.2009 23:25:50.641995','28.07.2009 09:34:42.494991','Go to bed'],
]

lst_to_del = [
	['27.07.2009 23:25:08.868997','27.07.2009 23:25:50.551993','Use toilet'],
	['27.07.2009 23:25:10.640995','28.07.2009 09:34:42.494991','Go to bed'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=2
# Bigger activity envelopes smaller

lst_to_add_2 = [
	['28.07.2009 10:12:57.675994','28.07.2009 10:18:16.603996','Prepare brunch'],
	['28.07.2009 10:18:16.604996','28.07.2009 10:20:17.115000','Play piano'],
	['28.07.2009 10:20:17.116000','28.07.2009 10:23:51.618997','Prepare brunch'],
]

lst_to_del = [
	['28.07.2009 10:12:57.675994','28.07.2009 10:23:51.618997','Prepare brunch'],
	['28.07.2009 10:18:16.603996','28.07.2009 10:20:17.115000','Play piano'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=3
# Bigger activity envelopes smaller

lst_to_add_3 = [
	['31.07.2009 11:09:41.497000','31.07.2009 11:12:05.657001','Prepare for leaving'],
	['31.07.2009 11:12:05.658001','31.07.2009 11:13:33.633993','Use toilet'],
	['31.07.2009 11:13:33.634993','31.07.2009 11:16:23.422993','Prepare for leaving'],
]

lst_to_del = [
	['31.07.2009 11:09:41.497000','31.07.2009 11:16:23.422993','Prepare for leaving'],
	['31.07.2009 11:12:05.657001','31.07.2009 11:13:33.633993','Use toilet'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=4
# Bigger activity envelopes smaller

lst_to_add_4 = [
	['31.07.2009 22:35:15.441993','01.08.2009 05:19:06.419997','Go to bed'],
	['01.08.2009 05:19:06.420997','01.08.2009 05:20:08.461000','Use toilet'],
	['01.08.2009 05:20:08.462000','01.08.2009 10:45:20.829998','Go to bed'],
]

lst_to_del = [
	['31.07.2009 22:35:15.441993','01.08.2009 10:45:20.829998','Go to bed'],
	['01.08.2009 05:19:06.419997','01.08.2009 05:20:08.461000','Use toilet'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)



# id=5
# Smaller activity trumps bigger

lst_to_add_5 = [
	['04.08.2009 15:41:39.666666','05.08.2009 00:09:59.737996','Leaving the house'],
]

lst_to_del = [
	['04.08.2009 15:41:37.011994','05.08.2009 00:09:59.737996','Leaving the house'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=6
# Bigger activity envelopes smaller

lst_to_add_6 = [
	['05.08.2009 00:11:06.200993','05.08.2009 05:53:01.324993','Go to bed'],
	['05.08.2009 05:53:01.325993','05.08.2009 05:54:47.140995','Use toilet'],
	['05.08.2009 05:54:47.141995','05.08.2009 12:40:30.691000','Go to bed'],
]

lst_to_del = [
	['05.08.2009 00:11:06.200993','05.08.2009 12:40:30.691000','Go to bed'],
	['05.08.2009 05:53:01.324993','05.08.2009 05:54:47.140995','Use toilet'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)
print(idxs_to_delete)

#----------------------------------------------------------------------------------------------------
new_rows_to_add = [*lst_to_add_0, *lst_to_add_1, *lst_to_add_2, *lst_to_add_3, *lst_to_add_4, *lst_to_add_5, *lst_to_add_6, ]
df_acts = df_acts.drop(idxs_to_delete)
df_news = pd.DataFrame(data=new_rows_to_add, columns=['start_time', 'end_time', 'activity'])
df_news['start_time'] = pd.to_datetime(df_news['start_time'], dayfirst=True)
df_news['end_time'] = pd.to_datetime(df_news['end_time'], dayfirst=True)
df_acts = pd.concat([df_acts, df_news], axis=0)
