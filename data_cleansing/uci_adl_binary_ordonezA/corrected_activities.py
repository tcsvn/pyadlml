# Awesome script 
import pandas as pd
from pyadlml.dataset._core.activities import get_index_matching_rows

# TODO add original activity dataframe in line below
idxs_to_delete = []

# Content
#----------------------------------------------------------------------------------------------------


# id=0
# Shorter activity trumps longer activity

lst_to_add_0 = [
	['01.12.2011 16:27:04.100000','01.12.2011 16:46:40.000000','Spare_Time/TV'],
]

lst_to_del = [
	['01.12.2011 16:27:03.001000','01.12.2011 16:46:40.000000','Spare_Time/TV'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)


# id=1
# Shorter activity trumps longer activity

lst_to_add_1 = [
	['03.12.2011 13:30:00.000000','03.12.2011 14:01:20.000000','Spare_Time/TV'],
]

lst_to_del = [
	['03.12.2011 13:29:48.001000','03.12.2011 14:01:20.000000','Spare_Time/TV'],
]
idxs_to_delete += get_index_matching_rows(df_acts, lst_to_del)

print(idxs_to_delete)
#----------------------------------------------------------------------------------------------------
new_rows_to_add = [*lst_to_add_0, *lst_to_add_1, ]
df_acts = df_acts.drop(idxs_to_delete)
df_news = pd.DataFrame(data=new_rows_to_add, columns=['start_time', 'end_time', 'activity'])
df_news['start_time'] = pd.to_datetime(df_news['start_time'], dayfirst=True)
df_news['end_time'] = pd.to_datetime(df_news['end_time'], dayfirst=True)
df_acts = pd.concat([df_acts, df_news], axis=0)
