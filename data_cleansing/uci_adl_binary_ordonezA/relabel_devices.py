# Awesome script
import pandas as pd
from pyadlml.constants import TIME, DEVICE, VALUE
from pyadlml.dataset._core.devices import get_index_matching_rows

#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['09.12.2011 17:59:25.000000','Bathroom Basin PIR',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['09.12.2011 17:59:27.000000','Bathroom Basin PIR',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['29.11.2011 12:21:56.000000','Bathroom Toilet Flush',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['29.11.2011 12:21:58.000000','Bathroom Toilet Flush',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)


#----------------------------------------
# 

idx_to_del = get_index_matching_rows(df_devs, [['01.12.2011 14:08:19.000000','Bathroom Basin PIR',True]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

idx_to_del = get_index_matching_rows(df_devs, [['01.12.2011 14:08:22.000000','Bathroom Basin PIR',False]])
df_devs = df_devs.drop(index=idx_to_del).reset_index(drop=True)

