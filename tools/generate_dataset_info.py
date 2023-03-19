from pyadlml.constants import  TIME, START_TIME, END_TIME, ACTIVITY, DEVICE
from pyadlml.dataset._core.activities import ActivityDict
from pyadlml.dataset.stats.activities import coverage
from pyadlml.dataset.util import infer_dtypes, fetch_by_name, DATASET_STRINGS
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='amsterdam',

                        choices=DATASET_STRINGS,
                        help='Select an avaliable dataset.'
                        )
    parser.add_argument('-i', '--identifier', type=str, default=None,
                        help='Specify dataset specific stuff'
    )
    args = parser.parse_args()

    data = fetch_by_name(args.dataset, args.identifier, cache=False)

    df_devs = data['devices']
    dct_acts = ActivityDict(data['activities'])

    # Determine start and end_time
    start_time = min(df_devs[TIME].iloc[0], dct_acts.min_starttime())
    end_time = max(df_devs[TIME].iloc[-1], dct_acts.max_endtime())
    # Compute device statistics
    nr_devices = len(df_devs[DEVICE].unique())
    nr_device_recordings = len(df_devs)
    dev_types = [k for k, v in infer_dtypes(df_devs).items() if len(v) > 0]


    residents = {r:{} for r in dct_acts.keys()}


    print(f':From: {start_time}')
    print(f':To: {end_time}')
    print(f':Devices: {nr_devices}/{nr_device_recordings}')
    print(f':DeviceType: {str(dev_types)}')

    for r in residents.keys():
        df_acts = dct_acts[r]

        # Compute activity statistics
        nr_activities = len(df_acts[ACTIVITY].unique())
        nr_activity_recordings = len(df_acts)

        act_cov_dp = coverage(df_acts, df_devs, datapoints=True)
        act_cov_time = coverage(df_acts, df_devs, datapoints=False)

        print(f":Activites {r}:\t{nr_activities}/{nr_activity_recordings}")
        print(f":Coverage {r}:\t{act_cov_time:.2f}/{act_cov_dp:.2f}")
