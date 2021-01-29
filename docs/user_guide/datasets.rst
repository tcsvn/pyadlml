.. _Dataset user guide:

Datasets
********

.. _activity_dataframe:

Usually a dataset is composed of two dataframes, the logged activities and the recorded device readings.
An entry of the activity dataframe consist of the *start_time*, the *end_time*  and the *activity*
that is being performed.

.. csv-table:: df_activities
   :header: "start_time", "end_time", "activity"
   :widths: 30, 30, 10

    27-12-2020 16:35:13.936,27-12-2020 16:35:23.397,eating
    27-12-2020 16:35:23.400,27-12-2020 16:35:29.806,sleeping
    27-12-2020 16:35:29.808,27-12-2020 16:35:36.057,eating
    ...

.. _device_dataframe:

A device dataframe entry consists of the *time* a certain *device* reported a
specific *val*\ue. Usually the values are, but don't necessarily have to be, binary.

.. csv-table:: df_devices
   :header: "time", "device", "val"
   :widths: 30, 20, 10

    2020-12-27 15:35:08.124228,light.bathroom,0
    2020-12-27 15:45:10.470072,switch bedroom,1
    2020-12-27 17:35:11.961696,temp.sensor 3,13.84
    ...

.. Note::
    Pyadlml supports 8 datasets so far. If you happen to come by a dataset, not included in this list
    please open an issue on github and I will add the dataset to the library. It is hard to find datasets
    online. For a full list and detailed dataset information visit :ref:`datasets <Dataset View>`.


Getting the data
================

A dataset is loaded using a function following the schema

.. py:function:: pyadlml.dataset.fetch_datasetname(cache=True, keep_original=True)

   Returns a data object possessing the attributes *df_activities* and *df_devices*



The data object functions as a container for relevant attributes of the dataset. The attributes can differ
from dataset to dataset. All datasets and the way to fetch them are listed at :ref:`datasets <Dataset View>`.
The example below shows the :ref:`amsterdam <amsterdam>` dataset being loaded

.. code:: python

    >>> from pyadlml.dataset import fetch_amsterdam

    >>> data = fetch_amsterdam(cache=True, keep_original=True)
    >>> dir(data)
    [..., df_activities, df_devices, ...]

    >>> print(data.df_devices)
                            time         device   val
    0 2020-12-27 15:35:08.124228 light.bathroom False
    1 2020-12-27 15:45:10.470072 switch.bedroom  True
    ..
    2 2020-12-27 17:35:11.961696 temp.sensor    13.84
    [263 rows x 3 columns]

    >>> print(data.df_activities)
                     start_time            end_time        activity
    0   2008-02-25 19:40:26.000 2008-02-25 20:22:58  prepare Dinner
    1   2008-02-25 20:23:12.000 2008-02-25 20:23:35       get drink
    ..                      ...                 ...             ...
    262 2008-03-21 19:10:36.000 2008-03-23 19:04:58     leave house
    [263 rows x 3 columns]


.. attention::
    Sometimes activities for multiple inhabitants are recorded and can only be accessed via other
    attribute names. If ``data.df_activities`` returns ``None`` make sure to check for other attributes
    with ``dir(data)``.

    .. code:: python

        from pyadlml.dataset import fetch_aras
        data = fetch_aras(cache=True, keep_original=True)
        dir(data)
        >>> [..., df_activities_subject_1, df_activities_subject_2, df_devices, ...]

.. _storage:

Storage and cache
=================

By default datasets are stored in the folder where python is executed. Many datasets are not
in the representation given above and have to be transformed beforehand. As the preprocessing takes time to compute,
it can be reasonable to use the ``cache=True`` option storing and reusing the dataset as binary file after
the first load. You can change the folder where the data is stored with

.. code:: python

    from pyadlml.dataset import set_data_home

    set_data_home('path/to/folder')

Coming from activity-assistant
==============================
If you collect your own data with :ref:`activity-assistant` you can load the dataset
by extracting the ``data_name.zip`` and pointing pyadlml to the folder

.. code:: python

    from pyadlml.dataset import load_act_assist

    data = load_act_assist('path/to/data_name/' subjects=['chris'])

.. note::
    Activity-assistant creates an activity file using the naming convention ``activities_[subject_name].csv``.
    Pyadlml loads the file into a dataframe referenced by the attribute ``data.df_activities_[subject_name]``.

.. _error_correction:

Error correction
================
Some datasets are in a desolate state. The ``fetch_datasetname`` method does some data cleaning beforehand.
This includes e.g deleting succeeding events reporting the same value. Some corrections deal with errors
done by researches like having overlapping activity intervals when they were defined as exclusive ect. Pyadlml
stores altered activity values under ``data.activities_corr_lst`` and omitted device values under ``data.todo``.
(TODO write more about this subject and how the different error correction strategies are done).


        Overlapping activity
        intervals and their correction are saved in *data.corr_acts_intervals*. Dropped
        duplicates can be accessed in *data.corr_acts_duplicates* or

        *data.corr_devs_duplicates*

.. _activity-assistant: