.. _Dataset user guide:

Datasets
********

.. _activity_dataframe:

Usually a dataset is composed of two dataframes, the logged activities and the recorded device readings.
An entry of the activity dataframe consists of the *start_time*, the *end_time*  and the *activity*
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
specific *val*\ue. The most common devices produce binary values but pyadlml also supports
categorical or numerical values.

.. csv-table:: df_devices
   :header: "time", "device", "value"
   :widths: 30, 20, 10

    2020-12-27 15:35:08.124228,light.bathroom,0
    2020-12-27 15:45:10.470072,switch bedroom,1
    2020-12-27 17:35:11.961696,temp.sensor 3,13.84
    ...

.. Note::
    Pyadlml supports 8 datasets so far. If you happen to come by a dataset, not included in this list
    please open an issue on github and I will add the dataset to the library.


Getting the data
================

A dataset is loaded using a function following the schema

.. py:function:: pyadlml.dataset.fetch_datasetname(cache=True, keep_original=False, retain_corrections=False)

   Returns a data object with attributes *df_activities* and *df_devices*



The data object serves as a container for relevant dataset attributes, that may differ
from dataset to dataset. For an exhaustive list and detailed dataset information visit :ref:`datasets <Dataset View>`.
The example below shows the :ref:`amsterdam <amsterdam>` dataset being loaded

.. code:: python

    >>> from pyadlml.dataset import fetch_amsterdam

    >>> data = fetch_amsterdam()
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
    [2832 rows x 3 columns]


.. attention::
    Sometimes activities for multiple inhabitants are recorded and can only be accessed via other
    attribute names. If ``data.df_activities`` returns ``None`` make sure to check for other attributes
    with ``dir(data)``.

    .. code:: python

        from pyadlml.dataset import fetch_aras

        data = fetch_aras()
        dir(data)
        >>> [..., df_activities_subject_1, df_activities_subject_2, df_devices, ...]

.. _storage_and_cache:

Storage and cache
=================

By default datasets are stored in the folder where python is executed. Many datasets use different formats
to represent device readings and activities and thus have to be transformed beforehand. As preprocessing takes time
to compute, it can be reasonable setting the ``fetch_dataset``\s parameter ``cache=True`` to store the dataset
as binary file after the first fetch for faster access. The folder where data is stored can be changed with

.. code:: python

    from pyadlml.dataset import fetch_aras, set_data_home

    set_data_home('/path/to/folder/')
    data = fetch_aras(cache=True, keep_original=True)

    # the original aras dataset as well as the cached version are stored in '/path/to/folder'

For more methods utilising the data home directory refer to the :ref:`api <todo>`

Coming from activity-assistant
==============================
If you collected your own data using `activity-assistant`_, the dataset can be loaded
by extracting the ``data_name.zip`` file and pointing pyadlml to the folder containing the zip's content (``devices.csv``,...)

.. code:: python

    from pyadlml.dataset import load_act_assist

    data = load_act_assist('path/to/data_name/', subjects=['chris'])

.. note::
    Activity-assistant creates an activity file using the naming convention ``activities_[subject_name].csv``.
    Pyadlml loads a dataframe referenced by the attribute ``data.df_activities_[subject_name]``.

.. _error_correction:

Data cleaning
=============
In order to correctly compute all summary statistics or data transformations, pyadlml places some
constraints on how the activity and device dataframe ought to look like. For example activity intervals are not
allowed to overlap, devices should not trigger at exactly the same moment or directly succeeding binary device
triggers have to differ. As some datasets are in a rather desolate state, the ``fetch_dataset`` method already
cleans some data beforehand. To offer transparency on what values were altered, passing
the parameter ``retain_correction=True`` to the ``fetch_dataset`` method, stores activity as well
as device corrections in the ``data`` objects attributes.

Activity correction
~~~~~~~~~~~~~~~~~~~

Altered activity entries can be accessed by the attribute ``data.correction_activities``.
The list contains tuples, where the first item is a list of the affected activities before
and the second item after the correction.

.. code:: python

   >>> from pyadlml.dataset import fetch_amsterdam
   >>> data = fetch_aras(retain_corrections=True)
   >>> data.__dict__
   [..., correction_activities, ...]

    >>> print(data.correction_activities[0][0])
    [(),
    ....
    ]
    >>> print(data.correction_activities[0][1])


Device correction
~~~~~~~~~~~~~~~~~

Devices are corrected by dropping duplicate entries, altering entries where the timestamps
coincide and disregarding equal pairwise succeeding values of binary devices.
When timestamps of two entries are the same, one of the two entries is randomly chosen
and a small offset is added onto the timestamp. Device entries where the timestamps
where altered can be accessed with the attribute ``data.correction_devices_ts_duplicates``.
For binary devices, that report the same value in direct succession the redundant entry is
dropped. The list of rows that were dropped during the correction can be accessed via
the attribute ``data.correction_devices_on_off_inconsistent``. In the following example
either list is printed

.. code:: python

   >>> from pyadlml.dataset import fetch_amsterdam
   >>> data = fetch_aras(retain_corrections=True)
   >>> data.__dict__
   [..., correction_devices_ts_duplicates, , ...]

    >>> print(data.correction_devices_ts_duplicates[0][0])
    >>> print(data.correction_devices_on_off_inconsistent[0][1])



.. _activity-assistant: https://www.google.de