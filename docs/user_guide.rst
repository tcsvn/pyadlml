User Guide
==========

Installation
------------
The setup is pretty straight forward and easy.
At the command line type

``pip install pyadlml``

and you are done.

.. _Dataset user guide:

Datasets
--------
Pyadlml supports 8 datasets so far. If you happen to come by a dataset, that is not included in this list
please let me know and I will add the dataset to the library. It can be very hard to find datasets online.
For a full list and more information about each dataset visit :ref:`Dataset View`.

Usually a dataset is composed of two dataframes, the logged activities and the recorded device readings.
An entry of the activity dataframe consist of the *start_time*, the *end_time*  and the *activity*
that is being performed. Below is an example how this could look like.

.. csv-table:: df_activities
   :header: "start_time", "end_time", "activity"
   :widths: 30, 30, 10

    27-12-2020 16:35:13.936,27-12-2020 16:35:23.397,eating
    27-12-2020 16:35:23.400,27-12-2020 16:35:29.806,sleeping
    27-12-2020 16:35:29.808,27-12-2020 16:35:36.057,eating
    ...

An entry of the device dataframe consist of the *time* a certain *device* reported a
specific *val*\ue. Most of the time the values are, but don't necessarily have to be, binary. The following table
shows how a typical device dataframe could like.

.. csv-table:: df_devices
   :header: "time", "device", "val"
   :widths: 30, 20, 10

    2020-12-27 15:35:08.124228,light.bathroom,0
    2020-12-27 15:45:10.470072,switch bedroom,1
    2020-12-27 17:35:11.961696,temp.sensor 3,13.84
    ...


Getting the data
~~~~~~~~~~~~~~~~

A dataset is loaded by using a function following the schema

.. py:function:: pyadlml.dataset.fetch_datasetname(cache=True, keep_original=True)

   Returns a data object possessing the attributes *df_activities* and *df_devices*



The data object functions as a container for relevant attributes of the dataset. The attributes can differ
from dataset to dataset. All datasets and the way to fetch them are listed at :ref:`Dataset View`.
The example below shows the :ref:`amsterdam` dataset being loaded

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
    2   2008-02-25 21:51:29.000 2008-02-25 21:52:36      use toilet
    3   2008-02-25 23:21:15.000 2008-02-25 23:28:30       go to bed
    4   2008-02-25 23:28:30.001 2008-02-25 23:29:14      use toilet
    ..                      ...                 ...             ...
    258 2008-03-21 09:42:28.000 2008-03-21 15:51:38     leave house
    259 2008-03-21 15:51:58.000 2008-03-21 15:53:10      use toilet
    260 2008-03-21 17:03:48.000 2008-03-21 18:05:18       go to bed
    261 2008-03-21 18:24:24.000 2008-03-21 18:25:05      use toilet
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

Storage and cache
^^^^^^^^^^^^^^^^^

By default datasets are stored in the folder where python is executed. Many datasets are not
in the representation given above and the preprocessing takes time to compute. Therefore it can
be reasonable to use the ``cache=True`` option storing and reusing a binary file of the result after the first load.
You can change the folder where the data is stored with

.. code:: python

    from pyadlml.dataset import set_data_home

    set_data_home('path/to/folder')

setting an environment variable used by pyadlml.

Coming from activity-assistant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you collected your own data with `activity-assistant`_, you can load the dataset
by extracting the ``data_name.zip`` and pointing pyadlml to the folder

.. code:: python

    from pyadlml.dataset import load_act_assist

    data = load_act_assist('path/to/data_name/')

.. note::
    Activity-assistant creates an activity file using the naming convention ``activities_[subject_name].csv``.
    Pyadlml loads the file into a dataframe referenced by the attribute ``data.df_activities_[subject_name]``.


Error correction
~~~~~~~~~~~~~~~~
Some datasets are in a desolate state. Therefore the fetch method does some data cleaning beforehand.
This includes e.g deleting succeeding events that report the same value. Some corrections deal with errors
done by researches like having overlapping activity intervals, when they were defined as exclusive ect. Pyadlml
stores altered activity values under ``data.activities_corr_lst`` and omitted device values under ``data.todo``.
(TODO write more about this subject and how the different error correction strategies are done).

Statistics
----------
Pyadlml supports methods to calculate rudimentary but interesting information about a dataset. The methods for devices
and activities respectively can be found in the modules ``pyadlml.dataset.stats.devices`` or  ``pyadlml.dataset.stats.activities``.
Statistics combining activities and devices reside in ``pyadlml.dataset.stats``.

The following examples use the :ref:`Amsterdam` dataset for presentation.

Activities
~~~~~~~~~~

Get the count of a device by

.. code:: python

    from pyadlml.dataset.stats.activities import activities_count

    counts = activities_count(data.df_activities)

TODO add ouput and description

Compute a markovian transition matrix

.. code:: python

    from pyadlml.dataset.stats.activities import activities_transitions

    transitions = activities_transitions(data.df_activities)

TODO add ouput and description

Compute how much total time the inhabitant spent in an activity

.. code:: python

    from pyadlml.dataset.stats.activities import activities_duration_dist

    act_durs = activities_duration_dist(data.df_activities)

TODO add ouput and description

.. code:: python

    from pyadlml.dataset.stats.activities import activity_durations

    transitions = activities_transitions(data.df_activities)

TODO add ouput and description

Approximate the activity density over one day for all activities using monte-carlo sampling

.. code:: python

    from pyadlml.dataset.stats.activities import activities_dist

    transitions = activities_dist(data.df_activities, n=1000)

.. note::
    You can pass every method with the optional parameter ``activity_list`` a list of activities. This
    can be useful if some activities were never recorded but should still be included in the statistics.

Devices
~~~~~~~
Compute the similarity between the devices themselves. High values mean they are on at the
same time and off at the same time. This is bad because their mutual information is high.

.. code:: python

    from pyadlml.dataset.stats.devices import duration_correlation

    dcorr = duration_correlation(data.df_devices)

TODO add ouput and description

Want to know how many times a device was triggered? here you go

.. code:: python

    from pyadlml.dataset.stats.devices import device_trigger_count

    dtc = device_trigger_count(data.df_devices)

TODO add ouput and description

Compute the pairwise differences between succedding device triggers for all devices

.. code:: python

    from pyadlml.dataset.stats.devices import trigger_time_diff

    tdf = trigger_time_diff(data.df_devices)

TODO add ouput and description

Compute the amount of triggers for a selected time resolution integrated to one day

.. code:: python

    from pyadlml.dataset.stats.devices import device_triggers_one_day

    tdf = device_triggers_one_day(data.df_devices, t_res='1h')

TODO add ouput and description

Compute for a certain time window how much devices trigger in that same window. Is
a way to show temporal relationships between devices

.. code:: python

    from pyadlml.dataset.stats.devices import device_tcorr

    tdf = device_tcorr(data.df_devices, t_res='1h')

TODO add ouput and description

Compute the time and the proportion a device was on or off

.. code:: python

    from pyadlml.dataset.stats.devices import devices_on_off_stats

    tdf = devices_on_off_stats(data.df_devices)

Activites and devices
~~~~~~~~~~~~~~~~~~~~~
blabla


Visualizations
--------------

Most of the plots visualize the statistics from above. The methods for devices and activities
can be found in the modules ``pyadlml.dataset.plot.devices`` or  ``pyadlml.dataset.plot.activities``. Visualizations
combining activities and devices reside in ``pyadlml.dataset.plot``.

Activities
~~~~~~~~~~

TODO add visualization for activities

Devices
~~~~~~~

TODO add visualization for devices

Theming
~~~~~~~

There are global options to set the color and colormaps of the plots.

.. code:: python

    from pyadlml.dataset import set_primary_color, set_secondary_color

    set_primary_color("#1234567")
    set_secondary_color("#1234567")

You can set global values for diverging and converging colormaps.

.. code:: python

    from pyadlml.dataset import set_converging_cmap, set_diverging_cmap

    set_primary_color()


Representations
---------------
Besides plotting there is not much we can do with the data as it is. So lets turn it into
formats digestible by models. Pyadlml supports three discrete and one image representation of the timeseries.
The overall procedure is transforming the device dataframe into a specific representation and than labeling
the new representation with activities:

.. code:: python

    from pyadlml.preprocessing import SomeEncoder, LabelEncoder

    rep_enc = SomeEncoder(rep='some_representation', *args)
    enc_devs = rep_enc.fit(data.df_devices)

    lbl_enc = LabelEncoder(data.df_activities, *args)
    enc_lbls = lbl_enc.fit(rep_enc)

    X = enc_devs.values
    y = enc_lbls.values

For now all representations regard only devices that are binary, meaning that they either have the state
``False`` for *off/0* or ``True`` for *on/1*. All representation assume a datapoint as binary vector

.. math::
    x_t = \begin{bmatrix} 1 & 0 & ... & 1\end{bmatrix}^T


representing the state of the smart home at a given point *t* in time. Each field corresponds to
the representation of a specific devices.

Raw
~~~

.. image:: images/raw.svg
   :height: 300px
   :width: 500 px
   :scale: 90 %
   :alt: alternate text
   :align: center

The raw representation is an ordered sequence of binary vectors, where the binary
vector represent the state of the smart home at a given point *t* in time.

.. image:: images/raw_matrix.svg
   :height: 300px
   :width: 500 px
   :scale: 60 %
   :alt: alternate text
   :align: center

Create a raw representation from your data by

.. code:: python

    from pyadlml.preprocessing import DiscreteEncoder, LabelEncoder

    raw = DiscreteEncoder(rep='raw').fit_transform(data.df_devices)
    labels = LabelEncoder(raw).fit_transform(data.df_activities)

    X = raw.values
    y = labels.values

Changepoint
~~~~~~~~~~~

.. image:: images/cp.svg
   :height: 300px
   :width: 500 px
   :scale: 90 %
   :alt: alternate text
   :align: center


The changepoint representation is a ordered sequence of binary vectors. Each field in the vector
corresponds to a device. A field is only "on" when the device changes its state. This representation
tries to capture the notion that device triggers may convey more information about the activity than
the state of the smart home.

.. image:: images/cp_matrix.svg
   :height: 300px
   :width: 500 px
   :scale: 60 %
   :alt: alternate text
   :align: center

The changepoint representation can be loaded by passing the right keyword to the ``rep`` argument

.. code:: python

    from pyadlml.preprocessing import DiscreteEncoder, LabelEncoder

    raw = DiscreteEncoder(rep='changepoint').fit_transform(data.df_devices)
    labels = LabelEncoder(raw).fit_transform(data.df_activities)

    X = raw.values
    y = labels.values

LastFired
~~~~~~~~~

.. image:: images/lf.svg
   :height: 300px
   :width: 500 px
   :scale: 90 %
   :alt: alternate text
   :align: center


The changepoint representation is a ordered sequence of binary vectors. Each field in the vector
corresponds to a device. A field is only "on" for the device that changed its state last.

.. image:: images/lf_matrix.svg
   :height: 300px
   :width: 500 px
   :scale: 60 %
   :alt: alternate text
   :align: center

Here is a code example to load the last fired representation

.. code:: python

    from pyadlml.preprocessing import DiscreteEncoder, LabelEncoder

    raw = DiscreteEncoder(rep='last_fired').fit_transform(data.df_devices)
    labels = LabelEncoder(raw).fit_transform(data.df_activities)

    X = raw.values
    y = labels.values

i.i.d
~~~~~
To transform the data into a format that assumes identical independently distributed data

.. math::
    X = \{x_1 ,..., x_N \}

use

.. code:: python

    from pyadlml.preprocessing import DiscreteEncoder, LabelEncoder
    raw = DiscreteEncoder(data.df_devices, rep='raw')
    y = LabelEncoder(raw, data.df_activities).values
    x = raw.drop_duplicates().values
    # maybe shuffle the data

Obviously the i.i.d assumption doesn't hold for data in smart homes.
- As ADLs have a temporal dependency and are thought of as the generating process behind the observations in a smart home, the recorded device readings
can't be independent of each other.
- You could add features being selectively "on" for a specific time of the day
or the day itself. However this doesn't consider one important characteristic of ADLs. Their order is time invariant.
For example an inhabitant is very likely to go to bed after he brushes his teeth, but the point in time when he goes
to bed varies a lot. I.i.d data correlates certain times of a day with certain activities but neglects the activity
orders time invariance. In Addition it is difficult to choose the right resolution for these features as there
is a tradeoff between resolution and number of features.

Sequential
~~~~~~~~~~

This and more reasons motivate the use of sequential representations and models, where data *X* is an ordered list

.. math::
    X = [x_1, ..., x_N]

of binary state vectors :math:`x_t`.

.. math::
    x_t = \begin{bmatrix} 1 & 0 & ... & 1\end{bmatrix}^T





.. code:: python

    from pyadlml.preprocessing import DiscreteEncoder, LabelEncoder

    raw = DiscreteEncoder(rep='raw').fit_transform(data.df_devices)
    lbls = LabelEncoder(raw).fit_transform(data.df_activities)

    y = lbls.values
    x = raw.drop_duplicates().values

Timeslice
~~~~~~~~~
The drawback of these representations is that they assume data in a sequential manner but disregard the
time between the device triggers in the smart home. One way to account for this is by assigning binary
state vectors not to events (when a device changes its state) but to timeslices. From the first event
to the last the data is divided into timeslices with the same length. A timeslices binary vector entry is
assigned either the last known device state or the current device state of an event that falls into that timeslice.
If multiple events of the same device fall into the same timeslice the most prominent state is assumed and
the succeeding timeslice is set to the last known state.

The following picture depicts the how the different representations *raw*, *changepoint* and *last_fired*
behave under the sequential assumption:




You do this by passing the parameter ``t_res='freq'`` to the DiscreteEncoder where ``t_res`` is a string
representing the timeslice length. Here is an example for the *raw* representation with a timeslice of 10 seconds:

.. code:: python

    from pyadlml.preprocessing import DiscreteEncoder, LabelEncoder
    raw = DiscreteEncoder(data.df_devices, rep='raw', t_res='10s')
    labels = LabelEncoder(raw, data.df_activities)

    X = raw.values
    y = labels.values


Image
~~~~~

With the rise of machine learning models that are good at recognizing images it can
be reasonable to represent a time series as an image to make use of these models capabilities.
The image is being generated by sliding a window over the sequential data. All
representations mentioned above can be transformed with this method. An example is

.. code:: python

    from pyadlml.preprocessing import ImageEncoder, LabelEncoder

    raw = ImageEncoder(data.df_devices, window_length='30s', rep='raw', t_res='10s')
    labels = LabelEncoder(raw, data.df_activities)

    X = raw.values
    y = labels.values


Sklearn Pipelines
-----------------
One goal of pyadlml is to integrate seamlessly into a machine learning workflow. Most of the
methods can be used in combination with the sklearn pipeline.

.. code:: python

    from pyadlml.preprocessing import ImageEncoder, LabelEncoder
    raw = ImageEncoder(data.df_devices, window_length='30s', rep='raw', t_res='10s')
    labels = LabelEncoder(raw, data.df_activities)

    list = []


Miscellaneous
-------------
This is the section where everything goes that didn't fit so far.


Home Assistant
~~~~~~~~~~~~~~

It is possible to load a device representation from a Home Assistant database . Every valid database url
will suffice

.. code:: python

    from pyadlml.dataset import load_homeassistant

    db_url = "sqlite:///config/homeassistant-v2.db"
    df_devices = load_homeassistant(db_url)




.. _activity-assistant: http://github.com/tcsvn/activity-assistant/