User Guide
==========

Installation
------------
The setup is pretty straight forward and easy.
At the command line type

``pip install pyadlml``

and you are done.


Datasets
--------
There are 8 supported datasets so far:

- aras
- casas aruba
- mitlab subject 1
- mitlab subject 2
- tuebingen 2019
- casas milan

Usually a dataset consists of 2 dataframes, the logged activities and the recorded device readings.
An entry of the activity dataframe consist of the *start_time*, the *end_time*  and the *activity*
that is performed. Here is an example how this could look like:

.. csv-table:: df_activities
   :header: "start_time", "end_time", "activity"
   :widths: 30, 30, 10

    27-12-2020 16:35:13.936,27-12-2020 16:35:23.397,eating
    27-12-2020 16:35:23.400,27-12-2020 16:35:29.806,sleeping
    27-12-2020 16:35:29.808,27-12-2020 16:35:36.057,eating
    ...

An entry of the device dataframe consist of the *time* a certain *device* reported a
specific *val* ue. Here is an example how a typical device dataframe looks like:

.. csv-table:: df_devices
   :header: "time", "device", "val"
   :widths: 30, 20, 5

    2020-12-27 15:35:08.124228,light bathroom,0
    2020-12-27 15:45:10.470072,switch bedroom,1
    2020-12-27 17:35:11.961696,motion sensor 3,1
    ...


Getting the data
~~~~~~~~~~~~~~~~

You can load a dataset by using a function of the schema

.. py:function:: pyadlml.dataset.fetch_datasetname(cache=True, keep_original=True)

   Returns a data object that has attributes *df_activities* and *df_devices*

.. attention::
    Sometimes activities for multiple inhabitants are recorded and can only be accessed via other
    attribute names. If ``data.df_activities`` returns ``None`` make sure to check for other attributes
    with ``dir(data)``.

All datasets and the way to fetch them are listed at (TODO link). Here is an example where the aras dataset is loaded

.. code:: python

    from pyadlml.dataset import fetch_aras
    data = fetch_aras(cache=True, keep_original=True)
    dir(data)
    >>> [..., df_activities, df_devices, ...]

By default datasets are stored in the folder where python is executed. Many datasets are not
in the representation given above and the preprocessing takes time to compute. Therefore it can
be reasonable to use the ``cache=True`` option storing and reusing a binary file of the result after the first load.
You can change the folder where the data is stored with

.. code:: python

    from pyadlml.dataset import set_data_home
    set_data_home('path/to/folder')

setting an environment variable that is used by pyadlml.

Coming from activity-assistant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you collected your own data with [activity-assistant](asdf.com), you can load the dataset
by extracting the ``data_name.zip`` and pointing pyadlml to the folder

.. code:: python

    from pyadlml.dataset import load_act_assist
    set_data_home('path/to/datahome')
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
Pyadlml supports methods to calculate some dataset statistics. The methods for devices and activities
can be found in the modules ``pyadlml.dataset.stats.devices`` or  ``pyadlml.dataset.stats.activities``. Statistics
combining activities and devices reside in ``pyadlml.dataset.stats``.

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

Devices
~~~~~~~
blabla

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

Devices
~~~~~~~

Representations
---------------
Besides plotting there is not much we can do with the data as it is. So lets turn them into a
formats digestible by models. Pyadlml supports three discrete and one image representation.
The overall procedure is transforming the device dataframe into a specific representation and than labeling
the new representation with activities:

.. code:: python

    from pyadlml.preprocessing import SomeEncoder, LabelEncoder
    rep = SomeEncoder(data.df_devices, rep='some_representation', *args)
    labels = LabelEncoder(rep, data.df_activities)

    X = rep.values
    y = labels.values

For now all representations regard only devices that are binary, meaning that they either have the state
``False`` for *off/0* or ``True`` for *on/1* . All representation assume a datapoint is a binary vector
representing the state of the smart home at a given point *t* in time

.. math::
    x_t = \begin{bmatrix} 1 & 0 & ... & 1\end{bmatrix}^T

where each field corresponds to the state of a specific devices.

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

Obviously the i.i.d assumption doesn't hold for data in smart homes. As ADLs have a temporal dependency
and are thought of as the generating process behind the observations in a smart home, the recorded device readings
can't be independent of each other. You could add features being selectively "on" for a specific time of the day
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

of binary state vectors x_t.

.. math::
    x_t = \begin{bmatrix} 1 & 0 & ... & 1\end{bmatrix}^T

Raw
~~~

The raw representation is a ordered sequence of binary vectors, where the binary
vector represent the state of the smart home at a given point *t* in time.

.. csv-table:: df_devices
   :header: "time", "light bathroom", "...", "motion sensor"
   :widths: 20, 10, 5, 10

    2020-12-27 15:35:08.124228,0,...,0
    2020-12-27 15:45:10.470072,1,...,0
    2020-12-27 17:35:11.961696,0,...,1
    ...

Create a raw representation from your data by

.. code:: python

    from pyadlml.preprocessing import DiscreteEncoder, LabelEncoder
    raw = DiscreteEncoder(data.df_devices, rep='raw')
    labels = LabelEncoder(raw, data.df_activities)

    X = raw.values
    y = labels.values

Changepoint
~~~~~~~~~~~

The changepoint representation is a ordered sequence of binary vectors. Each field in the vector
corresponds to a device. A field is only "on" when the device changes its state. This representation
tries to capture the notion that device triggers may convey more information about the activity than
the state of the smart home.

.. csv-table:: df_devices
   :header: "time", "light bathroom", "...", "motion sensor"
   :widths: 20, 10, 5, 10

    2020-12-27 15:35:08.124228,0,...,0
    2020-12-27 15:45:10.470072,1,...,0
    2020-12-27 17:35:11.961696,0,...,0
    ...

.. code:: python

    from pyadlml.preprocessing import DiscreteEncoder, LabelEncoder
    cp = DiscreteEncoder(data.df_devices, rep='changepoint')
    labels = LabelEncoder(cp, data.df_activities)

    X = raw.values
    y = labels.values

LastFired
~~~~~~~~~

The changepoint representation is a ordered sequence of binary vectors. Each field in the vector
corresponds to a device. A field is only "on" for the device that changed its state last.

.. csv-table:: df_devices
   :header: "time", "light bathroom", "...", "motion sensor"
   :widths: 20, 10, 5, 10

    2020-12-27 15:35:08.124228,0,...,0
    2020-12-27 15:45:10.470072,1,...,0
    2020-12-27 17:35:11.961696,0,...,0
    ...

.. code:: python

    from pyadlml.preprocessing import DiscreteEncoder, LabelEncoder
    lf = DiscreteEncoder(data.df_devices, rep='lastfired')
    labels = LabelEncoder(lf, data.df_activities)

    X = raw.values
    y = labels.values

Timeslice
~~~~~~~~~
The drawback of these representations is that they assume data in a sequential manner but disregard the
time between the device triggers in the smart home. One way to account for this is by assigning binary
state vectors not to events (when a device changes its state) but to timeslices. From the first event
to the last the data is divided into timeslices with the same length. A timeslices binary vector entry is
assigned either the last known device state or the current device state of an event that falls into that timeslice.
If multiple events of the same device fall into the same timeslice the most prominent state is assumed and
setting the succeeding timeslice to the last known state.

For every representations *raw*, *changepoint* and *lastfired* the discretization via timeslices is supported.
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

.. rubric:: title