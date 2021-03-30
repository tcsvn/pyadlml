Timeseries Representation
*************************

TODO include image of all timeslices


I.i.d
=====
There are various models that assume the data to be identical independently distributed (i.i.d).

.. math::
    X = \{x_1 ,..., x_N \}
The following example shows how you would typically load the data when using a model that
presumes the i.i.d assumption:

.. code:: python

    from pyadlml.preprocessing import DiscreteEncoder, LabelEncoder
    from pyadlml.dataset import fetch_aras
    from sklearn.utils import shuffle

    data = fetch_aras()

    raw = DiscreteEncoder(rep='raw').fit_transform(data.df_devices)

    y = LabelEncoder(data.df_activities).fit_transform(raw).values
    X = raw.values

    # shuffle the data as it is still ordered
    X, y = shuffle(X, y, random_state=0)



.. Note::
    Obviously the i.i.d assumption doesn't hold for data in smart homes.

    - As ADLs have a temporal dependency and are thought of as the generating process behind the observations in a smart home, the recorded device readings
    can't be independent of each other.

    - You could add features being selectively "on" for a specific time of the day

    or the day itself. However this doesn't consider one important characteristic of ADLs. Their order is time invariant.
    For example an inhabitant is very likely to go to bed after he brushes his teeth, but the point in time when he goes
    to bed varies a lot.
    - I.i.d data correlates certain times of a day with certain activities but neglects the activity
    TODO rewrite
    orders time invariance. In Addition it is difficult to choose the right resolution for these features as there
    is a tradeoff between resolution and number of features.

    This and more reasons motivate the use of sequential representations

References for models that use i.i.d data
- TODO link to userguide example
- TODO link to notebooks


Sequential
==========

Data is in the form of an ordered list

.. math::
    X = [x_1, ..., x_N]

of binary vectors

.. math::
    x_t = \begin{bmatrix} 1 & 0 & ... & 1\end{bmatrix}^T

Transforming the data into one of the representations *raw*, *changepoint* or *last_fired* usually yields the
datapoints already being ordered. There is no change in loading the dataset assuming a sequential format.

.. code:: python

    from pyadlml.preprocessing import DiscreteEncoder, LabelEncoder

    raw = DiscreteEncoder(rep='raw').fit_transform(data.df_devices)
    lbls = LabelEncoder(raw).fit_transform(data.df_activities)

    y = lbls.values
    x = raw.drop_duplicates().values

.. Note::
    The drawback using only an ordered event list is neglecting the time passed between consecutive
    event triggers. One way to account for this is to discretize time and assigning binary state
    vectors to timeslices rather than to events.

References for models that use i.i.d data
- TODO link to userguide example
- TODO link to notebooks


Timeslice
=========

.. image:: ../_static/images/reps/timeslice.svg
   :height: 200px
   :width: 500 px
   :scale: 90%
   :alt: alternate text
   :align: center


From the first unto the last event, the data is divided into equal-length timeslices. Each timeslice is
assigned a binary vector. How the vectors are assigned differs for each representation. For the *raw*
representation a timeslices binary vector entry is assigned either the last known device state or
the current device state of an event that falls into the timeslice. If multiple events originating from
the same device fall into the same timeslice, the most prominent state is assumed and the succeeding
timeslice is set to the last known event state. The *changepoint* representation sets a field to 1 if at
least one event of the specific device falls into the timeslice. The *last fired* representation TODO
look up.

The timeslices can be created by passing a resolution ``t_res='freq'`` to the DiscreteEncoder. Here is
an example for the *raw* representation with a timeslice-length of 10 seconds.

.. code:: python

    from pyadlml.preprocessing import DiscreteEncoder, LabelEncoder

    raw = DiscreteEncoder(rep='raw', t_res='10s').fit_transform(data.df_devices)
    labels = LabelEncoder(raw).fit_transform(data.df_activities)

    X = raw.values
    y = labels.values

.. Note::
    The drawback using timeslices as data representation is a trade-off originating in the choice of
    timeslice resolution. The greater the timeslice-length the higher the probability multiple events
    fall into the same timeslice, leading to a higher information loss. Smaller timeslice-length lead to
    a higher dataset size, which can lead to problems when learning the parameters of some models. Looking
    at you HSMM :/. If a model is used in a real-time context the time for performing inference
    must not exceed the timeslice-length to ensure reliable predictions.

References for models that use timesliced data
- TODO link to userguide example
- TODO link to notebooks

Temporal points
===============

TODO include description
- add link to pyhawkes

Further transformations
=======================

Based on the different representations further transformations can be done. Sequence models like LSTMs need
to have batches of samples. With the rise of machine learning models that are good at recognizing images it can
be reasonable to represent a timeseries as an image in order to make use of these models capabilities.

.. code:: python
    from pyadlml.preprocessing import ImageEncoder, ImageLabelEncoder
    from pyadlml.model_selection import

    img_enc = ImageEncoder(rep='raw', t_res='10s', window_length='30s')
    raw_img = img_enc.fit_transform(data.df_devices)
    labels = ImageLabelEncoder(raw_img, data.df_activities)



.. image:: ../_static/images/reps/image.svg
   :height: 200px
   :width: 500 px
   :scale: 80%
   :alt: alternate text
   :align: center


The image is being generated by sliding a window over the sequential data. For each image the
corresponding activity is that of the images last timestamp. *Raw*, *changepoint* and *last_fired* representation
can be transformed into images.

.. code:: python

    from pyadlml.preprocessing import ImageEncoder, ImageLabelEncoder

    img_enc = ImageEncoder(rep='raw', t_res='10s', window_length='30s')
    raw_img = img_enc.fit_transform(data.df_devices)

    labels = ImageLabelEncoder(raw_img, data.df_activities)

    X = raw.values
    y = labels.values
