5. Feature extraction
=====================

Pyadlml provides various transforms to extract features from the device event timestamps :math:`t_i`.

!!! SECTION IS UNDER CONSTRUCTION !!!


Time of day
~~~~~~~~~~~

.. image:: ../_static/images/feature_extractor_one_day.svg
   :height: 100px
   :width: 300px
   :scale: 100%
   :align: right

The visualization (link) clearly indicates that certain activities 
take place at different times throughout the day. As a result, it may 
prove beneficial to partition a day into distinct time intervals.
This is accomplished by providing the  ``TimeOfDayExtractor`` 
with the desired time bin size as demonstrated in the following example

.. code:: python

    >>> from pyadlml.dataset import fetch_aras
    >>> from pyadlml.feature_extraction import TimeOfDay

    >>> # load data and encode as state-vector
    >>> data = fetch_aras()
    >>> state = Event2Vec(encode='state')\
           .fit_transform(data['devices'])

    >>> # extract the time of a day with a bin size of 2h, will result in 12 new features
    >>> tode = TimeOfDay(dt='2h')

    >>> # return device dataframe containing one row with strings representing the bins
    >>> tode.fit_transform(state)
    index, time_of_day
    1, '09:00:00 - 10:00:00'
    2, '09:00:00 - 10:00:00'
    ...
    2901, '21:00:00 - 22:00:00'

    >>> # Append to the current encoded vectors  as one hot encoding
    >>> tode = TimeOfDay(dt='2h', inline=True, one_hot_encoding=True)
    >>> tode.fit_transform(state)
    , time, features, '00:00:00 - 02:00:00', ..., '22:00:00 - 24:00:00'
    1, timestamp1, ..., 1, 0, ..., 0
    2, timestamp1, ..., 1, 0, ..., 0
    ...
    2901, timestamp1, ..., 0, 0, ..., 1, 0


.. note::
    Reducing ``TimeOfDay`` bin resolution :math:`dt` increases the number of features,
    but to many features may hinder a models capability to generalize. Therefore,
    it is crucial to exercise caution when deciding on the resolution 
    to ensure optimal model performance.


Day of week
~~~~~~~~~~~

.. image:: ../_static/images/feature_extractor_dow.svg
   :height: 100px
   :width: 300px
   :scale: 100%
   :align: right

The timing and set of activities throughout a day may vary depending on the weekday on which they occur. 
For instance, the activity pattern on weekends hopefully differs from that on workdays.
To accommodate this, *pyadlml* provides a transformer for extracting the day of the week:


.. code:: python

    >>> from pyadlml.feature_extraction import DayOfWeek

    >>> dowe = DayOfWeek()

    >>> # return device dataframe containing one row with strings representing the bins
    >>> dowe.fit_transform(state)
    index, day_of_week
    1, 'Monday'
    2, 'Monday'
    ...
    2901, 'Saturday'

    >>> # Append to the current encoded vectors  as one hot encoding
    >>> dowe = DayOfWeekExtractor(inline=True, one_hot_encoding=True)
    >>> dowe.fit_transform(state)
    , time, features, 'Monday', ..., 'Sunday'
    1, timestamp1, ..., 1, 0, ..., 0
    2, timestamp1, ..., 1, 0, ..., 0
    ...
    2901, timestamp1, ..., 0, 0, ..., 1, 0


.. _inter-event-interval : https//todo



Inter-event-times
~~~~~~~~~~~~~~~~~

.. image:: ../_static/images/td_extractor.svg
   :height: 100px
   :width: 300px
   :scale: 100%
   :align: right


For an event sequence :math:`(e_1, ...., e_N)` with timings :math:`(t_1, ..., t_N)` the inter-event-times 
:math:`(\tau_1, ..., \tau_{N+1})` are defined as the time elapsing between 
two succeeding events. The :math:`i`-th inter-event-time  :math:`\tau_i` is either taken with respect to
:math:`e_i`'s predecessor :math:`e_{i-1}` with :math:`\tau_i:=t_i-t_{i-1}` or to
its successor :math:`e_{i+1}` with :math:`\tau_i:=t_{i+1} - t_i`. Setting the parameter ``to=predecessor`` 
or ``to=successor`` leads to the respective interval being computed:


.. code:: python

    >>> from pyadlml.dataset import fetch_amsterdam
    >>> from pyadlml.feature_extraction import InterEventTime
    >>> data = fetch_amsterdam()

    >>> # Add a dataframe column that includes the time differences to the predecessor in seconds
    >>> tde = InterEventTime(to='predecessor', unit='s')

    >>> # Returns a device dataframe containing one row representing the bins
    >>> tde.fit_transform(state)
    index, td
    1, 101231981
    2, 101231981
    ...
    2901, 101231981

    >>> # TODO
    >>> tde = InterEventTime(to='predecessor', inline=True, unit='s')
    >>> tde.fit_transform(state)
    , time, features, td
    1, timestamp1, ..., 1101231981
    2, timestamp1, ..., 1101231981
    ...
    2901, timestamp1, ..., 0101231981


.. note::
    A inter-event time distribution accounts for the regularity or irregularity of an event train.
    Check out the accompanied statistic at (TODO insert LINK)

Time2Vec
~~~~~~~~


TODO include decsription 


.. code:: python

    >>> from pyadlml.feature_extraction import Time2Vec 
    >>> df_devs = 
    >>> t2v = Time2Vec(emb_dim=10)
    >>> t2v.fit_transform(df_devs)
 

MeanFiringRate
~~~~~~~~~~~~~~

Pyadlml offers two notions of firing rate. First, a temporal average 
or an average over a population of devices i.e. group by room.


Temporal average
****************
The firing rate for device :math:`d` is the event count :math:`n^{ev}_d` in 
an interval of duration :math:`dt`

.. math::
    v_d(t) = \frac{n^{ev}_d}{dt}

The length of time window :math:`dt` is set by the user. In practice, 
to get sensiblae averages several events should occur within the time window.


Population average
******************
.. math::
    A(t) = \frac{n_{act}(t; t+\Delta t)}{\Delta t \cdot N}


.. https://neuronaldynamics.epfl.ch/online/Ch7.S2.html

FanoFactor
~~~~~~~~~~
Fano factor measures repeatability of event train between repititions 
of the same activity.
TODO can be used to check if events follow a poisson process (F=1)
TODO move to statistics section



Conditional Intensity
~~~~~~~~~~~~~~~~~~~~~

The conditional intensity :math:`\lambda^*(t)=g(t)` represents the instantaneous rate of arrival 
for new events at time :math:`t` conditioned on all past events :math:`H(t)=\{t_j \in T : t_j < t\}`. 


Hawkes Process
**************

The hawkes process conditional intensity is defined as
:math:`\lambda^*(t) = \mu(t) + \alpha \sum_{t_j \in H(t)} \gamma (t-t_j)`
where :math:`\alpha` represents the amount of intensity increments for each
occuring event and the base rate :math:`\mu(t)`. A commmon choice for the triggering
kernel :math:`\gamma(\tau)` is the exponential kernel given by
:math:`\gamma(t-t_j)\beta e^{-\beta (t-t_j))}`. 

TODO finish

.. code:: python

    >>> from pyadlml.feature_extraction import HawkesIntensity
    >>> hp = HP(alpha=1, kernel='exponential', kernel_param={beta:2})
    >>> hp.transform(state, td='1ms')
    , time, intensity
    1, timestamp1, 2.23254
    2, timestmap2, 1.23123
