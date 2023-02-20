4. Device encoding
******************

.. image:: ../_static/images/encodings/state_vector_encoding.svg
   :height: 90px
   :width: 300px
   :scale: 200%
   :alt: alternate text
   :align: center


The Smart Home device datastream is sequence of events :math:`[e_1, ..., e_T]` where the :math:`i`-th event 
consists of a triplet :math:`e_i=(t_i, d_i, o_i)`, containing the device :math:`d_i`, that produces the observation :math:`o_i` 
recorded at time :math:`t_i`.
Since different algorithms require the event stream to be formatted in certain ways, the 
``pyadlml.preprocessing`` module provides several device encoders.
To generate encoded data and correct labels, the overall procedure
involves transforming the device dataframe into a specific format using 
a suitable ``[Specific]Encoder`` and subsequently labeling the new representation
with the recorded activities using the ``LabelMatcher``:


.. code:: python

    from pyadlml.preprocessing import LabelMatcher, SpecificEncoder

    enc = SpecificEncoder(*args, **kwargs)
    X_enc = enc.fit_transform(data['devices'])

    lm = LabelMatcher(*args, **kwargs)
    y_enc = lm.fit_transform(data['activities'], X_enc)

    X = X_enc.values
    y = y_enc.values


Index
^^^^^

When training neural networks, a popular method is to embed categories or tokens into a learnable 
layer. One approach is the lookup embedding, where each token indexes some trainable weights. 
To enable the use of a lookup embedding, the ``IndexEncoder`` maps each device to a positive natural number.


.. code:: python

    >>> from pyadlml.preprocessing import IndexEncoder
    >>> from pyadlml.dataset import fetch_amsterdam
    >>> data = fetch_amsterdam()

    >>> enc = IndexEncoder()
    >>> enc.fit_transform(data['devices'])
                        time  device  value
    0    2008-02-25 00:20:14       7   True
    1    2008-02-25 00:22:57       7  False

    ...                  ...     ...    ...
    2619 2008-03-23 19:04:47       4  False
    [2620 rows x 3 columns]




StateVector
^^^^^^^^^^^
Pyadlml supports three different types of state-vectors, the *raw*, *changepoint* and *lastfired*
encoding.




Raw
~~~

Definition
==========

.. image:: ../_static/images/encodings/raw.svg
   :height: 90px
   :width: 300 px
   :scale: 200 %
   :alt: alternate text
   :align: center

The *raw* representation is a vector representing the state of all Smart Home devices at 
the given moment in time. For example, the illustration above depicts the raw representation
of a binary, categorcal and a numerical devices. 

.. math::
    x_i = \begin{bmatrix}t_i &  1 & 0 & -1.348 & \text{ open } & ... & 1\end{bmatrix}^T \\
    \text{ where } x_{i,k_{/1}} \in \{0,1\} \cup \mathbb{R} \cup \text{Categorical}


Example
=======

To transform a device dataframe to the *raw* representation use the *StateVectorEncoder* with the 
``encode=raw`` parameter

.. code:: python

    >>> from pyadlml.preprocessing import StateVectorEncoder
    >>> raw = StateVectorEncoder(encode='raw').fit_transform(data['devices'])
    >>> print(raw.head())
                        time  Hall-Toilet door  ...  Croceries Cupboard     Pans Cupboard
    0    2008-02-25 00:20:14                 1  ...                   0                 0
    1    2008-02-25 00:22:57                 0  ...                   0                 0
    ...
    2619 2008-02-25 09:33:47                 0  ...                   0                 0
    [2620 rows x 15 columns]


Unknown values
==============

When encoding state vectors, device values from previous events are used to fill in the fields for all devices
except for the firing device. For events that occur prior to the timepoint at which a device fires 
for the first time, the values must be inferred.

.. image:: ../_static/images/rep_value_imp.svg
   :height: 90px
   :width: 300 px
   :scale: 200 %
   :alt: alternate text
   :align: center

In the binary case, the correct values are inferred by inverting the first observed value. 
For categorical values, the *StateVectorEncoder* fills in the preceding category with the most 
likely category given the first known succeeding category :math:`argmax[p(c_{<t}|c_{t})]`. 
Numerical values in a state-vector at timepoints where the device does
not emit observations are populated with ``NaN``'s. 


.. note::

    To guarantee working with correct values only, determine the timestamp
    at which all devices fired at least once and use the dataframe starting
    from that point onward 

    .. code:: python

        raw = StateVectorEncoder(encode='raw').fit_transform(data['devices'])

        # get time string of last device that fired for the first time
        timestr = TODO

        # select all values after the device
        raw = raw[raw['time'] > timestr]


Changepoint
~~~~~~~~~~~

Definition
==========
.. image:: ../_static/images/encodings/changepoint.svg
   :height: 90px
   :width: 300 px
   :scale: 200 %
   :alt: alternate text
   :align: center


The changepoint representation one-hot encodes all devices indicating the device that generated the event.
A field is assigned a value of one at timepoint :math:`t_i` if the device :math:`d_i` is responsible for producing
the current event :math:`e_i`. Conversely, if the device did not generate the current event,
the fields value is set to zero

.. math::
    x_i = \begin{bmatrix} t_i & 0 & 1  & ... & 0 \end{bmatrix}^T \text{ where } x_{i, k_{/1}} \in \{0,1\}


Example
=======

Load the changepoint representation by using the ``encode='changepoint'`` argument.

.. code:: python

    >>> from pyadlml.preprocessing import StateVectorEncoder

    >>> cp = StateVectorEncoder(encode='changepoint').fit_transform(data['devices'])
    >>> print(cp.head())
                        time  Hall-Toilet door  ...  Croceries Cupboard     Pans Cupboard
    0    2008-02-25 00:20:14                 1  ...                   0                 0
    1    2008-02-25 00:22:57                 0  ...                   0                 0
    ...
    2619 2008-02-25 09:33:47                 0  ...                   0                 0
    [2620 rows x 15 columns]



LastFired
~~~~~~~~~

Definition
==========

.. image:: ../_static/images/encodings/lastfired.svg
   :height: 90px
   :width: 300 px
   :scale: 200 %
   :alt: alternate text
   :align: center

The *last_fired* representation is a device one-hot-encoding signifying the device to fired last. 
A field contains the value one at timepoint :math:`t`, if the device was the most recent to change its state. 
Conversely, for devices firing earlier all fields are assigned a zero.

.. math::
    x_i = \begin{bmatrix} t_i & 0 & 1  & ... & 0 \end{bmatrix}^T \text{ where } x_{t, k_{/1}} \in \{0,1\}

.. note::

    Notice, that for data that is not up- or downsampled, the *last_fired* and 
    the *changepoint* representation will be identical.

Example
=======
To transform a device dataframe into the *last_fired* representation use the ``encode='last_fired'`` argument

.. code:: python

    from pyadlml.preprocessing import StateVectorEncoder

    lf = StateVectorEncoder(encode='last_fired').fit_transform(data['devices'])
    X = lf.values


Combining Encodings
~~~~~~~~~~~~~~~~~~~

In the majority of cases, it is practical to combine multiple encodings,
such as i.e. the *raw* and the *last_fired* representation. To do this,
concatenate the different encodings string-representations using the ``+`` 
operator and provide the resultant string as parameter. Below is an example snippet, 
that combines the *raw* and the *changepoint* encoding:

.. code:: python

    X = StateVectorEncoder(encode='raw+changepoint')\
        .fit_transform(data['devices'])\
        .values
