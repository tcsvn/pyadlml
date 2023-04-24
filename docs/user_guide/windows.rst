6. Windows
==========

Each dataset is a single sequence covering multiple weeks. For many algorithms proves useful 
to slice the single sequence into multiple batches. There exist multiple ways to do this Each
coming with tradeoffs listed below.

For activity classification it you can use give a window an activity label, we refer 
to this representation as the ``many-to-one``. The activity label
corresponds to the activity present at the end of the window. The second representation 
is the ``many-to-many`` where for each observation in the window there exists a corresponding
activity label.

    .. image:: ../_static/images/many_to_many.svg
       :height: 200px
       :width: 500 px
       :scale: 90%
       :alt: alternate text
       :align: center


Explicit window
~~~~~~~~~~~~~~~


.. code:: python

    from pyadlml.preprocessing import ExplicitWindows
    from pyadlml.dataset


.. danger:: 
    Do not use explicit windows for learning algorithms that depend on a fixed training 
    input size, since the number of events in a window may leak the label. For example,
    a neural net may deduce the label 


Event window
~~~~~~~~~~~~

TODO 

Temporal window
~~~~~~~~~~~~~~~

TODO 

FuzzyTime window
~~~~~~~~~~~~~~~~
 
 TODO