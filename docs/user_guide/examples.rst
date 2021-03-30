Examples
********

This is the section where everything goes that didn't fit so far.

Random Forests
==============

It is possible to load a device representation from a Home Assistant database . Every valid database url
will suffice

.. code:: python

    from pyadlml.dataset import load_homeassistant

    db_url = "sqlite:///config/homeassistant-v2.db"
    df_devices = load_homeassistant(db_url)

.. _activity-assistant: http://github.com/tcsvn/activity-assistant/


Hidden Markov Model
===================

TODO

Recurrent Neural Net
====================

TODO

Hawkes Process
==============

TODO
