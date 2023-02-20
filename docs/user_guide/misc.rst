11. Miscellaneous
*****************

This section pertains to miscellaneous content that was not previously covered.

Home Assistant
==============

It is possible to load a devices from a Home Assistant database directly by providing
a valid database URL:


.. code:: python

    from pyadlml.dataset import load_homeassistant

    db_url = "sqlite:///config/homeassistant-v2.db"
    df = load_homeassistant(db_url)

To get the devices in the typical device dataframe representation (*time*, *device*, *timestamp*) use


.. code:: python

    from pyadlml.dataset import load_homeassistant_devices

    db_url = "sqlite:///config/homeassistant-v2.db"
    df_devices = load_homeassistant_devices(db_url)


.. attention::

    Home Assistant saves all database entries in UTC and not in local time. When working with 
    data produced by different devices, a conversion to local time may be necessary. 

.. _activity-assistant: http://github.com/tcsvn/activity-assistant/