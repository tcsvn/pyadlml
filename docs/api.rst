API
===

TODO here will be a full docu of all features supported by pyadlml. Listed after modules

Dataset
=======

Get data
~~~~~~~~

.. autofunction:: pyadlml.dataset.fetch_amsterdam

.. autofunction:: pyadlml.dataset.fetch_aras

.. autofunction:: pyadlml.dataset.fetch_tuebingen_2019

.. autofunction:: pyadlml.dataset.fetch_uci_adl_binary

.. autofunction:: pyadlml.dataset.fetch_casas_aruba

.. autofunction:: pyadlml.dataset.load_act_assist

.. autofunction:: pyadlml.dataset.load_homeassistant

.. autofunction:: pyadlml.dataset.load_homeassistant_devices


Input/Output
~~~~~~~~~~~~

.. autofunction:: pyadlml.dataset.set_data_home

.. autofunction:: pyadlml.dataset.get_data_home

.. autofunction:: pyadlml.dataset.load_from_data_home

.. autofunction:: pyadlml.dataset.dump_in_data_home

.. autofunction:: pyadlml.dataset.clear_data_home

Preprocessing
=============

Encoder
~~~~~~~

.. autoclass:: pyadlml.dataset._dataset.DiscreteEncoder

Feature creation
~~~~~~~~~~~~~~~~

.. autofunction::  pyadlml.feature_creation.add_day_of_week

.. autofunction::  pyadlml.feature_creation.add_time_bins

Evaluation
~~~~~~~~~~

.. autoclass:: pyadlml.model_selection.LeaveOneDayOut

Statistic
=========

Activities
~~~~~~~~~~

.. autofunction:: pyadlml.stats.activity_dist

.. autofunction:: pyadlml.stats.activity_count

.. autofunction:: pyadlml.stats.activity_transition

.. autofunction:: pyadlml.stats.activity_duration

.. autofunction:: pyadlml.stats.activity_duration_dist


Devices
~~~~~~~

.. autofunction:: pyadlml.stats.device_trigger_sliding_window

.. autofunction:: pyadlml.stats.device_on_off

.. autofunction:: pyadlml.stats.device_on_time

.. autofunction:: pyadlml.stats.device_time_diff

.. autofunction:: pyadlml.stats.device_trigger_one_day

.. autofunction:: pyadlml.stats.device_trigger_count

.. autofunction:: pyadlml.stats.device_duration_corr

Activities ~ Devices
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pyadlml.stats.contingency_triggers

.. autofunction:: pyadlml.stats.contingency_triggers_01

.. autofunction:: pyadlml.stats.contingency_duration


Plots
=====

Activities
~~~~~~~~~~

.. autofunction:: pyadlml.plot.plot_activity_bar_count

.. autofunction:: pyadlml.plot.plot_activity_bp_duration

.. autofunction:: pyadlml.plot.plot_activity_bar_duration

.. autofunction:: pyadlml.plot.plot_activity_hm_transition

.. autofunction:: pyadlml.plot.plot_activity_rl_daily_density


Devices
~~~~~~~

.. autofunction:: pyadlml.plot.plot_device_bar_count

.. autofunction:: pyadlml.plot.plot_device_hist_time_diff_trigger

.. autofunction:: pyadlml.plot.plot_device_bp_on_duration

.. autofunction:: pyadlml.plot.plot_device_hm_similarity

.. autofunction:: pyadlml.plot.plot_device_hm_time_trigger

.. autofunction:: pyadlml.plot.plot_device_on_off


Activities ~ Devices
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pyadlml.plot.plot_hm_contingency_duration

.. autofunction:: pyadlml.plot.plot_hm_contingency_trigger

.. autofunction:: pyadlml.plot.plot_hm_contingency_trigger_01