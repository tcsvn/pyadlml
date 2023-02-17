.. _dataset view:

Datasets
========

.. warning::
    Project is under heavy development. Dataset information is wrong or incomplete. I am going to 
    update this as soon as possible. Therefore the following information is given without 
    any warranty to be correct. 


This page presents an exhaustive list of all supported datasets.
The attributes ``Activities`` and ``Devices`` follow the schemata  ``#x/#X``, where ``#x`` is
the number of distinct activities and ``#X`` is the total number of activity recordings.
For example ``activities: 7/263`` means the dataset has 263 recordings of 7 different activities.
In addition, the activity ``Coverage`` per dataset is given in the same way. Hereby the first number
indicates the total time covered by activities for the dataset. The second number represents the amount of
datapoints where an activity label is available. The activity coverage can be taken as a measure for a datasets quality.
Finally, the ``DeviceType`` informs whether the dataset contains  *boolean*, *numerical* or *categorical* devices.

If a cleaned version is available for a dataset it is indicated. The data cleaning process is documented
in jupyter `notebooks`_. To get a feel for the datasets this is the recommended place to look.

Amsterdam
~~~~~~~~~

:Authors: T.L.M. van Kasteren; A. K. Noulas; G. Englebienne and B.J.A. Kroese
:Contact: t.l.m.vanKasteren@uva.nl
:Paper: Tenth International Conference on Ubiquitous Computing 2008 (Ubicomp '08)
:Organization: Universiteit van Amsterdam
:Link: https://sites.google.com/site/tim0306/
:From: 2008-02-25 00:20:14
:To: 2008-03-23 19:04:58
:Activities: 7/263
:Devices: 14/2620
:DeviceType: boolean
:Coverage: 0.88/0.77

To download the dataset execute

.. code:: python

    >>> from pyadlml.dataset import fetch_amsterdam
    >>> data = fetch_amsterdam()


.. note::

    A cleaned version of the amsterdam dataset is available and can be loaded with

    .. code:: python

        clean_data = fetch_amsterdam(load_cleaned=True)

    The cleaned dataset can be reproduced with the `amsterdam notebook`_.


.. warning::

    The amsterdam's data cleaning removes intermediate and shifts successive days. Consequently, week-days
    can not be used as features since the timing of device activations and activities do
    not correspond to the correct days anymore.


Casas Aruba
~~~~~~~~~~~

:Authors: D. Cook
:Contact: hande.alemdar@boun.edu.tr
:Paper: WSU CASAS smart home project: D. Cook. Learning setting-generalized activity models for smart spaces. IEEE Intelligent Systems, 2011.
:organization: Washington State University
:Link: http://casas.wsu.edu/datasets/aruba.zip
:From: 2010-11-04 00:03:50
:To: 2011-06-11 23:58:10
:Activities: 11/6474
:Devices: 39/1713065
:DeviceType: categorical, boolean, numerical

To download the dataset execute

.. code:: python

    from pyadlml.dataset import fetch_casas_aruba

    data = fetch_casas_aruba()

for more information visit `casas aruba notebook`_.


Aras
~~~~

:Authors: H. Alemdar, H. Ertan, O.D. Incel, C. Ersoy
:Contact: hande.alemdar@boun.edu.tr
:Paper: ARAS Human Activity Datasets in Multiple Homes with Multiple Residents, Pervasive Health, Venice, May 2013.
:Organization: Boğaziçi University Department of Computer Engineering
:Link: https://cmpe.boun.edu.tr/aras/
:From: 2000-01-01 00:00:00
:To: 2000-01-30 23:54:02
:Activities r1: 26/1308
:Coverage r1: 0.97/0.99
:Activities r2: 23/811
:Coverage r2:
:Devices: 20/102233
:DeviceType: boolean

To download the dataset execute

.. code:: python

    from pyadlml.dataset import fetch_aras

    data = fetch_aras(subject="Resident 1")
    dir(data)
    >>> [..., df_activities_resident1, df_activities_resident2, df_devices, ...]

for more information visit `aras notebook`_.


Kasteren 2010
~~~~~~~~~~~~~

:Authors: T.L.M. van Kasteren, G. Englebienne and B.J.A. Kroesse
:Contact: tim0306@gmail.com
:Paper: Transferring Knowledge of Activity Recognition across Sensor Networks. In Proceedings of the Eighth
        International Conference on Pervasive Computing. Helsinki, Finland, 2010.
:organization: todo
:Link: http://sites.google.com/site/tim0306/

House A
-------

:From: 2008-02-25 00:19:32
:To: 2008-03-23 19:04:47
:Activities: 16/327
:Devices: 14/2442
:DeviceType: boolean
:Coverage: 0.88/0.89

To download the dataset execute

.. code:: python

    from pyadlml.dataset import fetch_kasteren_2010

    data = fetch_kasteren_2010(house='A')

for more information visit `kasteren 2010 house A`_.

House B
-------

:From: 2009-07-21 13:30:12
:To: 2009-08-17 13:49:19
:Activities: 24/204
:Devices: 22/36600
:DeviceType: boolean
:Coverage: 0.95/0.64

To download the dataset execute

.. code:: python

    from pyadlml.dataset import fetch_kasteren_2010

    data = fetch_kasteren_2010()

for more information visit `casas aruba notebook`_.


House C
-------

:From: 2008-11-19 22:47:46
:To: 2008-12-08 08:15:00
:Activities: 17/374
:Devices: 21/43840
:DeviceType: boolean
:Coverage: 0.88/0.95

To download the dataset execute

.. code:: python

    from pyadlml.dataset import fetch_kasteren_2010

    data = fetch_kasteren_2010()

for more information visit `casas aruba notebook`_.



MitLab
~~~~~~

:Authors: Emmanuel Munguia Tapia
:Contact: emunguia@media.mit.edu
:Paper: E. Munguia Tapia. Activity Recognition in the Home Setting Using Simple and Ubiquitous sensors. S.M Thesis
:Organization: Massachusetts Institute of Technology

Subject 1
---------

:From: 2003-03-27 06:42:04
:To: 2003-04-11 22:26:46
:Activities: 22/296
:Devices: 72/5196
:DeviceType: boolean
:Coverage: 0.16/0.95

To download execute

.. code:: python

    from pyadlml.dataset import fetch_mitlab

    data = fetch_mitlab(subject="subject1")

for more information visit `mitlab subject1 notebook`_.

Subject 2
---------

:From: 2003-04-19 02:56:53
:To: 2003-05-04 22:23:42
:Activities: 24/219
:Devices: 68/3198
:DeviceType: boolean
:Coverage: 0.24/0.94

To download the dataset execute

.. code:: python

    from pyadlml.dataset import fetch_mitlab

    data = fetch_mitlab(subject="subject2")

for more information visit `mitlab subject2 notebook`_.

UCI_ADL_Binary
~~~~~~~~~~~~~~

:Authors: OrdÃ³Ã±ez, F.J.; de Toledo, P.; Sanchis, A. A
:Contact: fordonez@inf.uc3m.es
:Publication: Activity Recognition Using Hybrid Generative/Discriminative Models on Home Environments Using Binary Sensors. Sensors 2013, 13, 5460-5477.
:Organization: Carlos III University of Madrid
:Link: https://archive.ics.uci.edu/ml/datasets/Activities+of+Daily+Living+%28ADLs%29+Recognition+Using+Binary+Sensors

Ordonez A
---------

:From: 2011-11-28 02:27:59
:To: 2011-12-12 07:22:21
:Activities: 9/248
:Devices: 12/816
:DeviceType: boolean
:Coverage: 	0.95/0.93

To download the dataset use the ``subject`` parameter with  ``OrdonezA``

.. code:: python

    from pyadlml.dataset import fetch_uci_adl_binary

    data = fetch_uci_adl_binary(subject='OrdonezA')

for more information visit `uci adl binary subjectB notebook`_.

Ordonez B
---------

:From: 2012-11-11 21:14:00
:To: 2012-12-03 01:03:59
:Activities: 10/493
:Devices: 12/4666
:DeviceType: boolean
:Coverage: 	0.88/0.64

To download the dataset use the ``subject`` parameter with  ``OrdonezB``

.. code:: python

    from pyadlml.dataset import fetch_uci_adl_binary

    data = fetch_uci_adl_binary(subject='OrdonezB')


for more information visit `uci adl binary subjectA notebook`_

Tuebingen 2019
~~~~~~~~~~~~~~

:Authors: Christian Meier
:Contact: christian.meier@student.uni-tuebingen.de
:Thesis: Activity Recognition in Smart Home Environments using Hidden Markov Models. B.A. Thesis
:Organization: Eberhardt Karl University Tuebingen
:From: 2019-05-05 10:35:42
:To: 2019-07-23 07:21:59
:Activities: 11/313
:Coverage: 0.88/0.39
:Devices: 22/197847
:DeviceType: boolean

.. code:: python

    from pyadlml.dataset import fetch_tuebingen_2019

    data = fetch_tuebingen_2019()

for more information visit `tuebingen 2019 notebook`_.

.. _notebooks: https://github.com/tcsvn/pyadlml/blob/master/notebooks/datasets/
.. _amsterdam notebook: https://github.com/tcsvn/pyadlml/blob/master/notebooks/datasets/amsterdam.ipynb
.. _aras notebook: https://github.com/tcsvn/pyadlml/blob/master/notebooks/datasets/aras.ipynb
.. _casas aruba notebook: https://github.com/tcsvn/pyadlml/blob/master/notebooks/datasets/casas_aruba.ipynb
.. _mitlab subject1 notebook: https://github.com/tcsvn/pyadlml/blob/master/notebooks/datasets/mitlab_subject1.ipynb
.. _mitlab subject2 notebook: https://github.com/tcsvn/pyadlml/blob/master/notebooks/datasets/mitlab_subject2.ipynb
.. _tuebingen 2019 notebook: https://github.com/tcsvn/pyadlml/blob/master/notebooks/datasets/tuebingen_2019.ipynb
.. _uci adl binary subjectA notebook: https://github.com/tcsvn/pyadlml/blob/master/notebooks/datasets/uci_adl_binary_subjectA.ipynb
.. _uci adl binary subjectB notebook: https://github.com/tcsvn/pyadlml/blob/master/notebooks/datasets/uci_adl_binary_subjectB.ipynb
