.. _dataset view:

Datasets
========

.. warning::
    Project is under heavy development. Dataset information is wrong or incomplete. I am
    going to update this as soon as possible.

The page displays an exhaustive list of all supported datasets so far.
The attributes *activities* and *devices* follow the scheme  ``#x/#X``, where ``#x`` is
the number of distinct activities and ``#X`` is the total number of activity recordings.
For example ``activities: 7/263`` means the dataset has 263 recordings of 7 different activities.
The ``DeviceType`` informs whether the dataset contains  *boolean*, *numerical* or *categorical* devices.
Make sure to also check out the `notebooks`_.

Amsterdam
~~~~~~~~~

:Authors: T.L.M. van Kasteren; A. K. Noulas; G. Englebienne and B.J.A. Kroese
:Contact: t.l.m.vanKasteren@uva.nl
:Paper: Tenth International Conference on Ubiquitous Computing 2008 (Ubicomp '08)
:Organization: Universiteit van Amsterdam
:Link: https://sites.google.com/site/tim0306/
:From: 2008-02-25 19:40:26
:To: 2008-03-23 19:04:58
:Activities: 7/263
:Devices: 14/2620
:DeviceType: boolean
:Abstract: todo

To download the dataset execute

.. code:: python

    from pyadlml.dataset import fetch_amsterdam

    data = fetch_amsterdam()
    dir(data)
    >>> [..., df_activities, df_devices, ...]

for more information visit `amsterdam notebook`_.

Casas Aruba
~~~~~~~~~~~

:Authors: D. Cook
:Contact: hande.alemdar@boun.edu.tr
:Paper: WSU CASAS smart home project: D. Cook. Learning setting-generalized activity models for smart spaces. IEEE Intelligent Systems, 2011.
:organization: Washington State University
:Link: http://casas.wsu.edu/datasets/aruba.zip
:From: 2010-11-04 00:03:50
:To: 2011-06-11 23:58:10
:Activities: 11/6475
:Devices: 31/1589824
:DeviceType: boolean, numerical, categorical
:Abstract: todo

To download the dataset execute

.. code:: python

    from pyadlml.dataset import fetch_casas_aruba

    data = fetch_casas_aruba()
    dir(data)
    >>> [..., df_activities, df_devices, ...]

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
:Activities r2: 23/811
:Devices: 20/102233
:DeviceType: boolean
:Abstract: The data was recorded in one home for two subjects, Resident 1 and Resident 2.

To download the dataset execute

.. code:: python

    from pyadlml.dataset import fetch_aras

    data = fetch_aras(cache=True, keep_original=True, subject="Resident 1")
    dir(data)
    >>> [..., df_activities_resident1, df_activities_resident2, df_devices, ...]

for more information visit `aras notebook`_.

MitLab
~~~~~~


Subject 1
---------

:Authors: Emmanuel Munguia Tapia
:Contact: emunguia@media.mit.edu
:Paper: E. Munguia Tapia. Activity Recognition in the Home Setting Using Simple and Ubiquitous sensors. S.M Thesis
:Organization: Massachusetts Institute of Technology
:From: 2003-03-27 06:42:04
:To: 2003-04-11 22:26:46
:Activities: 22/296
:Devices: 72/5196
:DeviceType: boolean

.. code:: python

    from pyadlml.dataset import fetch_mitlab

    data = fetch_mitlab(cache=True, keep_original=True, subject="subject1")

for more information visit `mitlab subject1 notebook`_.

Subject 2
---------

:Authors: Emmanuel Munguia Tapia
:Contact: emunguia@media.mit.edu
:Paper: E. Munguia Tapia. Activity Recognition in the Home Setting Using Simple and Ubiquitous sensors. S.M Thesis
:Organization: Massachusetts Institute of Technology
:From: 2003-04-19 02:56:53
:To: 2003-05-04 22:16:02
:Activities: 24/219
:Devices: 68/3198
:DeviceType: boolean

To download the dataset execute

.. code:: python

    from pyadlml.dataset import fetch_mitlab

    data = fetch_mitlab(cache=True, keep_original=True, subject="subject2")

for more information visit `mitlab subject2 notebook`_.

UCI_ADL_Binary
~~~~~~~~~~~~~~

Ordonez A
---------

:Authors: OrdÃ³Ã±ez, F.J.; de Toledo, P.; Sanchis, A. A
:Contact: fordonez@inf.uc3m.es
:Publication: Activity Recognition Using Hybrid Generative/Discriminative Models on Home Environments Using Binary Sensors. Sensors 2013, 13, 5460-5477.
:Organization: Carlos III University of Madrid
:Link: https://archive.ics.uci.edu/ml/datasets/ Activities+of+Daily+Living+%28 ADLs%29+Recognition+Using+Binary+Sensors
:From: 2011-11-28 02:27:59
:To: 2011-12-12 07:22:21
:Activities: 9/248
:Devices: 12/816
:DeviceType: boolean

To download the dataset use the ``subject`` parameter with  ``OrdonezA``

.. code:: python

    from pyadlml.dataset import fetch_uci_adl_binary

    data = fetch_uci_adl_binary(subject='OrdonezA')

for more information visit `uci adl binary subjectB notebook`_.

Ordonez B
---------

:Authors: OrdÃ³Ã±ez, F.J.; de Toledo, P.; Sanchis, A. A
:Contact: fordonez@inf.uc3m.es
:Publication: Activity Recognition Using Hybrid Generative/Discriminative Models on Home Environments Using Binary Sensors. Sensors 2013, 13, 5460-5477.
:Organization: Carlos III University of Madrid
:Link: https://archive.ics.uci.edu/ml/datasets/ Activities+of+Daily+Living+%28 ADLs%29+Recognition+Using+Binary+Sensors
:From: 2012-11-11 21:14:00
:To: 2012-12-03 01:03:59
:Activities: 10/493
:Devices: 12/4666
:DeviceType: boolean
:Abstract: This dataset comprises information regarding the ADLs performed by two users on a daily basis in their
    own homes. This dataset is composed by two instances of data, each one corresponding to a different
    user and summing up to 35 days of fully labelled data. Each instance of the dataset is described by
    three text files, namely: description, sensors events (features), activities of the daily living (labels).
    Sensor events were recorded using a wireless sensor network and data were labelled manually.

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
:Activity Coverage: 0.8 TODO
:Devices: 22/197847
:DeviceType: boolean
:Short summary: todo

.. code:: python

    from pyadlml.dataset import fetch_tuebingen_2019

    data = fetch_tuebingen_2019(cache=True, keep_original=True)
    dir(data)
    >>> [..., df_activities, df_devices, ...]

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
