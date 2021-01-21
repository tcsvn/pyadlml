Sklearn Pipelines
=================

One goal of pyadlml is to integrate seamlessly into a machine learning workflow. Most of the
methods can be used in combination with the sklearn pipeline.

.. code:: python

    from pyadlml.preprocessing import ImageEncoder, LabelEncoder

    raw = ImageEncoder(data.df_devices, window_length='30s', rep='raw', t_res='10s')
    labels = LabelEncoder(raw, data.df_activities)

    # TODO full code example
    list = []