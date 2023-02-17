1. Installation
***************

Pyadlml can be installed with ``pip`` or from `source`_.

Pip
~~~

The setup is pretty straight forward. At the command line type

::

    $ python -m pip install pyadlml[complete]

This will install all dependencies that are required to use every functionality offered
by pyadlml. The complete dependency set is particulary useful as starting point for developers.
However, this installation option involves downloading numerous packages, many of which
may not be used. To address this issue, other dependency subsets tailored 
for specific usecases are

::

    $ python -m pip install "pyadlml[light]"    # dataloading, statistics, pipeline, preprocessing and feature_extraction
    $ python -m pip install "pyadlml[datavis]"  # light's functionality and visualization

Github
~~~~~~
Since the `pipy`_ repository may lag behind, the latest version can be installed directly from `github`_ with

::

    $ git clone https://github.com/tcsvn/pyadlml
    $ cd pyadlml
    $ pip install .



.. _source: https://github.com/tcsvn/pyadlml
.. _github: https://github.com/tcsvn/pyadlml
.. _pipy: https://pypi.python.org/pypi/pyadlml/
